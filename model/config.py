from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LayerType = Literal["ttt", "sparse_attn"]


@dataclass(frozen=True, slots=True)
class HybridTTTConfig:
    """Configuration for Hybrid-TTT-Coder supporting 16B/2.6B-active production
    and lightweight CI testing presets."""

    vocab_size: int = 65_536
    hidden_size: int = 3_072
    num_heads: int = 24
    num_kv_heads: int = 8
    max_position_embeddings: int = 1_048_576
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6

    # Prelude / Recurrent Core / Coda depth
    num_prelude_layers: int = 2
    num_core_layers: int = 12
    num_coda_layers: int = 2
    core_layer_pattern: tuple[LayerType, ...] = (
        "ttt",
        "ttt",
        "ttt",
        "ttt",
        "ttt",
        "sparse_attn",
        "ttt",
        "ttt",
        "ttt",
        "ttt",
        "ttt",
        "sparse_attn",
    )

    # Test-Time Training (TTT-Linear)
    ttt_chunk_size: int = 64
    ttt_base_lr: float = 0.05
    ttt_weight_decay: float = 0.995

    # Native Sparse Attention (NSA)
    sparse_block_size: int = 64
    sparse_top_k_blocks: int = 8
    sliding_window_size: int = 256

    # Fine-grained Mixture of Experts (64 routed + 1 shared, Top-6 active)
    moe_num_experts: int = 64
    moe_top_k: int = 6
    moe_intermediate_size: int = 1_536
    shared_intermediate_size: int = 3_072
    moe_aux_loss_coef: float = 0.01
    moe_z_loss_coef: float = 0.001

    # Looped Transformer (Recurrent Depth) + Adaptive Compute + Truncated BPTT
    min_recurrent_steps: int = 1
    default_recurrent_steps: int = 4
    max_recurrent_steps: int = 16
    truncated_bptt_steps: int = 4
    halting_threshold: float = 0.90
    ponder_loss_coef: float = 0.01
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.hidden_size <= 0 or self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be positive and divisible by num_heads")
        if self.num_kv_heads <= 0 or self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if len(self.core_layer_pattern) != self.num_core_layers:
            raise ValueError("len(core_layer_pattern) must match num_core_layers")
        for layer_type in self.core_layer_pattern:
            if layer_type not in ("ttt", "sparse_attn"):
                raise ValueError(f"unsupported layer type: {layer_type}")
        if self.ttt_chunk_size <= 0:
            raise ValueError("ttt_chunk_size must be positive")
        if not (0.0 < self.ttt_weight_decay <= 1.0):
            raise ValueError("ttt_weight_decay must lie in (0, 1]")
        if self.sparse_block_size <= 0 or self.sparse_top_k_blocks <= 0:
            raise ValueError("sparse_block_size and sparse_top_k_blocks must be positive")
        if self.sliding_window_size <= 0:
            raise ValueError("sliding_window_size must be positive")
        if not (0 < self.moe_top_k <= self.moe_num_experts):
            raise ValueError("moe_top_k must satisfy 0 < moe_top_k <= moe_num_experts")
        if not (
            1 <= self.min_recurrent_steps <= self.default_recurrent_steps <= self.max_recurrent_steps
        ):
            raise ValueError(
                "recurrent step bounds must satisfy "
                "1 <= min_recurrent_steps <= default_recurrent_steps <= max_recurrent_steps"
            )
        if not (1 <= self.truncated_bptt_steps <= self.max_recurrent_steps):
            raise ValueError("truncated_bptt_steps must satisfy 1 <= bptt <= max_recurrent_steps")
        if not (0.0 < self.halting_threshold <= 1.0):
            raise ValueError("halting_threshold must lie in (0, 1]")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_physical_layers(self) -> int:
        return self.num_prelude_layers + self.num_core_layers + self.num_coda_layers

    @property
    def ttt_ratio(self) -> float:
        ttt_count = sum(1 for t in self.core_layer_pattern if t == "ttt")
        return ttt_count / max(1, len(self.core_layer_pattern))

    def estimate_parameter_counts(self) -> dict[str, int]:
        """Compute exact analytical total and per-token active parameter counts."""
        d = self.hidden_size
        head_dim = self.head_dim
        kv_dim = self.num_kv_heads * head_dim

        # Embedding + untied LM head
        embed_params = self.vocab_size * d
        lm_head_params = self.vocab_size * d

        # Mixing params per layer type
        ttt_mix_params = 5 * d * d + d * self.num_heads + self.num_heads * (head_dim * head_dim + head_dim)
        sparse_mix_params = 2 * d * d + 2 * d * kv_dim + 3 * d * self.num_heads

        # Fine-grained MoE params per layer (SwiGLU = 3 projections)
        single_expert_params = 3 * d * self.moe_intermediate_size
        shared_expert_params = 3 * d * self.shared_intermediate_size
        router_params = d * self.moe_num_experts

        moe_total_per_layer = (
            self.moe_num_experts * single_expert_params + shared_expert_params + router_params
        )
        moe_active_per_layer = (
            self.moe_top_k * single_expert_params + shared_expert_params + router_params
        )

        # Prelude & Coda use TTT + MoE; Core uses core_layer_pattern + MoE
        total_mix = (self.num_prelude_layers + self.num_coda_layers) * ttt_mix_params
        for ltype in self.core_layer_pattern:
            total_mix += ttt_mix_params if ltype == "ttt" else sparse_mix_params

        # Input injection + adaptive router in Recurrent Core
        core_overhead = 2 * d * d + self.max_recurrent_steps * d + d * 2

        total_params = (
            embed_params
            + lm_head_params
            + total_mix
            + self.num_physical_layers * moe_total_per_layer
            + core_overhead
        )
        active_params = (
            lm_head_params
            + total_mix
            + self.num_physical_layers * moe_active_per_layer
            + core_overhead
        )
        return {
            "total_params": total_params,
            "active_params_per_token": active_params,
        }

    @classmethod
    def full_16b(cls) -> HybridTTTConfig:
        """Production 16B total / ~2.6B active configuration supporting 1M context."""
        return cls()

    @classmethod
    def tiny_test(cls) -> HybridTTTConfig:
        """Compact configuration for fast CPU/small-GPU testing and CI verification."""
        return cls(
            vocab_size=512,
            hidden_size=64,
            num_heads=4,
            num_kv_heads=2,
            max_position_embeddings=8_192,
            num_prelude_layers=1,
            num_core_layers=8,
            num_coda_layers=1,
            core_layer_pattern=(
                "ttt",
                "ttt",
                "ttt",
                "ttt",
                "ttt",
                "ttt",
                "ttt",
                "sparse_attn",
            ),
            ttt_chunk_size=16,
            sparse_block_size=8,
            sparse_top_k_blocks=2,
            sliding_window_size=16,
            moe_num_experts=16,
            moe_top_k=4,
            moe_intermediate_size=32,
            shared_intermediate_size=64,
            min_recurrent_steps=1,
            default_recurrent_steps=4,
            max_recurrent_steps=8,
            truncated_bptt_steps=2,
            halting_threshold=0.85,
        )
