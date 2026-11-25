from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from model.config import HybridTTTConfig, LayerType
from model.moe import FineGrainedMoE
from model.sparse_attention import NativeSparseAttention
from model.ttt_layer import TTTLinearLayer


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return normed * self.weight


class HybridDecoderBlock(nn.Module):
    """Pre-RMSNorm Transformer block pairing either TTT-Linear or Native Sparse
    Attention with Fine-Grained MoE."""

    def __init__(self, config: HybridTTTConfig, layer_type: LayerType) -> None:
        super().__init__()
        self.layer_type: LayerType = layer_type
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mix_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.ttt_mixer: TTTLinearLayer | None = None
        self.sparse_mixer: NativeSparseAttention | None = None
        if layer_type == "ttt":
            self.ttt_mixer = TTTLinearLayer(config)
        else:
            self.sparse_mixer = NativeSparseAttention(config)

        self.moe = FineGrainedMoE(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.input_norm(hidden_states)
        if self.ttt_mixer is not None:
            mixed, _ = self.ttt_mixer(normed)
        elif self.sparse_mixer is not None:
            mixed = self.sparse_mixer(normed)
        else:
            raise RuntimeError("HybridDecoderBlock missing sequence mixer")

        h = hidden_states + mixed
        moe_out = self.moe(self.post_mix_norm(h))
        h = h + moe_out.hidden_states
        return h, moe_out.aux_loss


class InputInjectionAdapter(nn.Module):
    """Fuses prelude context representation e_0 into the recurrent latent state h_k
    at every loop step so the looped core never suffers contextual forgetting."""

    def __init__(self, config: HybridTTTConfig) -> None:
        super().__init__()
        self.state_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fuse_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.step_embed = nn.Embedding(config.max_recurrent_steps, config.hidden_size)
        self.residual_scale = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        latent_state: torch.Tensor,
        prelude_input: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        norm_h = self.state_norm(latent_state)
        norm_e = self.input_norm(prelude_input)
        fused: torch.Tensor = self.fuse_proj(torch.cat((norm_h, norm_e), dim=-1))
        step_vec = self.step_embed.weight[step_idx].view(1, 1, -1)
        return fused + self.residual_scale * prelude_input + step_vec


class AdaptiveComputeRouter(nn.Module):
    """Estimates per-sequence halting probability p_halt in (0, 1) after each recurrent loop."""

    def __init__(self, config: HybridTTTConfig) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2, bias=True),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, 1, bias=True),
        )
        # Initialize bias moderately negative so early iterations prefer thinking before halting
        final_layer = self.proj[2]
        if isinstance(final_layer, nn.Linear) and final_layer.bias is not None:
            nn.init.constant_(final_layer.bias, -1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pool along sequence dimension to compute per-sample halting probability (B,)
        pooled = self.norm(hidden_states).mean(dim=1)
        logits: torch.Tensor = self.proj(pooled).squeeze(-1)
        return torch.sigmoid(logits)


@dataclass(slots=True)
class RecurrentCoreOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    ponder_loss: torch.Tensor
    steps_used: int
    per_sample_steps: torch.Tensor
    halting_probs: torch.Tensor


class RecurrentDepthCore(nn.Module):
    """Weight-tied Looped Transformer Core (8–12 layers, ~85% TTT / ~15% Sparse Attention)
    featuring Input Injection, Adaptive Compute Halting (1..16 loops), and Truncated BPTT."""

    def __init__(self, config: HybridTTTConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [HybridDecoderBlock(config, ltype) for ltype in config.core_layer_pattern]
        )
        self.input_injection = InputInjectionAdapter(config)
        self.compute_router = AdaptiveComputeRouter(config)
        self.output_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _run_core_pass(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass_aux = h.new_zeros(())
        for layer in self.layers:
            h, layer_aux = layer(h)
            pass_aux = pass_aux + layer_aux
        return h, pass_aux / float(len(self.layers))

    def forward(
        self,
        prelude_states: torch.Tensor,
        num_steps: int | None = None,
        adaptive_halt: bool | None = None,
    ) -> RecurrentCoreOutput:
        bsz = prelude_states.shape[0]
        device = prelude_states.device
        dtype = prelude_states.dtype

        use_adaptive = (not self.training) if adaptive_halt is None else adaptive_halt
        max_steps = (
            self.config.max_recurrent_steps
            if (num_steps is None and use_adaptive)
            else (
                num_steps
                if num_steps is not None
                else self.config.default_recurrent_steps
            )
        )
        if not (1 <= max_steps <= self.config.max_recurrent_steps):
            raise ValueError(
                f"num_steps ({max_steps}) must be in [1, {self.config.max_recurrent_steps}]"
            )

        bptt_window = self.config.truncated_bptt_steps
        grad_start_step = max(0, max_steps - bptt_window) if self.training else 0

        h = prelude_states
        halted_output = torch.zeros_like(prelude_states)
        cumulative_halt = torch.zeros(bsz, device=device, dtype=torch.float32)
        still_running = torch.ones(bsz, device=device, dtype=torch.bool)
        per_sample_steps = torch.zeros(bsz, device=device, dtype=torch.float32)

        total_aux_loss = prelude_states.new_zeros(())
        grad_aux_count = 0
        halting_history: list[torch.Tensor] = []
        steps_executed = 0

        for step in range(max_steps):
            steps_executed = step + 1
            in_no_grad_prefix = self.training and step < grad_start_step

            if in_no_grad_prefix:
                with torch.no_grad():
                    h_in = self.input_injection(h, prelude_states, step_idx=step)
                    h, _ = self._run_core_pass(h_in)
                    p_halt = self.compute_router(h)
                halting_history.append(p_halt.detach())
                per_sample_steps = per_sample_steps + still_running.float()
                if step + 1 == grad_start_step:
                    # Detach latent state boundary explicitly for Truncated BPTT while
                    # keeping prelude_states attached for Input Injection gradient flow.
                    h = h.detach()
                continue

            h_in = self.input_injection(h, prelude_states, step_idx=step)
            h, pass_aux = self._run_core_pass(h_in)
            total_aux_loss = total_aux_loss + pass_aux
            grad_aux_count += 1

            p_halt = self.compute_router(h)
            halting_history.append(p_halt)
            per_sample_steps = per_sample_steps + still_running.float()

            if use_adaptive:
                cumulative_halt = cumulative_halt + p_halt.float() * still_running.float()
                can_halt = (step + 1) >= self.config.min_recurrent_steps
                if can_halt:
                    newly_halted = still_running & (cumulative_halt >= self.config.halting_threshold)
                    if newly_halted.any():
                        mask = newly_halted.view(bsz, 1, 1)
                        halted_output = torch.where(mask, h, halted_output)
                        still_running = still_running & (~newly_halted)
                    if not still_running.any():
                        break

        if use_adaptive and still_running.any():
            mask = still_running.view(bsz, 1, 1)
            final_h = torch.where(mask, h, halted_output)
        elif use_adaptive:
            final_h = halted_output
        else:
            final_h = h

        halting_probs = torch.stack(halting_history, dim=1)  # (B, steps_executed)

        # Ponder regularization encourages calibrated compute usage during training
        if grad_aux_count > 0:
            avg_aux_loss = total_aux_loss / float(grad_aux_count)
        else:
            avg_aux_loss = prelude_states.new_zeros(())

        # Differentiable expected halting step regularization on active grad window
        active_halts = halting_probs[:, grad_start_step:] if self.training else halting_probs
        target_halt = min(0.95, 1.0 / float(max(1, self.config.default_recurrent_steps)))
        ponder_loss = (
            self.config.ponder_loss_coef
            * (active_halts.mean() - target_halt).pow(2).to(dtype=dtype)
        )

        return RecurrentCoreOutput(
            hidden_states=self.output_norm(final_h),
            aux_loss=avg_aux_loss,
            ponder_loss=ponder_loss,
            steps_used=steps_executed,
            per_sample_steps=per_sample_steps,
            halting_probs=halting_probs,
        )
