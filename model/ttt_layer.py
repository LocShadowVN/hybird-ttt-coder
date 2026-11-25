from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import HybridTTTConfig


@dataclass(slots=True)
class TTTState:
    """Constant-size O(1) fast-weight state persisted across chunks/generations."""

    weight: torch.Tensor  # (B, H, d_h, d_h)
    bias: torch.Tensor  # (B, H, 1, d_h)


class TTTLinearLayer(nn.Module):
    """Test-Time Training (TTT-Linear) sequence mixer.

    Updates per-head fast weights (W, b) via strictly causal mini-batch gradient
    descent in dual form, achieving O(N) time complexity and O(1) recurrent state
    memory regardless of sequence length up to 1M tokens.
    """

    def __init__(self, config: HybridTTTConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.chunk_size = config.ttt_chunk_size
        self.base_lr = config.ttt_base_lr
        self.weight_decay = config.ttt_weight_decay

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Input-dependent per-head learning rate gate
        self.lr_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        nn.init.zeros_(self.lr_proj.bias)

        # Meta-learned initial fast weights (W0, b0) per head
        self.init_weight = nn.Parameter(
            torch.randn(1, self.num_heads, self.head_dim, self.head_dim)
            * (self.head_dim**-0.5)
            * 0.1
        )
        self.init_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim))

        # Per-head post-update LayerNorm parameters (token-local, preserves strict causality)
        self.ttt_norm_weight = nn.Parameter(torch.ones(1, self.num_heads, 1, self.head_dim))
        self.ttt_norm_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, self.head_dim))
        self.norm_eps = config.rms_norm_eps

    def init_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> TTTState:
        w = self.init_weight.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1).clone()
        b = self.init_bias.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1).clone()
        return TTTState(weight=w, bias=b)

    def _head_layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        normed = (x - mean) * torch.rsqrt(var + self.norm_eps)
        return normed * self.ttt_norm_weight + self.ttt_norm_bias

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: TTTState | None = None,
    ) -> tuple[torch.Tensor, TTTState]:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        gate = F.silu(self.gate_proj(hidden_states))

        # Transpose to (B, H, T, d_h) and L2-normalize Q, K per token for fast-weight stability
        q = F.normalize(q.transpose(1, 2), p=2, dim=-1, eps=1e-6)
        k = F.normalize(k.transpose(1, 2), p=2, dim=-1, eps=1e-6)
        v = v.transpose(1, 2)

        # Token-local dynamic learning rate eta_t in (0, base_lr)
        eta = torch.sigmoid(self.lr_proj(hidden_states)).transpose(1, 2).unsqueeze(-1) * self.base_lr

        if state is None:
            w = self.init_weight.to(device=device, dtype=dtype).expand(bsz, -1, -1, -1).clone()
            b = self.init_bias.to(device=device, dtype=dtype).expand(bsz, -1, -1, -1).clone()
        else:
            w = state.weight
            b = state.bias

        outputs: list[torch.Tensor] = []

        # Process sequence in fixed-size chunks for O(N) complexity & strict causality
        for start in range(0, seq_len, self.chunk_size):
            end = min(start + self.chunk_size, seq_len)
            chunk_len = end - start

            q_c = q[:, :, start:end, :]  # (B, H, L, d_h)
            k_c = k[:, :, start:end, :]
            v_c = v[:, :, start:end, :]
            eta_c = eta[:, :, start:end, :]  # (B, H, L, 1)

            # Reconstruction prediction using chunk-entry fast weights
            z_c = torch.matmul(k_c, w) + b  # (B, H, L, d_h)
            target_c = v_c - k_c
            raw_grad = z_c - target_c

            # Token-local soft norm bound guarantees numerical stability over 1M tokens
            grad_norm = raw_grad.norm(dim=-1, keepdim=True)
            grad_c = raw_grad / (grad_norm + 1.0)
            scaled_grad = eta_c * grad_c  # (B, H, L, d_h)

            # Dual-form strictly causal inner-chunk update:
            # y_i = q_i W + b - sum_{j<=i} (q_i . k_j) * scaled_grad_j - sum_{j<=i} scaled_grad_j
            attn_qk = torch.matmul(q_c, k_c.transpose(-1, -2))  # (B, H, L, L)
            causal_mask = torch.tril(
                torch.ones(chunk_len, chunk_len, device=device, dtype=dtype)
            )
            causal_qk = attn_qk * causal_mask

            delta_w_out = torch.matmul(causal_qk, scaled_grad)
            delta_b_out = torch.cumsum(scaled_grad, dim=-2)

            y_c = torch.matmul(q_c, w) + b - delta_w_out - delta_b_out
            y_c = self._head_layer_norm(y_c) + q_c
            outputs.append(y_c)

            # End-of-chunk fast-weight state transition with gentle decay
            w = self.weight_decay * w - torch.matmul(k_c.transpose(-1, -2), scaled_grad)
            b = self.weight_decay * b - scaled_grad.sum(dim=-2, keepdim=True)

        merged = torch.cat(outputs, dim=2).transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out: torch.Tensor = self.o_proj(merged * gate)
        return out, TTTState(weight=w, bias=b)
