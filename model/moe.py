from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import HybridTTTConfig


@dataclass(slots=True)
class MoEOutput:
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    expert_counts: torch.Tensor


class SwiGLUExpert(nn.Module):
    """Single SwiGLU feed-forward expert."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return out


class FineGrainedMoE(nn.Module):
    """Fine-grained Mixture of Experts (default: 64 routed experts, Top-6 + 1 shared expert)
    with Switch-style load-balancing auxiliary loss and router z-loss."""

    def __init__(self, config: HybridTTTConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.aux_loss_coef = config.moe_aux_loss_coef
        self.z_loss_coef = config.moe_z_loss_coef

        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(self.hidden_size, config.moe_intermediate_size)
                for _ in range(self.num_experts)
            ]
        )
        self.shared_expert = SwiGLUExpert(self.hidden_size, config.shared_intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> MoEOutput:
        bsz, seq_len, dim = hidden_states.shape
        num_tokens = bsz * seq_len
        flat_x = hidden_states.reshape(num_tokens, dim)

        # Compute routing logits and probabilities in float32 for numerical stability
        router_logits = self.router(flat_x).float()  # (N, E)
        router_probs = torch.softmax(router_logits, dim=-1)

        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)  # (N, K)
        topk_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights.to(dtype=flat_x.dtype)

        # Load-balancing auxiliary loss: alpha * E * sum_i (f_i * P_i)
        flat_indices = topk_indices.reshape(-1)  # (N * K,)
        expert_counts = torch.bincount(flat_indices, minlength=self.num_experts).float()
        fraction_tokens = expert_counts / float(max(1, num_tokens * self.top_k))
        mean_router_prob = router_probs.mean(dim=0)
        load_balance_loss = self.num_experts * torch.sum(fraction_tokens * mean_router_prob)

        # Router z-loss discourages unbounded logit drift
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        aux_loss = (self.aux_loss_coef * load_balance_loss + self.z_loss_coef * z_loss).to(
            dtype=flat_x.dtype
        )

        # Sparse grouped token dispatch: zero FLOPs wasted on inactive experts
        token_ids = (
            torch.arange(num_tokens, device=flat_x.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        flat_weights = topk_weights.reshape(-1)

        sort_order = torch.argsort(flat_indices, stable=True)
        sorted_experts = flat_indices[sort_order]
        sorted_tokens = token_ids[sort_order]
        sorted_weights = flat_weights[sort_order]

        routed_out = torch.zeros_like(flat_x)
        counts_list = torch.bincount(sorted_experts, minlength=self.num_experts).tolist()

        cursor = 0
        for expert_idx, count in enumerate(counts_list):
            if count == 0:
                continue
            end = cursor + count
            tok_slice = sorted_tokens[cursor:end]
            weight_slice = sorted_weights[cursor:end].unsqueeze(-1)
            expert_in = flat_x.index_select(0, tok_slice)
            expert_out: torch.Tensor = self.experts[expert_idx](expert_in)
            routed_out.index_add_(0, tok_slice, expert_out * weight_slice)
            cursor = end

        shared_out = self.shared_expert(flat_x)
        combined = (routed_out + shared_out).view(bsz, seq_len, dim)

        return MoEOutput(
            hidden_states=combined,
            aux_loss=aux_loss,
            expert_counts=expert_counts.detach(),
        )
