from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import HybridTTTConfig


class RotaryEmbedding(nn.Module):
    """On-the-fly Rotary Position Embeddings supporting up to 1M context without
    pre-allocating gigantic sin/cos tables."""

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, theta: float = 1_000_000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype, offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bsz, n_kv, seq_len, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bsz, n_kv, n_rep, seq_len, head_dim)
        .reshape(bsz, n_kv * n_rep, seq_len, head_dim)
    )


def _safe_masked_softmax(scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Compute numerically safe float32 softmax returning 0.0 (and finite 0 gradients)
    for query rows that have no valid keys."""
    has_valid = valid_mask.any(dim=-1, keepdim=True)
    safe_scores = scores.float().masked_fill(~valid_mask, -1e9)
    probs = torch.softmax(safe_scores, dim=-1)
    probs = probs * has_valid.to(probs.dtype)
    return probs.to(scores.dtype)


class NativeSparseAttention(nn.Module):
    """Native Sparse Attention (NSA) designed for 1M-token coding contexts.

    Combines three strictly causal branches via learned per-head gates:
      1. Compressed Coarse-Grained Attention over strictly completed past blocks (m < b(t)).
      2. Selected Top-K Fine-Grained Block Attention retrieving full token resolution from
         the highest-importance past blocks.
      3. Chunked Sliding Window Attention over [max(0, t - W + 1), t] in O(T * W) memory.
    """

    def __init__(self, config: HybridTTTConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.block_size = config.sparse_block_size
        self.top_k_blocks = config.sparse_top_k_blocks
        self.window_size = config.sliding_window_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Intra-block positional compression weights + projections for K, V
        self.k_block_pos = nn.Parameter(torch.full((self.block_size,), 1.0 / self.block_size))
        self.v_block_pos = nn.Parameter(torch.full((self.block_size,), 1.0 / self.block_size))
        self.k_compress = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.v_compress = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Learned per-head branch mixing gates: [compressed, selected, sliding_window]
        self.branch_gate = nn.Linear(self.hidden_size, self.num_heads * 3, bias=True)
        nn.init.zeros_(self.branch_gate.bias)

        self.rotary = RotaryEmbedding(self.head_dim, theta=config.rope_theta)

    def _sliding_window_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Exact causal sliding-window attention in O(T * W) memory without T x T allocation."""
        bsz, n_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype
        w = self.window_size

        num_chunks = (seq_len + w - 1) // w
        pad_right = num_chunks * w - seq_len

        q_pad = F.pad(q, (0, 0, 0, pad_right)) if pad_right > 0 else q
        # Pad W zeros on the left so chunk c slices [c*W : (c+2)*W] cleanly
        k_pad = F.pad(k, (0, 0, w, pad_right))
        v_pad = F.pad(v, (0, 0, w, pad_right))

        q_chunks = q_pad.view(bsz, n_heads, num_chunks, w, head_dim)
        chunk_outputs: list[torch.Tensor] = []

        local_q = torch.arange(w, device=device)[:, None]  # (W, 1)
        local_k = torch.arange(2 * w, device=device)[None, :]  # (1, 2W)

        for c in range(num_chunks):
            qc = q_chunks[:, :, c, :, :]  # (B, H, W, d_h)
            kc = k_pad[:, :, c * w : (c + 2) * w, :]  # (B, H, 2W, d_h)
            vc = v_pad[:, :, c * w : (c + 2) * w, :]

            global_q = c * w + local_q  # (W, 1)
            global_k = (c - 1) * w + local_k  # (1, 2W)

            # Strictly causal & inside sliding window & within real sequence bounds
            valid_mask = (
                (global_k >= 0)
                & (global_k <= global_q)
                & ((global_q - global_k) < w)
                & (global_q < seq_len)
            )
            scores = torch.matmul(qc, kc.transpose(-1, -2)) * self.scale
            probs = _safe_masked_softmax(scores, valid_mask[None, None, :, :])
            out_c = torch.matmul(probs, vc)
            chunk_outputs.append(out_c)

        out = torch.cat(chunk_outputs, dim=2)
        return out[:, :, :seq_len, :]

    def _compressed_and_selected_attention(
        self,
        q: torch.Tensor,
        k_kv: torch.Tensor,
        v_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute strictly causal block-compressed attention and Top-K selected
        fine-grained block attention over strictly past blocks (m < b(t))."""
        bsz, n_heads, seq_len, head_dim = q.shape
        device = q.device
        dtype = q.dtype
        bs = self.block_size

        num_blocks = (seq_len + bs - 1) // bs
        if num_blocks <= 1:
            zeros = torch.zeros_like(q)
            return zeros, zeros

        pad_len = num_blocks * bs - seq_len
        k_padded = F.pad(k_kv, (0, 0, 0, pad_len)) if pad_len > 0 else k_kv
        v_padded = F.pad(v_kv, (0, 0, 0, pad_len)) if pad_len > 0 else v_kv

        # Shape: (B, H_kv, N_b, B_s, d_h)
        k_blocks = k_padded.view(bsz, self.num_kv_heads, num_blocks, bs, head_dim)
        v_blocks = v_padded.view(bsz, self.num_kv_heads, num_blocks, bs, head_dim)

        k_weights = torch.softmax(self.k_block_pos, dim=0).to(dtype=dtype)[None, None, None, :, None]
        v_weights = torch.softmax(self.v_block_pos, dim=0).to(dtype=dtype)[None, None, None, :, None]

        # Compressed block tokens: (B, H_kv, N_b, d_h)
        k_cmp_kv = self.k_compress((k_blocks * k_weights).sum(dim=3))
        v_cmp_kv = self.v_compress((v_blocks * v_weights).sum(dim=3))

        k_cmp = _repeat_kv(k_cmp_kv, self.num_kv_groups)
        v_cmp = _repeat_kv(v_cmp_kv, self.num_kv_groups)

        # Strict causal block mask: token t in block b(t) may ONLY attend to blocks m < b(t)
        q_block_ids = torch.arange(seq_len, device=device) // bs  # (T,)
        block_ids = torch.arange(num_blocks, device=device)  # (N_b,)
        causal_block_mask = block_ids[None, :] < q_block_ids[:, None]  # (T, N_b)

        # 1. Compressed coarse-grained attention
        cmp_scores = torch.matmul(q, k_cmp.transpose(-1, -2)) * self.scale  # (B, H, T, N_b)
        cmp_probs = _safe_masked_softmax(cmp_scores, causal_block_mask[None, None, :, :])
        out_cmp = torch.matmul(cmp_probs, v_cmp)

        # 2. Top-K block selection grouped across KV head groups for memory efficiency
        # Aggregate importance probabilities across each GQA group: (B, H_kv, T, N_b)
        group_importance = cmp_probs.view(
            bsz, self.num_kv_heads, self.num_kv_groups, seq_len, num_blocks
        ).mean(dim=2)
        masked_importance = group_importance.masked_fill(
            ~causal_block_mask[None, None, :, :], -1.0
        )

        eff_k = min(self.top_k_blocks, num_blocks - 1)
        _, topk_indices = torch.topk(masked_importance, k=eff_k, dim=-1)  # (B, H_kv, T, eff_k)

        # Verify each selected block is strictly in the past (m < b(t))
        valid_selected_blocks = topk_indices < q_block_ids[None, None, :, None]  # (B, H_kv, T, eff_k)

        # Gather fine-grained K, V tokens for the selected Top-K blocks
        # k_blocks: (B, H_kv, N_b, B_s, d_h) -> gather along N_b using flat indices
        flat_idx = topk_indices.reshape(bsz, self.num_kv_heads, seq_len * eff_k)
        gather_idx = flat_idx[:, :, :, None, None].expand(-1, -1, -1, bs, head_dim)

        sel_k_kv = torch.gather(k_blocks, dim=2, index=gather_idx).view(
            bsz, self.num_kv_heads, seq_len, eff_k * bs, head_dim
        )
        sel_v_kv = torch.gather(v_blocks, dim=2, index=gather_idx).view(
            bsz, self.num_kv_heads, seq_len, eff_k * bs, head_dim
        )

        # Expand valid block mask to per-token mask: (B, H_kv, T, eff_k * B_s)
        valid_token_mask_kv = (
            valid_selected_blocks.unsqueeze(-1)
            .expand(-1, -1, -1, -1, bs)
            .reshape(bsz, self.num_kv_heads, seq_len, eff_k * bs)
        )

        # Expand across GQA groups to match Q heads: (B, H_kv, G, T, ...)
        q_grouped = q.view(bsz, self.num_kv_heads, self.num_kv_groups, seq_len, head_dim)
        # Dot product: (B, H_kv, G, T, 1, d_h) @ (B, H_kv, 1, T, d_h, K*B_s) -> (B, H_kv, G, T, K*B_s)
        sel_scores = (
            torch.einsum("bngtd,bntsd->bngts", q_grouped, sel_k_kv) * self.scale
        )
        sel_probs = _safe_masked_softmax(sel_scores, valid_token_mask_kv[:, :, None, :, :])
        out_slc_grouped = torch.einsum("bngts,bntsd->bngtd", sel_probs, sel_v_kv)
        out_slc = out_slc_grouped.reshape(bsz, n_heads, seq_len, head_dim)

        return out_cmp, out_slc

    def forward(self, hidden_states: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_kv = (
            self.k_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_kv = (
            self.v_proj(hidden_states)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary(seq_len, device=device, dtype=dtype, offset=position_offset)
        q = _apply_rope(q, cos, sin)
        k_kv = _apply_rope(k_kv, cos, sin)

        k_full = _repeat_kv(k_kv, self.num_kv_groups)
        v_full = _repeat_kv(v_kv, self.num_kv_groups)

        out_win = self._sliding_window_attention(q, k_full, v_full)
        out_cmp, out_slc = self._compressed_and_selected_attention(q, k_kv, v_kv)

        # Per-head input-dependent branch gating
        gates = torch.sigmoid(
            self.branch_gate(hidden_states).view(bsz, seq_len, self.num_heads, 3)
        ).permute(0, 2, 1, 3)  # (B, H, T, 3)
        g_cmp = gates[..., 0:1]
        g_slc = gates[..., 1:2]
        g_win = gates[..., 2:3]

        fused = g_cmp * out_cmp + g_slc * out_slc + g_win * out_win
        merged = fused.transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)
        out: torch.Tensor = self.o_proj(merged)
        return out
