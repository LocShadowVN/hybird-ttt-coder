from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import HybridTTTConfig
from model.recurrent_core import HybridDecoderBlock, RecurrentDepthCore, RMSNorm


@dataclass(slots=True)
class ModelOutput:
    """Structured forward output of HybridTTTLoopedCoder."""

    logits: torch.Tensor
    loss: torch.Tensor | None
    ce_loss: torch.Tensor | None
    aux_loss: torch.Tensor
    ponder_loss: torch.Tensor
    steps_used: int
    per_sample_steps: torch.Tensor
    halting_probs: torch.Tensor


class HybridTTTLoopedCoder(nn.Module):
    """Complete Hybrid-TTT-Coder model: Prelude -> Looped Recurrent Core -> Coda."""

    def __init__(self, config: HybridTTTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)

        self.prelude = nn.ModuleList(
            [HybridDecoderBlock(config, "ttt") for _ in range(config.num_prelude_layers)]
        )
        self.recurrent_core = RecurrentDepthCore(config)
        self.coda = nn.ModuleList(
            [HybridDecoderBlock(config, "ttt") for _ in range(config.num_coda_layers)]
        )

        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        num_steps: int | None = None,
        adaptive_halt: bool | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must have shape (B, T), got {tuple(input_ids.shape)}")

        h = self.embed_dropout(self.token_embeddings(input_ids))
        aux_losses: list[torch.Tensor] = []

        # 1. Prelude: initial contextual encoding
        for block in self.prelude:
            h, block_aux = block(h)
            aux_losses.append(block_aux)

        # 2. Recurrent Core: adaptive looped deep reasoning with Input Injection & Truncated BPTT
        core_out = self.recurrent_core(
            h,
            num_steps=num_steps,
            adaptive_halt=adaptive_halt,
        )
        h = core_out.hidden_states
        aux_losses.append(core_out.aux_loss)

        # 3. Coda: decode refined latent reasoning into token logits
        for block in self.coda:
            h, block_aux = block(h)
            aux_losses.append(block_aux)

        logits: torch.Tensor = self.lm_head(self.final_norm(h))
        total_aux_loss = torch.stack(aux_losses).mean()

        ce_loss: torch.Tensor | None = None
        total_loss: torch.Tensor | None = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if shift_labels.numel() > 0 and (shift_labels != -100).any():
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            else:
                ce_loss = logits.new_zeros(())
            total_loss = ce_loss + total_aux_loss + core_out.ponder_loss

        return ModelOutput(
            logits=logits,
            loss=total_loss,
            ce_loss=ce_loss,
            aux_loss=total_aux_loss,
            ponder_loss=core_out.ponder_loss,
            steps_used=core_out.steps_used,
            per_sample_steps=core_out.per_sample_steps,
            halting_probs=core_out.halting_probs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        num_steps: int | None = None,
        eos_token_id: int | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        """Autoregressive decoding returning generated sequence and recurrent steps per token."""
        if max_new_tokens <= 0:
            return input_ids, []

        was_training = self.training
        self.eval()
        generated = input_ids
        step_trace: list[int] = []

        try:
            for _ in range(max_new_tokens):
                window = generated[:, -self.config.max_position_embeddings :]
                out = self.forward(window, num_steps=num_steps, adaptive_halt=(num_steps is None))
                step_trace.append(out.steps_used)
                next_logits = out.logits[:, -1, :]

                if temperature <= 0.0:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    scaled = next_logits / temperature
                    sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cum_probs = torch.cumsum(probs, dim=-1)

                    remove_mask = cum_probs > top_p
                    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
                    remove_mask[..., 0] = False
                    filtered_logits = sorted_logits.masked_fill(remove_mask, -1e9)
                    filtered_probs = torch.softmax(filtered_logits, dim=-1)
                    sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
                    next_token = torch.gather(sorted_indices, dim=-1, index=sampled_idx)

                generated = torch.cat((generated, next_token), dim=1)
                if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                    break
        finally:
            if was_training:
                self.train()

        return generated, step_trace
