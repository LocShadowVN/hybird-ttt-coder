from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model.architecture import HybridTTTLoopedCoder


class ByteCodeTokenizer:
    """Self-contained byte-level coding tokenizer with structural diff tokens.
    Requires vocab_size >= 264 and guarantees zero OOV failures on arbitrary source code."""

    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    COMMIT_ID: int = 3
    CONTEXT_ID: int = 4
    CODE_DIFF_ID: int = 5
    TEST_DIFF_ID: int = 6
    SEP_ID: int = 7
    BYTE_OFFSET: int = 8

    def __init__(self, vocab_size: int = 512) -> None:
        if vocab_size < 264:
            raise ValueError("ByteCodeTokenizer requires vocab_size >= 264")
        self.vocab_size = vocab_size

    def encode_text(self, text: str) -> list[int]:
        return [self.BYTE_OFFSET + b for b in text.encode("utf-8", errors="replace")]

    def decode_ids(self, token_ids: list[int]) -> str:
        raw = bytes(
            tid - self.BYTE_OFFSET
            for tid in token_ids
            if self.BYTE_OFFSET <= tid < self.BYTE_OFFSET + 256
        )
        return raw.decode("utf-8", errors="replace")

    def format_git_sample(
        self,
        commit_message: str,
        context_files: list[str],
        code_diff: str,
        test_diff: str,
        max_length: int,
    ) -> tuple[list[int], list[int]]:
        """Encode (Commit + Context -> Code Diff + Test Diff) with prompt labels masked to -100."""
        prompt_ids: list[int] = [
            self.BOS_ID,
            self.COMMIT_ID,
            *self.encode_text(commit_message),
            self.CONTEXT_ID,
            *self.encode_text(", ".join(context_files)),
            self.CODE_DIFF_ID,
        ]
        target_ids: list[int] = [
            *self.encode_text(code_diff),
            self.TEST_DIFF_ID,
            *self.encode_text(test_diff),
            self.EOS_ID,
        ]

        # Reserve at least half the sequence budget for target diff tokens if prompt is long
        max_prompt = max(8, max_length // 2)
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[:max_prompt]

        combined = (prompt_ids + target_ids)[:max_length]
        prompt_len = min(len(prompt_ids), len(combined))

        labels = [-100] * prompt_len + combined[prompt_len:]
        if all(lbl == -100 for lbl in labels):
            labels = list(combined)

        return combined, labels


class GitDiffJSONLDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset reading JSONL files emitted directly by Rust `GitHistoryExtractor`."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: ByteCodeTokenizer,
        max_seq_len: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[tuple[list[int], list[int]]] = []

        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"SFT dataset not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                ids, labels = tokenizer.format_git_sample(
                    commit_message=str(row.get("commit_message", "")),
                    context_files=[str(f) for f in row.get("context_files", [])],
                    code_diff=str(row.get("code_diff", "")),
                    test_diff=str(row.get("test_diff", "")),
                    max_length=max_seq_len,
                )
                if len(ids) >= 2:
                    self.samples.append((ids, labels))

        if not self.samples:
            raise ValueError(f"no valid commit diff samples found in {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        ids, labels = self.samples[index]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


def collate_padded_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    pad_id: int = ByteCodeTokenizer.PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(item[0].shape[0] for item in batch)
    bsz = len(batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for idx, (ids, lbl) in enumerate(batch):
        length = ids.shape[0]
        input_ids[idx, :length] = ids
        labels[idx, :length] = lbl
    return input_ids, labels


@dataclass(frozen=True, slots=True)
class SFTConfig:
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_steps: int = 10
    warmup_steps: int = 2
    max_grad_norm: float = 1.0
    recurrent_steps: int | None = None
    max_seq_len: int = 256


@dataclass(frozen=True, slots=True)
class SFTTrainMetrics:
    steps_completed: int
    initial_loss: float
    final_loss: float
    mean_aux_loss: float
    mean_ponder_loss: float
    tokens_per_sec: float
    loss_history: tuple[float, ...]


class SFTTrainer:
    """Supervised Fine-Tuning / Distillation trainer for HybridTTTLoopedCoder."""

    def __init__(
        self,
        model: HybridTTTLoopedCoder,
        config: SFTConfig,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device if device is not None else next(model.parameters()).device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

    def _compute_lr(self, step: int) -> float:
        if step < self.config.warmup_steps:
            return self.config.learning_rate * float(step + 1) / float(max(1, self.config.warmup_steps))
        progress = float(step - self.config.warmup_steps) / float(
            max(1, self.config.max_steps - self.config.warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return self.config.learning_rate * (0.1 + 0.9 * cosine)

    def train(self, dataset: GitDiffJSONLDataset) -> SFTTrainMetrics:
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_padded_batch,
        )

        self.model.train()
        loss_history: list[float] = []
        aux_losses: list[float] = []
        ponder_losses: list[float] = []
        total_tokens = 0
        start_time = time.perf_counter()

        step = 0
        while step < self.config.max_steps:
            for input_ids, labels in loader:
                if step >= self.config.max_steps:
                    break
                lr = self._compute_lr(step)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(
                    input_ids=input_ids,
                    labels=labels,
                    num_steps=self.config.recurrent_steps,
                    adaptive_halt=False,
                )
                if out.loss is None:
                    raise RuntimeError("SFT forward pass did not compute loss")
                if not torch.isfinite(out.loss):
                    raise FloatingPointError(f"non-finite SFT loss at step {step}: {out.loss.item()}")

                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                loss_history.append(float(out.loss.detach().cpu().item()))
                aux_losses.append(float(out.aux_loss.detach().cpu().item()))
                ponder_losses.append(float(out.ponder_loss.detach().cpu().item()))
                total_tokens += int(input_ids.numel())
                step += 1

        elapsed = max(1e-6, time.perf_counter() - start_time)
        return SFTTrainMetrics(
            steps_completed=step,
            initial_loss=loss_history[0],
            final_loss=loss_history[-1],
            mean_aux_loss=sum(aux_losses) / float(len(aux_losses)),
            mean_ponder_loss=sum(ponder_losses) / float(len(ponder_losses)),
            tokens_per_sec=float(total_tokens) / elapsed,
            loss_history=tuple(loss_history),
        )
