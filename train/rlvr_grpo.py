from __future__ import annotations

import copy
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import hybrid_ttt_native as native
from model.architecture import HybridTTTLoopedCoder
from train.sft_distill import ByteCodeTokenizer


@dataclass(frozen=True, slots=True)
class RLVRTask:
    """Single verifiable coding problem for RLVR-GRPO training."""

    problem_id: str
    prompt: str
    solution_filename: str
    test_code: str
    test_filename: str = "test_eval.py"
    code_extractor: Callable[[str], str] | None = None


@dataclass(frozen=True, slots=True)
class GRPOConfig:
    group_size: int = 4
    max_new_tokens: int = 32
    temperature: float = 0.9
    top_p: float = 0.95
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    recurrent_steps: int = 2
    sandbox_timeout_ms: int = 2_000
    sandbox_memory_mb: int = 1_024
    num_sandbox_threads: int | None = None


@dataclass(frozen=True, slots=True)
class GRPOStepMetrics:
    total_loss: float
    policy_loss: float
    kl_divergence: float
    aux_loss: float
    mean_reward: float
    pass_rate: float
    mean_advantage_std: float
    verified_completions: int


def _sequence_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Extract per-token log-probabilities strictly over the generated completion span."""
    # logits[:, t, :] predicts token_ids[:, t + 1]
    shift_logits = logits[:, prompt_len - 1 : -1, :]
    target_tokens = token_ids[:, prompt_len:]
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    gathered = torch.gather(log_probs, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
    return gathered


class GRPOTrainer:
    """Reinforcement Learning with Verifiable Rewards (RLVR) using Group Relative Policy
    Optimization (GRPO) coupled directly to Rust `hybrid_ttt_native.SandboxVerifier`."""

    def __init__(
        self,
        policy_model: HybridTTTLoopedCoder,
        tokenizer: ByteCodeTokenizer,
        config: GRPOConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy_model
        self.tokenizer = tokenizer
        self.config = config if config is not None else GRPOConfig()
        self.device = device if device is not None else next(policy_model.parameters()).device
        self.policy.to(self.device)

        # Frozen reference policy for KL regularization
        self.ref_policy = copy.deepcopy(policy_model).to(self.device)
        self.ref_policy.eval()
        for param in self.ref_policy.parameters():
            param.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
        )
        self.verifier = native.SandboxVerifier(
            config=native.SandboxConfig(
                timeout_ms=self.config.sandbox_timeout_ms,
                memory_limit_mb=self.config.sandbox_memory_mb,
            ),
            num_threads=self.config.num_sandbox_threads,
        )

    def _verify_rollout_batch(
        self,
        tasks: list[RLVRTask],
        decoded_groups: list[list[str]],
    ) -> list[list[native.VerificationResult]]:
        """Verify all (B x G) sampled code completions concurrently in Rust."""
        with tempfile.TemporaryDirectory(prefix="hybrid_ttt_grpo_") as root_tmp:
            root_path = Path(root_tmp)
            native_tasks: list[native.VerificationTask] = []

            for task_idx, (task, completions) in enumerate(
                zip(tasks, decoded_groups, strict=True)
            ):
                for g_idx, completion_text in enumerate(completions):
                    ws = root_path / f"p{task_idx}_g{g_idx}"
                    ws.mkdir(parents=True, exist_ok=True)
                    code_str = (
                        task.code_extractor(completion_text)
                        if task.code_extractor is not None
                        else completion_text
                    )
                    (ws / task.solution_filename).write_text(code_str, encoding="utf-8")
                    (ws / task.test_filename).write_text(task.test_code, encoding="utf-8")

                    native_tasks.append(
                        native.VerificationTask(
                            task_id=f"{task.problem_id}_g{g_idx}",
                            workdir=str(ws),
                            compile_cmd=[sys.executable, "-m", "py_compile", task.solution_filename],
                            lint_cmd=None,
                            test_cmd=[sys.executable, task.test_filename],
                            env_vars=[("PYTHONDONTWRITEBYTECODE", "1")],
                        )
                    )

            flat_results = self.verifier.verify_batch(native_tasks)

        grouped: list[list[native.VerificationResult]] = []
        g = self.config.group_size
        for task_idx in range(len(tasks)):
            grouped.append(flat_results[task_idx * g : (task_idx + 1) * g])
        return grouped

    def train_step(self, tasks: list[RLVRTask]) -> GRPOStepMetrics:
        if not tasks:
            raise ValueError("GRPO train_step requires at least one RLVRTask")

        g = self.config.group_size
        rollout_sequences: list[torch.Tensor] = []
        prompt_lengths: list[int] = []
        old_logprobs_list: list[torch.Tensor] = []
        ref_logprobs_list: list[torch.Tensor] = []
        decoded_groups: list[list[str]] = []

        # 1. Sample G completions per task under current policy without autograd
        self.policy.eval()
        with torch.no_grad():
            for task in tasks:
                prompt_tokens = [
                    self.tokenizer.BOS_ID,
                    *self.tokenizer.encode_text(task.prompt),
                ]
                prompt_len = len(prompt_tokens)
                prompt_tensor = torch.tensor(
                    [prompt_tokens] * g, dtype=torch.long, device=self.device
                )

                gen_ids, _ = self.policy.generate(
                    input_ids=prompt_tensor,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    num_steps=self.config.recurrent_steps,
                    eos_token_id=self.tokenizer.EOS_ID,
                )

                old_out = self.policy(
                    gen_ids,
                    num_steps=self.config.recurrent_steps,
                    adaptive_halt=False,
                )
                ref_out = self.ref_policy(
                    gen_ids,
                    num_steps=self.config.recurrent_steps,
                    adaptive_halt=False,
                )

                old_lp = _sequence_token_logprobs(old_out.logits, gen_ids, prompt_len)
                ref_lp = _sequence_token_logprobs(ref_out.logits, gen_ids, prompt_len)

                completions = [
                    self.tokenizer.decode_ids(gen_ids[i, prompt_len:].tolist()) for i in range(g)
                ]

                rollout_sequences.append(gen_ids)
                prompt_lengths.append(prompt_len)
                old_logprobs_list.append(old_lp)
                ref_logprobs_list.append(ref_lp)
                decoded_groups.append(completions)

        # 2. Verify all completions in parallel using Rust SandboxVerifier
        grouped_results = self._verify_rollout_batch(tasks, decoded_groups)

        # 3. Compute GRPO clipped surrogate objective + unbiased k3 KL divergence
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_policy_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_kl_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        total_aux_loss = torch.zeros((), device=self.device, dtype=torch.float32)
        all_rewards: list[float] = []
        passed_count = 0
        adv_stds: list[float] = []

        for gen_ids, prompt_len, old_lp, ref_lp, results in zip(
            rollout_sequences,
            prompt_lengths,
            old_logprobs_list,
            ref_logprobs_list,
            grouped_results,
            strict=True,
        ):
            rewards_vec = torch.tensor(
                [r.reward for r in results], device=self.device, dtype=torch.float32
            )
            all_rewards.extend(r.reward for r in results)
            passed_count += sum(1 for r in results if r.passed)

            # Group-relative advantage normalization (critic-free GRPO)
            mean_r = rewards_vec.mean()
            std_r = rewards_vec.std(unbiased=False)
            adv_stds.append(float(std_r.item()))
            advantages = (rewards_vec - mean_r) / (std_r + 1e-6)  # (G,)

            curr_out = self.policy(
                gen_ids,
                num_steps=self.config.recurrent_steps,
                adaptive_halt=False,
            )
            curr_lp = _sequence_token_logprobs(curr_out.logits, gen_ids, prompt_len)

            # Importance sampling ratio rho = exp(log pi_theta - log pi_old)
            log_ratio = torch.clamp(curr_lp - old_lp, min=-10.0, max=10.0)
            ratio = torch.exp(log_ratio)
            clipped_ratio = torch.clamp(
                ratio,
                min=1.0 - self.config.clip_epsilon,
                max=1.0 + self.config.clip_epsilon,
            )

            adv_expanded = advantages.unsqueeze(-1)  # (G, 1)
            surr1 = ratio * adv_expanded
            surr2 = clipped_ratio * adv_expanded
            policy_loss = -torch.minimum(surr1, surr2).mean()

            # Unbiased k3 KL estimator: exp(log_ref - log_curr) - (log_ref - log_curr) - 1 >= 0
            kl_diff = torch.clamp(ref_lp - curr_lp, min=-10.0, max=10.0)
            kl_div = (torch.exp(kl_diff) - kl_diff - 1.0).mean()

            total_policy_loss = total_policy_loss + policy_loss
            total_kl_loss = total_kl_loss + kl_div
            total_aux_loss = total_aux_loss + curr_out.aux_loss.float() + curr_out.ponder_loss.float()

        num_tasks = float(len(tasks))
        mean_policy_loss = total_policy_loss / num_tasks
        mean_kl = total_kl_loss / num_tasks
        mean_aux = total_aux_loss / num_tasks
        total_loss = mean_policy_loss + self.config.kl_beta * mean_kl + mean_aux

        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite GRPO loss: {total_loss.item()}")

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        total_verified = len(all_rewards)
        return GRPOStepMetrics(
            total_loss=float(total_loss.detach().cpu().item()),
            policy_loss=float(mean_policy_loss.detach().cpu().item()),
            kl_divergence=float(mean_kl.detach().cpu().item()),
            aux_loss=float(mean_aux.detach().cpu().item()),
            mean_reward=sum(all_rewards) / float(max(1, total_verified)),
            pass_rate=float(passed_count) / float(max(1, total_verified)),
            mean_advantage_std=sum(adv_stds) / float(max(1, len(adv_stds))),
            verified_completions=total_verified,
        )
