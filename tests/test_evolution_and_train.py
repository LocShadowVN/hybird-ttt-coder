from __future__ import annotations

import json
from pathlib import Path

import torch

from eval.benchmark import format_markdown_report, run_benchmark_suite
from model.architecture import HybridTTTLoopedCoder
from model.config import HybridTTTConfig
from runtime.evolutionary_loop import (
    ArchitectureCandidate,
    EvolutionConfig,
    EvolutionaryLoop,
    analyze_source_bottlenecks,
)
from runtime.skill_library import SkillLibrary
from train.rlvr_grpo import GRPOConfig, GRPOTrainer, RLVRTask
from train.sft_distill import ByteCodeTokenizer, GitDiffJSONLDataset, SFTConfig, SFTTrainer


def test_skill_library_graph_query_and_persistence(tmp_path: Path) -> None:
    lib = SkillLibrary()
    lib.upsert_skill(
        skill_id="memo",
        name="Memoization",
        category="optimization",
        principle="Cache recursive calls using lru_cache to reduce exponential to linear time.",
        code_pattern="@lru_cache",
        tags=("recursion", "cache"),
        reward=0.95,
        passed=True,
    )
    lib.upsert_skill(
        skill_id="dp_table",
        name="Bottom-Up DP",
        category="optimization",
        principle="Convert memoized recursion into iterative table filling for zero stack overhead.",
        code_pattern="dp = [0] * (n + 1)",
        tags=("iterative", "table"),
        reward=0.98,
        passed=True,
    )
    lib.link_skills("memo", "dp_table", relation="refines", weight=1.2)
    lib.record_outcome("memo", reward=0.99, passed=True)

    # Query matches "recursion" directly on `memo` and reaches `dp_table` via 1-hop graph propagation
    results = lib.query("fix recursion bottleneck", top_k=5, hops=2)
    ids = [r.skill.skill_id for r in results]
    assert ids[0] == "memo"
    assert "dp_table" in ids

    save_path = tmp_path / "skills.json"
    lib.save(save_path)
    reloaded = SkillLibrary.load(save_path)
    assert len(reloaded) == 2
    assert len(reloaded.list_edges()) == 1


def test_evolutionary_loop_end_to_end_optimization() -> None:
    lib = SkillLibrary()
    lib.upsert_skill(
        skill_id="hash_set",
        name="Hash Set Lookup",
        category="optimization",
        principle="Replace O(N) list membership checks inside loops with O(1) hash set lookups.",
        code_pattern="seen = set()",
        tags=("linear_membership_scan_in_loop", "hashing"),
        reward=0.92,
        passed=True,
    )

    slow_dedup = ArchitectureCandidate(
        candidate_id="dedup_v0",
        module_filename="dedup.py",
        source_code=(
            "def unique_preserve(items: list[int]) -> list[int]:\n"
            "    seen = []\n"
            "    for x in items:\n"
            "        if x not in seen:\n"
            "            seen.append(x)\n"
            "    return list(seen)\n"
        ),
        mutation_name="baseline",
    )
    test_suite = (
        "from dedup import unique_preserve\n"
        "data = list(range(3500)) + list(range(3500))\n"
        "out = unique_preserve(data)\n"
        "assert len(out) == 3500\n"
        "assert set(out) == set(range(3500))\n"
        "print('2 passed')\n"
    )

    diff_info = analyze_source_bottlenecks("", slow_dedup.source_code)
    assert "linear_membership_scan_in_loop" in diff_info.bottlenecks

    loop = EvolutionaryLoop(
        skill_library=lib,
        config=EvolutionConfig(generations=1, mutations_per_generation=3, timeout_ms=3_000),
    )
    reports = loop.run(slow_dedup, test_suite, problem_description="optimize dedup scan")
    assert len(reports) == 1
    assert reports[0].selected_result.passed is True
    assert reports[0].improved is True
    assert "seen = set()" in reports[0].selected_candidate.source_code
    assert reports[0].learned_skill_id is not None
    assert lib.get(reports[0].learned_skill_id) is not None


def test_sft_distill_and_rlvr_grpo_pipelines(tmp_path: Path) -> None:
    torch.manual_seed(101)
    cfg = HybridTTTConfig.tiny_test()
    model = HybridTTTLoopedCoder(cfg)
    tokenizer = ByteCodeTokenizer(vocab_size=cfg.vocab_size)

    # 1. SFT Distillation reading JSONL matching Rust GitHistoryExtractor schema
    jsonl_path = tmp_path / "git_commits.jsonl"
    rows = [
        {
            "commit_hash": "a1b2c3d4",
            "author_timestamp": 1_730_000_000,
            "commit_message": "feat: implement integer clamp helper",
            "context_files": ["src/util.py", "tests/test_util.py"],
            "code_diff": "+def clamp(x: int, lo: int, hi: int) -> int:\n+    return max(lo, min(hi, x))\n",
            "test_diff": "+def test_clamp():\n+    assert clamp(10, 0, 5) == 5\n",
            "files_changed": 2,
            "lines_added": 4,
            "lines_deleted": 0,
        },
        {
            "commit_hash": "e5f6a7b8",
            "author_timestamp": 1_730_000_100,
            "commit_message": "fix: guard empty list mean",
            "context_files": ["src/stats.py", "tests/test_stats.py"],
            "code_diff": "+def mean(xs: list[float]) -> float:\n+    return sum(xs) / len(xs) if xs else 0.0\n",
            "test_diff": "+def test_mean():\n+    assert mean([]) == 0.0\n",
            "files_changed": 2,
            "lines_added": 4,
            "lines_deleted": 0,
        },
    ]
    jsonl_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    dataset = GitDiffJSONLDataset(jsonl_path, tokenizer, max_seq_len=96)
    sft_trainer = SFTTrainer(
        model,
        SFTConfig(batch_size=2, max_steps=3, warmup_steps=1, recurrent_steps=2),
    )
    sft_metrics = sft_trainer.train(dataset)
    assert sft_metrics.steps_completed == 3
    assert sft_metrics.tokens_per_sec > 0.0

    # 2. RLVR-GRPO step backed by live Rust SandboxVerifier
    # Deterministic extractor converts noisy random tiny-model tokens into valid/invalid variants
    # so we verify genuine reward variance and GRPO advantage updates end-to-end.
    def extract_candidate(raw_text: str) -> str:
        if len(raw_text) % 2 == 0:
            return "def add(a: int, b: int) -> int:\n    return a + b\n"
        return "def add(a: int, b: int) -> int:\n    return a - b\n"

    rlvr_task = RLVRTask(
        problem_id="add_two",
        prompt="Implement add(a, b) returning a + b",
        solution_filename="sol.py",
        test_code="from sol import add\nassert add(2, 3) == 5\nprint('1 passed')\n",
        code_extractor=extract_candidate,
    )
    grpo_trainer = GRPOTrainer(
        policy_model=model,
        tokenizer=tokenizer,
        config=GRPOConfig(
            group_size=4,
            max_new_tokens=8,
            recurrent_steps=2,
            sandbox_timeout_ms=2_000,
        ),
    )
    grpo_metrics = grpo_trainer.train_step([rlvr_task])
    assert grpo_metrics.verified_completions == 4
    assert grpo_metrics.kl_divergence >= 0.0


def test_benchmark_suite_and_markdown_rendering() -> None:
    report = run_benchmark_suite(preset="tiny", seed=7)
    assert report.loss_is_finite is True
    assert report.sandbox_timeout_enforced is True
    assert report.git_samples_extracted > 0
    assert report.sample_suite_pass_rate == 1.0
    md = format_markdown_report(report)
    assert "Hybrid-TTT-Coder Automated Evaluation Report" in md
    assert "Context Length Scaling" in md
