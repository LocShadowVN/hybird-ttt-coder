from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

import hybrid_ttt_native as native
from model.architecture import HybridTTTLoopedCoder
from model.config import HybridTTTConfig
from runtime.evolutionary_loop import ArchitectureCandidate, EvolutionConfig, EvolutionaryLoop
from runtime.skill_library import SkillLibrary


@dataclass(frozen=True, slots=True)
class ContextMemoryPoint:
    seq_len: int
    latency_ms: float
    peak_python_mb: float
    peak_cuda_mb: float
    steps_used: int


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    preset: str
    device: str
    total_parameters: int
    full_16b_estimated_total: int
    full_16b_estimated_active: int
    forward_loss: float
    loss_is_finite: bool
    grad_norm: float
    prefill_tokens_per_sec: float
    generate_tokens_per_sec: float
    avg_recurrent_steps: float
    sandbox_batch_tasks: int
    sandbox_tasks_per_sec: float
    sandbox_timeout_enforced: bool
    sandbox_timeout_duration_ms: int
    git_commits_scanned: int
    git_samples_extracted: int
    git_commits_per_sec: float
    sample_suite_pass_rate: float
    evolution_improved: bool
    memory_scaling: tuple[ContextMemoryPoint, ...]


def _measure_peak_memory_for_context(
    model: HybridTTTLoopedCoder,
    seq_lengths: tuple[int, ...],
    device: torch.device,
) -> tuple[ContextMemoryPoint, ...]:
    model.eval()
    points: list[ContextMemoryPoint] = []
    vocab_size = model.config.vocab_size

    for seq_len in seq_lengths:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        tracemalloc.start()
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids, num_steps=2, adaptive_halt=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_cuda_mb = (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else 0.0
        )
        points.append(
            ContextMemoryPoint(
                seq_len=seq_len,
                latency_ms=round(elapsed_ms, 3),
                peak_python_mb=round(peak_bytes / (1024.0 * 1024.0), 3),
                peak_cuda_mb=round(peak_cuda_mb, 3),
                steps_used=out.steps_used,
            )
        )
    return tuple(points)


def _benchmark_rust_sandbox(batch_size: int = 12) -> tuple[int, float, bool, int]:
    with tempfile.TemporaryDirectory(prefix="hybrid_ttt_bench_sb_") as tmp:
        root = Path(tmp)
        tasks: list[native.VerificationTask] = []
        for idx in range(batch_size):
            ws = root / f"task_{idx}"
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "sol.py").write_text(
                f"def compute(x: int) -> int:\n    return x * 2 + {idx}\n",
                encoding="utf-8",
            )
            (ws / "test_sol.py").write_text(
                f"from sol import compute\nassert compute(5) == {10 + idx}\nprint('1 passed')\n",
                encoding="utf-8",
            )
            tasks.append(
                native.VerificationTask(
                    task_id=f"bench_{idx}",
                    workdir=str(ws),
                    compile_cmd=[sys.executable, "-m", "py_compile", "sol.py"],
                    test_cmd=[sys.executable, "test_sol.py"],
                )
            )

        verifier = native.SandboxVerifier(native.SandboxConfig(timeout_ms=3_000))
        t0 = time.perf_counter()
        results = verifier.verify_batch(tasks)
        elapsed = max(1e-6, time.perf_counter() - t0)
        if not all(r.passed for r in results):
            raise RuntimeError("benchmark sandbox verification failed unexpectedly")

        # Hard timeout verification probe
        timeout_verifier = native.SandboxVerifier(native.SandboxConfig(timeout_ms=150))
        timeout_task = native.VerificationTask(
            task_id="timeout_probe",
            workdir=str(root),
            test_cmd=[sys.executable, "-c", "import time; time.sleep(5)"],
        )
        timeout_res = timeout_verifier.verify(timeout_task)
        enforced = bool(
            timeout_res.timed_out
            and (not timeout_res.passed)
            and timeout_res.reward <= -0.99
            and timeout_res.duration_ms < 2_000
        )

    return batch_size, float(batch_size) / elapsed, enforced, int(timeout_res.duration_ms)


def _create_synthetic_git_repo(repo_dir: Path, num_commits: int = 30) -> None:
    def run_git(*args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repo_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    run_git("init")
    run_git("config", "user.name", "Hybrid Bench")
    run_git("config", "user.email", "bench@hybrid-ttt.dev")

    (repo_dir / "src").mkdir(parents=True, exist_ok=True)
    (repo_dir / "tests").mkdir(parents=True, exist_ok=True)

    for idx in range(num_commits):
        (repo_dir / "src" / "kernel.py").write_text(
            f"def scale_{idx}(x: int) -> int:\n    return x * {idx + 1}\n",
            encoding="utf-8",
        )
        (repo_dir / "tests" / "test_kernel.py").write_text(
            f"from src.kernel import scale_{idx}\n"
            f"def test_scale():\n    assert scale_{idx}(3) == {3 * (idx + 1)}\n",
            encoding="utf-8",
        )
        if idx % 5 == 0:
            (repo_dir / "Cargo.lock").write_text(f"# lock rev {idx}\n", encoding="utf-8")
            (repo_dir / "bundle.min.js").write_text(f"var x={idx};", encoding="utf-8")
        run_git("add", "-A")
        run_git("commit", "-m", f"feat: optimize kernel stage {idx}")


def _benchmark_rust_git_extractor(num_commits: int = 30) -> tuple[int, int, float]:
    with tempfile.TemporaryDirectory(prefix="hybrid_ttt_bench_git_") as tmp:
        root = Path(tmp)
        repo_dir = root / "repo"
        repo_dir.mkdir(parents=True, exist_ok=True)
        _create_synthetic_git_repo(repo_dir, num_commits=num_commits)

        out_jsonl = root / "commits.jsonl"
        cfg = native.ExtractorConfig(max_commits=num_commits + 10, require_tests=True)
        t0 = time.perf_counter()
        stats = native.extract_git_history(str(repo_dir), str(out_jsonl), cfg)
        elapsed = max(1e-6, time.perf_counter() - t0)

    return (
        int(stats.total_commits_scanned),
        int(stats.samples_extracted),
        float(stats.total_commits_scanned) / elapsed,
    )


def _run_sample_accuracy_and_evolution() -> tuple[float, bool]:
    lib = SkillLibrary()
    lib.upsert_skill(
        skill_id="seed_memo",
        name="Memoize Pure Recursion",
        category="optimization",
        principle="Apply lru_cache to overlapping recursive subproblems.",
        code_pattern="lru_cache",
        tags=("unmemoized_recursion", "dynamic_programming"),
        reward=0.95,
        passed=True,
    )

    loop = EvolutionaryLoop(
        skill_library=lib,
        config=EvolutionConfig(generations=1, mutations_per_generation=3, timeout_ms=2_000),
    )

    naive_fib = ArchitectureCandidate(
        candidate_id="fib_v0",
        module_filename="fib.py",
        source_code=(
            "def fib(n: int) -> int:\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n"
        ),
        mutation_name="baseline",
    )
    fib_tests = (
        "from fib import fib\n"
        "assert fib(0) == 0\n"
        "assert fib(1) == 1\n"
        "assert fib(10) == 55\n"
        "assert fib(26) == 121393\n"
        "print('4 passed')\n"
    )
    evo_reports = loop.run(naive_fib, fib_tests, problem_description="optimize recursive fib")
    evo_improved = any(rep.improved for rep in evo_reports)
    final_passed = evo_reports[-1].selected_result.passed if evo_reports else False

    # Additional deterministic coding suite verified in Rust sandbox
    with tempfile.TemporaryDirectory(prefix="hybrid_ttt_acc_") as tmp:
        root = Path(tmp)
        cases = [
            (
                "dedup",
                "def dedup(xs: list[int]) -> list[int]:\n"
                "    seen: set[int] = set()\n"
                "    out: list[int] = []\n"
                "    for x in xs:\n"
                "        if x not in seen:\n"
                "            seen.add(x)\n"
                "            out.append(x)\n"
                "    return out\n",
                "from sol import dedup\nassert dedup([3, 1, 3, 2, 1]) == [3, 1, 2]\n",
            ),
            (
                "binary_search",
                "def bsearch(arr: list[int], target: int) -> int:\n"
                "    lo, hi = 0, len(arr) - 1\n"
                "    while lo <= hi:\n"
                "        mid = (lo + hi) // 2\n"
                "        if arr[mid] == target:\n"
                "            return mid\n"
                "        if arr[mid] < target:\n"
                "            lo = mid + 1\n"
                "        else:\n"
                "            hi = mid - 1\n"
                "    return -1\n",
                "from sol import bsearch\nassert bsearch([1, 4, 7, 9], 7) == 2\n"
                "assert bsearch([1, 4, 7, 9], 5) == -1\n",
            ),
        ]
        tasks: list[native.VerificationTask] = []
        for name, code, test in cases:
            ws = root / name
            ws.mkdir(parents=True, exist_ok=True)
            (ws / "sol.py").write_text(code, encoding="utf-8")
            (ws / "test_sol.py").write_text(test, encoding="utf-8")
            tasks.append(
                native.VerificationTask(
                    task_id=name,
                    workdir=str(ws),
                    compile_cmd=[sys.executable, "-m", "py_compile", "sol.py"],
                    test_cmd=[sys.executable, "test_sol.py"],
                )
            )
        results = native.verify_batch(tasks, native.SandboxConfig(timeout_ms=2_000))

    total_cases = len(results) + 1
    passed_cases = sum(1 for r in results if r.passed) + (1 if final_passed else 0)
    return float(passed_cases) / float(total_cases), evo_improved


def run_benchmark_suite(preset: str = "tiny", seed: int = 42) -> BenchmarkReport:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if preset == "tiny":
        config = HybridTTTConfig.tiny_test()
        seq_lengths = (64, 128, 256, 512)
        gen_tokens = 24
    elif preset == "small":
        config = HybridTTTConfig(
            vocab_size=1_024,
            hidden_size=128,
            num_heads=4,
            num_kv_heads=2,
            num_prelude_layers=1,
            num_core_layers=8,
            num_coda_layers=1,
            core_layer_pattern=("ttt", "ttt", "ttt", "ttt", "ttt", "ttt", "ttt", "sparse_attn"),
            ttt_chunk_size=16,
            sparse_block_size=16,
            sparse_top_k_blocks=2,
            sliding_window_size=32,
            moe_num_experts=16,
            moe_top_k=4,
            moe_intermediate_size=64,
            shared_intermediate_size=128,
            max_recurrent_steps=8,
            truncated_bptt_steps=2,
        )
        seq_lengths = (64, 128, 256, 512)
        gen_tokens = 20
    else:
        raise ValueError(f"unsupported benchmark preset: {preset}")

    full_estimates = HybridTTTConfig.full_16b().estimate_parameter_counts()
    model = HybridTTTLoopedCoder(config).to(device)

    # 1. Forward + Backward numerical sanity check
    model.train()
    batch_ids = torch.randint(0, config.vocab_size, (2, 64), device=device)
    t_prefill_start = time.perf_counter()
    out = model(batch_ids, labels=batch_ids, num_steps=4, adaptive_halt=False)
    if out.loss is None:
        raise RuntimeError("expected loss from forward pass with labels")
    out.loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_elapsed = max(1e-6, time.perf_counter() - t_prefill_start)
    prefill_tps = float(batch_ids.numel()) / prefill_elapsed

    grad_sq_sum = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_sq_sum += float(param.grad.detach().float().pow(2).sum().item())
    grad_norm = math.sqrt(grad_sq_sum)
    loss_val = float(out.loss.detach().cpu().item())
    loss_finite = math.isfinite(loss_val) and math.isfinite(grad_norm)

    # 2. Autoregressive token generation throughput
    model.eval()
    prompt_ids = torch.randint(0, config.vocab_size, (1, 16), device=device)
    t_gen_start = time.perf_counter()
    _, step_trace = model.generate(prompt_ids, max_new_tokens=gen_tokens, temperature=0.8)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gen_elapsed = max(1e-6, time.perf_counter() - t_gen_start)
    gen_tps = float(len(step_trace)) / gen_elapsed
    avg_steps = sum(step_trace) / float(max(1, len(step_trace)))

    # 3. Context-length memory & latency scaling
    memory_scaling = _measure_peak_memory_for_context(model, seq_lengths, device)

    # 4. Rust Sandbox & Git Extractor throughput benchmarks
    sb_tasks, sb_tps, sb_timeout_ok, sb_timeout_ms = _benchmark_rust_sandbox()
    git_scanned, git_extracted, git_cps = _benchmark_rust_git_extractor()

    # 5. Accuracy & Self-Evolution check
    pass_rate, evo_improved = _run_sample_accuracy_and_evolution()

    return BenchmarkReport(
        preset=preset,
        device=str(device),
        total_parameters=model.count_parameters(),
        full_16b_estimated_total=full_estimates["total_params"],
        full_16b_estimated_active=full_estimates["active_params_per_token"],
        forward_loss=round(loss_val, 5),
        loss_is_finite=loss_finite,
        grad_norm=round(grad_norm, 5),
        prefill_tokens_per_sec=round(prefill_tps, 2),
        generate_tokens_per_sec=round(gen_tps, 2),
        avg_recurrent_steps=round(avg_steps, 2),
        sandbox_batch_tasks=sb_tasks,
        sandbox_tasks_per_sec=round(sb_tps, 2),
        sandbox_timeout_enforced=sb_timeout_ok,
        sandbox_timeout_duration_ms=sb_timeout_ms,
        git_commits_scanned=git_scanned,
        git_samples_extracted=git_extracted,
        git_commits_per_sec=round(git_cps, 2),
        sample_suite_pass_rate=round(pass_rate, 4),
        evolution_improved=evo_improved,
        memory_scaling=memory_scaling,
    )


def format_markdown_report(report: BenchmarkReport) -> str:
    lines = [
        "## Hybrid-TTT-Coder Automated Evaluation Report",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Model Preset / Device | `{report.preset}` / `{report.device}` |",
        f"| Active Test Parameters | `{report.total_parameters:,}` |",
        f"| Full 16B Spec (Total / Active) | `{report.full_16b_estimated_total / 1e9:.2f}B` / `{report.full_16b_estimated_active / 1e9:.2f}B` |",
        f"| Forward Loss (Finite) | `{report.forward_loss:.5f}` (`{report.loss_is_finite}`) |",
        f"| Backward Grad Norm | `{report.grad_norm:.5f}` |",
        f"| Prefill Throughput | `{report.prefill_tokens_per_sec:.2f} tokens/sec` |",
        f"| Decode Throughput | `{report.generate_tokens_per_sec:.2f} tokens/sec` |",
        f"| Avg Adaptive Recurrent Steps | `{report.avg_recurrent_steps:.2f}` |",
        f"| Rust Sandbox Throughput ({report.sandbox_batch_tasks} tasks) | `{report.sandbox_tasks_per_sec:.2f} tasks/sec` |",
        f"| Rust Sandbox Hard Timeout Enforced | `{report.sandbox_timeout_enforced}` (`{report.sandbox_timeout_duration_ms} ms`) |",
        f"| Rust Git Extractor Throughput | `{report.git_commits_per_sec:.2f} commits/sec` (`{report.git_samples_extracted}/{report.git_commits_scanned}`) |",
        f"| Sample Coding Suite Pass Rate | `{report.sample_suite_pass_rate * 100.0:.1f}%` |",
        f"| Evolutionary Loop Improved | `{report.evolution_improved}` |",
        "",
        "### Context Length Scaling",
        "",
        "| Context Length | Latency (ms) | Peak Python RAM (MB) | Peak CUDA VRAM (MB) | Recurrent Steps |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for pt in report.memory_scaling:
        lines.append(
            f"| `{pt.seq_len}` | `{pt.latency_ms:.2f}` | `{pt.peak_python_mb:.2f}` "
            f"| `{pt.peak_cuda_mb:.2f}` | `{pt.steps_used}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _serialize_report(report: BenchmarkReport) -> dict[str, Any]:
    raw = asdict(report)
    raw["memory_scaling"] = [asdict(pt) for pt in report.memory_scaling]
    return raw


def _enforce_ci_gates(report: BenchmarkReport) -> None:
    failures: list[str] = []
    if not report.loss_is_finite:
        failures.append(f"non-finite forward loss or grad norm ({report.forward_loss})")
    if report.generate_tokens_per_sec <= 1.0:
        failures.append(f"generation throughput too low: {report.generate_tokens_per_sec} tok/s")
    if not report.sandbox_timeout_enforced:
        failures.append("Rust sandbox hard timeout enforcement check failed")
    if report.sandbox_tasks_per_sec <= 1.0:
        failures.append(f"Rust sandbox throughput too low: {report.sandbox_tasks_per_sec} tasks/s")
    if report.git_samples_extracted == 0:
        failures.append("Rust git history extractor returned zero samples")
    if report.sample_suite_pass_rate < 1.0:
        failures.append(f"sample coding suite pass rate below 100%: {report.sample_suite_pass_rate}")
    if failures:
        raise SystemExit("CI evaluation gate failed:\n- " + "\n- ".join(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hybrid-TTT-Coder Benchmark & CI Gate")
    parser.add_argument("--preset", choices=("tiny", "small"), default="tiny")
    parser.add_argument("--output", type=Path, default=Path("eval_report.json"))
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--strict-ci", action="store_true")
    args = parser.parse_args(argv)

    report = run_benchmark_suite(preset=args.preset)
    markdown = format_markdown_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_serialize_report(report), indent=2), encoding="utf-8")

    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")

    print(markdown)
    if args.strict_ci:
        _enforce_ci_gates(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
