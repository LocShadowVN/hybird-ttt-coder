from __future__ import annotations

import json
import subprocess
import sys
import tracemalloc
from pathlib import Path

import pytest

import hybrid_ttt_native as native


def test_sandbox_verifier_parallel_batch_and_timeout(tmp_path: Path) -> None:
    pass_dir = tmp_path / "pass_ws"
    pass_dir.mkdir()
    (pass_dir / "mod.py").write_text("def double(x: int) -> int:\n    return x * 2\n")
    (pass_dir / "test_mod.py").write_text(
        "from mod import double\nassert double(21) == 42\nprint('2 passed in 0.01s')\n"
    )

    fail_dir = tmp_path / "fail_ws"
    fail_dir.mkdir()
    (fail_dir / "mod.py").write_text("def double(x: int) -> int:\n    return x + 1\n")
    (fail_dir / "test_mod.py").write_text(
        "import sys\nprint('1 passed, 1 failed in 0.01s')\nsys.exit(1)\n"
    )

    timeout_dir = tmp_path / "timeout_ws"
    timeout_dir.mkdir()

    tasks = [
        native.VerificationTask(
            task_id="pass",
            workdir=str(pass_dir),
            compile_cmd=[sys.executable, "-m", "py_compile", "mod.py"],
            test_cmd=[sys.executable, "test_mod.py"],
        ),
        native.VerificationTask(
            task_id="partial_fail",
            workdir=str(fail_dir),
            compile_cmd=[sys.executable, "-m", "py_compile", "mod.py"],
            test_cmd=[sys.executable, "test_mod.py"],
        ),
        native.VerificationTask(
            task_id="infinite_sleep",
            workdir=str(timeout_dir),
            test_cmd=[sys.executable, "-c", "import time; time.sleep(5)"],
        ),
    ]

    cfg = native.SandboxConfig(timeout_ms=250, memory_limit_mb=1024)
    results = native.verify_batch(tasks, config=cfg, num_threads=3)
    assert len(results) == 3

    res_pass, res_fail, res_timeout = results
    assert res_pass.passed is True
    assert res_pass.timed_out is False
    assert 0.90 <= res_pass.reward <= 1.0

    assert res_fail.passed is False
    assert res_fail.timed_out is False
    assert 0.40 <= res_fail.reward < 0.90

    assert res_timeout.passed is False
    assert res_timeout.timed_out is True
    assert res_timeout.reward == pytest.approx(-1.0)
    assert res_timeout.duration_ms < 2_000


def test_sandbox_invalid_arguments_raise_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        native.SandboxConfig(timeout_ms=0)
    with pytest.raises(ValueError):
        native.VerificationTask(task_id="", workdir=str(tmp_path), test_cmd=["echo"])
    with pytest.raises(ValueError):
        native.VerificationTask(task_id="x", workdir=str(tmp_path), test_cmd=[])


def test_git_history_extractor_python_bridge(tmp_path: Path) -> None:
    repo = tmp_path / "demo_repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

    git("init")
    git("config", "user.name", "PyO3 Tester")
    git("config", "user.email", "pyo3@hybrid-ttt.dev")

    (repo / "pkg").mkdir()
    (repo / "tests").mkdir()
    (repo / "pkg" / "calc.py").write_text("def sub(a: int, b: int) -> int:\n    return a - b\n")
    (repo / "tests" / "test_calc.py").write_text(
        "from pkg.calc import sub\ndef test_sub():\n    assert sub(5, 2) == 3\n"
    )
    (repo / "yarn.lock").write_text("# lockfile noise\n")
    (repo / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    git("add", "-A")
    git("commit", "-m", "feat: add sub helper with unit test")

    out_jsonl = tmp_path / "out.jsonl"
    cfg = native.ExtractorConfig(max_commits=50, require_tests=True, num_threads=2)
    extractor = native.GitHistoryExtractor(cfg)

    samples = extractor.extract_samples(str(repo))
    assert len(samples) == 1
    assert "pkg/calc.py" in samples[0].code_diff
    assert "tests/test_calc.py" in samples[0].test_diff
    assert "yarn.lock" not in samples[0].code_diff
    assert "image.png" not in samples[0].code_diff

    stats = extractor.extract_to_jsonl(str(repo), str(out_jsonl))
    assert stats.samples_extracted == 1
    row = json.loads(out_jsonl.read_text(encoding="utf-8").strip())
    assert row["commit_message"] == "feat: add sub helper with unit test"


def test_native_bridge_no_memory_leak_under_repeated_calls(tmp_path: Path) -> None:
    verifier = native.SandboxVerifier(native.SandboxConfig(timeout_ms=2_000))
    task = native.VerificationTask(
        task_id="leak_check",
        workdir=str(tmp_path),
        test_cmd=[sys.executable, "-c", "print('ok')"],
    )

    # Warmup
    for _ in range(3):
        verifier.verify(task)

    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()
    for _ in range(15):
        res = verifier.verify(task)
        assert res.passed
    snap_after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    total_growth_bytes = sum(stat.size_diff for stat in snap_after.compare_to(snap_before, "lineno"))
    # Bound Python-side retained object growth well under 1 MiB across 15 FFI roundtrips
    assert total_growth_bytes < 1_048_576
