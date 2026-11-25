from __future__ import annotations

class SandboxConfig:
    timeout_ms: int
    memory_limit_mb: int
    max_output_bytes: int
    compiler_weight: float
    lint_weight: float
    test_weight: float
    time_bonus_weight: float

    def __init__(
        self,
        timeout_ms: int = 5000,
        memory_limit_mb: int = 1024,
        max_output_bytes: int = 1048576,
        compiler_weight: float = 0.35,
        lint_weight: float = 0.15,
        test_weight: float = 0.50,
        time_bonus_weight: float = 0.05,
    ) -> None: ...

class VerificationTask:
    task_id: str
    workdir: str
    compile_cmd: list[str] | None
    lint_cmd: list[str] | None
    test_cmd: list[str]
    env_vars: list[tuple[str, str]]

    def __init__(
        self,
        task_id: str,
        workdir: str,
        test_cmd: list[str],
        compile_cmd: list[str] | None = None,
        lint_cmd: list[str] | None = None,
        env_vars: list[tuple[str, str]] | None = None,
    ) -> None: ...

class CommandStageResult:
    @property
    def stage(self) -> str: ...
    @property
    def exit_code(self) -> int | None: ...
    @property
    def timed_out(self) -> bool: ...
    @property
    def duration_ms(self) -> int: ...
    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...

class VerificationResult:
    @property
    def task_id(self) -> str: ...
    @property
    def passed(self) -> bool: ...
    @property
    def timed_out(self) -> bool: ...
    @property
    def exit_code(self) -> int | None: ...
    @property
    def duration_ms(self) -> int: ...
    @property
    def stdout(self) -> str: ...
    @property
    def stderr(self) -> str: ...
    @property
    def reward(self) -> float: ...
    @property
    def stages(self) -> list[CommandStageResult]: ...

class SandboxVerifier:
    def __init__(
        self,
        config: SandboxConfig | None = None,
        num_threads: int | None = None,
    ) -> None: ...
    @property
    def config(self) -> SandboxConfig: ...
    def verify(self, task: VerificationTask) -> VerificationResult: ...
    def verify_batch(self, tasks: list[VerificationTask]) -> list[VerificationResult]: ...

def verify_batch(
    tasks: list[VerificationTask],
    config: SandboxConfig | None = None,
    num_threads: int | None = None,
) -> list[VerificationResult]: ...

class ExtractorConfig:
    max_commits: int
    max_diff_bytes: int
    min_diff_lines: int
    require_tests: bool
    num_threads: int | None

    def __init__(
        self,
        max_commits: int = 10000,
        max_diff_bytes: int = 65536,
        min_diff_lines: int = 1,
        require_tests: bool = False,
        num_threads: int | None = None,
    ) -> None: ...

class CommitSample:
    @property
    def commit_hash(self) -> str: ...
    @property
    def author_timestamp(self) -> int: ...
    @property
    def commit_message(self) -> str: ...
    @property
    def context_files(self) -> list[str]: ...
    @property
    def code_diff(self) -> str: ...
    @property
    def test_diff(self) -> str: ...
    @property
    def files_changed(self) -> int: ...
    @property
    def lines_added(self) -> int: ...
    @property
    def lines_deleted(self) -> int: ...

class ExtractionStats:
    @property
    def total_commits_scanned(self) -> int: ...
    @property
    def samples_extracted(self) -> int: ...
    @property
    def skipped_merge_commits(self) -> int: ...
    @property
    def skipped_filtered_commits(self) -> int: ...
    @property
    def duration_ms(self) -> int: ...
    @property
    def output_path(self) -> str: ...

class GitHistoryExtractor:
    def __init__(self, config: ExtractorConfig | None = None) -> None: ...
    @property
    def config(self) -> ExtractorConfig: ...
    def extract_to_jsonl(self, repo_path: str, output_jsonl: str) -> ExtractionStats: ...
    def extract_samples(self, repo_path: str) -> list[CommitSample]: ...

def extract_git_history(
    repo_path: str,
    output_jsonl: str,
    config: ExtractorConfig | None = None,
) -> ExtractionStats: ...
