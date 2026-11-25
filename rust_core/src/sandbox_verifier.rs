use crate::error::{NativeError, NativeResult};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::io::Read;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

const POLL_INTERVAL_MS: u64 = 3;
const SAFE_ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "RUSTUP_HOME",
    "CARGO_HOME",
    "SYSTEMROOT",
    "COMSPEC",
];

#[pyclass(get_all, set_all)]
#[derive(Clone, Debug)]
pub struct SandboxConfig {
    pub timeout_ms: u64,
    pub memory_limit_mb: u64,
    pub max_output_bytes: usize,
    pub compiler_weight: f64,
    pub lint_weight: f64,
    pub test_weight: f64,
    pub time_bonus_weight: f64,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 5_000,
            memory_limit_mb: 1_024,
            max_output_bytes: 1_048_576,
            compiler_weight: 0.35,
            lint_weight: 0.15,
            test_weight: 0.50,
            time_bonus_weight: 0.05,
        }
    }
}

impl SandboxConfig {
    pub fn validate(&self) -> NativeResult<()> {
        if self.timeout_ms == 0 {
            return Err(NativeError::InvalidArgument(
                "timeout_ms must be greater than 0".to_string(),
            ));
        }
        if self.max_output_bytes == 0 {
            return Err(NativeError::InvalidArgument(
                "max_output_bytes must be greater than 0".to_string(),
            ));
        }
        let weights = [
            self.compiler_weight,
            self.lint_weight,
            self.test_weight,
            self.time_bonus_weight,
        ];
        if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(NativeError::InvalidArgument(
                "all reward weights must be finite and non-negative".to_string(),
            ));
        }
        if self.compiler_weight + self.lint_weight + self.test_weight <= 0.0 {
            return Err(NativeError::InvalidArgument(
                "sum of stage weights must be strictly positive".to_string(),
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl SandboxConfig {
    #[new]
    #[pyo3(signature = (
        timeout_ms = 5_000,
        memory_limit_mb = 1_024,
        max_output_bytes = 1_048_576,
        compiler_weight = 0.35,
        lint_weight = 0.15,
        test_weight = 0.50,
        time_bonus_weight = 0.05
    ))]
    pub fn new(
        timeout_ms: u64,
        memory_limit_mb: u64,
        max_output_bytes: usize,
        compiler_weight: f64,
        lint_weight: f64,
        test_weight: f64,
        time_bonus_weight: f64,
    ) -> PyResult<Self> {
        let cfg = Self {
            timeout_ms,
            memory_limit_mb,
            max_output_bytes,
            compiler_weight,
            lint_weight,
            test_weight,
            time_bonus_weight,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn __repr__(&self) -> String {
        format!(
            "SandboxConfig(timeout_ms={}, memory_limit_mb={}, max_output_bytes={})",
            self.timeout_ms, self.memory_limit_mb, self.max_output_bytes
        )
    }
}

#[pyclass(get_all, set_all)]
#[derive(Clone, Debug)]
pub struct VerificationTask {
    pub task_id: String,
    pub workdir: String,
    pub compile_cmd: Option<Vec<String>>,
    pub lint_cmd: Option<Vec<String>>,
    pub test_cmd: Vec<String>,
    pub env_vars: Vec<(String, String)>,
}

#[pymethods]
impl VerificationTask {
    #[new]
    #[pyo3(signature = (
        task_id,
        workdir,
        test_cmd,
        compile_cmd = None,
        lint_cmd = None,
        env_vars = None
    ))]
    pub fn new(
        task_id: String,
        workdir: String,
        test_cmd: Vec<String>,
        compile_cmd: Option<Vec<String>>,
        lint_cmd: Option<Vec<String>>,
        env_vars: Option<Vec<(String, String)>>,
    ) -> PyResult<Self> {
        if task_id.trim().is_empty() {
            return Err(NativeError::InvalidArgument("task_id must not be empty".to_string()).into());
        }
        if test_cmd.is_empty() {
            return Err(
                NativeError::InvalidArgument("test_cmd must contain an executable".to_string())
                    .into(),
            );
        }
        if let Some(ref cmd) = compile_cmd {
            if cmd.is_empty() {
                return Err(NativeError::InvalidArgument(
                    "compile_cmd cannot be an empty list".to_string(),
                )
                .into());
            }
        }
        if let Some(ref cmd) = lint_cmd {
            if cmd.is_empty() {
                return Err(NativeError::InvalidArgument(
                    "lint_cmd cannot be an empty list".to_string(),
                )
                .into());
            }
        }
        Ok(Self {
            task_id,
            workdir,
            compile_cmd,
            lint_cmd,
            test_cmd,
            env_vars: env_vars.unwrap_or_default(),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "VerificationTask(task_id='{}', workdir='{}')",
            self.task_id, self.workdir
        )
    }
}

#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct CommandStageResult {
    pub stage: String,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub duration_ms: u64,
    pub stdout: String,
    pub stderr: String,
}

#[pymethods]
impl CommandStageResult {
    fn __repr__(&self) -> String {
        format!(
            "CommandStageResult(stage='{}', exit_code={:?}, timed_out={}, duration_ms={})",
            self.stage, self.exit_code, self.timed_out, self.duration_ms
        )
    }
}

#[pyclass(get_all)]
#[derive(Clone, Debug)]
pub struct VerificationResult {
    pub task_id: String,
    pub passed: bool,
    pub timed_out: bool,
    pub exit_code: Option<i32>,
    pub duration_ms: u64,
    pub stdout: String,
    pub stderr: String,
    pub reward: f64,
    pub stages: Vec<CommandStageResult>,
}

#[pymethods]
impl VerificationResult {
    fn __repr__(&self) -> String {
        format!(
            "VerificationResult(task_id='{}', passed={}, timed_out={}, reward={:.4}, duration_ms={})",
            self.task_id, self.passed, self.timed_out, self.reward, self.duration_ms
        )
    }
}

fn read_bounded<R: Read>(mut reader: R, max_bytes: usize) -> Vec<u8> {
    let mut captured = Vec::with_capacity(max_bytes.min(8_192));
    let mut buf = [0_u8; 4_096];
    let mut truncated = false;

    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                if captured.len() < max_bytes {
                    let remaining = max_bytes - captured.len();
                    let to_copy = n.min(remaining);
                    captured.extend_from_slice(&buf[..to_copy]);
                    if to_copy < n {
                        truncated = true;
                    }
                } else {
                    truncated = true;
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }

    if truncated {
        captured.extend_from_slice(b"\n[sandbox: output truncated]");
    }
    captured
}

#[cfg(unix)]
fn terminate_process_tree(child: &mut Child) {
    let pid = child.id();
    if let Ok(pgid) = i32::try_from(pid) {
        if pgid > 0 {
            // SAFETY: `libc::kill` with negative PID targets the isolated process group
            // created via `setpgid(0, 0)` in `pre_exec`. `pgid > 0` guarantees we never
            // signal PID 0 or -1 (broadcast).
            unsafe {
                let _ = libc::kill(-pgid, libc::SIGKILL);
            }
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[cfg(not(unix))]
fn terminate_process_tree(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn run_isolated_command(
    stage: &str,
    cmd_args: &[String],
    workdir: &Path,
    env_vars: &[(String, String)],
    timeout: Duration,
    memory_limit_mb: u64,
    max_output_bytes: usize,
) -> NativeResult<CommandStageResult> {
    let program = cmd_args.first().ok_or_else(|| {
        NativeError::InvalidArgument(format!("stage `{stage}` command cannot be empty"))
    })?;

    if !workdir.exists() {
        return Err(NativeError::PathNotFound(workdir.display().to_string()));
    }

    let mut command = Command::new(program);
    if cmd_args.len() > 1 {
        command.args(&cmd_args[1..]);
    }
    command
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear();

    for key in SAFE_ENV_ALLOWLIST {
        if let Ok(val) = std::env::var(key) {
            command.env(key, val);
        }
    }
    for (k, v) in env_vars {
        command.env(k, v);
    }

    #[cfg(unix)]
    {
        let mem_bytes = memory_limit_mb.saturating_mul(1024 * 1024);
        let cpu_secs = timeout.as_secs().saturating_add(2);
        // SAFETY:
        // 1. Preconditions: `pre_exec` runs strictly in the forked child before `execve`.
        // 2. Async-signal-safety: Only POSIX async-signal-safe syscalls (`setpgid`, `getrlimit`,
        //    `setrlimit`) are invoked. Zero heap allocations, zero mutex acquisitions, and zero
        //    Rust panics occur inside this closure.
        // 3. Pointer validity: `current` and `next` are initialized stack-allocated `libc::rlimit`
        //    structs passed as valid aligned mutable/const pointers exclusively for the syscall duration.
        unsafe {
            command.pre_exec(move || {
                let _ = libc::setpgid(0, 0);

                if mem_bytes > 0 {
                    let mut current = libc::rlimit {
                        rlim_cur: 0,
                        rlim_max: 0,
                    };
                    if libc::getrlimit(libc::RLIMIT_AS, &mut current) == 0 {
                        let requested = mem_bytes as libc::rlim_t;
                        let bounded = if current.rlim_max == libc::RLIM_INFINITY {
                            requested
                        } else {
                            requested.min(current.rlim_max)
                        };
                        let next = libc::rlimit {
                            rlim_cur: bounded,
                            rlim_max: current.rlim_max,
                        };
                        let _ = libc::setrlimit(libc::RLIMIT_AS, &next);
                    }
                }

                let mut cpu_current = libc::rlimit {
                    rlim_cur: 0,
                    rlim_max: 0,
                };
                if libc::getrlimit(libc::RLIMIT_CPU, &mut cpu_current) == 0 {
                    let requested = cpu_secs as libc::rlim_t;
                    let bounded = if cpu_current.rlim_max == libc::RLIM_INFINITY {
                        requested
                    } else {
                        requested.min(cpu_current.rlim_max)
                    };
                    let next = libc::rlimit {
                        rlim_cur: bounded,
                        rlim_max: cpu_current.rlim_max,
                    };
                    let _ = libc::setrlimit(libc::RLIMIT_CPU, &next);
                }

                Ok(())
            });
        }
    }

    #[cfg(not(unix))]
    let _ = memory_limit_mb;

    let start = Instant::now();
    let mut child = command.spawn().map_err(|source| NativeError::ProcessSpawn {
        command: program.clone(),
        source,
    })?;

    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();

    let (exit_code, timed_out, stdout_bytes, stderr_bytes) = std::thread::scope(|scope| {
        let stdout_handle = scope.spawn(move || match stdout_pipe {
            Some(pipe) => read_bounded(pipe, max_output_bytes),
            None => Vec::new(),
        });
        let stderr_handle = scope.spawn(move || match stderr_pipe {
            Some(pipe) => read_bounded(pipe, max_output_bytes),
            None => Vec::new(),
        });

        let mut exit_code: Option<i32> = None;
        let mut timed_out = false;

        loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    exit_code = status.code();
                    break;
                }
                Ok(None) => {
                    if start.elapsed() >= timeout {
                        timed_out = true;
                        terminate_process_tree(&mut child);
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
                }
                Err(_) => {
                    terminate_process_tree(&mut child);
                    break;
                }
            }
        }

        let out = stdout_handle.join().unwrap_or_default();
        let err = stderr_handle.join().unwrap_or_default();
        (exit_code, timed_out, out, err)
    });

    let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    Ok(CommandStageResult {
        stage: stage.to_string(),
        exit_code,
        timed_out,
        duration_ms,
        stdout: String::from_utf8_lossy(&stdout_bytes).into_owned(),
        stderr: String::from_utf8_lossy(&stderr_bytes).into_owned(),
    })
}

fn parse_partial_test_pass_ratio(stdout: &str, stderr: &str) -> Option<f64> {
    let combined = format!("{stdout}\n{stderr}");
    let mut passed: Option<u64> = None;
    let mut failed: Option<u64> = None;

    for line in combined.lines().rev() {
        let trimmed = line.trim();
        // Matches pytest summary lines like "3 passed, 1 failed in 0.02s"
        // or cargo test summary lines like "test result: FAILED. 3 passed; 1 failed;"
        if trimmed.contains("passed") || trimmed.contains("failed") {
            let tokens: Vec<&str> = trimmed
                .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '.')
                .filter(|t| !t.is_empty())
                .collect();
            for pair in tokens.windows(2) {
                if let Ok(num) = pair[0].parse::<u64>() {
                    match pair[1] {
                        "passed" => passed = Some(num),
                        "failed" | "error" | "errors" => {
                            failed = Some(failed.unwrap_or(0).saturating_add(num));
                        }
                        _ => {}
                    }
                }
            }
            if passed.is_some() || failed.is_some() {
                break;
            }
        }
    }

    let p = passed.unwrap_or(0);
    let f = failed.unwrap_or(0);
    let total = p.saturating_add(f);
    if total > 0 {
        Some((p as f64) / (total as f64))
    } else {
        None
    }
}

pub fn compute_objective_reward(
    config: &SandboxConfig,
    compile_res: Option<&CommandStageResult>,
    lint_res: Option<&CommandStageResult>,
    test_res: Option<&CommandStageResult>,
    total_duration_ms: u64,
) -> (bool, bool, f64) {
    let any_timeout = [compile_res, lint_res, test_res]
        .into_iter()
        .flatten()
        .any(|r| r.timed_out);

    if any_timeout {
        return (false, true, -1.0);
    }

    let mut active_weight = 0.0_f64;
    let mut earned = 0.0_f64;

    let compile_ok = match compile_res {
        Some(res) => {
            active_weight += config.compiler_weight;
            let ok = res.exit_code == Some(0);
            if ok {
                earned += config.compiler_weight;
            }
            ok
        }
        None => true,
    };

    let lint_ok = match lint_res {
        Some(res) => {
            active_weight += config.lint_weight;
            let ok = res.exit_code == Some(0);
            if ok {
                earned += config.lint_weight;
            }
            ok
        }
        None => true,
    };

    let test_ok = match test_res {
        Some(res) => {
            active_weight += config.test_weight;
            let ok = res.exit_code == Some(0);
            if ok {
                earned += config.test_weight;
            } else if let Some(ratio) = parse_partial_test_pass_ratio(&res.stdout, &res.stderr) {
                earned += config.test_weight * ratio.clamp(0.0, 0.95);
            }
            ok
        }
        None => false,
    };

    let passed = compile_ok && lint_ok && test_ok;
    let base_score = if active_weight > 0.0 {
        (earned / active_weight).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let speed_bonus = if passed {
        let ratio = 1.0 - (total_duration_ms as f64 / config.timeout_ms.max(1) as f64);
        config.time_bonus_weight * ratio.clamp(0.0, 1.0)
    } else {
        0.0
    };

    // Scale base + speed_bonus cleanly into [0.0, 1.0]
    let max_possible = 1.0 + config.time_bonus_weight;
    let reward = ((base_score + speed_bonus) / max_possible).clamp(0.0, 1.0);
    (passed, false, reward)
}

pub fn execute_verification_task(
    config: &SandboxConfig,
    task: &VerificationTask,
) -> NativeResult<VerificationResult> {
    let workdir = Path::new(&task.workdir);
    let total_budget = Duration::from_millis(config.timeout_ms);
    let overall_start = Instant::now();

    let mut stages = Vec::with_capacity(3);
    let mut compile_idx: Option<usize> = None;
    let mut lint_idx: Option<usize> = None;
    let mut test_idx: Option<usize> = None;

    if let Some(ref compile_cmd) = task.compile_cmd {
        let remaining = total_budget.saturating_sub(overall_start.elapsed());
        let res = run_isolated_command(
            "compile",
            compile_cmd,
            workdir,
            &task.env_vars,
            remaining.max(Duration::from_millis(1)),
            config.memory_limit_mb,
            config.max_output_bytes,
        )?;
        let failed = res.timed_out || res.exit_code != Some(0);
        compile_idx = Some(stages.len());
        stages.push(res);
        if failed {
            return Ok(finalize_result(config, task, stages, compile_idx, None, None, overall_start));
        }
    }

    if let Some(ref lint_cmd) = task.lint_cmd {
        let remaining = total_budget.saturating_sub(overall_start.elapsed());
        if remaining.is_zero() {
            stages.push(CommandStageResult {
                stage: "lint".to_string(),
                exit_code: None,
                timed_out: true,
                duration_ms: 0,
                stdout: String::new(),
                stderr: "sandbox timeout budget exhausted before lint stage".to_string(),
            });
            lint_idx = Some(stages.len() - 1);
            return Ok(finalize_result(
                config,
                task,
                stages,
                compile_idx,
                lint_idx,
                None,
                overall_start,
            ));
        }
        let res = run_isolated_command(
            "lint",
            lint_cmd,
            workdir,
            &task.env_vars,
            remaining,
            config.memory_limit_mb,
            config.max_output_bytes,
        )?;
        let timed_out = res.timed_out;
        lint_idx = Some(stages.len());
        stages.push(res);
        if timed_out {
            return Ok(finalize_result(
                config,
                task,
                stages,
                compile_idx,
                lint_idx,
                None,
                overall_start,
            ));
        }
    }

    let remaining = total_budget.saturating_sub(overall_start.elapsed());
    if remaining.is_zero() {
        stages.push(CommandStageResult {
            stage: "test".to_string(),
            exit_code: None,
            timed_out: true,
            duration_ms: 0,
            stdout: String::new(),
            stderr: "sandbox timeout budget exhausted before test stage".to_string(),
        });
        test_idx = Some(stages.len() - 1);
    } else {
        let res = run_isolated_command(
            "test",
            &task.test_cmd,
            workdir,
            &task.env_vars,
            remaining,
            config.memory_limit_mb,
            config.max_output_bytes,
        )?;
        test_idx = Some(stages.len());
        stages.push(res);
    }

    Ok(finalize_result(
        config,
        task,
        stages,
        compile_idx,
        lint_idx,
        test_idx,
        overall_start,
    ))
}

fn finalize_result(
    config: &SandboxConfig,
    task: &VerificationTask,
    stages: Vec<CommandStageResult>,
    compile_idx: Option<usize>,
    lint_idx: Option<usize>,
    test_idx: Option<usize>,
    overall_start: Instant,
) -> VerificationResult {
    let total_duration_ms = u64::try_from(overall_start.elapsed().as_millis()).unwrap_or(u64::MAX);
    let compile_ref = compile_idx.and_then(|idx| stages.get(idx));
    let lint_ref = lint_idx.and_then(|idx| stages.get(idx));
    let test_ref = test_idx.and_then(|idx| stages.get(idx));

    let (passed, timed_out, reward) =
        compute_objective_reward(config, compile_ref, lint_ref, test_ref, total_duration_ms);

    let last_exit_code = stages.last().and_then(|s| s.exit_code);
    let combined_stdout = stages
        .iter()
        .filter(|s| !s.stdout.is_empty())
        .map(|s| format!("[{}]\n{}", s.stage, s.stdout))
        .collect::<Vec<_>>()
        .join("\n");
    let combined_stderr = stages
        .iter()
        .filter(|s| !s.stderr.is_empty())
        .map(|s| format!("[{}]\n{}", s.stage, s.stderr))
        .collect::<Vec<_>>()
        .join("\n");

    VerificationResult {
        task_id: task.task_id.clone(),
        passed,
        timed_out,
        exit_code: last_exit_code,
        duration_ms: total_duration_ms,
        stdout: combined_stdout,
        stderr: combined_stderr,
        reward,
        stages,
    }
}

pub fn execute_batch_internal(
    config: &SandboxConfig,
    tasks: &[VerificationTask],
    num_threads: Option<usize>,
) -> NativeResult<Vec<VerificationResult>> {
    config.validate()?;
    if tasks.is_empty() {
        return Ok(Vec::new());
    }

    let run_par = || {
        tasks
            .par_iter()
            .map(|task| execute_verification_task(config, task))
            .collect::<NativeResult<Vec<VerificationResult>>>()
    };

    match num_threads {
        Some(n) if n > 0 => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| NativeError::ThreadPool(e.to_string()))?;
            pool.install(run_par)
        }
        Some(_) => Err(NativeError::InvalidArgument(
            "num_threads must be greater than 0 when specified".to_string(),
        )),
        None => run_par(),
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct SandboxVerifier {
    config: SandboxConfig,
    num_threads: Option<usize>,
}

#[pymethods]
impl SandboxVerifier {
    #[new]
    #[pyo3(signature = (config = None, num_threads = None))]
    pub fn new(config: Option<SandboxConfig>, num_threads: Option<usize>) -> PyResult<Self> {
        let cfg = config.unwrap_or_default();
        cfg.validate()?;
        if let Some(0) = num_threads {
            return Err(NativeError::InvalidArgument(
                "num_threads must be greater than 0 when specified".to_string(),
            )
            .into());
        }
        Ok(Self {
            config: cfg,
            num_threads,
        })
    }

    #[getter]
    pub fn config(&self) -> SandboxConfig {
        self.config.clone()
    }

    pub fn verify(
        &self,
        py: Python<'_>,
        task: VerificationTask,
    ) -> PyResult<VerificationResult> {
        let cfg = self.config.clone();
        let result = py.allow_threads(move || execute_verification_task(&cfg, &task))?;
        Ok(result)
    }

    pub fn verify_batch(
        &self,
        py: Python<'_>,
        tasks: Vec<VerificationTask>,
    ) -> PyResult<Vec<VerificationResult>> {
        let cfg = self.config.clone();
        let threads = self.num_threads;
        let results =
            py.allow_threads(move || execute_batch_internal(&cfg, &tasks, threads))?;
        Ok(results)
    }
}

#[pyfunction]
#[pyo3(signature = (tasks, config = None, num_threads = None))]
pub fn verify_batch(
    py: Python<'_>,
    tasks: Vec<VerificationTask>,
    config: Option<SandboxConfig>,
    num_threads: Option<usize>,
) -> PyResult<Vec<VerificationResult>> {
    let cfg = config.unwrap_or_default();
    cfg.validate()?;
    let results = py.allow_threads(move || execute_batch_internal(&cfg, &tasks, num_threads))?;
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_sandbox_passing_and_reward_scoring() {
        let dir = tempdir().expect("create tempdir");
        let cfg = SandboxConfig {
            timeout_ms: 3_000,
            memory_limit_mb: 512,
            ..SandboxConfig::default()
        };

        let task = VerificationTask {
            task_id: "pass_case".to_string(),
            workdir: dir.path().display().to_string(),
            compile_cmd: Some(vec![
                "python3".to_string(),
                "-c".to_string(),
                "print('compile ok')".to_string(),
            ]),
            lint_cmd: Some(vec![
                "python3".to_string(),
                "-c".to_string(),
                "print('lint ok')".to_string(),
            ]),
            test_cmd: vec![
                "python3".to_string(),
                "-c".to_string(),
                "print('3 passed in 0.01s')".to_string(),
            ],
            env_vars: vec![],
        };

        let res = execute_verification_task(&cfg, &task).expect("execute passing task");
        assert!(res.passed);
        assert!(!res.timed_out);
        assert_eq!(res.exit_code, Some(0));
        assert!(res.reward > 0.90 && res.reward <= 1.0);
        assert_eq!(res.stages.len(), 3);
        assert!(res.stdout.contains("compile ok"));
        assert!(res.stdout.contains("3 passed"));
    }

    #[test]
    fn test_sandbox_hard_timeout_enforcement() {
        let dir = tempdir().expect("create tempdir");
        let cfg = SandboxConfig {
            timeout_ms: 150,
            memory_limit_mb: 512,
            ..SandboxConfig::default()
        };

        let task = VerificationTask {
            task_id: "timeout_case".to_string(),
            workdir: dir.path().display().to_string(),
            compile_cmd: None,
            lint_cmd: None,
            test_cmd: vec![
                "python3".to_string(),
                "-c".to_string(),
                "import time; time.sleep(5)".to_string(),
            ],
            env_vars: vec![],
        };

        let res = execute_verification_task(&cfg, &task).expect("execute timeout task");
        assert!(!res.passed);
        assert!(res.timed_out);
        assert!((res.reward - (-1.0)).abs() < f64::EPSILON);
        assert!(res.duration_ms >= 140);
        assert!(res.duration_ms < 2_000);
    }

    #[test]
    fn test_sandbox_partial_reward_and_batch_execution() {
        let dir = tempdir().expect("create tempdir");
        let cfg = SandboxConfig::default();

        let partial_task = VerificationTask {
            task_id: "partial".to_string(),
            workdir: dir.path().display().to_string(),
            compile_cmd: Some(vec![
                "python3".to_string(),
                "-c".to_string(),
                "pass".to_string(),
            ]),
            lint_cmd: None,
            test_cmd: vec![
                "python3".to_string(),
                "-c".to_string(),
                "import sys; print('2 passed, 2 failed in 0.02s'); sys.exit(1)".to_string(),
            ],
            env_vars: vec![],
        };

        let fail_compile = VerificationTask {
            task_id: "fail_compile".to_string(),
            workdir: dir.path().display().to_string(),
            compile_cmd: Some(vec![
                "python3".to_string(),
                "-c".to_string(),
                "import sys; sys.exit(2)".to_string(),
            ]),
            lint_cmd: None,
            test_cmd: vec!["python3".to_string(), "-c".to_string(), "pass".to_string()],
            env_vars: vec![],
        };

        let results =
            execute_batch_internal(&cfg, &[partial_task, fail_compile], Some(2)).expect("batch");
        assert_eq!(results.len(), 2);
        assert!(!results[0].passed);
        assert!(results[0].reward > 0.5 && results[0].reward < 0.9);
        assert!(!results[1].passed);
        assert!((results[1].reward - 0.0).abs() < f64::EPSILON);
        assert_eq!(results[1].stages.len(), 1);
    }
}
