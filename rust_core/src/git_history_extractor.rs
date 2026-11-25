use crate::error::{NativeError, NativeResult};
use git2::{DiffFormat, DiffOptions, Oid, Repository, Sort};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const MAX_LINE_LEN_NON_MINIFIED: usize = 2_000;
const WRITER_BUF_CAPACITY: usize = 256 * 1024;

const IGNORED_EXACT_FILENAMES: &[&str] = &[
    "cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "uv.lock",
    "gemfile.lock",
    "composer.lock",
    "go.sum",
    "flake.lock",
    ".ds_store",
];

const IGNORED_PATH_SEGMENTS: &[&str] = &[
    "node_modules/",
    "dist/",
    "build/",
    "target/",
    "vendor/",
    ".venv/",
    "__pycache__/",
    ".git/",
    "third_party/",
];

const IGNORED_SUFFIXES: &[&str] = &[
    ".min.js",
    ".min.css",
    ".bundle.js",
    ".chunk.js",
    ".map",
    ".pb.go",
    "_pb2.py",
    ".g.dart",
    ".generated.ts",
    ".snap",
    ".lock",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".pdf",
    ".wasm",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
    ".o",
    ".a",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".onnx",
    ".pyc",
    ".pyo",
    ".class",
    ".jar",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp4",
    ".mp3",
    ".wav",
    ".sqlite",
    ".db",
];

const SOURCE_EXTENSIONS: &[&str] = &[
    ".rs", ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp",
    ".go", ".java", ".kt", ".kts", ".swift", ".zig", ".cs", ".rb", ".php", ".scala", ".lua",
    ".sql", ".sh", ".bash", ".zsh", ".toml", ".yaml", ".yml", ".json", ".dockerfile",
];

#[pyclass(get_all, set_all)]
#[derive(Clone, Debug)]
pub struct ExtractorConfig {
    pub max_commits: usize,
    pub max_diff_bytes: usize,
    pub min_diff_lines: usize,
    pub require_tests: bool,
    pub num_threads: Option<usize>,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            max_commits: 10_000,
            max_diff_bytes: 65_536,
            min_diff_lines: 1,
            require_tests: false,
            num_threads: None,
        }
    }
}

impl ExtractorConfig {
    pub fn validate(&self) -> NativeResult<()> {
        if self.max_commits == 0 {
            return Err(NativeError::InvalidArgument(
                "max_commits must be greater than 0".to_string(),
            ));
        }
        if self.max_diff_bytes == 0 {
            return Err(NativeError::InvalidArgument(
                "max_diff_bytes must be greater than 0".to_string(),
            ));
        }
        if let Some(0) = self.num_threads {
            return Err(NativeError::InvalidArgument(
                "num_threads must be greater than 0 when specified".to_string(),
            ));
        }
        Ok(())
    }
}

#[pymethods]
impl ExtractorConfig {
    #[new]
    #[pyo3(signature = (
        max_commits = 10_000,
        max_diff_bytes = 65_536,
        min_diff_lines = 1,
        require_tests = false,
        num_threads = None
    ))]
    pub fn new(
        max_commits: usize,
        max_diff_bytes: usize,
        min_diff_lines: usize,
        require_tests: bool,
        num_threads: Option<usize>,
    ) -> PyResult<Self> {
        let cfg = Self {
            max_commits,
            max_diff_bytes,
            min_diff_lines,
            require_tests,
            num_threads,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtractorConfig(max_commits={}, max_diff_bytes={}, require_tests={})",
            self.max_commits, self.max_diff_bytes, self.require_tests
        )
    }
}

#[pyclass(get_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitSample {
    pub commit_hash: String,
    pub author_timestamp: i64,
    pub commit_message: String,
    pub context_files: Vec<String>,
    pub code_diff: String,
    pub test_diff: String,
    pub files_changed: usize,
    pub lines_added: usize,
    pub lines_deleted: usize,
}

#[pymethods]
impl CommitSample {
    fn __repr__(&self) -> String {
        format!(
            "CommitSample(commit_hash='{}', files_changed={}, +{}, -{})",
            &self.commit_hash[..self.commit_hash.len().min(8)],
            self.files_changed,
            self.lines_added,
            self.lines_deleted
        )
    }
}

#[pyclass(get_all)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractionStats {
    pub total_commits_scanned: usize,
    pub samples_extracted: usize,
    pub skipped_merge_commits: usize,
    pub skipped_filtered_commits: usize,
    pub duration_ms: u64,
    pub output_path: String,
}

#[pymethods]
impl ExtractionStats {
    fn __repr__(&self) -> String {
        format!(
            "ExtractionStats(scanned={}, extracted={}, duration_ms={})",
            self.total_commits_scanned, self.samples_extracted, self.duration_ms
        )
    }
}

pub fn should_ignore_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_ascii_lowercase();
    let filename = normalized.rsplit('/').next().unwrap_or(normalized.as_str());

    if IGNORED_EXACT_FILENAMES.contains(&filename) {
        return true;
    }

    for segment in IGNORED_PATH_SEGMENTS {
        if normalized.starts_with(segment) || normalized.contains(&format!("/{segment}")) {
            return true;
        }
    }

    for suffix in IGNORED_SUFFIXES {
        if normalized.ends_with(suffix) {
            return true;
        }
    }

    let is_known_source = SOURCE_EXTENSIONS.iter().any(|ext| normalized.ends_with(ext))
        || filename == "makefile"
        || filename == "dockerfile";

    !is_known_source
}

pub fn is_test_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/").to_ascii_lowercase();
    let filename = normalized.rsplit('/').next().unwrap_or(normalized.as_str());

    normalized.starts_with("tests/")
        || normalized.starts_with("test/")
        || normalized.contains("/tests/")
        || normalized.contains("/test/")
        || normalized.contains("/__tests__/")
        || normalized.contains("/spec/")
        || filename.starts_with("test_")
        || filename.ends_with("_test.rs")
        || filename.ends_with("_test.go")
        || filename.ends_with("_test.py")
        || filename.ends_with(".test.ts")
        || filename.ends_with(".test.tsx")
        || filename.ends_with(".test.js")
        || filename.ends_with(".spec.ts")
        || filename.ends_with(".spec.js")
        || filename.ends_with("_spec.rb")
        || filename.ends_with("test.java")
        || filename.ends_with("tests.java")
}

enum CommitOutcome {
    Extracted(CommitSample),
    SkippedMerge,
    SkippedFiltered,
}

fn extract_single_commit(
    repo: &Repository,
    oid: Oid,
    config: &ExtractorConfig,
) -> NativeResult<CommitOutcome> {
    let commit = repo.find_commit(oid)?;
    if commit.parent_count() > 1 {
        return Ok(CommitOutcome::SkippedMerge);
    }

    let commit_tree = commit.tree()?;
    let parent_tree = if commit.parent_count() == 1 {
        Some(commit.parent(0)?.tree()?)
    } else {
        None
    };

    let mut diff_opts = DiffOptions::new();
    diff_opts
        .context_lines(3)
        .ignore_submodules(true)
        .include_typechange(false);

    let diff = repo.diff_tree_to_tree(
        parent_tree.as_ref(),
        Some(&commit_tree),
        Some(&mut diff_opts),
    )?;

    let mut code_diff = String::new();
    let mut test_diff = String::new();
    let mut context_files = BTreeSet::new();
    let mut lines_added: usize = 0;
    let mut lines_deleted: usize = 0;
    let mut exceeded_budget = false;
    let mut saw_minified_line = false;

    diff.print(DiffFormat::Patch, |delta, _hunk, line| {
        if exceeded_budget || saw_minified_line {
            return false;
        }
        if delta.flags().is_binary() {
            return true;
        }

        let path_opt = delta.new_file().path().or_else(|| delta.old_file().path());
        let Some(path) = path_opt else {
            return true;
        };
        let path_str = path.to_string_lossy();
        if should_ignore_path(&path_str) {
            return true;
        }

        let Ok(content) = std::str::from_utf8(line.content()) else {
            return true;
        };
        if content.len() > MAX_LINE_LEN_NON_MINIFIED {
            saw_minified_line = true;
            return false;
        }

        let is_test = is_test_path(&path_str);
        context_files.insert(path_str.into_owned());

        let target = if is_test {
            &mut test_diff
        } else {
            &mut code_diff
        };

        let origin = line.origin();
        match origin {
            '+' => {
                lines_added = lines_added.saturating_add(1);
                target.push('+');
                target.push_str(content);
            }
            '-' => {
                lines_deleted = lines_deleted.saturating_add(1);
                target.push('-');
                target.push_str(content);
            }
            ' ' => {
                target.push(' ');
                target.push_str(content);
            }
            'F' | 'H' => {
                target.push_str(content);
            }
            _ => {}
        }

        if code_diff.len().saturating_add(test_diff.len()) > config.max_diff_bytes {
            exceeded_budget = true;
            return false;
        }
        true
    })?;

    if exceeded_budget || saw_minified_line {
        return Ok(CommitOutcome::SkippedFiltered);
    }

    let total_changed_lines = lines_added.saturating_add(lines_deleted);
    if total_changed_lines < config.min_diff_lines {
        return Ok(CommitOutcome::SkippedFiltered);
    }

    if code_diff.trim().is_empty() && test_diff.trim().is_empty() {
        return Ok(CommitOutcome::SkippedFiltered);
    }

    if config.require_tests && (code_diff.trim().is_empty() || test_diff.trim().is_empty()) {
        return Ok(CommitOutcome::SkippedFiltered);
    }

    let commit_message = commit.message().unwrap_or_default().trim().to_string();
    let files_changed = context_files.len();

    Ok(CommitOutcome::Extracted(CommitSample {
        commit_hash: oid.to_string(),
        author_timestamp: commit.time().seconds(),
        commit_message,
        context_files: context_files.into_iter().collect(),
        code_diff,
        test_diff,
        files_changed,
        lines_added,
        lines_deleted,
    }))
}

pub fn collect_samples_internal(
    repo_path: &Path,
    config: &ExtractorConfig,
) -> NativeResult<(Vec<CommitSample>, usize, usize, usize)> {
    config.validate()?;
    if !repo_path.exists() {
        return Err(NativeError::PathNotFound(repo_path.display().to_string()));
    }

    let repo = Repository::open(repo_path)?;
    let mut revwalk = repo.revwalk()?;
    revwalk.set_sorting(Sort::TOPOLOGICAL | Sort::TIME)?;
    revwalk.push_head()?;

    let mut oids = Vec::with_capacity(config.max_commits.min(4_096));
    for oid_res in revwalk.take(config.max_commits) {
        oids.push(oid_res?);
    }
    let total_scanned = oids.len();

    let skipped_merge = AtomicUsize::new(0);
    let skipped_filtered = AtomicUsize::new(0);
    let repo_path_owned = repo_path.to_path_buf();

    let run_par = || -> NativeResult<Vec<CommitSample>> {
        let results: Vec<NativeResult<Option<CommitSample>>> = oids
            .par_iter()
            .map_init(
                || Repository::open(&repo_path_owned),
                |worker_repo, &oid| {
                    let repo_ref = match worker_repo {
                        Ok(r) => r,
                        Err(err) => {
                            return Err(NativeError::Git(git2::Error::from_str(err.message())));
                        }
                    };
                    match extract_single_commit(repo_ref, oid, config)? {
                        CommitOutcome::Extracted(sample) => Ok(Some(sample)),
                        CommitOutcome::SkippedMerge => {
                            skipped_merge.fetch_add(1, Ordering::Relaxed);
                            Ok(None)
                        }
                        CommitOutcome::SkippedFiltered => {
                            skipped_filtered.fetch_add(1, Ordering::Relaxed);
                            Ok(None)
                        }
                    }
                },
            )
            .collect();

        let mut samples = Vec::with_capacity(results.len());
        for item in results {
            if let Some(sample) = item? {
                samples.push(sample);
            }
        }
        Ok(samples)
    };

    let samples = match config.num_threads {
        Some(n) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| NativeError::ThreadPool(e.to_string()))?;
            pool.install(run_par)?
        }
        None => run_par()?,
    };

    Ok((
        samples,
        total_scanned,
        skipped_merge.load(Ordering::Relaxed),
        skipped_filtered.load(Ordering::Relaxed),
    ))
}

pub fn extract_to_jsonl_internal(
    repo_path: &Path,
    output_jsonl: &Path,
    config: &ExtractorConfig,
) -> NativeResult<ExtractionStats> {
    let start = Instant::now();
    let (samples, total_scanned, skipped_merge, skipped_filtered) =
        collect_samples_internal(repo_path, config)?;

    if let Some(parent) = output_jsonl.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(output_jsonl)?;
    let mut writer = BufWriter::with_capacity(WRITER_BUF_CAPACITY, file);
    for sample in &samples {
        serde_json::to_writer(&mut writer, sample)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;

    let duration_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
    Ok(ExtractionStats {
        total_commits_scanned: total_scanned,
        samples_extracted: samples.len(),
        skipped_merge_commits: skipped_merge,
        skipped_filtered_commits: skipped_filtered,
        duration_ms,
        output_path: output_jsonl.display().to_string(),
    })
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct GitHistoryExtractor {
    config: ExtractorConfig,
}

#[pymethods]
impl GitHistoryExtractor {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<ExtractorConfig>) -> PyResult<Self> {
        let cfg = config.unwrap_or_default();
        cfg.validate()?;
        Ok(Self { config: cfg })
    }

    #[getter]
    pub fn config(&self) -> ExtractorConfig {
        self.config.clone()
    }

    pub fn extract_to_jsonl(
        &self,
        py: Python<'_>,
        repo_path: String,
        output_jsonl: String,
    ) -> PyResult<ExtractionStats> {
        let cfg = self.config.clone();
        let stats = py.allow_threads(move || {
            extract_to_jsonl_internal(Path::new(&repo_path), Path::new(&output_jsonl), &cfg)
        })?;
        Ok(stats)
    }

    pub fn extract_samples(
        &self,
        py: Python<'_>,
        repo_path: String,
    ) -> PyResult<Vec<CommitSample>> {
        let cfg = self.config.clone();
        let (samples, _, _, _) =
            py.allow_threads(move || collect_samples_internal(Path::new(&repo_path), &cfg))?;
        Ok(samples)
    }
}

#[pyfunction]
#[pyo3(signature = (repo_path, output_jsonl, config = None))]
pub fn extract_git_history(
    py: Python<'_>,
    repo_path: String,
    output_jsonl: String,
    config: Option<ExtractorConfig>,
) -> PyResult<ExtractionStats> {
    let cfg = config.unwrap_or_default();
    cfg.validate()?;
    let stats = py.allow_threads(move || {
        extract_to_jsonl_internal(Path::new(&repo_path), Path::new(&output_jsonl), &cfg)
    })?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use git2::Signature;
    use tempfile::tempdir;

    fn add_and_commit(
        repo: &Repository,
        files: &[(&str, &[u8])],
        message: &str,
        parent: Option<Oid>,
    ) -> Oid {
        let workdir = repo.workdir().expect("repo workdir");
        let mut index = repo.index().expect("repo index");
        for (rel_path, content) in files {
            let full_path = workdir.join(rel_path);
            if let Some(dir) = full_path.parent() {
                fs::create_dir_all(dir).expect("create parent dirs");
            }
            fs::write(&full_path, content).expect("write file");
            index.add_path(Path::new(rel_path)).expect("stage file");
        }
        index.write().expect("write index");
        let tree_oid = index.write_tree().expect("write tree");
        let tree = repo.find_tree(tree_oid).expect("find tree");
        let sig = Signature::now("Hybrid Tester", "tester@hybrid-ttt.dev").expect("signature");

        match parent {
            Some(pid) => {
                let parent_commit = repo.find_commit(pid).expect("find parent");
                repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &[&parent_commit])
                    .expect("commit with parent")
            }
            None => repo
                .commit(Some("HEAD"), &sig, &sig, message, &tree, &[])
                .expect("root commit"),
        }
    }

    #[test]
    fn test_git_extractor_filters_binaries_lockfiles_and_splits_tests() {
        let repo_dir = tempdir().expect("create temp repo");
        let out_dir = tempdir().expect("create temp out");
        let repo = Repository::init(repo_dir.path()).expect("init git repo");

        let c1 = add_and_commit(
            &repo,
            &[
                ("src/math.py", b"def add(a: int, b: int) -> int:\n    return a + b\n"),
                ("tests/test_math.py", b"from src.math import add\n\ndef test_add():\n    assert add(1, 2) == 3\n"),
                ("Cargo.lock", b"# auto-generated lockfile\nversion = 3\n"),
                ("assets/logo.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"),
                ("web/app.min.js", b"var a=1;console.log(a);"),
            ],
            "feat: implement add with unit test",
            None,
        );

        let _c2 = add_and_commit(
            &repo,
            &[
                (
                    "src/math.py",
                    b"def add(a: int, b: int) -> int:\n    return a + b\n\ndef mul(a: int, b: int) -> int:\n    return a * b\n",
                ),
                (
                    "tests/test_math.py",
                    b"from src.math import add, mul\n\ndef test_add():\n    assert add(1, 2) == 3\n\ndef test_mul():\n    assert mul(3, 4) == 12\n",
                ),
                ("package-lock.json", b"{\"lockfileVersion\": 3}"),
            ],
            "feat: add mul function and test coverage",
            Some(c1),
        );

        let out_jsonl = out_dir.path().join("commits.jsonl");
        let cfg = ExtractorConfig {
            max_commits: 100,
            require_tests: true,
            num_threads: Some(2),
            ..ExtractorConfig::default()
        };

        let stats =
            extract_to_jsonl_internal(repo_dir.path(), &out_jsonl, &cfg).expect("extract jsonl");
        assert_eq!(stats.total_commits_scanned, 2);
        assert_eq!(stats.samples_extracted, 2);

        let content = fs::read_to_string(&out_jsonl).expect("read jsonl");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        for line in lines {
            let sample: CommitSample = serde_json::from_str(line).expect("parse sample");
            assert!(sample.code_diff.contains("src/math.py"));
            assert!(sample.test_diff.contains("tests/test_math.py"));
            assert!(!sample.code_diff.contains("Cargo.lock"));
            assert!(!sample.code_diff.contains("package-lock.json"));
            assert!(!sample.code_diff.contains("logo.png"));
            assert!(!sample.code_diff.contains("app.min.js"));
        }
    }
}
