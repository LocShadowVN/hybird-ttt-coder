pub mod error;
pub mod git_history_extractor;
pub mod sandbox_verifier;

use crate::git_history_extractor::{
    extract_git_history, CommitSample, ExtractionStats, ExtractorConfig, GitHistoryExtractor,
};
use crate::sandbox_verifier::{
    verify_batch, CommandStageResult, SandboxConfig, SandboxVerifier, VerificationResult,
    VerificationTask,
};
use pyo3::prelude::*;

#[pymodule]
fn hybrid_ttt_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SandboxConfig>()?;
    m.add_class::<VerificationTask>()?;
    m.add_class::<CommandStageResult>()?;
    m.add_class::<VerificationResult>()?;
    m.add_class::<SandboxVerifier>()?;
    m.add_class::<ExtractorConfig>()?;
    m.add_class::<CommitSample>()?;
    m.add_class::<ExtractionStats>()?;
    m.add_class::<GitHistoryExtractor>()?;
    m.add_function(wrap_pyfunction!(verify_batch, m)?)?;
    m.add_function(wrap_pyfunction!(extract_git_history, m)?)?;
    Ok(())
}
