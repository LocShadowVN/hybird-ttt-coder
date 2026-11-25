use pyo3::exceptions::{PyFileNotFoundError, PyIOError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum NativeError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("path not found: {0}")]
    PathNotFound(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("git error: {0}")]
    Git(#[from] git2::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("thread pool initialization error: {0}")]
    ThreadPool(String),

    #[error("failed to spawn process `{command}`: {source}")]
    ProcessSpawn {
        command: String,
        #[source]
        source: std::io::Error,
    },
}

impl From<NativeError> for PyErr {
    fn from(err: NativeError) -> Self {
        match err {
            NativeError::InvalidArgument(msg) => PyValueError::new_err(msg),
            NativeError::PathNotFound(msg) => PyFileNotFoundError::new_err(msg),
            NativeError::Io(io_err) => PyIOError::new_err(io_err.to_string()),
            NativeError::Git(git_err) => PyRuntimeError::new_err(git_err.to_string()),
            NativeError::Json(json_err) => PyRuntimeError::new_err(json_err.to_string()),
            NativeError::ThreadPool(msg) => PyRuntimeError::new_err(msg),
            NativeError::ProcessSpawn { command, source } => {
                PyRuntimeError::new_err(format!("failed to spawn `{command}`: {source}"))
            }
        }
    }
}

pub type NativeResult<T> = Result<T, NativeError>;
