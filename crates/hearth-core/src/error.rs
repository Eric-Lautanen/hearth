use std::fmt;

#[derive(Debug, Clone)]
pub enum ModelError {
    FileNotFound(String),
    ParseError(String),
    UnsupportedArchitecture(String),
    UnsupportedDtype(String),
    LoadError(String),
    RuntimeError(String),
    OutOfMemory(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::FileNotFound(p) => write!(f, "File not found: {}", p),
            ModelError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            ModelError::UnsupportedArchitecture(a) => {
                write!(f, "Unsupported architecture: {}", a)
            }
            ModelError::UnsupportedDtype(d) => write!(f, "Unsupported dtype: {}", d),
            ModelError::LoadError(msg) => write!(f, "Load error: {}", msg),
            ModelError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            ModelError::OutOfMemory(msg) => write!(f, "Out of memory: {}", msg),
        }
    }
}

impl std::error::Error for ModelError {}
