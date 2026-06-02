use crate::error::ModelError;
use crate::stats::RunStats;

pub enum EngineOutput {
    TextToken(String),
    ImageStep {
        step: u32,
        total: u32,
        preview: Option<Vec<u8>>,
    },
    ImageDone {
        rgba: Vec<u8>,
        width: u32,
        height: u32,
        stats: RunStats,
    },
    VideoFrame {
        index: u32,
        total: u32,
        rgba: Vec<u8>,
        width: u32,
        height: u32,
    },
    Done(RunStats),
    Error(ModelError),
}
