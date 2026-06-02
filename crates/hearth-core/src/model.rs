use std::sync::{atomic::AtomicBool, mpsc::Sender, Arc};

use crate::output::EngineOutput;
use crate::sampler::SamplerConfig;
pub enum PipelineRequest {
    TextGen {
        prompt: String,
        sampler: SamplerConfig,
        max_new_tokens: usize,
    },
    ImageGen {
        prompt: String,
        negative: String,
        steps: u32,
        cfg_scale: f32,
        width: u32,
        height: u32,
        seed: u64,
    },
    VideoGen {
        prompt: String,
        frames: u32,
        fps: u32,
        width: u32,
        height: u32,
        seed: u64,
    },
    Embed {
        text: String,
    },
}

pub enum LoadStrategy {
    GpuFull,
    CpuOffload,
    LayerSplit(usize),
    CpuOnly,
}

pub trait Model: Send + Sync {
    fn run(&self, request: PipelineRequest, cancel: Arc<AtomicBool>, output: Sender<EngineOutput>);

    /// Build a chat-formatted prompt from system prompt, history pairs, and the
    /// latest user message. Uses the model's tokenizer (including Jinja2 chat
    /// templates from GGUF metadata when available).
    ///
    /// `thinking` controls whether reasoning/thinking sections are shown in the
    /// output (applied via ThinkFilter at the consumer level; some templates
    /// also use this parameter in their chat template).
    fn build_chat_prompt(
        &self,
        system: &str,
        history: &[(&str, &str)],
        user: &str,
        thinking: bool,
    ) -> String;
}

#[derive(Debug)]
pub enum ModelFit {
    GpuFull,
    CpuRam,
    TooBig(String),
}
