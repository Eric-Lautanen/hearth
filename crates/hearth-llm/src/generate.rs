use crate::config::ModelConfig;
use hearth_core::SamplerConfig;

pub struct GenerateRequest {
    pub prompt: String,
    pub max_new_tokens: usize,
    pub sampler: SamplerConfig,
    pub config: ModelConfig,
}

pub struct GenerateResponse {
    pub text: String,
    pub tokens_generated: usize,
    pub tokens_per_second: f32,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub id: u32,
    pub text: String,
    pub logprob: Option<f32>,
}
