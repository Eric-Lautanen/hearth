#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_pen: f32,
    pub min_p: f32,
    pub typical_p: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_pen: 1.1,
            min_p: 0.0,
            typical_p: 0.0,
            seed: 0,
        }
    }
}

pub type CancelFlag = std::sync::Arc<std::sync::atomic::AtomicBool>;
