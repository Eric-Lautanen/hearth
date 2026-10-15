use hearth_gguf::GgufFile;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub d_model: u32,
    pub d_ffn: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub rope_dim: u32,
    pub rope_scaling_type: Option<String>,
    pub rope_scaling_factor: Option<f32>,
    pub original_ctx_len: Option<u32>,
    pub n_experts: u32,
    pub n_experts_per_tok: u32,
    pub post_norm: bool,
    pub embed_scale: bool,
    /// True when the GGUF stated attention.head_dim explicitly (vs. derived).
    /// When set, load() must not re-derive head_dim from tensor shapes —
    /// architectures like qwen35 (attn_q = query+gate, 2x width) would mis-derive.
    pub explicit_head_dim: bool,
    /// Hybrid linear-attention arch guard (gated delta net layers present).
    pub recurrent_layers: Option<Vec<bool>>,
    /// Per-model sampling recommendations from `general.sampling.*` metadata.
    /// When present these are the model author's intended defaults and are
    /// preferred over Hearth's generic fallbacks.
    pub sampling_temperature: Option<f32>,
    pub sampling_top_p: Option<f32>,
    pub sampling_top_k: Option<u32>,
    pub sampling_min_p: Option<f32>,
}

impl ModelConfig {
    /// Sampling defaults the UI/service should start from when the user has not
    /// overridden them. Falls back to Hearth's generic values when the GGUF
    /// carries no `general.sampling.*` metadata.
    pub fn sampler_defaults(&self) -> hearth_core::SamplerConfig {
        hearth_core::SamplerConfig {
            temperature: self.sampling_temperature.unwrap_or(0.7),
            top_k: self.sampling_top_k.unwrap_or(40) as usize,
            top_p: self.sampling_top_p.unwrap_or(0.9),
            repeat_pen: 1.1,
            min_p: self.sampling_min_p.unwrap_or(0.0),
            typical_p: 0.0,
            seed: 0,
        }
    }
}

impl ModelConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        let arch = gguf
            .meta_str("general.architecture")
            .ok_or_else(|| "Missing general.architecture".to_string())?
            .to_string();

        let prefixes = get_arch_prefixes(&arch);

        let meta_u32_prefixed = |key: &str| -> Option<u32> {
            prefixes
                .iter()
                .filter_map(|p| {
                    if p.is_empty() {
                        None
                    } else {
                        gguf.meta_u32(&format!("{}.{}", p, key))
                    }
                })
                .next()
                .or_else(|| gguf.meta_u32(&format!("llama.{}", key)))
        };

        let meta_f32_prefixed = |key: &str| -> Option<f32> {
            prefixes
                .iter()
                .filter_map(|p| {
                    if p.is_empty() {
                        None
                    } else {
                        gguf.meta_f32(&format!("{}.{}", p, key))
                    }
                })
                .next()
                .or_else(|| gguf.meta_f32(&format!("llama.{}", key)))
        };

        let meta_str_prefixed = |key: &str| -> Option<String> {
            prefixes
                .iter()
                .filter_map(|p| {
                    if p.is_empty() {
                        None
                    } else {
                        gguf.meta_str(&format!("{}.{}", p, key))
                    }
                })
                .next()
                .or_else(|| gguf.meta_str(&format!("llama.{}", key)))
                .map(|s| s.to_string())
        };

        let d_model = meta_u32_prefixed("embedding_length")
            .ok_or_else(|| "Missing embedding_length".to_string())?;

        let n_layers =
            meta_u32_prefixed("block_count").ok_or_else(|| "Missing block_count".to_string())?;

        let n_heads = meta_u32_prefixed("attention.head_count")
            .ok_or_else(|| "Missing head_count".to_string())?;

        let n_kv_heads = meta_u32_prefixed("attention.head_count_kv").unwrap_or(n_heads);

        let head_dim_meta = meta_u32_prefixed("attention.head_dim")
            .or_else(|| meta_u32_prefixed("attention.key_length"))
            .or_else(|| meta_u32_prefixed("attention.value_length"));
        let head_dim = head_dim_meta.unwrap_or_else(|| {
            // Fall back to d_model / n_heads (standard for most Llama models)
            d_model / n_heads
        });
        let explicit_head_dim = head_dim_meta.is_some();

        let d_ffn = meta_u32_prefixed("feed_forward_length").unwrap_or(d_model * 4);

        let vocab_size = gguf
            .meta_u32("tokenizer.ggml.tokens")
            .or_else(|| gguf.meta_u32("llama.vocab_size"))
            .or_else(|| {
                gguf.meta_array("tokenizer.ggml.tokens")
                    .map(|a| a.len() as u32)
            })
            .ok_or_else(|| "Missing vocab_size".to_string())?;

        let max_seq_len = meta_u32_prefixed("context_length").unwrap_or(2048);

        let rope_theta = meta_f32_prefixed("rope.freq_base").unwrap_or(10000.0);

        let rms_norm_eps = meta_f32_prefixed("attention.layer_norm_rms_epsilon").unwrap_or(1e-5);

        let rope_dim = meta_u32_prefixed("rope.dimension_count").unwrap_or(head_dim);

        let rope_scaling_type = meta_str_prefixed("rope.scaling.type");
        let rope_scaling_factor = meta_f32_prefixed("rope.scaling.factor");
        let original_ctx_len = meta_u32_prefixed("rope.scaling.original_context_length");

        let n_experts = meta_u32_prefixed("expert_count").unwrap_or(0);
        let n_experts_per_tok = meta_u32_prefixed("expert_used_count").unwrap_or(0);

        // Model-author sampling recommendations (`general.sampling.*`). These
        // keys are global, not architecture-prefixed, and every Bonsai GGUF
        // ships them — ignoring them meant every model ran on Hearth's
        // hardcoded 0.7/0.9/40 regardless of what it was built for.
        let sampling_temperature = gguf.meta_f32("general.sampling.temp");
        let sampling_top_p = gguf.meta_f32("general.sampling.top_p");
        let sampling_top_k = gguf.meta_u32("general.sampling.top_k");
        let sampling_min_p = gguf.meta_f32("general.sampling.min_p");

        let post_norm = arch == "gemma" || arch == "gemma2";
        let embed_scale = post_norm;

        // Per-layer recurrent (linear attention) flags for hybrid archs.
        let recurrent_layers = if arch == "qwen35" {
            // Prefer explicit metadata when present, else full_attention_interval
            // (recurrent = every layer NOT at a multiple of the interval).
            if let Some(arr) = gguf.meta_array("qwen35.attention.recurrent_layers") {
                Some(
                    arr.iter()
                        .filter_map(|v| v.as_bool())
                        .collect::<Vec<bool>>(),
                )
            } else {
                let interval = gguf
                    .meta_u32(&format!("{}.full_attention_interval", prefixes[0]))
                    .or_else(|| gguf.meta_u32("qwen35.full_attention_interval"))
                    .unwrap_or(4) as usize;
                let n = n_layers as usize;
                Some((0..n).map(|i| (i + 1) % interval != 0).collect())
                // Note: for qwen35 the fork marks layer i recurrent when
                // (i+1) % interval != 0, i.e. layers 0,1,2 recurrent, 3 full.
            }
        } else {
            None
        };

        Ok(ModelConfig {
            arch,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            d_model,
            d_ffn,
            vocab_size,
            max_seq_len,
            rope_theta,
            rms_norm_eps,
            rope_dim,
            rope_scaling_type,
            rope_scaling_factor,
            original_ctx_len,
            n_experts,
            n_experts_per_tok,
            post_norm,
            embed_scale,
            explicit_head_dim,
            recurrent_layers,
            sampling_temperature,
            sampling_top_p,
            sampling_top_k,
            sampling_min_p,
        })
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_heads == 0 {
            return Err("Config validation: n_heads is 0".to_string());
        }
        if self.d_model == 0 {
            return Err("Config validation: d_model is 0".to_string());
        }
        if self.head_dim == 0 {
            return Err("Config validation: head_dim is 0".to_string());
        }
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(format!(
                "Config validation: d_model ({}) not divisible by n_heads ({})",
                self.d_model, self.n_heads
            ));
        }
        if !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(format!(
                "Config validation: n_heads ({}) not divisible by n_kv_heads ({})",
                self.n_heads, self.n_kv_heads
            ));
        }
        // d_model need not divide by n_heads on hybrid archs that emit query+gate
        // from a single projection (qwen35: 24 heads x 256 dim = 6144 > 5120).
        if self.explicit_head_dim
            && self.head_dim * self.n_heads < self.d_model
            && !matches!(self.arch.as_str(), "qwen35" | "qwen3next" | "qwen35moe")
        {
            return Err(format!(
                "Config validation: head_dim ({}) * n_heads ({}) < d_model ({})",
                self.head_dim, self.n_heads, self.d_model
            ));
        }
        if self.d_ffn == 0 {
            return Err("Config validation: d_ffn is 0".to_string());
        }
        if self.n_layers == 0 {
            return Err("Config validation: n_layers is 0".to_string());
        }
        if self.vocab_size == 0 {
            return Err("Config validation: vocab_size is 0".to_string());
        }
        if self.max_seq_len == 0 {
            return Err("Config validation: max_seq_len is 0".to_string());
        }
        if self.n_experts > 0 && self.n_experts_per_tok == 0 {
            return Err(format!(
                "Config validation: n_experts ({}) > 0 but n_experts_per_tok is 0",
                self.n_experts
            ));
        }
        if self.n_experts_per_tok > self.n_experts {
            return Err(format!(
                "Config validation: n_experts_per_tok ({}) > n_experts ({})",
                self.n_experts_per_tok, self.n_experts
            ));
        }
        if self.rms_norm_eps <= 0.0 {
            return Err(format!(
                "Config validation: rms_norm_eps ({}) must be positive",
                self.rms_norm_eps
            ));
        }
        if self.rope_theta <= 0.0 {
            return Err(format!(
                "Config validation: rope_theta ({}) must be positive",
                self.rope_theta
            ));
        }
        Ok(())
    }
}

fn get_arch_prefixes(arch: &str) -> [&str; 3] {
    match arch {
        "llama" | "mistral" | "mixtral" | "gemma" => ["llama", arch, ""],
        "qwen2" | "qwen2.5" => ["qwen2", arch, ""],
        "qwen3" | "qwen3moe" | "qwen3next" | "qwen35" | "qwen35moe" => ["qwen3", arch, ""],
        "phi3" | "phi-3" => ["phi3", arch, ""],
        "starcoder2" => ["starcoder2", arch, ""],
        "gemma2" => ["gemma", arch, ""],
        "command-r" => ["command-r", arch, ""],
        "dbrx" => ["dbrx", arch, ""],
        "falcon" => ["falcon", arch, ""],
        "yi" => ["yi", arch, ""],
        "deepseek2" => ["deepseek2", arch, ""],
        _ => ["llama", arch, ""],
    }
}
