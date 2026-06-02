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

        let head_dim = meta_u32_prefixed("attention.head_dim").unwrap_or_else(|| {
            // Fall back to d_model / n_heads (standard for most Llama models)
            d_model / n_heads
        });

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

        let post_norm = arch == "gemma" || arch == "gemma2";
        let embed_scale = post_norm;

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
        if self.head_dim * self.n_heads < self.d_model {
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
        "qwen3" | "qwen3moe" | "qwen3next" => ["qwen3", arch, ""],
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
