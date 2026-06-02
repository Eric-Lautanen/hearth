use hearth_core::LoadStrategy;
use hearth_llm::LlamaModel;

pub struct TextEncoder {
    model: LlamaModel,
    qwen_dim: usize,
    joint_dim: usize,
    d_model: usize,
}

impl TextEncoder {
    pub fn load(model_path: &str) -> Result<Self, String> {
        let model = LlamaModel::load(model_path, LoadStrategy::CpuOnly)?;
        Ok(TextEncoder {
            model,
            qwen_dim: 2560,
            joint_dim: 7680,
            d_model: 3072,
        })
    }

    pub fn encode(&self, prompt: &str, max_tokens: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
        let tokens = {
            let mut tok = self
                .model
                .tokenizer()
                .lock()
                .map_err(|e| format!("Tokenizer lock: {}", e))?;
            tok.encode(prompt, false)
        };

        let tokens: Vec<u32> = tokens.into_iter().take(max_tokens).collect();
        let seq_len = tokens.len();
        eprintln!(
            "[text_encoder] {} tokens for: \"{}\"",
            seq_len,
            &prompt[..prompt.len().min(60)]
        );

        let (hidden_states, pooled_raw) = self.model.encode_text(&tokens)?;
        eprintln!(
            "[text_encoder] Hidden: {}×{}, pooled: {}",
            seq_len,
            self.qwen_dim,
            pooled_raw.len()
        );

        let prompt_embeds = zero_pad(&hidden_states, seq_len, self.qwen_dim, self.joint_dim);
        let pooled_embeds = zero_pad(&pooled_raw, 1, self.qwen_dim, self.d_model);

        Ok((prompt_embeds, pooled_embeds))
    }
}

fn zero_pad(src: &[f32], seq_len: usize, src_dim: usize, dst_dim: usize) -> Vec<f32> {
    let mut dst = vec![0.0f32; seq_len * dst_dim];
    for s in 0..seq_len {
        let src_off = s * src_dim;
        let dst_off = s * dst_dim;
        let copy_len = src_dim.min(dst_dim);
        dst[dst_off..dst_off + copy_len].copy_from_slice(&src[src_off..src_off + copy_len]);
    }
    dst
}
