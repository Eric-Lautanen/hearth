pub(crate) struct ForwardTimers {
    pub embed_us: u128,
    pub attn_norm_us: u128,
    pub qkv_quant_us: u128,
    pub qkv_matmul_us: u128,
    pub qk_head_norm_us: u128,
    pub rope_us: u128,
    pub kv_cache_write_us: u128,
    pub attention_us: u128,
    pub attn_output_matmul_us: u128,
    pub ffn_norm_us: u128,
    pub ffn_quant_us: u128,
    pub ffn_gate_up_matmul_us: u128,
    pub silu_mul_us: u128,
    pub ffn_down_matmul_us: u128,
    pub output_norm_us: u128,
    pub lm_head_matmul_us: u128,
    pub total_us: u128,
}

impl ForwardTimers {
    pub fn reset(&mut self) {
        self.embed_us = 0;
        self.attn_norm_us = 0;
        self.qkv_quant_us = 0;
        self.qkv_matmul_us = 0;
        self.qk_head_norm_us = 0;
        self.rope_us = 0;
        self.kv_cache_write_us = 0;
        self.attention_us = 0;
        self.attn_output_matmul_us = 0;
        self.ffn_norm_us = 0;
        self.ffn_quant_us = 0;
        self.ffn_gate_up_matmul_us = 0;
        self.silu_mul_us = 0;
        self.ffn_down_matmul_us = 0;
        self.output_norm_us = 0;
        self.lm_head_matmul_us = 0;
        self.total_us = 0;
    }
}

pub(crate) struct ForwardScratch {
    pub(crate) hidden: Vec<f32>,
    pub(crate) residual: Vec<f32>,
    pub(crate) attn_out: Vec<f32>,
    pub(crate) q_buf: Vec<f32>,
    pub(crate) k_buf: Vec<f32>,
    pub(crate) v_buf: Vec<f32>,
    pub(crate) q_heads: Vec<f32>,
    pub(crate) k_heads: Vec<f32>,
    pub(crate) v_heads: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) ffn_tmp: Vec<f32>,
    pub(crate) norm_tmp: Vec<f32>,
    pub(crate) attn_scores: Vec<f32>,
    pub(crate) moe_gate: Vec<f32>,
    pub(crate) moe_ffn: Vec<f32>,
    pub(crate) x_q8: Vec<u8>,
    pub(crate) ffn_q8: Vec<u8>,
    pub(crate) scratch_q8: Vec<u8>,
    pub(crate) head_norm_tmp: Vec<f32>,
    pub(crate) timers: ForwardTimers,
}

pub(crate) struct BatchScratch {
    pub(crate) hidden: Vec<f32>,
    pub(crate) residual: Vec<f32>,
    pub(crate) attn_out: Vec<f32>,
    pub(crate) q_buf: Vec<f32>,
    pub(crate) k_buf: Vec<f32>,
    pub(crate) v_buf: Vec<f32>,
    pub(crate) q_heads: Vec<f32>,
    pub(crate) k_heads: Vec<f32>,
    pub(crate) v_heads: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) ffn_tmp: Vec<f32>,
    pub(crate) norm_tmp: Vec<f32>,
    pub(crate) attn_scores: Vec<f32>,
    pub(crate) batch_q8: Vec<u8>,
    pub(crate) head_norm_tmp: Vec<f32>,
}
