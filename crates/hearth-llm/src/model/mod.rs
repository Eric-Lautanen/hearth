mod gpu;
mod kv_budget;
mod matmul;
mod moe;
mod qwen35;
mod scratch;
mod tensor;

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
    Arc,
};

use hearth_core::{
    EngineOutput, LoadStrategy, Model, ModelError, PipelineRequest, RunStats, SamplerConfig,
};
use hearth_gguf::GgufFile;
use hearth_tokenizer::Tokenizer;

use crate::config::ModelConfig;
use crate::kvcache::{KVCache, KVStorage};
use crate::ops;
use crate::pool::ThreadPool;
use hearth_compute::GpuCompute;

use scratch::{BatchScratch, ForwardScratch, ForwardTimers};
use tensor::TensorEntry;

struct LayerTensorNames {
    attn_norm: String,
    attn_q: String,
    attn_k: String,
    attn_v: String,
    attn_q_norm: String,
    attn_k_norm: String,
    attn_output: String,
    ffn_norm: String,
    ffn_gate: String,
    ffn_up: String,
    ffn_down: String,
}

pub struct LlamaModel {
    config: ModelConfig,
    tokenizer: std::sync::Mutex<Tokenizer>,
    tensors: HashMap<String, TensorEntry>,
    strategy: LoadStrategy,
    gpu: Option<GpuCompute>,
    gpu_layers: usize,
    scratch: std::sync::Mutex<ForwardScratch>,
    row_buf: std::sync::Mutex<Vec<f32>>,
    /// Pre-allocated CPU buffer for token embedding lookup in forward_gpu.
    /// Avoids a per-token Vec allocation on the GPU decode path.
    hidden_gpu: std::sync::Mutex<Vec<f32>>,
    norm_cache: HashMap<String, Vec<f32>>,
    batch: std::sync::Mutex<BatchScratch>,
    layer_names: Vec<LayerTensorNames>,
    lm_head_name: String,
    pool: ThreadPool,
    rope_cache: ops::RopeCache,
    /// Hybrid architectures (e.g. qwen35 / Ternary Bonsai 2) own their entire
    /// forward path; every other field is inert placeholder state.
    hybrid: Option<qwen35::Qwen35Core>,
    /// Bytes the f32 KV cache occupies at the allocated context. Allocated at
    /// load, so this is the model's real RAM cost beyond its weight file — the
    /// UI shows it so a big-context model can be spotted before chatting.
    kv_cache_bytes: u64,
}

impl LlamaModel {
    pub fn load(path: &str, strategy: LoadStrategy) -> Result<Self, String> {
        let gguf = GgufFile::open(path).map_err(|e| format!("GGUF: {}", e))?;
        let mut config = ModelConfig::from_gguf(&gguf)?;

        // Hybrid linear-attention architectures have their own forward path, state
        // layout and geometry (e.g. d_model not divisible by n_heads); they must
        // dispatch BEFORE the dense-architecture validation rules.
        if config.arch == "qwen35" {
            return qwen35::load(path, strategy);
        }
        config.validate()?;

        let tokenizer = Tokenizer::from_gguf(&gguf).map_err(|e| format!("Tokenizer: {}", e))?;

        // Large tensors borrow the GGUF mmap instead of being copied, so a
        // multi-GB model loads on a machine with less RAM than the file.
        // HEARTH_PIN_WEIGHTS overrides that and copies everything into RAM.
        const COPY_THRESHOLD: usize = 16 * 1024 * 1024;
        let copy_all = qwen35::pin_weights();
        let mut tensors = HashMap::new();
        for info in &gguf.tensors {
            let byte_len = info.byte_size();
            let entry = if byte_len > COPY_THRESHOLD && !copy_all {
                let start = (gguf.data_offset + info.offset) as usize;
                TensorEntry::mapped(
                    gguf.mmap_arc(),
                    start,
                    byte_len,
                    info.dtype,
                    info.shape.clone(),
                )
            } else {
                TensorEntry::owned(
                    gguf.tensor_data(info).to_vec(),
                    info.dtype,
                    info.shape.clone(),
                )
            };
            tensors.insert(info.name.clone(), entry);
        }

        let d = config.d_model as usize;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let mut head_dim = config.head_dim as usize;
        let d_ffn = config.d_ffn as usize;
        // Derive head_dim from the Q projection only when the GGUF metadata did
        // not state it explicitly. Note this would be wrong for qwen35 (attn_q
        // emits query+gate, 2x width) — but that arch never reaches this code.
        if !config.explicit_head_dim {
            if let Some(q_entry) = tensors.get("blk.0.attn_q.weight") {
                let q_rows = q_entry.n_rows();
                let computed_head_dim = q_rows / n_heads;
                if computed_head_dim * n_heads == q_rows && computed_head_dim != head_dim {
                    head_dim = computed_head_dim;
                }
            }
        }
        config.head_dim = head_dim as u32;

        // Cap the KV cache context by a byte budget *after* head_dim is settled:
        // every dense layer keeps K+V, so a model declaring a huge context would
        // allocate an F32 cache beyond system memory at load. Applied to
        // config.max_seq_len itself so every consumer (generate_text, warmup,
        // scratch) sees the capped value.
        let kv_per_token =
            kv_budget::kv_bytes_per_token(config.n_layers as usize, n_kv_heads, head_dim);
        let max_seq = kv_budget::ctx_cap(
            config.max_seq_len as usize,
            config.n_layers as usize,
            n_kv_heads,
            head_dim,
            Some(kv_budget::DENSE_CTX_CAP),
        );
        config.max_seq_len = max_seq as u32;
        eprintln!(
            "[kv] context {} tokens · {} per token · {} KV cache",
            max_seq,
            kv_budget::fmt_bytes(kv_per_token),
            kv_budget::fmt_bytes(max_seq as u64 * kv_per_token)
        );

        let max_row = d.max(d_ffn).max(head_dim);

        let nq = n_heads * head_dim;
        let nkv = n_kv_heads * head_dim;
        let scratch = ForwardScratch {
            hidden: vec![0.0f32; d],
            residual: vec![0.0f32; d],
            attn_out: vec![0.0f32; nq],
            q_buf: vec![0.0f32; nq],
            k_buf: vec![0.0f32; nkv],
            v_buf: vec![0.0f32; nkv],
            q_heads: vec![0.0f32; n_heads * head_dim],
            k_heads: vec![0.0f32; n_kv_heads * head_dim],
            v_heads: vec![0.0f32; n_kv_heads * head_dim],
            gate: vec![0.0f32; d_ffn],
            up: vec![0.0f32; d_ffn],
            ffn_tmp: vec![0.0f32; d_ffn],
            norm_tmp: vec![0.0f32; d],
            attn_scores: vec![0.0f32; max_seq],
            moe_gate: Vec::new(),
            moe_ffn: Vec::new(),
            x_q8: Vec::new(),
            ffn_q8: Vec::new(),
            scratch_q8: Vec::new(),
            head_norm_tmp: vec![0.0f32; head_dim],
            timers: ForwardTimers {
                embed_us: 0,
                attn_norm_us: 0,
                qkv_quant_us: 0,
                qkv_matmul_us: 0,
                qk_head_norm_us: 0,
                rope_us: 0,
                kv_cache_write_us: 0,
                attention_us: 0,
                attn_output_matmul_us: 0,
                ffn_norm_us: 0,
                ffn_quant_us: 0,
                ffn_gate_up_matmul_us: 0,
                silu_mul_us: 0,
                ffn_down_matmul_us: 0,
                output_norm_us: 0,
                lm_head_matmul_us: 0,
                total_us: 0,
            },
        };

        let (mut gpu, gpu_layers, effective_strategy) = match strategy {
            LoadStrategy::GpuFull | LoadStrategy::CpuOffload => {
                let gpu = pollster::block_on(GpuCompute::new());
                if let Some(ref g) = gpu {
                    eprintln!("[hearth-llm] GPU compute initialized, uploading tensors...");
                    gpu::upload_tensors_to_gpu(g, &tensors);
                } else {
                    eprintln!("[hearth-llm] GPU unavailable, falling back to CPU-only");
                }
                let layers = if gpu.is_some() {
                    config.n_layers as usize
                } else {
                    0
                };
                let eff = if gpu.is_some() {
                    LoadStrategy::GpuFull
                } else {
                    LoadStrategy::CpuOnly
                };
                (gpu, layers, eff)
            }
            LoadStrategy::LayerSplit(n) => {
                let gpu = pollster::block_on(GpuCompute::new());
                if let Some(ref g) = gpu {
                    eprintln!(
                        "[hearth-llm] GPU compute initialized ({} layers), uploading tensors...",
                        n
                    );
                    gpu::upload_tensors_to_gpu(g, &tensors);
                } else {
                    eprintln!("[hearth-llm] GPU unavailable, falling back to CPU-only");
                }
                let layers = if gpu.is_some() { n } else { 0 };
                let eff = if gpu.is_some() {
                    LoadStrategy::LayerSplit(n)
                } else {
                    LoadStrategy::CpuOnly
                };
                (gpu, layers, eff)
            }
            LoadStrategy::CpuOnly => {
                eprintln!("[hearth-llm] Loading CPU-only");
                (None, 0, LoadStrategy::CpuOnly)
            }
        };

        let mut norm_cache: HashMap<String, Vec<f32>> = HashMap::new();
        {
            let norm_suffixes = [
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn_q_norm.weight",
                "attn_k_norm.weight",
                "post_attention_norm.weight",
                "post_ffn_norm.weight",
            ];
            for (name, entry) in &tensors {
                let is_norm =
                    norm_suffixes.iter().any(|s| name.ends_with(s)) || name == "output_norm.weight";
                if is_norm && entry.shape.len() == 1 {
                    let len = entry.shape[0] as usize;
                    let mut buf = vec![0.0f32; len];
                    if hearth_quant::dequantize(entry.dtype, entry.data(), &mut buf).is_ok() {
                        norm_cache.insert(name.clone(), buf);
                    }
                }
            }
        }

        // Initialize GPU-side KV cache (one buffer per layer, capped to fit iGPU memory)
        if let Some(ref mut gpu_inner) = gpu {
            // Each f32 per position = 28 layers × 2 (K+V) × n_kv_heads × head_dim × 4 bytes
            // With 512MB iGPU and 237MB weights already uploaded, ~200-250MB remain.
            // Cap cache to 512 positions (~115MB total) to leave safety margin.
            let cache_max_seq = max_seq.min(512);
            let per_layer = (n_kv_heads * cache_max_seq * head_dim) as u64 * 2 * 4; // K + V in f32 (4 bytes each)
            gpu_inner.kv_cache = (0..config.n_layers)
                .map(|_| gpu_inner.create_storage_buffer(per_layer, "kv_cache"))
                .collect();
            gpu_inner.cache_max_seq = cache_max_seq as u32;
            // Upload norm weights to GPU once (avoids create_buffer_init per layer in forward_gpu)
            for (name, f32_data) in &norm_cache {
                let buf = gpu_inner.upload_f32(f32_data, &format!("norm:{}", name));
                gpu_inner.norm_buffers.insert(name.clone(), buf);
            }
        }

        let batch = BatchScratch {
            hidden: Vec::new(),
            residual: Vec::new(),
            attn_out: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            q_heads: Vec::new(),
            k_heads: Vec::new(),
            v_heads: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            ffn_tmp: Vec::new(),
            norm_tmp: Vec::new(),
            attn_scores: Vec::new(),
            batch_q8: Vec::new(),
            head_norm_tmp: Vec::new(),
        };

        let layer_names: Vec<LayerTensorNames> = (0..config.n_layers as usize)
            .map(|i| LayerTensorNames {
                attn_norm: format!("blk.{}.attn_norm.weight", i),
                attn_q: format!("blk.{}.attn_q.weight", i),
                attn_k: format!("blk.{}.attn_k.weight", i),
                attn_v: format!("blk.{}.attn_v.weight", i),
                attn_q_norm: format!("blk.{}.attn_q_norm.weight", i),
                attn_k_norm: format!("blk.{}.attn_k_norm.weight", i),
                attn_output: format!("blk.{}.attn_output.weight", i),
                ffn_norm: format!("blk.{}.ffn_norm.weight", i),
                ffn_gate: format!("blk.{}.ffn_gate.weight", i),
                ffn_up: format!("blk.{}.ffn_up.weight", i),
                ffn_down: format!("blk.{}.ffn_down.weight", i),
            })
            .collect();

        let lm_head_name = if tensors.contains_key("output.weight") {
            "output.weight".to_string()
        } else {
            "token_embd.weight".to_string()
        };

        let pool = ThreadPool::new(8);

        let rope_cache = ops::RopeCache::new(
            max_seq,
            config.rope_dim as usize,
            config.rope_theta,
            config.rope_scaling_type.as_deref(),
            config.rope_scaling_factor,
            config.original_ctx_len,
        );

        Ok(LlamaModel {
            config,
            tokenizer: std::sync::Mutex::new(tokenizer),
            tensors,
            strategy: effective_strategy,
            gpu,
            gpu_layers,
            scratch: std::sync::Mutex::new(scratch),
            row_buf: std::sync::Mutex::new(vec![0.0f32; max_row]),
            hidden_gpu: std::sync::Mutex::new(vec![0.0f32; d]),
            norm_cache,
            batch: std::sync::Mutex::new(batch),
            layer_names,
            lm_head_name,
            pool,
            rope_cache,
            hybrid: None,
            kv_cache_bytes: max_seq as u64 * kv_per_token,
        })
    }

    /// Bytes of f32 KV cache this model allocated at load.
    pub fn kv_cache_bytes(&self) -> u64 {
        self.kv_cache_bytes
    }

    /// Wrap a hybrid-architecture core (qwen35) in the LlamaModel shell.
    ///
    /// The shell always adopts the core's final (capped / corrected) config, so
    /// the caller's config is not read here.
    pub(crate) fn from_hybrid(
        core: qwen35::Qwen35Core,
        _config: ModelConfig,
        tokenizer: Tokenizer,
    ) -> Self {
        let config = core.config_snapshot();
        LlamaModel {
            config,
            tokenizer: std::sync::Mutex::new(tokenizer),
            tensors: HashMap::new(),
            strategy: LoadStrategy::CpuOnly,
            gpu: None,
            gpu_layers: 0,
            scratch: std::sync::Mutex::new(ForwardScratch::placeholder()),
            row_buf: std::sync::Mutex::new(Vec::new()),
            hidden_gpu: std::sync::Mutex::new(Vec::new()),
            norm_cache: HashMap::new(),
            batch: std::sync::Mutex::new(BatchScratch::placeholder()),
            layer_names: Vec::new(),
            lm_head_name: "output.weight".to_string(),
            pool: ThreadPool::new(1),
            rope_cache: ops::RopeCache::new(16, 16, 10000.0, None, None, None),
            kv_cache_bytes: core.kv_cache_bytes(),
            hybrid: Some(core),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::needless_range_loop)]
    fn forward(
        &self,
        token_ids: &[u32],
        pos: usize,
        caches: &mut [KVCache],
        logits: &mut [f32],
        cancel: &AtomicBool,
        sc: &mut ForwardScratch,
        rb: &mut Vec<f32>,
    ) -> Result<(), String> {
        let fwd_t0 = std::time::Instant::now();
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }

        let d = self.config.d_model as usize;
        let n_heads = self.config.n_heads as usize;
        let n_kv_heads = self.config.n_kv_heads as usize;
        let head_dim = self.config.head_dim as usize;
        let n_layers = self.config.n_layers as usize;
        let ffn_dim = self.config.d_ffn as usize;
        let nq = n_heads * head_dim;
        let nkv = n_kv_heads * head_dim;
        let seq_len = caches[0].current_len + 1;
        let token_id = token_ids[0] as usize;

        if sc.attn_scores.len() < seq_len {
            sc.attn_scores.resize(seq_len, 0.0f32);
        }

        sc.timers.reset();

        let t0 = std::time::Instant::now();
        {
            let entry = self
                .tensors
                .get("token_embd.weight")
                .ok_or_else(|| "Missing token_embd.weight".to_string())?;
            let n = entry.n_cols().min(d);
            sc.hidden[..d].fill(0.0f32);
            if token_id < entry.n_rows() {
                hearth_quant::dequantize(
                    entry.dtype,
                    entry.row_data(token_id),
                    &mut sc.hidden[..n],
                )
                .map_err(|e| format!("Embed dequant: {}", e))?;
            }
            if self.config.embed_scale {
                let scale = (d as f32).sqrt();
                for i in 0..d {
                    sc.hidden[i] *= scale;
                }
            }
        }
        sc.timers.embed_us = t0.elapsed().as_micros();

        // FORWARD ALL LAYERS
        let mut has_qk_norm = false;
        for layer in 0..n_layers {
            if cancel.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            let ln = &self.layer_names[layer];

            let t0 = std::time::Instant::now();
            sc.norm_tmp[..d].copy_from_slice(&sc.hidden[..d]);
            let attn_norm = self.get_1d(&ln.attn_norm, d)?;
            self.norm(
                &sc.norm_tmp[..d],
                attn_norm,
                self.config.rms_norm_eps,
                &mut sc.residual[..d],
            );
            sc.timers.attn_norm_us += t0.elapsed().as_micros();

            let t0 = std::time::Instant::now();
            sc.x_q8.clear();
            hearth_quant::quantize_q8_0(&sc.residual[..d], &mut sc.x_q8);
            sc.timers.qkv_quant_us += t0.elapsed().as_micros();

            let t0 = std::time::Instant::now();
            self.matmul_fused_qkv(
                &ln.attn_q,
                &ln.attn_k,
                &ln.attn_v,
                &sc.residual[..d],
                &mut sc.q_buf[..nq],
                &mut sc.k_buf[..nkv],
                &mut sc.v_buf[..nkv],
                rb,
                layer,
                Some(&sc.x_q8[..]),
            )?;

            sc.q_heads[..nq].copy_from_slice(&sc.q_buf[..nq]);
            sc.k_heads[..nkv].copy_from_slice(&sc.k_buf[..nkv]);
            sc.v_heads[..nkv].copy_from_slice(&sc.v_buf[..nkv]);
            sc.timers.qkv_matmul_us += t0.elapsed().as_micros();

            let t0 = std::time::Instant::now();
            if self.tensors.contains_key(&ln.attn_q_norm) {
                has_qk_norm = true;
                let q_norm = self.get_1d(&ln.attn_q_norm, head_dim)?;
                for h in 0..n_heads {
                    let s = h * head_dim;
                    sc.head_norm_tmp[..head_dim].copy_from_slice(&sc.q_heads[s..s + head_dim]);
                    self.norm(
                        &sc.head_norm_tmp[..head_dim],
                        q_norm,
                        self.config.rms_norm_eps,
                        &mut sc.q_heads[s..s + head_dim],
                    );
                }
            }
            if self.tensors.contains_key(&ln.attn_k_norm) {
                has_qk_norm = true;
                let k_norm = self.get_1d(&ln.attn_k_norm, head_dim)?;
                for h in 0..n_kv_heads {
                    let s = h * head_dim;
                    sc.head_norm_tmp[..head_dim].copy_from_slice(&sc.k_heads[s..s + head_dim]);
                    self.norm(
                        &sc.head_norm_tmp[..head_dim],
                        k_norm,
                        self.config.rms_norm_eps,
                        &mut sc.k_heads[s..s + head_dim],
                    );
                }
            }
            sc.timers.qk_head_norm_us += t0.elapsed().as_micros();

            let t0 = std::time::Instant::now();
            let rope_dim = self.config.rope_dim as usize;
            for h in 0..n_heads {
                let s = h * head_dim;
                self.rope_cache.apply(&mut sc.q_heads[s..s + rope_dim], pos);
            }
            for h in 0..n_kv_heads {
                let s = h * head_dim;
                self.rope_cache.apply(&mut sc.k_heads[s..s + rope_dim], pos);
            }
            sc.timers.rope_us += t0.elapsed().as_micros();

            let t0 = std::time::Instant::now();
            for h in 0..n_kv_heads {
                let s = h * head_dim;
                caches[layer].write_kv(
                    pos,
                    h,
                    &sc.k_heads[s..s + head_dim],
                    &sc.v_heads[s..s + head_dim],
                );
            }
            sc.timers.kv_cache_write_us += t0.elapsed().as_micros();

            sc.attn_out[..nq].fill(0.0f32);
            let kv_repeat = n_heads / n_kv_heads;
            let t0 = std::time::Instant::now();
            let gpu_attn = if layer < self.gpu_layers && self.gpu_supports_model_dtypes() {
                if let Some(ref gpu) = self.gpu {
                    if !caches[layer].is_q8_0() {
                        // Compact K/V from cache: [n_kv_heads x seq_len x head_dim]
                        let max_s = self.config.max_seq_len as usize;
                        let total_kv = n_kv_heads * seq_len * head_dim;
                        let mut compact_k = vec![0.0f32; total_kv];
                        let mut compact_v = vec![0.0f32; total_kv];
                        for kvh in 0..n_kv_heads {
                            let src_off = kvh * max_s * head_dim;
                            let dst_off = kvh * seq_len * head_dim;
                            compact_k[dst_off..dst_off + seq_len * head_dim].copy_from_slice(
                                &caches[layer].k[src_off..src_off + seq_len * head_dim],
                            );
                            compact_v[dst_off..dst_off + seq_len * head_dim].copy_from_slice(
                                &caches[layer].v[src_off..src_off + seq_len * head_dim],
                            );
                        }
                        let q_buf = gpu.upload_f32(&sc.q_heads[..nq], "q_attn");
                        // Combine K and V into one buffer for flash_attention (cache API)
                        let mut kv_combined = compact_k;
                        kv_combined.extend_from_slice(&compact_v);
                        let kv_buf = gpu.upload_f32(&kv_combined, "kv_attn");
                        if let Some(out_buf) = gpu.flash_attention(
                            &q_buf,
                            &kv_buf,
                            0,
                            (total_kv as u64) * 4,
                            n_heads as u32,
                            n_kv_heads as u32,
                            head_dim as u32,
                            seq_len as u32,
                            seq_len as u32,
                            pos as u32,
                        ) {
                            let out = gpu.readback_f32(&out_buf, nq);
                            sc.attn_out[..nq].copy_from_slice(&out[..nq]);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };
            if !gpu_attn {
                if caches[layer].is_q8_0() {
                    let blocks_per_pos = head_dim.div_ceil(32);
                    let q8_row = blocks_per_pos * 34;
                    let q8_stride = caches[layer].max_seq_len * q8_row;
                    for kvh in 0..n_kv_heads {
                        let head_off = kvh * q8_stride;
                        let head_len = seq_len * q8_row;
                        let k_q8 = &caches[layer].k_q8[head_off..head_off + head_len];
                        let v_q8 = &caches[layer].v_q8[head_off..head_off + head_len];
                        for r in 0..kv_repeat {
                            let qh = kvh * kv_repeat + r;
                            let qs = qh * head_dim;
                            let os = qh * head_dim;
                            sc.norm_tmp[..head_dim].copy_from_slice(&sc.q_heads[qs..qs + head_dim]);
                            ops::attention_q8_0(
                                &sc.norm_tmp[..head_dim],
                                k_q8,
                                v_q8,
                                seq_len,
                                head_dim,
                                &mut sc.attn_out[os..os + head_dim],
                                &mut sc.attn_scores,
                            );
                        }
                    }
                } else {
                    for qh in 0..n_heads {
                        let kvh = qh / kv_repeat;
                        let qs = qh * head_dim;
                        let os = qh * head_dim;
                        let ks = caches[layer].k_slice(kvh, seq_len);
                        let vs = caches[layer].v_slice(kvh, seq_len);
                        sc.norm_tmp[..head_dim].copy_from_slice(&sc.q_heads[qs..qs + head_dim]);
                        ops::attention(
                            &sc.norm_tmp[..head_dim],
                            ks,
                            vs,
                            seq_len,
                            head_dim,
                            &mut sc.attn_out[os..os + head_dim],
                            &mut sc.attn_scores,
                        );
                    }
                }
            }
            sc.timers.attention_us += t0.elapsed().as_micros();

            let t0 = std::time::Instant::now();
            let attn_q8 = self.q8_scratch(&sc.attn_out[..nq], &mut sc.scratch_q8);
            self.matmul(
                &ln.attn_output,
                &sc.attn_out[..nq],
                &mut sc.q_buf[..d],
                rb,
                layer,
                attn_q8,
            )?;
            for i in 0..d {
                sc.hidden[i] += sc.q_buf[i];
            }
            sc.timers.attn_output_matmul_us += t0.elapsed().as_micros();

            let t0 = std::time::Instant::now();
            sc.norm_tmp[..d].copy_from_slice(&sc.hidden[..d]);
            let ffn_norm = self.get_1d(&ln.ffn_norm, d)?;
            self.norm(
                &sc.norm_tmp[..d],
                ffn_norm,
                self.config.rms_norm_eps,
                &mut sc.residual[..d],
            );
            sc.timers.ffn_norm_us += t0.elapsed().as_micros();

            if self.config.n_experts > 0 {
                let blk = format!("blk.{}", layer);
                let residual_copy = sc.residual[..d].to_vec();
                self.moe_forward(&blk, &residual_copy, sc, rb, layer)?;
            } else {
                let t0 = std::time::Instant::now();
                sc.ffn_q8.clear();
                hearth_quant::quantize_q8_0(&sc.residual[..d], &mut sc.ffn_q8);
                sc.timers.ffn_quant_us += t0.elapsed().as_micros();

                let t0 = std::time::Instant::now();
                self.matmul_fused2(
                    &ln.ffn_gate,
                    &ln.ffn_up,
                    &sc.residual[..d],
                    &mut sc.gate[..ffn_dim],
                    &mut sc.up[..ffn_dim],
                    rb,
                    layer,
                    Some(&sc.ffn_q8[..]),
                )?;
                sc.timers.ffn_gate_up_matmul_us += t0.elapsed().as_micros();

                let t0 = std::time::Instant::now();
                ops::silu(&mut sc.gate[..ffn_dim]);
                ops::mul_elem(
                    &sc.gate[..ffn_dim],
                    &sc.up[..ffn_dim],
                    &mut sc.ffn_tmp[..ffn_dim],
                );
                sc.timers.silu_mul_us += t0.elapsed().as_micros();

                let t0 = std::time::Instant::now();
                let ffn_q8 = self.q8_scratch(&sc.ffn_tmp[..ffn_dim], &mut sc.scratch_q8);
                self.matmul(
                    &ln.ffn_down,
                    &sc.ffn_tmp[..ffn_dim],
                    &mut sc.q_buf[..d],
                    rb,
                    layer,
                    ffn_q8,
                )?;
                for i in 0..d {
                    sc.hidden[i] += sc.q_buf[i];
                }
                sc.timers.ffn_down_matmul_us += t0.elapsed().as_micros();
            }
        }

        let t0 = std::time::Instant::now();
        sc.norm_tmp[..d].copy_from_slice(&sc.hidden[..d]);
        let out_norm = self.get_1d("output_norm.weight", d)?;
        self.norm(
            &sc.norm_tmp[..d],
            out_norm,
            self.config.rms_norm_eps,
            &mut sc.residual[..d],
        );
        sc.timers.output_norm_us = t0.elapsed().as_micros();

        let t0 = std::time::Instant::now();
        let head_q8 = self.q8_scratch(&sc.residual[..d], &mut sc.scratch_q8);
        self.matmul(
            &self.lm_head_name,
            &sc.residual[..d],
            logits,
            rb,
            self.config.n_layers as usize,
            head_q8,
        )?;
        sc.timers.lm_head_matmul_us = t0.elapsed().as_micros();

        sc.timers.total_us = fwd_t0.elapsed().as_micros();

        let layer_ns = n_layers as u128;
        eprintln!(
            "\n[timing] per-token forward pass ({} layers, {} heads, {} head_dim, {} d_model) [μs]:",
            n_layers, n_heads, head_dim, d
        );
        eprintln!(
            "[timing]   {:<30} {:>8} {:>8} {:>8}",
            "section", "total", "/layer", "%total"
        );
        let section_data: [(&str, u128); 17] = [
            ("embed", sc.timers.embed_us),
            ("attn_norm", sc.timers.attn_norm_us),
            ("qkv_quant", sc.timers.qkv_quant_us),
            ("qkv_matmul", sc.timers.qkv_matmul_us),
            ("qk_head_norm", sc.timers.qk_head_norm_us),
            ("rope", sc.timers.rope_us),
            ("kv_cache_write", sc.timers.kv_cache_write_us),
            ("attention", sc.timers.attention_us),
            ("attn_output_matmul", sc.timers.attn_output_matmul_us),
            ("ffn_norm", sc.timers.ffn_norm_us),
            ("ffn_quant", sc.timers.ffn_quant_us),
            ("ffn_gate_up_matmul", sc.timers.ffn_gate_up_matmul_us),
            ("silu_mul", sc.timers.silu_mul_us),
            ("ffn_down_matmul", sc.timers.ffn_down_matmul_us),
            ("output_norm", sc.timers.output_norm_us),
            ("lm_head_matmul", sc.timers.lm_head_matmul_us),
            ("TOTAL", sc.timers.total_us),
        ];
        for (name, val) in &section_data {
            let per_layer = if *name != "embed"
                && *name != "output_norm"
                && *name != "lm_head_matmul"
                && *name != "TOTAL"
            {
                val / layer_ns
            } else {
                0
            };
            let pct = if sc.timers.total_us > 0 {
                (val * 10000 / sc.timers.total_us) as f64 / 100.0
            } else {
                0.0
            };
            eprintln!(
                "[timing]   {:<30} {:>8} {:>8} {:>7.2}%",
                name, val, per_layer, pct
            );
        }
        if has_qk_norm {
            eprintln!("[timing]   (qk_head_norm active for this model)");
        }

        Ok(())
    }

    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn ensure_batch_size(
        &self,
        bs: &mut BatchScratch,
        seq_len: usize,
        d: usize,
        nq: usize,
        nkv: usize,
        ffn_dim: usize,
        head_dim: usize,
        max_seq: usize,
    ) {
        let need = |v: &mut Vec<f32>, n: usize| {
            if v.len() < n {
                v.resize(n, 0.0f32);
            }
        };
        need(&mut bs.hidden, seq_len * d);
        need(&mut bs.residual, seq_len * d);
        need(&mut bs.attn_out, seq_len * nq);
        need(&mut bs.q_buf, seq_len * nq);
        need(&mut bs.k_buf, seq_len * nkv);
        need(&mut bs.v_buf, seq_len * nkv);
        need(&mut bs.q_heads, seq_len * nq);
        need(&mut bs.k_heads, seq_len * nkv);
        need(&mut bs.v_heads, seq_len * nkv);
        need(&mut bs.gate, seq_len * ffn_dim);
        need(&mut bs.up, seq_len * ffn_dim);
        need(&mut bs.ffn_tmp, seq_len * ffn_dim);
        need(&mut bs.norm_tmp, seq_len * d);
        need(&mut bs.attn_scores, max_seq.max(seq_len));
        need(&mut bs.head_norm_tmp, head_dim);
        let q8_needed = seq_len * (d.div_ceil(32) * 34);
        if bs.batch_q8.len() < q8_needed {
            bs.batch_q8.resize(q8_needed, 0u8);
        }
    }

    #[allow(dead_code)]
    #[allow(clippy::needless_range_loop)]
    fn forward_batch(
        &self,
        prompt_ids: &[u32],
        caches: &mut [KVCache],
        logits: &mut [f32],
        cancel: &AtomicBool,
        bs: &mut BatchScratch,
        rb: &mut Vec<f32>,
    ) -> Result<(), String> {
        let seq_len = prompt_ids.len();
        let d = self.config.d_model as usize;
        let n_heads = self.config.n_heads as usize;
        let n_kv_heads = self.config.n_kv_heads as usize;
        let head_dim = self.config.head_dim as usize;
        let n_layers = self.config.n_layers as usize;
        let ffn_dim = self.config.d_ffn as usize;
        let nq = n_heads * head_dim;
        let nkv = n_kv_heads * head_dim;
        let max_seq = self.config.max_seq_len as usize;

        self.ensure_batch_size(bs, seq_len, d, nq, nkv, ffn_dim, head_dim, max_seq);

        {
            let entry = self
                .tensors
                .get("token_embd.weight")
                .ok_or_else(|| "Missing token_embd.weight".to_string())?;
            let n = entry.n_cols().min(d);
            for (s, &tid) in prompt_ids.iter().enumerate() {
                let token_id = tid as usize;
                let row = &mut bs.hidden[s * d..(s + 1) * d];
                row[..d].fill(0.0f32);
                if token_id < entry.n_rows() {
                    hearth_quant::dequantize(entry.dtype, entry.row_data(token_id), &mut row[..n])
                        .map_err(|e| format!("Embed dequant: {}", e))?;
                }
                if self.config.embed_scale {
                    let scale = (d as f32).sqrt();
                    for v in row[..d].iter_mut() {
                        *v *= scale;
                    }
                }
            }
        }

        for layer in 0..n_layers {
            if cancel.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            let p = format!("blk.{}", layer);

            // --- ATTENTION NORM ---
            // Write f32 normed values to residual, then parallel-quantize to Q8 for fast matmul
            let attn_norm = self.get_1d(&format!("{}.attn_norm.weight", p), d)?;
            let aq8_needed = seq_len * (d.div_ceil(32) * 34);
            if bs.batch_q8.len() < aq8_needed {
                bs.batch_q8.resize(aq8_needed, 0u8);
            }
            ops::rms_norm_batch(
                &bs.hidden[..seq_len * d],
                attn_norm,
                self.config.rms_norm_eps,
                &mut bs.residual[..seq_len * d],
                seq_len,
                d,
            );
            self.pool.par_quantize(
                seq_len,
                d,
                bs.residual.as_ptr() as usize,
                bs.batch_q8.as_mut_ptr() as usize,
            );
            self.matmul_batch(
                &format!("{}.attn_q.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.q_buf[..seq_len * nq],
                rb,
                nq,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            self.matmul_batch(
                &format!("{}.attn_k.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.k_buf[..seq_len * nkv],
                rb,
                nkv,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            self.matmul_batch(
                &format!("{}.attn_v.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.v_buf[..seq_len * nkv],
                rb,
                nkv,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;

            bs.q_heads[..seq_len * nq].copy_from_slice(&bs.q_buf[..seq_len * nq]);
            bs.k_heads[..seq_len * nkv].copy_from_slice(&bs.k_buf[..seq_len * nkv]);
            bs.v_heads[..seq_len * nkv].copy_from_slice(&bs.v_buf[..seq_len * nkv]);

            if self
                .tensors
                .contains_key(&format!("{}.attn_q_norm.weight", p))
            {
                let q_norm = self.get_1d(&format!("{}.attn_q_norm.weight", p), head_dim)?;
                for s in 0..seq_len {
                    for h in 0..n_heads {
                        let off = s * nq + h * head_dim;
                        bs.head_norm_tmp[..head_dim]
                            .copy_from_slice(&bs.q_heads[off..off + head_dim]);
                        self.norm(
                            &bs.head_norm_tmp[..head_dim],
                            q_norm,
                            self.config.rms_norm_eps,
                            &mut bs.q_heads[off..off + head_dim],
                        );
                    }
                }
            }
            if self
                .tensors
                .contains_key(&format!("{}.attn_k_norm.weight", p))
            {
                let k_norm = self.get_1d(&format!("{}.attn_k_norm.weight", p), head_dim)?;
                for s in 0..seq_len {
                    for h in 0..n_kv_heads {
                        let off = s * nkv + h * head_dim;
                        bs.head_norm_tmp[..head_dim]
                            .copy_from_slice(&bs.k_heads[off..off + head_dim]);
                        self.norm(
                            &bs.head_norm_tmp[..head_dim],
                            k_norm,
                            self.config.rms_norm_eps,
                            &mut bs.k_heads[off..off + head_dim],
                        );
                    }
                }
            }

            let rope_dim = self.config.rope_dim as usize;
            for s in 0..seq_len {
                let pos = s;
                for h in 0..n_heads {
                    let off = s * nq + h * head_dim;
                    self.rope_cache
                        .apply(&mut bs.q_heads[off..off + rope_dim], pos);
                }
                for h in 0..n_kv_heads {
                    let off = s * nkv + h * head_dim;
                    self.rope_cache
                        .apply(&mut bs.k_heads[off..off + rope_dim], pos);
                }
            }

            for s in 0..seq_len {
                for h in 0..n_kv_heads {
                    let k_off = s * nkv + h * head_dim;
                    let v_off = s * nkv + h * head_dim;
                    caches[layer].write_kv(
                        s,
                        h,
                        &bs.k_heads[k_off..k_off + head_dim],
                        &bs.v_heads[v_off..v_off + head_dim],
                    );
                }
            }

            if caches[layer].is_q8_0() {
                let scratch_needed = self.pool.num_threads() * max_seq;
                if bs.attn_scores.len() < scratch_needed {
                    bs.attn_scores.resize(scratch_needed, 0.0f32);
                }
                ops::attention_batch_q8_0_parallel(
                    &bs.q_heads[..seq_len * nq],
                    &caches[layer].k_q8,
                    &caches[layer].v_q8,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq,
                    &mut bs.attn_out[..seq_len * nq],
                    &mut bs.attn_scores,
                    &self.pool,
                );
            } else {
                let scratch_needed = self.pool.num_threads() * max_seq;
                if bs.attn_scores.len() < scratch_needed {
                    bs.attn_scores.resize(scratch_needed, 0.0f32);
                }
                ops::attention_batch_parallel(
                    &bs.q_heads[..seq_len * nq],
                    &caches[layer].k,
                    &caches[layer].v,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq,
                    &mut bs.attn_out[..seq_len * nq],
                    &mut bs.attn_scores,
                    &self.pool,
                );
            }

            let aq8 = nq.div_ceil(32) * 34;
            let attn_q8_needed = seq_len * aq8;
            if bs.batch_q8.len() < attn_q8_needed {
                bs.batch_q8.resize(attn_q8_needed, 0u8);
            }
            self.pool.par_quantize(
                seq_len,
                nq,
                bs.attn_out.as_ptr() as usize,
                bs.batch_q8.as_mut_ptr() as usize,
            );
            self.matmul_batch(
                &format!("{}.attn_output.weight", p),
                &bs.attn_out[..seq_len * nq],
                &mut bs.q_buf[..seq_len * d],
                rb,
                d,
                nq,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            for i in 0..seq_len * d {
                bs.hidden[i] += bs.q_buf[i];
            }

            // --- FEED-FORWARD NORM ---
            let ffn_norm = self.get_1d(&format!("{}.ffn_norm.weight", p), d)?;
            let fq8_needed = seq_len * (d.div_ceil(32) * 34);
            if bs.batch_q8.len() < fq8_needed {
                bs.batch_q8.resize(fq8_needed, 0u8);
            }
            ops::rms_norm_batch(
                &bs.hidden[..seq_len * d],
                ffn_norm,
                self.config.rms_norm_eps,
                &mut bs.residual[..seq_len * d],
                seq_len,
                d,
            );
            self.pool.par_quantize(
                seq_len,
                d,
                bs.residual.as_ptr() as usize,
                bs.batch_q8.as_mut_ptr() as usize,
            );
            self.matmul_batch(
                &format!("{}.ffn_gate.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.gate[..seq_len * ffn_dim],
                rb,
                ffn_dim,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            self.matmul_batch(
                &format!("{}.ffn_up.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.up[..seq_len * ffn_dim],
                rb,
                ffn_dim,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            let fd_q8_needed = seq_len * (ffn_dim.div_ceil(32) * 34);
            if bs.batch_q8.len() < fd_q8_needed {
                bs.batch_q8.resize(fd_q8_needed, 0u8);
            }
            let fd_q8row = ffn_dim / 32 * 34;
            for s in 0..seq_len {
                let off = s * ffn_dim;
                ops::silu_mul_q8(
                    &bs.gate[off..off + ffn_dim],
                    &bs.up[off..off + ffn_dim],
                    &mut bs.batch_q8[s * fd_q8row..(s + 1) * fd_q8row],
                    ffn_dim,
                );
            }
            self.matmul_batch(
                &format!("{}.ffn_down.weight", p),
                &bs.gate[..seq_len * ffn_dim],
                &mut bs.q_buf[..seq_len * d],
                rb,
                d,
                ffn_dim,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            for i in 0..seq_len * d {
                bs.hidden[i] += bs.q_buf[i];
            }
        }

        for c in caches.iter_mut() {
            c.current_len = seq_len;
        }

        let last_off = (seq_len - 1) * d;
        bs.norm_tmp[..d].copy_from_slice(&bs.hidden[last_off..last_off + d]);
        let out_norm = self.get_1d("output_norm.weight", d)?;
        self.norm(
            &bs.norm_tmp[..d],
            out_norm,
            self.config.rms_norm_eps,
            &mut bs.residual[..d],
        );

        self.matmul(
            &self.lm_head_name,
            &bs.residual[..d],
            logits,
            rb,
            self.config.n_layers as usize,
            None,
        )?;

        Ok(())
    }

    pub fn encode_text(&self, tokens: &[u32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return Err("Empty token sequence".into());
        }
        let d = self.config.d_model as usize;
        let n_layers = self.config.n_layers as usize;
        let n_heads = self.config.n_heads as usize;
        let n_kv_heads = self.config.n_kv_heads as usize;
        let head_dim = self.config.head_dim as usize;
        let ffn_dim = self.config.d_ffn as usize;
        let nq = n_heads * head_dim;
        let nkv = n_kv_heads * head_dim;
        let max_seq = self.config.max_seq_len as usize;

        let mut caches: Vec<KVCache> = (0..n_layers)
            .map(|_| KVCache::new_with_storage(n_kv_heads, head_dim, max_seq, KVStorage::F32))
            .collect();

        let mut bs = BatchScratch {
            hidden: Vec::new(),
            residual: Vec::new(),
            attn_out: Vec::new(),
            q_buf: Vec::new(),
            k_buf: Vec::new(),
            v_buf: Vec::new(),
            q_heads: Vec::new(),
            k_heads: Vec::new(),
            v_heads: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            ffn_tmp: Vec::new(),
            norm_tmp: Vec::new(),
            attn_scores: Vec::new(),
            batch_q8: Vec::new(),
            head_norm_tmp: Vec::new(),
        };
        let mut rb = self.row_buf.lock().unwrap().clone();
        self.ensure_batch_size(&mut bs, seq_len, d, nq, nkv, ffn_dim, head_dim, max_seq);

        {
            let entry = self
                .tensors
                .get("token_embd.weight")
                .ok_or_else(|| "Missing token_embd.weight".to_string())?;
            let n = entry.n_cols().min(d);
            for (s, &tid) in tokens.iter().enumerate() {
                let token_id = tid as usize;
                let row = &mut bs.hidden[s * d..(s + 1) * d];
                row[..d].fill(0.0f32);
                if token_id < entry.n_rows() {
                    hearth_quant::dequantize(entry.dtype, entry.row_data(token_id), &mut row[..n])
                        .map_err(|e| format!("Embed dequant: {}", e))?;
                }
                if self.config.embed_scale {
                    let scale = (d as f32).sqrt();
                    for v in row[..d].iter_mut() {
                        *v *= scale;
                    }
                }
            }
        }

        for layer in 0..n_layers {
            let p = format!("blk.{}", layer);

            // --- ATTENTION NORM ---
            let attn_norm = self.get_1d(&format!("{}.attn_norm.weight", p), d)?;
            let eaq8_needed = seq_len * (d.div_ceil(32) * 34);
            if bs.batch_q8.len() < eaq8_needed {
                bs.batch_q8.resize(eaq8_needed, 0u8);
            }
            ops::rms_norm_batch(
                &bs.hidden[..seq_len * d],
                attn_norm,
                self.config.rms_norm_eps,
                &mut bs.residual[..seq_len * d],
                seq_len,
                d,
            );
            self.pool.par_quantize(
                seq_len,
                d,
                bs.residual.as_ptr() as usize,
                bs.batch_q8.as_mut_ptr() as usize,
            );
            self.matmul_batch(
                &format!("{}.attn_q.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.q_buf[..seq_len * nq],
                &mut rb,
                nq,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            self.matmul_batch(
                &format!("{}.attn_k.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.k_buf[..seq_len * nkv],
                &mut rb,
                nkv,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            self.matmul_batch(
                &format!("{}.attn_v.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.v_buf[..seq_len * nkv],
                &mut rb,
                nkv,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;

            bs.q_heads[..seq_len * nq].copy_from_slice(&bs.q_buf[..seq_len * nq]);
            bs.k_heads[..seq_len * nkv].copy_from_slice(&bs.k_buf[..seq_len * nkv]);
            bs.v_heads[..seq_len * nkv].copy_from_slice(&bs.v_buf[..seq_len * nkv]);

            if self
                .tensors
                .contains_key(&format!("{}.attn_q_norm.weight", p))
            {
                let q_norm = self.get_1d(&format!("{}.attn_q_norm.weight", p), head_dim)?;
                for s in 0..seq_len {
                    for h in 0..n_heads {
                        let off = s * nq + h * head_dim;
                        bs.head_norm_tmp[..head_dim]
                            .copy_from_slice(&bs.q_heads[off..off + head_dim]);
                        self.norm(
                            &bs.head_norm_tmp[..head_dim],
                            q_norm,
                            self.config.rms_norm_eps,
                            &mut bs.q_heads[off..off + head_dim],
                        );
                    }
                }
            }
            if self
                .tensors
                .contains_key(&format!("{}.attn_k_norm.weight", p))
            {
                let k_norm = self.get_1d(&format!("{}.attn_k_norm.weight", p), head_dim)?;
                for s in 0..seq_len {
                    for h in 0..n_kv_heads {
                        let off = s * nkv + h * head_dim;
                        bs.head_norm_tmp[..head_dim]
                            .copy_from_slice(&bs.k_heads[off..off + head_dim]);
                        self.norm(
                            &bs.head_norm_tmp[..head_dim],
                            k_norm,
                            self.config.rms_norm_eps,
                            &mut bs.k_heads[off..off + head_dim],
                        );
                    }
                }
            }

            let rope_dim = self.config.rope_dim as usize;
            for s in 0..seq_len {
                let pos = s;
                for h in 0..n_heads {
                    let off = s * nq + h * head_dim;
                    self.rope_cache
                        .apply(&mut bs.q_heads[off..off + rope_dim], pos);
                }
                for h in 0..n_kv_heads {
                    let off = s * nkv + h * head_dim;
                    self.rope_cache
                        .apply(&mut bs.k_heads[off..off + rope_dim], pos);
                }
            }

            for s in 0..seq_len {
                for h in 0..n_kv_heads {
                    let k_off = s * nkv + h * head_dim;
                    let v_off = s * nkv + h * head_dim;
                    caches[layer].write_kv(
                        s,
                        h,
                        &bs.k_heads[k_off..k_off + head_dim],
                        &bs.v_heads[v_off..v_off + head_dim],
                    );
                }
            }

            if caches[layer].is_q8_0() {
                ops::attention_batch_q8_0(
                    &bs.q_heads[..seq_len * nq],
                    &caches[layer].k_q8,
                    &caches[layer].v_q8,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq,
                    &mut bs.attn_out[..seq_len * nq],
                    &mut bs.attn_scores,
                );
            } else {
                ops::attention_batch(
                    &bs.q_heads[..seq_len * nq],
                    &caches[layer].k,
                    &caches[layer].v,
                    seq_len,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    max_seq,
                    &mut bs.attn_out[..seq_len * nq],
                    &mut bs.attn_scores,
                );
            }

            let eaq8 = nq.div_ceil(32) * 34;
            let eattn_q8_needed = seq_len * eaq8;
            if bs.batch_q8.len() < eattn_q8_needed {
                bs.batch_q8.resize(eattn_q8_needed, 0u8);
            }
            self.pool.par_quantize(
                seq_len,
                nq,
                bs.attn_out.as_ptr() as usize,
                bs.batch_q8.as_mut_ptr() as usize,
            );
            self.matmul_batch(
                &format!("{}.attn_output.weight", p),
                &bs.attn_out[..seq_len * nq],
                &mut bs.q_buf[..seq_len * d],
                &mut rb,
                d,
                nq,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            for i in 0..seq_len * d {
                bs.hidden[i] += bs.q_buf[i];
            }

            let ffn_norm = self.get_1d(&format!("{}.ffn_norm.weight", p), d)?;
            let efnq8_needed = seq_len * (d.div_ceil(32) * 34);
            if bs.batch_q8.len() < efnq8_needed {
                bs.batch_q8.resize(efnq8_needed, 0u8);
            }
            ops::rms_norm_batch(
                &bs.hidden[..seq_len * d],
                ffn_norm,
                self.config.rms_norm_eps,
                &mut bs.residual[..seq_len * d],
                seq_len,
                d,
            );
            self.pool.par_quantize(
                seq_len,
                d,
                bs.residual.as_ptr() as usize,
                bs.batch_q8.as_mut_ptr() as usize,
            );
            self.matmul_batch(
                &format!("{}.ffn_gate.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.gate[..seq_len * ffn_dim],
                &mut rb,
                ffn_dim,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            self.matmul_batch(
                &format!("{}.ffn_up.weight", p),
                &bs.residual[..seq_len * d],
                &mut bs.up[..seq_len * ffn_dim],
                &mut rb,
                ffn_dim,
                d,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;

            let efd_q8_needed = seq_len * (ffn_dim.div_ceil(32) * 34);
            if bs.batch_q8.len() < efd_q8_needed {
                bs.batch_q8.resize(efd_q8_needed, 0u8);
            }
            let efd_q8row = ffn_dim / 32 * 34;
            for s in 0..seq_len {
                let off = s * ffn_dim;
                ops::silu_mul_q8(
                    &bs.gate[off..off + ffn_dim],
                    &bs.up[off..off + ffn_dim],
                    &mut bs.batch_q8[s * efd_q8row..(s + 1) * efd_q8row],
                    ffn_dim,
                );
            }
            self.matmul_batch(
                &format!("{}.ffn_down.weight", p),
                &bs.gate[..seq_len * ffn_dim],
                &mut bs.q_buf[..seq_len * d],
                &mut rb,
                d,
                ffn_dim,
                seq_len,
                Some(&bs.batch_q8[..]),
            )?;
            for i in 0..seq_len * d {
                bs.hidden[i] += bs.q_buf[i];
            }
        }

        let out_norm = self.get_1d("output_norm.weight", d)?;
        let mut hidden_states = vec![0.0f32; seq_len * d];
        for s in 0..seq_len {
            let off = s * d;
            bs.norm_tmp[..d].copy_from_slice(&bs.hidden[off..off + d]);
            self.norm(
                &bs.norm_tmp[..d],
                out_norm,
                self.config.rms_norm_eps,
                &mut hidden_states[off..off + d],
            );
        }

        let pooled: Vec<f32> = (0..d)
            .map(|i| {
                let sum: f32 = (0..seq_len).map(|s| hidden_states[s * d + i]).sum();
                sum / seq_len as f32
            })
            .collect();

        Ok((hidden_states, pooled))
    }

    #[allow(dead_code)]
    fn prefill_batch(
        &self,
        prompt_ids: &[u32],
        caches: &mut [KVCache],
        logits: &mut [f32],
        cancel: &AtomicBool,
        bs: &mut BatchScratch,
        rb: &mut Vec<f32>,
    ) -> Result<(), String> {
        self.forward_batch(prompt_ids, caches, logits, cancel, bs, rb)
    }

    pub fn tokenizer(&self) -> &std::sync::Mutex<hearth_tokenizer::Tokenizer> {
        &self.tokenizer
    }

    pub fn template_kind(&self) -> hearth_tokenizer::TemplateKind {
        self.tokenizer.lock().unwrap().template_kind
    }

    pub fn is_gpu(&self) -> bool {
        self.gpu.is_some() && self.gpu_layers > 0
    }

    /// True if the GPU has dequant pipelines for all weight dtypes in this model.
    fn gpu_supports_model_dtypes(&self) -> bool {
        let gpu = match self.gpu.as_ref() {
            Some(g) => g,
            None => return false,
        };
        for entry in self.tensors.values() {
            if matches!(
                entry.dtype,
                hearth_gguf::GgmlDType::F32 | hearth_gguf::GgmlDType::F16
            ) {
                continue;
            }
            let kind = hearth_compute::QuantKind::from_gguf_dtype(entry.dtype);
            match kind {
                Some(k) => {
                    if !gpu.has_dequant(&k) {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }

    pub fn effective_strategy(&self) -> &LoadStrategy {
        &self.strategy
    }

    pub fn debug_template_info(&self) {
        let tok = self.tokenizer.lock().unwrap();
        let has_gguf_template = tok.chat_template.is_some();
        let using_gguf = has_gguf_template
            && tok.chat_template.as_ref().is_some_and(|t| {
                !t.contains("{% set")
                    && !t.contains("namespace")
                    && !t.contains("| ")
                    && !t.contains("is defined")
            });
        if using_gguf {
            eprintln!("Template source: GGUF chat_template (direct evaluation)");
        } else if has_gguf_template {
            eprintln!(
                "Template source: synthetic ({:?}) — GGUF template too complex",
                tok.template_kind
            );
        } else {
            eprintln!(
                "Template source: synthetic ({:?}) — no GGUF template",
                tok.template_kind
            );
        }
        eprintln!("Template kind: {:?}", tok.template_kind);
        eprintln!("GGUF template present: {}", has_gguf_template);
        eprintln!(
            "BOS ID: {} -> {:?}",
            tok.bos_id,
            tok.decode_token(tok.bos_id)
        );
        eprintln!(
            "EOS ID: {} -> {:?}",
            tok.eos_id,
            tok.decode_token(tok.eos_id)
        );
        eprintln!(
            "Special tokens: {:?}",
            tok.special_tokens.keys().collect::<Vec<_>>()
        );
    }

    pub fn debug_config(&self) -> crate::config::ModelConfig {
        self.config.clone()
    }

    fn forward_gpu(
        &self,
        token_ids: &[u32],
        pos: usize,
        _caches: &mut [KVCache],
        logits: &mut [f32],
        cancel: &AtomicBool,
        greedy: bool,
        rb: &mut Vec<f32>,
        repeat_pen: f32,
        past_tokens: &[u32],
    ) -> Result<(), String> {
        if cfg!(debug_assertions) {
            eprintln!(
                "[forward_gpu] executing GPU forward pass token={} pos={}",
                token_ids[0], pos
            );
        }
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let gpu = self.gpu.as_ref().ok_or("GPU not available")?;
        let d = self.config.d_model as usize;
        let n_heads = self.config.n_heads as usize;
        let n_kv_heads = self.config.n_kv_heads as usize;
        let head_dim = self.config.head_dim as usize;
        let n_layers = self.config.n_layers as usize;
        let ffn_dim = self.config.d_ffn as usize;
        let nq = n_heads * head_dim;
        let rope_dim = self.config.rope_dim as usize;
        let max_s = self.config.max_seq_len as usize;
        let rms_norm_eps = self.config.rms_norm_eps;
        let post_norm = self.config.post_norm;

        let create_buf = |n: usize, label: &str| gpu.create_storage_buffer((n as u64) * 4, label);

        // Token embed -> hidden_buf
        let token_id = token_ids[0] as usize;
        let embed_entry = self
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight".to_string())?;
        let mut hidden = self.hidden_gpu.lock().unwrap();
        hidden[..d].fill(0.0f32);
        if token_id < embed_entry.n_rows() {
            hearth_quant::dequantize(
                embed_entry.dtype,
                embed_entry.row_data(token_id),
                &mut hidden,
            )
            .map_err(|e| format!("Embed dequant: {}", e))?;
        }
        if self.config.embed_scale {
            let scale = (d as f32).sqrt();
            for h in hidden.iter_mut() {
                *h *= scale;
            }
        }
        let hidden_buf = gpu.upload_f32(&hidden, "hidden");

        let has_qk_norm = self.tensors.contains_key("blk.0.attn_q_norm.weight");

        // Pre-allocate scratch buffers reused across all layers (avoids per-layer create_buffer)
        let norm_tmp = create_buf(d, "norm_tmp");
        let ffn_norm_tmp = create_buf(d, "ffn_norm_tmp");
        let ffn_tmp_buf = create_buf(ffn_dim, "ffn_tmp");
        let out_norm_tmp = create_buf(d, "output_norm_tmp");
        // Pre-allocate QKV output buffers for fused_qkv (sizes constant across layers)
        let q_buf = create_buf(nq, "q_out");
        let k_buf = create_buf(n_kv_heads * head_dim, "k_out");
        let v_buf = create_buf(n_kv_heads * head_dim, "v_out");
        // Pre-allocate gate/up output buffers for fused_gate_up
        let gate_buf = create_buf(ffn_dim, "gate_out");
        let up_buf = create_buf(ffn_dim, "up_out");

        // Single batch for entire forward pass — avoids pipeline bubbles from partial submits
        gpu.begin_batch();
        let fwd_t0 = std::time::Instant::now();
        let mut sum_layer_us: u64 = 0;

        // Pre-compute YaRN constants (model-wide, same for all layers/positions)
        let rope_theta = self.config.rope_theta;
        let (yarn_corr_low, yarn_corr_high, yarn_mscale, yarn_n_dims) =
            match self.config.rope_scaling_type.as_deref() {
                Some("yarn") => {
                    let factor = self.config.rope_scaling_factor.unwrap_or(1.0);
                    let orig_ctx = self.config.original_ctx_len.unwrap_or(2048) as f32;
                    let n_dims_f = head_dim as f32;
                    let corr_dim = |beta: f32| {
                        n_dims_f * (orig_ctx / (beta * 2.0 * std::f32::consts::PI)).ln()
                            / (2.0 * rope_theta.ln())
                    };
                    let corr_low = (corr_dim(32.0)).floor().max(0.0);
                    let corr_high = (corr_dim(1.0)).ceil().min(n_dims_f - 1.0);
                    let mscale = 1.0 + 0.1 * factor.ln();
                    (corr_low, corr_high, mscale, n_dims_f)
                }
                _ => (0.0, 0.0, 1.0, 0.0), // n_dims=0 signals standard RoPE to shader
            };
        let rope_freq_scale = 1.0 / self.config.rope_scaling_factor.unwrap_or(1.0).max(1.0);

        // Pre-compute per-layer weight names (avoids format!() on every token)
        let wnames: Vec<[String; 7]> = (0..n_layers)
            .map(|i| {
                let p = format!("blk.{}", i);
                [
                    format!("{}.attn_q.weight", p),
                    format!("{}.attn_k.weight", p),
                    format!("{}.attn_v.weight", p),
                    format!("{}.attn_output.weight", p),
                    format!("{}.ffn_gate.weight", p),
                    format!("{}.ffn_up.weight", p),
                    format!("{}.ffn_down.weight", p),
                ]
            })
            .collect();

        // Pre-resolve norm buffer references (avoids HashMap lookup per layer per token)
        let mut anb = Vec::with_capacity(n_layers);
        let mut fnb = Vec::with_capacity(n_layers);
        let mut qnb = Vec::with_capacity(n_layers);
        let mut knb = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let p = format!("blk.{}", i);
            anb.push(gpu.norm_buffers[&format!("{}.attn_norm.weight", p)].clone());
            fnb.push(gpu.norm_buffers[&format!("{}.ffn_norm.weight", p)].clone());
            qnb.push(if has_qk_norm {
                gpu.norm_buffers
                    .get(&format!("{}.attn_q_norm.weight", p))
                    .cloned()
            } else {
                None
            });
            knb.push(if has_qk_norm {
                gpu.norm_buffers
                    .get(&format!("{}.attn_k_norm.weight", p))
                    .cloned()
            } else {
                None
            });
        }

        #[allow(clippy::needless_range_loop)]
        for layer in 0..n_layers {
            if cancel.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            let layer_t0 = std::time::Instant::now();

            // --- ATTENTION RMS NORM ---
            gpu.rms_norm(
                &hidden_buf,
                &anb[layer],
                &norm_tmp,
                d as u32,
                rms_norm_eps,
                post_norm,
            );

            // --- FUSED QKV MATMUL — one dispatch for all three (saves 2 dispatches per layer) ---
            {
                let pool = gpu.pool.lock().unwrap();
                let qw = pool
                    .get(&wnames[layer][0])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][0]))?;
                let kw = pool
                    .get(&wnames[layer][1])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][1]))?;
                let vw = pool
                    .get(&wnames[layer][2])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][2]))?;
                drop(pool);
                let qk_q = hearth_compute::QuantKind::from_gguf_dtype(
                    self.tensors[&wnames[layer][0]].dtype,
                )
                .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][0]));
                // Fallback: if non-Q1_0_G128 or fused_qkv unavailable, use separate matmuls
                if qk_q != Ok(hearth_compute::QuantKind::Q1_0G128)
                    || !gpu.fused_qkv(
                        &qw,
                        &kw,
                        &vw,
                        &norm_tmp,
                        &q_buf,
                        &k_buf,
                        &v_buf,
                        nq as u32,
                        (n_kv_heads * head_dim) as u32,
                        d as u32,
                    )
                {
                    let wr = |i| self.tensors[&wnames[layer][i]].n_rows();
                    let qb = gpu
                        .dequant_matmul_fused(&qw, &qk_q?, &norm_tmp, wr(0) as u32, 1, d as u32)
                        .ok_or_else(|| format!("Failed matmul {}", &wnames[layer][0]))?;
                    let qk_k = hearth_compute::QuantKind::from_gguf_dtype(
                        self.tensors[&wnames[layer][1]].dtype,
                    )
                    .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][1]))?;
                    let kb = gpu
                        .dequant_matmul_fused(&kw, &qk_k, &norm_tmp, wr(1) as u32, 1, d as u32)
                        .ok_or_else(|| format!("Failed matmul {}", &wnames[layer][1]))?;
                    let qk_v = hearth_compute::QuantKind::from_gguf_dtype(
                        self.tensors[&wnames[layer][2]].dtype,
                    )
                    .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][2]))?;
                    let vb = gpu
                        .dequant_matmul_fused(&vw, &qk_v, &norm_tmp, wr(2) as u32, 1, d as u32)
                        .ok_or_else(|| format!("Failed matmul {}", &wnames[layer][2]))?;
                    // Copy fallback outputs into pre-allocated buffers
                    gpu.copy_buffer(&qb, &q_buf);
                    gpu.copy_buffer(&kb, &k_buf);
                    gpu.copy_buffer(&vb, &v_buf);
                }
            }

            // --- Q/K HEAD NORMS ---
            if let Some(ref qnb_ref) = qnb[layer] {
                gpu.head_rms_norm(
                    &q_buf,
                    qnb_ref,
                    n_heads as u32,
                    head_dim as u32,
                    rms_norm_eps,
                    post_norm,
                );
            }
            if let Some(ref knb_ref) = knb[layer] {
                gpu.head_rms_norm(
                    &k_buf,
                    knb_ref,
                    n_kv_heads as u32,
                    head_dim as u32,
                    rms_norm_eps,
                    post_norm,
                );
            }

            // --- RoPE ---
            gpu.rope_combined(
                &q_buf,
                n_heads as u32,
                head_dim as u32,
                rope_dim as u32,
                pos as u32,
                rope_theta,
                rope_freq_scale,
                yarn_corr_low,
                yarn_corr_high,
                yarn_mscale,
                yarn_n_dims,
            );
            gpu.rope_combined(
                &k_buf,
                n_kv_heads as u32,
                head_dim as u32,
                rope_dim as u32,
                pos as u32,
                rope_theta,
                rope_freq_scale,
                yarn_corr_low,
                yarn_corr_high,
                yarn_mscale,
                yarn_n_dims,
            );

            // KV cache write
            let cache_max_seq = gpu.cache_max_seq as usize;
            let cache_stride = cache_max_seq.min(max_s);
            let _seq_len = pos + 1;
            let layer_buf = &gpu.kv_cache[layer];
            let per_layer_kv = n_kv_heads * cache_stride * head_dim;
            let v_offset = (per_layer_kv as u64) * 4;
            gpu.write_cache_kv(
                &k_buf,
                &v_buf,
                layer_buf,
                pos as u32,
                n_kv_heads as u32,
                head_dim as u32,
                cache_stride as u32,
            );

            // Flash attention
            let attn_out_buf = gpu
                .flash_attention(
                    &q_buf,
                    layer_buf,
                    0,
                    v_offset,
                    n_heads as u32,
                    n_kv_heads as u32,
                    head_dim as u32,
                    1,                   // seq_len_q = 1 for single-token decode
                    cache_stride as u32, // seq_len_kv = cache stride for head indexing
                    pos as u32,
                )
                .ok_or_else(|| format!("flash_attention failed layer {}", layer))?;

            // Output projection + residual (attn_out, gate, up, down — second pool lock)
            {
                let pool = gpu.pool.lock().unwrap();
                let aw = pool
                    .get(&wnames[layer][3])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][3]))?;
                let gw = pool
                    .get(&wnames[layer][4])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][4]))?;
                let uw = pool
                    .get(&wnames[layer][5])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][5]))?;
                let dw = pool
                    .get(&wnames[layer][6])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][6]))?;
                drop(pool);
                let wr = |i| self.tensors[&wnames[layer][i]].n_rows();
                let qk = |i| {
                    hearth_compute::QuantKind::from_gguf_dtype(
                        self.tensors[&wnames[layer][i]].dtype,
                    )
                    .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][i]))
                };

                // attn_out projection (fused matmul+add avoids intermediate buffer)
                if !gpu.dequant_matmul_fused_add_inplace(
                    &aw,
                    &qk(3)?,
                    &attn_out_buf,
                    &hidden_buf,
                    wr(3) as u32,
                    nq as u32,
                ) {
                    let aob = gpu
                        .dequant_matmul_fused(
                            &aw,
                            &qk(3)?,
                            &attn_out_buf,
                            wr(3) as u32,
                            1,
                            nq as u32,
                        )
                        .ok_or_else(|| format!("Failed matmul {}", &wnames[layer][3]))?;
                    gpu.add_inplace(&hidden_buf, &aob, d as u32);
                }

                // FFN norm
                gpu.rms_norm(
                    &hidden_buf,
                    &fnb[layer],
                    &ffn_norm_tmp,
                    d as u32,
                    rms_norm_eps,
                    post_norm,
                );

                // FFN gate + up (fused dispatch saves 1 dispatch per layer)
                let qk_gu = hearth_compute::QuantKind::from_gguf_dtype(
                    self.tensors[&wnames[layer][4]].dtype,
                )
                .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][4]));
                if qk_gu != Ok(hearth_compute::QuantKind::Q1_0G128)
                    || !gpu.fused_gate_up(
                        &gw,
                        &uw,
                        &ffn_norm_tmp,
                        &gate_buf,
                        &up_buf,
                        ffn_dim as u32,
                        d as u32,
                    )
                {
                    let gb = gpu
                        .dequant_matmul_fused(
                            &gw,
                            &qk_gu?,
                            &ffn_norm_tmp,
                            wr(4) as u32,
                            1,
                            d as u32,
                        )
                        .ok_or_else(|| format!("Failed matmul {}", &wnames[layer][4]))?;
                    let qk_u = hearth_compute::QuantKind::from_gguf_dtype(
                        self.tensors[&wnames[layer][5]].dtype,
                    )
                    .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][5]))?;
                    let ub = gpu
                        .dequant_matmul_fused(&uw, &qk_u, &ffn_norm_tmp, wr(5) as u32, 1, d as u32)
                        .ok_or_else(|| format!("Failed matmul {}", &wnames[layer][5]))?;
                    gpu.copy_buffer(&gb, &gate_buf);
                    gpu.copy_buffer(&ub, &up_buf);
                }

                gpu.silu_mul(&gate_buf, &up_buf, &ffn_tmp_buf, ffn_dim as u32);

                // FFN down + residual (fused matmul+add avoids intermediate buffer)
                if !gpu.dequant_matmul_fused_add_inplace(
                    &dw,
                    &qk(6)?,
                    &ffn_tmp_buf,
                    &hidden_buf,
                    wr(6) as u32,
                    ffn_dim as u32,
                ) {
                    let db = gpu
                        .dequant_matmul_fused(
                            &dw,
                            &qk(6)?,
                            &ffn_tmp_buf,
                            wr(6) as u32,
                            1,
                            ffn_dim as u32,
                        )
                        .ok_or_else(|| format!("Failed matmul {}", &wnames[layer][6]))?;
                    gpu.add_inplace(&hidden_buf, &db, d as u32);
                }
            }

            let layer_us = layer_t0.elapsed().as_micros();
            sum_layer_us += layer_us as u64;
        }
        // --- FINAL NORM + LM HEAD ---
        let out_norm_buf = &gpu.norm_buffers["output_norm.weight"];
        gpu.rms_norm(
            &hidden_buf,
            out_norm_buf,
            &out_norm_tmp,
            d as u32,
            rms_norm_eps,
            post_norm,
        );

        let vocab_size = self.config.vocab_size as usize;

        // Always compute lm_head on CPU: read back hidden state (8KB) after the final norm,
        // then compute the full 151669×2048 matmul on CPU using parallel Q8_0 dot product.
        // This eliminates the 196K-workgroup GPU lm_head dispatch and 606KB readback,
        // saving ~5ms per token. The CPU lm_head takes ~3ms via parallel Q8_0 kernels.
        gpu.copy_to_readback_staging(&out_norm_tmp, (d as u64) * 4);
        gpu.end_batch();
        let hidden = gpu.finish_readback_f32(d);

        // CPU lm_head (forces CPU path by passing layer > gpu_layers)
        self.matmul(
            &self.lm_head_name,
            &hidden,
            logits,
            rb,
            n_layers + 1, // > gpu_layers → CPU path
            None,         // no pre-quantized activation
        )?;

        if greedy {
            // Apply repetition penalty before argmax (matches sampler behavior)
            if (repeat_pen - 1.0).abs() > 1e-6 {
                for &tid in past_tokens {
                    let idx = tid as usize;
                    if idx < vocab_size {
                        if logits[idx] > 0.0 {
                            logits[idx] /= repeat_pen;
                        } else {
                            logits[idx] *= repeat_pen;
                        }
                    }
                }
            }

            // Find argmax from CPU-computed logits and store in logits[0]
            let argmax = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .unwrap_or(0);
            logits[0] = argmax as f32;
        }
        // For non-greedy: logits already contain the full CPU-computed results,
        // and the caller will apply temperature/sampling via the standard sampler.

        let fwd_total_us = fwd_t0.elapsed().as_micros();
        eprintln!(
            "[fwd] total={}us  layers={}us  read={}us",
            fwd_total_us,
            sum_layer_us,
            fwd_total_us - sum_layer_us as u128,
        );
        Ok(())
    }

    /// GPU batch forward pass for prefill (processes seq_len > 1 tokens in one batch).
    /// Uploads all token embeddings at once, runs all layers with batch operations,
    /// and reads back logits for the last token only.
    fn forward_gpu_batch(
        &self,
        prompt_ids: &[u32],
        _caches: &mut [KVCache],
        logits: &mut [f32],
        cancel: &AtomicBool,
    ) -> Result<(), String> {
        if cfg!(debug_assertions) {
            eprintln!(
                "[forward_gpu_batch] GPU batch forward pass seq_len={}",
                prompt_ids.len()
            );
        }
        if cancel.load(Ordering::Relaxed) {
            return Err("Cancelled".into());
        }
        let gpu = self.gpu.as_ref().ok_or("GPU not available")?;
        let seq_len = prompt_ids.len() as u32;
        let d = self.config.d_model;
        let n_heads = self.config.n_heads;
        let n_kv_heads = self.config.n_kv_heads;
        let head_dim = self.config.head_dim;
        let n_layers = self.config.n_layers;
        let ffn_dim = self.config.d_ffn;
        let nq = n_heads * head_dim;
        let nkv = n_kv_heads * head_dim;
        let rope_dim = self.config.rope_dim;
        let max_s = self.config.max_seq_len;
        let rms_norm_eps = self.config.rms_norm_eps;
        let post_norm = self.config.post_norm;

        let create_buf = |n: u64, label: &str| gpu.create_storage_buffer(n, label);

        // Embed all tokens on CPU, upload as [seq_len × d] f32
        let needed = (seq_len as usize) * (d as usize);
        let mut hidden_cpu = self.hidden_gpu.lock().unwrap();
        if hidden_cpu.len() < needed {
            hidden_cpu.resize(needed, 0.0f32);
        }
        let embed_entry = self
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight".to_string())?;
        for (s, &tid) in prompt_ids.iter().enumerate() {
            let token_id = tid as usize;
            let row = &mut hidden_cpu[s * d as usize..(s + 1) * d as usize];
            row.fill(0.0f32);
            if token_id < embed_entry.n_rows() {
                hearth_quant::dequantize(embed_entry.dtype, embed_entry.row_data(token_id), row)
                    .map_err(|e| format!("Embed dequant: {}", e))?;
            }
            if self.config.embed_scale {
                let scale = (d as f32).sqrt();
                for v in row.iter_mut() {
                    *v *= scale;
                }
            }
        }
        let hidden_buf = gpu.upload_f32(&hidden_cpu[..needed], "hidden_batch");
        drop(hidden_cpu);

        let has_qk_norm = self.tensors.contains_key("blk.0.attn_q_norm.weight");

        // Pre-allocate scratch GPU buffers
        let norm_tmp = create_buf((needed as u64) * 4, "norm_tmp_batch");
        let ffn_norm_tmp = create_buf((needed as u64) * 4, "ffn_norm_tmp_batch");
        let q_buf = create_buf((seq_len * nq) as u64 * 4, "q_buf_batch");
        let k_buf = create_buf((seq_len * nkv) as u64 * 4, "k_buf_batch");
        let v_buf = create_buf((seq_len * nkv) as u64 * 4, "v_buf_batch");
        let attn_out_buf = create_buf((seq_len * nq) as u64 * 4, "attn_out_batch");
        let gate_buf = create_buf((seq_len * ffn_dim) as u64 * 4, "gate_buf_batch");
        let up_buf = create_buf((seq_len * ffn_dim) as u64 * 4, "up_buf_batch");
        let ffn_tmp_buf = create_buf((seq_len * ffn_dim) as u64 * 4, "ffn_tmp_batch");
        let out_norm_tmp = create_buf(d as u64 * 4, "out_norm_tmp");

        // Pre-compute per-layer weight names
        let wnames: Vec<[String; 7]> = (0..n_layers as usize)
            .map(|i| {
                let p = format!("blk.{}", i);
                [
                    format!("{}.attn_q.weight", p),
                    format!("{}.attn_k.weight", p),
                    format!("{}.attn_v.weight", p),
                    format!("{}.attn_output.weight", p),
                    format!("{}.ffn_gate.weight", p),
                    format!("{}.ffn_up.weight", p),
                    format!("{}.ffn_down.weight", p),
                ]
            })
            .collect();

        // Pre-resolve norm buffer references
        let mut anb = Vec::with_capacity(n_layers as usize);
        let mut fnb = Vec::with_capacity(n_layers as usize);
        let mut qnb = Vec::with_capacity(n_layers as usize);
        let mut knb = Vec::with_capacity(n_layers as usize);
        for i in 0..n_layers as usize {
            let p = format!("blk.{}", i);
            anb.push(gpu.norm_buffers[&format!("{}.attn_norm.weight", p)].clone());
            fnb.push(gpu.norm_buffers[&format!("{}.ffn_norm.weight", p)].clone());
            qnb.push(if has_qk_norm {
                gpu.norm_buffers
                    .get(&format!("{}.attn_q_norm.weight", p))
                    .cloned()
            } else {
                None
            });
            knb.push(if has_qk_norm {
                gpu.norm_buffers
                    .get(&format!("{}.attn_k_norm.weight", p))
                    .cloned()
            } else {
                None
            });
        }

        // Pre-compute RoPE constants (same as forward_gpu)
        let rope_theta = self.config.rope_theta;
        let (yarn_corr_low, yarn_corr_high, yarn_mscale, yarn_n_dims) =
            match self.config.rope_scaling_type.as_deref() {
                Some("yarn") => {
                    let factor = self.config.rope_scaling_factor.unwrap_or(1.0);
                    let orig_ctx = self.config.original_ctx_len.unwrap_or(2048) as f32;
                    let n_dims_f = head_dim as f32;
                    let corr_dim = |beta: f32| {
                        n_dims_f * (orig_ctx / (beta * 2.0 * std::f32::consts::PI)).ln()
                            / (2.0 * rope_theta.ln())
                    };
                    let corr_low = (corr_dim(32.0)).floor().max(0.0);
                    let corr_high = (corr_dim(1.0)).ceil().min(n_dims_f - 1.0);
                    let mscale = 1.0 + 0.1 * factor.ln();
                    (corr_low, corr_high, mscale, n_dims_f)
                }
                _ => (0.0, 0.0, 1.0, 0.0),
            };
        let rope_freq_scale = 1.0 / self.config.rope_scaling_factor.unwrap_or(1.0).max(1.0);
        let cache_stride = gpu.cache_max_seq.min(max_s);

        gpu.begin_batch();

        let fwd_t0 = std::time::Instant::now();
        let mut sum_layer_us: u64 = 0;

        for layer in 0..n_layers as usize {
            if cancel.load(Ordering::Relaxed) {
                return Err("Cancelled".into());
            }
            let layer_t0 = std::time::Instant::now();

            // --- ATTENTION RMS NORM (batch) ---
            gpu.rms_norm_batch(
                &hidden_buf,
                &anb[layer],
                &norm_tmp,
                seq_len,
                d,
                rms_norm_eps,
                post_norm,
            );

            // --- BATCH QKV ---
            {
                let pool = gpu.pool.lock().unwrap();
                let qw = pool
                    .get(&wnames[layer][0])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][0]))?;
                let kw = pool
                    .get(&wnames[layer][1])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][1]))?;
                let vw = pool
                    .get(&wnames[layer][2])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][2]))?;
                drop(pool);

                let qk = |i| {
                    hearth_compute::QuantKind::from_gguf_dtype(
                        self.tensors[&wnames[layer][i]].dtype,
                    )
                    .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][i]))
                };

                // Fused QKV batch: one dispatch replaces 3 matmuls + 3 copies
                let qk_q = qk(0)?;
                if qk_q == hearth_compute::QuantKind::Q1_0G128
                    && gpu.fused_qkv_batch(
                        &qw, &kw, &vw, &norm_tmp, &q_buf, &k_buf, &v_buf, nq, nkv, d, seq_len,
                    )
                {
                    // Fused path succeeded — Q, K, V already written to buffers
                } else {
                    // Fallback: three separate batch matmuls
                    let qk_k = qk(1)?;
                    let qk_v = qk(2)?;

                    let q_out = gpu
                        .dequant_matmul_batch(&qw, &qk_q, &norm_tmp, nq, d, seq_len)
                        .ok_or_else(|| format!("Batch Q matmul failed layer {}", layer))?;
                    gpu.copy_buffer(&q_out, &q_buf);

                    let k_out = gpu
                        .dequant_matmul_batch(&kw, &qk_k, &norm_tmp, nkv, d, seq_len)
                        .ok_or_else(|| format!("Batch K matmul failed layer {}", layer))?;
                    gpu.copy_buffer(&k_out, &k_buf);

                    let v_out = gpu
                        .dequant_matmul_batch(&vw, &qk_v, &norm_tmp, nkv, d, seq_len)
                        .ok_or_else(|| format!("Batch V matmul failed layer {}", layer))?;
                    gpu.copy_buffer(&v_out, &v_buf);
                }
            }

            // --- Q/K HEAD NORMS (batch) ---
            if let Some(ref qnb_ref) = qnb[layer] {
                gpu.head_rms_norm_batch(
                    &q_buf,
                    qnb_ref,
                    n_heads,
                    head_dim,
                    seq_len,
                    rms_norm_eps,
                    post_norm,
                );
            }
            if let Some(ref knb_ref) = knb[layer] {
                gpu.head_rms_norm_batch(
                    &k_buf,
                    knb_ref,
                    n_kv_heads,
                    head_dim,
                    seq_len,
                    rms_norm_eps,
                    post_norm,
                );
            }

            // --- BATCH RoPE on Q and K ---
            gpu.rope_batch(
                &q_buf,
                n_heads,
                head_dim,
                rope_dim,
                seq_len,
                rope_theta,
                rope_freq_scale,
                yarn_corr_low,
                yarn_corr_high,
                yarn_mscale,
                yarn_n_dims,
            );
            gpu.rope_batch(
                &k_buf,
                n_kv_heads,
                head_dim,
                rope_dim,
                seq_len,
                rope_theta,
                rope_freq_scale,
                yarn_corr_low,
                yarn_corr_high,
                yarn_mscale,
                yarn_n_dims,
            );

            // --- BATCH KV CACHE WRITE ---
            let layer_buf = &gpu.kv_cache[layer];
            gpu.write_cache_kv_batch(
                &k_buf,
                &v_buf,
                layer_buf,
                seq_len,
                n_kv_heads,
                head_dim,
                cache_stride,
            );

            // --- FLASH ATTENTION PREFILL (causal) ---
            let attn_out = gpu
                .flash_attention_prefill(
                    &q_buf,
                    layer_buf,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    seq_len,
                    cache_stride,
                )
                .ok_or_else(|| format!("flash_attention_prefill failed layer {}", layer))?;
            gpu.copy_buffer(&attn_out, &attn_out_buf);

            // --- OUTPUT PROJECTION + RESIDUAL (batch) ---
            {
                let pool = gpu.pool.lock().unwrap();
                let aw = pool
                    .get(&wnames[layer][3])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][3]))?;
                let gw = pool
                    .get(&wnames[layer][4])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][4]))?;
                let uw = pool
                    .get(&wnames[layer][5])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][5]))?;
                let dw = pool
                    .get(&wnames[layer][6])
                    .cloned()
                    .ok_or_else(|| format!("Missing {}", &wnames[layer][6]))?;
                drop(pool);

                let qk_a = hearth_compute::QuantKind::from_gguf_dtype(
                    self.tensors[&wnames[layer][3]].dtype,
                )
                .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][3]))?;

                // attn_out projection: batch matmul then add to hidden
                let a_out = gpu
                    .dequant_matmul_batch(&aw, &qk_a, &attn_out_buf, d, nq, seq_len)
                    .ok_or_else(|| format!("Batch attn_out matmul failed layer {}", layer))?;
                gpu.add_inplace(&hidden_buf, &a_out, seq_len * d);

                // FFN norm (batch)
                gpu.rms_norm_batch(
                    &hidden_buf,
                    &fnb[layer],
                    &ffn_norm_tmp,
                    seq_len,
                    d,
                    rms_norm_eps,
                    post_norm,
                );

                // FFN gate + up (batch fused — one dispatch replaces 2 matmuls + 2 copies)
                let qk_gu = hearth_compute::QuantKind::from_gguf_dtype(
                    self.tensors[&wnames[layer][4]].dtype,
                )
                .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][4]))?;

                if qk_gu == hearth_compute::QuantKind::Q1_0G128
                    && gpu.fused_gate_up_batch(
                        &gw,
                        &uw,
                        &ffn_norm_tmp,
                        &gate_buf,
                        &up_buf,
                        ffn_dim,
                        d,
                        seq_len,
                    )
                {
                    // Fused path succeeded
                } else {
                    // Fallback: two separate batch matmuls
                    let qk_u = hearth_compute::QuantKind::from_gguf_dtype(
                        self.tensors[&wnames[layer][5]].dtype,
                    )
                    .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][5]))?;

                    let g_out = gpu
                        .dequant_matmul_batch(&gw, &qk_gu, &ffn_norm_tmp, ffn_dim, d, seq_len)
                        .ok_or_else(|| format!("Batch gate matmul failed layer {}", layer))?;
                    gpu.copy_buffer(&g_out, &gate_buf);

                    let u_out = gpu
                        .dequant_matmul_batch(&uw, &qk_u, &ffn_norm_tmp, ffn_dim, d, seq_len)
                        .ok_or_else(|| format!("Batch up matmul failed layer {}", layer))?;
                    gpu.copy_buffer(&u_out, &up_buf);
                }

                // SiLU(gate) * up (element-wise, more elements = seq_len * ffn_dim)
                gpu.silu_mul(&gate_buf, &up_buf, &ffn_tmp_buf, seq_len * ffn_dim);

                // FFN down + residual (batch)
                let qk_d = hearth_compute::QuantKind::from_gguf_dtype(
                    self.tensors[&wnames[layer][6]].dtype,
                )
                .ok_or_else(|| format!("Unsupported dtype for {}", &wnames[layer][6]))?;

                let d_out = gpu
                    .dequant_matmul_batch(&dw, &qk_d, &ffn_tmp_buf, d, ffn_dim, seq_len)
                    .ok_or_else(|| format!("Batch down matmul failed layer {}", layer))?;
                gpu.add_inplace(&hidden_buf, &d_out, seq_len * d);
            }

            let layer_us = layer_t0.elapsed().as_micros();
            sum_layer_us += layer_us as u64;
        }

        // --- OUTPUT NORM + LM HEAD (only for last position in batch) ---
        // Copy last position's hidden to single-vec buffer for output norm + lm_head
        let last_pos = seq_len - 1;
        let hidden_last = gpu.create_storage_buffer(d as u64 * 4, "hidden_last");

        // copy_buffer_range: copy the last row from hidden_buf
        let last_offset = (last_pos as u64) * (d as u64) * 4;
        gpu.copy_buffer_range(&hidden_buf, last_offset, &hidden_last, 0, d as u64 * 4);

        let out_norm_buf = &gpu.norm_buffers["output_norm.weight"];
        gpu.rms_norm(
            &hidden_last,
            out_norm_buf,
            &out_norm_tmp,
            d,
            rms_norm_eps,
            post_norm,
        );

        let vocab_size = self.config.vocab_size as usize;
        let lm_head_w = {
            let pool = gpu.pool.lock().unwrap();
            pool.get(&self.lm_head_name)
                .cloned()
                .ok_or_else(|| format!("Missing {}", self.lm_head_name))?
        };
        let lm_head_buf = gpu
            .dequant_matmul_fused(
                &lm_head_w,
                &hearth_compute::QuantKind::from_gguf_dtype(self.tensors[&self.lm_head_name].dtype)
                    .ok_or_else(|| format!("Unsupported dtype for {}", self.lm_head_name))?,
                &out_norm_tmp,
                vocab_size as u32,
                1,
                d,
            )
            .ok_or_else(|| "Failed lm_head matmul".to_string())?;

        gpu.end_batch();

        let logits_vec = gpu.readback_f32(&lm_head_buf, vocab_size);
        logits[..vocab_size].copy_from_slice(&logits_vec[..vocab_size]);

        let fwd_total_us = fwd_t0.elapsed().as_micros();
        eprintln!(
            "[fwd_batch] seq_len={} total={}us  layers={}us",
            seq_len, fwd_total_us, sum_layer_us,
        );
        Ok(())
    }

    fn get_1d(&self, name: &str, _len: usize) -> Result<&[f32], String> {
        self.norm_cache
            .get(name)
            .map(|v| v.as_slice())
            .ok_or_else(|| format!("Norm weight not in cache: {}", name))
    }

    /// Reusable Q8_0 quantization buffer for matmuls.
    /// Large models (d_model >= 2560) benefit from reusing a scratch buffer to avoid
    /// per-call Vec allocation/deallocation. Small models use the original `None` path
    /// (fresh alloc per call) since allocation cost is negligible at that scale and
    /// buffer reuse can cause cache interference in fast matmuls.
    ///
    /// Returns `Some(&[u8])` with the quantized activation when buffer reuse is beneficial,
    /// or `None` to let the matmul allocate its own temp buffer.
    ///
    /// Caller must ensure `scratch` is not aliased — the returned slice borrows `scratch`.
    fn q8_scratch<'a>(&self, x: &[f32], scratch: &'a mut Vec<u8>) -> Option<&'a [u8]> {
        if self.config.d_model >= 2560 {
            scratch.clear();
            hearth_quant::quantize_q8_0(x, scratch);
            Some(&scratch[..])
        } else {
            None
        }
    }

    fn norm(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
        if self.config.post_norm {
            ops::rms_norm_gemma(x, weight, eps, out);
        } else {
            ops::rms_norm(x, weight, eps, out);
        }
    }

    #[allow(dead_code)]
    fn norm_batch(
        &self,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
        seq_len: usize,
        dim: usize,
    ) {
        if self.config.post_norm {
            ops::rms_norm_gemma_batch(x, weight, eps, out, seq_len, dim);
        } else {
            ops::rms_norm_batch(x, weight, eps, out, seq_len, dim);
        }
    }
}

impl Model for LlamaModel {
    fn run(&self, request: PipelineRequest, cancel: Arc<AtomicBool>, output: Sender<EngineOutput>) {
        // Hybrid-architecture models fully own their forward path.
        if let Some(core) = self.hybrid.as_ref() {
            return core.run(request, cancel, output);
        }
        let result = match &request {
            PipelineRequest::TextGen {
                prompt,
                sampler,
                max_new_tokens,
                max_context_len,
            } => self.generate_text(
                prompt,
                sampler,
                *max_new_tokens,
                *max_context_len,
                &cancel,
                &output,
            ),
            _ => {
                let _ = output.send(EngineOutput::Error(ModelError::RuntimeError(
                    "Unsupported request type for LlamaModel".to_string(),
                )));
                Ok(())
            }
        };

        if let Err(e) = result {
            let _ = output.send(EngineOutput::Error(ModelError::RuntimeError(e)));
        }
    }

    fn build_chat_prompt(
        &self,
        system: &str,
        history: &[(&str, &str)],
        user: &str,
        thinking: bool,
    ) -> String {
        // Hybrid-architecture models route through their own core.
        if let Some(core) = self.hybrid.as_ref() {
            return core.build_chat_prompt(system, history, user, thinking);
        }
        self.tokenizer
            .lock()
            .unwrap()
            .apply_template(system, history, user, thinking)
    }
}

impl LlamaModel {
    fn generate_text(
        &self,
        prompt: &str,
        sampler_config: &SamplerConfig,
        max_new_tokens: usize,
        max_context_len: Option<usize>,
        cancel: &Arc<AtomicBool>,
        output: &Sender<EngineOutput>,
    ) -> Result<(), String> {
        let start = std::time::Instant::now();
        let vocab_size = self.config.vocab_size as usize;

        // Warm up GPU to avoid DVFS ramp-up penalty on first tokens
        let use_gpu = self.gpu_layers >= self.config.n_layers as usize
            && self.gpu.is_some()
            && self.gpu_supports_model_dtypes();
        if use_gpu {
            self.gpu.as_ref().unwrap().warmup();
        }

        let mut input_ids = self.tokenizer.lock().unwrap().encode(prompt, false);
        if input_ids.is_empty() {
            return Err("Empty prompt after tokenization".to_string());
        }
        if let Some(max_ctx) = max_context_len {
            if input_ids.len() > max_ctx {
                input_ids.drain(..input_ids.len() - max_ctx);
            }
        }
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }

        let n_layers = self.config.n_layers as usize;
        let mut caches: Vec<KVCache> = (0..n_layers)
            .map(|_| {
                KVCache::new_with_storage(
                    self.config.n_kv_heads as usize,
                    self.config.head_dim as usize,
                    self.config.max_seq_len as usize,
                    KVStorage::F32,
                )
            })
            .collect();

        let mut logits = vec![0.0f32; vocab_size];
        let mut sample_buf = vec![0.0f32; vocab_size];
        let mut past_tokens: Vec<u32> = Vec::new();
        let prompt_tokens = input_ids.len();
        let mut total_tokens = 0;

        let n_layers_usize = n_layers;
        let use_gpu = self.gpu_layers >= n_layers_usize
            && self.gpu.is_some()
            && self.gpu_supports_model_dtypes();
        eprintln!(
            "[generate_text] use_gpu={} n_layers={} gpu_layers={} gpu_some={}",
            use_gpu,
            n_layers_usize,
            self.gpu_layers,
            self.gpu.is_some(),
        );

        let mut sc = self.scratch.lock().unwrap();
        let mut rb = self.row_buf.lock().unwrap();
        let mut bs = self.batch.lock().unwrap();

        // CPU warmup: run one forward pass to get clock frequency up before timed work.
        // Windows 11 frequency scaling can take ~16ms to ramp from base to boost.
        if !use_gpu {
            let warmup_t0 = std::time::Instant::now();
            // Use BOS token (0) or first prompt token as warmup, at pos=0
            let warmup_tok = input_ids[0];
            let warmup_logits = &mut vec![0.0f32; vocab_size];
            let mut warmup_caches: Vec<KVCache> = (0..n_layers)
                .map(|_| {
                    KVCache::new_with_storage(
                        self.config.n_kv_heads as usize,
                        self.config.head_dim as usize,
                        self.config.max_seq_len as usize,
                        KVStorage::F32,
                    )
                })
                .collect();
            self.forward(
                &[warmup_tok],
                0,
                &mut warmup_caches,
                warmup_logits,
                cancel,
                &mut sc,
                &mut rb,
            )?;
            let warmup_us = warmup_t0.elapsed().as_micros();
            eprintln!("[warmup] CPU clock ramp: {}ms", warmup_us / 1000);
        }

        if input_ids.len() > 1 && use_gpu {
            let t0 = std::time::Instant::now();
            self.forward_gpu_batch(&input_ids, &mut caches, &mut logits, cancel)?;
            let us = t0.elapsed().as_micros();
            eprintln!(
                "[prefill] GPU batch {} tokens in {}ms ({:.1}ms/tok)",
                input_ids.len(),
                us / 1000,
                us as f64 / input_ids.len() as f64 / 1000.0
            );
        } else if input_ids.len() > 1 && !use_gpu {
            let t0 = std::time::Instant::now();
            self.forward_batch(
                &input_ids,
                &mut caches,
                &mut logits,
                cancel,
                &mut bs,
                &mut rb,
            )?;
            let us = t0.elapsed().as_micros();
            eprintln!(
                "[prefill] {} tokens in {}ms ({:.1}ms/tok)",
                input_ids.len(),
                us / 1000,
                us as f64 / input_ids.len() as f64 / 1000.0
            );
        } else {
            for (i, &tid) in input_ids.iter().enumerate() {
                if cancel.load(Ordering::Relaxed) {
                    return Ok(());
                }
                if use_gpu {
                    self.forward_gpu(
                        &[tid],
                        i,
                        &mut caches,
                        &mut logits,
                        cancel,
                        sampler_config.temperature == 0.0,
                        &mut rb,
                        sampler_config.repeat_pen,
                        &past_tokens,
                    )?;
                } else {
                    self.forward(
                        &[tid],
                        i,
                        &mut caches,
                        &mut logits,
                        cancel,
                        &mut sc,
                        &mut rb,
                    )?;
                }
            }
        }
        // `total_tokens` counts generated tokens only; the prompt is reported
        // separately so a 1-token reply from a 30-token prompt is not
        // misreported as "31 tokens generated".
        past_tokens.push(input_ids[input_ids.len() - 1]);

        let mut generated = 0;
        let mut cpu_overhead_us: u128 = 0;
        let greedy = sampler_config.temperature == 0.0;
        while generated < max_new_tokens {
            if cancel.load(Ordering::Relaxed) {
                // Emit any multi-byte character still held by the streaming
                // decoder so a cancelled stream isn't left one byte short.
                let flush = self.tokenizer.lock().unwrap().flush_stream_decode();
                if !flush.is_empty() {
                    let _ = output.send(EngineOutput::TextToken(flush));
                }
                let elapsed = start.elapsed().as_millis();
                let _ = output.send(EngineOutput::Done(RunStats {
                    elapsed_ms: elapsed,
                    tokens: generated,
                    prompt_tokens,
                    tokens_per_s: 0.0,
                }));
                return Ok(());
            }

            let pos = input_ids.len() + generated;

            if pos >= self.config.max_seq_len as usize {
                for c in caches.iter_mut() {
                    c.truncate_left(64);
                }
            }

            let cpu_t0 = std::time::Instant::now();

            let token = if use_gpu && greedy {
                // GPU-side argmax: forward_gpu stores argmax index in logits[0]
                logits[0] as u32
            } else {
                sample_buf.copy_from_slice(&logits);
                hearth_sampler::sample(&mut sample_buf, sampler_config, &past_tokens)
            };

            let cpu_sampling_us = cpu_t0.elapsed().as_micros();

            let tok = self.tokenizer.lock().unwrap();
            let eos_id = tok.eos_id;
            if token == eos_id {
                eprintln!("[generate] token={} EOS, stopping", token);
                drop(tok);
                break;
            }

            if tok.is_control_token(token) {
                let raw = tok
                    .vocab
                    .get(token as usize)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                eprintln!(
                    "[generate] token={} raw={:?} is control, stopping",
                    token, raw
                );
                drop(tok);
                break;
            }

            let token_text = tok.decode_token(token);
            drop(tok);
            let cpu_decode_us = cpu_t0.elapsed().as_micros() - cpu_sampling_us;
            let _ = output.send(EngineOutput::TextToken(token_text.to_string()));

            past_tokens.push(token);
            generated += 1;

            if use_gpu {
                self.forward_gpu(
                    &[token],
                    pos,
                    &mut caches,
                    &mut logits,
                    cancel,
                    sampler_config.temperature == 0.0,
                    &mut rb,
                    sampler_config.repeat_pen,
                    &past_tokens,
                )?;
            } else {
                self.forward(
                    &[token],
                    pos,
                    &mut caches,
                    &mut logits,
                    cancel,
                    &mut sc,
                    &mut rb,
                )?;
            }
            total_tokens += 1;
            let cpu_total_us = cpu_t0.elapsed().as_micros();
            cpu_overhead_us += cpu_total_us;
            eprintln!(
                "[cpu] sampling={}us decode={}us total={}us",
                cpu_sampling_us as u64, cpu_decode_us as u64, cpu_total_us as u64,
            );
        }

        let elapsed = start.elapsed().as_millis();
        eprintln!(
            "[generate] avg_cpu_overhead={}us/token",
            (cpu_overhead_us / generated.max(1) as u128) as u64,
        );
        let tokens_per_s = if elapsed > 0 {
            total_tokens as f32 / (elapsed as f32 / 1000.0)
        } else {
            0.0
        };

        // Emit any multi-byte character still held by the streaming decoder
        // (e.g. the reply's final character was cut mid-sequence).
        let flush = self.tokenizer.lock().unwrap().flush_stream_decode();
        if !flush.is_empty() {
            let _ = output.send(EngineOutput::TextToken(flush));
        }

        let _ = output.send(EngineOutput::Done(RunStats {
            elapsed_ms: elapsed,
            tokens: total_tokens,
            prompt_tokens,
            tokens_per_s,
        }));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_decode_with_real_model() {
        let models_dir = std::env::var("USERPROFILE").unwrap_or_else(|_| ".".into());
        let model_path = format!(
            "{}\\AppData\\Roaming\\hearth\\models\\Bonsai-1.7B-Q1_0.gguf",
            models_dir
        );

        if !std::path::Path::new(&model_path).exists() {
            eprintln!("[test] Model not found — skipping");
            return;
        }

        eprintln!("[test] Loading model with GpuFull...");
        let gpu_model = match LlamaModel::load(&model_path, LoadStrategy::GpuFull) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[test] GPU load failed: {} — skipping", e);
                return;
            }
        };

        if gpu_model.gpu.is_none() {
            eprintln!("[test] GPU not detected — skipping");
            return;
        }

        eprintln!("[test] Loading model CPU-only for comparison...");
        let cpu_model = match LlamaModel::load(&model_path, LoadStrategy::CpuOnly) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[test] CPU load failed: {} — skipping", e);
                return;
            }
        };

        let n_layers = cpu_model.config.n_layers as usize;
        let vocab_size = cpu_model.config.vocab_size as usize;
        let n_kv_heads = cpu_model.config.n_kv_heads as usize;
        let head_dim = cpu_model.config.head_dim as usize;
        let max_seq = cpu_model.config.max_seq_len as usize;
        let d = cpu_model.config.d_model as usize;

        let cancel = std::sync::atomic::AtomicBool::new(false);

        // 1) Compare CPU vs GPU token embedding first
        eprintln!("[test] Checking token embeddings...");
        let embed_entry = cpu_model.tensors.get("token_embd.weight").unwrap();
        let mut cpu_embed = vec![0.0f32; d];
        hearth_quant::dequantize(embed_entry.dtype, embed_entry.row_data(0), &mut cpu_embed)
            .unwrap();

        // GPU embedding
        let gpu = gpu_model.gpu.as_ref().unwrap();
        let _embed_gpu_buf = gpu
            .pool
            .lock()
            .unwrap()
            .get("token_embd.weight")
            .cloned()
            .unwrap();
        drop(gpu);
        // Don't need to upload - the CPU path dequantizes directly

        // The CPU embedding is the correct reference. For GPU, we can't easily read back
        // the dequantized value. But the CPU embedding values are what we expect.
        eprintln!("[test] CPU embed[0..5]: {:?}", &cpu_embed[..5]);
        let embed_sum_sq: f32 = cpu_embed.iter().map(|x| x * x).sum();
        eprintln!("[test] CPU embed sum_sq: {:.4}", embed_sum_sq);

        // 2) CPU forward
        eprintln!("[test] Running CPU forward...");
        let mut cpu_caches: Vec<crate::kvcache::KVCache> = (0..n_layers)
            .map(|_| {
                crate::kvcache::KVCache::new_with_storage(
                    n_kv_heads,
                    head_dim,
                    max_seq,
                    crate::kvcache::KVStorage::F32,
                )
            })
            .collect();
        let mut logits_cpu = vec![0.0f32; vocab_size];
        let mut sc = cpu_model.scratch.lock().unwrap();
        let mut rb = cpu_model.row_buf.lock().unwrap();
        cpu_model
            .forward(
                &[0u32],
                0,
                &mut cpu_caches,
                &mut logits_cpu,
                &cancel,
                &mut sc,
                &mut rb,
            )
            .expect("CPU forward failed");

        // 3) GPU forward
        eprintln!("[test] Running GPU forward...");
        let mut gpu_caches: Vec<crate::kvcache::KVCache> = (0..n_layers)
            .map(|_| {
                crate::kvcache::KVCache::new_with_storage(
                    n_kv_heads,
                    head_dim,
                    max_seq,
                    crate::kvcache::KVStorage::F32,
                )
            })
            .collect();
        let mut logits_gpu = vec![0.0f32; vocab_size];
        let mut dummy_rb = Vec::new();
        let empty_tokens = Vec::new();
        gpu_model
            .forward_gpu(
                &[0u32],
                0,
                &mut gpu_caches,
                &mut logits_gpu,
                &cancel,
                false,
                &mut dummy_rb,
                1.0,
                &empty_tokens,
            )
            .expect("GPU forward failed");

        // Compare
        let mut cpu_sorted: Vec<(u32, f32)> = logits_cpu
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        cpu_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut gpu_sorted: Vec<(u32, f32)> = logits_gpu
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        gpu_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("[test] CPU Top-5: {:?}", &cpu_sorted[..5]);
        eprintln!("[test] GPU Top-5: {:?}", &gpu_sorted[..5]);

        let dot: f32 = logits_cpu
            .iter()
            .zip(logits_gpu.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_cpu: f32 = logits_cpu.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm_gpu: f32 = logits_gpu.iter().map(|a| a * a).sum::<f32>().sqrt();
        eprintln!(
            "[test] Cosine similarity: {:.6}",
            dot / (norm_cpu * norm_gpu + 1e-10)
        );

        if cpu_sorted[0].0 == gpu_sorted[0].0 {
            eprintln!("[test] ✓ Top-1 token matches!");
        } else {
            eprintln!(
                "[test] ✗ Top-1 MISMATCH: CPU={} GPU={}",
                cpu_sorted[0].0, gpu_sorted[0].0
            );
        }

        // 4) Debug: manually run first layer operations on GPU and compare
        eprintln!("[test] Layer-by-layer debug...");
        let _n_heads = cpu_model.config.n_heads as usize;
        let _ffn_dim = cpu_model.config.d_ffn as usize;

        // Manually run embedding on GPU and read back
        let mut hidden_cpu_input = vec![0.0f32; d];
        hearth_quant::dequantize(
            embed_entry.dtype,
            embed_entry.row_data(0),
            &mut hidden_cpu_input,
        )
        .unwrap();

        // Upload to GPU
        let gpu = gpu_model.gpu.as_ref().unwrap();
        let hidden_buf = gpu.upload_f32(&hidden_cpu_input, "hidden_debug");

        // Run RMS norm on GPU for layer 0
        let anb = gpu
            .norm_buffers
            .get("blk.0.attn_norm.weight")
            .unwrap()
            .clone();
        let norm_tmp = gpu.create_storage_buffer(d as u64 * 4, "norm_tmp_debug");
        gpu.rms_norm(
            &hidden_buf,
            &anb,
            &norm_tmp,
            d as u32,
            cpu_model.config.rms_norm_eps,
            false,
        );

        // Read back RMS norm output
        let norm_out = gpu.readback_f32(&norm_tmp, d);

        // Compute CPU RMS norm for comparison
        let attn_norm_weight = cpu_model.norm_cache.get("blk.0.attn_norm.weight").unwrap();
        let mut cpu_norm_out = vec![0.0f32; d];
        ops::rms_norm(
            &hidden_cpu_input,
            attn_norm_weight,
            cpu_model.config.rms_norm_eps,
            &mut cpu_norm_out,
        );

        // Compare RMS norm outputs
        eprintln!("[test] RMS norm comparison (first 10):");
        for i in 0..10 {
            eprintln!(
                "  [{}] cpu={:.6} gpu={:.6}",
                i, cpu_norm_out[i], norm_out[i]
            );
        }
        let rms_match = norm_out
            .iter()
            .zip(cpu_norm_out.iter())
            .all(|(a, b)| (a - b).abs() < 0.001);
        eprintln!("[test] RMS norm match: {}", rms_match);

        // 5) Check QKV matmul on GPU
        eprintln!("[test] Checking QKV matmul...");
        let qw = gpu_model.tensors.get("blk.0.attn_q.weight").unwrap();
        let kw = gpu_model.tensors.get("blk.0.attn_k.weight").unwrap();
        let vw = gpu_model.tensors.get("blk.0.attn_v.weight").unwrap();
        let wr_q = qw.n_rows();
        let wr_k = kw.n_rows();
        let wr_v = vw.n_rows();

        // GPU QKV
        gpu.begin_batch();
        let qk_q = hearth_compute::QuantKind::from_gguf_dtype(qw.dtype).unwrap();
        let qb = gpu
            .dequant_matmul_fused(
                &gpu_model.gpu.as_ref().unwrap().pool.lock().unwrap()["blk.0.attn_q.weight"],
                &qk_q,
                &norm_tmp,
                wr_q as u32,
                1,
                d as u32,
            )
            .unwrap();
        let qk_k = hearth_compute::QuantKind::from_gguf_dtype(kw.dtype).unwrap();
        let kb = gpu
            .dequant_matmul_fused(
                &gpu_model.gpu.as_ref().unwrap().pool.lock().unwrap()["blk.0.attn_k.weight"],
                &qk_k,
                &norm_tmp,
                wr_k as u32,
                1,
                d as u32,
            )
            .unwrap();
        let qk_v = hearth_compute::QuantKind::from_gguf_dtype(vw.dtype).unwrap();
        let vb = gpu
            .dequant_matmul_fused(
                &gpu_model.gpu.as_ref().unwrap().pool.lock().unwrap()["blk.0.attn_v.weight"],
                &qk_v,
                &norm_tmp,
                wr_v as u32,
                1,
                d as u32,
            )
            .unwrap();
        gpu.end_batch();

        let gpu_q = gpu.readback_f32(&qb, wr_q);
        let gpu_k = gpu.readback_f32(&kb, wr_k);
        let gpu_v = gpu.readback_f32(&vb, wr_v);

        // CPU QKV
        let mut cpu_q = vec![0.0f32; wr_q];
        let mut cpu_k = vec![0.0f32; wr_k];
        let mut cpu_v = vec![0.0f32; wr_v];
        {
            let entry = qw;
            let n_cols = entry.n_cols();
            let n_rows = entry.n_rows();
            for row in 0..n_rows {
                let w = entry.row_data(row);
                cpu_q[row] = hearth_quant::dot_q1_0g128_f32(w, &cpu_norm_out, n_cols);
            }
        }
        {
            let entry = kw;
            let n_cols = entry.n_cols();
            let n_rows = entry.n_rows();
            for row in 0..n_rows {
                let w = entry.row_data(row);
                cpu_k[row] = hearth_quant::dot_q1_0g128_f32(w, &cpu_norm_out, n_cols);
            }
        }
        {
            let entry = vw;
            let n_cols = entry.n_cols();
            let n_rows = entry.n_rows();
            for row in 0..n_rows {
                let w = entry.row_data(row);
                cpu_v[row] = hearth_quant::dot_q1_0g128_f32(w, &cpu_norm_out, n_cols);
            }
        }

        let q_match = gpu_q
            .iter()
            .zip(cpu_q.iter())
            .all(|(a, b)| (a - b).abs() < 0.1);
        eprintln!("[test] Q matmul match: {} (wr_q={})", q_match, wr_q);
        if !q_match {
            eprintln!(
                "[test] Q first 10: cpu={:?} gpu={:?}",
                &cpu_q[..10],
                &gpu_q[..10]
            );
        }
        let k_match = gpu_k
            .iter()
            .zip(cpu_k.iter())
            .all(|(a, b)| (a - b).abs() < 0.1);
        eprintln!("[test] K matmul match: {}", k_match);
        if !k_match {
            eprintln!(
                "[test] K first 10: cpu={:?} gpu={:?}",
                &cpu_k[..10],
                &gpu_k[..10]
            );
        }
        let v_match = gpu_v
            .iter()
            .zip(cpu_v.iter())
            .all(|(a, b)| (a - b).abs() < 0.1);
        eprintln!("[test] V matmul match: {}", v_match);
        if !v_match {
            eprintln!(
                "[test] V first 10: cpu={:?} gpu={:?}",
                &cpu_v[..10],
                &gpu_v[..10]
            );
        }

        // 6) Check head RMS norm + RoPE
        eprintln!("[test] Checking head_norm + RoPE...");
        let n_heads = cpu_model.config.n_heads as usize;
        let n_kv_heads = cpu_model.config.n_kv_heads as usize;
        let head_dim = cpu_model.config.head_dim as usize;
        let nq = n_heads * head_dim;
        let nkv = n_kv_heads * head_dim;
        let rope_dim = cpu_model.config.rope_dim as usize;

        // Upload Q/K to GPU buffers
        let gpu_q_buf = gpu.upload_f32(&gpu_q, "q_debug");
        let gpu_k_buf = gpu.upload_f32(&gpu_k, "k_debug");

        // Apply head RMS norm on GPU
        let q_norm_w = gpu
            .norm_buffers
            .get("blk.0.attn_q_norm.weight")
            .unwrap()
            .clone();
        let k_norm_w = gpu
            .norm_buffers
            .get("blk.0.attn_k_norm.weight")
            .unwrap()
            .clone();
        // head_rms_norm works in-place on the buffer
        gpu.head_rms_norm(
            &gpu_q_buf,
            &q_norm_w,
            n_heads as u32,
            head_dim as u32,
            cpu_model.config.rms_norm_eps,
            false,
        );
        gpu.head_rms_norm(
            &gpu_k_buf,
            &k_norm_w,
            n_kv_heads as u32,
            head_dim as u32,
            cpu_model.config.rms_norm_eps,
            false,
        );

        // Read back
        let gpu_q_hnorm = gpu.readback_f32(&gpu_q_buf, nq);
        let gpu_k_hnorm = gpu.readback_f32(&gpu_k_buf, nkv);

        // CPU head RMS norm
        let q_norm_cpu = cpu_model
            .norm_cache
            .get("blk.0.attn_q_norm.weight")
            .unwrap();
        let k_norm_cpu = cpu_model
            .norm_cache
            .get("blk.0.attn_k_norm.weight")
            .unwrap();
        let mut cpu_q_hnorm = cpu_q.clone();
        let mut cpu_k_hnorm = cpu_k.clone();
        for h in 0..n_heads {
            let s = h * head_dim;
            ops::rms_norm(
                &cpu_q[s..s + head_dim],
                q_norm_cpu,
                cpu_model.config.rms_norm_eps,
                &mut cpu_q_hnorm[s..s + head_dim],
            );
        }
        for h in 0..n_kv_heads {
            let s = h * head_dim;
            ops::rms_norm(
                &cpu_k[s..s + head_dim],
                k_norm_cpu,
                cpu_model.config.rms_norm_eps,
                &mut cpu_k_hnorm[s..s + head_dim],
            );
        }

        let qh_match = gpu_q_hnorm
            .iter()
            .zip(cpu_q_hnorm.iter())
            .all(|(a, b)| (a - b).abs() < 0.01);
        eprintln!("[test] Q head_norm match: {}", qh_match);
        if !qh_match {
            eprintln!(
                "[test] Q head_norm first 10: cpu={:?} gpu={:?}",
                &cpu_q_hnorm[..10],
                &gpu_q_hnorm[..10]
            );
        }
        let kh_match = gpu_k_hnorm
            .iter()
            .zip(cpu_k_hnorm.iter())
            .all(|(a, b)| (a - b).abs() < 0.01);
        eprintln!("[test] K head_norm match: {}", kh_match);
        if !kh_match {
            eprintln!(
                "[test] K head_norm first 10: cpu={:?} gpu={:?}",
                &cpu_k_hnorm[..10],
                &gpu_k_hnorm[..10]
            );
        }

        // 7) Check RoPE
        eprintln!("[test] Checking RoPE...");
        let rope_theta = cpu_model.config.rope_theta;
        let rope_freq_scale = 1.0 / cpu_model.config.rope_scaling_factor.unwrap_or(1.0).max(1.0);
        // We need to re-upload because head_norm modified the buffers in-place
        let gpu_q_rope = gpu.upload_f32(&gpu_q_hnorm, "q_rope_debug");
        let gpu_k_rope = gpu.upload_f32(&gpu_k_hnorm, "k_rope_debug");
        gpu.rope_combined(
            &gpu_q_rope,
            n_heads as u32,
            head_dim as u32,
            rope_dim as u32,
            0,
            rope_theta,
            rope_freq_scale,
            0.0,
            0.0,
            1.0,
            0.0,
        );
        gpu.rope_combined(
            &gpu_k_rope,
            n_kv_heads as u32,
            head_dim as u32,
            rope_dim as u32,
            0,
            rope_theta,
            rope_freq_scale,
            0.0,
            0.0,
            1.0,
            0.0,
        );
        let gpu_q_roped = gpu.readback_f32(&gpu_q_rope, nq);
        let gpu_k_roped = gpu.readback_f32(&gpu_k_rope, nkv);

        // CPU RoPE
        let mut cpu_q_roped = cpu_q_hnorm.clone();
        let mut cpu_k_roped = cpu_k_hnorm.clone();
        for h in 0..n_heads {
            let s = h * head_dim;
            cpu_model
                .rope_cache
                .apply(&mut cpu_q_roped[s..s + rope_dim], 0);
        }
        for h in 0..n_kv_heads {
            let s = h * head_dim;
            cpu_model
                .rope_cache
                .apply(&mut cpu_k_roped[s..s + rope_dim], 0);
        }

        let qr_match = gpu_q_roped
            .iter()
            .zip(cpu_q_roped.iter())
            .all(|(a, b)| (a - b).abs() < 0.01);
        eprintln!("[test] Q RoPE match: {}", qr_match);
        if !qr_match {
            eprintln!(
                "[test] Q RoPE first 10: cpu={:?} gpu={:?}",
                &cpu_q_roped[..10],
                &gpu_q_roped[..10]
            );
            // Also check specific head pairs to see if the rotation is correct
            for h in 0..n_heads.min(2) {
                let s = h * head_dim;
                eprintln!(
                    "[test] Q head {} first 15: cpu={:?} gpu={:?}",
                    h,
                    &cpu_q_roped[s..s + 15],
                    &gpu_q_roped[s..s + 15]
                );
            }
        }
        let kr_match = gpu_k_roped
            .iter()
            .zip(cpu_k_roped.iter())
            .all(|(a, b)| (a - b).abs() < 0.01);
        eprintln!("[test] K RoPE match: {}", kr_match);
        if !kr_match {
            eprintln!(
                "[test] K RoPE first 10: cpu={:?} gpu={:?}",
                &cpu_k_roped[..10],
                &gpu_k_roped[..10]
            );
            for h in 0..n_kv_heads.min(2) {
                let s = h * head_dim;
                eprintln!(
                    "[test] K head {} first 15: cpu={:?} gpu={:?}",
                    h,
                    &cpu_k_roped[s..s + 15],
                    &gpu_k_roped[s..s + 15]
                );
            }
        }
    }
}
