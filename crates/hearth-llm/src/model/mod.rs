mod gpu;
mod matmul;
mod moe;
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
    norm_cache: HashMap<String, Vec<f32>>,
    batch: std::sync::Mutex<BatchScratch>,
    layer_names: Vec<LayerTensorNames>,
    lm_head_name: String,
    pool: ThreadPool,
    rope_cache: ops::RopeCache,
}

impl LlamaModel {
    pub fn load(path: &str, strategy: LoadStrategy) -> Result<Self, String> {
        let gguf = GgufFile::open(path).map_err(|e| format!("GGUF: {}", e))?;
        let mut config = ModelConfig::from_gguf(&gguf)?;
        config.validate()?;

        let tokenizer = Tokenizer::from_gguf(&gguf).map_err(|e| format!("Tokenizer: {}", e))?;

        let mut tensors = HashMap::new();
        for info in &gguf.tensors {
            let data = gguf.tensor_data(info).to_vec();
            tensors.insert(
                info.name.clone(),
                TensorEntry {
                    data,
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                },
            );
        }

        let d = config.d_model as usize;
        let n_heads = config.n_heads as usize;
        let n_kv_heads = config.n_kv_heads as usize;
        let mut head_dim = config.head_dim as usize;
        let d_ffn = config.d_ffn as usize;
        let max_seq = config.max_seq_len as usize;

        if let Some(q_entry) = tensors.get("blk.0.attn_q.weight") {
            let q_rows = q_entry.n_rows();
            let computed_head_dim = q_rows / n_heads;
            if computed_head_dim * n_heads == q_rows && computed_head_dim != head_dim {
                head_dim = computed_head_dim;
            }
        }
        config.head_dim = head_dim as u32;

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
                    if hearth_quant::dequantize(entry.dtype, &entry.data, &mut buf).is_ok() {
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
            let per_layer = (n_kv_heads * cache_max_seq * head_dim) as u64 * 2; // K + V in f32
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
            norm_cache,
            batch: std::sync::Mutex::new(batch),
            layer_names,
            lm_head_name,
            pool,
            rope_cache,
        })
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
                    for kvh in 0..n_kv_heads {
                        let ks = caches[layer].k_slice_dequant(kvh, seq_len).to_vec();
                        let vs = caches[layer].v_slice_dequant(kvh, seq_len).to_vec();
                        for r in 0..kv_repeat {
                            let qh = kvh * kv_repeat + r;
                            let qs = qh * head_dim;
                            let os = qh * head_dim;
                            sc.norm_tmp[..head_dim].copy_from_slice(&sc.q_heads[qs..qs + head_dim]);
                            ops::attention(
                                &sc.norm_tmp[..head_dim],
                                &ks,
                                &vs,
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

            bs.norm_tmp[..seq_len * d].copy_from_slice(&bs.hidden[..seq_len * d]);
            let attn_norm = self.get_1d(&format!("{}.attn_norm.weight", p), d)?;
            self.norm_batch(
                &bs.norm_tmp[..seq_len * d],
                attn_norm,
                self.config.rms_norm_eps,
                &mut bs.residual[..seq_len * d],
                seq_len,
                d,
            );

            let q8_needed = seq_len * (d.div_ceil(32) * 34);
            if bs.batch_q8.len() < q8_needed {
                bs.batch_q8.resize(q8_needed, 0u8);
            }
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
                let kv_total = n_kv_heads * max_seq * head_dim;
                if bs.k_heads.len() < kv_total {
                    bs.k_heads.resize(kv_total, 0.0f32);
                }
                if bs.v_heads.len() < kv_total {
                    bs.v_heads.resize(kv_total, 0.0f32);
                }
                for h in 0..n_kv_heads {
                    let ks = caches[layer].k_slice_dequant(h, seq_len).to_vec();
                    let vs = caches[layer].v_slice_dequant(h, seq_len).to_vec();
                    let k_off = h * max_seq * head_dim;
                    let v_off = h * max_seq * head_dim;
                    bs.k_heads[k_off..k_off + seq_len * head_dim].copy_from_slice(&ks);
                    bs.v_heads[v_off..v_off + seq_len * head_dim].copy_from_slice(&vs);
                }
                ops::attention_batch(
                    &bs.q_heads[..seq_len * nq],
                    &bs.k_heads,
                    &bs.v_heads,
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

            bs.norm_tmp[..seq_len * d].copy_from_slice(&bs.hidden[..seq_len * d]);
            let ffn_norm = self.get_1d(&format!("{}.ffn_norm.weight", p), d)?;
            self.norm_batch(
                &bs.norm_tmp[..seq_len * d],
                ffn_norm,
                self.config.rms_norm_eps,
                &mut bs.residual[..seq_len * d],
                seq_len,
                d,
            );

            let rq8 = d.div_ceil(32) * 34;
            let gate_q8_needed = seq_len * rq8;
            if bs.batch_q8.len() < gate_q8_needed {
                bs.batch_q8.resize(gate_q8_needed, 0u8);
            }
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

            for s in 0..seq_len {
                let off = s * ffn_dim;
                ops::silu(&mut bs.gate[off..off + ffn_dim]);
                ops::mul_elem(
                    &bs.gate[off..off + ffn_dim],
                    &bs.up[off..off + ffn_dim],
                    &mut bs.ffn_tmp[off..off + ffn_dim],
                );
            }

            let fq8 = ffn_dim.div_ceil(32) * 34;
            let ffn_q8_needed = seq_len * fq8;
            if bs.batch_q8.len() < ffn_q8_needed {
                bs.batch_q8.resize(ffn_q8_needed, 0u8);
            }
            self.pool.par_quantize(
                seq_len,
                ffn_dim,
                bs.ffn_tmp.as_ptr() as usize,
                bs.batch_q8.as_mut_ptr() as usize,
            );
            self.matmul_batch(
                &format!("{}.ffn_down.weight", p),
                &bs.ffn_tmp[..seq_len * ffn_dim],
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

            bs.norm_tmp[..seq_len * d].copy_from_slice(&bs.hidden[..seq_len * d]);
            let attn_norm = self.get_1d(&format!("{}.attn_norm.weight", p), d)?;
            self.norm_batch(
                &bs.norm_tmp[..seq_len * d],
                attn_norm,
                self.config.rms_norm_eps,
                &mut bs.residual[..seq_len * d],
                seq_len,
                d,
            );

            let erq8 = d.div_ceil(32) * 34;
            let eqkv_needed = seq_len * erq8;
            if bs.batch_q8.len() < eqkv_needed {
                bs.batch_q8.resize(eqkv_needed, 0u8);
            }
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
                let kv_total = n_kv_heads * max_seq * head_dim;
                if bs.k_heads.len() < kv_total {
                    bs.k_heads.resize(kv_total, 0.0f32);
                }
                if bs.v_heads.len() < kv_total {
                    bs.v_heads.resize(kv_total, 0.0f32);
                }
                for h in 0..n_kv_heads {
                    let ks = caches[layer].k_slice_dequant(h, seq_len).to_vec();
                    let vs = caches[layer].v_slice_dequant(h, seq_len).to_vec();
                    let k_off = h * max_seq * head_dim;
                    let v_off = h * max_seq * head_dim;
                    bs.k_heads[k_off..k_off + seq_len * head_dim].copy_from_slice(&ks);
                    bs.v_heads[v_off..v_off + seq_len * head_dim].copy_from_slice(&vs);
                }
                ops::attention_batch(
                    &bs.q_heads[..seq_len * nq],
                    &bs.k_heads,
                    &bs.v_heads,
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

            bs.norm_tmp[..seq_len * d].copy_from_slice(&bs.hidden[..seq_len * d]);
            let ffn_norm = self.get_1d(&format!("{}.ffn_norm.weight", p), d)?;
            self.norm_batch(
                &bs.norm_tmp[..seq_len * d],
                ffn_norm,
                self.config.rms_norm_eps,
                &mut bs.residual[..seq_len * d],
                seq_len,
                d,
            );

            let erg8 = d.div_ceil(32) * 34;
            let egate_q8_needed = seq_len * erg8;
            if bs.batch_q8.len() < egate_q8_needed {
                bs.batch_q8.resize(egate_q8_needed, 0u8);
            }
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

            for s in 0..seq_len {
                let off = s * ffn_dim;
                ops::silu(&mut bs.gate[off..off + ffn_dim]);
                ops::mul_elem(
                    &bs.gate[off..off + ffn_dim],
                    &bs.up[off..off + ffn_dim],
                    &mut bs.ffn_tmp[off..off + ffn_dim],
                );
            }

            let efq8 = ffn_dim.div_ceil(32) * 34;
            let effn_q8_needed = seq_len * efq8;
            if bs.batch_q8.len() < effn_q8_needed {
                bs.batch_q8.resize(effn_q8_needed, 0u8);
            }
            self.pool.par_quantize(
                seq_len,
                ffn_dim,
                bs.ffn_tmp.as_ptr() as usize,
                bs.batch_q8.as_mut_ptr() as usize,
            );
            self.matmul_batch(
                &format!("{}.ffn_down.weight", p),
                &bs.ffn_tmp[..seq_len * ffn_dim],
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

        let create_buf = |n: usize, label: &str| gpu.create_storage_buffer(n as u64, label);

        // Token embed -> hidden_buf
        let token_id = token_ids[0] as usize;
        let embed_entry = self
            .tensors
            .get("token_embd.weight")
            .ok_or_else(|| "Missing token_embd.weight".to_string())?;
        let mut hidden = vec![0.0f32; d];
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
            for h in &mut hidden {
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
                _ => (0.0, 0.0, 0.0, 0.0), // n_dims=0 signals standard RoPE to shader
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
            let seq_len = pos + 1;
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
                    seq_len as u32,
                    cache_stride as u32,
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

        // --- FINAL NORM + LM HEAD (GPU+reads back 151669f32) ---
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
        // GPU lm_head matmul (full 151669×2048 dispatch) — keeps GPU busy → DVFS boost
        let w_buf = {
            let pool = gpu.pool.lock().unwrap();
            pool.get(&self.lm_head_name)
                .cloned()
                .ok_or_else(|| format!("Missing {}", self.lm_head_name))?
        };
        let lm_head_buf = gpu
            .dequant_matmul_fused(
                &w_buf,
                &hearth_compute::QuantKind::from_gguf_dtype(self.tensors[&self.lm_head_name].dtype)
                    .ok_or_else(|| format!("Unsupported dtype for {}", self.lm_head_name))?,
                &out_norm_tmp,
                vocab_size as u32,
                1,
                d as u32,
            )
            .ok_or_else(|| "Failed lm_head matmul".to_string())?;

        gpu.end_batch();

        let logits_vec = gpu.readback_f32(&lm_head_buf, vocab_size);
        logits[..vocab_size].copy_from_slice(&logits_vec[..vocab_size]);

        let fwd_total_us = fwd_t0.elapsed().as_micros();
        eprintln!(
            "[fwd] total={}us  layers={}us  read={}us",
            fwd_total_us,
            sum_layer_us,
            fwd_total_us - sum_layer_us as u128,
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
        let result = match &request {
            PipelineRequest::TextGen {
                prompt,
                sampler,
                max_new_tokens,
            } => self.generate_text(prompt, sampler, *max_new_tokens, &cancel, &output),
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

        let input_ids = self.tokenizer.lock().unwrap().encode(prompt, false);
        if input_ids.is_empty() {
            return Err("Empty prompt after tokenization".to_string());
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

        if input_ids.len() > 1 && !use_gpu {
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
                    self.forward_gpu(&[tid], i, &mut caches, &mut logits, cancel)?;
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
        total_tokens += input_ids.len();
        past_tokens.push(input_ids[input_ids.len() - 1]);

        let mut generated = 0;
        let mut cpu_overhead_us: u128 = 0;
        while generated < max_new_tokens {
            if cancel.load(Ordering::Relaxed) {
                let elapsed = start.elapsed().as_millis();
                let _ = output.send(EngineOutput::Done(RunStats {
                    elapsed_ms: elapsed,
                    tokens: total_tokens,
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
            sample_buf.copy_from_slice(&logits);

            let token = hearth_sampler::sample(&mut sample_buf, sampler_config, &past_tokens);

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
                self.forward_gpu(&[token], pos, &mut caches, &mut logits, cancel)?;
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

        let _ = output.send(EngineOutput::Done(RunStats {
            elapsed_ms: elapsed,
            tokens: total_tokens,
            tokens_per_s,
        }));

        Ok(())
    }
}
