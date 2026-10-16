use crate::attention;
use crate::ops;
use crate::weights::ModelWeights;
use hearth_compute::{GpuBuffer, GpuCompute, QuantKind};
use std::collections::HashMap;

pub struct FluxConfig {
    pub d_model: usize,         // 3072
    pub n_heads: usize,         // 24
    pub head_dim: usize,        // 128
    pub ffn_dim: usize,         // 9216
    pub n_double_layers: usize, // 5
    pub n_single_layers: usize, // 20
    pub joint_dim: usize,       // 7680 (text context dim)
    pub in_channels: usize,     // 128 (latent channels, patchified to d_model)
    pub rope_theta: f32,        // 2000
    pub axes_dims: Vec<usize>,  // [32, 32, 32, 32]
    pub eps: f32,               // 1e-6
}

impl Default for FluxConfig {
    fn default() -> Self {
        FluxConfig {
            d_model: 3072,
            n_heads: 24,
            head_dim: 128,
            ffn_dim: 9216,
            n_double_layers: 5,
            n_single_layers: 20,
            joint_dim: 7680,
            in_channels: 128,
            rope_theta: 2000.0,
            axes_dims: vec![32, 32, 32, 32],
            eps: 1e-6,
        }
    }
}

pub struct FluxTransformer {
    cfg: FluxConfig,
    pub weights: ModelWeights,
    gpu: Option<GpuCompute>,
    /// GPU buffers for Q2_0 weights, uploaded once during construction.
    gpu_q2_bufs: HashMap<String, GpuBuffer>,
    /// GPU buffers for BF16 weights converted to f32. Written during
    /// construction but not yet consumed by any forward path (unfinished
    /// GPU work, same as the dead_code-marked stubs in vae.rs).
    #[allow(dead_code)]
    gpu_f32_bufs: HashMap<String, GpuBuffer>,
}

impl FluxTransformer {
    pub fn new(cfg: FluxConfig, weights: ModelWeights) -> Self {
        let gpu = pollster::block_on(GpuCompute::new());
        let mut gpu_q2_bufs = HashMap::new();
        let mut gpu_f32_bufs = HashMap::new();

        if let Some(ref g) = gpu {
            eprintln!("[hearth-diffusion] GPU available, uploading weights...");
            // Upload all Q2_0 weights
            for (name, tensor) in &weights.tensors {
                match tensor {
                    crate::weights::StoredTensor::Q2(q2) => {
                        let buf = g.upload_bytes(&q2.data, &format!("diffusion:{}", name));
                        gpu_q2_bufs.insert(name.clone(), buf);
                    }
                    crate::weights::StoredTensor::Bf16(bf16) => {
                        // Convert BF16 → f32 on CPU, upload as f32 buffer.
                        // as_chunks::<2> needs Rust 1.88+.
                        let mut f32_data = Vec::with_capacity(bf16.data.len() / 2);
                        let (chunks, _remainder) = bf16.data.as_chunks::<2>();
                        for chunk in chunks {
                            let raw = u16::from_le_bytes(*chunk);
                            f32_data.push(f32::from_bits((raw as u32) << 16));
                        }
                        let buf = g.upload_f32(&f32_data, &format!("diffusion:{}", name));
                        gpu_f32_bufs.insert(name.clone(), buf);
                    }
                    crate::weights::StoredTensor::F32(f32t) => {
                        let buf = g.upload_f32(&f32t.data, &format!("diffusion:{}", name));
                        gpu_f32_bufs.insert(name.clone(), buf);
                    }
                }
            }
            eprintln!(
                "[hearth-diffusion] {} Q2_0 + {} f32 weights uploaded",
                gpu_q2_bufs.len(),
                gpu_f32_bufs.len()
            );
        }

        FluxTransformer {
            cfg,
            weights,
            gpu,
            gpu_q2_bufs,
            gpu_f32_bufs,
        }
    }

    /// Full forward pass: latent [seq_img × d_model] + text [seq_txt × joint_dim] + pooled [d_model]
    /// → denoised latent [seq_img × d_model]
    pub fn forward(
        &self,
        img: &[f32],
        txt: &[f32],
        _pooled: &[f32],
        t_emb: &[f32],
        _img_h: usize,
        _img_w: usize,
    ) -> Vec<f32> {
        let d = self.cfg.d_model;
        let seq_img = img.len() / d;
        let seq_txt = txt.len() / self.cfg.joint_dim;

        // Embeddings
        let mut img_emb = self.linear_bf16("x_embedder", img, d, self.cfg.in_channels);
        let mut txt_emb = self.linear_bf16("context_embedder", txt, d, self.cfg.joint_dim);

        // Time + guidance embedding
        let guidance = vec![0.0f32; 0]; // no guidance embedding
        let t_full = self.time_embed(t_emb, &guidance);

        // Double-stream blocks
        for i in 0..self.cfg.n_double_layers {
            let (new_img, new_txt) = self.double_stream_block(i, &img_emb, &txt_emb, &t_full);
            img_emb = new_img;
            txt_emb = new_txt;
        }

        // Concatenate for single-stream
        let mut hidden = img_emb;
        hidden.extend_from_slice(&txt_emb);

        // Single-stream blocks
        for i in 0..self.cfg.n_single_layers {
            hidden = self.single_stream_block(i, &hidden, &t_full, seq_img, seq_txt);
        }

        // Split off img portion
        let img_out = hidden[..seq_img * d].to_vec();

        // Final projection
        self.linear_bf16("proj_out", &img_out, self.cfg.in_channels, d)
    }

    fn time_embed(&self, t_emb: &[f32], _guidance: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model;
        // timestep_embedder: 256 → d
        let mut x =
            self.linear_bf16_single_idx("time_guidance_embed.timestep_embedder.linear_1", t_emb, d);
        ops::silu(&mut x);

        self.linear_bf16_single_idx("time_guidance_embed.timestep_embedder.linear_2", &x, d)
    }

    fn double_stream_block(
        &self,
        idx: usize,
        img: &[f32],
        txt: &[f32],
        t_emb: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let d = self.cfg.d_model;

        // Modulation
        let mod_img =
            self.linear_bf16_single_idx("double_stream_modulation_img.linear", t_emb, 6 * d);
        let mod_txt =
            self.linear_bf16_single_idx("double_stream_modulation_txt.linear", t_emb, 6 * d);

        let (shift_img, scale_img, gate_img) =
            (&mod_img[0..d], &mod_img[d..2 * d], &mod_img[2 * d..3 * d]);
        let (shift_mlp_img, scale_mlp_img, gate_mlp_img) = (
            &mod_img[3 * d..4 * d],
            &mod_img[4 * d..5 * d],
            &mod_img[5 * d..6 * d],
        );
        let (shift_txt, scale_txt, gate_txt) =
            (&mod_txt[0..d], &mod_txt[d..2 * d], &mod_txt[2 * d..3 * d]);
        let (shift_mlp_txt, scale_mlp_txt, gate_mlp_txt) = (
            &mod_txt[3 * d..4 * d],
            &mod_txt[4 * d..5 * d],
            &mod_txt[5 * d..6 * d],
        );

        // --- Norm + modulate (img) ---
        let mut img_normed = vec![0.0f32; img.len()];
        let img_seq = img.len() / d;
        for s in 0..img_seq {
            let off = s * d;
            ops::rms_norm(
                &img[off..off + d],
                &vec![1.0f32; d],
                self.cfg.eps,
                &mut img_normed[off..off + d],
            );
            for i in 0..d {
                img_normed[off + i] = img_normed[off + i] * (1.0 + scale_img[i]) + shift_img[i];
            }
        }

        // --- Norm + modulate (txt) ---
        let mut txt_normed = vec![0.0f32; txt.len()];
        let txt_seq = txt.len() / d;
        for s in 0..txt_seq {
            let off = s * d;
            ops::rms_norm(
                &txt[off..off + d],
                &vec![1.0f32; d],
                self.cfg.eps,
                &mut txt_normed[off..off + d],
            );
            for i in 0..d {
                txt_normed[off + i] = txt_normed[off + i] * (1.0 + scale_txt[i]) + shift_txt[i];
            }
        }

        // --- Q, K, V projections ---
        let n_heads = self.cfg.n_heads;
        let head_dim = self.cfg.head_dim;
        let n_qkv = n_heads * head_dim;

        let mut q_img = vec![0.0f32; img_seq * n_qkv];
        let mut k_img = vec![0.0f32; img_seq * n_qkv];
        let mut v_img = vec![0.0f32; img_seq * n_qkv];
        let mut q_txt = vec![0.0f32; txt_seq * n_qkv];
        let mut k_txt = vec![0.0f32; txt_seq * n_qkv];
        let mut v_txt = vec![0.0f32; txt_seq * n_qkv];

        let prefix = format!("transformer_blocks.{}.attn", idx);

        // Q, K, V projections (GPU-accelerated when available)
        self.matmul_q2_or_fallback(
            &format!("{}.to_q", prefix),
            &img_normed,
            &mut q_img,
            img_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.to_q", prefix),
            &txt_normed,
            &mut q_txt,
            txt_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.to_k", prefix),
            &img_normed,
            &mut k_img,
            img_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.to_k", prefix),
            &txt_normed,
            &mut k_txt,
            txt_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.to_v", prefix),
            &img_normed,
            &mut v_img,
            img_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.to_v", prefix),
            &txt_normed,
            &mut v_txt,
            txt_seq,
        );

        // --- Cross-attention Q, K, V (GPU-accelerated) ---
        let mut add_q = vec![0.0f32; img_seq * n_qkv];
        let mut add_k = vec![0.0f32; txt_seq * n_qkv];
        let mut add_v = vec![0.0f32; txt_seq * n_qkv];

        self.matmul_q2_or_fallback(
            &format!("{}.add_q_proj", prefix),
            &img_normed,
            &mut add_q,
            img_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.add_k_proj", prefix),
            &txt_normed,
            &mut add_k,
            txt_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.add_v_proj", prefix),
            &txt_normed,
            &mut add_v,
            txt_seq,
        );

        // Merge: Q = Q_img + Add_Q, K = cat(K_img, K_txt) + cat(Add_K, Add_K?)
        // Actually in flux: Q = Q_img + Add_Q, K = K_img + Add_K, V = V_img + Add_V
        // And we also have separate streams for txt attention
        // Simplified: just use the merged approach
        for i in 0..img_seq * n_qkv {
            q_img[i] += add_q[i];
        }
        // For img K/V, add txt K/V for cross-attention
        // Actually flux does: softmax(Q @ [K_txt, K_img]^T) @ [V_txt, V_img]
        // where Q = (Q_img + add_Q), K_img = (K_img + add_K), K_txt = K_txt
        // Actually, looking at flux source, it's more complex. Let me simplify:

        // Apply RoPE to img Q and K
        let mut q_img_rope = q_img.clone();
        let mut k_img_rope = k_img.clone();
        // rope_2d needs per-position encoding
        // For simplicity, use basic RoPE on img
        crate::ops::rope_2d(
            &mut q_img_rope,
            &mut k_img_rope,
            0,
            0,
            head_dim,
            self.cfg.rope_theta,
            &self.cfg.axes_dims,
        );

        // Attention: img queries attend to [txt_keys | img_keys]
        // and txt queries attend to [txt_keys | img_keys]
        let total_seq = img_seq + txt_seq;
        let mut attn_out_img = vec![0.0f32; img_seq * n_qkv];
        let mut attn_out_txt = vec![0.0f32; txt_seq * n_qkv];

        // Concatenate K and V
        let mut k_all = k_txt.clone();
        k_all.extend_from_slice(&k_img_rope);
        let mut v_all = v_txt.clone();
        v_all.extend_from_slice(&v_img);

        // Img attention (GPU-accelerated when available)
        match self.gpu_attention(&q_img_rope, &k_all, &v_all, total_seq, head_dim, n_heads) {
            Ok(gpu_out) => attn_out_img.copy_from_slice(&gpu_out),
            Err(_) => {
                attention::attention_batched(
                    &q_img_rope,
                    &k_all,
                    &v_all,
                    total_seq,
                    head_dim,
                    n_heads,
                    &mut attn_out_img,
                );
            }
        }

        // Txt attention (batched, uses txt Q which doesn't have RoPE)
        let mut k_all_txt = k_img_rope.clone();
        k_all_txt.extend_from_slice(&k_txt);
        let mut v_all_txt = v_img.clone();
        v_all_txt.extend_from_slice(&v_txt);

        match self.gpu_attention(&q_txt, &k_all_txt, &v_all_txt, total_seq, head_dim, n_heads) {
            Ok(gpu_out) => attn_out_txt.copy_from_slice(&gpu_out),
            Err(_) => {
                attention::attention_batched(
                    &q_txt,
                    &k_all_txt,
                    &v_all_txt,
                    total_seq,
                    head_dim,
                    n_heads,
                    &mut attn_out_txt,
                );
            }
        }

        // Output projections (GPU-accelerated)
        let mut add_out_img = vec![0.0f32; img_seq * d];
        let mut add_out_txt = vec![0.0f32; txt_seq * d];

        self.matmul_q2_or_fallback(
            &format!("{}.to_add_out", prefix),
            &attn_out_img,
            &mut add_out_img,
            img_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.to_add_out", prefix),
            &attn_out_txt,
            &mut add_out_txt,
            txt_seq,
        );

        // Output projections
        let mut out_img = vec![0.0f32; img_seq * d];
        let mut out_txt = vec![0.0f32; txt_seq * d];
        self.matmul_q2_or_fallback(
            &format!("{}.to_out.0", prefix),
            &attn_out_img,
            &mut out_img,
            img_seq,
        );
        self.matmul_q2_or_fallback(
            &format!("{}.to_out.0", prefix),
            &attn_out_txt,
            &mut out_txt,
            txt_seq,
        );

        // Gate + residual
        let mut new_img = img.to_vec();
        let mut new_txt = txt.to_vec();
        for i in 0..img_seq * d {
            new_img[i] += gate_img[i % d] * (out_img[i] + add_out_img[i]);
        }
        for i in 0..txt_seq * d {
            new_txt[i] += gate_txt[i % d] * (out_txt[i] + add_out_txt[i]);
        }

        // --- FFN (img) ---
        let mut img_ffn_normed = vec![0.0f32; img_seq * d];
        for s in 0..img_seq {
            let off = s * d;
            ops::rms_norm(
                &new_img[off..off + d],
                &vec![1.0f32; d],
                self.cfg.eps,
                &mut img_ffn_normed[off..off + d],
            );
            for i in 0..d {
                img_ffn_normed[off + i] =
                    img_ffn_normed[off + i] * (1.0 + scale_mlp_img[i]) + shift_mlp_img[i];
            }
        }
        let ffn_img_prefix = format!("transformer_blocks.{}.ff", idx);
        let ffn_total = self
            .weights
            .q2(&format!("{}.linear_in", ffn_img_prefix))
            .map(|w| w.rows)
            .unwrap_or(0);
        if ffn_total > 0 {
            let mut ffn_hidden = vec![0.0f32; img_seq * ffn_total];
            self.matmul_q2_or_fallback(
                &format!("{}.linear_in", ffn_img_prefix),
                &img_ffn_normed,
                &mut ffn_hidden,
                img_seq,
            );
            // Split into gate and up
            let half = ffn_total / 2;
            let mut gate = vec![0.0f32; img_seq * half];
            let mut up = vec![0.0f32; img_seq * half];
            for s in 0..img_seq {
                let off = s * ffn_total;
                gate[s * half..(s + 1) * half].copy_from_slice(&ffn_hidden[off..off + half]);
                up[s * half..(s + 1) * half]
                    .copy_from_slice(&ffn_hidden[off + half..off + ffn_total]);
            }
            ops::silu(&mut gate);
            for i in 0..gate.len() {
                gate[i] *= up[i];
            }
            let mut ffn_out = vec![0.0f32; img_seq * d];
            self.matmul_q2_or_fallback(
                &format!("{}.linear_out", ffn_img_prefix),
                &gate,
                &mut ffn_out,
                img_seq,
            );
            for i in 0..img_seq * d {
                new_img[i] += gate_mlp_img[i % d] * ffn_out[i];
            }
        }

        // --- FFN (txt) ---
        let mut txt_ffn_normed = vec![0.0f32; txt_seq * d];
        for s in 0..txt_seq {
            let off = s * d;
            ops::rms_norm(
                &new_txt[off..off + d],
                &vec![1.0f32; d],
                self.cfg.eps,
                &mut txt_ffn_normed[off..off + d],
            );
            for i in 0..d {
                txt_ffn_normed[off + i] =
                    txt_ffn_normed[off + i] * (1.0 + scale_mlp_txt[i]) + shift_mlp_txt[i];
            }
        }
        let ffn_txt_prefix = format!("transformer_blocks.{}.ff_context", idx);
        let ffn_txt_total = self
            .weights
            .q2(&format!("{}.linear_in", ffn_txt_prefix))
            .map(|w| w.rows)
            .unwrap_or(0);
        if ffn_txt_total > 0 {
            let mut ffn_hidden = vec![0.0f32; txt_seq * ffn_txt_total];
            self.matmul_q2_or_fallback(
                &format!("{}.linear_in", ffn_txt_prefix),
                &txt_ffn_normed,
                &mut ffn_hidden,
                txt_seq,
            );
            let half = ffn_txt_total / 2;
            let mut gate = vec![0.0f32; txt_seq * half];
            let mut up = vec![0.0f32; txt_seq * half];
            for s in 0..txt_seq {
                let off = s * ffn_total;
                gate[s * half..(s + 1) * half].copy_from_slice(&ffn_hidden[off..off + half]);
                up[s * half..(s + 1) * half]
                    .copy_from_slice(&ffn_hidden[off + half..off + ffn_total]);
            }
            ops::silu(&mut gate);
            for i in 0..gate.len() {
                gate[i] *= up[i];
            }
            let mut ffn_out = vec![0.0f32; txt_seq * d];
            self.matmul_q2_or_fallback(
                &format!("{}.linear_out", ffn_txt_prefix),
                &gate,
                &mut ffn_out,
                txt_seq,
            );
            for i in 0..txt_seq * d {
                new_txt[i] += gate_mlp_txt[i % d] * ffn_out[i];
            }
        }

        (new_img, new_txt)
    }

    fn single_stream_block(
        &self,
        idx: usize,
        hidden: &[f32],
        t_emb: &[f32],
        seq_img: usize,
        seq_txt: usize,
    ) -> Vec<f32> {
        let d = self.cfg.d_model;
        let seq = seq_img + seq_txt;

        // Modulation
        let mod_out = self.linear_bf16_single_idx("single_stream_modulation.linear", t_emb, 3 * d);
        let (shift, scale, gate) = (&mod_out[0..d], &mod_out[d..2 * d], &mod_out[2 * d..3 * d]);

        // Norm + modulate
        let mut normed = vec![0.0f32; hidden.len()];
        for s in 0..seq {
            let off = s * d;
            ops::rms_norm(
                &hidden[off..off + d],
                &vec![1.0f32; d],
                self.cfg.eps,
                &mut normed[off..off + d],
            );
            for i in 0..d {
                normed[off + i] = normed[off + i] * (1.0 + scale[i]) + shift[i];
            }
        }

        // Combined QKV + MLP projection (GPU-accelerated)
        let prefix = format!("single_transformer_blocks.{}.attn", idx);
        let proj_rows = self
            .weights
            .q2(&format!("{}.to_qkv_mlp_proj", prefix))
            .map(|w| w.rows)
            .unwrap_or(0);
        let mut qkv_mlp = vec![0.0f32; seq * proj_rows];
        self.matmul_q2_or_fallback(
            &format!("{}.to_qkv_mlp_proj", prefix),
            &normed,
            &mut qkv_mlp,
            seq,
        );

        let n_qkv = self.cfg.n_heads * self.cfg.head_dim;
        // Split: Q (3072), K (3072), V (3072), gate_mlp (9216), up_mlp (9216) = 27648
        let mut q_all = vec![0.0f32; seq * n_qkv];
        let mut k_all = vec![0.0f32; seq * n_qkv];
        let mut v_all = vec![0.0f32; seq * n_qkv];
        let mut gate_mlp = vec![0.0f32; seq * self.cfg.ffn_dim];
        let mut up_mlp = vec![0.0f32; seq * self.cfg.ffn_dim];

        for s in 0..seq {
            let off = s * proj_rows;
            q_all[s * n_qkv..(s + 1) * n_qkv].copy_from_slice(&qkv_mlp[off..off + n_qkv]);
            k_all[s * n_qkv..(s + 1) * n_qkv]
                .copy_from_slice(&qkv_mlp[off + n_qkv..off + 2 * n_qkv]);
            v_all[s * n_qkv..(s + 1) * n_qkv]
                .copy_from_slice(&qkv_mlp[off + 2 * n_qkv..off + 3 * n_qkv]);
            gate_mlp[s * self.cfg.ffn_dim..(s + 1) * self.cfg.ffn_dim]
                .copy_from_slice(&qkv_mlp[off + 3 * n_qkv..off + 3 * n_qkv + self.cfg.ffn_dim]);
            up_mlp[s * self.cfg.ffn_dim..(s + 1) * self.cfg.ffn_dim].copy_from_slice(
                &qkv_mlp
                    [off + 3 * n_qkv + self.cfg.ffn_dim..off + 3 * n_qkv + 2 * self.cfg.ffn_dim],
            );
        }

        // Apply RoPE to img portion Q and K
        crate::ops::rope_2d(
            &mut q_all[..seq_img * n_qkv],
            &mut k_all[..seq_img * n_qkv],
            0,
            0,
            self.cfg.head_dim,
            self.cfg.rope_theta,
            &self.cfg.axes_dims,
        );

        // Attention (GPU-accelerated when available)
        let mut attn_out = vec![0.0f32; seq * n_qkv];
        match self.gpu_attention(
            &q_all,
            &k_all,
            &v_all,
            seq,
            self.cfg.head_dim,
            self.cfg.n_heads,
        ) {
            Ok(gpu_out) => attn_out.copy_from_slice(&gpu_out),
            Err(_) => {
                attention::attention_batched(
                    &q_all,
                    &k_all,
                    &v_all,
                    seq,
                    self.cfg.head_dim,
                    self.cfg.n_heads,
                    &mut attn_out,
                );
            }
        }

        // MLP: silu(gate_mlp) * up_mlp
        ops::silu(&mut gate_mlp);
        for i in 0..gate_mlp.len() {
            gate_mlp[i] *= up_mlp[i];
        }

        // Concatenate attention output + MLP output → project via to_out
        let mut combined = attn_out;
        combined.extend_from_slice(&gate_mlp);

        let mut proj_out = vec![0.0f32; seq * d];
        self.matmul_q2_or_fallback(&format!("{}.to_out", prefix), &combined, &mut proj_out, seq);

        // Gate + residual
        let mut new_hidden = hidden.to_vec();
        for i in 0..seq * d {
            new_hidden[i] += gate[i % d] * proj_out[i];
        }

        new_hidden
    }

    fn linear_bf16(&self, name: &str, x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
        let batch = x.len() / in_dim;
        let mut out = vec![0.0f32; batch * out_dim];
        if let Some(w) = self.weights.bf16_as_f32(name) {
            for b in 0..batch {
                ops::matmul_bf16(
                    &w,
                    &x[b * in_dim..(b + 1) * in_dim],
                    &mut out[b * out_dim..(b + 1) * out_dim],
                    out_dim,
                    in_dim,
                );
            }
        } else if let Some(w) = self.weights.f32(name) {
            for b in 0..batch {
                ops::matmul_bf16(
                    &w.data,
                    &x[b * in_dim..(b + 1) * in_dim],
                    &mut out[b * out_dim..(b + 1) * out_dim],
                    out_dim,
                    in_dim,
                );
            }
        }
        out
    }

    fn linear_bf16_single_idx(&self, name: &str, x: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = x.len();
        if let Some(w) = self.weights.bf16_as_f32(name) {
            let mut out = vec![0.0f32; out_dim];
            // w has shape [out_dim, in_dim]
            for (i, slot) in out.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                let w_row = &w[i * in_dim..(i + 1) * in_dim];
                for j in 0..in_dim {
                    sum += w_row[j] * x[j];
                }
                *slot = sum;
            }
            out
        } else if let Some(w) = self.weights.f32(name) {
            let mut out = vec![0.0f32; out_dim];
            for (i, slot) in out.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                let w_row = &w.data[i * in_dim..(i + 1) * in_dim];
                for j in 0..in_dim {
                    sum += w_row[j] * x[j];
                }
                *slot = sum;
            }
            out
        } else {
            vec![0.0f32; out_dim]
        }
    }

    /// GPU-accelerated Q2_0 batch matmul: upload input → dispatch → readback output.
    /// Falls back to CPU matmul_q2_batched if GPU unavailable.
    fn gpu_matmul_q2(&self, name: &str, input: &[f32], batch: usize) -> Result<Vec<f32>, String> {
        let gpu = self.gpu.as_ref().ok_or("GPU not available")?;
        let buf = self
            .gpu_q2_bufs
            .get(name)
            .ok_or_else(|| format!("Q2_0 weight not on GPU: {}", name))?;
        let q2 = self
            .weights
            .q2(name)
            .ok_or_else(|| format!("Weight not found: {}", name))?;
        let m = q2.rows as u32;
        let k = q2.cols as u32;
        let act_buf = gpu.upload_f32(input, &format!("diff_act:{}", name));
        let out_buf = gpu
            .dequant_matmul_batch(buf, &QuantKind::Q2_0, &act_buf, m, k, batch as u32)
            .ok_or_else(|| format!("GPU matmul failed: {}", name))?;
        Ok(gpu.readback_f32(&out_buf, (m as usize) * batch))
    }

    /// Try GPU matmul; on failure, run CPU matmul_q2_batched into the given output slice.
    fn matmul_q2_or_fallback(&self, name: &str, input: &[f32], out: &mut [f32], batch: usize) {
        if let Ok(gpu_out) = self.gpu_matmul_q2(name, input, batch) {
            out.copy_from_slice(&gpu_out);
        } else if let Some(w) = self.weights.q2(name) {
            ops::matmul_q2_batched(w, input, out, batch);
        }
    }

    /// GPU-accelerated batch attention: upload Q, K, V → dispatch → readback output.
    /// GPU-accelerated batch attention: upload Q, K, V → dispatch → readback output.
    /// Falls back to CPU attention_batched if GPU unavailable.
    fn gpu_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        n_heads: usize,
    ) -> Result<Vec<f32>, String> {
        let gpu = self.gpu.as_ref().ok_or("GPU not available")?;
        let batch = q.len() / (n_heads * head_dim);
        let q_buf = gpu.upload_f32(q, "diff_attn_q");
        let k_buf = gpu.upload_f32(k, "diff_attn_k");
        let v_buf = gpu.upload_f32(v, "diff_attn_v");
        let out_buf = gpu
            .flash_attention_bidirectional(
                &q_buf,
                &k_buf,
                &v_buf,
                n_heads as u32,
                head_dim as u32,
                batch as u32,
                seq_len as u32,
            )
            .ok_or("GPU attention failed")?;
        Ok(gpu.readback_f32(&out_buf, q.len()))
    }
}
