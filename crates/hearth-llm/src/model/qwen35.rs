// qwen35 (Ternary Bonsai 2 / Qwen3.5 dense) hybrid architecture.
//
// Layer mix: every 4th block (3, 7, 11, ...) is gated full attention; the rest
// are gated delta net (GDN) linear-attention layers. Dense 27B: 64 blocks.
//
// Reference: PrismML-Eng/llama.cpp (prism branch), src/models/qwen35.cpp,
// src/models/deltanet.cpp and the Hadamard fold loader in src/llama-model.cpp.
//
// Decode-path math (single token; the n_tokens==1 limit of the chunked delta
// rule, matching FLA's fused_recurrent_gated_delta_rule):
//   conv:  y[c] = sum_k conv_w[k][c] * win[k][c], then silu
//   qk:    per head, l2-normalize(q), l2-normalize(k) (eps = rms eps)
//   g:     g = softplus(alpha + dt_bias) * ssm_a          (per v-head)
//   beta:  beta = sigmoid(beta_proj)                      (per v-head)
//   delta = v - S^T k
//   S <- S * exp(g) + k (beta * delta)^T
//   o = S^T (q / sqrt(k_dim))
//
// Hadamard fold: prism.hadamard.* metadata marks weights whose stored basis is
// rotated. Equivalence W x = (W H) H x for orthogonal H, so the runtime applies
// H (with the metadata sign vector) to every folded matmul input, per
// prism.hadamard.block_size (1024) on the input dimension:
//   x' = signs . x, then normalized WHT per block.
// token_embd is an inverse (latent) table: after row lookup apply
//   signs . (H . row)  (H is self-inverse up to the 1/sqrt(n) normalization).

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc::Sender,
    Arc,
};

use hearth_core::{EngineOutput, Model, ModelError, PipelineRequest, RunStats, SamplerConfig};
use hearth_gguf::{GgmlDType, GgufFile};
use hearth_tokenizer::Tokenizer;

use super::super::LlamaModel;
use super::kv_budget;
use super::tensor::TensorEntry;
use crate::config::ModelConfig;
use crate::kvcache::KVCache;
use crate::pool::ThreadPool;

// ---------------------------------------------------------------------------
// Hadamard transform
// ---------------------------------------------------------------------------

/// Normalized Sylvester Walsh-Hadamard: H[i][j] = s * (-1)^popcount(i & j),
/// s = 1/sqrt(n). In-place butterfly, O(n log n).
fn walsh_hadamard_in_place(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let s = 1.0 / (n as f32).sqrt();
    let mut h = 1usize;
    while h < n {
        for i in 0..n {
            if i & h == 0 {
                let a = x[i];
                let b = x[i | h];
                x[i] = a + b;
                x[i | h] = a - b;
            }
        }
        h <<= 1;
    }
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// In-place transform of `x`: multiply by the per-element sign vector (when
/// present), then apply the normalized WHT per block_size chunk.
/// Forward fold transform `R = D·H`: sign pass first, then the normalized WHT
/// per block. This is what quantized weight rows were folded by, so activations
/// feeding those weights are rotated the same way.
fn hadamard_rotate(x: &mut [f32], signs: Option<&[f32]>, block_size: usize) {
    if let Some(sgn) = signs {
        debug_assert_eq!(sgn.len(), x.len());
        for (v, s) in x.iter_mut().zip(sgn.iter()) {
            *v *= s;
        }
    }
    for chunk in x.chunks_exact_mut(block_size) {
        walsh_hadamard_in_place(chunk);
    }
}

/// Debug aid: with `HEARTH_DUMP_TENSORS=<dir-or-prefix>` set, layer-0 activations
/// at position 0 are written as `<prefix><name>.f32` so they can be diffed
/// element-wise against dumps from the reference implementation. Off by default.
fn dump_layer0(pos: usize, name: &str, buf: &[f32]) {
    if pos != 0 {
        return;
    }
    if let Some(dir) = std::env::var_os("HEARTH_DUMP_TENSORS") {
        let dir = dir.to_string_lossy();
        let _ = std::fs::write(format!("{}{}.f32", dir, name), unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 4)
        });
    }
}

/// Inverse fold transform `R⁻¹ = H·D`: normalized WHT per block first, then the
/// sign pass. Embedding-table rows are stored already folded by `R`, so a lookup
/// must apply `R⁻¹` — the opposite order from `hadamard_rotate`. Verified
/// bit-exact (cos = 1.000000) against the reference fork's `attn_norm-0` dump.
fn hadamard_unrotate(x: &mut [f32], signs: Option<&[f32]>, block_size: usize) {
    for chunk in x.chunks_exact_mut(block_size) {
        walsh_hadamard_in_place(chunk);
    }
    if let Some(sgn) = signs {
        debug_assert_eq!(sgn.len(), x.len());
        for (v, s) in x.iter_mut().zip(sgn.iter()) {
            *v *= s;
        }
    }
}

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// Whether weights should be read into RAM at load instead of borrowing the
/// GGUF mmap. `HEARTH_PIN_WEIGHTS=1` (or `--pin-weights`).
///
/// Both modes end up resident in `Working Set` during decode, but they differ in
/// ownership: a mapped tensor is *shared, file-backed* memory that the OS may
/// trim back to the standby cache at any time (Task Manager's "Memory" column
/// only counts the private portion, which is why a mapped 5.5 GB model shows as
/// ~350 MB). An owned tensor is private memory that stays put unless the system
/// is actually under memory pressure — at the cost of holding a second copy
/// alongside the file's cache pages.
pub(crate) fn pin_weights() -> bool {
    std::env::var_os("HEARTH_PIN_WEIGHTS").is_some_and(|v| v != "0")
}

/// Weight tensor: either borrowing the GGUF mmap (zero copy) or owned (pinned).
struct W(TensorEntry);

impl W {
    fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self, String> {
        let info = gguf
            .find_tensor(name)
            .ok_or_else(|| format!("qwen35: missing tensor {}", name))?;
        let start = (gguf.data_offset + info.offset) as usize;
        let len = info.byte_size();
        if pin_weights() {
            // Touch the bytes once, in one copy: the model lands in our own
            // memory and never depends on the OS file cache being warm.
            let bytes = gguf.mmap_arc()[start..start + len].to_vec();
            return Ok(W(TensorEntry::owned(bytes, info.dtype, info.shape.clone())));
        }
        Ok(W(TensorEntry::mapped(
            gguf.mmap_arc(),
            start,
            len,
            info.dtype,
            info.shape.clone(),
        )))
    }

    fn required(gguf: &GgufFile, name: &str) -> Result<Self, String> {
        Self::from_gguf(gguf, name)
    }
}

/// Dequantized 1-D norm weight (tiny, used every token).
struct Norm1D(Vec<f32>);

/// Dequantize a small dense tensor to f32 at load, row-major as stored. Used for
/// tensors that are read once per token and too small to justify a pool
/// dispatch, so they can stay f32 for the whole session instead of being
/// decoded per call.
fn load_dense(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| format!("qwen35: missing tensor {}", name))?;
    let n = info.shape.iter().product::<u64>() as usize;
    let mut v = vec![0.0f32; n];
    hearth_quant::dequantize(info.dtype, gguf.tensor_data(info), &mut v)?;
    Ok(v)
}

fn load_norm(gguf: &GgufFile, name: &str) -> Result<Norm1D, String> {
    Ok(Norm1D(load_dense(gguf, name)?))
}

fn meta_int(v: &hearth_gguf::MetaValue) -> Option<i64> {
    Some(match v {
        hearth_gguf::MetaValue::U8(x) => *x as i64,
        hearth_gguf::MetaValue::I8(x) => *x as i64,
        hearth_gguf::MetaValue::U16(x) => *x as i64,
        hearth_gguf::MetaValue::I16(x) => *x as i64,
        hearth_gguf::MetaValue::U32(x) => *x as i64,
        hearth_gguf::MetaValue::I32(x) => *x as i64,
        hearth_gguf::MetaValue::U64(x) => *x as i64,
        hearth_gguf::MetaValue::I64(x) => *x,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Per-layer tensors
// ---------------------------------------------------------------------------

struct Layer {
    attn_norm: Norm1D,
    post_attention_norm: Norm1D,
    // attention (full attn layers)
    attn_q: Option<W>,
    attn_k: Option<W>,
    attn_v: Option<W>,
    attn_q_norm: Option<Norm1D>,
    attn_k_norm: Option<Norm1D>,
    attn_output: Option<W>,
    // GDN layers
    attn_qkv: Option<W>,
    attn_gate: Option<W>,
    ssm_conv1d: Option<Vec<f32>>,
    ssm_dt_bias: Option<Vec<f32>>,
    ssm_a: Option<Vec<f32>>,
    /// `[n_v_heads][d_model]`, row-major, dequantized at load. The packs differ
    /// here: the 27B stores these as bf16, the 2B/4B/9B keep the whole
    /// linear-attention state path in f32 — so the storage dtype must not be
    /// baked into the kernel that reads them.
    ssm_alpha: Option<Vec<f32>>,
    ssm_beta: Option<Vec<f32>>,
    ssm_norm: Option<Norm1D>,
    ssm_out: Option<W>,
    // FFN (all layers)
    ffn_gate: W,
    ffn_up: W,
    ffn_down: W,
}

// ---------------------------------------------------------------------------
// Runtime state
// ---------------------------------------------------------------------------

/// GDN recurrent state: per layer, per v-head a [k_dim x v_dim] matrix.
struct GdnState {
    n_v_heads: usize,
    dim: usize,
    s: Vec<f32>, // [layer][head][k][v]
}

impl GdnState {
    fn new(n_layers: usize, n_v_heads: usize, dim: usize) -> Self {
        GdnState {
            n_v_heads,
            dim,
            s: vec![0.0; n_layers * n_v_heads * dim * dim],
        }
    }

    fn layer_slice(&mut self, layer: usize) -> &mut [f32] {
        let per = self.n_v_heads * self.dim * self.dim;
        &mut self.s[layer * per..(layer + 1) * per]
    }
}

/// Sliding conv state: the last (kernel-1) conv inputs per GDN layer.
struct ConvState {
    conv_dim: usize,
    kernel: usize,
    buf: Vec<f32>, // [layer][kernel-1][conv_dim]
}

impl ConvState {
    fn new(n_layers: usize, conv_dim: usize, kernel: usize) -> Self {
        ConvState {
            conv_dim,
            kernel,
            buf: vec![0.0; n_layers * (kernel - 1) * conv_dim],
        }
    }

    /// Push the current conv input; fill `window` (layout [channel][tap]) with
    /// tap 0 = oldest input, tap kernel-1 = the current input x.
    fn push(&mut self, layer: usize, x: &[f32], window: &mut [f32]) {
        let k = self.kernel;
        let c = self.conv_dim;
        let hist = (k - 1) * c;
        let base = layer * hist;
        // shift history left by one tap
        assert!(x.len() >= c);
        for (ch, &xv) in x.iter().enumerate().take(c) {
            let w = ch * k;
            window.copy_within(w..w + k - 1, w);
            window[w + k - 1] = xv;
        }
        // store back the newest kernel-1 taps for the next step
        for ch in 0..c {
            let w = ch * k;
            self.buf[base + ch * (k - 1)..base + (ch + 1) * (k - 1)]
                .copy_from_slice(&window[w + 1..w + k]);
        }
    }

    /// Restore the history portion of `window` from the cache (taps 0..k-2),
    /// leaving tap k-1 to be filled by the current input.
    fn prepare(&self, layer: usize, window: &mut [f32]) {
        let k = self.kernel;
        let c = self.conv_dim;
        let hist = (k - 1) * c;
        let base = layer * hist;
        for ch in 0..c {
            let w = ch * k;
            window[w..w + k - 1]
                .copy_from_slice(&self.buf[base + ch * (k - 1)..base + (ch + 1) * (k - 1)]);
            window[w + k - 1] = 0.0;
        }
    }
}

/// Hybrid caches: KV for attention layers, conv + SSM state for GDN layers.
pub struct Qwen35Caches {
    kv: Vec<KVCache>,
    gdn: GdnState,
    conv: ConvState,
}

// ---------------------------------------------------------------------------
// Scratch
// ---------------------------------------------------------------------------

/// Per-stage wall-clock totals for one generation, accumulated inside `forward`.
/// Enabled with `HEARTH_Q35_TIMERS=1`.
///
/// The hybrid path has no `ForwardTimers` equivalent: it is not a stack of the
/// same matmul shape, and the interesting question is which *kind* of work
/// dominates (weight-streaming matmul vs. the GDN recurrence vs. the lm_head,
/// which is 248k rows on this family). `total_us` is the sum of the five stages
/// for the forward passes in one phase, so `TOTAL / tokens` is the per-token
/// cost with the sampler and tokenizer excluded.
#[derive(Default)]
struct Q35Timers {
    embed_us: u128,
    norm_us: u128,
    /// GDN recurrence or full attention for a layer (`gdn_layer`/`attn_layer`).
    mix_us: u128,
    /// Sandwich norm + FFN branch + both residuals for one layer.
    ffn_us: u128,
    /// `output_norm` plus the language-model head.
    head_us: u128,
    total_us: u128,
}

impl Q35Timers {
    fn report(&self, label: &str, tokens: usize) {
        let total = self.total_us.max(1);
        let per_token = total as f64 / 1000.0 / tokens.max(1) as f64;
        eprintln!(
            "[q35timing] {}: {} tokens, {:.3} ms/token",
            label, tokens, per_token
        );
        let rows: [(&str, u128); 6] = [
            ("embed", self.embed_us),
            ("attn_norm", self.norm_us),
            ("gdn/attn", self.mix_us),
            ("ffn", self.ffn_us),
            ("head", self.head_us),
            ("TOTAL", self.total_us),
        ];
        for (name, us) in rows {
            eprintln!(
                "[q35timing]   {:<12} {:>10} us  {:>6.2}%",
                name,
                us,
                us as f64 * 100.0 / total as f64
            );
        }
    }
}

struct FwdScratch {
    hidden: Vec<f32>,     // d_model
    normed: Vec<f32>,     // d_model
    tmp_f32: Vec<f32>,    // hadamard / qk scratch (widest folded input)
    qkv: Vec<f32>,        // conv_dim = 2*k_dim + v_dim
    zgate: Vec<f32>,      // v_dim (gdn z gate)
    conv_win: Vec<f32>,   // kernel * conv_dim
    conv_out: Vec<f32>,   // conv_dim
    q_buf: Vec<f32>,      // 2*nq (attn q|gate)
    k_buf: Vec<f32>,      // nkv
    v_buf: Vec<f32>,      // nkv
    attn_gate: Vec<f32>,  // nq
    attn_out: Vec<f32>,   // max(nq, v_dim)
    scores: Vec<f32>,     // attention scores (seq_len)
    abuf: Vec<f32>,       // alpha sums (dt_rank)
    bbuf: Vec<f32>,       // beta sums (dt_rank)
    gdec: Vec<f32>,       // decay gates (dt_rank)
    beta: Vec<f32>,       // beta values (dt_rank)
    normed_out: Vec<f32>, // v_dim
    ffn_gate: Vec<f32>,   // d_ffn
    ffn_up: Vec<f32>,     // d_ffn
    act: Vec<f32>,        // layer output (d_model)
    proj_q8: Vec<u8>,     // activation q8 scratch
    /// Stage totals, accumulated only when `timers_on`.
    timers: Q35Timers,
}

// ---------------------------------------------------------------------------
// Core model
// ---------------------------------------------------------------------------

pub struct Qwen35Core {
    config: ModelConfig,
    pool: ThreadPool,
    tokenizer: std::sync::Mutex<Tokenizer>,

    token_embd: W,
    /// LM head. `None` for a tied-output model (hadamard version 2), where the
    /// latent embedding table is rotated the same way and used in its place.
    output: Option<W>,
    output_norm: Norm1D,

    had_block_size: usize,
    had_signs: HashMap<usize, Vec<f32>>,
    had_embed_inverse: bool,
    // Diagnostic kill-switches, cached at load (see the accessors below).
    kill_had: bool,
    kill_ffn: bool,
    kill_had_mm: bool,
    kill_emb_inv: bool,
    kill_gdn: bool,
    kill_attn: bool,
    /// `HEARTH_Q35_TIMERS` — cache the env lookup (it takes a process-wide lock
    /// on Windows and this is consulted on every stage of every token).
    timers_on: bool,

    layers: Vec<Layer>,
    /// true when the layer is a GDN (recurrent) layer
    layer_is_gdn: Vec<bool>,
    /// attention-cache index per layer (unused marker for GDN layers)
    attn_index: Vec<usize>,

    d_model: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    k_dim: usize,
    v_dim: usize,
    conv_dim: usize,
    dt_rank: usize,
    n_groups: usize,
    state_size: usize,
    conv_kernel: usize,

    max_seq: usize,
    /// Bytes the f32 attention KV cache occupies at `max_seq` (allocated at
    /// load, reported to the UI so the RAM cost is visible).
    kv_cache_bytes: u64,
    rope_cache: crate::ops::RopeCache,
    scratch: std::sync::Mutex<FwdScratch>,
}

impl Qwen35Core {
    pub fn from_gguf(gguf: &GgufFile, tokenizer: Tokenizer) -> Result<Self, String> {
        let mut config = ModelConfig::from_gguf(gguf)?;
        // Dense-architecture validation rules do not apply here (d_model need not
        // divide by n_heads); check qwen35-specific geometry instead.
        if config.n_layers == 0 || config.d_model == 0 || config.vocab_size == 0 {
            return Err("qwen35: incomplete model config".into());
        }

        let d = config.d_model as usize;
        let n_heads = config.n_heads as usize;
        let n_kv = config.n_kv_heads as usize;

        // head_dim: derive from the first attention layer's K projection
        // (rows = n_kv_heads x head_dim). d_model/n_heads is wrong here (5120/24).
        let head_dim = {
            let mut hd = config.head_dim as usize;
            for i in 0..config.n_layers as usize {
                if let Some(info) = gguf.find_tensor(&format!("blk.{}.attn_k.weight", i)) {
                    let rows = *info.shape.get(1).ok_or("qwen35: bad attn_k shape")? as usize;
                    if rows.is_multiple_of(n_kv) && rows / n_kv > 0 {
                        hd = rows / n_kv;
                    }
                    break;
                }
            }
            hd
        };
        config.head_dim = head_dim as u32;
        let rope_dim = config.rope_dim as usize;

        // GDN geometry (mirrors the fork's load_arch_tensors calculations:
        // head_k_dim = head_v_dim = ssm state_size, n_k_heads = group_count,
        // n_v_heads = time_step_rank).
        let state_size = gguf
            .meta_u32("qwen35.ssm.state_size")
            .ok_or("qwen35: missing qwen35.ssm.state_size")? as usize;
        let n_groups = gguf
            .meta_u32("qwen35.ssm.group_count")
            .ok_or("qwen35: missing qwen35.ssm.group_count")? as usize;
        let dt_rank = gguf
            .meta_u32("qwen35.ssm.time_step_rank")
            .ok_or("qwen35: missing qwen35.ssm.time_step_rank")? as usize;
        let conv_kernel = gguf
            .meta_u32("qwen35.ssm.conv_kernel")
            .ok_or("qwen35: missing qwen35.ssm.conv_kernel")? as usize;
        let v_dim = state_size * dt_rank;
        let k_dim = state_size * n_groups;
        let conv_dim = 2 * k_dim + v_dim;
        let d_inner_declared = gguf
            .meta_u32("qwen35.ssm.inner_size")
            .unwrap_or(v_dim as u32) as usize;
        if d_inner_declared != v_dim {
            return Err(format!(
                "qwen35: ssm.inner_size {} != state_size*time_step_rank {}",
                d_inner_declared, v_dim
            ));
        }

        // Recurrent layer flags.
        let layer_is_gdn: Vec<bool> = match &config.recurrent_layers {
            Some(v) if v.len() == config.n_layers as usize => v.clone(),
            _ => {
                let interval =
                    gguf.meta_u32("qwen35.full_attention_interval").unwrap_or(4) as usize;
                (0..config.n_layers as usize)
                    .map(|i| (i + 1) % interval != 0)
                    .collect()
            }
        };

        // Hadamard fold metadata. Refuse to run rotated weights without it.
        let had_block_size = gguf.meta_u32("prism.hadamard.block_size").unwrap_or(0) as usize;
        let had_version = gguf.meta_u32("prism.hadamard.version").unwrap_or(0);
        // Version 2 of the fold metadata means the LM head is *not* stored: the
        // (latent, folded) `token_embd.weight` table doubles as the output
        // projection. The reference folds it into the ordinary rotated-weight
        // path, so the head matmul is a plain `mm(.., had = true)` against the
        // latent rows — see the `output` field for how it is wired up.
        let had_tied_output = gguf
            .meta_bool("prism.hadamard.tied_output")
            .unwrap_or(false);
        let mut had_signs: HashMap<usize, Vec<f32>> = HashMap::new();
        let mut had_embed_inverse = false;
        if had_block_size > 0 {
            if had_version != 1 && had_version != 2 {
                return Err(format!(
                    "qwen35: unsupported prism.hadamard.version {}",
                    had_version
                ));
            }
            let transform = gguf.meta_str("prism.hadamard.transform").unwrap_or("");
            if transform != "normalized-sylvester-walsh-hadamard" {
                return Err(format!(
                    "qwen35: unsupported hadamard transform {}",
                    transform
                ));
            }
            let widths = gguf
                .meta_array("prism.hadamard.sign_widths")
                .ok_or("qwen35: missing prism.hadamard.sign_widths")?;
            let values = gguf
                .meta_array("prism.hadamard.sign_values")
                .ok_or("qwen35: missing prism.hadamard.sign_values")?;
            let mut off = 0usize;
            for wv in widths {
                let width = meta_int(wv).ok_or("qwen35: bad sign width")? as usize;
                if width == 0 || !width.is_multiple_of(had_block_size) || off + width > values.len()
                {
                    return Err(format!("qwen35: invalid hadamard sign width {}", width));
                }
                let vec: Vec<f32> = values[off..off + width]
                    .iter()
                    .map(|v| {
                        let x = meta_int(v).unwrap_or(0);
                        if x != 1 && x != -1 {
                            return Err("qwen35: hadamard sign values must be +/-1".to_string());
                        }
                        Ok(x as f32)
                    })
                    .collect::<Result<Vec<f32>, String>>()?;
                off += width;
                had_signs.insert(width, vec);
            }
            if off != values.len() {
                return Err("qwen35: hadamard sign_values length mismatch".into());
            }
            // Inverse (latent) embedding table.
            if let Some(inv) = gguf.meta_array("prism.hadamard.inverse_weight_names") {
                had_embed_inverse = inv.iter().any(|v| v.as_str() == Some("token_embd.weight"));
            }
        }

        // Tied-output invariants, mirrored from the reference loader so a
        // malformed file fails loudly at load instead of producing silently
        // wrong logits (a latent table used as a head without its fold is still
        // the right magnitude, just decorrelated).
        if had_tied_output {
            if had_version != 2 || had_block_size == 0 {
                return Err("qwen35: tied output requires prism.hadamard version 2".into());
            }
            if gguf.find_tensor("output.weight").is_some() {
                return Err(
                    "qwen35: prism.hadamard.tied_output requires output.weight to be absent".into(),
                );
            }
            if !had_embed_inverse {
                return Err(
                    "qwen35: prism.hadamard.tied_output requires a latent token_embd".into(),
                );
            }
        }

        let token_embd = W::required(gguf, "token_embd.weight")?;
        // `None` = tied head; the forward pass falls back to `token_embd`.
        let output = if had_tied_output {
            None
        } else {
            Some(W::required(gguf, "output.weight")?)
        };
        let output_norm = load_norm(gguf, "output_norm.weight")?;

        let n_layers = config.n_layers as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for (i, &is_gdn) in layer_is_gdn.iter().enumerate().take(n_layers) {
            let p = format!("blk.{}.", i);
            let layer = if is_gdn {
                let conv_w = gguf
                    .find_tensor(&format!("{}ssm_conv1d.weight", p))
                    .ok_or_else(|| format!("qwen35: missing {}ssm_conv1d.weight", p))?;
                let mut conv1d = vec![0.0f32; conv_kernel * conv_dim];
                hearth_quant::dequantize(conv_w.dtype, gguf.tensor_data(conv_w), &mut conv1d)?;
                let dt_info = gguf
                    .find_tensor(&format!("{}ssm_dt.bias", p))
                    .ok_or_else(|| format!("qwen35: missing {}ssm_dt.bias", p))?;
                let mut dt_bias = vec![0.0f32; dt_rank];
                hearth_quant::dequantize(dt_info.dtype, gguf.tensor_data(dt_info), &mut dt_bias)?;
                let a_info = gguf
                    .find_tensor(&format!("{}ssm_a", p))
                    .ok_or_else(|| format!("qwen35: missing {}ssm_a", p))?;
                let mut ssm_a = vec![0.0f32; dt_rank];
                hearth_quant::dequantize(a_info.dtype, gguf.tensor_data(a_info), &mut ssm_a)?;
                Layer {
                    attn_norm: load_norm(gguf, &format!("{}attn_norm.weight", p))?,
                    post_attention_norm: load_norm(
                        gguf,
                        &format!("{}post_attention_norm.weight", p),
                    )?,
                    attn_q: None,
                    attn_k: None,
                    attn_v: None,
                    attn_q_norm: None,
                    attn_k_norm: None,
                    attn_output: None,
                    attn_qkv: Some(W::required(gguf, &format!("{}attn_qkv.weight", p))?),
                    attn_gate: Some(W::required(gguf, &format!("{}attn_gate.weight", p))?),
                    ssm_conv1d: Some(conv1d),
                    ssm_dt_bias: Some(dt_bias),
                    ssm_a: Some(ssm_a),
                    ssm_alpha: Some(load_dense(gguf, &format!("{}ssm_alpha.weight", p))?),
                    ssm_beta: Some(load_dense(gguf, &format!("{}ssm_beta.weight", p))?),
                    ssm_norm: Some(load_norm(gguf, &format!("{}ssm_norm.weight", p))?),
                    ssm_out: Some(W::required(gguf, &format!("{}ssm_out.weight", p))?),
                    ffn_gate: W::required(gguf, &format!("{}ffn_gate.weight", p))?,
                    ffn_up: W::required(gguf, &format!("{}ffn_up.weight", p))?,
                    ffn_down: W::required(gguf, &format!("{}ffn_down.weight", p))?,
                }
            } else {
                Layer {
                    attn_norm: load_norm(gguf, &format!("{}attn_norm.weight", p))?,
                    post_attention_norm: load_norm(
                        gguf,
                        &format!("{}post_attention_norm.weight", p),
                    )?,
                    attn_q: Some(W::required(gguf, &format!("{}attn_q.weight", p))?),
                    attn_k: Some(W::required(gguf, &format!("{}attn_k.weight", p))?),
                    attn_v: Some(W::required(gguf, &format!("{}attn_v.weight", p))?),
                    attn_q_norm: Some(load_norm(gguf, &format!("{}attn_q_norm.weight", p))?),
                    attn_k_norm: Some(load_norm(gguf, &format!("{}attn_k_norm.weight", p))?),
                    attn_output: Some(W::required(gguf, &format!("{}attn_output.weight", p))?),
                    attn_qkv: None,
                    attn_gate: None,
                    ssm_conv1d: None,
                    ssm_dt_bias: None,
                    ssm_a: None,
                    ssm_alpha: None,
                    ssm_beta: None,
                    ssm_norm: None,
                    ssm_out: None,
                    ffn_gate: W::required(gguf, &format!("{}ffn_gate.weight", p))?,
                    ffn_up: W::required(gguf, &format!("{}ffn_up.weight", p))?,
                    ffn_down: W::required(gguf, &format!("{}ffn_down.weight", p))?,
                }
            };
            layers.push(layer);
        }

        // Attention-cache index per layer.
        let mut attn_index = vec![usize::MAX; n_layers];
        let mut a = 0usize;
        for (i, gdn) in layer_is_gdn.iter().enumerate() {
            if !gdn {
                attn_index[i] = a;
                a += 1;
            }
        }

        // Cap the allocated KV cache so the declared context does not blow up
        // memory. The cap is a *ceiling*, not a fixed setting: the UI passes a
        // smaller per-request window and the engine honors it, and
        // `HEARTH_CTX_CAP` overrides the ceiling for machines with headroom.
        //
        // Only the full-attention layers keep a cache (the gated-delta layers
        // hold a fixed-size recurrence instead), but each such layer stores K+V
        // as f32: 16 attn layers x 4 kv heads x head_dim 256 x 2 x 4 B = 128 KiB
        // per token, so the declared 262144 tokens is 32 GiB — far past the RAM
        // of the machines that run this model. Budget it by bytes instead.
        let n_attn_layers = layer_is_gdn.iter().filter(|g| !**g).count();
        let n_kv_heads = config.n_kv_heads as usize;
        let kv_per_token = kv_budget::kv_bytes_per_token(n_attn_layers, n_kv_heads, head_dim);
        let max_seq = kv_budget::ctx_cap(
            config.max_seq_len as usize,
            n_attn_layers,
            n_kv_heads,
            head_dim,
            None,
        );
        config.max_seq_len = max_seq as u32;
        eprintln!(
            "[kv] context {} tokens · {} per token ({} attn layers) · {} KV cache",
            max_seq,
            kv_budget::fmt_bytes(kv_per_token),
            n_attn_layers,
            kv_budget::fmt_bytes(max_seq as u64 * kv_per_token)
        );

        let rope_cache = crate::ops::RopeCache::new(
            max_seq,
            rope_dim,
            config.rope_theta,
            config.rope_scaling_type.as_deref(),
            config.rope_scaling_factor,
            config.original_ctx_len,
        );

        let max_width = had_signs.keys().copied().max().unwrap_or(d);
        let nq = n_heads * head_dim;
        let nkv = n_kv * head_dim;
        let ffn_dim = config.d_ffn as usize;
        let scratch = FwdScratch {
            hidden: vec![0.0; d],
            normed: vec![0.0; d],
            tmp_f32: vec![0.0; max_width.max(2 * k_dim)],
            qkv: vec![0.0; conv_dim],
            zgate: vec![0.0; v_dim],
            conv_win: vec![0.0; conv_kernel * conv_dim],
            conv_out: vec![0.0; conv_dim],
            q_buf: vec![0.0; 2 * nq],
            k_buf: vec![0.0; nkv],
            v_buf: vec![0.0; nkv],
            attn_gate: vec![0.0; nq],
            attn_out: vec![0.0; nq.max(v_dim)],
            scores: Vec::new(),
            abuf: vec![0.0; dt_rank],
            bbuf: vec![0.0; dt_rank],
            gdec: vec![0.0; dt_rank],
            beta: vec![0.0; dt_rank],
            normed_out: vec![0.0; v_dim],
            ffn_gate: vec![0.0; ffn_dim],
            ffn_up: vec![0.0; ffn_dim],
            act: vec![0.0; d],
            proj_q8: Vec::new(),
            timers: Q35Timers::default(),
        };

        Ok(Qwen35Core {
            config,
            // The ternary dots are memory-bound; the main thread only spins while
            // the workers run, so worker count is tunable (HEARTH_THREADS) and
            // defaults to the physical-core count where we can detect it.
            pool: ThreadPool::new(
                std::env::var("HEARTH_THREADS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|n| *n > 0)
                    .unwrap_or_else(|| {
                        // SMT siblings contend with the spinning main thread for
                        // memory bandwidth, so back off from the logical count on
                        // wider machines (measured optimum: 12 of 16 on an 8c/16t).
                        let logical = std::thread::available_parallelism()
                            .map(|n| n.get())
                            .unwrap_or(4);
                        if logical > 8 {
                            (logical * 3 / 4).max(8)
                        } else {
                            logical
                        }
                    }),
            ),
            tokenizer: std::sync::Mutex::new(tokenizer),
            token_embd,
            output,
            output_norm,
            had_block_size,
            had_signs,
            had_embed_inverse,
            kill_had: std::env::var_os("HEARTH_Q35_NOHAD").is_some(),
            kill_ffn: std::env::var_os("HEARTH_Q35_NOFFN").is_some(),
            kill_had_mm: std::env::var_os("HEARTH_Q35_NOHADMM").is_some(),
            kill_emb_inv: std::env::var_os("HEARTH_Q35_NOEMBINV").is_some(),
            kill_gdn: std::env::var_os("HEARTH_Q35_NOGDN").is_some(),
            kill_attn: std::env::var_os("HEARTH_Q35_NOATTN").is_some(),
            timers_on: std::env::var_os("HEARTH_Q35_TIMERS").is_some(),
            layers,
            layer_is_gdn,
            attn_index,
            d_model: d,
            n_heads,
            n_kv_heads: n_kv,
            head_dim,
            rope_dim,
            k_dim,
            v_dim,
            conv_dim,
            dt_rank,
            n_groups,
            state_size,
            conv_kernel,
            max_seq,
            kv_cache_bytes: max_seq as u64 * kv_per_token,
            rope_cache,
            scratch: std::sync::Mutex::new(scratch),
        })
    }

    /// Bytes the f32 attention KV cache occupies at the allocated context.
    pub(crate) fn kv_cache_bytes(&self) -> u64 {
        self.kv_cache_bytes
    }

    /// Zero the stage totals. No-op unless `HEARTH_Q35_TIMERS` is set.
    fn timers_reset(&self) {
        if self.timers_on {
            self.scratch.lock().unwrap().timers = Q35Timers::default();
        }
    }

    /// Print the stage split for one phase. No-op unless timing is enabled.
    fn timers_report(&self, label: &str, tokens: usize) {
        if self.timers_on {
            self.scratch.lock().unwrap().timers.report(label, tokens);
        }
    }

    fn signs_for(&self, width: usize) -> Option<&[f32]> {
        self.had_signs.get(&width).map(|v| v.as_slice())
    }

    // Diagnostic kill-switches (HEARTH_Q35_NOHAD / NOFFN / NOHADMM / NOEMBINV /
    // NOGDN / NOATTN). Read once at load: they are consulted per layer and per
    // matmul, and `std::env::var_os` takes a process-wide lock on Windows.
    fn no_had(&self) -> bool {
        self.kill_had
    }

    fn no_ffn(&self) -> bool {
        self.kill_ffn
    }

    fn no_had_mm(&self) -> bool {
        self.kill_had_mm
    }

    fn no_emb_inv(&self) -> bool {
        self.kill_emb_inv
    }

    fn no_gdn(&self) -> bool {
        self.kill_gdn
    }

    fn no_attn(&self) -> bool {
        self.kill_attn
    }

    /// Final post-load config (head_dim corrections, KV context cap applied).
    pub fn config_snapshot(&self) -> ModelConfig {
        self.config.clone()
    }

    /// Matmul with optional Hadamard rotation applied to the activation.
    /// `out` must not alias `x` (tmp breaks the alias when rotating).
    fn mm(
        &self,
        w: &W,
        x: &[f32],
        out: &mut [f32],
        tmp: &mut [f32],
        q8: &mut Vec<u8>,
        had: bool,
    ) -> Result<(), String> {
        let entry = &w.0;
        let n_cols = entry.n_cols();
        let n_rows = entry.n_rows();
        if x.len() != n_cols {
            return Err(format!(
                "qwen35: matmul width mismatch (w cols {} vs x {})",
                n_cols,
                x.len()
            ));
        }
        if out.len() != n_rows {
            return Err(format!(
                "qwen35: matmul out width mismatch (w rows {} vs out {})",
                n_rows,
                out.len()
            ));
        }
        if std::env::var_os("HEARTH_MM_DEBUG").is_some() && w.0.n_rows() >= 1024 {
            let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
            for &v in x {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            eprintln!(
                "[mmdbg] rows={} cols={} dtype={} x_mean={:.4} x_min={:.3} x_max={:.3}",
                w.0.n_rows(),
                w.0.n_cols(),
                w.0.dtype.name(),
                mean_abs(x),
                mn,
                mx
            );
        }
        let x_rot: &[f32] = if had && self.had_block_size > 0 && !self.no_had() && !self.no_had_mm()
        {
            // tmp is a fixed-capacity shared scratch; never resize it (other
            // callers slice it at larger widths).
            if x.len() > tmp.len() {
                return Err(format!(
                    "qwen35: hadamard scratch too small ({} < {})",
                    tmp.len(),
                    x.len()
                ));
            }
            let tmp_slice: &mut [f32] = &mut tmp[..x.len()];
            tmp_slice.copy_from_slice(x);
            hadamard_rotate(tmp_slice, self.signs_for(n_cols), self.had_block_size);
            tmp_slice
        } else {
            x
        };
        q8.clear();
        hearth_quant::quantize_q8_0(x_rot, q8);
        let w_base = entry.data().as_ptr() as usize;
        let a_ptr = q8.as_ptr() as usize;
        let out_ptr = out.as_mut_ptr() as usize;
        let row_bytes = entry.dtype.byte_size(n_cols);
        let dot = match entry.dtype {
            GgmlDType::PTQ1_0 => hearth_quant::dot_ptq1_0_q8_0_ptr,
            GgmlDType::PQ2_0 | GgmlDType::Q2_0 => hearth_quant::dot_q2_0_q8_0_ptr,
            GgmlDType::Q1_0 | GgmlDType::Q10G128 => hearth_quant::dot_q1_0g128_q8_0_ptr,
            GgmlDType::Q8_0 => hearth_quant::dot_q8_0_q8_0_ptr,
            other => {
                return Err(format!(
                    "qwen35: unsupported weight dtype {} (needs a q8 dot kernel)",
                    other.name()
                ))
            }
        };
        self.pool.par_dot_rows(crate::pool::DotRowsParams {
            n: n_rows,
            w_base,
            a_ptr,
            out_ptr,
            row_bytes,
            n_cols,
            dot_fn: dot,
        });
        Ok(())
    }

    /// Matmul with a pre-quantized Q8 activation (used after manual transforms).
    fn mm_q8(&self, w: &W, q8: &[u8], out: &mut [f32]) -> Result<(), String> {
        let entry = &w.0;
        let n_cols = entry.n_cols();
        let n_rows = entry.n_rows();
        if out.len() != n_rows {
            return Err(format!(
                "qwen35: matmul out width mismatch (w rows {} vs out {})",
                n_rows,
                out.len()
            ));
        }
        let dot = match entry.dtype {
            GgmlDType::PTQ1_0 => hearth_quant::dot_ptq1_0_q8_0_ptr,
            GgmlDType::PQ2_0 | GgmlDType::Q2_0 => hearth_quant::dot_q2_0_q8_0_ptr,
            GgmlDType::Q1_0 | GgmlDType::Q10G128 => hearth_quant::dot_q1_0g128_q8_0_ptr,
            GgmlDType::Q8_0 => hearth_quant::dot_q8_0_q8_0_ptr,
            other => {
                return Err(format!(
                    "qwen35: unsupported weight dtype {} (needs a q8 dot kernel)",
                    other.name()
                ))
            }
        };
        let w_base = entry.data().as_ptr() as usize;
        let a_ptr = q8.as_ptr() as usize;
        let out_ptr = out.as_mut_ptr() as usize;
        let row_bytes = entry.dtype.byte_size(n_cols);
        self.pool.par_dot_rows(crate::pool::DotRowsParams {
            n: n_rows,
            w_base,
            a_ptr,
            out_ptr,
            row_bytes,
            n_cols,
            dot_fn: dot,
        });
        Ok(())
    }

    /// Serial f32 matvec over a small load-dequantized weight
    /// (`ssm_alpha`/`ssm_beta`: `dt_rank` rows of `d_model`). Only ~32k MACs, so
    /// a pool dispatch would cost more than the work itself.
    fn matvec_dense(&self, w: &[f32], x: &[f32], out: &mut [f32]) {
        let n_cols = x.len();
        for (r, o) in out.iter_mut().enumerate() {
            let row = &w[r * n_cols..(r + 1) * n_cols];
            let mut acc = 0.0f32;
            for (c, v) in row.iter().enumerate() {
                acc += v * x[c];
            }
            *o = acc;
        }
    }

    // -- forward ------------------------------------------------------------
    fn forward(
        &self,
        token: u32,
        pos: usize,
        caches: &mut Qwen35Caches,
        logits: &mut [f32],
    ) -> Result<(), String> {
        let sc = &mut *self.scratch.lock().unwrap();
        let d = self.d_model;
        let eps = self.config.rms_norm_eps;
        let t_total = std::time::Instant::now();
        let mut t_stage = std::time::Instant::now();

        // Embedding (latent table: inverse Hadamard after lookup).
        let emb_rows = self.token_embd.0.n_rows();
        if token as usize >= emb_rows {
            return Err(format!("qwen35: token id {} out of range", token));
        }
        hearth_quant::dequantize(
            self.token_embd.0.dtype,
            self.token_embd.0.row_data(token as usize),
            &mut sc.hidden,
        )?;
        // Embedding (latent table): the stored rows are folded by R = D·H, so
        // the lookup inverts the fold with R⁻¹ = H·D (WHT first, then signs).
        // Verified bit-exact against the reference fork's layer-0 attn_norm.
        if self.had_embed_inverse && self.had_block_size > 0 && !self.no_had() && !self.no_emb_inv()
        {
            if std::env::var_os("HEARTH_Q35_DEBUG").is_some() && pos < 2 {
                let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
                eprintln!("[q35chk] emb_raw mean={:.6}", mean_abs(&sc.hidden));
            }
            hadamard_unrotate(&mut sc.hidden, self.signs_for(d), self.had_block_size);
            if std::env::var_os("HEARTH_Q35_DEBUG").is_some() && pos < 2 {
                let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
                eprintln!("[q35chk] emb_inv mean={:.6}", mean_abs(&sc.hidden));
            }
        }

        if self.timers_on {
            sc.timers.embed_us += t_stage.elapsed().as_micros();
        }

        let ffn_dim = self.config.d_ffn as usize;
        for layer in 0..self.layers.len() {
            let lyr = &self.layers[layer];

            // (reborrow dance: rms_norm borrows hidden immutably + normed mutably)
            let hidden_ref: &[f32] = &sc.hidden;
            let normed_ref: &mut [f32] = &mut sc.normed;
            crate::ops::rms_norm(hidden_ref, &lyr.attn_norm.0, eps, normed_ref);
            if self.timers_on {
                sc.timers.norm_us += t_stage.elapsed().as_micros();
                t_stage = std::time::Instant::now();
            }

            if layer == 0 {
                dump_layer0(pos, "attn_norm-0", &sc.normed);
            }
            if self.layer_is_gdn[layer] {
                self.gdn_layer(layer, pos, lyr, sc, caches)?;
                if self.no_gdn() {
                    for v in sc.act.iter_mut() {
                        *v = 0.0;
                    }
                }
            } else {
                self.attn_layer(layer, lyr, pos, sc, caches)?;
                if self.no_attn() {
                    for v in sc.act.iter_mut() {
                        *v = 0.0;
                    }
                }
            }

            if self.timers_on {
                sc.timers.mix_us += t_stage.elapsed().as_micros();
                t_stage = std::time::Instant::now();
            }

            // residual
            for i in 0..d {
                sc.hidden[i] += sc.act[i];
            }
            if layer == 0 {
                dump_layer0(pos, "attn_residual-0", &sc.hidden);
            }

            if std::env::var_os("HEARTH_Q35_DEBUG").is_some() && pos < 2 && layer < 6 {
                let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
                eprintln!(
                    "[q35dbg] pos={} layer={} is_gdn={} hidden_mean={:.4}",
                    pos,
                    layer,
                    self.layer_is_gdn[layer],
                    mean_abs(&sc.hidden),
                );
            }

            // sandwich norm + FFN
            let hidden_ref: &[f32] = &sc.hidden;
            let normed_ref: &mut [f32] = &mut sc.normed;
            crate::ops::rms_norm(hidden_ref, &lyr.post_attention_norm.0, eps, normed_ref);
            if self.no_ffn() {
                for v in sc.act.iter_mut() {
                    *v = 0.0;
                }
                continue;
            }
            self.mm(
                &lyr.ffn_gate,
                &sc.normed,
                &mut sc.ffn_gate,
                &mut sc.tmp_f32,
                &mut sc.proj_q8,
                true,
            )?;
            self.mm(
                &lyr.ffn_up,
                &sc.normed,
                &mut sc.ffn_up,
                &mut sc.tmp_f32,
                &mut sc.proj_q8,
                true,
            )?;
            crate::ops::silu(&mut sc.ffn_gate);
            for i in 0..ffn_dim {
                sc.ffn_gate[i] *= sc.ffn_up[i];
            }
            if layer == 0 {
                dump_layer0(pos, "attn_post_norm-0", &sc.normed);
                if pos == 0 {
                    dump_layer0(pos, "ffn_down_in-0", &sc.ffn_gate);
                }
            }
            self.mm(
                &lyr.ffn_down,
                &sc.ffn_gate,
                &mut sc.act,
                &mut sc.tmp_f32,
                &mut sc.proj_q8,
                true,
            )?;
            if layer == 0 {
                dump_layer0(pos, "post_ffn-0", &sc.act);
            }

            // second residual: hidden += ffn_branch
            for i in 0..d {
                sc.hidden[i] += sc.act[i];
            }
            if layer == 0 {
                dump_layer0(pos, "ffn_out-0", &sc.hidden);
            }
            if self.timers_on {
                sc.timers.ffn_us += t_stage.elapsed().as_micros();
                t_stage = std::time::Instant::now();
            }

            if std::env::var_os("HEARTH_Q35_DEBUG").is_some() && pos < 2 {
                let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
                eprintln!(
                    "[q35chk] post_ffn layer={} hidden_mean={:.4}",
                    layer,
                    mean_abs(&sc.hidden),
                );
            }
        }

        // (reborrow dance)
        let hidden_ref: &[f32] = &sc.hidden;
        let normed_ref: &mut [f32] = &mut sc.normed;
        crate::ops::rms_norm(hidden_ref, &self.output_norm.0, eps, normed_ref);
        // Tied head: `token_embd.weight` is the head, folded on its input dim
        // exactly like every other rotated weight, so the same `had = true`
        // matmul is correct for both the separate and the tied case.
        let out_w = self.output.as_ref().unwrap_or(&self.token_embd);
        self.mm(
            out_w,
            &sc.normed,
            logits,
            &mut sc.tmp_f32,
            &mut sc.proj_q8,
            true,
        )?;
        if self.timers_on {
            sc.timers.head_us += t_stage.elapsed().as_micros();
            sc.timers.total_us += t_total.elapsed().as_micros();
        }

        if std::env::var_os("HEARTH_Q35_DEBUG").is_some() && pos < 200 {
            let mut best = (0usize, f32::NEG_INFINITY);
            for (i, &v) in logits.iter().enumerate() {
                if v > best.1 {
                    best = (i, v);
                }
            }
            let tok_str = self.tokenizer.lock().unwrap().decode_token(best.0 as u32);
            eprintln!(
                "[q35dbg] pos={} hidden_mean={:.4} argmax_id={} ({:?}) val={:.3}",
                pos,
                sc.hidden.iter().map(|x| x.abs()).sum::<f32>() / sc.hidden.len() as f32,
                best.0,
                tok_str,
                best.1,
            );
        }
        Ok(())
    }

    // -- GDN layer ----------------------------------------------------------
    fn gdn_layer(
        &self,
        layer: usize,
        pos: usize,
        lyr: &Layer,
        sc: &mut FwdScratch,
        caches: &mut Qwen35Caches,
    ) -> Result<(), String> {
        let eps = self.config.rms_norm_eps;
        let qkv_w = lyr.attn_qkv.as_ref().ok_or("qwen35: gdn missing qkv")?;
        let gate_w = lyr.attn_gate.as_ref().ok_or("qwen35: gdn missing gate")?;
        let conv = lyr.ssm_conv1d.as_ref().ok_or("qwen35: gdn missing conv")?;
        let dt_bias = lyr
            .ssm_dt_bias
            .as_ref()
            .ok_or("qwen35: gdn missing dt bias")?;
        let ssm_a = lyr.ssm_a.as_ref().ok_or("qwen35: gdn missing a")?;
        let alpha_w = lyr.ssm_alpha.as_ref().ok_or("qwen35: gdn missing alpha")?;
        let beta_w = lyr.ssm_beta.as_ref().ok_or("qwen35: gdn missing beta")?;
        let ssm_norm = lyr
            .ssm_norm
            .as_ref()
            .ok_or("qwen35: gdn missing ssm norm")?;
        let out_w = lyr.ssm_out.as_ref().ok_or("qwen35: gdn missing out")?;

        // Projections (folded weights: rotate the activation).
        self.mm(
            qkv_w,
            &sc.normed,
            &mut sc.qkv,
            &mut sc.tmp_f32,
            &mut sc.proj_q8,
            true,
        )?;
        self.mm(
            gate_w,
            &sc.normed,
            &mut sc.zgate,
            &mut sc.tmp_f32,
            &mut sc.proj_q8,
            true,
        )?;
        if layer == 0 {
            dump_layer0(pos, "linear_attn_qkv_mixed-0", &sc.qkv);
        }
        // alpha/beta read sc.normed through small serial matvecs.
        {
            let normed_ref: &[f32] = &sc.normed;
            let abuf_ref: &mut [f32] = &mut sc.abuf;
            self.matvec_dense(alpha_w, normed_ref, abuf_ref);
            let bbuf_ref: &mut [f32] = &mut sc.bbuf;
            self.matvec_dense(beta_w, normed_ref, bbuf_ref);
        }

        // Causal depthwise conv + silu.
        caches.conv.prepare(layer, &mut sc.conv_win);
        caches.conv.push(layer, &sc.qkv, &mut sc.conv_win);
        depthwise_conv(
            &sc.conv_win,
            conv,
            self.conv_kernel,
            self.conv_dim,
            &mut sc.conv_out,
        );
        crate::ops::silu(&mut sc.conv_out);
        let q35dbg = std::env::var_os("HEARTH_Q35_DEBUG").is_some() && layer < 2;
        if q35dbg {
            let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            eprintln!(
                "[q35chk] L{} qkv_mixed mean={:.4} (ref 1.5135)",
                layer,
                mean_abs(&sc.qkv)
            );
            eprintln!(
                "[q35chk] L{} z mean={:.4} (ref 1.0743)",
                layer,
                mean_abs(&sc.zgate)
            );
            eprintln!(
                "[q35chk] L{} conv_silu mean={:.4} (ref 0.0328)",
                layer,
                mean_abs(&sc.conv_out)
            );
        }

        let k_dim = self.k_dim;
        let hd = self.state_size;
        let q_conv = &sc.conv_out[..k_dim];
        let k_conv = &sc.conv_out[k_dim..2 * k_dim];
        let v_conv = &sc.conv_out[2 * k_dim..];

        // l2-normalize q and k per head (heads are adjacent groups of the conv
        // output: q heads then k heads, each n_groups wide).
        {
            let normed_qk = &mut sc.tmp_f32[..2 * k_dim];
            normed_qk[..k_dim].copy_from_slice(q_conv);
            normed_qk[k_dim..].copy_from_slice(k_conv);
            for h in 0..2 * self.n_groups {
                let s = h * hd;
                let mut ss = 0.0f32;
                for v in &normed_qk[s..s + hd] {
                    ss += v * v;
                }
                let inv = 1.0 / (ss + eps).sqrt();
                for v in &mut normed_qk[s..s + hd] {
                    *v *= inv;
                }
            }
        }
        let (q_n, k_n) = sc.tmp_f32[..2 * k_dim].split_at(k_dim);
        if q35dbg {
            let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            eprintln!(
                "[q35chk] L{} qk_l2 mean={:.4} (ref 0.0424)",
                layer,
                mean_abs(q_n)
            );
        }

        // Gates.
        let n_v = self.dt_rank;
        for h in 0..n_v {
            sc.gdec[h] = softplus(sc.abuf[h] + dt_bias[h]) * ssm_a[h];
            sc.beta[h] = 1.0 / (1.0 + (-sc.bbuf[h]).exp());
        }
        if layer == 0 {
            dump_layer0(pos, "beta-0", &sc.bbuf);
        }

        // Recurrence per v-head. Head mapping follows ggml's tiling semantics:
        // v-head h pairs with k-group h % n_groups (ggml_repeat tiles the head
        // axis), and the trained alpha/beta rows are indexed by the same h.
        let scale_q = 1.0 / (hd as f32).sqrt();
        let state = caches.gdn.layer_slice(layer);
        for h in 0..n_v {
            let kg = h % self.n_groups;
            let decay = sc.gdec[h].exp();
            let b = sc.beta[h];
            let q_off = kg * hd;
            let k_off = kg * hd;
            let v_off = h * hd;
            let s_base = h * hd * hd;

            // delta_j = v[j] - sum_i S[i][j]*k[i]  (decay applies to S first)
            for j in 0..hd {
                let mut dot = 0.0f32;
                let col = s_base + j;
                for i in 0..hd {
                    dot += state[col + i * hd] * k_n[k_off + i];
                }
                let bd = b * (v_conv[v_off + j] - dot);
                for i in 0..hd {
                    state[col + i * hd] = state[col + i * hd] * decay + k_n[k_off + i] * bd;
                }
            }
            // o_j = sum_i S[i][j] * q[i] * scale (tiled head order, matching ggml)
            let out_off = h * hd;
            for j in 0..hd {
                let mut acc = 0.0f32;
                let col = s_base + j;
                for i in 0..hd {
                    acc += state[col + i * hd] * q_n[q_off + i];
                }
                sc.attn_out[out_off + j] = acc * scale_q;
            }
        }

        if q35dbg {
            let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            eprintln!(
                "[q35chk] L{} gdec mean={:.4} (ref 0.4829)",
                layer,
                mean_abs(&sc.gdec)
            );
            eprintln!(
                "[q35chk] L{} beta mean={:.4} (ref 0.7268)",
                layer,
                mean_abs(&sc.beta)
            );
            eprintln!(
                "[q35chk] L{} attn_out mean={:.6} (ref 0.0005)",
                layer,
                mean_abs(&sc.attn_out)
            );
        }

        // Gated RMS norm: silu(z) * rmsnorm(o); the norm weight (head_v_dim wide)
        // is shared across all v-heads.
        for h in 0..n_v {
            let s = h * hd;
            crate::ops::rms_norm(
                &sc.attn_out[s..s + hd],
                &ssm_norm.0,
                eps,
                &mut sc.normed_out[s..s + hd],
            );
        }
        crate::ops::silu(&mut sc.zgate);
        for i in 0..self.v_dim {
            sc.normed_out[i] *= sc.zgate[i];
        }
        if q35dbg {
            let mean_abs = |v: &[f32]| v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32;
            eprintln!(
                "[q35chk] L{} final_out mean={:.4} (ref 0.0110)",
                layer,
                mean_abs(&sc.normed_out)
            );
        }
        if layer == 0 {
            dump_layer0(pos, "final_output-0", &sc.normed_out);
        }

        // ssm_out is Hadamard-folded with gdn_v_grouped: the activation is in
        // tiled head order [hd, nk, rep] and must be permuted to the grouped
        // order [hd, rep, nk] the fold was computed in, before signs + WHT.
        if self.had_block_size > 0 && !self.no_had() {
            let nk = self.n_groups;
            let rep = n_v / nk;
            let perm: &mut [f32] = &mut sc.tmp_f32[..self.v_dim];
            if rep > 1 {
                for h in 0..n_v {
                    let kg = h % nk;
                    let r = h / nk;
                    let dst = (kg * rep + r) * hd;
                    perm[dst..dst + hd].copy_from_slice(&sc.normed_out[h * hd..(h + 1) * hd]);
                }
            } else {
                perm.copy_from_slice(&sc.normed_out);
            }
            hadamard_rotate(perm, self.signs_for(self.v_dim), self.had_block_size);
            // quantize_q8_0 APPENDS to the Vec, so the scratch must be cleared
            // first — otherwise the kernel dots against a stale activation from
            // an earlier (different-width) matmul.
            sc.proj_q8.clear();
            hearth_quant::quantize_q8_0(perm, &mut sc.proj_q8);
            self.mm_q8(out_w, &sc.proj_q8, &mut sc.act)?;
            if layer == 0 {
                dump_layer0(pos, "linear_attn_out-0", &sc.act);
            }
            return Ok(());
        }

        // Non-folded (or Q2_0-style) ssm_out: ordinary rotated matmul.
        self.mm(
            out_w,
            &sc.normed_out,
            &mut sc.act,
            &mut sc.tmp_f32,
            &mut sc.proj_q8,
            true,
        )?;
        Ok(())
    }

    // -- gated full attention layer ------------------------------------------
    fn attn_layer(
        &self,
        layer: usize,
        lyr: &Layer,
        pos: usize,
        sc: &mut FwdScratch,
        caches: &mut Qwen35Caches,
    ) -> Result<(), String> {
        let eps = self.config.rms_norm_eps;
        let hd = self.head_dim;
        let n_heads = self.n_heads;
        let n_kv = self.n_kv_heads;
        let nq = n_heads * hd;
        let _ = n_kv;

        let qw = lyr.attn_q.as_ref().ok_or("qwen35: attn missing q")?;
        let kw = lyr.attn_k.as_ref().ok_or("qwen35: attn missing k")?;
        let vw = lyr.attn_v.as_ref().ok_or("qwen35: attn missing v")?;
        let q_norm = lyr
            .attn_q_norm
            .as_ref()
            .ok_or("qwen35: attn missing q norm")?;
        let k_norm = lyr
            .attn_k_norm
            .as_ref()
            .ok_or("qwen35: attn missing k norm")?;
        let ow = lyr
            .attn_output
            .as_ref()
            .ok_or("qwen35: attn missing output")?;

        // Q projection emits query + gate interleaved per head: [q_h | gate_h].
        self.mm(
            qw,
            &sc.normed,
            &mut sc.q_buf,
            &mut sc.tmp_f32,
            &mut sc.proj_q8,
            true,
        )?;
        self.mm(
            kw,
            &sc.normed,
            &mut sc.k_buf,
            &mut sc.tmp_f32,
            &mut sc.proj_q8,
            true,
        )?;
        self.mm(
            vw,
            &sc.normed,
            &mut sc.v_buf,
            &mut sc.tmp_f32,
            &mut sc.proj_q8,
            true,
        )?;

        // Split q|gate. Per head ascending: copy gate out first (its source
        // region is untouched by q compaction of this or earlier heads), then
        // compact q to the front of q_buf.
        for h in 0..n_heads {
            let src = h * 2 * hd;
            sc.attn_gate[h * hd..(h + 1) * hd].copy_from_slice(&sc.q_buf[src + hd..src + 2 * hd]);
            sc.q_buf.copy_within(src..src + hd, h * hd);
        }

        // Head norms.
        let mut tmp = vec![0.0f32; hd];
        for h in 0..n_heads {
            let s = h * hd;
            crate::ops::rms_norm(&sc.q_buf[s..s + hd], &q_norm.0, eps, &mut tmp);
            sc.q_buf[s..s + hd].copy_from_slice(&tmp);
        }
        for h in 0..n_kv {
            let s = h * hd;
            crate::ops::rms_norm(&sc.k_buf[s..s + hd], &k_norm.0, eps, &mut tmp);
            sc.k_buf[s..s + hd].copy_from_slice(&tmp);
        }

        // Partial RoPE (MRoPE sections collapse to standard RoPE for text).
        for h in 0..n_heads {
            let s = h * hd;
            self.rope_cache
                .apply(&mut sc.q_buf[s..s + self.rope_dim], pos);
        }
        for h in 0..n_kv {
            let s = h * hd;
            self.rope_cache
                .apply(&mut sc.k_buf[s..s + self.rope_dim], pos);
        }

        // KV cache.
        let ai = self.attn_index[layer];
        let cache = &mut caches.kv[ai];
        for h in 0..n_kv {
            cache.write_kv(
                pos,
                h,
                &sc.k_buf[h * hd..(h + 1) * hd],
                &sc.v_buf[h * hd..(h + 1) * hd],
            );
        }

        // GQA attention per q head.
        let kv_repeat = n_heads / n_kv;
        let scale = 1.0 / (hd as f32).sqrt();
        let seq_len = cache.current_len;
        sc.scores.resize(seq_len, 0.0);
        for qh in 0..n_heads {
            let kvh = qh / kv_repeat;
            let qs = qh * hd;
            let ks = cache.k_slice(kvh, seq_len);
            let vs = cache.v_slice(kvh, seq_len);
            for t in 0..seq_len {
                let koff = t * hd;
                let mut acc = 0.0f32;
                for i in 0..hd {
                    acc += sc.q_buf[qs + i] * ks[koff + i];
                }
                sc.scores[t] = acc * scale;
            }
            crate::ops::softmax(&mut sc.scores);
            for i in 0..hd {
                let mut acc = 0.0f32;
                for t in 0..seq_len {
                    acc += sc.scores[t] * vs[t * hd + i];
                }
                sc.attn_out[qs + i] = acc;
            }
        }

        // Gating: attn_out * sigmoid(gate).
        for i in 0..nq {
            sc.attn_out[i] /= 1.0 + (-sc.attn_gate[i]).exp();
        }

        self.mm(
            ow,
            &sc.attn_out[..nq],
            &mut sc.act,
            &mut sc.tmp_f32,
            &mut sc.proj_q8,
            true,
        )?;
        Ok(())
    }

    pub fn new_caches(&self) -> Qwen35Caches {
        let n_attn = self.layer_is_gdn.iter().filter(|g| !**g).count();
        Qwen35Caches {
            kv: (0..n_attn)
                .map(|_| KVCache::new(self.n_kv_heads, self.head_dim, self.max_seq))
                .collect(),
            gdn: GdnState::new(self.layers.len(), self.dt_rank, self.state_size),
            conv: ConvState::new(self.layers.len(), self.conv_dim, self.conv_kernel),
        }
    }

    // -- generation -----------------------------------------------------------
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
        let vocab = self.config.vocab_size as usize;

        let mut input_ids = {
            let mut tok = self.tokenizer.lock().unwrap();
            tok.encode(prompt, false)
        };
        if input_ids.is_empty() {
            return Err("Empty prompt after tokenization".into());
        }

        // Effective window for this request: the caller's context setting when
        // present, otherwise the model's capped KV size.
        let ctx_limit = match max_context_len {
            Some(n) if n > 0 => n.min(self.max_seq),
            _ => self.max_seq,
        };
        if input_ids.len() > ctx_limit {
            let drop = input_ids.len() - ctx_limit;
            input_ids.drain(..drop);
        }

        let mut caches = self.new_caches();
        let mut logits = vec![0.0f32; vocab];
        let mut sample_buf = vec![0.0f32; vocab];
        let mut past_tokens: Vec<u32> = Vec::new();
        let prompt_tokens = input_ids.len();
        let mut total_tokens = 0usize;
        let greedy = sampler_config.temperature == 0.0;

        // Prefill: token by token (GDN recurrence is inherently sequential).
        self.timers_reset();
        for (i, &tid) in input_ids.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            self.forward(tid, i, &mut caches, &mut logits)?;
        }
        self.timers_report("prefill", prompt_tokens);
        self.timers_reset();
        // `total_tokens` counts generated tokens only: the prompt is reported
        // separately so a short reply from a long prompt is not misreported as
        // a long reply (which is exactly what made "28 tokens generated, no
        // visible text" look like a model failure).
        past_tokens.push(input_ids[input_ids.len() - 1]);

        let mut generated = 0usize;
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
            if pos >= ctx_limit {
                for c in caches.kv.iter_mut() {
                    c.truncate_left(64);
                }
            }

            let token = if greedy {
                // argmax
                let mut best = 0usize;
                let mut bv = f32::NEG_INFINITY;
                for (i, &v) in logits.iter().enumerate() {
                    if v > bv {
                        bv = v;
                        best = i;
                    }
                }
                best as u32
            } else {
                sample_buf.copy_from_slice(&logits);
                hearth_sampler::sample(&mut sample_buf, sampler_config, &past_tokens)
            };

            let stop = {
                let tok = self.tokenizer.lock().unwrap();
                if token == tok.eos_id || tok.is_control_token(token) {
                    true
                } else {
                    if std::env::var_os("HEARTH_Q35_TOKEN_DEBUG").is_some() {
                        eprintln!(
                            "[q35tok] id={} raw={:?} dec={:?}",
                            token,
                            tok.vocab
                                .get(token as usize)
                                .map(|s| s.as_str())
                                .unwrap_or(""),
                            tok.decode_token(token),
                        );
                    }
                    let text = tok.decode_token(token);
                    let _ = output.send(EngineOutput::TextToken(text.to_string()));
                    false
                }
            };
            if stop {
                break;
            }

            past_tokens.push(token);
            generated += 1;
            total_tokens += 1;

            self.forward(token, pos, &mut caches, &mut logits)?;
        }

        self.timers_report("decode", generated);

        // Emit any multi-byte character still held by the streaming decoder
        // (e.g. the reply's final character was cut mid-sequence).
        let flush = self.tokenizer.lock().unwrap().flush_stream_decode();
        if !flush.is_empty() {
            let _ = output.send(EngineOutput::TextToken(flush));
        }
        let elapsed = start.elapsed().as_millis();
        let tokens_per_s = if elapsed > 0 {
            total_tokens as f32 / (elapsed as f32 / 1000.0)
        } else {
            0.0
        };
        let _ = output.send(EngineOutput::Done(RunStats {
            elapsed_ms: elapsed,
            tokens: total_tokens,
            prompt_tokens,
            tokens_per_s,
        }));
        Ok(())
    }
}

// helpers -------------------------------------------------------------------

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Causal depthwise conv over the time window. Weight layout is [channel][tap]
/// (matching ggml_ssm_conv: c[i0 + i1*nc] where i0 = tap, i1 = channel), and the
/// window uses the same [channel][tap] layout with tap 0 = oldest.
fn depthwise_conv(window: &[f32], w: &[f32], kernel: usize, conv_dim: usize, out: &mut [f32]) {
    for (c, slot) in out.iter_mut().enumerate().take(conv_dim) {
        let wo = c * kernel;
        let mut acc = 0.0f32;
        for k in 0..kernel {
            acc += w[wo + k] * window[wo + k];
        }
        *slot = acc;
    }
}

// ---------------------------------------------------------------------------
// Integration glue: qwen35 files load as a LlamaModel that wraps a Qwen35Core.
// ---------------------------------------------------------------------------

pub fn load(path: &str, _strategy: hearth_core::LoadStrategy) -> Result<LlamaModel, String> {
    let gguf = GgufFile::open(path).map_err(|e| format!("GGUF: {}", e))?;
    let tokenizer = Tokenizer::from_gguf(&gguf).map_err(|e| format!("Tokenizer: {}", e))?;
    let config = ModelConfig::from_gguf(&gguf)?;
    // The core re-parses its own tokenizer instance (parser is cheap, metadata only).
    let core = Qwen35Core::from_gguf(
        &gguf,
        Tokenizer::from_gguf(&gguf).map_err(|e| format!("Tokenizer: {}", e))?,
    )?;
    // Which kernel each weight format will use, for the `[kernel]` load banner
    // and the service's `kernels` JSON. Captured here because the shell that is
    // returned does not retain the tensor map.
    let dispatch = hearth_quant::dispatch_report(gguf.tensors.iter().map(|t| t.dtype));
    Ok(LlamaModel::from_hybrid(core, config, tokenizer, dispatch))
}

// ---------------------------------------------------------------------------
// Model trait impl
// ---------------------------------------------------------------------------

impl Model for Qwen35Core {
    fn run(&self, request: PipelineRequest, cancel: Arc<AtomicBool>, output: Sender<EngineOutput>) {
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
            _ => Err("Unsupported request type for qwen35 model".to_string()),
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
