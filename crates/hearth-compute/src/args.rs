//! Parameter objects for the GPU dispatch methods.
//!
//! Every one of these exists because a method's argument list had grown past what a reader can
//! hold: a dispatch like `flash_attention` was taking two buffers plus eight numbers, and the
//! call site was a positional list where the only thing distinguishing `seq_len_q` from
//! `seq_len_kv` was the order they happened to be written in. The split is always the same —
//! **buffers stay explicit** (they are the things whose ownership and aliasing matter, and a
//! struct that hid them would hide the interesting part), and **the numbers move into a named,
//! `Copy` shape** that a call site can construct with field names.
//!
//! These are plain data. Nothing here validates anything: a shape is a faithful transcription of
//! what the caller knows, and the kernels are where a nonsensical one becomes a wrong answer.
//! Fields keep the names their call sites used, so a diff that moves an argument into a struct
//! reads as a move and not as a rename.

/// Shapes for a single-token attention dispatch over a KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashAttnShape {
    /// Byte offset of this layer's cache inside the flat KV buffer.
    pub kv_offset: u64,
    /// Total size of the flat KV buffer, for the bounds the kernel needs.
    pub kv_total_bytes: u64,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    /// Query positions being computed (1 for decode).
    pub seq_len_q: u32,
    /// Positions already in the cache.
    pub seq_len_kv: u32,
    /// The position this token occupies.
    pub pos: u32,
}

/// Shapes for a prefill attention dispatch over a KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillAttnShape {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub seq_len_q: u32,
    /// Elements between one head's cache rows in the flat buffer.
    pub cache_stride: u32,
}

/// Shapes for attention with no cache at all, where K and V are full tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BidirAttnShape {
    pub n_heads: u32,
    pub head_dim: u32,
    pub batch: u32,
    pub seq_len: u32,
}

/// Shapes for a batched RMS norm over a whole sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormShape {
    pub seq_len: u32,
    pub d: u32,
    pub eps: f32,
    /// Gemma-style post-norm (weight applied after the scale) rather than the usual pre-norm.
    pub post_norm: bool,
}

/// Shapes for a batched RMS norm applied per attention head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeadNormShape {
    pub n_heads: u32,
    pub head_dim: u32,
    pub seq_len: u32,
    pub eps: f32,
    pub post_norm: bool,
}

/// Everything about RoPE that does not depend on which position is being rotated.
///
/// Split from the position on purpose: the configuration is a property of the model and the
/// position is a property of the call, and keeping them in one struct meant every dispatch had
/// to carry both even when one of them was a constant for the whole run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeConfig {
    pub n_heads: u32,
    pub head_dim: u32,
    /// The number of dims the rotation is applied to, which the shader uses as its pairing stride.
    pub rope_dim: u32,
    pub theta: f32,
    pub freq_scale: f32,
    pub yarn_corr_low: f32,
    pub yarn_corr_high: f32,
    pub yarn_mscale: f32,
    pub yarn_n_dims: f32,
}

/// Shapes for writing one position's K and V into the flat cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheWriteShape {
    pub pos: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub cache_stride: u32,
}

/// Shapes for writing a whole sequence of K and V into the flat cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheWriteBatchShape {
    pub seq_len: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub cache_stride: u32,
}

/// Shapes for a single-token fused gate/up projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlpShape {
    pub m: u32,
    pub n: u32,
}

/// Shapes for a batched fused gate/up projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MlpBatchShape {
    pub ffn_dim: u32,
    pub k: u32,
    pub seq_len: u32,
}

/// The seven buffers a fused Q/K/V projection reads and writes.
///
/// A struct rather than seven arguments because the seven are one object in every sense that
/// matters: they are always passed together, they are always in the same roles, and a positional
/// call site put `qw` and `q_out` four parameters apart with three lookalikes between them.
#[derive(Debug, Clone, Copy)]
pub struct FusedQkvBuffers<'a> {
    pub qw: &'a crate::GpuBuffer,
    pub kw: &'a crate::GpuBuffer,
    pub vw: &'a crate::GpuBuffer,
    pub x: &'a crate::GpuBuffer,
    pub q_out: &'a crate::GpuBuffer,
    pub k_out: &'a crate::GpuBuffer,
    pub v_out: &'a crate::GpuBuffer,
}

/// Shapes for a single-token fused Q/K/V projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// Shapes for a batched fused Q/K/V projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QkvBatchShape {
    pub nq: u32,
    pub nkv: u32,
    pub k: u32,
    pub seq_len: u32,
}

/// Shapes for a 3×3 convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvShape {
    pub c_in: u32,
    pub c_out: u32,
    pub h: u32,
    pub w: u32,
}

/// Shapes for a group norm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupNormShape {
    pub c: u32,
    /// Height times width: the norm is taken over each group's spatial extent.
    pub hw: u32,
    pub n_groups: u32,
    pub eps: f32,
}
