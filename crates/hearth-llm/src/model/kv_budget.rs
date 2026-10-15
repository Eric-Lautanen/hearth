//! How large a context the engine allocates at load time.
//!
//! The KV cache is allocated once, at load, at the full context ceiling — not
//! per request — so a model whose metadata declares a huge context can sink a
//! machine before it generates a token. The 27B hybrid is the acute case: 16
//! attention layers × 4 KV heads × head_dim 256 × 2 (K and V) × 4 B =
//! **128 KiB per token**, so its declared 262144-token context asks for 32 GiB.
//! Bounded by a byte budget instead, it lands on a context that fits alongside
//! the 6 GB of weights.
//!
//! `HEARTH_CTX_CAP` overrides the computed ceiling in either direction (raise it
//! on a machine with RAM to spare), still bounded by what the model declares.

/// Default ceiling for the f32 KV cache allocation: 1.25 GiB. Chosen so every
/// dense Bonsai model keeps the 4096-token context it has always been run and
/// benchmarked at (an 8B dense model needs 288 KiB per token, so 4096 costs
/// 1.125 GiB), while the 27B hybrid — which has no historic cap worth keeping —
/// gets a context that fits next to its weights.
pub const DEFAULT_KV_BUDGET_BYTES: u64 = 1280 << 20; // 1.25 GiB

/// Dense models are tuned, benchmarked and documented at this context; the
/// budget only lowers it. An explicit `HEARTH_CTX_CAP` replaces both.
pub const DENSE_CTX_CAP: usize = 4096;

/// Never allocate less than this from the budget, however expensive a token is —
/// a model that cannot hold a short conversation is not usable at all.
pub const MIN_CTX: usize = 512;

/// Bytes of f32 KV cache one token costs: K and V, per KV head, per layer that
/// actually keeps a cache. `n_kv_layers` is the number of *attention* layers
/// (the 27B hybrid's 48 gated-delta layers keep a fixed-size recurrence instead).
pub fn kv_bytes_per_token(n_kv_layers: usize, n_kv_heads: usize, head_dim: usize) -> u64 {
    2 * n_kv_layers as u64 * n_kv_heads as u64 * head_dim as u64 * 4
}

/// The context ceiling to allocate for a model.
///
/// * `model_declared` — the context the GGUF advertises (often far larger than
///   the machine can hold).
/// * `legacy_cap` — the context this architecture has always been run at when
///   the model declares more ([`DENSE_CTX_CAP`] for dense models, `None` for the
///   hybrid, which has no such history to preserve). The byte budget can only
///   lower it.
pub fn ctx_cap(
    model_declared: usize,
    n_kv_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    legacy_cap: Option<usize>,
) -> usize {
    let per_token = kv_bytes_per_token(n_kv_layers, n_kv_heads, head_dim).max(1);
    let by_budget = ((DEFAULT_KV_BUDGET_BYTES / per_token) as usize).max(MIN_CTX);
    let default_cap = match legacy_cap {
        Some(c) => c.min(by_budget),
        None => by_budget,
    };
    model_declared.min(env_ctx_cap().unwrap_or(default_cap))
}

/// `HEARTH_CTX_CAP`, when set to a positive integer.
fn env_ctx_cap() -> Option<usize> {
    std::env::var("HEARTH_CTX_CAP")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|n| *n > 0)
}

/// Human-readable size, for the load banner and the UI.
pub fn fmt_bytes(bytes: u64) -> String {
    if bytes >= 1 << 30 {
        format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
    } else if bytes >= 1 << 20 {
        format!("{:.0} MiB", bytes as f64 / (1u64 << 20) as f64)
    } else {
        format!("{:.0} KiB", bytes as f64 / 1024.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense() -> (usize, usize, usize) {
        (28, 8, 128) // Bonsai 1.7B
    }

    /// The models that already work must not move: 224 KiB per token, still 4096.
    #[test]
    fn dense_models_keep_their_context() {
        let (l, h, d) = dense();
        assert_eq!(kv_bytes_per_token(l, h, d), 229_376); // 224 KiB per token
        assert_eq!(ctx_cap(4096, l, h, d, Some(DENSE_CTX_CAP)), 4096);
        // Declaring more than the legacy cap does not raise it.
        assert_eq!(ctx_cap(32768, l, h, d, Some(DENSE_CTX_CAP)), 4096);
    }

    /// The 8B dense model is the worst per-token case that must stay at 4096:
    /// 36 layers × 8 kv heads × 128 × 2 × 4 = 288 KiB per token (1.125 GiB).
    #[test]
    fn expensive_dense_model_still_fits_the_budget() {
        let per_token = kv_bytes_per_token(36, 8, 128);
        assert_eq!(per_token, 294_912);
        let cap = ctx_cap(32768, 36, 8, 128, Some(DENSE_CTX_CAP));
        assert_eq!(cap, 4096);
        assert!(cap as u64 * per_token <= DEFAULT_KV_BUDGET_BYTES);
    }

    /// The regression this module exists for: the 27B hybrid declared 262144
    /// tokens, which is 32 GiB of f32 KV cache at 128 KiB per token.
    #[test]
    fn twenty_seven_b_gets_a_context_that_fits() {
        let per_token = kv_bytes_per_token(16, 4, 256);
        assert_eq!(per_token, 131_072); // 128 KiB per token
        assert!(
            262_144u64 * per_token > 30 * (1 << 30),
            "declared KV is 32 GiB"
        );
        let cap = ctx_cap(262_144, 16, 4, 256, None);
        assert_eq!(cap, (DEFAULT_KV_BUDGET_BYTES / per_token) as usize);
        assert!(cap as u64 * per_token <= DEFAULT_KV_BUDGET_BYTES);
        // Half the RAM the broken 32768 cap allocated, and 5x its context.
        assert!(cap > 2048);
    }

    /// A model so expensive that the budget allows almost nothing is floored at
    /// MIN_CTX rather than allocated a context it cannot converse in.
    #[test]
    fn absurdly_expensive_kv_is_floored() {
        assert_eq!(kv_bytes_per_token(64, 32, 1024), 16 << 20); // 16 MiB per token
        assert_eq!(ctx_cap(1 << 20, 64, 32, 1024, None), MIN_CTX);
    }

    #[test]
    fn bytes_are_formatted_readably() {
        assert_eq!(fmt_bytes(229_376), "224 KiB");
        assert_eq!(fmt_bytes(294_912), "288 KiB");
        assert_eq!(fmt_bytes(1 << 20), "1 MiB");
        assert_eq!(fmt_bytes(DEFAULT_KV_BUDGET_BYTES), "1.25 GiB");
    }
}
