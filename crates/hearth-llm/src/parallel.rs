use rayon::prelude::*;
use std::sync::OnceLock;

fn num_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| rayon::current_num_threads().max(1))
}

/// Parallel for-each using rayon::broadcast for minimal per-call overhead.
/// Each thread processes its own static chunk — no work-stealing needed for
/// balanced matmul rows.
pub fn par_for_static(n: usize, f: impl Fn(usize) + Send + Sync) {
    if n == 0 {
        return;
    }
    let nt = num_threads();
    if nt <= 1 || n <= 1 {
        for i in 0..n {
            f(i);
        }
        return;
    }
    let f = &f;
    rayon::broadcast(|ctx| {
        let chunk = n.div_ceil(nt);
        let start = ctx.index() * chunk;
        let end = (start + chunk).min(n);
        for i in start..end {
            f(i);
        }
    });
}
