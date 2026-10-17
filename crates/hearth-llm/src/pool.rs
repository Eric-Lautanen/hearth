use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

pub type DotFn = unsafe fn(*const u8, *const u8, usize) -> f32;
/// Paired-token dot: one weight row against two activations, returning both results.
/// Lets prefill expand each weight row once for two prompt tokens.
pub type DotFn2 = unsafe fn(*const u8, *const u8, *const u8, usize) -> (f32, f32);
pub type ParForFn = unsafe fn(worker_id: usize, begin: usize, end: usize, ctx: usize);

unsafe fn dummy_dot_fn(_w: *const u8, _a: *const u8, _n: usize) -> f32 {
    0.0
}

pub struct WorkParams {
    pub n: usize,
    pub seq_len: usize,
    pub q8_stride: usize,
    pub w_base: usize,
    pub a_ptr: usize,
    pub out_ptr: usize,
    pub row_bytes: usize,
    pub n_cols: usize,
    pub dot_fn: DotFn,
    pub dot_fn2: Option<DotFn2>,
    pub tile_size: usize,
    pub is_quantize: bool,
    pub is_par_for: bool,
    pub par_for_fn: ParForFn,
    pub par_for_ctx: usize,
}

/// Rows x tokens dot product over one pool chunk of rows `begin..end`.
///
/// `out` is laid out token-major (`out[s * n + row]`). The weight rows are walked
/// in L2-sized tiles with the token loop inside so a tile stays cached across the
/// whole prompt; when a paired kernel is available each weight tile is expanded
/// once for two tokens instead of once per token.
unsafe fn batched_rows_chunk(wp: &WorkParams, begin: usize, end: usize) {
    if begin >= end {
        return;
    }
    let seq_len = wp.seq_len.max(1);
    let n = wp.n;
    let mut r = begin;
    while r < end {
        let tile_end = (r + wp.tile_size).min(end);
        let mut s = 0;
        if let Some(f2) = wp.dot_fn2 {
            while s + 1 < seq_len {
                let a0 = (wp.a_ptr + s * wp.q8_stride) as *const u8;
                let a1 = (wp.a_ptr + (s + 1) * wp.q8_stride) as *const u8;
                let out0 = s * n;
                let out1 = out0 + n;
                for row in r..tile_end {
                    let (v0, v1) = f2(
                        (wp.w_base + row * wp.row_bytes) as *const u8,
                        a0,
                        a1,
                        wp.n_cols,
                    );
                    *((wp.out_ptr + (row + out0) * 4) as *mut f32) = v0;
                    *((wp.out_ptr + (row + out1) * 4) as *mut f32) = v1;
                }
                s += 2;
            }
        }
        while s < seq_len {
            let a = (wp.a_ptr + s * wp.q8_stride) as *const u8;
            let out_off = s * n;
            for row in r..tile_end {
                *((wp.out_ptr + (row + out_off) * 4) as *mut f32) =
                    (wp.dot_fn)((wp.w_base + row * wp.row_bytes) as *const u8, a, wp.n_cols);
            }
            s += 1;
        }
        r = tile_end;
    }
}

struct Worker {
    handle: Option<thread::JoinHandle<()>>,
    done_flag: Arc<AtomicBool>,
}

/// Spin iterations before a worker stops looking for work, and before the
/// dispatching thread stops spinning on a worker and yields instead. A dispatch
/// normally follows the previous one within microseconds while tokens are being
/// produced, so the spin catches almost every dispatch without a syscall.
const SPIN_BUDGET: u64 = 65536;

/// How long an idle worker sleeps before it double-checks for work. Only a
/// backstop for the notify path: a worker is woken directly by `dispatch`.
const IDLE_WAIT: Duration = Duration::from_millis(100);

/// Machine-derived worker count for the **dense** forward path.
///
/// Honors `HEARTH_THREADS` first. Otherwise it uses the *physical* core count:
/// the dots are memory-bandwidth bound, so SMT siblings mostly steal bandwidth
/// from each other and from the spinning main thread. Backing off from the
/// logical count is not enough, because the loss is a cliff rather than a slope.
/// Measured on an 8C/16T box, the dense 8B Q1_0 runs 12.1 tok/s at 8 workers and
/// 12.0 at 10, but 6.0 at 12 and 4.0 at 16. `std` only reports the logical count,
/// so the SMT width comes from CPUID on x86_64 and the logical count is the
/// fallback everywhere else.
///
/// The hybrid path keeps its own, *higher* count on purpose: it was measured to
/// want more threads than physical cores on the same box, so one shared number
/// would cost about 2x on whichever path it did not suit. See the comment at its
/// `ThreadPool::new` in `model/qwen35.rs`.
pub fn default_worker_count() -> usize {
    if let Ok(v) = std::env::var("HEARTH_THREADS") {
        if let Some(n) = v.parse::<usize>().ok().filter(|n| *n > 0) {
            return n;
        }
    }
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    physical_cores(logical)
}

/// Physical (non-SMT) core count, when the CPU will say.
///
/// Reads the hardware threads per core from the SMT topology leaf. A part
/// without SMT, or an unhelpful hypervisor, reports 1, which leaves the logical
/// count in place.
#[cfg(target_arch = "x86_64")]
fn physical_cores(logical: usize) -> usize {
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    // Leaf 0x1F is the modern extended-topology leaf and 0xB is its older
    // equivalent; sub-leaf 0's EBX[15:0] is the thread count per core. The
    // topology level in ECX is ignored deliberately: sub-leaf 0 is the SMT
    // level on every shipping part.
    let max_leaf = __cpuid(0).eax;
    let leaf = if max_leaf >= 0x1f {
        0x1f
    } else if max_leaf >= 0x0b {
        0x0b
    } else {
        return logical;
    };
    let threads_per_core = (__cpuid_count(leaf, 0).ebx & 0xffff) as usize;
    if threads_per_core >= 2 && threads_per_core <= logical {
        (logical / threads_per_core).max(1)
    } else {
        logical
    }
}

/// Non-x86_64 hosts do not expose SMT width through a portable CPU query, so
/// the logical count stands (Apple silicon and most ARM parts have no SMT).
#[cfg(not(target_arch = "x86_64"))]
fn physical_cores(logical: usize) -> usize {
    logical
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    params: Arc<AtomicPtr<WorkParams>>,
    _params_box: Box<WorkParams>,
    gen: Arc<AtomicU64>,
    /// Idle workers block here instead of spinning. A spin-then-`yield_now`
    /// loop never actually blocks — `yield_now` returns immediately when the
    /// machine has nothing else to run — so every worker sat at ~100% of a core
    /// while the process idled (measured: an idle service with the 27B loaded
    /// burned 414% of one core doing nothing).
    wait: Arc<(Mutex<()>, Condvar)>,
    /// How many workers are asleep. A dispatch only pays for a wake-up call when
    /// it has to; a pool that is still spinning sees the new generation for free.
    parked: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    num_threads: usize,
}

/// Row-parallel dot-product dispatch parameters. Grouped so the pool API
/// stays under the argument-count lint as dispatch variants grow.
#[derive(Clone, Copy)]
pub struct DotRowsParams {
    pub n: usize,
    pub w_base: usize,
    pub a_ptr: usize,
    pub out_ptr: usize,
    pub row_bytes: usize,
    pub n_cols: usize,
    pub dot_fn: DotFn,
}

/// Batched (prefill) variant: one weight pass serves `seq_len` activation rows.
#[derive(Clone, Copy)]
pub struct BatchedDotRowsParams {
    pub n: usize,
    pub seq_len: usize,
    pub q8_stride: usize,
    pub w_base: usize,
    pub a_ptr: usize,
    pub out_ptr: usize,
    pub row_bytes: usize,
    pub n_cols: usize,
    pub dot_fn: DotFn,
    pub dot_fn2: Option<DotFn2>,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut params_box = Box::new(WorkParams {
            n: 0,
            seq_len: 1,
            q8_stride: 0,
            w_base: 0,
            a_ptr: 0,
            out_ptr: 0,
            row_bytes: 0,
            n_cols: 0,
            dot_fn: dummy_dot_fn,
            dot_fn2: None,
            tile_size: usize::MAX,
            is_quantize: false,
            is_par_for: false,
            par_for_fn: |_, _, _, _| {},
            par_for_ctx: 0,
        });
        let params = Arc::new(AtomicPtr::new(&mut *params_box));
        let gen = Arc::new(AtomicU64::new(0));
        let wait = Arc::new((Mutex::new(()), Condvar::new()));
        let parked = Arc::new(AtomicUsize::new(0));

        let mut workers = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            let done = Arc::new(AtomicBool::new(false));
            let done_clone = done.clone();
            let sd = shutdown.clone();
            let p = params.clone();
            let g = gen.clone();
            let nt = num_threads;
            let wait_worker = wait.clone();
            let parked_worker = parked.clone();
            let handle = thread::spawn(move || {
                let mut local_gen = 0u64;
                let mut spins = 0u64;
                loop {
                    let cur_gen = g.load(Ordering::Acquire);
                    if cur_gen != local_gen {
                        local_gen = cur_gen;
                        spins = 0;
                        if sd.load(Ordering::Acquire) {
                            return;
                        }
                        let wp = unsafe { &*p.load(Ordering::Relaxed) };
                        let chunk = wp.n.div_ceil(nt);
                        let begin = i * chunk;
                        let end = (begin + chunk).min(wp.n);
                        if wp.is_par_for {
                            unsafe {
                                (wp.par_for_fn)(i, begin, end, wp.par_for_ctx);
                            }
                        } else if wp.is_quantize {
                            let q8_size = wp.row_bytes;
                            let dim = wp.n_cols;
                            for token in begin..end {
                                unsafe {
                                    let src = std::slice::from_raw_parts(
                                        (wp.w_base + token * dim * 4) as *const f32,
                                        dim,
                                    );
                                    let dst = std::slice::from_raw_parts_mut(
                                        (wp.a_ptr + token * q8_size) as *mut u8,
                                        q8_size,
                                    );
                                    hearth_quant::quantize_q8_0_into(src, dst);
                                }
                            }
                        } else {
                            unsafe {
                                batched_rows_chunk(wp, begin, end);
                            }
                        }
                        done_clone.store(true, Ordering::Release);
                    } else {
                        if sd.load(Ordering::Acquire) {
                            return;
                        }
                        spins += 1;
                        if spins < SPIN_BUDGET {
                            std::hint::spin_loop();
                        } else {
                            // Genuinely idle: sleep until a dispatch wakes us
                            // rather than burning a core in a yield loop.
                            // `spins` stays at the budget so a wake-up goes
                            // straight back to sleep unless there is work.
                            parked_worker.fetch_add(1, Ordering::AcqRel);
                            // Double-check under the lock: a dispatch that bumped
                            // the generation after our first read must not be
                            // missed between counting ourselves parked and
                            // sleeping.
                            let guard = wait_worker.0.lock().unwrap();
                            if g.load(Ordering::Acquire) == local_gen {
                                let _ = wait_worker.1.wait_timeout(guard, IDLE_WAIT);
                            }
                            parked_worker.fetch_sub(1, Ordering::AcqRel);
                        }
                    }
                }
            });
            workers.push(Worker {
                handle: Some(handle),
                done_flag: done,
            });
        }

        ThreadPool {
            workers,
            params,
            _params_box: params_box,
            gen,
            wait,
            parked,
            shutdown,
            num_threads,
        }
    }

    /// Publish new work: bump the generation, and wake any worker that has gone
    /// to sleep. Workers still spinning see the new generation on their next
    /// iteration, so the in-token dispatch path stays syscall-free.
    #[inline]
    fn dispatch(&self) {
        self.gen.fetch_add(1, Ordering::Release);
        if self.parked.load(Ordering::Acquire) > 0 {
            self.wait.1.notify_all();
        }
    }

    /// Wait for every worker to finish the chunk it was given.
    ///
    /// The dispatching thread spins first: a dispatch normally completes in
    /// microseconds, so the spin catches it without a syscall. It must not spin
    /// *unconditionally*, though — the pool routinely has more runnable threads than
    /// physical cores, and a thread that never yields keeps its own workers off the
    /// core they need (8B Q1_0 on an 8C/16T box: 13.2 tok/s at 8 workers against 6.9
    /// at 12 now, 12.1 against 6.0 when it spun straight through).
    fn wait_for_workers(&self) {
        for w in &self.workers {
            let mut spins = 0u64;
            while !w.done_flag.load(Ordering::Acquire) {
                spins += 1;
                if spins < SPIN_BUDGET {
                    std::hint::spin_loop();
                } else {
                    std::thread::yield_now();
                }
            }
        }
    }

    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    pub fn par_for(&self, n: usize, func: ParForFn, ctx: usize) {
        if n == 0 {
            return;
        }
        if self.num_threads <= 1 || n <= 1 {
            unsafe {
                (func)(0, 0, n, ctx);
            }
            return;
        }
        unsafe {
            *self.params.load(Ordering::Relaxed) = WorkParams {
                n,
                seq_len: 1,
                q8_stride: 0,
                w_base: 0,
                a_ptr: 0,
                out_ptr: 0,
                row_bytes: 0,
                n_cols: 0,
                dot_fn: dummy_dot_fn,
                dot_fn2: None,
                tile_size: usize::MAX,
                is_quantize: false,
                is_par_for: true,
                par_for_fn: func,
                par_for_ctx: ctx,
            };
        }
        for w in &self.workers {
            w.done_flag.store(false, Ordering::Relaxed);
        }
        self.dispatch();
        self.wait_for_workers();
    }

    pub fn par_quantize(&self, seq_len: usize, dim: usize, f32_base: usize, q8_base: usize) {
        if seq_len == 0 {
            return;
        }
        let q8_size = dim.div_ceil(32) * 34;
        if self.num_threads <= 1 || seq_len <= 1 {
            for s in 0..seq_len {
                let src = unsafe {
                    std::slice::from_raw_parts((f32_base + s * dim * 4) as *const f32, dim)
                };
                let dst = unsafe {
                    std::slice::from_raw_parts_mut((q8_base + s * q8_size) as *mut u8, q8_size)
                };
                hearth_quant::quantize_q8_0_into(src, dst);
            }
            return;
        }
        unsafe {
            *self.params.load(Ordering::Relaxed) = WorkParams {
                n: seq_len,
                seq_len: 1,
                q8_stride: 0,
                w_base: f32_base,
                a_ptr: q8_base,
                out_ptr: 0,
                row_bytes: q8_size,
                n_cols: dim,
                dot_fn: dummy_dot_fn,
                dot_fn2: None,
                tile_size: usize::MAX,
                is_quantize: true,
                is_par_for: false,
                par_for_fn: |_, _, _, _| {},
                par_for_ctx: 0,
            };
        }
        for w in &self.workers {
            w.done_flag.store(false, Ordering::Relaxed);
        }
        self.dispatch();
        self.wait_for_workers();
    }

    pub fn par_dot_rows(&self, p: DotRowsParams) {
        let DotRowsParams {
            n,
            w_base,
            a_ptr,
            out_ptr,
            row_bytes,
            n_cols,
            dot_fn,
        } = p;
        if n == 0 {
            return;
        }
        if self.num_threads <= 1 || n <= 1 {
            let tile_size = (1048576 / row_bytes).max(1);
            let mut r = 0;
            while r < n {
                let tile_end = (r + tile_size).min(n);
                for row in r..tile_end {
                    unsafe {
                        *((out_ptr + row * 4) as *mut f32) = (dot_fn)(
                            (w_base + row * row_bytes) as *const u8,
                            a_ptr as *const u8,
                            n_cols,
                        );
                    }
                }
                r = tile_end;
            }
            return;
        }
        let tile_size = (1048576 / row_bytes).max(1);
        unsafe {
            *self.params.load(Ordering::Relaxed) = WorkParams {
                n,
                seq_len: 1,
                q8_stride: 0,
                w_base,
                a_ptr,
                out_ptr,
                row_bytes,
                n_cols,
                dot_fn,
                dot_fn2: None,
                tile_size,
                is_quantize: false,
                is_par_for: false,
                par_for_fn: |_, _, _, _| {},
                par_for_ctx: 0,
            };
        }
        for w in &self.workers {
            w.done_flag.store(false, Ordering::Relaxed);
        }
        self.dispatch();
        self.wait_for_workers();
    }

    pub fn par_dot_rows_batched(&self, p: BatchedDotRowsParams) {
        let BatchedDotRowsParams {
            n,
            seq_len,
            q8_stride,
            w_base,
            a_ptr,
            out_ptr,
            row_bytes,
            n_cols,
            dot_fn,
            dot_fn2,
        } = p;
        if n == 0 || seq_len == 0 {
            return;
        }
        let tile_size = (1048576 / row_bytes).max(1);
        // Pairing only pays off when a weight row is reused across tokens, so it is
        // disabled for a single-token batch.
        let wp = WorkParams {
            n,
            seq_len,
            q8_stride,
            w_base,
            a_ptr,
            out_ptr,
            row_bytes,
            n_cols,
            dot_fn,
            dot_fn2: if seq_len > 1 { dot_fn2 } else { None },
            tile_size,
            is_quantize: false,
            is_par_for: false,
            par_for_fn: |_, _, _, _| {},
            par_for_ctx: 0,
        };
        if self.num_threads <= 1 || (n <= 1 && seq_len <= 1) {
            unsafe {
                batched_rows_chunk(&wp, 0, n);
            }
            return;
        }
        unsafe {
            *self.params.load(Ordering::Relaxed) = wp;
        }
        for w in &self.workers {
            w.done_flag.store(false, Ordering::Relaxed);
        }
        self.dispatch();
        self.wait_for_workers();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        // Sleeping workers must see the shutdown flag instead of waiting out
        // their idle timeout.
        self.wait.1.notify_all();
        for mut w in self.workers.drain(..) {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SMT query feeds a thread count straight into `ThreadPool::new`, so it
    /// has to be a usable count for any host, including ones whose topology leaf
    /// is missing or nonsense (containers, some hypervisors).
    #[test]
    fn physical_cores_is_always_a_usable_count() {
        for logical in [1usize, 2, 3, 4, 8, 16, 64, 256] {
            let cores = physical_cores(logical);
            assert!(
                (1..=logical).contains(&cores),
                "logical={logical} yielded {cores}"
            );
        }
    }

    /// A count of zero would spin the pool up with no workers and hang every
    /// dispatch; the override is normalized rather than trusted.
    #[test]
    fn worker_count_is_never_zero() {
        assert!(default_worker_count() >= 1);
    }
}
