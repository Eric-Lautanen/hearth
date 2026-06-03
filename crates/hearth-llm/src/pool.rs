use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

use hearth_quant;

pub type DotFn = unsafe fn(*const u8, *const u8, usize) -> f32;
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
    pub tile_size: usize,
    pub is_quantize: bool,
    pub is_par_for: bool,
    pub par_for_fn: ParForFn,
    pub par_for_ctx: usize,
}

struct Worker {
    handle: Option<thread::JoinHandle<()>>,
    done_flag: Arc<AtomicBool>,
}

pub struct ThreadPool {
    workers: Vec<Worker>,
    params: Arc<AtomicPtr<WorkParams>>,
    _params_box: Box<WorkParams>,
    gen: Arc<AtomicU64>,
    shutdown: Arc<AtomicBool>,
    num_threads: usize,
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
            tile_size: usize::MAX,
            is_quantize: false,
            is_par_for: false,
            par_for_fn: |_, _, _, _| {},
            par_for_ctx: 0,
        });
        let params = Arc::new(AtomicPtr::new(&mut *params_box));
        let gen = Arc::new(AtomicU64::new(0));

        let mut workers = Vec::with_capacity(num_threads);
        for i in 0..num_threads {
            let done = Arc::new(AtomicBool::new(false));
            let done_clone = done.clone();
            let sd = shutdown.clone();
            let p = params.clone();
            let g = gen.clone();
            let nt = num_threads;
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
                        let tile_size = wp.tile_size;
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
                            let seq_len = wp.seq_len.max(1);
                            let n = wp.n;
                            for s in 0..seq_len {
                                let a = (wp.a_ptr + s * wp.q8_stride) as *const u8;
                                let out_off = s * n;
                                let mut r = begin;
                                while r < end {
                                    let tile_end = (r + tile_size).min(end);
                                    for row in r..tile_end {
                                        unsafe {
                                            *((wp.out_ptr + (row + out_off) * 4) as *mut f32) =
                                                (wp.dot_fn)(
                                                    (wp.w_base + row * wp.row_bytes) as *const u8,
                                                    a,
                                                    wp.n_cols,
                                                );
                                        }
                                    }
                                    r = tile_end;
                                }
                            }
                        }
                        done_clone.store(true, Ordering::Release);
                    } else {
                        if sd.load(Ordering::Acquire) {
                            return;
                        }
                        spins += 1;
                        if spins < 65536 {
                            std::hint::spin_loop();
                        } else {
                            spins = 0;
                            thread::yield_now();
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
            shutdown,
            num_threads,
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
        self.gen.fetch_add(1, Ordering::Release);
        for w in &self.workers {
            while !w.done_flag.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
        }
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
        self.gen.fetch_add(1, Ordering::Release);
        for w in &self.workers {
            while !w.done_flag.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
        }
    }

    pub fn par_dot_rows(
        &self,
        n: usize,
        w_base: usize,
        a_ptr: usize,
        out_ptr: usize,
        row_bytes: usize,
        n_cols: usize,
        dot_fn: DotFn,
    ) {
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
        self.gen.fetch_add(1, Ordering::Release);
        for w in &self.workers {
            while !w.done_flag.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
        }
    }

    pub fn par_dot_rows_batched(
        &self,
        n: usize,
        seq_len: usize,
        q8_stride: usize,
        w_base: usize,
        a_ptr: usize,
        out_ptr: usize,
        row_bytes: usize,
        n_cols: usize,
        dot_fn: DotFn,
    ) {
        if n == 0 || seq_len == 0 {
            return;
        }
        if self.num_threads <= 1 || (n <= 1 && seq_len <= 1) {
            let tile_size = (1048576 / row_bytes).max(1);
            for s in 0..seq_len {
                let a = (a_ptr + s * q8_stride) as *const u8;
                let out_off = s * n;
                let mut r = 0;
                while r < n {
                    let tile_end = (r + tile_size).min(n);
                    for row in r..tile_end {
                        unsafe {
                            *((out_ptr + (row + out_off) * 4) as *mut f32) =
                                (dot_fn)((w_base + row * row_bytes) as *const u8, a, n_cols);
                        }
                    }
                    r = tile_end;
                }
            }
            return;
        }
        let tile_size = (1048576 / row_bytes).max(1);
        unsafe {
            *self.params.load(Ordering::Relaxed) = WorkParams {
                n,
                seq_len,
                q8_stride,
                w_base,
                a_ptr,
                out_ptr,
                row_bytes,
                n_cols,
                dot_fn,
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
        self.gen.fetch_add(1, Ordering::Release);
        for w in &self.workers {
            while !w.done_flag.load(Ordering::Acquire) {
                std::hint::spin_loop();
            }
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        for mut w in self.workers.drain(..) {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}
