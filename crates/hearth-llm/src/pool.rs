use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

pub type DotFn = unsafe fn(*const u8, *const u8, usize) -> f32;

unsafe fn dummy_dot_fn(_w: *const u8, _a: *const u8, _n: usize) -> f32 {
    0.0
}

pub struct WorkParams {
    pub n: usize,
    pub w_base: usize,
    pub a_ptr: usize,
    pub out_ptr: usize,
    pub row_bytes: usize,
    pub n_cols: usize,
    pub dot_fn: DotFn,
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
            w_base: 0,
            a_ptr: 0,
            out_ptr: 0,
            row_bytes: 0,
            n_cols: 0,
            dot_fn: dummy_dot_fn,
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
                        for row in begin..end {
                            unsafe {
                                *((wp.out_ptr + row * 4) as *mut f32) = (wp.dot_fn)(
                                    (wp.w_base + row * wp.row_bytes) as *const u8,
                                    wp.a_ptr as *const u8,
                                    wp.n_cols,
                                );
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
            for row in 0..n {
                unsafe {
                    *((out_ptr + row * 4) as *mut f32) = (dot_fn)(
                        (w_base + row * row_bytes) as *const u8,
                        a_ptr as *const u8,
                        n_cols,
                    );
                }
            }
            return;
        }
        unsafe {
            *self.params.load(Ordering::Relaxed) = WorkParams {
                n,
                w_base,
                a_ptr,
                out_ptr,
                row_bytes,
                n_cols,
                dot_fn,
            };
        }
        for w in &self.workers {
            w.done_flag.store(false, Ordering::Relaxed);
        }
        // Release store ensures the params write and done_flag reset are visible
        // to workers that see the gen change via Acquire load.
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
