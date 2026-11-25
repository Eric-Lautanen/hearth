//! Per-kernel microbenchmark: weight bandwidth (GB/s) and weight throughput
//! (weights/s) for every quantized format the engine can load, at each thread
//! count.
//!
//! This exists because kernel claims were being inferred from whole-model tok/s.
//! A tok/s reading mixes the kernel with prefill pairing, KV traffic, sampling,
//! thread-count policy and whatever else the machine was doing; two readings of
//! the same binary can spread 10-15% on frequency scaling alone. Here the
//! measurement is one row kernel walked over a weight buffer far larger than
//! L3, which is the quantity a kernel change actually moves.
//!
//! The kernel is not chosen by this file: pooled formats go through
//! [`hearth_llm::pool::dot_rows_fn`], the same table the decode path dispatches,
//! so a format whose kernel is misconfigured cannot look fast here.
//!
//! ```text
//! cargo run --release -p hearth-llm --bin kernel-bench
//! cargo run --release -p hearth-llm --bin kernel-bench -- --format ptq1_0 --threads 1,4,8
//! ```

use std::hint::black_box;
use std::time::Instant;

use hearth_gguf::GgmlDType;
use hearth_llm::pool::{dot_rows_fn, DotRowsParams, ThreadPool};
use rayon::prelude::*;

/// How the engine turns a weight format into row dot products.
#[derive(Clone, Copy)]
enum Dispatch {
    /// `ThreadPool::par_dot_rows` with q8_0 activations: the decode path.
    Pooled,
    /// rayon over rows with q8_0 activations (`Q4_0`'s arm in `matmul`).
    RayonQ8(fn(&[u8], &[u8], usize) -> f32),
    /// rayon over rows with f32 activations.
    RayonF32(fn(&[u8], &[f32], usize) -> f32),
}

struct Format {
    dtype: GgmlDType,
    /// The kernel function by name, so a report line can be quoted in the
    /// tracker without guessing which one ran.
    kernel: &'static str,
    dispatch: Dispatch,
}

const FORMATS: &[Format] = &[
    Format {
        dtype: GgmlDType::PTQ1_0,
        kernel: "dot_ptq1_0_q8_0",
        dispatch: Dispatch::Pooled,
    },
    Format {
        dtype: GgmlDType::Q2_0,
        kernel: "dot_q2_0_q8_0",
        dispatch: Dispatch::Pooled,
    },
    Format {
        dtype: GgmlDType::PQ2_0,
        kernel: "dot_q2_0_q8_0 (shared)",
        dispatch: Dispatch::Pooled,
    },
    Format {
        dtype: GgmlDType::Q1_0,
        kernel: "dot_q1_0g128_q8_0",
        dispatch: Dispatch::Pooled,
    },
    Format {
        dtype: GgmlDType::Q10G128,
        kernel: "dot_q1_0g128_q8_0 (shared)",
        dispatch: Dispatch::Pooled,
    },
    Format {
        dtype: GgmlDType::Q8_0,
        kernel: "dot_q8_0_q8_0",
        dispatch: Dispatch::Pooled,
    },
    Format {
        dtype: GgmlDType::Q4_0,
        kernel: "dot_q4_0_q8_0",
        dispatch: Dispatch::RayonQ8(hearth_quant::dot_q4_0_q8_0),
    },
    Format {
        dtype: GgmlDType::Q4_1,
        kernel: "dot_q4_1",
        dispatch: Dispatch::RayonF32(hearth_quant::dot_q4_1),
    },
    Format {
        dtype: GgmlDType::Q5_0,
        kernel: "dot_q5_0",
        dispatch: Dispatch::RayonF32(hearth_quant::dot_q5_0),
    },
    Format {
        dtype: GgmlDType::Q5_1,
        kernel: "dot_q5_1",
        dispatch: Dispatch::RayonF32(hearth_quant::dot_q5_1),
    },
    Format {
        dtype: GgmlDType::Q4K,
        kernel: "dot_q4_k",
        dispatch: Dispatch::RayonF32(hearth_quant::dot_q4_k),
    },
    Format {
        dtype: GgmlDType::Q6K,
        kernel: "dot_q6_k",
        dispatch: Dispatch::RayonF32(hearth_quant::dot_q6_k),
    },
];

struct Args {
    rows: usize,
    cols: usize,
    threads: Vec<usize>,
    reps: usize,
    only: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let logical = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let mut args = Args {
            // 32768 x 5120 is ~36 MB of PTQ1_0 weights: past this box's 16 MB L3
            // even for the smallest format, so the kernel streams from DRAM the
            // way decode does instead of measuring cache-resident throughput.
            rows: 32768,
            cols: 5120,
            threads: [1, 2, 4, 6, 8, 12, 16]
                .into_iter()
                .filter(|t| *t <= logical)
                .collect(),
            reps: 5,
            only: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut value = || it.next().ok_or_else(|| format!("{arg} needs a value"));
            match arg.as_str() {
                "--rows" => args.rows = parse_num(&value()?, "--rows")?,
                "--cols" => args.cols = parse_num(&value()?, "--cols")?,
                "--reps" => args.reps = parse_num(&value()?, "--reps")?.max(1),
                "--threads" => {
                    let list = value()?;
                    args.threads = list
                        .split(',')
                        .map(|t| parse_num(t.trim(), "--threads"))
                        .collect::<Result<Vec<_>, _>>()?;
                    args.threads.retain(|t| *t > 0);
                    if args.threads.is_empty() {
                        return Err("--threads needs at least one positive count".to_string());
                    }
                }
                "--format" => args.only = Some(value()?.to_ascii_lowercase()),
                "--help" | "-h" => {
                    println!(
                        "kernel-bench [--rows N] [--cols N] [--threads 1,4,8] [--reps N] \
                         [--format ptq1_0]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        if args.rows == 0 || args.cols == 0 {
            return Err("--rows and --cols must be positive".to_string());
        }
        Ok(args)
    }
}

fn parse_num(text: &str, flag: &str) -> Result<usize, String> {
    text.parse::<usize>()
        .map_err(|_| format!("{flag} expects a number, got `{text}`"))
}

struct Workload<'a> {
    rows: usize,
    cols: usize,
    row_bytes: usize,
    weights: &'a [u8],
    act_q8: &'a [u8],
    act_f32: &'a [f32],
    out: &'a mut [f32],
}

impl Workload<'_> {
    /// Weight bytes read by one pass over every row.
    fn weight_bytes(&self) -> u64 {
        (self.rows * self.row_bytes) as u64
    }

    /// Weight values multiplied by one pass over every row.
    fn weight_count(&self) -> u64 {
        (self.rows * self.cols) as u64
    }
}

fn run_once(
    fmt: &Format,
    pooled: &ThreadPool,
    rayon_pool: &rayon::ThreadPool,
    w: &mut Workload<'_>,
) {
    match fmt.dispatch {
        Dispatch::Pooled => {
            let dot_fn = dot_rows_fn(fmt.dtype).expect("pooled format has a dispatch entry");
            pooled.par_dot_rows(DotRowsParams {
                n: w.rows,
                w_base: w.weights.as_ptr() as usize,
                a_ptr: w.act_q8.as_ptr() as usize,
                out_ptr: w.out.as_mut_ptr() as usize,
                row_bytes: w.row_bytes,
                n_cols: w.cols,
                dot_fn,
            });
        }
        Dispatch::RayonQ8(kernel) => {
            let (weights, act, out, cols, row_bytes) =
                (w.weights, w.act_q8, &mut *w.out, w.cols, w.row_bytes);
            rayon_pool.install(|| {
                out.par_iter_mut().enumerate().for_each(|(row, slot)| {
                    let row_data = &weights[row * row_bytes..][..row_bytes];
                    *slot = kernel(row_data, act, cols);
                });
            });
        }
        Dispatch::RayonF32(kernel) => {
            let (weights, act, out, cols, row_bytes) =
                (w.weights, w.act_f32, &mut *w.out, w.cols, w.row_bytes);
            rayon_pool.install(|| {
                out.par_iter_mut().enumerate().for_each(|(row, slot)| {
                    let row_data = &weights[row * row_bytes..][..row_bytes];
                    *slot = kernel(row_data, act, cols);
                });
            });
        }
    }
    black_box(&w.out[..1]);
}

/// Deterministic byte pattern. The benchmark measures memory traffic and
/// arithmetic, not numerics, so a filled buffer measures the same as real
/// weights without needing a multi-gigabyte file; every byte is touched so the
/// first pass is not paying page faults.
fn pattern_bytes(len: usize) -> Vec<u8> {
    let mut buf = vec![0u8; len];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for byte in buf.iter_mut() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (state >> 33) as u8;
    }
    buf
}

fn activation_f32(cols: usize) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn main() -> Result<(), String> {
    let args = Args::parse()?;

    let report = hearth_quant::dispatch_report(FORMATS.iter().map(|f| f.dtype));
    println!(
        "kernel-bench · {} · avx512_full_width={} · {} rows × {} cols · best of {} passes",
        report.cpu, report.avx512_full_width, args.rows, args.cols, args.reps
    );
    println!(
        "weights are read once per pass, sized to exceed L3 so this is the streaming path \
         decode uses;\nthe activation row stays in cache and is not counted in GB/s"
    );
    // The engine walks the rayon formats with rayon's global pool, which is sized
    // from the logical core count; here each thread count gets its own pool so the
    // curve can be read at all.
    println!(
        "pooled formats use ThreadPool::par_dot_rows (the decode dispatch); the rayon \
         formats get a local pool per\nrow count here, while the engine walks them with \
         rayon's global pool (one worker per logical core)\n"
    );

    let act_f32 = activation_f32(args.cols);
    let mut act_q8 = Vec::new();
    hearth_quant::quantize_q8_0(&act_f32, &mut act_q8);

    let rayon_pools: Vec<(usize, rayon::ThreadPool)> = args
        .threads
        .iter()
        .map(|&threads| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| format!("rayon pool of {threads}: {e}"))?;
            Ok((threads, pool))
        })
        .collect::<Result<_, String>>()?;

    // format name -> (threads, weight GB/s)
    let mut summary: Vec<(String, Vec<(usize, f64)>)> = Vec::new();

    for fmt in FORMATS {
        let name = fmt.dtype.name().to_string();
        if let Some(only) = &args.only {
            if !name.to_ascii_lowercase().contains(only) {
                continue;
            }
        }

        let row_bytes = fmt.dtype.byte_size(args.cols);
        let weights = pattern_bytes(row_bytes * args.rows);
        let mut out = vec![0f32; args.rows];
        let mut workload = Workload {
            rows: args.rows,
            cols: args.cols,
            row_bytes,
            weights: &weights,
            act_q8: &act_q8,
            act_f32: &act_f32,
            out: &mut out,
        };
        let weight_bytes = workload.weight_bytes();
        let weight_count = workload.weight_count();

        let isa = hearth_quant::selected_kernel(fmt.dtype).unwrap_or("row loop");
        println!(
            "{} · {} · {} · {:.1} MiB · {:.2} B/weight · isa {}",
            name,
            fmt.kernel,
            match fmt.dispatch {
                Dispatch::Pooled | Dispatch::RayonQ8(_) => "q8_0 act",
                Dispatch::RayonF32(_) => "f32 act",
            },
            weight_bytes as f64 / (1024.0 * 1024.0),
            weight_bytes as f64 / weight_count as f64,
            isa
        );
        println!(" threads   best ms  median ms   weight GB/s   Gweights/s   speedup");

        let mut single_thread_gbs: Option<f64> = None;
        let mut rows = Vec::new();
        for (threads, rayon_pool) in &rayon_pools {
            let pooled = ThreadPool::new(*threads);
            // One warmup pass: the pool spins its workers up and the buffer
            // pages are first touched here, not inside the timed loop.
            run_once(fmt, &pooled, rayon_pool, &mut workload);

            let mut samples = Vec::with_capacity(args.reps);
            for _ in 0..args.reps {
                let start = Instant::now();
                run_once(fmt, &pooled, rayon_pool, &mut workload);
                samples.push(start.elapsed().as_secs_f64());
            }
            samples.sort_by(f64::total_cmp);
            // Reported throughput comes from the median, not the best: a spin-wait
            // pool scheduled over shared cores can hand back one easy pass that no
            // other pass reproduces, and a claim built on that pass is the noise
            // this bench exists to remove. The best pass stays visible beside it.
            let best = samples[0];
            let median = samples[samples.len() / 2];

            let gbs = weight_bytes as f64 / median / 1e9;
            let gweights = weight_count as f64 / median / 1e9;
            if *threads == 1 {
                single_thread_gbs = Some(gbs);
            }
            let speedup = match single_thread_gbs {
                Some(base) if base > 0.0 => format!("{:.2}x", gbs / base),
                _ => "-".to_string(),
            };
            println!(
                "{:>8} {:>10.3} {:>10.3} {:>13.2} {:>12.2} {:>9}",
                *threads,
                best * 1e3,
                median * 1e3,
                gbs,
                gweights,
                speedup
            );
            rows.push((*threads, gbs));
        }
        if let Some((threads, gbs)) = rows.iter().copied().max_by(|a, b| a.1.total_cmp(&b.1)) {
            println!("   best: {threads} threads, {gbs:.2} GB/s\n");
        } else {
            println!();
        }
        summary.push((name, rows));
    }

    if summary.is_empty() {
        return Err("no format matched --format".to_string());
    }

    let header: Vec<String> = args.threads.iter().map(|t| format!("{t:>7}")).collect();
    println!("summary · weight GB/s by thread count (median pass)");
    println!("{:<10}{}", "format", header.join(""));
    for (name, rows) in &summary {
        let mut line = format!("{name:<10}");
        for &threads in &args.threads {
            match rows.iter().find(|(t, _)| *t == threads) {
                Some((_, gbs)) => line.push_str(&format!("{gbs:>7.2}")),
                None => line.push_str(&format!("{:>7}", "-")),
            }
        }
        println!("{line}");
    }
    Ok(())
}
