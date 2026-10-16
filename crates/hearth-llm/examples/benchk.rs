// Raw throughput of the quantized dot kernels: how many GB/s of weights can a
// single thread process?
//
// Usage: benchk [model.gguf] [dtype] [block_prefix]
//
// dtype is one of ptq1_0 | pq2_0 | q2_0 | q1_0 | q1_0_g128 (default ptq1_0);
// the model defaults to the 4B PTQ1_0 file and the prefix to `blk.0.`. Where a
// format has more than one kernel implementation, every one is timed on
// identical data so the dispatch heuristic can be checked against measurement.
use hearth_gguf::{GgmlDType, GgufFile};
use std::time::Instant;

type DotFn = unsafe fn(*const u8, *const u8, usize) -> f32;

fn parse_dtype(s: Option<String>) -> GgmlDType {
    match s.as_deref() {
        Some("pq2_0") | Some("PQ2_0") => GgmlDType::PQ2_0,
        Some("q2_0") | Some("Q2_0") => GgmlDType::Q2_0,
        Some("q1_0") | Some("Q1_0") => GgmlDType::Q1_0,
        Some("q1_0_g128") | Some("Q1_0_G128") => GgmlDType::Q1_0_G128,
        _ => GgmlDType::PTQ1_0,
    }
}

#[cfg(target_arch = "x86_64")]
fn have(feature: &str) -> bool {
    match feature {
        "avx2" => std::arch::is_x86_feature_detected!("avx2"),
        "avx512f" => std::arch::is_x86_feature_detected!("avx512f"),
        "avx512bw" => std::arch::is_x86_feature_detected!("avx512bw"),
        "avx512vl" => std::arch::is_x86_feature_detected!("avx512vl"),
        "avx512vnni" => std::arch::is_x86_feature_detected!("avx512vnni"),
        _ => false,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn have(_feature: &str) -> bool {
    false
}

/// Print the CPU features and the AMD family the way `hearth_quant`'s AVX-512
/// gate decodes it: the 4-bit base family field on its own, then the real
/// family (base + extended when the base field is 0xF, which is how AMD reports
/// every family from 15h onward).
#[cfg(target_arch = "x86_64")]
fn cpu_report() {
    use std::arch::x86_64::__cpuid;
    let vendor = __cpuid(0);
    let id = __cpuid(1);
    let base = (id.eax >> 8) & 0xF;
    let ext = (id.eax >> 20) & 0xFF;
    let real = if base == 0xF { base + ext } else { base };
    let name = [vendor.ebx, vendor.edx, vendor.ecx]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .map(|b| b as char)
        .collect::<String>();
    println!(
        "cpu {name} base_family={base} ext_family=0x{ext:X} real_family=0x{real:X}\n\
         avx2={} avx512f={} avx512bw={} avx512vl={} avx512vnni={}",
        have("avx2"),
        have("avx512f"),
        have("avx512bw"),
        have("avx512vl"),
        have("avx512vnni"),
    );
}

fn make_variants(dtype: GgmlDType) -> Vec<(&'static str, DotFn)> {
    let mut v: Vec<(&'static str, DotFn)> = match dtype {
        GgmlDType::PTQ1_0 => vec![("dispatched", hearth_quant::dot_ptq1_0_q8_0_ptr)],
        GgmlDType::PQ2_0 | GgmlDType::Q2_0 => {
            vec![("dispatched", hearth_quant::dot_q2_0_q8_0_ptr)]
        }
        _ => vec![("dispatched", hearth_quant::dot_q1_0g128_q8_0_ptr)],
    };
    #[cfg(target_arch = "x86_64")]
    {
        if dtype == GgmlDType::PTQ1_0 {
            v.push(("scalar", hearth_quant::ptq1_0::dot_ptq1_0_q8_0_ptr_scalar));
            if have("avx2") {
                v.push(("avx2", hearth_quant::ptq1_0::dot_ptq1_0_q8_0_avx2));
            }
            if have("avx512f") && have("avx512bw") && have("avx512vl") {
                v.push(("avx512", hearth_quant::ptq1_0::dot_ptq1_0_q8_0_avx512));
            }
        }
        if matches!(dtype, GgmlDType::PQ2_0 | GgmlDType::Q2_0) {
            if have("avx2") {
                v.push(("avx2_lut", hearth_quant::q2_0::dot_q2_0_q8_0_lut_avx2));
            }
            if have("avx512f") && have("avx512vnni") {
                v.push(("vnni256", hearth_quant::q2_0::dot_q2_0_q8_0_vnni_avx512));
            }
            if have("avx512f") && have("avx512vnni") && have("avx512bw") {
                v.push((
                    "vnni512x2",
                    hearth_quant::q2_0::dot_q2_0_q8_0_vnni_avx512_2sub,
                ));
            }
        }
    }
    v
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "models/Ternary-Bonsai-2-4B-PTQ1_0.gguf".to_string());
    let dtype = parse_dtype(args.next());
    let prefix = args.next().unwrap_or_else(|| "blk.0.".to_string());

    let gguf = GgufFile::open(&path).unwrap();
    let names: Vec<String> = gguf
        .tensors
        .iter()
        .filter(|t| t.name.starts_with(&prefix) && t.dtype == dtype)
        .map(|t| t.name.clone())
        .collect();
    println!("{path}  dtype={dtype:?}  {prefix}");
    println!("{} matching tensors", names.len());
    #[cfg(target_arch = "x86_64")]
    cpu_report();
    if dtype == GgmlDType::PTQ1_0 {
        println!("dispatch selects: {}", hearth_quant::ptq1_0::kernel_name());
    }
    println!();

    let variants = make_variants(dtype);
    let mut secs = vec![0.0f64; variants.len()];
    let mut bytes_total = 0usize;
    let mut macs_total = 0usize;

    println!(
        "{:24} {:>8} {:>11} {:>10}",
        "tensor", "MB", "kernel", "GB/s"
    );
    for name in &names {
        let t = gguf.find_tensor(name).unwrap();
        let n_in = t.shape[0] as usize;
        let n_out = t.shape[1] as usize;
        let row_bytes = t.dtype.byte_size(n_in);
        let data = gguf.tensor_data(t);
        let bytes = row_bytes * n_out;

        // Activation: deterministic pseudo-random, then q8.
        let x: Vec<f32> = (0..n_in)
            .map(|i| ((i * 2654435761usize % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let mut q8: Vec<u8> = Vec::new();
        hearth_quant::quantize_q8_0(&x, &mut q8);

        for (vi, (label, f)) in variants.iter().enumerate() {
            // Warm the clock and the page tables for a fixed wall-clock time
            // before measuring: otherwise whichever kernel is timed first is
            // penalised by the CPU ramping up out of its idle frequency, which
            // is worth ~2x here and swamps the difference under test.
            let mut acc = 0.0f32;
            let warm = Instant::now();
            while warm.elapsed().as_millis() < 300 {
                for r in 0..n_out {
                    acc += unsafe { f(data.as_ptr().add(r * row_bytes), q8.as_ptr(), n_in) };
                }
            }
            let mut pass = f64::MAX;
            for _ in 0..2 {
                let t0 = Instant::now();
                for r in 0..n_out {
                    acc += unsafe { f(data.as_ptr().add(r * row_bytes), q8.as_ptr(), n_in) };
                }
                pass = pass.min(t0.elapsed().as_secs_f64());
            }
            std::hint::black_box(acc);
            secs[vi] += pass;
            println!(
                "{:24} {:>8.2} {:>11} {:>10.2}",
                name.trim_start_matches(&prefix),
                bytes as f64 / 1e6,
                label,
                bytes as f64 / pass / 1e9
            );
        }
        println!();
        bytes_total += bytes;
        macs_total += n_out * n_in;
    }

    println!(
        "block totals: {:.2} GB weights, {:.2} G MACs",
        bytes_total as f64 / 1e9,
        macs_total as f64 / 1e9
    );
    for (vi, (label, _)) in variants.iter().enumerate() {
        println!(
            "  {label:>11}: {:.2} GB/s single-thread aggregate",
            bytes_total as f64 / secs[vi] / 1e9
        );
    }
}
