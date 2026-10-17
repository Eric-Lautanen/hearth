pub mod pq2_0;
pub mod ptq1_0;
pub mod q1_0g128;
pub mod q2_0;
pub mod q4_0;
pub mod q4_1;
pub mod q4_k;
pub mod q5_0;
pub mod q5_1;
pub mod q8_0;
pub mod qk_quant;

use hearth_gguf::GgmlDType;

/// Whether 512-bit vector kernels are expected to beat their 256-bit versions
/// on this CPU.
///
/// AMD family 19h (Zen 3 and Zen 4) splits every 512-bit operation into two
/// 256-bit micro-ops, so wider registers buy no throughput while wider-kernel
/// overhead (packing, masked loads) is real: measured on a Ryzen 7 8840HS, the
/// AVX-512 PTQ1_0 dot ran at 1.28 GB/s against 1.76 GB/s for the AVX2 one.
/// Full-width implementations (Intel server parts, Zen 5+) do benefit.
///
/// This is not a blanket "no AVX-512 on Zen 4": the VNNI kernels reduce the
/// front-end work enough to stay ahead there and are gated on the VNNI feature
/// alone (see `q2_0::dot_q2_0_q8_0_ptr`).
#[cfg(target_arch = "x86_64")]
pub(crate) fn avx512_full_width() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static STATE: AtomicU8 = AtomicU8::new(2); // 2 = unknown
    let v = STATE.load(Ordering::Relaxed);
    if v < 2 {
        return v == 1;
    }
    let ok = avx512_full_width_for(
        cpu_is_amd(),
        cpu_family(),
        std::arch::is_x86_feature_detected!("avx512f"),
        std::arch::is_x86_feature_detected!("avx512bw"),
    );
    STATE.store(if ok { 1 } else { 0 }, Ordering::Relaxed);
    ok
}

/// One f16 scale → f32, for kernels that declare `f16c` themselves.
///
/// `half`'s `to_f32` is feature-detected at runtime, which is the right thing for code that
/// has to run anywhere. Inside a kernel that already declares `f16c` it is the wrong thing:
/// the runtime probe and *both* of its arms inline into the block loop, and the register
/// pressure that costs is larger than the conversion itself. Measured on Bonsai-1.7B-Q1_0 at
/// `x86-64-v2` (`benchk`): 0.52 GB/s with `half` and no declaration, 1.76 with the
/// declaration alone, 3.06 with this — against 3.11 at `x86-64-v3`, where `half` lowers to
/// the same single instruction anyway. See `docs/BUG_TRACKER.md`.
///
/// # Safety
/// The calling function must declare `#[target_feature(enable = "f16c")]`; this executes
/// `vcvtph2ps`, which is a hard requirement wherever it is reached.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "f16c")]
pub(crate) unsafe fn f16_bits_to_f32(bits: u16) -> f32 {
    use core::arch::x86_64::{_mm_cvtph_ps, _mm_cvtsi32_si128, _mm_cvtss_f32};
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(bits))))
}

/// AMD family 19h (Zen 3 / Zen 4). The only AVX-512 part in it is Zen 4, whose
/// 512-bit datapath is two 256-bit halves.
pub(crate) const AMD_FAMILY_SPLIT_AVX512: u8 = 0x19;

/// Effective CPUID family from leaf 1 `EAX`.
///
/// The family is a 4-bit field, so every part from 15h up is encoded as base
/// `0xF` plus an 8-bit *extended* family: Zen 4's 19h arrives as base 15 with
/// extended 0xA. Reading only the raw 4-bit field yields `0xF` on every Zen part,
/// so a comparison against 19h — or against the decimal 25 it is sometimes
/// written as — is false on all of them. That mistake silently kept every
/// AVX-512 kernel on the slower path here until the `raw_family_field_can_never_
/// match_19h` test below pinned it down.
///
/// Kept free of any hardware access so the decode can be tested for parts this
/// machine is not.
pub(crate) const fn cpuid_family(eax: u32) -> u8 {
    let base = ((eax >> 8) & 0xF) as u8;
    let ext = ((eax >> 20) & 0xFF) as u8;
    if base == 0xF {
        base + ext
    } else {
        base
    }
}

/// Whether a leaf-0 `EBX`/`ECX`/`EDX` triple reads `AuthenticAMD`.
pub(crate) const fn cpuid_is_amd(ebx: u32, ecx: u32, edx: u32) -> bool {
    ebx == u32::from_le_bytes(*b"Auth")
        && edx == u32::from_le_bytes(*b"enti")
        && ecx == u32::from_le_bytes(*b"cAMD")
}

/// The policy behind [`avx512_full_width`], with the CPU facts passed in so the
/// decision can be checked for parts this machine is not.
///
/// A part gets full-width AVX-512 unless it is an AMD part in family 19h, whose
/// 512-bit datapath is two 256-bit halves. The features must still be present.
pub(crate) fn avx512_full_width_for(
    is_amd: bool,
    family: u8,
    has_avx512f: bool,
    has_avx512bw: bool,
) -> bool {
    has_avx512f && has_avx512bw && !(is_amd && family == AMD_FAMILY_SPLIT_AVX512)
}

/// Reads a CPUID leaf.
///
/// An `unsafe fn` wrapper rather than a direct call, because `__cpuid`'s safety is a
/// property of the **build**: with SSE available the intrinsic is safe and an `unsafe`
/// block around it is redundant — which `unused_unsafe` says, and which is the case for
/// every build made with this repository's `-C target-cpu=native` — while without SSE the
/// call must be unsafe. Going through a wrapper makes the call site's `unsafe` necessary
/// in both configurations, so neither needs a lint suppressed.
#[cfg(target_arch = "x86_64")]
unsafe fn cpuid(leaf: u32) -> core::arch::x86_64::CpuidResult {
    core::arch::x86_64::__cpuid(leaf)
}

#[cfg(target_arch = "x86_64")]
fn cpu_is_amd() -> bool {
    let v = unsafe { cpuid(0) };
    cpuid_is_amd(v.ebx, v.ecx, v.edx)
}

#[cfg(target_arch = "x86_64")]
fn cpu_family() -> u8 {
    cpuid_family(unsafe { cpuid(1) }.eax)
}

pub fn dequantize(dtype: GgmlDType, data: &[u8], out: &mut [f32]) -> Result<(), String> {
    match dtype {
        GgmlDType::F32 => {
            let src = bytemuck::cast_slice::<u8, f32>(data);
            let n = out.len().min(src.len());
            out[..n].copy_from_slice(&src[..n]);
        }
        GgmlDType::F16 => {
            let src = bytemuck::cast_slice::<u8, half::f16>(data);
            for (i, &v) in src.iter().enumerate() {
                if i >= out.len() {
                    break;
                }
                out[i] = v.to_f32();
            }
        }
        GgmlDType::BF16 => {
            // as_chunks::<2> needs Rust 1.88+ (repo previously said 1.85).
            let (chunks, _remainder) = data.as_chunks::<2>();
            for (i, chunk) in chunks.iter().enumerate() {
                if i >= out.len() {
                    break;
                }
                let bits = u16::from_le_bytes(*chunk);
                out[i] = f32::from_bits((bits as u32) << 16);
            }
        }
        GgmlDType::F64 => {
            let src = bytemuck::cast_slice::<u8, f64>(data);
            for (i, &v) in src.iter().enumerate() {
                if i >= out.len() {
                    break;
                }
                out[i] = v as f32;
            }
        }
        GgmlDType::Q8_0 => q8_0::dequantize(data, out),
        GgmlDType::Q4_0 => q4_0::dequantize(data, out),
        GgmlDType::Q4_1 => q4_1::dequantize(data, out),
        GgmlDType::Q5_0 => q5_0::dequantize(data, out),
        GgmlDType::Q5_1 => q5_1::dequantize(data, out),
        GgmlDType::Q4K => q4_k::dequantize(data, out),
        GgmlDType::Q2_0 => q2_0::dequantize(data, out),
        GgmlDType::PQ2_0 => pq2_0::dequantize(data, out),
        GgmlDType::PTQ1_0 => ptq1_0::dequantize(data, out),
        GgmlDType::Q1_0 | GgmlDType::Q10G128 => q1_0g128::dequantize(data, out),
        GgmlDType::Q2K => qk_quant::dequantize_q2_k(data, out),
        GgmlDType::Q3K => qk_quant::dequantize_q3_k(data, out),
        GgmlDType::Q5K => qk_quant::dequantize_q5_k(data, out),
        GgmlDType::Q6K => qk_quant::dequantize_q6_k(data, out),
        GgmlDType::Q8_1 => {
            return Err(format!(
                "Dequant for {} not yet implemented - try a Q8_0 or Q4_K model instead",
                dtype.name()
            ));
        }
        GgmlDType::Q8K => {
            return Err(format!(
                "Dequant for {} not yet implemented - try a Q8_0 or Q4_K model instead",
                dtype.name()
            ));
        }
        _ => {
            return Err(format!(
                "Unsupported quantization type: {} (requires a dequant kernel)",
                dtype.name()
            ));
        }
    }
    Ok(())
}

pub fn dot_q8_0(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q8_0::dot_q8_0_f32(row, vec, n)
}

pub fn dot_q4_0(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q4_0::dot_q4_0_f32(row, vec, n)
}

pub fn dot_q4_1(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q4_1::dot_q4_1_f32(row, vec, n)
}

pub fn dot_q5_0(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q5_0::dot_q5_0_f32(row, vec, n)
}

pub fn dot_q5_1(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q5_1::dot_q5_1_f32(row, vec, n)
}

pub fn dot_q4_k(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q4_k::dot_q4_k_f32(row, vec, n)
}

pub fn dot_q6_k(row: &[u8], vec: &[f32], n: usize) -> f32 {
    qk_quant::dot_q6_k_f32(row, vec, n)
}

pub fn dot_q2_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q2_0::dot_q2_0_f32(row, vec, n)
}

pub fn dot_q2_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q2_0::dot_q2_0_q8_0(row, act, n)
}

pub fn dot_pq2_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    pq2_0::dot_pq2_0_f32(row, vec, n)
}

pub fn dot_ptq1_0_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    ptq1_0::dot_ptq1_0_f32(row, vec, n)
}

/// # Safety
/// w_ptr valid for n/128*28 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_ptq1_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    ptq1_0::dot_ptq1_0_q8_0_ptr(w_ptr, a_ptr, n)
}

/// Paired-token PTQ1_0 dot: one weight row against two Q8_0 activation rows,
/// sharing the ternary decode block. Output is bit-identical to two calls to
/// [`dot_ptq1_0_q8_0_ptr`], so only the prefill path is affected.
/// # Safety
/// w_ptr valid for n/128*28 bytes, a0_ptr and a1_ptr each valid for n/32*34 bytes.
pub unsafe fn dot_ptq1_0_q8_0x2_ptr(
    w_ptr: *const u8,
    a0_ptr: *const u8,
    a1_ptr: *const u8,
    n: usize,
) -> (f32, f32) {
    ptq1_0::dot_ptq1_0_q8_0x2_ptr(w_ptr, a0_ptr, a1_ptr, n)
}

/// # Safety
/// w_ptr valid for n/128*34 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q2_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q2_0::dot_q2_0_q8_0_ptr(w_ptr, a_ptr, n)
}

/// Paired-token Q2_0 dot: one weight row against two Q8_0 activation rows,
/// sharing the Q2V_I16 weight expansion. Output is bit-identical to two calls to
/// [`dot_q2_0_q8_0_ptr`].
/// # Safety
/// w_ptr valid for n/128*34 bytes, a0_ptr and a1_ptr each valid for n/32*34 bytes.
pub unsafe fn dot_q2_0_q8_0x2_ptr(
    w_ptr: *const u8,
    a0_ptr: *const u8,
    a1_ptr: *const u8,
    n: usize,
) -> (f32, f32) {
    q2_0::dot_q2_0_q8_0x2_ptr(w_ptr, a0_ptr, a1_ptr, n)
}

pub fn dot_q1_0(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_f32(row, vec, n)
}

pub fn dot_q1_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_q8_0(row, act, n)
}

pub fn dot_q1_0g128_f32(row: &[u8], vec: &[f32], n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_f32(row, vec, n)
}

pub fn dot_q1_0g128_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_q8_0(row, act, n)
}

/// # Safety
/// w_ptr valid for n/32*6 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q1_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q1_0g128::dot_q1_0_q8_0_ptr(w_ptr, a_ptr, n)
}

/// # Safety
/// w_ptr valid for n/128*18 bytes, a_ptr valid for n/32*34 bytes.
pub unsafe fn dot_q1_0g128_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q1_0g128::dot_q1_0g128_q8_0_ptr(w_ptr, a_ptr, n)
}

/// Paired-token Q1_0_G128 dot: one weight row against two Q8_0 activation rows,
/// sharing the weight bit expansion. Output is bit-identical to two calls to
/// [`dot_q1_0g128_q8_0_ptr`].
/// # Safety
/// w_ptr valid for n/128*18 bytes, a0_ptr and a1_ptr each valid for n/32*34 bytes.
pub unsafe fn dot_q1_0g128_q8_0x2_ptr(
    w_ptr: *const u8,
    a0_ptr: *const u8,
    a1_ptr: *const u8,
    n: usize,
) -> (f32, f32) {
    q1_0g128::dot_q1_0g128_q8_0x2_ptr(w_ptr, a0_ptr, a1_ptr, n)
}

pub fn dot_q4_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q4_0::dot_q4_0_q8_0(row, act, n)
}

pub fn dot_q8_0_q8_0(row: &[u8], act: &[u8], n: usize) -> f32 {
    q8_0::dot_q8_0_q8_0(row, act, n)
}

/// # Safety
/// Both pointers valid for n/32*34 bytes.
pub unsafe fn dot_q8_0_q8_0_ptr(w_ptr: *const u8, a_ptr: *const u8, n: usize) -> f32 {
    q8_0::dot_q8_0_q8_0_ptr(w_ptr, a_ptr, n)
}

pub fn quantize_q8_0(src: &[f32], dst: &mut Vec<u8>) {
    q8_0::quantize(src, dst)
}

pub fn quantize_q8_0_into(src: &[f32], dst: &mut [u8]) {
    q8_0::quantize_into(src, dst);
}

pub fn quantize_f32(dtype: GgmlDType, src: &[f32], dst: &mut Vec<u8>) {
    match dtype {
        GgmlDType::Q8_0 => q8_0::quantize(src, dst),
        _ => {
            if dtype == GgmlDType::F32 {
                let bytes = bytemuck::cast_slice::<f32, u8>(src);
                dst.extend_from_slice(bytes);
            }
        }
    }
}

/// Name of the kernel Hearth will use for a quantized *weight* dtype on this
/// CPU, or `None` when the dtype has no reported selection.
///
/// This exists so a stale dispatch is visible without running a benchmark. The
/// bug it guards against — a feature gate that could never be true — silently
/// cost 34% on PTQ1_0 (1.28 vs 1.76 GB/s) and was invisible in anything a user
/// or a benchmark harness prints at startup.
///
/// The mapping mirrors the `DotFn` table in `hearth-llm`'s `model/matmul.rs`:
/// `PQ2_0` shares the Q2_0 kernels and `Q1_0` shares the Q1_0_G128 ones.
pub fn selected_kernel(dtype: GgmlDType) -> Option<&'static str> {
    match dtype {
        GgmlDType::PTQ1_0 => Some(ptq1_0::kernel_name()),
        GgmlDType::Q2_0 | GgmlDType::PQ2_0 => Some(q2_0::kernel_name()),
        GgmlDType::Q1_0 | GgmlDType::Q10G128 => Some(q1_0g128::kernel_name()),
        _ => None,
    }
}

/// This machine's AVX-512 class plus the kernel each weight format will use.
///
/// Built for service and UI consumers so neither has to parse the `[kernel]`
/// load banner: `cpu` is a short label (`zen4`, `zen5`, `zen3`, `intel`, `amd`,
/// `other`), `cpu_family` the reconstructed CPUID family as hex (`0x19`), and
/// `kernels` is keyed by the lowercase format name (`ptq1_0`, `q1_0_g128`, ...).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchReport {
    pub cpu: &'static str,
    pub cpu_family: String,
    pub avx512_full_width: bool,
    pub kernels: Vec<(String, &'static str)>,
}

impl DispatchReport {
    /// Human-readable one-liner for a load banner, e.g.
    /// `[kernel] PTQ1_0=avx2 · zen4 · avx512_full_width=false`.
    pub fn banner(&self) -> String {
        let kernels = self
            .kernels
            .iter()
            .map(|(key, kernel)| format!("{}={}", key.to_ascii_uppercase(), kernel))
            .collect::<Vec<_>>()
            .join(" ");
        format!(
            "[kernel] {} · {} · avx512_full_width={}",
            kernels, self.cpu, self.avx512_full_width
        )
    }
}

/// Build the [`DispatchReport`] for the weight formats a model uses.
///
/// Formats with no reported selection are skipped rather than guessed at;
/// repeats collapse to the first occurrence.
pub fn dispatch_report(dtypes: impl IntoIterator<Item = GgmlDType>) -> DispatchReport {
    let mut kernels: Vec<(String, &'static str)> = Vec::new();
    for dtype in dtypes {
        let key = dtype.name().to_ascii_lowercase();
        if kernels.iter().any(|(seen, _)| *seen == key) {
            continue;
        }
        if let Some(kernel) = selected_kernel(dtype) {
            kernels.push((key, kernel));
        }
    }
    let (cpu, cpu_family) = cpu_label();
    DispatchReport {
        cpu,
        cpu_family,
        avx512_full_width: full_width_avx512(),
        kernels,
    }
}

/// Public form of [`avx512_full_width`]; `false` where the features are absent
/// or the target is not x86_64.
pub fn full_width_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        avx512_full_width()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Short CPU class label and the reconstructed family as hex.
///
/// 19h is shared by Zen 3 and Zen 4, so the two are told apart by whether
/// AVX-512 is actually present rather than by the family alone.
fn cpu_label() -> (&'static str, String) {
    #[cfg(target_arch = "x86_64")]
    {
        let is_amd = cpu_is_amd();
        let family = cpu_family();
        let has_avx512 = std::arch::is_x86_feature_detected!("avx512f");
        let label = match (is_amd, family) {
            (true, AMD_FAMILY_SPLIT_AVX512) if has_avx512 => "zen4",
            (true, AMD_FAMILY_SPLIT_AVX512) => "zen3",
            (true, 0x1A) => "zen5",
            (true, _) => "amd",
            (false, 6) => "intel",
            (false, _) => "other",
        };
        (label, format!("{:#04x}", family))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        ("other", "?".to_string())
    }
}

#[cfg(test)]
mod cpu_decode {
    use super::*;

    // Synthetic leaf-1 `EAX`: base family in bits 8..12, extended family in
    // bits 20..28. These are the values the real parts report.
    const EAX_ZEN4: u32 = (0xF << 8) | (0xA << 20); // 19h - Ryzen 7 8840HS
    const EAX_ZEN3: u32 = (0xF << 8) | (0xA << 20); // 19h too - same family
    const EAX_ZEN5: u32 = (0xF << 8) | (0xB << 20); // 1Ah
    const EAX_BULLDOZER: u32 = 0xF << 8; // 15h, extended 0
    const EAX_INTEL: u32 = 6 << 8; // family 6

    /// The bug this module exists to prevent. A real Zen 4 part reports 0xF in
    /// the 4-bit family field, so any test written against that field — 19h or
    /// its decimal twin 25 — answers "no" on every Zen 4 machine. The
    /// reconstructed family is what actually matches.
    #[test]
    fn raw_family_field_can_never_match_19h() {
        let raw = (EAX_ZEN4 >> 8) & 0xF;
        assert_eq!(raw, 0xF);
        assert_ne!(raw, 0x19);
        assert_ne!(raw, 25);
        assert_eq!(cpuid_family(EAX_ZEN4), AMD_FAMILY_SPLIT_AVX512);
    }

    #[test]
    fn reconstructs_extended_families_and_passes_through_base_ones() {
        assert_eq!(cpuid_family(EAX_ZEN4), 0x19);
        assert_eq!(cpuid_family(EAX_ZEN5), 0x1A);
        // Zen 3 shares the 19h family with Zen 4 - the two are told apart by the
        // features they advertise, not by the family, which is why the AVX-512
        // check also requires the feature bits.
        assert_eq!(cpuid_family(EAX_ZEN3), 0x19);
        assert_eq!(cpuid_family(EAX_BULLDOZER), 0xF);
        assert_eq!(cpuid_family(EAX_INTEL), 6);
    }

    /// The decision the way the crate makes it: leaf-1 `EAX` in, kernel choice out.
    fn full_width(is_amd: bool, eax: u32, has_f: bool, has_bw: bool) -> bool {
        avx512_full_width_for(is_amd, cpuid_family(eax), has_f, has_bw)
    }

    #[test]
    fn only_amd_family_19h_loses_full_width_avx512() {
        // This box: AuthenticAMD 19h with AVX-512 present.
        assert!(!full_width(true, EAX_ZEN4, true, true));
        // Zen 5 keeps it, so the exclusion must stay narrow.
        assert!(full_width(true, EAX_ZEN5, true, true));
        // Intel keeps it, including an Intel part that reported 19h.
        assert!(full_width(false, EAX_INTEL, true, true));
        assert!(avx512_full_width_for(false, 0x19, true, true));
        // Zen 3 is 19h too but has no AVX-512, so the family check never decides it.
        assert!(!full_width(true, EAX_ZEN3, false, false));
        // The features are still required on any vendor.
        assert!(!full_width(false, EAX_INTEL, true, false));
        assert!(!full_width(false, EAX_INTEL, false, true));
        assert!(!full_width(false, EAX_INTEL, false, false));
    }

    #[test]
    fn decodes_the_vendor_string() {
        let amd = (
            u32::from_le_bytes(*b"Auth"),
            u32::from_le_bytes(*b"cAMD"),
            u32::from_le_bytes(*b"enti"),
        );
        assert!(cpuid_is_amd(amd.0, amd.1, amd.2));
        let intel = (
            u32::from_le_bytes(*b"Geni"),
            u32::from_le_bytes(*b"ntel"),
            u32::from_le_bytes(*b"ineI"),
        );
        assert!(!cpuid_is_amd(intel.0, intel.1, intel.2));
    }

    /// Every hot weight dtype must be able to say which kernel it landed on, so a
    /// future regression shows up in a log line rather than a 34% slowdown.
    #[test]
    fn every_hot_weight_dtype_reports_a_kernel() {
        for dtype in [
            GgmlDType::PTQ1_0,
            GgmlDType::PQ2_0,
            GgmlDType::Q2_0,
            GgmlDType::Q1_0,
            GgmlDType::Q10G128,
        ] {
            let name = selected_kernel(dtype);
            assert!(name.is_some(), "{dtype:?} reports no kernel");
            assert!(!name.unwrap_or_default().is_empty());
        }
        assert_eq!(selected_kernel(GgmlDType::F32), None);
    }
    #[test]
    fn dispatch_report_dedupes_and_keeps_first_seen_order() {
        let report = dispatch_report([
            GgmlDType::PTQ1_0,
            GgmlDType::F32,
            GgmlDType::Q1_0,
            GgmlDType::PTQ1_0,
            GgmlDType::F32,
        ]);
        // Untracked and repeated formats drop; keys are lowercase for JSON.
        assert_eq!(report.kernels.len(), 2, "{:?}", report.kernels);
        assert_eq!(report.kernels[0].0, "ptq1_0");
        assert_eq!(report.kernels[1].0, "q1_0");
        for (_, kernel) in &report.kernels {
            assert!(!kernel.is_empty());
        }
        assert!(!report.cpu.is_empty());
        // A model whose formats are all untracked reports no kernels, not a guess.
        let empty = dispatch_report([GgmlDType::F32, GgmlDType::F16]);
        assert!(empty.kernels.is_empty());
    }

    #[test]
    fn banner_names_every_selected_kernel() {
        let report = dispatch_report([GgmlDType::PTQ1_0, GgmlDType::PQ2_0]);
        let banner = report.banner();
        assert!(banner.starts_with("[kernel] "), "{banner}");
        assert!(banner.contains("PTQ1_0="), "{banner}");
        assert!(banner.contains("PQ2_0="), "{banner}");
        assert!(
            banner.contains(&format!("avx512_full_width={}", report.avx512_full_width)),
            "{banner}"
        );
        assert!(banner.contains(report.cpu), "{banner}");
    }
}
