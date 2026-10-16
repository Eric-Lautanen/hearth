//! PCLMULQDQ carry-less multiplication for GHASH (x86_64).
//!
//! # Why there is no magic constant in this file
//!
//! The usual implementation of this is a byte-swap plus a 128-bit constant
//! (`0xc200000000000000` and friends) that nobody can re-derive from the code.
//! That constant *is* the key exchange between PCLMULQDQ's bit order and GCM's,
//! and getting it wrong produces plausible-looking garbage, so it is worth
//! spelling out the reasoning instead:
//!
//! 1. GCM (SP 800-38D §6.3) reads a 128-bit block with the **leftmost** bit as
//!    the coefficient of `x^0`, and its field is `GF(2^128)` modulo
//!    `x^128 + x^7 + x^2 + x + 1`. In that reading the block's bit index from
//!    the left *is* the degree, so the modulus relation is
//!    `x^128 ≡ x^7 + x^2 + x + 1`.
//! 2. PCLMULQDQ multiplies with the **rightmost** bit of each 64-bit lane as the
//!    coefficient of `x^0`.
//! 3. Reversing the 128 bits of a block maps one reading onto the other, and
//!    reversal commutes with multiplication (it is `P ↦ x^d·P(1/x)`, so
//!    `rev(A·B) = rev(A)·rev(B)`). Therefore: reverse the operands, multiply
//!    with PCLMULQDQ, reverse the result — and the same modulus relation from
//!    (1) still holds in the reversed domain.
//! 4. In the reversed domain, multiplying by `x^k` is a **left shift by `k`
//!    bits**, so the reduction becomes pure integer arithmetic:
//!    split the 256-bit product as `P = P_lo + P_hi·x^128`, then
//!    `P ≡ P_lo ^ (P_hi · (x^7 + x^2 + x + 1))`, i.e. XOR the shifted-and-XORed
//!    `P_hi` back in. `P_hi`'s degree is at most 126, so one round leaves at
//!    most degree 133 → the next round's high part is at most 6 bits → two
//!    rounds finish, and the result is already the canonical degree ≤ 127
//!    representative that the software path returns.
//!
//! So the only SIMD needed is the 128×128 carry-less product (three
//! PCLMULQDQ, Karatsuba) and the reduction is shifts and XORs — no constants to
//! trust, and the differential test against the software path pins the whole
//! chain.
//!
//! Still one thing to remember: unlike AES-NI, PCLMULQDQ does not remove a
//! table lookup because there is no lookup to remove — the software GHASH was
//! already branchless and table-free. This is purely throughput.

use std::arch::x86_64::{
    _mm_clmulepi64_si128, _mm_cvtsi128_si64, _mm_set_epi64x, _mm_unpackhi_epi64,
};

/// True if this CPU implements PCLMULQDQ.
#[inline]
pub(crate) fn available() -> bool {
    std::arch::is_x86_feature_detected!("pclmulqdq")
}

/// Multiplies two field elements in GCM's block convention.
///
/// Both arguments and the result are the same 128-bit values the software
/// implementation produces; the bit reversal in and out is what makes
/// PCLMULQDQ's lane ordering irrelevant to the caller.
///
/// # Safety
///
/// Requires the `pclmulqdq` CPU feature, confirmed by the caller via
/// [`available`].
pub(crate) unsafe fn multiply(x: u128, y: u128) -> u128 {
    unsafe { reverse128(multiply_reversed(reverse128(x), reverse128(y))) }
}

/// Reverses the 128 bits of a value: bit `i` goes to bit `127 - i`.
///
/// Public within the crate because the record layer wants to keep GHASH's whole
/// accumulator in one domain rather than converting per block.
pub(crate) fn reverse128(v: u128) -> u128 {
    const M1: u128 = 0x5555_5555_5555_5555_5555_5555_5555_5555;
    const M2: u128 = 0x3333_3333_3333_3333_3333_3333_3333_3333;
    const M4: u128 = 0x0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f_0f0f;
    const M8: u128 = 0x00ff_00ff_00ff_00ff_00ff_00ff_00ff_00ff;
    const M16: u128 = 0x0000_ffff_0000_ffff_0000_ffff_0000_ffff;
    const M32: u128 = 0x0000_0000_ffff_ffff_0000_0000_ffff_ffff;

    let mut x = v;
    x = ((x & M1) << 1) | ((x >> 1) & M1);
    x = ((x & M2) << 2) | ((x >> 2) & M2);
    x = ((x & M4) << 4) | ((x >> 4) & M4);
    x = ((x & M8) << 8) | ((x >> 8) & M8);
    x = ((x & M16) << 16) | ((x >> 16) & M16);
    x = ((x & M32) << 32) | ((x >> 32) & M32);
    x.rotate_right(64)
}

/// The multiply-and-reduce in the reversed domain (see the module docs).
///
/// # Safety
///
/// Requires the `pclmulqdq` CPU feature.
#[target_feature(enable = "pclmulqdq")]
unsafe fn multiply_reversed(a: u128, b: u128) -> u128 {
    let (mut lo, mut hi) = unsafe { clmul(a, b) };

    // Two folds: p ≡ p_lo ^ (p_hi · 0x87), where 0x87 = x^7 + x^2 + x + 1.
    for _ in 0..2 {
        let (folded, carry) = multiply_by_087(hi);
        lo ^= folded;
        hi = carry;
    }
    debug_assert_eq!(hi, 0, "second fold must clear the high part");

    lo
}

/// Computes `v · (x^7 + x^2 + x + 1)` as `(low 128 bits, overflowing high bits)`.
///
/// In the reversed domain `·x^k` is `<< k`, so this is four shifts and XORs.
#[inline]
fn multiply_by_087(v: u128) -> (u128, u128) {
    let lo = (v << 7) ^ (v << 2) ^ (v << 1) ^ v;
    // Bits shifted past 127 by each term, collected into degrees 128..134.
    let hi = (v >> 121) ^ (v >> 126) ^ (v >> 127);
    (lo, hi)
}

/// The 128×128 carry-less product as `(low, high)`, by Karatsuba on the two
/// 64-bit halves.
///
/// # Safety
///
/// Requires the `pclmulqdq` CPU feature.
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64, (a >> 64) as u64);
    let (b0, b1) = (b as u64, (b >> 64) as u64);

    let p00 = unsafe { clmul64(a0, b0) };
    let p11 = unsafe { clmul64(a1, b1) };
    // (a0 ^ a1)(b0 ^ b1) = p00 ^ p11 ^ middle, so middle is recoverable.
    let middle = unsafe { clmul64(a0 ^ a1, b0 ^ b1) } ^ p00 ^ p11;

    (p00 ^ (middle << 64), (middle >> 64) ^ p11)
}

/// 64×64 carry-less multiply into a 128-bit result.
///
/// # Safety
///
/// Requires the `pclmulqdq` CPU feature.
#[target_feature(enable = "pclmulqdq")]
unsafe fn clmul64(a: u64, b: u64) -> u128 {
    // SSE2-only extraction: `_mm_extract_epi64` would need sse4.1, and the
    // feature gate here is pclmulqdq alone. No inner `unsafe` block: the whole
    // body of an `unsafe fn` with the feature enabled is already one.
    let va = _mm_set_epi64x(0, a as i64);
    let vb = _mm_set_epi64x(0, b as i64);
    let product = _mm_clmulepi64_si128(va, vb, 0x00);
    let lo = _mm_cvtsi128_si64(product) as u64;
    let hi = _mm_cvtsi128_si64(_mm_unpackhi_epi64(product, product)) as u64;
    ((hi as u128) << 64) | lo as u128
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gcm;

    /// The reversal is its own inverse and moves the end bits.
    #[test]
    fn reverse128_is_an_involution() {
        assert_eq!(reverse128(0), 0);
        assert_eq!(reverse128(1), 1u128 << 127);
        assert_eq!(reverse128(1u128 << 127), 1);
        for v in [
            0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128,
            u128::MAX,
            u128::MAX - 1,
        ] {
            assert_eq!(reverse128(reverse128(v)), v);
        }
    }

    /// `multiply_by_087` must agree with a plain carry-less multiply by 0x87.
    #[test]
    fn multiply_by_087_matches_a_shift_free_definition() {
        for v in [
            1u128,
            0x8000_0000_0000_0000_0000_0000_0000_0000u128,
            0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128,
            u128::MAX,
        ] {
            let (lo, hi) = multiply_by_087(v);
            // Reconstruct by the definition: for every set bit, add the four
            // shifted copies. XOR, not OR — adding polynomials in GF(2) is
            // XOR, and two shifted copies genuinely collide (writing OR here
            // made this test fail against a correct implementation).
            let mut expected = 0u128;
            let mut expected_hi = 0u128;
            for bit in 0..128 {
                if (v >> bit) & 1 == 1 {
                    for k in [0u32, 1, 2, 7] {
                        let target = bit as u32 + k;
                        if target < 128 {
                            expected ^= 1u128 << target;
                        } else {
                            expected_hi ^= 1u128 << (target - 128);
                        }
                    }
                }
            }
            assert_eq!((lo, hi), (expected, expected_hi), "v = {v:#x}");
        }
    }

    /// The decisive test: the hardware multiply must equal the software one,
    /// which is itself pinned by the published GCM vectors. Random-ish inputs
    /// chosen to include the awkward shapes (all ones, single high bit, a value
    /// whose product needs both folds).
    #[test]
    fn hardware_multiply_matches_software() {
        if !available() {
            eprintln!("skipping: no pclmulqdq on this CPU");
            return;
        }
        let mut state = 0x243f_6a88_85a3_08d3u128;
        let mut next = || {
            // xorshift128+, so the values are reproducible without a dependency.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };

        let mut values = vec![
            0u128,
            1,
            u128::MAX,
            1u128 << 127,
            0x87,
            0xc200_0000_0000_0000_0000_0000_0000_0000u128,
        ];
        // Kept small on purpose: the software side is a 128-step bit loop, so
        // the test suite's debug profile makes a larger cross product slow
        // without buying coverage (the shapes above are the interesting ones).
        for _ in 0..28 {
            values.push(next());
            values.push(1u128 << (next() % 128));
        }

        for &x in &values {
            for &y in &values {
                let soft = gcm::multiply_software(x, y);
                let hard = unsafe { multiply(x, y) };
                assert_eq!(hard, soft, "mul({x:#034x}, {y:#034x})");
            }
        }
    }
}
