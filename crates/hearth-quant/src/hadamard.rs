//! The Walsh-Hadamard fold the `qwen35` (hybrid) models use.
//!
//! The fold is `R = D·H`: a per-element sign vector `D` (±1) followed by the
//! normalized Sylvester-Walsh-Hadamard transform `H` applied to each
//! `block_size`-element block of a row. The inverse fold `R⁻¹ = H·D` is the
//! same transform with the two steps the other way round, and the embedding
//! lookup needs it because the stored rows are already folded by `R`.
//!
//! The vector path is **bit-identical** to the scalar one, and that is a
//! requirement, not a nicety: the folded weights are quantized, so a changed
//! rounding anywhere in the fold changes the activation bytes the dot kernel
//! sees and the model starts emitting different tokens.
//!
//! * Every stage of the transform computes `a + b` and `a - b` on *disjoint*
//!   pairs, so the arithmetic each element sees is unchanged - only the order
//!   in which the pairs are visited changes.
//! * The normalization stays one multiply by `1/sqrt(n)` at the very end, with
//!   the same f32 constant the scalar path uses. Folding it into the stages
//!   would change the arithmetic.
//! * The sign pass stays a multiply by ±1.0, which is exact, so it is also
//!   bit-identical - it is the same multiply the scalar path did, down to the
//!   sign bit of a zero.
//!
//! The transform itself has an AVX2 path; the sign pass uses [`wide`], which
//! is portable (NEON on aarch64) and more than fast enough for one multiply
//! per element.

use wide::f32x8;

/// Multiply a row by its sign vector, in place.
///
/// The signs are `+1.0` or `-1.0`, so the product is exact: this is a sign
/// flip, not a rounding step, and `-0.0 * 1.0` stays `-0.0`.
pub fn apply_signs_in_place(x: &mut [f32], signs: &[f32]) {
    debug_assert_eq!(signs.len(), x.len());
    let n = x.len();
    if !crate::scalar_only() {
        let chunks = n / 8;
        for i in 0..chunks {
            let vx = f32x8::from(&x[i * 8..(i + 1) * 8]);
            let vs = f32x8::from(&signs[i * 8..(i + 1) * 8]);
            let vo = vx * vs;
            x[i * 8..(i + 1) * 8].copy_from_slice(&vo.to_array());
        }
        for i in chunks * 8..n {
            x[i] *= signs[i];
        }
        return;
    }
    for (v, s) in x.iter_mut().zip(signs.iter()) {
        *v *= s;
    }
}

/// The normalized Walsh-Hadamard transform, in place, on one block.
///
/// `x.len()` must be a power of two. The scalar path is the reference; the
/// AVX2 path reproduces it bit for bit (see the module docs).
pub fn fwht_in_place(x: &mut [f32]) {
    let n = x.len();
    if n >= 8 && n.is_power_of_two() {
        #[cfg(target_arch = "x86_64")]
        {
            if !crate::scalar_only() && crate::avx2_available() {
                // SAFETY: the AVX2 path is entered only when the CPU reports
                // AVX2, and it reads and writes exactly `x.len()` elements.
                unsafe { avx2::fwht_in_place(x) };
                return;
            }
        }
    }
    fwht_in_place_scalar(x);
}

/// Fold `R = D·H`: the sign pass first, then the transform per block.
///
/// This is the transform the quantized weight rows were folded by, so an
/// activation feeding those weights is rotated the same way.
pub fn rotate_in_place(x: &mut [f32], signs: Option<&[f32]>, block_size: usize) {
    if let Some(sgn) = signs {
        debug_assert_eq!(sgn.len(), x.len());
        apply_signs_in_place(x, sgn);
    }
    for chunk in x.chunks_exact_mut(block_size) {
        fwht_in_place(chunk);
    }
}

/// Unfold `R⁻¹ = H·D`: the transform per block first, then the sign pass.
///
/// The opposite order from [`rotate_in_place`]: the stored rows are folded by
/// `R`, so a lookup has to invert it.
pub fn unrotate_in_place(x: &mut [f32], signs: Option<&[f32]>, block_size: usize) {
    for chunk in x.chunks_exact_mut(block_size) {
        fwht_in_place(chunk);
    }
    if let Some(sgn) = signs {
        debug_assert_eq!(sgn.len(), x.len());
        apply_signs_in_place(x, sgn);
    }
}

/// The scalar reference transform: the stages, then one normalization pass.
fn fwht_in_place_scalar(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let s = 1.0 / (n as f32).sqrt();
    let mut h = 1usize;
    while h < n {
        for i in 0..n {
            if i & h == 0 {
                let a = x[i];
                let b = x[i | h];
                x[i] = a + b;
                x[i | h] = a - b;
            }
        }
        h <<= 1;
    }
    for v in x.iter_mut() {
        *v *= s;
    }
}

/// The x86_64 transform.
///
/// Stages `h = 1, 2, 4` pair lanes inside one 8-lane group, so the partner of
/// lane `i` is lane `i ^ h` - a lane permutation, and the butterfly is a
/// blend of `v + p` and `p - v`. Stages `h >= 8` pair the low half of a
/// `2h`-element group with the high half at the same offset, which is a plain
/// load/add/store.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::*;

    /// The blend mask for the stage whose lanes have `bit` set: those lanes
    /// take `p - v`, the others `v + p`. `bit` is the index of the bit that
    /// the stage flips - 0 for `h = 1`, 1 for `h = 2`, 2 for `h = 4`.
    const fn mask_bits(bit: u32) -> [i32; 8] {
        let mut lanes = [0i32; 8];
        let mut i = 0usize;
        while i < 8 {
            lanes[i] = if (i >> bit) & 1 != 0 { -1 } else { 0 };
            i += 1;
        }
        lanes
    }

    /// The mask as a vector: all-ones per selected lane, all-zeros elsewhere.
    ///
    /// # Safety
    /// The CPU must support AVX2 (see [`crate::avx2_available`]).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn lane_mask(lanes: [i32; 8]) -> __m256 {
        _mm256_castsi256_ps(_mm256_setr_epi32(
            lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7],
        ))
    }

    /// The butterfly stage `h = 1`: the partner of lane `i` is lane `i ^ 1`,
    /// i.e. the lane permutation [1,0,3,2,5,4,7,6].
    ///
    /// Lanes with bit 0 set take `p - v`, the others `v + p`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn stage_h1(ptr: *mut f32, n: usize, mask: __m256) {
        let idx = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(ptr.add(i));
            let p = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(_mm256_castps_si256(v), idx));
            let sum = _mm256_add_ps(v, p);
            let dif = _mm256_sub_ps(p, v);
            _mm256_storeu_ps(ptr.add(i), _mm256_blendv_ps(sum, dif, mask));
            i += 8;
        }
    }

    /// The butterfly stage `h = 2`: the partner of lane `i` is lane `i ^ 2`,
    /// i.e. the lane permutation [2,3,0,1,6,7,4,5].
    ///
    /// Lanes with bit 1 set take `p - v`, the others `v + p`.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn stage_h2(ptr: *mut f32, n: usize, mask: __m256) {
        let idx = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(ptr.add(i));
            let p = _mm256_castsi256_ps(_mm256_permutevar8x32_epi32(_mm256_castps_si256(v), idx));
            let sum = _mm256_add_ps(v, p);
            let dif = _mm256_sub_ps(p, v);
            _mm256_storeu_ps(ptr.add(i), _mm256_blendv_ps(sum, dif, mask));
            i += 8;
        }
    }

    /// A butterfly stage whose partner lane is in the other 128-bit half.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn stage_half(ptr: *mut f32, n: usize, mask: __m256) {
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(ptr.add(i));
            let p = _mm256_permute2f128_ps(v, v, 0x01);
            let sum = _mm256_add_ps(v, p);
            let dif = _mm256_sub_ps(p, v);
            _mm256_storeu_ps(ptr.add(i), _mm256_blendv_ps(sum, dif, mask));
            i += 8;
        }
    }

    /// The transform.
    ///
    /// # Safety
    /// The CPU must support AVX2 (see [`crate::avx2_available`]).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn fwht_in_place(x: &mut [f32]) {
        let n = x.len();
        let ptr = x.as_mut_ptr();

        // Stage h = 1: partner lane is `i ^ 1`, i.e. [1,0,3,2] inside each
        // 4-lane group. Lanes with bit 0 set take `p - v`, the others `v + p`.
        stage_h1(ptr, n, lane_mask(mask_bits(0)));
        // Stage h = 2: partner lane is `i ^ 2`, i.e. [2,3,0,1] inside each
        // 4-lane group. Lanes with bit 1 set take `p - v`.
        stage_h2(ptr, n, lane_mask(mask_bits(1)));
        // Stage h = 4: partner lane is `i ^ 4`, i.e. the two 128-bit halves
        // swapped. Lanes with bit 2 set take `p - v`.
        stage_half(ptr, n, lane_mask(mask_bits(2)));

        // Stages h >= 8: the low half of each 2h group pairs with the high
        // half at the same offset.
        let mut h = 8usize;
        while h < n {
            let mut base = 0usize;
            while base < n {
                let mut off = 0usize;
                while off < h {
                    let a = _mm256_loadu_ps(ptr.add(base + off));
                    let b = _mm256_loadu_ps(ptr.add(base + h + off));
                    _mm256_storeu_ps(ptr.add(base + off), _mm256_add_ps(a, b));
                    _mm256_storeu_ps(ptr.add(base + h + off), _mm256_sub_ps(a, b));
                    off += 8;
                }
                base += 2 * h;
            }
            h <<= 1;
        }

        // One normalization pass at the end, with the same f32 constant the
        // scalar path uses.
        let s = _mm256_set1_ps(1.0 / (n as f32).sqrt());
        let mut i = 0usize;
        while i + 8 <= n {
            let v = _mm256_loadu_ps(ptr.add(i));
            _mm256_storeu_ps(ptr.add(i), _mm256_mul_ps(v, s));
            i += 8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic pseudo-random row, so the tests do not depend on a
    /// random number generator's implementation.
    fn pseudo_random(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect()
    }

    /// The AVX2 transform must reproduce the scalar one bit for bit.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_bit_for_bit() {
        if !crate::avx2_available() {
            return;
        }
        for n in [8usize, 16, 64, 1024, 2048] {
            let src = pseudo_random(n, 0x51ed_0000 ^ n as u32);
            let mut scalar = src.clone();
            fwht_in_place_scalar(&mut scalar);
            let mut vector = src;
            fwht_in_place(&mut vector);
            assert_eq!(
                scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "n = {n}"
            );
        }
    }

    /// A row that is not a multiple of the block size still transforms
    /// block by block, and the sign pass runs over the whole row.
    #[test]
    fn rotate_is_the_sign_pass_then_the_transform() {
        let signs: Vec<f32> = (0..64)
            .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
            .collect();
        let src = pseudo_random(64, 7);
        let mut scalar = src.clone();
        for (v, s) in scalar.iter_mut().zip(signs.iter()) {
            *v *= s;
        }
        for chunk in scalar.as_chunks_mut::<32>().0 {
            fwht_in_place_scalar(chunk);
        }
        let mut vector = src;
        rotate_in_place(&mut vector, Some(&signs), 32);
        assert_eq!(
            scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// The inverse fold is the transform first and the sign pass second.
    #[test]
    fn unrotate_is_the_transform_then_the_sign_pass() {
        let signs: Vec<f32> = (0..32)
            .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
            .collect();
        let src = pseudo_random(32, 11);
        let mut scalar = src.clone();
        fwht_in_place_scalar(&mut scalar);
        for (v, s) in scalar.iter_mut().zip(signs.iter()) {
            *v *= s;
        }
        let mut vector = src;
        unrotate_in_place(&mut vector, Some(&signs), 32);
        assert_eq!(
            scalar.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            vector.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// The transform is its own inverse: the normalized WHT applied twice
    /// gives the original row back.
    #[test]
    fn fwht_twice_is_the_identity() {
        let src = pseudo_random(64, 13);
        let mut x = src.clone();
        fwht_in_place(&mut x);
        fwht_in_place(&mut x);
        for (a, b) in x.iter().zip(src.iter()) {
            assert!((a - b).abs() <= b.abs() * 1e-4 + 1e-5, "{a} vs {b}");
        }
    }

    /// The sign pass is exact: multiplying by ±1 flips the sign and leaves
    /// the magnitude alone. `-0.0 * 1.0` stays `-0.0`, and `-0.0 * -1.0` is
    /// `+0.0` - the multiply, not a sign-bit flip, is what the scalar path
    /// does, and the two differ on a signed zero.
    #[test]
    fn sign_pass_is_exact() {
        let signs = vec![1.0f32, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let mut x = vec![0.0f32, -0.0, 1.5, -1.5, 2.0, -2.0, 0.25, -0.25];
        apply_signs_in_place(&mut x, &signs);
        assert_eq!(x[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(x[1].to_bits(), (-0.0f32).to_bits());
        assert_eq!(&x[2..], &[-1.5, 1.5, 2.0, 2.0, 0.25, 0.25]);

        // The same zero under a -1 sign comes out positive, like the
        // multiply it replaces.
        let mut y = vec![-0.0f32; 8];
        apply_signs_in_place(&mut y, &[-1.0f32; 8]);
        assert_eq!(y[0].to_bits(), 0.0f32.to_bits());
    }
}
