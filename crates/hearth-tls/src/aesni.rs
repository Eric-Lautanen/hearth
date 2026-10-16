//! AES-NI block encryption (x86_64).
//!
//! The point of this module is not cleverness: `_mm_aesenc_si128` performs
//! `ShiftRows → SubBytes → MixColumns → AddRoundKey` in hardware, which is both
//! far faster than the software path and — unlike the software S-box, a 256-byte
//! table — **free of data-dependent cache accesses**. On any x86_64 CPU that
//! reports the `aes` feature, the timing caveat documented in `aes.rs` does not
//! apply to this path.
//!
//! The key schedule is *not* recomputed here: AES-128 key expansion is the same
//! arithmetic either way, so this module consumes the round keys the software
//! implementation already produced, which keeps one implementation of the
//! schedule and therefore one thing to get right.
//!
//! Selected at runtime by [`available`]; the software path remains the
//! reference and is what runs on every other architecture (including macOS on
//! Apple silicon, whose crypto extensions are a further slice).

use std::arch::x86_64::{
    __m128i, _mm_aesenc_si128, _mm_aesenclast_si128, _mm_loadu_si128, _mm_storeu_si128,
    _mm_xor_si128,
};

/// True if this CPU implements AES-NI. Cheap enough to call per key setup, and
/// cached by `std` after the first probe.
#[inline]
pub(crate) fn available() -> bool {
    std::arch::is_x86_feature_detected!("aes")
}

/// Encrypts one block using the expanded round keys.
///
/// # Safety
///
/// Requires the `aes` CPU feature, which the caller must have confirmed with
/// [`available`]; calling it on a CPU without AES-NI is undefined behaviour
/// (it will execute an illegal instruction).
#[target_feature(enable = "aes")]
pub(crate) unsafe fn encrypt_block(round_keys: &[u8; 176], block: &mut [u8; 16]) {
    let load_key = |round: usize| unsafe {
        _mm_loadu_si128(round_keys.as_ptr().add(round * 16) as *const __m128i)
    };

    let mut state = unsafe { _mm_loadu_si128(block.as_ptr() as *const __m128i) };
    state = _mm_xor_si128(state, load_key(0));
    for round in 1..10 {
        state = _mm_aesenc_si128(state, load_key(round));
    }
    state = _mm_aesenclast_si128(state, load_key(10));

    unsafe { _mm_storeu_si128(block.as_mut_ptr() as *mut __m128i, state) };
}
