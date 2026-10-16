//! Small shared helpers.
//!
//! `constant_time_eq` lives here rather than beside its first user because a
//! second copy is how a security-relevant comparison drifts: one caller
//! eventually "optimises" theirs with `==` and early exit, and nothing notices.
//! Both the GCM tag check and the Finished MAC check use this one.

/// Compares two byte strings without leaking where they differ.
///
/// No early exit and no data-dependent memory access: every byte of the shorter
/// length is always read, and the result is a single OR accumulation. The length
/// check itself is not secret (our lengths are public: a tag is 16 bytes, a
/// Finished MAC is 32), so returning early on a length mismatch leaks nothing.
#[inline]
pub(crate) fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_and_unequal_are_reported() {
        assert!(constant_time_eq(&[], &[]));
        assert!(constant_time_eq(b"hearth", b"hearth"));
        assert!(!constant_time_eq(b"hearth", b"hearta"));
        assert!(!constant_time_eq(b"hearth", b"hearrh"));
        assert!(!constant_time_eq(b"hearth", b"earth"));
        // A difference in the first byte and in the last byte must both fail.
        assert!(!constant_time_eq(&[0x00, 1, 2, 3], &[0x01, 1, 2, 3]));
        assert!(!constant_time_eq(&[0x00, 1, 2, 3], &[0x00, 1, 2, 4]));
        // Length mismatch, in both directions.
        assert!(!constant_time_eq(b"hearth", b"hearth!"));
        assert!(!constant_time_eq(b"hearth!", b"hearth"));
    }

    /// Every single-bit change must be detected, so no bit position is skipped
    /// by an off-by-one in the accumulation.
    #[test]
    fn every_bit_matters() {
        let mut base = [0u8; 32];
        for (i, b) in base.iter_mut().enumerate() {
            *b = i as u8;
        }
        assert!(constant_time_eq(&base, &base));
        for byte in 0..base.len() {
            for bit in 0..8 {
                let mut other = base;
                other[byte] ^= 1 << bit;
                assert!(!constant_time_eq(&base, &other), "byte {byte} bit {bit}");
            }
        }
    }
}
