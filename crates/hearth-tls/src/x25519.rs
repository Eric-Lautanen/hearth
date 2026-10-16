//! X25519 (RFC 7748) — the key exchange TLS 1.3 uses with `x25519` named groups.
//!
//! Arithmetic is over `GF(2^255 - 19)` on five 51-bit limbs, which is the
//! standard representation: limb products fit in `u128` with room to spare, and
//! 51 bits keeps the reduction shifts cheap. Everything is branchless and
//! table-free — no data-dependent lookup in the ladder, no early return in the
//! zero check — so a secret scalar cannot be recovered through timing or
//! cache behaviour.
//!
//! # Fail closed on small-order points
//!
//! RFC 7748 §6.1 requires that an implementation check the output of the
//! scalar multiplication and **abort** if it is all zero, which is what a
//! small-order peer public key produces. [`diffie_hellman`] therefore returns a
//! `Result` and has no unchecked twin, and [`x25519`] exists only because the
//! published vectors need the raw function. Cross-checked against OpenSSL,
//! which likewise refuses these points rather than returning zeros.
//!
//! Note that clamping is applied to the scalar by this module (RFC 7748 §5), so
//! callers must not clamp as well and expect a different result — the test
//! `clamping_is_applied_to_the_scalar` pins the exact bits that are forced.

/// Mask for the 51-bit limbs.
const MASK51: u64 = (1 << 51) - 1;

/// `(A - 2) / 4` for Curve25519, the ladder constant (RFC 7748 §5).
const A24: u64 = 121665;

/// `2 * p` in 51-bit limbs, used to keep subtraction non-negative.
const TWO_P: [u64; 5] = [
    0x000F_FFFF_FFFF_FFDA,
    0x000F_FFFF_FFFF_FFFE,
    0x000F_FFFF_FFFF_FFFE,
    0x000F_FFFF_FFFF_FFFE,
    0x000F_FFFF_FFFF_FFFE,
];

/// The X25519 base point, `u = 9` (RFC 7748 §6).
pub const BASEPOINT: [u8; 32] = {
    let mut b = [0u8; 32];
    b[0] = 9;
    b
};

/// Why a shared secret was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedSecretError {
    /// The scalar multiplication produced zero. The peer sent a small-order
    /// point, so the "shared" secret is a value the attacker chose: using it
    /// would hand them the session key. The handshake must abort.
    LowOrderPoint,
}

impl std::fmt::Display for SharedSecretError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LowOrderPoint => write!(f, "peer public key is a low-order point"),
        }
    }
}

impl std::error::Error for SharedSecretError {}

/// A field element modulo `2^255 - 19`, as five 51-bit limbs, little-endian.
#[derive(Clone, Copy)]
struct Fe([u64; 5]);

impl Fe {
    const ZERO: Self = Self([0; 5]);
    const ONE: Self = Self([1, 0, 0, 0, 0]);

    /// Loads a 32-byte little-endian value, ignoring bit 255 as RFC 7748 §5
    /// requires ("mask the most significant bit in the final byte").
    fn from_bytes(bytes: &[u8; 32]) -> Self {
        let load = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().expect("8 bytes"));
        Self([
            load(0) & MASK51,
            (load(6) >> 3) & MASK51,
            (load(12) >> 6) & MASK51,
            (load(19) >> 1) & MASK51,
            (load(24) >> 12) & MASK51,
        ])
    }

    /// Serialises to 32 bytes little-endian, fully reduced modulo `p`.
    fn to_bytes(self) -> [u8; 32] {
        let mut l = self.0;
        let mut carry = l[0] >> 51;
        l[0] &= MASK51;
        l[1] += carry;
        carry = l[1] >> 51;
        l[1] &= MASK51;
        l[2] += carry;
        carry = l[2] >> 51;
        l[2] &= MASK51;
        l[3] += carry;
        carry = l[3] >> 51;
        l[3] &= MASK51;
        l[4] += carry;
        carry = l[4] >> 51;
        l[4] &= MASK51;
        l[0] += carry * 19;
        carry = l[0] >> 51;
        l[0] &= MASK51;
        l[1] += carry;

        // `l` is now below 2^255 + 19. Adding 19 and looking at the carry out of
        // the top limb tells us whether a subtraction of p is needed; the
        // select is a mask so it does not branch on secret data.
        let mut t = [l[0] + 19, l[1], l[2], l[3], l[4]];
        let mut c = t[0] >> 51;
        t[0] &= MASK51;
        t[1] += c;
        c = t[1] >> 51;
        t[1] &= MASK51;
        t[2] += c;
        c = t[2] >> 51;
        t[2] &= MASK51;
        t[3] += c;
        c = t[3] >> 51;
        t[3] &= MASK51;
        t[4] += c;
        c = t[4] >> 51;
        t[4] &= MASK51;

        let mask = c.wrapping_neg();
        let r = [
            (t[0] & mask) | (l[0] & !mask),
            (t[1] & mask) | (l[1] & !mask),
            (t[2] & mask) | (l[2] & !mask),
            (t[3] & mask) | (l[3] & !mask),
            (t[4] & mask) | (l[4] & !mask),
        ];

        let w0 = r[0] | (r[1] << 51);
        let w1 = (r[1] >> 13) | (r[2] << 38);
        let w2 = (r[2] >> 26) | (r[3] << 25);
        let w3 = (r[3] >> 39) | (r[4] << 12);
        let mut out = [0u8; 32];
        out[0..8].copy_from_slice(&w0.to_le_bytes());
        out[8..16].copy_from_slice(&w1.to_le_bytes());
        out[16..24].copy_from_slice(&w2.to_le_bytes());
        out[24..32].copy_from_slice(&w3.to_le_bytes());
        out
    }

    fn add(&self, o: &Self) -> Self {
        let r = [
            self.0[0] + o.0[0],
            self.0[1] + o.0[1],
            self.0[2] + o.0[2],
            self.0[3] + o.0[3],
            self.0[4] + o.0[4],
        ];
        Self(carry_all(r))
    }

    /// `self - o`, computed as `self + 2p - o` so no limb ever goes negative.
    /// Inputs must have limbs below `2^51 + 64`, which every operation here
    /// maintains.
    fn sub(&self, o: &Self) -> Self {
        let r = [
            self.0[0] + TWO_P[0] - o.0[0],
            self.0[1] + TWO_P[1] - o.0[1],
            self.0[2] + TWO_P[2] - o.0[2],
            self.0[3] + TWO_P[3] - o.0[3],
            self.0[4] + TWO_P[4] - o.0[4],
        ];
        Self(carry_all(r))
    }

    fn mul(&self, o: &Self) -> Self {
        let a0 = self.0[0] as u128;
        let a1 = self.0[1] as u128;
        let a2 = self.0[2] as u128;
        let a3 = self.0[3] as u128;
        let a4 = self.0[4] as u128;
        let b0 = o.0[0] as u128;
        let b1 = o.0[1] as u128;
        let b2 = o.0[2] as u128;
        let b3 = o.0[3] as u128;
        let b4 = o.0[4] as u128;

        // Limbs above 2^255 wrap: the 19 folds them back, since 2^255 = 19 mod p.
        let b1_19 = b1 * 19;
        let b2_19 = b2 * 19;
        let b3_19 = b3 * 19;
        let b4_19 = b4 * 19;

        let mut r0 = a0 * b0 + a1 * b4_19 + a2 * b3_19 + a3 * b2_19 + a4 * b1_19;
        let mut r1 = a0 * b1 + a1 * b0 + a2 * b4_19 + a3 * b3_19 + a4 * b2_19;
        let mut r2 = a0 * b2 + a1 * b1 + a2 * b0 + a3 * b4_19 + a4 * b3_19;
        let mut r3 = a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0 + a4 * b4_19;
        let mut r4 = a0 * b4 + a1 * b3 + a2 * b2 + a3 * b1 + a4 * b0;

        let mut c = r0 >> 51;
        r0 &= MASK51 as u128;
        r1 += c;
        c = r1 >> 51;
        r1 &= MASK51 as u128;
        r2 += c;
        c = r2 >> 51;
        r2 &= MASK51 as u128;
        r3 += c;
        c = r3 >> 51;
        r3 &= MASK51 as u128;
        r4 += c;
        c = r4 >> 51;
        r4 &= MASK51 as u128;
        r0 += c * 19;
        c = r0 >> 51;
        r0 &= MASK51 as u128;
        r1 += c;

        Self([r0 as u64, r1 as u64, r2 as u64, r3 as u64, r4 as u64])
    }

    fn sq(&self) -> Self {
        self.mul(self)
    }

    /// Multiplies by a small constant (used for the `a24` ladder term).
    fn mul_small(&self, s: u64) -> Self {
        let s = s as u128;
        let mut r0 = self.0[0] as u128 * s;
        let mut r1 = self.0[1] as u128 * s;
        let mut r2 = self.0[2] as u128 * s;
        let mut r3 = self.0[3] as u128 * s;
        let mut r4 = self.0[4] as u128 * s;

        let mut c = r0 >> 51;
        r0 &= MASK51 as u128;
        r1 += c;
        c = r1 >> 51;
        r1 &= MASK51 as u128;
        r2 += c;
        c = r2 >> 51;
        r2 &= MASK51 as u128;
        r3 += c;
        c = r3 >> 51;
        r3 &= MASK51 as u128;
        r4 += c;
        c = r4 >> 51;
        r4 &= MASK51 as u128;
        r0 += c * 19;
        c = r0 >> 51;
        r0 &= MASK51 as u128;
        r1 += c;

        Self([r0 as u64, r1 as u64, r2 as u64, r3 as u64, r4 as u64])
    }

    /// `self^(p - 2)`, i.e. the inverse, by the standard addition chain
    /// (2^255 - 21). Zero maps to zero, so callers must ensure the input is
    /// non-zero — the ladder's `z2` never is for a valid scalar.
    fn invert(&self) -> Self {
        let z2 = self.sq();
        let z4 = z2.sq();
        let z8 = z4.sq();
        let z9 = z8.mul(self);
        let z11 = z9.mul(&z2);
        let z22 = z11.sq();
        let z_5_0 = z22.mul(&z9);
        let z_10_5 = z_5_0.sq_n(5);
        let z_10_0 = z_10_5.mul(&z_5_0);
        let z_20_10 = z_10_0.sq_n(10);
        let z_20_0 = z_20_10.mul(&z_10_0);
        let z_40_20 = z_20_0.sq_n(20);
        let z_40_0 = z_40_20.mul(&z_20_0);
        let z_50_10 = z_40_0.sq_n(10);
        let z_50_0 = z_50_10.mul(&z_10_0);
        let z_100_50 = z_50_0.sq_n(50);
        let z_100_0 = z_100_50.mul(&z_50_0);
        let z_200_100 = z_100_0.sq_n(100);
        let z_200_0 = z_200_100.mul(&z_100_0);
        let z_250_50 = z_200_0.sq_n(50);
        let z_250_0 = z_250_50.mul(&z_50_0);
        z_250_0.sq_n(5).mul(&z11)
    }

    /// `self^(2^n)`.
    fn sq_n(&self, n: u32) -> Self {
        let mut out = *self;
        for _ in 0..n {
            out = out.sq();
        }
        out
    }

    /// True if the value is zero once reduced. Constant time: it looks at every
    /// byte and accumulates, instead of returning early.
    fn is_zero(&self) -> bool {
        let bytes = self.to_bytes();
        let mut acc = 0u8;
        for b in bytes {
            acc |= b;
        }
        acc == 0
    }
}

/// Carries limbs back under 2^51, folding the top limb's overflow by 19. Two
/// passes: the fold can push limb 0 back over the boundary once.
#[inline]
fn carry_all(mut l: [u64; 5]) -> [u64; 5] {
    for _ in 0..2 {
        let mut c = l[0] >> 51;
        l[0] &= MASK51;
        l[1] += c;
        c = l[1] >> 51;
        l[1] &= MASK51;
        l[2] += c;
        c = l[2] >> 51;
        l[2] &= MASK51;
        l[3] += c;
        c = l[3] >> 51;
        l[3] &= MASK51;
        l[4] += c;
        c = l[4] >> 51;
        l[4] &= MASK51;
        l[0] += c * 19;
        c = l[0] >> 51;
        l[0] &= MASK51;
        l[1] += c;
    }
    l
}

/// Clamps the scalar per RFC 7748 §5: clear the low 3 bits, clear bit 255, set
/// bit 254. Keeps the value in the prime-order subgroup and fixes the top bit
/// so the ladder's bit loop has a constant length.
#[inline]
fn clamp(scalar: &[u8; 32]) -> [u8; 32] {
    let mut k = *scalar;
    k[0] &= 248;
    k[31] &= 127;
    k[31] |= 64;
    k
}

/// Bit `i` of the little-endian byte string.
#[inline]
fn bit(bytes: &[u8; 32], i: usize) -> u64 {
    ((bytes[i >> 3] >> (i & 7)) & 1) as u64
}

/// Swaps `a` and `b` if `swap` is 1, in constant time (no branch, no
/// data-dependent memory access).
#[inline]
fn cswap(swap: u64, a: &mut Fe, b: &mut Fe) {
    let mask = swap.wrapping_neg();
    for (x, y) in a.0.iter_mut().zip(b.0.iter_mut()) {
        let t = mask & (*x ^ *y);
        *x ^= t;
        *y ^= t;
    }
}

/// The Montgomery ladder (RFC 7748 §5), returning `x2` before inversion.
fn ladder(scalar: &[u8; 32], u: &[u8; 32]) -> Fe {
    let k = clamp(scalar);
    let x1 = Fe::from_bytes(u);
    let mut x2 = Fe::ONE;
    let mut z2 = Fe::ZERO;
    let mut x3 = x1;
    let mut z3 = Fe::ONE;
    let mut swap = 0u64;

    for t in (0..255).rev() {
        let k_t = bit(&k, t);
        swap ^= k_t;
        cswap(swap, &mut x2, &mut x3);
        cswap(swap, &mut z2, &mut z3);
        swap = k_t;

        let a = x2.add(&z2);
        let aa = a.sq();
        let b = x2.sub(&z2);
        let bb = b.sq();
        let e = aa.sub(&bb);
        let c = x3.add(&z3);
        let d = x3.sub(&z3);
        let da = d.mul(&a);
        let cb = c.mul(&b);
        x3 = da.add(&cb).sq();
        z3 = x1.mul(&da.sub(&cb).sq());
        x2 = aa.mul(&bb);
        z2 = e.mul(&aa.add(&e.mul_small(A24)));
    }

    cswap(swap, &mut x2, &mut x3);
    cswap(swap, &mut z2, &mut z3);
    x2.mul(&z2.invert())
}

/// Raw X25519, as defined in RFC 7748 §5.
///
/// This returns the all-zero output for a small-order peer key, because the
/// published vectors and the RFC's `decodeScalar25519`/`decodeUCoordinate`
/// helpers are defined that way. **Protocol code must use
/// [`diffie_hellman`]**, which refuses that case instead of agreeing on a
/// secret the peer chose.
pub fn x25519(scalar: &[u8; 32], u: &[u8; 32]) -> [u8; 32] {
    ladder(scalar, u).to_bytes()
}

/// The public key for a private scalar: `X25519(scalar, 9)`.
pub fn public_key(scalar: &[u8; 32]) -> [u8; 32] {
    x25519(scalar, &BASEPOINT)
}

/// Diffie-Hellman, refusing a small-order peer key.
///
/// An all-zero result means the peer sent a point of small order, so the shared
/// secret is attacker-chosen: returning `Ok` would let the peer dictate the
/// session key. RFC 7748 §6.1 requires the check; TLS 1.3 requires the abort.
pub fn diffie_hellman(
    scalar: &[u8; 32],
    peer_public: &[u8; 32],
) -> Result<[u8; 32], SharedSecretError> {
    let shared = ladder(scalar, peer_public);
    if shared.is_zero() {
        return Err(SharedSecretError::LowOrderPoint);
    }
    Ok(shared.to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(s: &str) -> Vec<u8> {
        let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(cleaned.len().is_multiple_of(2), "odd hex length");
        (0..cleaned.len() / 2)
            .map(|i| u8::from_str_radix(&cleaned[i * 2..i * 2 + 2], 16).expect("hex"))
            .collect()
    }

    fn b32(s: &str) -> [u8; 32] {
        h(s).try_into().expect("32 bytes")
    }

    /// Every constant in this module was re-derived from the independent
    /// OpenSSL-backed `cryptography` package before being pinned here; the
    /// script that did it also asserts the published RFC 7748 values, so the
    /// oracle and the spec are checked against each other, not just against me.
    #[test]
    fn rfc7748_scalar_multiplication_vector_1() {
        let out = x25519(
            &b32("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4"),
            &b32("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c"),
        );
        assert_eq!(
            out[..],
            h("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552")[..]
        );
    }

    #[test]
    fn rfc7748_scalar_multiplication_vector_2() {
        let out = x25519(
            &b32("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d"),
            &b32("e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493"),
        );
        assert_eq!(
            out[..],
            h("95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957")[..]
        );
    }

    /// RFC 7748 §6.1: both parties must derive the same secret, and the two
    /// public keys are the published ones.
    #[test]
    fn rfc7748_diffie_hellman_example() {
        let alice_sk = b32("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        let bob_sk = b32("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb");
        let alice_pub = public_key(&alice_sk);
        let bob_pub = public_key(&bob_sk);

        assert_eq!(
            alice_pub[..],
            h("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a")[..]
        );
        assert_eq!(
            bob_pub[..],
            h("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f")[..]
        );

        let shared_a = diffie_hellman(&alice_sk, &bob_pub).expect("alice");
        let shared_b = diffie_hellman(&bob_sk, &alice_pub).expect("bob");
        assert_eq!(shared_a, shared_b);
        assert_eq!(
            shared_a[..],
            h("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742")[..]
        );
    }

    /// RFC 7748 §5.2's iterated test, which exercises the ladder repeatedly and
    /// would catch a stateful mistake that a single call hides.
    #[test]
    fn rfc7748_iterated_test() {
        let mut k = b32("0900000000000000000000000000000000000000000000000000000000000000");
        let mut u = k;
        for i in 1..=1000 {
            let next = x25519(&k, &u);
            u = k;
            k = next;
            if i == 1 {
                assert_eq!(
                    k[..],
                    h("422c8e7a6227d7bca1350b3e2bb7279f7897b87bb6854b783c60e80311ae3079")[..]
                );
            }
        }
        assert_eq!(
            k[..],
            h("684cf59ba83309552800ef566f2f4d3c1c3887c49360e3875f2eb94d99532c51")[..]
        );
    }

    /// Random vectors produced by the oracle, so the suite is not limited to
    /// the published cases (which were chosen by the RFC authors, not by a
    /// fuzzer).
    #[test]
    fn matches_oracle_on_random_inputs() {
        let cases = [
            (
                "fe8dedbb8b99a6935d5cd0d675727f5ad28e1dc611a2874261476793725f815d",
                "165c60ae748b05b37608a261bd16aa775ea58652e76de152373658479188aef4",
                "c7d8f4d6772c058cd35f7c1545564410b585c672f5d0dd7d0e6b6fe5d13abe7c",
            ),
            (
                "2fa472287d77f0d2cdc1f45186c60b5b784d29a7abc73e455af344816e961dd1",
                "55223986de2ba32e3b42c14243935ccdf794931fc8593a3502426a11e36ee6a5",
                "1e2fe76d4415ec398ed195798d31bc7d0e9a8bc8049c5f69aa9b137b69a1f16e",
            ),
            (
                "2cc29afbfe58f309488272210aedabf8d4aff4b7508aa6f9db34d7229ed119e5",
                "1b62cef77c16c70eb21a56f152cf61b28f45032ce55f93fcac81815d40d6b993",
                "7a178c36d051a836d10e79e5d13d8d0ba98badad0e7d0132345c81495277385b",
            ),
        ];
        for (k, u, expected) in cases {
            assert_eq!(x25519(&b32(k), &b32(u))[..], h(expected)[..], "u = {}", u);
        }
    }

    /// RFC 7748 §6.1: small-order points produce an all-zero secret, and the
    /// protocol-facing API must refuse rather than return it. OpenSSL rejects
    /// these as well.
    #[test]
    fn low_order_points_fail_closed() {
        let secret = b32("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a");
        let small_order = [
            "0000000000000000000000000000000000000000000000000000000000000000",
            "0100000000000000000000000000000000000000000000000000000000000000",
            "ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f",
            // The other order-4/order-8 points from the RFC's list.
            "0000000000000000000000000000000000000000000000000000000000000080",
            "e0eb7a7c3b41b8ae1656e3faf19fc46ada098deb9c32b1fd866205165f49b800",
        ];
        for u in small_order {
            let peer = b32(u);
            assert!(
                diffie_hellman(&secret, &peer).is_err(),
                "accepted low-order u = {}",
                u
            );
            // The raw function is defined to return zeros for these, which is
            // exactly the value that must never reach a KDF.
            assert_eq!(x25519(&secret, &peer), [0u8; 32], "u = {}", u);
        }
    }

    /// Clamping is applied inside this module: the low 3 bits and bit 255 of the
    /// scalar are ignored, and bit 254 is forced on.
    #[test]
    fn clamping_is_applied_to_the_scalar() {
        let u = b32("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
        let base = b32("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
        let expected = x25519(&base, &u);

        let mut low_bits = base;
        low_bits[0] |= 0b0000_0111; // cleared by the mask
        assert_eq!(x25519(&low_bits, &u), expected);

        let mut high_bit = base;
        high_bit[31] |= 0b1000_0000; // cleared by the mask
        assert_eq!(x25519(&high_bit, &u), expected);

        let mut bit_254 = base;
        bit_254[31] ^= 0b0100_0000; // forced on either way
        assert_eq!(x25519(&bit_254, &u), expected);
    }

    /// Bit 255 of the u-coordinate must be ignored (RFC 7748 §5), so the byte
    /// strings for `u` and `u + 2^255` are the same point.
    #[test]
    fn high_bit_of_u_coordinate_is_ignored() {
        let secret = b32("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4");
        let u = b32("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c");
        let mut u_high = u;
        u_high[31] |= 0b1000_0000;
        assert_eq!(x25519(&secret, &u), x25519(&secret, &u_high));
    }

    /// Field arithmetic sanity: `from_bytes`/`to_bytes` round-trip, and the
    /// ring identities hold on values that stress the reduction (p-1, p, p+1).
    #[test]
    fn field_element_round_trip_and_reduction() {
        let p_minus_1 = b32("ecffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f");
        assert_eq!(Fe::from_bytes(&p_minus_1).to_bytes(), p_minus_1);

        // p itself must reduce to zero.
        let p = b32("edffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff7f");
        assert!(Fe::from_bytes(&p).is_zero());

        // A value with bit 255 set is the same field element as without it.
        let mut masked = p_minus_1;
        masked[31] |= 0b1000_0000;
        assert_eq!(Fe::from_bytes(&masked).to_bytes(), p_minus_1);

        let a = Fe::from_bytes(&b32(
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        ));
        assert_eq!(a.mul(&Fe::ONE).to_bytes(), a.to_bytes());
        assert_eq!(a.mul(&Fe::ONE).add(&Fe::ZERO).to_bytes(), a.to_bytes());
        assert_eq!(a.mul(&a.invert()).to_bytes(), Fe::ONE.to_bytes());

        // (a + 1)^2 == a^2 + 2a + 1
        let lhs = a.add(&Fe::ONE).sq();
        let rhs = a.sq().add(&a.add(&a)).add(&Fe::ONE);
        assert_eq!(lhs.to_bytes(), rhs.to_bytes());

        // mul_small agrees with a full multiplication by the same constant.
        let a24 = {
            let mut b = [0u8; 32];
            b[..8].copy_from_slice(&A24.to_le_bytes());
            b
        };
        assert_eq!(
            a.mul_small(A24).to_bytes(),
            a.mul(&Fe::from_bytes(&a24)).to_bytes()
        );

        // and b + (p - b) == 0, which is where a broken subtraction shows up.
        let b = Fe::from_bytes(&b32(
            "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        ));
        assert!(b.add(&Fe::ZERO.sub(&b)).is_zero());
    }
}
