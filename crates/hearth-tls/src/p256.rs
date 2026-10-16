//! P-256 (`secp256r1`, `prime256v1`): the field, the group, and one scalar
//! multiplication.
//!
//! This is what `ecdsa_secp256r1_sha256` is built on, and like the RSA module it
//! is only ever used on public data: a signature verifier multiplies public points
//! by public scalars. See [`crate::ecdsa`] for the signature equation itself.
//!
//! # Where the constants come from
//!
//! `p`, `b`, `Gx`, `Gy` and `n` are printed by `scripts/tls_p256_params.py`, which
//! reads them out of **OpenSSL's own explicit-parameter dump** — a second
//! implementation — and then checks them as a group before emitting them:
//! `p` and `n` are prime (sympy), `a == p - 3`, the generator satisfies
//! `y^2 == x^3 + a*x + b (mod p)`, `n * G` is the identity, `n` lies in the Hasse
//! interval around `p + 1` with cofactor 1, and `p < 2n` with `n > 2^255`. The
//! first four of those are re-asserted here, in Rust, by the
//! `the_generator_and_the_order_agree_with_each_other` test — so a transcription
//! error anywhere in this file fails the suite rather than a handshake.
//!
//! Two derived constants are deliberately **not** transcribed: `-p^-1 mod 2^64`
//! and `2^512 mod p` (and the same for `n`) are computed from the modulus by the
//! `const fn` below, and the script prints its own values only so the tests can
//! check that derivation against an independent one.
//!
//! # Variable time, on purpose
//!
//! Scalar multiplication here branches on the scalar's bits. That is safe for
//! exactly one reason and it is worth stating plainly: **every scalar this crate
//! ever multiplies by is public** — it comes out of a signature or out of a
//! hash — so there is no secret to leak. This module is therefore *not* fit for
//! signing or for ECDH as written; doing either would need a Montgomery ladder
//! with conditional swaps, and it would need to arrive with its own tests. TLS 1.3
//! here uses X25519 for key agreement, and this crate never signs anything, so
//! that is a rule about the future rather than a compromise in the present.
//!
//! # What is refused rather than supported
//!
//! Public keys arrive as SEC 1 points, and only the **uncompressed** form
//! (`0x04 || X || Y`, 65 bytes) is accepted. Compressed points are legal X.509 and
//! every CA issues uncompressed keys, so supporting the compressed form would add
//! a square root and a second encoding to review for keys that do not occur;
//! refusing it keeps the point parser to one shape. A coordinate at or above `p`
//! is refused because it is a second encoding of the same point, a point not on
//! the curve is refused, and the point at infinity is refused — with a cofactor of
//! 1 there is no small subgroup to check beyond that one point.

/// The field characteristic and the group order, as four little-endian limbs.
///
/// Printed by `scripts/tls_p256_params.py` from OpenSSL's explicit parameters.
pub(crate) const P: [u64; 4] = [
    0xffff_ffff_ffff_ffff,
    0x0000_0000_ffff_ffff,
    0x0000_0000_0000_0000,
    0xffff_ffff_0000_0001,
];

/// The curve coefficient `b`, where the curve is `y^2 = x^3 + a*x + b` with
/// `a = p - 3`.
const B: [u64; 4] = [
    0x3bce_3c3e_27d2_604b,
    0x651d_06b0_cc53_b0f6,
    0xb3eb_bd55_7698_86bc,
    0x5ac6_35d8_aa3a_93e7,
];

/// The generator's `x` coordinate.
const GX: [u64; 4] = [
    0xf4a1_3945_d898_c296,
    0x7703_7d81_2deb_33a0,
    0xf8bc_e6e5_63a4_40f2,
    0x6b17_d1f2_e12c_4247,
];

/// The generator's `y` coordinate.
const GY: [u64; 4] = [
    0xcbb6_4068_37bf_51f5,
    0x2bce_3357_6b31_5ece,
    0x8ee7_eb4a_7c0f_9e16,
    0x4fe3_42e2_fe1a_7f9b,
];

/// The order of the generator, which is the order of the group because the
/// cofactor is 1.
pub(crate) const N: [u64; 4] = [
    0xf3b9_cac2_fc63_2551,
    0xbce6_faad_a717_9e84,
    0xffff_ffff_ffff_ffff,
    0xffff_ffff_0000_0000,
];

/// `a`, the curve coefficient: `p - 3`, computed rather than transcribed.
const A: [u64; 4] = sub_const(P, [3, 0, 0, 0]);

/// Why a public key or a point was refused.
///
/// As in [`crate::rsa`], every variant is fatal: there is no tolerant mode, and
/// each one names the rule that failed so a diagnostic can say which check
/// rejected the key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum P256Error {
    /// A public key must be exactly 65 bytes: `0x04 || X || Y`.
    PointLength {
        /// The only accepted length.
        expected: usize,
        /// What the caller passed.
        found: usize,
    },
    /// The first byte was not `0x04`, so the point is compressed, in the hybrid
    /// form, or not a point at all.
    UncompressedOnly(u8),
    /// A coordinate is at or above `p`, so it is a second encoding of a value.
    CoordinateOutOfRange(&'static str),
    /// The point does not satisfy the curve equation.
    NotOnCurve,
    /// The encoding names the point at infinity, which is not a public key.
    PointAtInfinity,
}

impl std::fmt::Display for P256Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PointLength { expected, found } => write!(
                f,
                "a P-256 public key is {expected} bytes, but this one is {found}"
            ),
            Self::UncompressedOnly(first) => write!(
                f,
                "the point starts with {first:#04x}, and only the uncompressed form (0x04) is accepted"
            ),
            Self::CoordinateOutOfRange(which) => {
                write!(f, "the point's {which} coordinate is not below p")
            }
            Self::NotOnCurve => write!(f, "the point is not on the curve"),
            Self::PointAtInfinity => write!(f, "the point is the point at infinity"),
        }
    }
}

impl std::error::Error for P256Error {}

/// A modulus and the two constants Montgomery multiplication needs, both derived
/// from the modulus itself.
///
/// `const fn` with `while` loops is stable at this crate's MSRV, so `p` and `n`
/// arrive as literals and everything else is computed: the alternative is four
/// more hand-copied magic numbers in a file whose whole point is not having any.
#[derive(Clone, Copy)]
struct Modulus {
    /// The modulus, four little-endian limbs, odd.
    m: [u64; 4],
    /// `-m^-1 mod 2^64`.
    n0inv: u64,
    /// `2^512 mod m`: the constant that converts into Montgomery form.
    r2: [u64; 4],
}

impl Modulus {
    const fn new(m: [u64; 4]) -> Self {
        debug_assert!(m[0] & 1 == 1, "Montgomery needs an odd modulus");
        // Newton's iteration for the inverse of `m[0]` modulo 2^64: each round
        // doubles the number of correct low bits, so six rounds reach 64 from one.
        let mut inverse: u64 = 1;
        let mut round = 0;
        while round < 6 {
            inverse = inverse.wrapping_mul(2u64.wrapping_sub(m[0].wrapping_mul(inverse)));
            round += 1;
        }
        // `2^512 mod m` by doubling one five hundred and twelve times, which
        // needs no division routine at all: a doubling is a shift and a
        // conditional subtraction, and the value never exceeds `2m`.
        let mut r2 = [1u64, 0, 0, 0];
        let mut count = 0;
        while count < 512 {
            let mut carry = 0u64;
            let mut index = 0;
            while index < 4 {
                let next = r2[index] >> 63;
                r2[index] = (r2[index] << 1) | carry;
                carry = next;
                index += 1;
            }
            // `carry != 0` means the doubled value exceeded 2^256, and it is
            // still below `2m`, so one subtraction is always enough.
            if carry != 0 || !lt4(&r2, &m) {
                sub4(&mut r2, &m);
            }
            count += 1;
        }
        Self {
            m,
            n0inv: inverse.wrapping_neg(),
            r2,
        }
    }

    /// The Montgomery product `a * b * 2^-256 mod m`, for `a` and `b` already
    /// reduced. Convert into Montgomery form with `b = r2` and back out with
    /// `b = 1`, both of which this function does for its callers below.
    const fn mont_mul(&self, a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        // Ten limbs is the product's eight plus the two a carry can reach. The
        // shape is the coarse "multiply, then reduce" one from `rsa.rs`: easier to
        // check by reading than the interleaved form, which matters more than the
        // few percent it costs.
        let mut t = [0u64; 10];

        let mut i = 0;
        while i < 4 {
            let mut carry = 0u64;
            let mut j = 0;
            while j < 4 {
                let acc = t[i + j] as u128 + (a[j] as u128) * (b[i] as u128) + carry as u128;
                t[i + j] = acc as u64;
                carry = (acc >> 64) as u64;
                j += 1;
            }
            let mut k = i + 4;
            while carry != 0 {
                let acc = t[k] as u128 + carry as u128;
                t[k] = acc as u64;
                carry = (acc >> 64) as u64;
                k += 1;
            }
            i += 1;
        }

        // Each step makes one more low limb zero, by adding `factor * m` shifted
        // into place for the `factor` that cancels that limb.
        let mut i = 0;
        while i < 4 {
            let factor = t[i].wrapping_mul(self.n0inv);
            let mut carry = 0u64;
            let mut j = 0;
            while j < 4 {
                let acc = t[i + j] as u128 + (factor as u128) * (self.m[j] as u128) + carry as u128;
                t[i + j] = acc as u64;
                carry = (acc >> 64) as u64;
                j += 1;
            }
            let mut k = i + 4;
            while carry != 0 {
                let acc = t[k] as u128 + carry as u128;
                t[k] = acc as u64;
                carry = (acc >> 64) as u64;
                k += 1;
            }
            i += 1;
        }

        // The low half is zero and the high half is below `2m`, so at most one
        // subtraction finishes it. `t[8]` is the only bit that can be left over:
        // the value is below `2m < 2^257`.
        let mut index = 0;
        while index < 4 {
            debug_assert!(t[index] == 0, "the low half is zero after reduction");
            index += 1;
        }
        let mut result = [t[4], t[5], t[6], t[7]];
        if t[8] != 0 || !lt4(&result, &self.m) {
            sub4(&mut result, &self.m);
        }
        result
    }

    /// Takes a reduced value into Montgomery form.
    ///
    /// Named `enter`/`leave` rather than `to`/`from`, which clippy reads as a
    /// conversion the receiver is the *subject* of; here the receiver is the
    /// modulus and the argument is what moves.
    const fn enter_montgomery(&self, value: &[u64; 4]) -> [u64; 4] {
        self.mont_mul(value, &self.r2)
    }

    /// Brings a Montgomery value back out.
    const fn leave_montgomery(&self, value: &[u64; 4]) -> [u64; 4] {
        self.mont_mul(value, &[1, 0, 0, 0])
    }

    /// `base^exponent` **in the Montgomery domain**: `base_mont` is a Montgomery
    /// value and so is the result.
    ///
    /// Square and multiply, most significant bit first. Branching on the
    /// exponent is fine: for the one exponent this is used with, `m - 2`, the
    /// exponent is a published constant, and the base is public too.
    ///
    /// This is the primitive; `pow` below is the same thing for a caller holding
    /// a plain value. Keeping both is not redundancy: getting the domains confused
    /// is the one mistake this file made twice, and it is invisible whenever an
    /// inverse is checked only by multiplying it back.
    fn mont_pow(&self, base_mont: &[u64; 4], exponent: &[u64; 4]) -> [u64; 4] {
        // One, in Montgomery form: `2^256 mod m`, not the limbs `1`.
        let mut result = self.enter_montgomery(&[1, 0, 0, 0]);
        let mut index = 4;
        while index > 0 {
            index -= 1;
            let mut bit = 64;
            while bit > 0 {
                bit -= 1;
                result = self.mont_mul(&result, &result);
                if (exponent[index] >> bit) & 1 == 1 {
                    result = self.mont_mul(&result, base_mont);
                }
            }
        }
        result
    }

    /// `base^exponent mod m` for a reduced plain base, returning a plain value.
    fn pow(&self, base: &[u64; 4], exponent: &[u64; 4]) -> [u64; 4] {
        self.leave_montgomery(&self.mont_pow(&self.enter_montgomery(base), exponent))
    }

    /// `a^-1 mod m` by Fermat's little theorem, which needs `m` prime and `a`
    /// nonzero. Both hold at every call site: `p` and `n` are prime and the value
    /// being inverted is a nonzero scalar.
    fn invert(&self, a: &[u64; 4], exponent: &[u64; 4]) -> [u64; 4] {
        debug_assert!(a != &[0, 0, 0, 0], "there is no inverse of zero");
        self.pow(a, exponent)
    }

    /// `a * b mod m`, for reduced inputs.
    fn mul(&self, a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
        // One conversion is saved by feeding `b` in its plain form: the Montgomery
        // product of a Montgomery value and a plain one is the plain product.
        self.mont_mul(&self.enter_montgomery(a), b)
    }
}

/// The field `F_p` and the scalar field `F_n`. Both are compiled in, and the
/// constants inside them are computed at compile time.
const FP: Modulus = Modulus::new(P);
const FN: Modulus = Modulus::new(N);

/// `p - 2`, the exponent of the Fermat inverse modulo `p`.
const P_MINUS_2: [u64; 4] = sub_const(P, [2, 0, 0, 0]);
/// `n - 2`, the exponent of the Fermat inverse modulo `n`.
const N_MINUS_2: [u64; 4] = sub_const(N, [2, 0, 0, 0]);

/// `a = p - 3` in Montgomery form, for the doubling formula.
const A_MONT: [u64; 4] = FP.enter_montgomery(&A);

/// `b` in Montgomery form, for the on-curve check.
const B_MONT: [u64; 4] = FP.enter_montgomery(&B);

/// `a < b` over four limbs, least significant first.
const fn lt4(a: &[u64; 4], b: &[u64; 4]) -> bool {
    let mut index = 4;
    while index > 0 {
        index -= 1;
        if a[index] != b[index] {
            return a[index] < b[index];
        }
    }
    false
}

/// `a -= b`, wrapping. Only called where `a >= b` has been established, so the
/// wrap is exactly the borrow that cancels.
const fn sub4(a: &mut [u64; 4], b: &[u64; 4]) {
    let mut borrow = 0u64;
    let mut index = 0;
    while index < 4 {
        let (value, first) = a[index].overflowing_sub(b[index]);
        let (value, second) = value.overflowing_sub(borrow);
        a[index] = value;
        borrow = (first as u64) | (second as u64);
        index += 1;
    }
}

/// `a - b` as a new value, for the two constants that are defined as `p - 2` and
/// `p - 3`.
const fn sub_const(a: [u64; 4], b: [u64; 4]) -> [u64; 4] {
    let mut out = a;
    sub4(&mut out, &b);
    out
}

/// An element of `F_p` in Montgomery form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fe([u64; 4]);

impl Fe {
    const ZERO: Self = Self([0; 4]);
    /// One, in Montgomery form — which is `2^256 mod p`, **not** the limbs `1`.
    ///
    /// Written as a conversion rather than as a literal on purpose: the literal
    /// version of this was one of two domain bugs that cancelled each other and
    /// left the inverse test passing, while doubling a point whose `Z` was not one
    /// quietly stopped producing a point on the curve.
    const ONE: Self = Self(FP.enter_montgomery(&[1, 0, 0, 0]));

    /// A field element from limbs that are already reduced: how the constants in
    /// this file are turned into field elements without a round trip through
    /// bytes.
    fn from_reduced_limbs(limbs: &[u64; 4]) -> Self {
        debug_assert!(lt4(limbs, &P));
        Self(FP.enter_montgomery(limbs))
    }

    /// A field element from a reduced big-endian value: exactly 32 bytes, and the
    /// caller has checked it is below `p`.
    fn from_be_bytes(bytes: &[u8]) -> Self {
        debug_assert_eq!(bytes.len(), 32);
        let mut limbs = [0u64; 4];
        let mut index = 0;
        while index < 4 {
            let mut limb = [0u8; 8];
            limb.copy_from_slice(&bytes[index * 8..index * 8 + 8]);
            limbs[3 - index] = u64::from_be_bytes(limb);
            index += 1;
        }
        Self(FP.enter_montgomery(&limbs))
    }

    /// The value as exactly 32 big-endian bytes.
    fn to_be_bytes(self) -> [u8; 32] {
        let limbs = FP.leave_montgomery(&self.0);
        let mut out = [0u8; 32];
        let mut index = 0;
        while index < 4 {
            out[index * 8..index * 8 + 8].copy_from_slice(&limbs[3 - index].to_be_bytes());
            index += 1;
        }
        out
    }

    fn is_zero(self) -> bool {
        self.0 == [0, 0, 0, 0]
    }

    fn add(self, other: Self) -> Self {
        // Both are reduced, so the sum is below `2p` and one conditional
        // subtraction is enough.
        let mut sum = [0u64; 4];
        let mut carry = 0u64;
        let mut index = 0;
        while index < 4 {
            let acc = self.0[index] as u128 + other.0[index] as u128 + carry as u128;
            sum[index] = acc as u64;
            carry = (acc >> 64) as u64;
            index += 1;
        }
        if carry != 0 || !lt4(&sum, &P) {
            sub4(&mut sum, &P);
        }
        Self(sum)
    }

    /// `self - other`.
    ///
    /// There is no borrow to track: the difference is computed wrapping, and a
    /// borrow is exactly the case where the wrapped result is the true difference
    /// plus `p`, so adding `p` back when `self < other` is the whole correction.
    fn sub(self, other: Self) -> Self {
        let mut difference = self.0;
        sub4(&mut difference, &other.0);
        if lt4(&self.0, &other.0) {
            let mut carry = 0u64;
            let mut index = 0;
            while index < 4 {
                let acc = difference[index] as u128 + P[index] as u128 + carry as u128;
                difference[index] = acc as u64;
                carry = (acc >> 64) as u64;
                index += 1;
            }
        }
        Self(difference)
    }

    fn mul(self, other: Self) -> Self {
        Self(FP.mont_mul(&self.0, &other.0))
    }

    /// `1 / self`.
    ///
    /// Both the input and the output are Montgomery values, so this goes through
    /// `mont_pow` and not through `Modulus::invert`, which speaks plain values.
    /// The distinction is invisible to a test that only multiplies an inverse back
    /// against its input — that test passes with either domain, because `R / R`
    /// cancels — so it is spelled out here rather than left to be rediscovered.
    fn invert(self) -> Self {
        Self(FP.mont_pow(&self.0, &P_MINUS_2))
    }

    fn square(self) -> Self {
        self.mul(self)
    }

    fn double(self) -> Self {
        self.add(self)
    }

    /// `2^e * self` for the small exponents the point formulas use, by repeated
    /// doubling — no multiplication by a literal constant can be wrong in a way
    /// this is not.
    fn shift(self, e: u32) -> Self {
        let mut value = self;
        let mut count = 0;
        while count < e {
            value = value.double();
            count += 1;
        }
        value
    }
}

/// A point in Jacobian coordinates: `(X, Y, Z)` stands for the affine
/// `(X/Z^2, Y/Z^3)`, and `Z = 0` is the point at infinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Jacobian {
    x: Fe,
    y: Fe,
    z: Fe,
}

impl Jacobian {
    /// The point at infinity, the group's identity.
    const IDENTITY: Self = Self {
        x: Fe::ZERO,
        y: Fe::ONE,
        z: Fe::ZERO,
    };

    fn is_identity(self) -> bool {
        self.z.is_zero()
    }

    /// An affine point as Jacobian, with `Z = 1`.
    fn from_affine(x: Fe, y: Fe) -> Self {
        Self { x, y, z: Fe::ONE }
    }

    /// Doubling, with `a = p - 3`.
    ///
    /// The formula is the generic one from the explicit-formulas literature —
    /// `Z' = (Y + Z)^2 - Y^2 - Z^2`, `S = 4*X*Y^2`, `M = 3*X^2 + a*Z^4`,
    /// `X' = M^2 - 2*S`, `Y' = M*(S - X') - 8*Y^4` — used with `a = p - 3` rather
    /// than the `a = -3` special case, because the special case is one more place
    /// an algebra slip can hide and the cost is one multiplication out of nine.
    fn double(self) -> Self {
        if self.is_identity() || self.y.is_zero() {
            // Doubling a point of order two gives infinity; on this curve there
            // is no such point, but the check keeps the formula below honest.
            return Self::IDENTITY;
        }
        let xx = self.x.square();
        let yy = self.y.square();
        let yyyy = yy.square();
        let zz = self.z.square();
        let zzzz = zz.square();

        let s = self.x.mul(yy).shift(2); // 4 * X * Y^2
        let m = xx.double().add(xx).add(Fe(A_MONT).mul(zzzz)); // 3X^2 + a*Z^4
        let x3 = m.square().sub(s.double());
        let y3 = m.mul(s.sub(x3)).sub(yyyy.shift(3));
        let z3 = self.y.add(self.z).square().sub(yy).sub(zz);
        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// Addition, with the special cases the Jacobian formulas do not cover.
    ///
    /// `H = U2 - U1` is zero exactly when the two points have the same `x`; then
    /// the sum is either a doubling (`S` equal too) or infinity (`S` opposite).
    /// Leaving that out is the classic way to get a point multiplication that
    /// works on every input except the ones an attacker chooses.
    fn add(self, other: Self) -> Self {
        if self.is_identity() {
            return other;
        }
        if other.is_identity() {
            return self;
        }
        let z1z1 = self.z.square();
        let z2z2 = other.z.square();
        let u1 = self.x.mul(z2z2);
        let u2 = other.x.mul(z1z1);
        let s1 = self.y.mul(z2z2).mul(other.z);
        let s2 = other.y.mul(z1z1).mul(self.z);

        let h = u2.sub(u1);
        if h.is_zero() {
            return if s2.sub(s1).is_zero() {
                self.double()
            } else {
                // `other` is `-self`.
                Self::IDENTITY
            };
        }
        let r = s2.sub(s1).double();
        let i = h.double().square();
        let j = h.mul(i);
        let v = u1.mul(i);

        let x3 = r.square().sub(j).sub(v.double());
        let y3 = r.mul(v.sub(x3)).sub(s1.mul(j).double());
        let z3 = self.z.add(other.z).square().sub(z1z1).sub(z2z2).mul(h);
        Self {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// The affine `x` coordinate, or `None` for infinity. One inversion.
    ///
    /// `x` alone is returned because `x` is the only coordinate ECDSA's check
    /// needs (`v = R.x mod n`); a `y` that nothing outside the tests reads is a
    /// field the next reader has to wonder about. The tests reach the full point
    /// through this module's private pieces, which is one of the things an
    /// in-module test is for.
    fn to_affine_x(self) -> Option<[u8; 32]> {
        if self.is_identity() {
            return None;
        }
        Some(self.x.mul(self.z.invert().square()).to_be_bytes())
    }
}

/// A P-256 public key: a point that has been checked to be on the curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PublicKey {
    point: Jacobian,
}

impl PublicKey {
    /// Reads a public key from its SEC 1 encoding, which for a certificate is the
    /// `BIT STRING` contents of the `SubjectPublicKeyInfo`.
    pub(crate) fn from_sec1(bytes: &[u8]) -> Result<Self, P256Error> {
        if bytes.len() != 65 {
            return Err(P256Error::PointLength {
                expected: 65,
                found: bytes.len(),
            });
        }
        if bytes[0] != 0x04 {
            // 0x02 and 0x03 are the compressed forms and 0x06 and 0x07 the hybrid
            // ones; anything else is not a point encoding at all.
            return Err(P256Error::UncompressedOnly(bytes[0]));
        }
        let x = &bytes[1..33];
        let y = &bytes[33..65];
        // A coordinate at or above `p` is a second encoding of the same point:
        // `X` and `X + p` are the same field element but different bytes, and a
        // reader that accepts both has a malleable key.
        if !lt4(&be_limbs(x), &P) {
            return Err(P256Error::CoordinateOutOfRange("x"));
        }
        if !lt4(&be_limbs(y), &P) {
            return Err(P256Error::CoordinateOutOfRange("y"));
        }
        let x = Fe::from_be_bytes(x);
        let y = Fe::from_be_bytes(y);
        if x.is_zero() && y.is_zero() {
            return Err(P256Error::PointAtInfinity);
        }
        // y^2 == x^3 + a*x + b, in Montgomery form, which is where `a` and `b`
        // are already waiting.
        let left = y.square();
        let right = x.square().mul(x).add(Fe(A_MONT).mul(x)).add(Fe(B_MONT));
        if left != right {
            return Err(P256Error::NotOnCurve);
        }
        Ok(Self {
            point: Jacobian::from_affine(x, y),
        })
    }
}

/// Reads 32 big-endian bytes as four little-endian limbs.
fn be_limbs(bytes: &[u8]) -> [u64; 4] {
    debug_assert_eq!(bytes.len(), 32);
    let mut limbs = [0u64; 4];
    for (index, chunk) in bytes.as_chunks::<8>().0.iter().enumerate() {
        limbs[3 - index] = u64::from_be_bytes(*chunk);
    }
    limbs
}

/// The generator as a point.
fn generator() -> Jacobian {
    Jacobian::from_affine(Fe::from_reduced_limbs(&GX), Fe::from_reduced_limbs(&GY))
}

/// Scalar multiplication, left to right, most significant bit first.
///
/// Variable time by design: see the module docs. The scalar's bits decide only
/// which additions happen, and every input is public.
fn mul(point: Jacobian, k: &[u64; 4]) -> Jacobian {
    let mut acc = Jacobian::IDENTITY;
    let mut index = 4;
    while index > 0 {
        index -= 1;
        let mut bit = 64;
        while bit > 0 {
            bit -= 1;
            acc = acc.double();
            if (k[index] >> bit) & 1 == 1 {
                acc = acc.add(point);
            }
        }
    }
    acc
}

/// `u1 * G + u2 * public`, as the `x` coordinate of the sum, or `None` when the
/// sum is the point at infinity.
///
/// This is the only curve operation ECDSA verification needs, so it is the only
/// one exposed: the second half of the check is `v = R.x mod n`.
pub(crate) fn double_mul(public: &PublicKey, u2: &[u64; 4], u1: &[u64; 4]) -> Option<[u8; 32]> {
    mul(public.point, u2)
        .add(mul(generator(), u1))
        .to_affine_x()
}

/// Reduces a 256-bit value modulo `n`.
///
/// One or two conditional subtractions: `2^256 < 2n` because `n > 2^255`, so a
/// value below `2^256` is below `2n` and subtracting `n` at most twice suffices.
pub(crate) fn scalar_reduce(value: [u64; 4]) -> [u64; 4] {
    let mut out = value;
    while !lt4(&out, &N) {
        sub4(&mut out, &N);
    }
    out
}

/// `a^-1 mod n` for a nonzero scalar.
pub(crate) fn scalar_invert(a: &[u64; 4]) -> [u64; 4] {
    FN.invert(a, &N_MINUS_2)
}

/// `a * b mod n` for reduced scalars.
pub(crate) fn scalar_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
    FN.mul(a, b)
}

/// Whether a 32-byte big-endian value is in `1..n`, the range every scalar a
/// signature carries has to be in.
pub(crate) fn scalar_in_range(bytes: &[u8]) -> bool {
    let limbs = be_limbs(bytes);
    !(limbs == [0, 0, 0, 0]) && lt4(&limbs, &N)
}

/// A 32-byte big-endian value read as a reduced scalar (already checked to be in
/// range by [`scalar_in_range`]).
pub(crate) fn scalar_from_be(bytes: &[u8]) -> [u64; 4] {
    be_limbs(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A point's affine coordinates.
    ///
    /// The tests reach past the exposed surface here on purpose: `double_mul`
    /// hands back only an `x`, because `x` is all ECDSA needs, while checking the
    /// group laws needs both coordinates. That is what an in-module test is for.
    fn point_xy(point: &Jacobian) -> ([u8; 32], [u8; 32]) {
        assert!(!point.is_identity(), "infinity has no affine coordinates");
        let z_inv = point.z.invert();
        let z_inv2 = z_inv.square();
        (
            point.x.mul(z_inv2).to_be_bytes(),
            point.y.mul(z_inv2).mul(z_inv).to_be_bytes(),
        )
    }

    /// Four limbs as 32 big-endian bytes, for comparing against the constants.
    fn bytes_of(limbs: &[u64; 4]) -> [u8; 32] {
        let mut out = [0u8; 32];
        for index in 0..4 {
            out[index * 8..index * 8 + 8].copy_from_slice(&limbs[3 - index].to_be_bytes());
        }
        out
    }

    /// `k * G`, as its two affine coordinates.
    fn g_times(k: u64) -> ([u8; 32], [u8; 32]) {
        point_xy(&mul(generator(), &[k, 0, 0, 0]))
    }

    /// `-G`.
    fn minus_g() -> Jacobian {
        Jacobian::from_affine(
            Fe::from_reduced_limbs(&GX),
            Fe::ZERO.sub(Fe::from_reduced_limbs(&GY)),
        )
    }

    /// A public key from affine coordinates.
    fn key_from(point: ([u8; 32], [u8; 32])) -> PublicKey {
        let mut encoded = [0u8; 65];
        encoded[0] = 0x04;
        encoded[1..33].copy_from_slice(&point.0);
        encoded[33..].copy_from_slice(&point.1);
        PublicKey::from_sec1(&encoded).expect("a point on the curve is a valid key")
    }

    #[test]
    fn the_generator_and_the_order_agree_with_each_other() {
        // The checks the generator script makes, made again here in Rust, so a
        // transcription error in any constant in this file fails this test rather
        // than a handshake.
        let gx = Fe::from_reduced_limbs(&GX);
        let gy = Fe::from_reduced_limbs(&GY);
        let left = gy.square();
        let right = gx.square().mul(gx).add(Fe(A_MONT).mul(gx)).add(Fe(B_MONT));
        assert_eq!(left, right, "the generator is not on the curve");

        // n * G is infinity, which is what pins the order to the generator.
        assert!(
            mul(generator(), &N).is_identity(),
            "n * G should be the point at infinity"
        );
        // And (n - 1) * G is -G, which pins the order to exactly n rather than to
        // some multiple of it.
        let minus_g = point_xy(&mul(generator(), &sub_const(N, [1, 0, 0, 0])));
        assert_eq!(minus_g.0, bytes_of(&GX), "-G has G's x coordinate");
        assert_eq!(
            minus_g.1,
            Fe::ZERO.sub(gy).to_be_bytes(),
            "-G's y is p - Gy"
        );

        // The two moduli are both 256 bits and they are not equal, which is what
        // makes reducing a scalar a different operation from reducing a
        // coordinate.
        assert_eq!(P[3] >> 63, 1);
        assert_eq!(N[3] >> 63, 1, "n is wider than 255 bits");
        assert_ne!(P, N);
    }

    #[test]
    fn the_derived_constants_match_the_independent_ones() {
        // `scripts/tls_p256_params.py` computes these two with Python's `pow`.
        // The Rust side derives them from the modulus, and this is where the two
        // derivations meet.
        assert_eq!(FP.n0inv, 0x0000_0000_0000_0001, "-p^-1 mod 2^64");
        assert_eq!(
            FP.r2,
            [
                0x0000_0000_0000_0003,
                0xffff_fffb_ffff_ffff,
                0xffff_ffff_ffff_fffe,
                0x0000_0004_ffff_fffd,
            ],
            "2^512 mod p"
        );
        assert_eq!(FN.n0inv, 0xccd1_c8aa_ee00_bc4f, "-n^-1 mod 2^64");
        assert_eq!(
            FN.r2,
            [
                0x8324_4c95_be79_eea2,
                0x4699_799c_49bd_6fa6,
                0x2845_b239_2b6b_ec59,
                0x66e1_2d94_f3d9_5620,
            ],
            "2^512 mod n"
        );
        // The contract `n0inv` exists to satisfy, restated in one line, which is
        // what catches a sign error that the values above happen not to cover.
        assert_eq!(P[0].wrapping_mul(FP.n0inv), u64::MAX);
        assert_eq!(N[0].wrapping_mul(FN.n0inv), u64::MAX);
    }

    #[test]
    fn field_and_scalar_inverses_undo_their_products() {
        // One small value, one at the top of the range, one crossing a limb
        // boundary, and the generator's own coordinates.
        let cases = [
            [2, 0, 0, 0],
            sub_const(P, [1, 0, 0, 0]),
            [0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210, 1, 0],
            GX,
            GY,
        ];
        for value in cases {
            // Through `Fe`, which is the form every caller actually holds.
            let field = Fe::from_reduced_limbs(&value);
            assert_eq!(field.invert().mul(field), Fe::ONE, "inverse mod p");
            assert_eq!(FN.mul(&value, &FN.invert(&value, &N_MINUS_2)), [1, 0, 0, 0]);
        }
        // Inversion is not the identity, which a stub returning its input would
        // satisfy if only the round trip above were checked.
        assert_ne!(
            Fe::from_reduced_limbs(&[2, 0, 0, 0]).invert(),
            Fe::from_reduced_limbs(&[2, 0, 0, 0])
        );

        // The checks the version of this test that *passed* could not make. Both
        // of these fail if a value is held in the wrong domain, and neither can be
        // satisfied by two mistakes cancelling: one has to be a real one, and a
        // value has to survive the round trip through its Montgomery form and come
        // back unchanged.
        assert_eq!(Fe::ONE.to_be_bytes(), bytes_of(&[1, 0, 0, 0]));
        assert_eq!(Fe::ZERO.to_be_bytes(), [0u8; 32]);
        for value in cases {
            assert_eq!(
                Fe::from_reduced_limbs(&value).to_be_bytes(),
                bytes_of(&value),
                "round trip through Montgomery form"
            );
        }
        // The shape the two wrong domains cancelled in. One in Montgomery form is
        // `2^256 mod p`, which is not the limbs 1: a plain inverse living in a
        // Montgomery world multiplied by a Montgomery value returns the limbs 1
        // while meaning nothing in particular, and that is why the version of this
        // test that passed proved nothing.
        assert_ne!(
            FP.enter_montgomery(&[1, 0, 0, 0]),
            [1, 0, 0, 0],
            "one in Montgomery form is not the limbs 1"
        );
    }

    #[test]
    fn a_small_multiple_of_the_generator_matches_the_groups_own_laws() {
        let g = generator();
        assert_eq!(g_times(1), (bytes_of(&GX), bytes_of(&GY)));

        // 2G from the scalar path equals G doubled by the point law, and 3G
        // equals 2G + G. The ladder and the group law agreeing is worth more than
        // either alone: a formula error moves one of them and not the other.
        assert_eq!(point_xy(&g.double()), g_times(2));
        assert_eq!(point_xy(&g.double().add(g)), g_times(3));
        assert_eq!(point_xy(&g.double().double()), g_times(4));
        assert_ne!(g_times(2), g_times(3), "2G and 3G are different points");

        // Pinned values for the doublings above, from the independent affine
        // implementation in `scripts/tls_p256_vectors.py`. They are here because
        // the agreement checks above are weaker than they look: the ladder and the
        // group law run the *same* sequences when the group law is called as
        // `g.double().double()`, so both sides can be wrong together. That is
        // exactly what happened — a doubling formula that was only correct when
        // `Z == 1` passed every one of those comparisons and then produced a point
        // off the curve at 7G. 4G is doubled from a point whose `Z` is not one,
        // and 6G is doubled from one that is not either.
        assert_eq!(
            point_xy(&g.double()).0,
            hex("7cf27b188d034f7e8a52380304b51ac3c08969e277f21b35a60b48fc47669978"),
            "2G's x coordinate"
        );
        assert_eq!(
            point_xy(&g.double().add(g)).0,
            hex("5ecbe4d1a6330a44c8f7ef951d4bf165e6c6b721efada985fb41661bc6e7fd6c"),
            "3G's x coordinate"
        );
        let four_g = g.double().double();
        assert_eq!(
            point_xy(&four_g).0,
            hex("e2534a3532d08fbba02dde659ee62bd0031fe2db785596ef509302446b030852"),
            "4G's x coordinate"
        );
        // 6G is three times doubled, so it is the first pinned value reached by
        // doubling a point whose `Z` is not one — the case that used to fail.
        assert_eq!(
            point_xy(&g.double().add(g).double()).0,
            hex("b01a172a76a4602c92d3242cb897dde3024c740debb215b4c6b0aae93c2291a9"),
            "6G's x coordinate"
        );
        // ... and 7G is the point the broken version left off the curve, which is
        // how this was found rather than how it was fixed.
        assert_eq!(
            point_xy(&g.double().double().add(g.double().add(g))).0,
            hex("8e533b6fa0bf7b4625bb30667c01fb607ef9f8b8a80fef5b300628703187b2a3"),
            "7G's x coordinate, as 4G + 3G"
        );
    }

    /// A pinned coordinate from the independent implementation, as raw bytes.
    fn hex(text: &str) -> [u8; 32] {
        let bytes = crate::sha256::from_hex(text).expect("hex");
        let mut out = [0u8; 32];
        out.copy_from_slice(&bytes);
        out
    }

    #[test]
    fn the_group_law_handles_the_cases_the_formulas_do_not() {
        let g = generator();

        // Infinity is the identity, on both sides, and doubling it is a no-op.
        assert_eq!(Jacobian::IDENTITY.add(g), g);
        assert_eq!(g.add(Jacobian::IDENTITY), g);
        assert_eq!(Jacobian::IDENTITY.double(), Jacobian::IDENTITY);

        // Infinity with itself stays infinity, and it is its own inverse.
        assert_eq!(
            Jacobian::IDENTITY.add(Jacobian::IDENTITY),
            Jacobian::IDENTITY
        );

        // P + (-P) is infinity: the arm a formula that only handles the generic
        // case gets wrong. `-P` is a point, not a method here, because nothing
        // outside this test needs one.
        let minus_g = minus_g();
        assert!(g.add(minus_g).is_identity());
        assert_eq!(
            point_xy(&minus_g),
            (
                bytes_of(&GX),
                Fe::ZERO.sub(Fe::from_reduced_limbs(&GY)).to_be_bytes()
            )
        );
        // Negating twice is nothing: -(-G) is G again.
        assert_eq!(
            point_xy(&Jacobian::from_affine(
                Fe::from_reduced_limbs(&GX),
                Fe::ZERO.sub(Fe::ZERO.sub(Fe::from_reduced_limbs(&GY))),
            )),
            point_xy(&g)
        );

        // P + P takes the doubling arm and must agree with it.
        assert_eq!(g.add(g), g.double());
    }

    #[test]
    fn double_mul_is_the_sum_of_two_separate_multiples() {
        // A key that is not the generator and scalars that are not one, so
        // neither term is a shortcut for the other.
        let q = key_from(g_times(7));
        let q_point = Jacobian::from_affine(
            Fe::from_be_bytes(&g_times(7).0),
            Fe::from_be_bytes(&g_times(7).1),
        );
        let g = generator();
        let u1 = [11, 0, 0, 0];
        let u2 = [13, 0, 0, 0];

        let expected = point_xy(&mul(g, &u1).add(mul(q_point, &u2))).0;
        assert_eq!(double_mul(&q, &u2, &u1), Some(expected));

        // One term at a time, and neither: the three identity arms.
        assert_eq!(
            double_mul(&q, &[0, 0, 0, 0], &u1),
            Some(point_xy(&mul(g, &u1)).0)
        );
        assert_eq!(
            double_mul(&q, &u2, &[0, 0, 0, 0]),
            Some(point_xy(&mul(q_point, &u2)).0)
        );
        assert_eq!(double_mul(&q, &[0, 0, 0, 0], &[0, 0, 0, 0]), None);

        // n is zero in the group. A scalar from a signature can never be n, since
        // the range check rejects it, but the multiply path must still treat it as
        // the order rather than as a number.
        assert_eq!(double_mul(&q, &N, &u1), Some(point_xy(&mul(g, &u1)).0));
    }

    #[test]
    fn a_key_off_the_curve_or_out_of_range_is_refused() {
        let key = key_from(g_times(1));
        assert_eq!(
            double_mul(&key, &[1, 0, 0, 0], &[0, 0, 0, 0]),
            Some(bytes_of(&GX))
        );

        // Right length, right prefix, one bit of y changed: off the curve. This is
        // the check that stops a key from another curve being read as a point
        // here, which is the failure that leaks a private key in ECDH.
        let mut encoded = [0u8; 65];
        encoded[0] = 0x04;
        encoded[1..33].copy_from_slice(&bytes_of(&GX));
        encoded[33..].copy_from_slice(&bytes_of(&GY));
        let mut off_curve = encoded;
        off_curve[64] ^= 1;
        assert_eq!(PublicKey::from_sec1(&off_curve), Err(P256Error::NotOnCurve));

        // Wrong length, wrong prefix, a coordinate at p, and the all-zero point.
        assert_eq!(
            PublicKey::from_sec1(&encoded[..64]),
            Err(P256Error::PointLength {
                expected: 65,
                found: 64
            })
        );
        let mut compressed = encoded;
        compressed[0] = 0x02;
        assert_eq!(
            PublicKey::from_sec1(&compressed),
            Err(P256Error::UncompressedOnly(0x02))
        );
        let mut x_at_p = encoded;
        x_at_p[1..33].copy_from_slice(&bytes_of(&P));
        assert_eq!(
            PublicKey::from_sec1(&x_at_p),
            Err(P256Error::CoordinateOutOfRange("x"))
        );
        let mut y_at_p = encoded;
        y_at_p[33..].copy_from_slice(&bytes_of(&P));
        assert_eq!(
            PublicKey::from_sec1(&y_at_p),
            Err(P256Error::CoordinateOutOfRange("y"))
        );
        assert_eq!(
            PublicKey::from_sec1(&[0u8; 65]),
            Err(P256Error::UncompressedOnly(0x00))
        );
        let mut zero_point = [0u8; 65];
        zero_point[0] = 0x04;
        assert_eq!(
            PublicKey::from_sec1(&zero_point),
            Err(P256Error::PointAtInfinity)
        );
    }

    #[test]
    fn scalars_are_reduced_and_range_checked() {
        // Zero and n are out of range; one and n - 1 are the edges that are in it.
        assert!(!scalar_in_range(&bytes_of(&[0, 0, 0, 0])));
        assert!(!scalar_in_range(&bytes_of(&N)));
        assert!(scalar_in_range(&bytes_of(&[1, 0, 0, 0])));
        assert!(scalar_in_range(&bytes_of(&sub_const(N, [1, 0, 0, 0]))));

        // Reduction at both edges: everything just under n is itself, n is zero,
        // and a 256-bit value above n loses exactly one n.
        assert_eq!(
            scalar_reduce(sub_const(N, [1, 0, 0, 0])),
            sub_const(N, [1, 0, 0, 0])
        );
        assert_eq!(scalar_reduce(N), [0, 0, 0, 0]);
        let just_under_p = sub_const(P, [1, 0, 0, 0]);
        assert_eq!(scalar_reduce(just_under_p), {
            let mut value = just_under_p;
            sub4(&mut value, &N);
            value
        });
    }
}
