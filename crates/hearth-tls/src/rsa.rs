//! RSA signature verification: the public-key operation, and EMSA-PSS as
//! RFC 8017 §9.1.2 defines it.
//!
//! This is the first half of certificate verification, and the half that can be
//! *proven* today: RFC 8448's recorded handshake contains a complete
//! `rsa_pss_rsae_sha256` signature over the exact 130 bytes the client state
//! machine already rebuilds, so the whole path — modulus, exponent, signature,
//! `s^e mod n`, PSS decoding, message hash — is checked against a published
//! vector rather than against our own output. `tests/rfc8448_flight.rs` drives that
//! path end to end, `tests/rsa_vectors.rs` covers the arithmetic at six limb counts
//! and both `emLen` cases, and `scripts/tls_rfc8448_rsa.py` plus
//! `scripts/tls_rsa_vectors.py` are the generators those constants came from, kept
//! in the repository so the claim is checkable rather than believed.
//!
//! # What this is not
//!
//! It does not read a certificate. Nothing here parses DER, builds a chain,
//! checks a name or a validity window, and **nothing calls this module yet**: the
//! only verifier the crate ships is still
//! [`RejectAll`](crate::client::RejectAll), so no `https://` path exists. The
//! public key arrives as two byte strings the caller has already extracted.
//!
//! # Constant time is not the goal here, and saying so is the point
//!
//! Every input to signature verification is public: the modulus, the exponent,
//! the signature and the message. There is nothing to leak, so the modular
//! exponentiation is allowed to branch on the exponent's bits and to skip the
//! leading zeros, and the PSS decode is allowed to compare fields with `==`. The
//! **one** comparison that is done in constant time is the final `H' == H`, not
//! because the values are secret but because a lazy `==` there is the habit that
//! spreads: the same `constant_time_eq` guards the Finished MAC, where it does
//! matter.
//!
//! # Two signature schemes, and which one is allowed where
//!
//! **`rsa_pss_rsae_sha256`** with MGF1-SHA-256 and a 32-byte salt, which is what
//! TLS 1.3 requires for a *handshake* signature: RFC 8446 §4.2.3 says of the PSS
//! schemes that "the length of the Salt MUST be equal to the length of the output
//! of the digest algorithm". A signature whose salt is a different length is
//! therefore **rejected**, not tolerated — that is the protocol, not a preference.
//! [`RsaPublicKey::verify_pss_sha256`] is the handshake check.
//!
//! **`sha256WithRSAEncryption`**, the PKCS#1 v1.5 form, which TLS 1.3 forbids for
//! a handshake signature and which *certificates use anyway* — nearly every RSA
//! chain in the world is signed with it, including five of the seven X.509
//! fixtures this crate tests against. So [`RsaPublicKey::verify_pkcs1v15_sha256`]
//! exists, and [`crate::verify`] is where the rule that it must never be reached
//! from `CertificateVerify` is enforced rather than promised.
//!
//! There is no third scheme. `ecdsa_secp256r1_sha256`, the other suite a TLS 1.3
//! client must support, is not RSA and lives in [`crate::ecdsa`].
//!
//! # The one policy this module deliberately does not make
//!
//! A minimum key size. A 1024-bit RSA key is still accepted by this code, because
//! RFC 8448's fixture *is* one and refusing it would mean giving up the only
//! published vector for the scheme. The policy belongs where the key comes from —
//! a verifier that will not accept a chain whose leaf key is below 2048 bits — and
//! it is recorded as a required negative test in `docs/RESUME.md` so that it
//! cannot be quietly forgotten.

use crate::sha256::Sha256;
use crate::util::constant_time_eq;

/// SHA-256 output length, which is also the salt length TLS 1.3 requires.
pub const SHA256_LEN: usize = 32;

/// The smallest modulus, in bits, that can carry PSS with SHA-256 and a 32-byte
/// salt: `ceil((modBits - 1) / 8) >= hLen + sLen + 2` rearranges to
/// `modBits >= 8 * (hLen + sLen + 2) - 6`.
const MIN_PSS_MODULUS_BITS: usize = 8 * (SHA256_LEN + SHA256_LEN + 2) - 6;

/// Why a signature, a modulus or an encoding was refused.
///
/// Every variant is fatal to verification: there is no "accept anyway" and no
/// recoverable case. The variants name the rule that failed, so a diagnostic says
/// which of RFC 8017's steps rejected the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RsaError {
    /// The modulus is structurally unusable, with the reason.
    BadModulus(&'static str),
    /// The exponent is zero, even, below 3, or wider than 64 bits.
    UnsupportedExponent,
    /// The signature is not `k` bytes, the modulus's length.
    SignatureLength {
        /// `k`, the modulus length in bytes.
        expected: usize,
        /// What the caller passed.
        found: usize,
    },
    /// The signature value is not less than the modulus, so it is not a
    /// signature at all (`0 <= s < n` is required).
    SignatureOutOfRange,
    /// The key is too small for PSS with a 32-byte salt.
    KeyTooSmall {
        /// The modulus size in bits.
        bits: usize,
        /// The smallest modulus size, in bits, that could carry the encoding.
        needed_bits: usize,
    },
    /// The key is too small for a PKCS#1 v1.5 `SHA-256` encoding, which needs
    /// `k >= 19 + 32 + 11` bytes.
    KeyTooSmallForPkcs1 {
        /// The modulus size in bits.
        bits: usize,
        /// The smallest modulus size, in bits, that could carry the encoding.
        needed_bits: usize,
    },
    /// The encoding did not match, with the rule that failed. Everything named
    /// here is public: it describes bytes the peer sent us.
    Inconsistent(&'static str),
}

impl std::fmt::Display for RsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadModulus(why) => write!(f, "unusable RSA modulus: {why}"),
            Self::UnsupportedExponent => write!(f, "unsupported RSA exponent"),
            Self::SignatureLength { expected, found } => write!(
                f,
                "the signature is {found} bytes, but this key's modulus is {expected}"
            ),
            Self::SignatureOutOfRange => write!(f, "the signature is not less than the modulus"),
            Self::KeyTooSmall { bits, needed_bits } => write!(
                f,
                "a {bits}-bit modulus is too small for PSS with a {SHA256_LEN}-byte salt \
                 (at least {needed_bits} bits are needed)"
            ),
            Self::KeyTooSmallForPkcs1 { bits, needed_bits } => write!(
                f,
                "a {bits}-bit modulus is too small for a PKCS#1 v1.5 SHA-256 encoding \
                 (at least {needed_bits} bits are needed)"
            ),
            Self::Inconsistent(why) => write!(f, "PSS encoding is inconsistent: {why}"),
        }
    }
}

impl std::error::Error for RsaError {}

/// An RSA public key: a modulus and a public exponent.
///
/// Built from the two integers as they appear in a certificate's `rsaEncryption`
/// `SubjectPublicKeyInfo`, both big-endian. The modulus must be odd (Montgomery
/// arithmetic requires it, and a real modulus is a product of two odd primes);
/// the exponent must be odd and at least 3, and is limited to 64 bits, which
/// every real key satisfies — a longer exponent only ever makes verification
/// slower, and refusing it is better than quietly truncating it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RsaPublicKey {
    /// The modulus, little-endian 64-bit limbs, with no leading zero limb.
    modulus: Vec<u64>,
    /// The public exponent.
    exponent: u64,
    /// `-n^-1 mod 2^64`, for the Montgomery reduction.
    n0inv: u64,
    /// `2^(2 * 64 * limbs) mod n`: what converts a value into Montgomery form.
    r2: Vec<u64>,
}

impl RsaPublicKey {
    /// Builds a key from the modulus and exponent as big-endian byte strings.
    pub fn from_be_bytes(modulus: &[u8], exponent: &[u8]) -> Result<Self, RsaError> {
        let modulus = limbs_from_be(modulus);
        let bits = bit_len(&modulus);
        if bits == 0 {
            return Err(RsaError::BadModulus("the modulus is zero"));
        }
        if modulus[0] & 1 == 0 {
            return Err(RsaError::BadModulus(
                "the modulus is even, so it is not a product of two odd primes",
            ));
        }
        if bits < 3 {
            return Err(RsaError::BadModulus("the modulus is smaller than 3"));
        }

        // DER allows a leading zero byte on a positive INTEGER, so leading zeros
        // are stripped rather than rejected; anything wider than 64 bits is
        // refused rather than truncated.
        let exponent = match exponent.iter().position(|&byte| byte != 0) {
            Some(first) => &exponent[first..],
            None => return Err(RsaError::UnsupportedExponent),
        };
        if exponent.len() > 8 {
            return Err(RsaError::UnsupportedExponent);
        }
        let e = exponent
            .iter()
            .fold(0u64, |acc, &byte| acc << 8 | byte as u64);
        if e < 3 || e & 1 == 0 {
            return Err(RsaError::UnsupportedExponent);
        }

        let mut key = Self {
            n0inv: montgomery_n0inv(modulus[0]),
            r2: Vec::new(),
            modulus,
            exponent: e,
        };
        key.r2 = key.compute_r2();
        Ok(key)
    }

    /// The modulus size in bits: RFC 8017's `modBits`.
    pub fn key_bits(&self) -> usize {
        bit_len(&self.modulus)
    }

    /// RFC 8017's `k`, the modulus length in bytes: `ceil(modBits / 8)`.
    ///
    /// This is the length of a signature and of the `I2OSP(m, k)` output. It is
    /// **not** the width of the limbs a value is stored in, and the difference is
    /// real: a 1025-bit modulus is 17 limbs wide but `k = 129`, while its PSS
    /// encoding is `emLen = 128` bytes, one byte shorter than `k`. That case is
    /// covered by a test with a real 1025-bit key precisely because an
    /// implementation that confuses the two passes every other check.
    pub fn modulus_bytes(&self) -> usize {
        self.key_bits().div_ceil(8)
    }

    /// The public exponent, for diagnostics.
    pub fn exponent(&self) -> u64 {
        self.exponent
    }

    /// RSAVP1 (RFC 8017 §5.2.2): `s^e mod n`, returned as exactly `k` bytes.
    ///
    /// `signature` must be `k` bytes and its value must be below the modulus.
    /// RFC 8017 §9.1.2 requires both — step 1 rejects any other length and step 2b
    /// rejects an out-of-range representative — and they are checked here rather
    /// than left to the caller, because a shorter encoding of the same integer is
    /// a *different byte string* that would otherwise verify.
    pub fn rsa_vp1(&self, signature: &[u8]) -> Result<Vec<u8>, RsaError> {
        let k = self.modulus_bytes();
        if signature.len() != k {
            return Err(RsaError::SignatureLength {
                expected: k,
                found: signature.len(),
            });
        }
        let value = limbs_from_be(signature);
        if !less_than(&value, &self.modulus) {
            return Err(RsaError::SignatureOutOfRange);
        }

        let mont = Montgomery {
            n: &self.modulus,
            n0inv: self.n0inv,
        };
        let result = mont.pow(&value, self.exponent, &self.r2);
        debug_assert!(result.len() == self.modulus.len());
        Ok(to_be_bytes_fixed(&result, k))
    }

    /// RSASSA-PSS-VERIFY (RFC 8017 §9.1.2) with SHA-256, MGF1-SHA-256 and a
    /// 32-byte salt, which is all TLS 1.3 accepts for `rsa_pss_rsae_sha256`.
    ///
    /// `message` is the data that was signed — for TLS, the 130-byte
    /// `signed_data` of RFC 8446 §4.4.3, not a pre-computed hash. The salt is
    /// *required* to be [`SHA256_LEN`] bytes, so a signature made with any other
    /// salt length is refused rather than accepted with whatever salt it carries.
    pub fn verify_pss_sha256(&self, message: &[u8], signature: &[u8]) -> Result<(), RsaError> {
        let h_len = SHA256_LEN;
        let s_len = SHA256_LEN;
        let em_bits = self.key_bits() - 1;
        let em_len = em_bits.div_ceil(8);

        // RFC 8017 §9.1.2 step 3, checked before any arithmetic so the bounds
        // below cannot underflow.
        if em_len < h_len + s_len + 2 {
            return Err(RsaError::KeyTooSmall {
                bits: self.key_bits(),
                needed_bits: MIN_PSS_MODULUS_BITS,
            });
        }

        let m = self.rsa_vp1(signature)?;
        let k = self.modulus_bytes();
        // Step 2c: EM = I2OSP(m, emLen). Anything above emLen must be zero, which
        // is exactly the `modBits == 1 (mod 8)` case where emLen is k - 1.
        let em = &m[k - em_len..];
        if m[..k - em_len].iter().any(|&byte| byte != 0) {
            return Err(RsaError::Inconsistent(
                "the RSA result is wider than emLen allows",
            ));
        }

        // Step 4: EM is maskedDB || H || 0xbc.
        let db_len = em_len - h_len - 1;
        let (masked_db, tail) = em.split_at(db_len);
        let h = &tail[..h_len];
        if tail[h_len] != 0xbc {
            return Err(RsaError::Inconsistent("the trailer byte is not 0xbc"));
        }

        // Step 5: the leftmost 8 * emLen - emBits bits of maskedDB are zero.
        // `unused` is in 0..=7 by construction, so the shift below cannot overflow.
        let unused = 8 * em_len - em_bits;
        debug_assert!(unused < 8);
        if unused > 0 && masked_db[0] & (0xffu8 << (8 - unused)) != 0 {
            return Err(RsaError::Inconsistent(
                "the bits of maskedDB above emBits are not zero",
            ));
        }

        // Step 6: dbMask = MGF1(H, emLen - hLen - 1); DB = maskedDB xor dbMask,
        // with the leftmost 8 * emLen - emBits bits cleared.
        let mut db = vec![0u8; db_len];
        mgf1_sha256(h, &mut db);
        for (byte, mask) in db.iter_mut().zip(masked_db) {
            *byte ^= mask;
        }
        if unused > 0 {
            db[0] &= 0xffu8 >> unused;
        }

        // Steps 7 and 8: DB is PS || 0x01 || salt, PS all zeros, salt exactly
        // sLen long. A shorter salt puts the 0x01 earlier and leaves that byte
        // inside PS, which this reads for what it is: a different encoding.
        let ps_len = db_len - s_len - 1;
        if db[..ps_len].iter().any(|&byte| byte != 0) {
            return Err(RsaError::Inconsistent(
                "the padding string is not all zeros",
            ));
        }
        if db[ps_len] != 0x01 {
            return Err(RsaError::Inconsistent(
                "the separator byte is not 0x01 where a 32-byte salt requires it",
            ));
        }
        let salt = &db[ps_len + 1..];

        // Step 9: H' = Hash(0x00 * 8 || mHash || salt), compared in constant time.
        let mut hasher = Sha256::new();
        hasher.update(&[0u8; 8]);
        hasher.update(&sha256(message));
        hasher.update(salt);
        if !constant_time_eq(&hasher.finalize(), h) {
            return Err(RsaError::Inconsistent(
                "the message hash does not match the one the signature encodes",
            ));
        }
        Ok(())
    }

    /// RSASSA-PKCS1-v1_5-VERIFY (RFC 8017 §8.2.2) with SHA-256.
    ///
    /// This is the scheme a *certificate* is signed with, not a handshake: TLS 1.3
    /// removed `rsa_pkcs1_*` from `signature_algorithms` for `CertificateVerify`,
    /// so nothing that handles a handshake may call this. It is here because RFC
    /// 5280 chains are full of it, and a chain verifier that cannot check its own
    /// links cannot verify anything.
    ///
    /// `message` is the data that was signed — for a certificate, the
    /// `tbsCertificate` bytes, not a pre-computed hash.
    pub fn verify_pkcs1v15_sha256(&self, message: &[u8], signature: &[u8]) -> Result<(), RsaError> {
        let needed_bits = 8 * (SHA256_DIGEST_INFO_PREFIX.len() + SHA256_LEN + PKCS1_MIN_PADDING);
        if self.modulus_bytes() < SHA256_DIGEST_INFO_PREFIX.len() + SHA256_LEN + PKCS1_MIN_PADDING {
            return Err(RsaError::KeyTooSmallForPkcs1 {
                bits: self.key_bits(),
                needed_bits,
            });
        }
        let em = self.rsa_vp1(signature)?;
        check_pkcs1v15_sha256(&em, &sha256(message))
    }

    /// `2^(2 * 64 * limbs) mod n`, reached by doubling `1` that many times
    /// rather than by dividing: modular doubling is one shift and one conditional
    /// subtract, so this module needs no division routine at all.
    fn compute_r2(&self) -> Vec<u64> {
        let mut value = one(self.modulus.len());
        for _ in 0..(2 * 64 * self.modulus.len()) {
            double_mod(&mut value, &self.modulus);
        }
        value
    }
}

/// The least `PS` a PKCS#1 v1.5 block may carry (`emLen - tLen - 3 >= 8`
/// rearranges the other way round).
const PKCS1_MIN_PADDING: usize = 8;

/// SHA-256's `DigestInfo` prefix, the DER of
/// `SEQUENCE { SEQUENCE { id-sha256, NULL }, OCTET STRING }` **without** the hash
/// that follows it: nineteen bytes, then the 32-byte hash.
///
/// Written out rather than assembled at run time, and then *checked* against the
/// pieces it is made of by
/// `the_sha256_digest_info_prefix_is_the_der_of_the_algorithm_it_names`: the OID
/// comes from [`crate::der::oid_from_dotted`], which is the same function the DER
/// tests check their OIDs against, so a wrong byte here cannot pass.
const SHA256_DIGEST_INFO_PREFIX: [u8; 19] = [
    0x30, 0x31, // SEQUENCE, 49 bytes
    0x30, 0x0d, // SEQUENCE, 13 bytes: the AlgorithmIdentifier
    0x06, 0x09, 0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01, // OID id-sha256
    0x05, 0x00, // NULL parameters
    0x04, 0x20, // OCTET STRING, 32 bytes: the hash
];

/// Checks `EM = 0x00 || 0x01 || PS || 0x00 || DigestInfo || H(m)` exactly.
///
/// **Structurally, not by search**, and that distinction is the whole security
/// content of this function. Bleichenbacher's 2006 attack works against verifiers
/// that look for the `DigestInfo` *anywhere* in `EM` and ignore what surrounds it:
/// with a small exponent an attacker can find an `EM` that contains a valid
/// `DigestInfo` followed by rubbish, sign nothing, and be accepted. So every byte
/// is accounted for by position — the block type, at least eight `0xFF`, the
/// separator, then a `DigestInfo` of exactly the length SHA-256 requires and the
/// hash in the last 32 bytes — and there is no code path here that searches for a
/// pattern and accepts whatever it finds around it.
///
/// `em` is exactly `k` bytes, which the caller has established.
fn check_pkcs1v15_sha256(em: &[u8], hash: &[u8; SHA256_LEN]) -> Result<(), RsaError> {
    if em.len() < SHA256_DIGEST_INFO_PREFIX.len() + SHA256_LEN + PKCS1_MIN_PADDING {
        return Err(RsaError::Inconsistent(
            "the encoding is shorter than a SHA-256 block can be",
        ));
    }
    if em[0] != 0x00 || em[1] != 0x01 {
        return Err(RsaError::Inconsistent(
            "the block type is not 0x00 0x01, which is the private-key operation",
        ));
    }
    // The padding runs to the separator, so its length is where the run of 0xFF
    // ends and the check on that length is what makes `PS` at least eight bytes.
    let mut separator = 2;
    while separator < em.len() && em[separator] == 0xff {
        separator += 1;
    }
    if separator < 2 + PKCS1_MIN_PADDING {
        return Err(RsaError::Inconsistent(
            "fewer than eight 0xFF padding bytes",
        ));
    }
    if em.get(separator) != Some(&0x00) {
        return Err(RsaError::Inconsistent(
            "the padding string is not followed by a zero separator byte",
        ));
    }
    let digest_info = &em[separator + 1..];
    if digest_info.len() != SHA256_DIGEST_INFO_PREFIX.len() + SHA256_LEN {
        return Err(RsaError::Inconsistent(
            "the DigestInfo is not exactly what SHA-256 requires, with nothing after it",
        ));
    }
    if !constant_time_eq(
        &digest_info[..SHA256_DIGEST_INFO_PREFIX.len()],
        &SHA256_DIGEST_INFO_PREFIX,
    ) {
        return Err(RsaError::Inconsistent(
            "the DigestInfo does not name SHA-256",
        ));
    }
    if !constant_time_eq(&digest_info[SHA256_DIGEST_INFO_PREFIX.len()..], hash) {
        return Err(RsaError::Inconsistent(
            "the message hash does not match the one the signature encodes",
        ));
    }
    Ok(())
}

/// One-shot SHA-256 (FIPS 180-4), which is all this module needs.
fn sha256(data: &[u8]) -> [u8; SHA256_LEN] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

/// MGF1 (RFC 8017 §B.2.1) with SHA-256: `Hash(seed || counter)` blocks, the
/// counter being four bytes big-endian, as specified.
fn mgf1_sha256(seed: &[u8], out: &mut [u8]) {
    for (counter, chunk) in out.chunks_mut(SHA256_LEN).enumerate() {
        let mut hasher = Sha256::new();
        hasher.update(seed);
        hasher.update(&(counter as u32).to_be_bytes());
        chunk.copy_from_slice(&hasher.finalize()[..chunk.len()]);
    }
}

/// Montgomery arithmetic modulo a fixed odd modulus.
///
/// Values are little-endian limb vectors of exactly `n.len()` limbs, all less
/// than `n`. `mont_mul(a, b)` returns `a * b * 2^-(64 * limbs) mod n` — the
/// Montgomery product — in the coarse-grained "multiply, then reduce" form
/// rather than the interleaved CIOS form: the product goes into a work buffer
/// twice the modulus's width plus two limbs, and the reduction then makes the low
/// half zero from the bottom up. That is easier to check by reading, which
/// matters more here than the few percent it costs.
struct Montgomery<'a> {
    /// The modulus, little-endian limbs, odd.
    n: &'a [u64],
    /// `-n^-1 mod 2^64`.
    n0inv: u64,
}

impl Montgomery<'_> {
    /// `a * b * R^-1 mod n`, for `a` and `b` already reduced: pass `b = r2` to
    /// convert `a` into Montgomery form, and `b = 1` to convert back out.
    fn mont_mul(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let l = self.n.len();
        let mut t = vec![0u64; 2 * l + 2];

        // t = a * b, schoolbook, with the carries propagated into the limbs above.
        for i in 0..l {
            let mut carry = 0u64;
            for j in 0..l {
                let acc = t[i + j] as u128 + (a[j] as u128) * (b[i] as u128) + carry as u128;
                t[i + j] = acc as u64;
                carry = (acc >> 64) as u64;
            }
            let mut k = i + l;
            while carry != 0 {
                let acc = t[k] as u128 + carry as u128;
                t[k] = acc as u64;
                carry = (acc >> 64) as u64;
                k += 1;
            }
        }

        // The reduction: each step makes one more low limb zero, by adding
        // `factor * n * 2^(64 i)` for the `factor` that cancels `t[i]`.
        for i in 0..l {
            let factor = t[i].wrapping_mul(self.n0inv);
            let mut carry = 0u64;
            for j in 0..l {
                let acc = t[i + j] as u128 + (factor as u128) * (self.n[j] as u128) + carry as u128;
                t[i + j] = acc as u64;
                carry = (acc >> 64) as u64;
            }
            let mut k = i + l;
            while carry != 0 {
                let acc = t[k] as u128 + carry as u128;
                t[k] = acc as u64;
                carry = (acc >> 64) as u64;
                k += 1;
            }
        }

        // The low half is now zero and the high half is below 2n, so one
        // conditional subtraction finishes it. Subtracting wraps the limb vector,
        // which is correct because the true difference is below 2^64l.
        let (low, rest) = t.split_at(l);
        debug_assert!(low.iter().all(|&limb| limb == 0));
        let mut result = rest[..l].to_vec();
        if rest[l] != 0 || !less_than(&result, self.n) {
            subtract_in_place(&mut result, self.n);
        }
        result
    }

    /// `base^exponent mod n`, converting in and out of Montgomery form.
    ///
    /// Square-and-multiply, left to right, skipping the exponent's leading zeros.
    /// Branching on the exponent is fine: see the module docs, the exponent is
    /// public.
    fn pow(&self, base: &[u64], exponent: u64, r2: &[u64]) -> Vec<u64> {
        let l = self.n.len();
        let base = pad(base, l);
        let one_mont = self.mont_mul(&one(l), r2);
        let base_mont = self.mont_mul(&base, r2);

        let mut result = one_mont;
        for bit in (0..64 - exponent.leading_zeros()).rev() {
            let squared = self.mont_mul(&result, &result);
            result = if (exponent >> bit) & 1 == 1 {
                self.mont_mul(&squared, &base_mont)
            } else {
                squared
            };
        }
        self.mont_mul(&result, &one(l))
    }
}

/// `-n^-1 mod 2^64` by Newton's iteration, which doubles the number of correct
/// bits each round: six rounds give 64 bits from a one-bit seed.
fn montgomery_n0inv(n0: u64) -> u64 {
    debug_assert_eq!(n0 & 1, 1, "Montgomery needs an odd modulus");
    let mut inverse = 1u64;
    for _ in 0..6 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(n0.wrapping_mul(inverse)));
    }
    inverse.wrapping_neg()
}

/// `1` as a limb vector of `l` limbs.
fn one(l: usize) -> Vec<u64> {
    let mut value = vec![0u64; l];
    value[0] = 1;
    value
}

/// Renders a limb vector as exactly `bytes` big-endian bytes, dropping any higher
/// limbs (which the caller has established are zero).
fn to_be_bytes_fixed(limbs: &[u64], bytes: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes);
    for index in (0..bytes).rev() {
        let limb = index / 8;
        let shift = (index % 8) * 8;
        out.push(if limb < limbs.len() {
            (limbs[limb] >> shift) as u8
        } else {
            0
        });
    }
    out
}

/// Parses a big-endian byte string into little-endian 64-bit limbs, dropping
/// leading zero limbs. An empty or all-zero input becomes `[0]`, which the caller
/// rejects as a zero modulus.
///
/// The chunks are taken **from the end** and each is zero-padded at its high
/// side: the last eight bytes of the input are the least significant limb. Taking
/// them from the front instead, as an earlier version of this did, silently
/// reverses the limbs of any value wider than one limb and doubles its apparent
/// width — a bug that a test caught and that no amount of staring at the
/// arithmetic would have.
fn limbs_from_be(bytes: &[u8]) -> Vec<u64> {
    let mut limbs = Vec::with_capacity(bytes.len().div_ceil(8).max(1));
    let mut end = bytes.len();
    while end > 0 {
        let start = end.saturating_sub(8);
        let mut chunk = [0u8; 8];
        chunk[8 - (end - start)..].copy_from_slice(&bytes[start..end]);
        limbs.push(u64::from_be_bytes(chunk));
        end = start;
    }
    if limbs.is_empty() {
        limbs.push(0);
    }
    while limbs.len() > 1 && limbs.last() == Some(&0) {
        limbs.pop();
    }
    limbs
}

/// Grows a limb vector to `l` limbs, which the Montgomery routines require of
/// both operands.
fn pad(limbs: &[u64], l: usize) -> Vec<u64> {
    let mut out = limbs.to_vec();
    out.resize(l, 0);
    out
}

/// The bit length of a value that has no leading zero limb.
fn bit_len(limbs: &[u64]) -> usize {
    match limbs.last() {
        Some(top) => limbs.len() * 64 - top.leading_zeros() as usize,
        None => 0,
    }
}

/// `a < b` over limb vectors that may be different lengths (the shorter is
/// zero-extended).
fn less_than(a: &[u64], b: &[u64]) -> bool {
    for index in (0..a.len().max(b.len())).rev() {
        let x = a.get(index).copied().unwrap_or(0);
        let y = b.get(index).copied().unwrap_or(0);
        if x != y {
            return x < y;
        }
    }
    false
}

/// `a -= b`. Legitimate only when the caller has established `a >= b`, which is
/// the case at both call sites.
fn subtract_in_place(a: &mut [u64], b: &[u64]) {
    let mut borrow = 0u64;
    for (index, limb) in a.iter_mut().enumerate() {
        let (value, first) = limb.overflowing_sub(b.get(index).copied().unwrap_or(0));
        let (value, second) = value.overflowing_sub(borrow);
        *limb = value;
        borrow = (first as u64) | (second as u64);
    }
}

/// `value = 2 * value mod n`: shift, then subtract `n` if the shift carried out
/// of the top limb or the doubled value is now at least `n`.
fn double_mod(value: &mut [u64], n: &[u64]) {
    let mut carry = 0u64;
    for limb in value.iter_mut() {
        let next = *limb >> 63;
        *limb = (*limb << 1) | carry;
        carry = next;
    }
    if carry != 0 || !less_than(value, n) {
        subtract_in_place(value, n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limbs(bytes: &[u8]) -> Vec<u64> {
        limbs_from_be(bytes)
    }

    #[test]
    fn the_inverse_is_the_negative_inverse_of_the_low_limb() {
        // `n * n0inv == -1 (mod 2^64)` is the whole contract, over values that
        // stress both ends of the word.
        for n0 in [
            1u64,
            3,
            0x0102_0304_0506_0709,
            u64::MAX,
            0x8000_0000_0000_0001,
        ] {
            let inverse = montgomery_n0inv(n0);
            assert_eq!(n0.wrapping_mul(inverse), u64::MAX, "{n0:#x}");
        }
    }

    /// A PKCS#1 v1.5 encoding built from its parts, so the negative cases below
    /// differ from a valid one in exactly one respect.
    fn em(padding: usize, prefix: &[u8], hash: &[u8], trailer: usize) -> Vec<u8> {
        let mut em = vec![0x00, 0x01];
        em.extend(std::iter::repeat_n(0xff, padding));
        em.push(0x00);
        em.extend_from_slice(prefix);
        em.extend_from_slice(hash);
        em.extend(std::iter::repeat_n(0x00, trailer));
        em
    }

    #[test]
    fn the_sha256_digest_info_prefix_is_the_der_of_the_algorithm_it_names() {
        // Rebuilt from the pieces `der` already knows — the OID encoder that the
        // DER tests check their OIDs against, and the crate's own `id-sha256`
        // constant — so a wrong byte in the literal above cannot pass this.
        let mut rebuilt = vec![0x30, 0x31, 0x30, 0x0d, 0x06, 0x09];
        rebuilt.extend_from_slice(&crate::der::oid_from_dotted("2.16.840.1.101.3.4.2.1"));
        rebuilt.extend_from_slice(&[0x05, 0x00, 0x04, 0x20]);
        assert_eq!(rebuilt, SHA256_DIGEST_INFO_PREFIX);
        assert_eq!(
            &SHA256_DIGEST_INFO_PREFIX[6..15],
            crate::der::oids::SHA256,
            "the OID inside the prefix is the one `der` names"
        );
        // And it is the length the SHA-256 encoding implies, which is what makes
        // the nineteen-byte literal above checkable rather than mysterious.
        assert_eq!(SHA256_DIGEST_INFO_PREFIX.len(), 19);
        assert_eq!(
            SHA256_DIGEST_INFO_PREFIX[1], 0x31,
            "49 bytes follow the header"
        );
        assert_eq!(SHA256_DIGEST_INFO_PREFIX[15], 0x05, "NULL parameters");
        assert_eq!(SHA256_DIGEST_INFO_PREFIX[17], 0x04, "an OCTET STRING");
        assert_eq!(SHA256_DIGEST_INFO_PREFIX[18], 0x20, "of 32 bytes");
    }

    #[test]
    fn a_pkcs1v15_encoding_is_checked_by_position_and_not_by_search() {
        let hash = [0x5au8; SHA256_LEN];
        let prefix = &SHA256_DIGEST_INFO_PREFIX;

        // The valid shape: eleven bytes of padding is the least RFC 8017 allows,
        // and a 2048-bit key's worth is what a real signature has.
        assert_eq!(
            check_pkcs1v15_sha256(&em(8, prefix, &hash, 0), &hash),
            Ok(())
        );
        assert_eq!(
            check_pkcs1v15_sha256(&em(200, prefix, &hash, 0), &hash),
            Ok(())
        );

        // Seven `0xFF` bytes: `emLen - tLen - 3 >= 8` is violated by one byte.
        assert_eq!(
            check_pkcs1v15_sha256(&em(7, prefix, &hash, 0), &hash),
            Err(RsaError::Inconsistent(
                "fewer than eight 0xFF padding bytes"
            ))
        );
        // The block type of a public-key operation, not a private-key one.
        let mut public_type = em(8, prefix, &hash, 0);
        public_type[1] = 0x02;
        assert!(check_pkcs1v15_sha256(&public_type, &hash).is_err());
        // No separator where the padding ends.
        let mut no_separator = em(8, prefix, &hash, 0);
        no_separator[10] = 0xff;
        assert!(check_pkcs1v15_sha256(&no_separator, &hash).is_err());
        // A `DigestInfo` of exactly the right length naming **SHA-512** — the same
        // nineteen-byte shape, a different OID — which is what algorithm confusion
        // looks like: a verifier that read the algorithm out of the signature and
        // then hashed with it would accept this and call it SHA-256.
        let mut sha512 = vec![0x30, 0x51, 0x30, 0x0d, 0x06, 0x09];
        sha512.extend_from_slice(&crate::der::oid_from_dotted("2.16.840.1.101.3.4.2.3"));
        sha512.extend_from_slice(&[0x05, 0x00, 0x04, 0x40]);
        assert_eq!(
            sha512.len(),
            19,
            "SHA-512's DigestInfo prefix is the same width"
        );
        assert_eq!(
            check_pkcs1v15_sha256(&em(8, &sha512, &hash, 0), &hash),
            Err(RsaError::Inconsistent(
                "the DigestInfo does not name SHA-256"
            ))
        );
        // And SHA-1, which is shorter rather than wider: it fails on the length
        // before the OID is ever compared, and the point of asserting the variant
        // is that those are two different refusals.
        let mut sha1 = vec![0x30, 0x21, 0x30, 0x09, 0x06, 0x05];
        sha1.extend_from_slice(&crate::der::oid_from_dotted("1.3.14.3.2.26"));
        sha1.extend_from_slice(&[0x05, 0x00, 0x04, 0x14]);
        // Padding chosen so the block clears the minimum length: this case has to
        // fail on the `DigestInfo`, not on being too short to hold one, or it is
        // asserting the wrong refusal.
        assert!(em(25, &sha1, &[0x5au8; 20], 0).len() >= 59);
        assert_eq!(
            check_pkcs1v15_sha256(&em(25, &sha1, &[0x5au8; 20], 0), &hash),
            Err(RsaError::Inconsistent(
                "the DigestInfo is not exactly what SHA-256 requires, with nothing after it"
            ))
        );
        // The right prefix and the right length, but a different hash.
        let mut other_hash = hash;
        other_hash[31] ^= 1;
        assert_eq!(
            check_pkcs1v15_sha256(&em(8, prefix, &other_hash, 0), &hash),
            Err(RsaError::Inconsistent(
                "the message hash does not match the one the signature encodes"
            ))
        );

        // The cases Bleichenbacher's 2006 attack depends on being refused. A
        // verifier that finds the `DigestInfo` and ignores what surrounds it
        // accepts all three of these; this one accounts for every byte.
        assert_eq!(
            check_pkcs1v15_sha256(&em(8, prefix, &hash, 1), &hash),
            Err(RsaError::Inconsistent(
                "the DigestInfo is not exactly what SHA-256 requires, with nothing after it"
            )),
            "a byte after the hash"
        );
        let mut moved = em(8, prefix, &hash, 0);
        moved.insert(11, 0x00);
        assert!(
            check_pkcs1v15_sha256(&moved, &hash).is_err(),
            "a byte between the separator and the DigestInfo"
        );
        assert!(
            check_pkcs1v15_sha256(&em(8, &[], &hash, 0), &hash).is_err(),
            "the hash where the DigestInfo should be"
        );
        assert!(
            check_pkcs1v15_sha256(&[], &hash).is_err(),
            "an empty encoding"
        );
    }

    #[test]
    fn pkcs1v15_needs_five_bytes_less_than_pss_does() {
        // 472 bits is the floor for a SHA-256 PKCS#1 v1.5 block (19 + 32 + 8
        // bytes), against 522 for PSS: a 512-bit key can carry one and not the
        // other, and the two refusals must not be confused with each other.
        assert_eq!(
            8 * (SHA256_DIGEST_INFO_PREFIX.len() + SHA256_LEN + PKCS1_MIN_PADDING),
            472
        );
        let mut small = [0xffu8; 59];
        small[58] = 0xfd;
        let key = RsaPublicKey::from_be_bytes(&small, &[1, 0, 1]).expect("a usable key");
        assert_eq!(key.key_bits(), 472);
        // Exactly enough: the refusal is about the encoding here, not the key.
        assert!(!matches!(
            key.verify_pkcs1v15_sha256(b"message", &[0x42; 59]),
            Err(RsaError::KeyTooSmallForPkcs1 { .. })
        ));
        // Five bytes too narrow for PSS, though, which is a different refusal.
        assert!(matches!(
            key.verify_pss_sha256(b"message", &[0x42; 59]),
            Err(RsaError::KeyTooSmall { .. })
        ));

        let mut too_small = [0xffu8; 58];
        too_small[57] = 0xfd;
        let key = RsaPublicKey::from_be_bytes(&too_small, &[1, 0, 1]).expect("a usable key");
        assert_eq!(
            key.verify_pkcs1v15_sha256(b"message", &[0x42; 58]),
            Err(RsaError::KeyTooSmallForPkcs1 {
                bits: 464,
                needed_bits: 472
            })
        );
    }

    #[test]
    fn doubling_wraps_around_the_modulus() {
        let n = limbs(&[0x01, 0x00, 0x01]); // 65537
        let mut value = one(n.len());
        for _ in 0..16 {
            double_mod(&mut value, &n);
        }
        assert_eq!(value, limbs(&[0x01, 0x00, 0x00]), "2^16 is below n");
        double_mod(&mut value, &n);
        assert_eq!(value, limbs(&[0xff, 0xff]), "2^17 mod 65537 = 65535");
        assert_eq!(bit_len(&value), 16);
    }

    #[test]
    fn the_smallest_usable_modulus_is_where_the_spec_puts_it() {
        // ceil((modBits - 1) / 8) >= 66 requires modBits >= 522.
        assert_eq!(MIN_PSS_MODULUS_BITS, 522);
        assert_eq!(
            (521usize - 1).div_ceil(8),
            65,
            "521 bits cannot carrying it"
        );
        assert_eq!((522usize - 1).div_ceil(8), 66, "522 bits can");
    }

    #[test]
    fn a_modulus_or_exponent_of_the_wrong_shape_is_refused() {
        let e = [1u8, 0, 1];
        assert!(matches!(
            RsaPublicKey::from_be_bytes(&[], &e),
            Err(RsaError::BadModulus("the modulus is zero"))
        ));
        assert!(matches!(
            RsaPublicKey::from_be_bytes(&[0, 0, 0], &e),
            Err(RsaError::BadModulus("the modulus is zero"))
        ));
        // Even: Montgomery cannot work, and no real modulus is even.
        assert!(matches!(
            RsaPublicKey::from_be_bytes(&[0x80, 0x00], &e),
            Err(RsaError::BadModulus(_))
        ));
        assert!(matches!(
            RsaPublicKey::from_be_bytes(&[0x01], &e),
            Err(RsaError::BadModulus("the modulus is smaller than 3"))
        ));
        // Exponent: even, below 3, empty, all zero, and wider than 64 bits.
        for bad in [
            &[2u8][..],
            &[1],
            &[],
            &[0, 0],
            &[0; 9],
            &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        ] {
            assert_eq!(
                RsaPublicKey::from_be_bytes(&[0xff, 0x01], bad),
                Err(RsaError::UnsupportedExponent),
                "exponent {bad:?}"
            );
        }
        // 3 is the smallest legal exponent, and leading zeros are harmless.
        assert_eq!(
            RsaPublicKey::from_be_bytes(&[0xff, 0x01], &[0, 0, 0, 3])
                .expect("e = 3")
                .exponent(),
            3
        );
    }

    /// Times the public-key operation on a 2048-bit modulus, which is the whole
    /// cost of verifying one signature apart from the hashing.
    ///
    /// Ignored by default so the suite stays quiet and fast; it exists because a
    /// number in a document should have been measured:
    /// `cargo test --release -p hearth-tls --lib -- --ignored --nocapture rsa::tests::measure`.
    #[test]
    #[ignore = "timing measurement"]
    fn measure_a_2048_bit_public_key_operation() {
        // A synthetic odd modulus of exactly 2048 bits: the arithmetic does not
        // care whether it factors, and generating a real key would only measure
        // OpenSSL.
        let mut n = [0xffu8; 256];
        n[255] = 0xfd;
        let key = RsaPublicKey::from_be_bytes(&n, &[1, 0, 1]).expect("a usable key");
        let signature = [0x42u8; 256];

        let rounds = 200u32;
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            let _ = key.rsa_vp1(&signature).expect("in range");
        }
        let each = start.elapsed() / rounds;
        println!("RSAVP1, 2048-bit modulus, e = 65537: {each:?} each ({rounds} rounds)");
    }

    #[test]
    fn the_modulus_and_exponent_are_reported_verbatim() {
        // A 65-bit modulus: 9 bytes, and two limbs once it is packed.
        let key =
            RsaPublicKey::from_be_bytes(&[1, 0xc0, 0, 0, 0, 0, 0, 0, 1], &[1, 0, 1]).expect("ok");
        assert_eq!(key.key_bits(), 65);
        assert_eq!(key.modulus_bytes(), 9, "k = ceil(65 / 8)");
        assert_eq!(key.exponent(), 65537);
        // The same value without its leading zero byte is the 64-bit one, which
        // is what shows the zero is dropped rather than counted.
        let shorter =
            RsaPublicKey::from_be_bytes(&[0xc0, 0, 0, 0, 0, 0, 0, 1], &[1, 0, 1]).expect("ok");
        assert_eq!(shorter.key_bits(), 64);
        assert_eq!(shorter.modulus_bytes(), 8);
        // A modulus whose width is a whole number of limbs reports k = limbs * 8.
        let wide = RsaPublicKey::from_be_bytes(&[0xff; 128], &[1, 0, 1]).expect("ok");
        assert_eq!(wide.modulus_bytes(), 128);
        assert_eq!(wide.key_bits(), 1024);
    }
}
