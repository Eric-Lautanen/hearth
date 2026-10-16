//! ECDSA over P-256 with SHA-256: `ecdsa_secp256r1_sha256`, the other signature
//! scheme TLS 1.3 requires a client to support.
//!
//! The verification equation is X9.62's:
//!
//! ```text
//! e  = SHA-256(message), read as a 256-bit big-endian integer
//! w  = s^-1 mod n
//! u1 = e * w mod n
//! u2 = r * w mod n
//! R  = u1 * G + u2 * Q          (Q the public key)
//! accept  iff  R is not infinity  and  R.x mod n == r
//! ```
//!
//! `e` needs no truncation here: SHA-256 is 256 bits and `n` is a 256-bit prime,
//! so the whole hash is already the leftmost 256 bits the standard asks for. It
//! does still need a reduction, because `e` is a hash output rather than a scalar
//! and can be at or above `n`.
//!
//! # The signature must be canonical DER, and that is checked structurally
//!
//! The signature is a DER `ECDSA-Sig-Value`, `SEQUENCE { r INTEGER, s INTEGER }`,
//! and it is read with [`crate::der`] — the same reader certificates go through —
//! rather than with a hand-rolled integer decoder. That buys three refusals for
//! free and they are exactly the ones that matter: a redundant leading zero on
//! `r` or `s` (which would make two byte strings mean one signature), a negative
//! integer, and trailing bytes after the sequence, which are bytes the transcript
//! hash covers but the verifier did not read. RFC 8446 §4.2.3 requires the DER
//! form; the `r || s` raw form that some ecosystems use is not accepted, and a
//! signature whose two integers are not exactly the sequence's contents is not
//! accepted either.
//!
//! # What is deliberately not checked
//!
//! **Low-S is not enforced.** s and n - s are both valid signatures over the same
//! message, and requiring the smaller one is a Bitcoin consensus rule, not an
//! X9.62 or TLS one. Rejecting it here would refuse signatures that conforming CAs
//! and servers are entitled to produce, for a malleability that costs nothing in
//! TLS: the signature is over the handshake transcript, so there is no
//! signature-as-identifier to grind and no second party to convince.
//!
//! **Constant time is not the goal**, for the same reason as in [`crate::rsa`]:
//! the key, the scalars, the hash and the signature are all public. The final
//! comparison is a plain `==` between two 32-byte values, and the module says so
//! rather than pretending otherwise.
//!
//! # What this is not
//!
//! It does not read a certificate, build a chain, check a name or a validity
//! window, and **nothing calls it yet**: the only verifier the crate ships is
//! [`RejectAll`](crate::client::RejectAll), so no `https://` path exists. The
//! public key arrives as the SEC 1 point bytes a caller has already extracted,
//! and `Key`-to-scheme matching — a P-256 key must not be verified with an RSA
//! scheme — belongs to the verifier that will call this.

use crate::der::{Der, DerError, Tag};
use crate::p256::{self, P256Error, PublicKey};
use crate::sha256::Sha256;

/// Why a signature, a key or an encoding was refused.
///
/// Every variant is fatal to verification: there is no "accept anyway" and no
/// recoverable case. The variants name the rule that failed, so a diagnostic says
/// whether the key, the encoding or the equation was the problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EcdsaError {
    /// The public key is not a point this crate will use.
    PublicKey(P256Error),
    /// The signature is not a well-formed DER `ECDSA-Sig-Value`.
    Signature(DerError),
    /// `r` or `s` is zero, or at least `n`. Named so the error says which one.
    ScalarOutOfRange(&'static str),
    /// The signature is structurally fine and the equation did not hold.
    VerificationFailed,
}

impl std::fmt::Display for EcdsaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PublicKey(why) => write!(f, "unusable P-256 public key: {why}"),
            Self::Signature(why) => write!(f, "malformed ECDSA signature: {why}"),
            Self::ScalarOutOfRange(which) => {
                write!(f, "{which} is not in the range a signature may carry")
            }
            Self::VerificationFailed => write!(
                f,
                "the ECDSA verification equation does not hold for this signature"
            ),
        }
    }
}

impl std::error::Error for EcdsaError {}

impl From<P256Error> for EcdsaError {
    fn from(error: P256Error) -> Self {
        Self::PublicKey(error)
    }
}

impl From<DerError> for EcdsaError {
    fn from(error: DerError) -> Self {
        Self::Signature(error)
    }
}

/// ECDSA-P256-SHA256 verification (X9.62 §7.4.1.4): does `signature` sign
/// `message` under the public key in `public_key_sec1`?
///
/// `public_key_sec1` is the `SubjectPublicKeyInfo`'s `BIT STRING` contents —
/// `0x04 || X || Y` — as it appears in a certificate. `message` is the data that
/// was signed, not a pre-computed hash: for TLS, the 130-byte `signed_data` of
/// RFC 8446 §4.4.3. The signature is DER.
pub fn verify_ecdsa_secp256r1_sha256(
    public_key_sec1: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), EcdsaError> {
    let key = PublicKey::from_sec1(public_key_sec1)?;
    let (r, s) = parse_signature(signature)?;

    // Step 1: e is the hash, reduced into the scalar field. `n` is a 256-bit
    // prime, so the second half of the reduction is real rather than a formality:
    // a SHA-256 output is at or above `n` about 2^-32 of the time.
    let e = p256::scalar_reduce(p256::scalar_from_be(&Sha256::digest(message)));
    // Steps 2 and 3: w = s^-1 mod n, then u1 = e*w and u2 = r*w.
    let w = p256::scalar_invert(&s);
    let u1 = p256::scalar_mul(&e, &w);
    let u2 = p256::scalar_mul(&r, &w);
    // Step 4: R = u1*G + u2*Q. Infinity cannot satisfy the check below, since r is
    // not zero, so failing here is the same as failing there — but it is reported
    // as the distinct thing it is.
    let x = p256::double_mul(&key, &u2, &u1).ok_or(EcdsaError::VerificationFailed)?;
    // Step 5: v = R.x mod n, accepted when v == r. Every input is public, so this
    // is a plain comparison; see the module docs.
    let v = p256::scalar_reduce(p256::scalar_from_be(&x));
    if v == r {
        Ok(())
    } else {
        Err(EcdsaError::VerificationFailed)
    }
}

/// Reads a DER `ECDSA-Sig-Value` into `r` and `s`.
///
/// The whole structure is required to be exactly one sequence of exactly two
/// integers, the integers required to be minimal non-negative encodings by
/// [`crate::der`], and both required to be in `1..n`.
fn parse_signature(signature: &[u8]) -> Result<([u64; 4], [u64; 4]), EcdsaError> {
    let mut outer = Der::new(signature);
    let sequence = outer.take(Tag::SEQUENCE)?;
    // A second sequence, or any trailing byte, is data the signature does not
    // cover and this verifier would not have read.
    outer.expect_end()?;

    let mut inner = sequence.reader()?;
    let r = inner.take(Tag::INTEGER)?.as_integer_bytes()?;
    let s = inner.take(Tag::INTEGER)?.as_integer_bytes()?;
    // Exactly two integers: a third is the same class of unread trailing data.
    inner.expect_end()?;
    Ok((scalar(r, "r")?, scalar(s, "s")?))
}

/// A signature integer as a scalar, after checking it is in `1..n`.
///
/// `magnitude` is the DER magnitude with leading zeros already stripped, so a
/// value above 32 bytes is above `n` by width alone; anything shorter is
/// left-padded, which is how DER writes a small `r`.
fn scalar(magnitude: &[u8], which: &'static str) -> Result<[u64; 4], EcdsaError> {
    if magnitude.len() > 32 {
        return Err(EcdsaError::ScalarOutOfRange(which));
    }
    let mut padded = [0u8; 32];
    padded[32 - magnitude.len()..].copy_from_slice(magnitude);
    // Zero and `n` and above are all refused by the one range check: a signature
    // with r = 0 verifies for no message and no key, and s = 0 has no inverse, so
    // neither is a signature that can be accepted.
    if !p256::scalar_in_range(&padded) {
        return Err(EcdsaError::ScalarOutOfRange(which));
    }
    Ok(p256::scalar_from_be(&padded))
}
