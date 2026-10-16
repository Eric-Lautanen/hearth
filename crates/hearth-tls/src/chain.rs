//! Chain building: which certificate issued which, and whether the path ends at
//! an anchor the caller trusts.
//!
//! [`verify_certificate_list`] is the entry point a verifier calls. It takes the
//! `certificate_list` bytes from a `Certificate` message, the trust anchors the
//! caller has decided to trust, the instant to check validity against, and a
//! [`Policy`], and answers one question: is every link of this path signed by the
//! key of the next one, does the path end at an anchor, and does it respect the
//! rules below?
//!
//! # The order the checks run in, and why
//!
//! 1. **Framing, then parsing.** The wire form is length-prefixed DER; a
//!    certificate that does not parse is not a certificate. Both are fatal.
//! 2. **Structure.** The chain is no longer than the policy allows, the leaf is
//!    not a CA, every other certificate in the path *is* one, every key is a kind
//!    this crate can verify with and at least the policy's minimum size, and every
//!    validity window contains the instant.
//! 3. **Links.** For each certificate in the presented chain, its issuer is the
//!    **subject of the next one, compared as the DER of the names**, and its
//!    signature verifies with that certificate's public key under the scheme its
//!    own `signatureAlgorithm` names.
//! 4. **The anchor.** The last certificate's issuer name must be the subject name
//!    of one of the caller's anchors, **and** its signature must verify with that
//!    anchor's key.
//!
//! Structure before links is deliberate: a chain of the wrong shape is refused for
//! its shape, before any public-key operation is done on hostile input.
//!
//! # A name match is not trust
//!
//! Step 4 is the one to read twice. An anchor's authority comes from the caller,
//! not from the anchor being self-signed, so an anchor that is *not* self-signed
//! is not an error — that is exactly what pinning one key looks like. But a name
//! match on its own decides nothing: the certificate's signature must verify with
//! the anchor's key, or the path is unanchored. A chain whose links all verify but
//! whose top certificate was signed by some other key with the same name is
//! refused, and that is the case a name-only match would have accepted.
//!
//! # What this does not do, stated plainly
//!
//! - **No revocation checking of any kind.** No CRL, no OCSP, no stapling. A
//!   revoked certificate verifies here. This is the largest gap in what a success
//!   from this module means, and it is not fixable from `std` alone.
//! - **No host name matching** and no `nameConstraints` processing: a chain that
//!   verifies here may still be for the wrong server. That is the next slice.
//! - **No `pathLenConstraint` and no `keyUsage.keyCertSign` check.** Both are read
//!   out of the certificate by [`crate::x509`] — `path_len` and `key_usage` — and
//!   neither has a fixture that can make the check *fail*, so neither can be
//!   proven here yet. `docs/RESUME.md` names the certificates to generate.
//! - **No `extendedKeyUsage` policy on issuers**, no `policyConstraints`, no
//!   `inhibitAnyPolicy`, no name constraints.
//!
//! Every check that *is* here is fatal on failure: there is no tolerant mode, no
//! flag and no environment variable that turns one off.

use std::fmt;

use crate::client::UnixTime;
use crate::rsa::{RsaError, RsaPublicKey};
use crate::verify::{verify_certificate_signature, VerifyError};
use crate::x509::{Certificate, PublicKey, X509Error};

/// The knobs a caller sets. Everything else is fixed by the certificates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    /// The most certificates a presented chain may contain, the leaf included.
    ///
    /// This is the caller's cap, and it is *not* the certificates'
    /// `pathLenConstraint` — see the module docs for why that one is not enforced.
    /// Real chains are three or four certificates, so a low cap costs nothing and
    /// bounds the work a hostile server can ask for.
    pub max_chain_len: usize,
    /// The smallest RSA modulus accepted anywhere in the path, in bits.
    ///
    /// There is no equivalent floor for EC keys because [`crate::x509`] refuses
    /// every curve but `prime256v1`: any EC key that reaches here is 256 bits, so
    /// a configurable floor could only ever be set to something wrong. Note that
    /// [`crate::rsa`] itself deliberately accepts a 1024-bit key — RFC 8448's
    /// fixture is one — which is why this policy lives here rather than there.
    pub min_rsa_bits: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            max_chain_len: 6,
            min_rsa_bits: 2048,
        }
    }
}

/// Where a certificate sits in the path, so an error can name it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    /// An index into the presented chain, which is leaf-first: `Chain(0)` is the
    /// server's own certificate.
    Chain(usize),
    /// An index into the caller's trust anchors.
    Anchor(usize),
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Chain(index) => write!(f, "the certificate at index {index} of the chain"),
            Self::Anchor(index) => write!(f, "the anchor at index {index}"),
        }
    }
}

/// Why a chain was refused.
///
/// As everywhere else in this crate every variant is fatal, and each one names the
/// rule that failed. Variants carrying a [`Position`] say *which* certificate was
/// refused, because "the chain is bad" is not a useful diagnostic and "the
/// certificate at index 1 is not a CA" is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// The `certificate_list` was empty.
    EmptyList,
    /// The list ended inside an entry's three-byte length prefix.
    EndsInsideLength {
        /// How many bytes were left when the prefix was being read.
        remaining: usize,
    },
    /// An entry declared its length as zero, which RFC 8446 does not allow: the
    /// entry's own type carries a lower bound of one byte.
    ZeroLengthEntry {
        /// The offset of the entry's length prefix within the list.
        offset: usize,
    },
    /// An entry declared more bytes than the list still holds, so the list is
    /// truncated or the lengths are lying.
    TruncatedEntry {
        /// The offset of the entry's length prefix within the list.
        offset: usize,
        /// What the prefix said.
        declared: usize,
        /// What was actually left behind it.
        remaining: usize,
    },
    /// A certificate did not parse.
    Certificate {
        /// Its index in the presented chain.
        index: usize,
        /// Why the DER was refused.
        error: X509Error,
    },
    /// The chain holds more certificates than the policy allows.
    ChainTooLong {
        /// How many were presented.
        len: usize,
        /// The policy's cap.
        max: usize,
    },
    /// The leaf's `basicConstraints` says it is a CA, so it is not a server
    /// certificate: a CA certificate cannot be used as an end entity.
    CaUsedAsLeaf,
    /// A certificate that must be a CA is not one.
    NotACa(Position),
    /// The issuer of one certificate is not the subject of the next, so the
    /// presented list is not a chain at all.
    BrokenLink {
        /// The index of the certificate whose issuer could not be found in the
        /// certificate that follows it.
        index: usize,
    },
    /// A public key's algorithm is not one this crate verifies with, so no link
    /// through it could be checked.
    UnsupportedKey {
        /// Where the key is.
        at: Position,
        /// The algorithm's OID, as its DER contents.
        oid: Vec<u8>,
    },
    /// An RSA key could not be read, or is not a usable key at all.
    UnusableKey {
        /// Where the key is.
        at: Position,
        /// Why the modulus or exponent was refused.
        error: RsaError,
    },
    /// An RSA key is smaller than the policy allows.
    KeyTooSmall {
        /// Where the key is.
        at: Position,
        /// The key's size.
        bits: usize,
        /// The policy's floor.
        min_bits: usize,
    },
    /// A validity window begins after the instant being checked.
    NotYetValid(Position),
    /// A validity window has already ended.
    Expired(Position),
    /// A signature did not verify with the key of the certificate that should
    /// have made it.
    Signature {
        /// The certificate whose signature was being checked.
        certificate: Position,
        /// Why the check failed, including whether the scheme and the key could
        /// not go together in the first place.
        error: VerifyError,
    },
    /// No anchor has the issuer name of the last certificate in the chain, so
    /// there is nothing to anchor the path to.
    Unanchored {
        /// The issuer name that could not be resolved, as the DER of the `Name`.
        issuer: Vec<u8>,
    },
}

/// Renders a name's DER for a diagnostic: hex, truncated, never as text.
///
/// A name is compared as bytes, so this is only for a human reading an error. It
/// deliberately does not decode the name into a string, because a message that
/// looks like a host name is exactly the thing that must not be trusted.
fn name_hex(der: &[u8]) -> String {
    const SHOWN: usize = 24;
    let mut out = String::new();
    for byte in der.iter().take(SHOWN) {
        out.push_str(&format!("{byte:02x}"));
    }
    if der.len() > SHOWN {
        out.push_str(&format!("... ({} bytes)", der.len()));
    }
    out
}

impl fmt::Display for ChainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyList => write!(f, "the certificate list is empty"),
            Self::EndsInsideLength { remaining } => write!(
                f,
                "the certificate list ends inside an entry's length, with {remaining} bytes left"
            ),
            Self::ZeroLengthEntry { offset } => write!(
                f,
                "the certificate list entry at offset {offset} declares a length of zero"
            ),
            Self::TruncatedEntry {
                offset,
                declared,
                remaining,
            } => write!(
                f,
                "the certificate list entry at offset {offset} declares {declared} bytes with \
                 {remaining} left"
            ),
            Self::Certificate { index, error } => {
                write!(f, "the certificate at index {index} was refused: {error}")
            }
            Self::ChainTooLong { len, max } => write!(
                f,
                "the chain holds {len} certificates, more than the {max} the policy allows"
            ),
            Self::CaUsedAsLeaf => write!(
                f,
                "the leaf is a CA certificate (basicConstraints says CA is true), so it cannot be \
                 used as an end entity"
            ),
            Self::NotACa(at) => write!(
                f,
                "{at} is in the chain as an issuer but is not a CA (basicConstraints)"
            ),
            Self::BrokenLink { index } => write!(
                f,
                "the issuer of the certificate at index {index} is not the subject of the next \
                 certificate, so this is not a chain"
            ),
            Self::UnsupportedKey { at, oid } => write!(
                f,
                "the public key of {at} is not a kind this crate verifies with ({})",
                oid.iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            ),
            Self::UnusableKey { at, error } => {
                write!(f, "the public key of {at} is not usable: {error}")
            }
            Self::KeyTooSmall { at, bits, min_bits } => write!(
                f,
                "the RSA key of {at} is {bits} bits, below the {min_bits} the policy requires"
            ),
            Self::NotYetValid(at) => {
                write!(f, "{at} is not valid yet at the instant being checked")
            }
            Self::Expired(at) => {
                write!(f, "{at} has expired at the instant being checked")
            }
            Self::Signature { certificate, error } => write!(
                f,
                "the signature on {certificate} does not verify with its issuer's key: {error}"
            ),
            Self::Unanchored { issuer } => write!(
                f,
                "no trust anchor has the issuer name of the last certificate ({}), so the path \
                 reaches nothing trusted",
                name_hex(issuer)
            ),
        }
    }
}

impl std::error::Error for ChainError {}

/// Why a `certificate_list`'s framing was refused.
///
/// Separate from [`ChainError`] so that `client.rs`, which validates the same
/// bytes inside the `Certificate` message and has its own error type, can share
/// this walk instead of keeping a second copy of the rule that could drift from
/// this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ListError {
    /// There are no entries at all.
    Empty,
    /// The list ended inside an entry's three-byte length prefix, with this many
    /// bytes left.
    EndsInsideLength { remaining: usize },
    /// An entry at this offset declared a length of zero.
    ZeroLengthEntry { offset: usize },
    /// An entry at this offset declared more bytes than the list holds.
    TruncatedEntry {
        offset: usize,
        declared: usize,
        remaining: usize,
    },
}

/// Splits a `certificate_list` into its DER entries per RFC 8446 §4.4.2.
///
/// Each entry is `opaque certificate<1..2^24-1>`: a three-byte big-endian length
/// followed by exactly that many bytes of DER, and the entries run to the end of
/// the list with nothing left over. A length of zero is refused rather than
/// treated as an empty entry, because the type's range does not include it.
pub(crate) fn split_entries(list: &[u8]) -> Result<Vec<&[u8]>, ListError> {
    let mut entries = Vec::new();
    let mut offset = 0usize;
    while offset < list.len() {
        let rest = &list[offset..];
        if rest.len() < 3 {
            return Err(ListError::EndsInsideLength {
                remaining: rest.len(),
            });
        }
        let declared = u32::from_be_bytes([0, rest[0], rest[1], rest[2]]) as usize;
        if declared == 0 {
            return Err(ListError::ZeroLengthEntry { offset });
        }
        if rest.len() < 3 + declared {
            return Err(ListError::TruncatedEntry {
                offset,
                declared,
                remaining: rest.len() - 3,
            });
        }
        entries.push(&rest[3..3 + declared]);
        offset += 3 + declared;
    }
    if entries.is_empty() {
        return Err(ListError::Empty);
    }
    Ok(entries)
}

/// The DER entries of a `certificate_list`, checked for framing.
///
/// Public because a caller that wants to hold the certificates — to look at the
/// leaf's names, or to parse them once and check several things — needs the split
/// without the rest of the chain logic. [`verify_certificate_list`] is the entry
/// point that does everything.
pub fn entries(list: &[u8]) -> Result<Vec<&[u8]>, ChainError> {
    split_entries(list).map_err(|error| match error {
        ListError::Empty => ChainError::EmptyList,
        ListError::EndsInsideLength { remaining } => ChainError::EndsInsideLength { remaining },
        ListError::ZeroLengthEntry { offset } => ChainError::ZeroLengthEntry { offset },
        ListError::TruncatedEntry {
            offset,
            declared,
            remaining,
        } => ChainError::TruncatedEntry {
            offset,
            declared,
            remaining,
        },
    })
}

/// Checks one certificate's validity window against `time`.
///
/// The comparison is the one [`UnixTime::outside`] makes — `time < notBefore ||
/// time > notAfter`, with both bounds inclusive — written out here rather than
/// called so that the refusal can say *which* end was crossed.
fn check_validity(
    certificate: &Certificate<'_>,
    at: Position,
    time: UnixTime,
) -> Result<(), ChainError> {
    if time < certificate.not_before() {
        return Err(ChainError::NotYetValid(at));
    }
    if time > certificate.not_after() {
        return Err(ChainError::Expired(at));
    }
    Ok(())
}

/// Checks a key's kind and, for RSA, its size against the policy.
fn check_key(key: &PublicKey<'_>, at: Position, policy: &Policy) -> Result<(), ChainError> {
    match key {
        PublicKey::Rsa { modulus, exponent } => {
            // The same reader the signature check uses, so a key this accepts is a
            // key that can be verified with.
            let key = RsaPublicKey::from_be_bytes(modulus, &exponent.to_be_bytes())
                .map_err(|error| ChainError::UnusableKey { at, error })?;
            let bits = key.key_bits();
            if bits < policy.min_rsa_bits {
                return Err(ChainError::KeyTooSmall {
                    at,
                    bits,
                    min_bits: policy.min_rsa_bits,
                });
            }
            Ok(())
        }
        // `x509` yields an `Ec` key only for `prime256v1` and only in the
        // uncompressed encoding, so this is a 256-bit P-256 point: there is no
        // size decision to make.
        PublicKey::Ec { .. } => Ok(()),
        PublicKey::Unsupported(oid) => Err(ChainError::UnsupportedKey {
            at,
            oid: oid.as_bytes().to_vec(),
        }),
    }
}

/// Checks a chain of already-parsed certificates against the caller's anchors.
///
/// `chain` is the presented path, leaf first, exactly as the `Certificate`
/// message carries it; `anchors` are the trust anchors, and this function makes no
/// judgement about them beyond requiring that one of them actually signed the top
/// of the path — an anchor that is not self-signed is allowed, because pinning a
/// key is a legitimate thing for a caller to do.
///
/// The chain may include the anchor itself; either way, the last certificate's
/// issuer must resolve to an anchor.
pub fn verify_chain(
    chain: &[Certificate<'_>],
    anchors: &[Certificate<'_>],
    time: UnixTime,
    policy: &Policy,
) -> Result<(), ChainError> {
    if chain.is_empty() {
        return Err(ChainError::EmptyList);
    }
    if chain.len() > policy.max_chain_len {
        return Err(ChainError::ChainTooLong {
            len: chain.len(),
            max: policy.max_chain_len,
        });
    }
    // The leaf is an end entity, not a CA. This is the check that stops a CA
    // certificate being presented as a server certificate.
    if chain[0].is_ca() {
        return Err(ChainError::CaUsedAsLeaf);
    }
    // Every other certificate in the path is an issuer, so it must be a CA: a
    // non-CA certificate has no business signing other certificates, whatever its
    // name says.
    for (index, issuer) in chain.iter().enumerate().skip(1) {
        if !issuer.is_ca() {
            return Err(ChainError::NotACa(Position::Chain(index)));
        }
    }
    for (index, certificate) in chain.iter().enumerate() {
        let at = Position::Chain(index);
        check_validity(certificate, at, time)?;
        check_key(certificate.public_key(), at, policy)?;
    }
    // The links: name, then signature. The name is compared as the DER of the two
    // `Name` values rather than as rendered text, because two names that print the
    // same can encode differently and it is the encoding the signature covers.
    for (index, pair) in chain.windows(2).enumerate() {
        let (certificate, issuer) = (&pair[0], &pair[1]);
        if certificate.issuer().as_der() != issuer.subject().as_der() {
            return Err(ChainError::BrokenLink { index });
        }
        verify_certificate_signature(certificate, issuer.public_key()).map_err(|error| {
            ChainError::Signature {
                certificate: Position::Chain(index),
                error,
            }
        })?;
    }
    // The anchor. Candidates are found by name and *then* must verify the
    // signature: a name match alone would accept any key with the right name.
    let last = &chain[chain.len() - 1];
    let last_index = chain.len() - 1;
    let mut failure = None;
    for (index, anchor) in anchors.iter().enumerate() {
        if anchor.subject().as_der() != last.issuer().as_der() {
            continue;
        }
        let at = Position::Anchor(index);
        if let Err(error) = check_validity(anchor, at, time) {
            failure = Some(error);
            continue;
        }
        if let Err(error) = check_key(anchor.public_key(), at, policy) {
            failure = Some(error);
            continue;
        }
        match verify_certificate_signature(last, anchor.public_key()) {
            Ok(()) => return Ok(()),
            Err(error) => {
                failure = Some(ChainError::Signature {
                    certificate: Position::Chain(last_index),
                    error,
                });
            }
        }
    }
    match failure {
        Some(error) => Err(error),
        None => Err(ChainError::Unanchored {
            issuer: last.issuer().as_der().to_vec(),
        }),
    }
}

/// The whole path from the wire bytes: framing, parsing, then [`verify_chain`].
///
/// `list` is `CertificateVerifyInput::certificate_list` verbatim, and `time` is
/// `CertificateVerifyInput::validation_time` — so a verifier's whole chain check is
/// one call with the fields the handshake already hands it.
///
/// What a success means, exactly: the entries frame correctly, each certificate
/// parses, the path respects the structural rules and the policy, each link is
/// signed by the next certificate's key, and the last one is signed by an anchor's
/// key. What it does **not** mean is that the leaf is for the server that was
/// asked for, or that nothing in the path was revoked — see the module docs.
pub fn verify_certificate_list(
    list: &[u8],
    anchors: &[Certificate<'_>],
    time: UnixTime,
    policy: &Policy,
) -> Result<(), ChainError> {
    let entries = entries(list)?;
    let mut chain = Vec::with_capacity(entries.len());
    for (index, der) in entries.iter().enumerate() {
        match Certificate::parse(der) {
            Ok(certificate) => chain.push(certificate),
            Err(error) => return Err(ChainError::Certificate { index, error }),
        }
    }
    verify_chain(&chain, anchors, time, policy)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames `bodies` as a `certificate_list`.
    fn framed(bodies: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for body in bodies {
            out.extend_from_slice(&(body.len() as u32).to_be_bytes()[1..]);
            out.extend_from_slice(body);
        }
        out
    }

    #[test]
    fn a_well_framed_list_splits_into_its_entries() {
        let list = framed(&[&[0xaa, 0xbb], &[0x01]]);
        assert_eq!(entries(&list), Ok(vec![&[0xaa, 0xbb][..], &[0x01][..]]));
        // And one entry is the common case.
        assert_eq!(entries(&framed(&[&[0xaa]])), Ok(vec![&[0xaa][..]]));
    }

    #[test]
    fn framing_negatives_are_refused_by_name() {
        // No entries at all.
        assert_eq!(entries(&[]), Err(ChainError::EmptyList));
        // A length prefix that is not three bytes long.
        assert_eq!(
            entries(&[0x00, 0x02]),
            Err(ChainError::EndsInsideLength { remaining: 2 })
        );
        // And the same through the public mapping, where an empty list is its own
        // variant rather than a framing failure.
        assert_eq!(entries(&[]), Err(ChainError::EmptyList));
        // A declared length that runs past the end of the list, and the same with
        // one byte of body present.
        assert_eq!(
            split_entries(&[0x00, 0x00, 0x05]),
            Err(ListError::TruncatedEntry {
                offset: 0,
                declared: 5,
                remaining: 0
            })
        );
        assert_eq!(
            split_entries(&[0x00, 0x00, 0x05, 0x01]),
            Err(ListError::TruncatedEntry {
                offset: 0,
                declared: 5,
                remaining: 1
            })
        );
        // The same, one entry in, so the offset is not the zero that a bug could
        // pass by only ever reporting.
        let mut with_first = framed(&[&[0xaa]]);
        with_first.extend_from_slice(&[0x00, 0x00, 0x09, 0x01]);
        assert_eq!(
            split_entries(&with_first),
            Err(ListError::TruncatedEntry {
                offset: 4,
                declared: 9,
                remaining: 1
            })
        );
        // A zero-length entry is not an entry: `opaque certificate<1..2^24-1>` has
        // no zero in its range.
        assert_eq!(
            split_entries(&[0x00, 0x00, 0x00]),
            Err(ListError::ZeroLengthEntry { offset: 0 })
        );
    }

    #[test]
    fn a_trailing_partial_length_is_not_a_valid_tail() {
        // One well-formed entry, then two bytes: the list ends inside a prefix.
        let mut list = framed(&[&[0xaa, 0xbb]]);
        list.extend_from_slice(&[0x00, 0x01]);
        assert_eq!(
            split_entries(&list),
            Err(ListError::EndsInsideLength { remaining: 2 })
        );
    }

    #[test]
    fn the_default_policy_is_the_documented_one() {
        let policy = Policy::default();
        assert_eq!(policy.max_chain_len, 6);
        assert_eq!(policy.min_rsa_bits, 2048);
        // RFC 8448's own certificate carries a 1024-bit key, and this floor is
        // what refuses it: the primitive in `rsa.rs` still accepts the key, which
        // is the split the module docs describe.
        assert!(1024 < policy.min_rsa_bits);
    }

    #[test]
    fn a_position_names_where_it_is() {
        assert_eq!(
            Position::Chain(0).to_string(),
            "the certificate at index 0 of the chain"
        );
        assert_eq!(Position::Anchor(2).to_string(), "the anchor at index 2");
    }

    #[test]
    fn a_name_in_a_diagnostic_is_hex_not_text() {
        // The point of the helper: a name that would decode to a plausible host
        // name must not appear as one in an error message.
        assert_eq!(name_hex(&[0x30, 0x03, 0x41, 0x62, 0x63]), "3003416263");
        assert_eq!(name_hex(&[]), "");
        // Long names are truncated and say how long they were.
        let long = vec![0x41; 40];
        let rendered = name_hex(&long);
        assert!(rendered.ends_with("... (40 bytes)"));
        assert_eq!(rendered.len(), 24 * 2 + "... (40 bytes)".len());
    }
}
