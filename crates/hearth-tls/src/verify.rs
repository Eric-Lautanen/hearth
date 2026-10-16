//! The decision half, in its first pieces: **which scheme may verify which key**,
//! and the two signature checks a chain needs.
//!
//! This is where the primitives in [`crate::rsa`], [`crate::ecdsa`] and
//! [`crate::p256`] meet the certificates in [`crate::x509`]. It is not a verifier:
//! nothing here builds a chain, matches a host name, or checks a validity window,
//! and nothing here is called by [`crate::client`]. What it does is the part that
//! has to be right *before* any of those can be, and it is the part that is a
//! security bug rather than a parse error when it is wrong.
//!
//! # Why a scheme and a key have to be matched
//!
//! A signature says which algorithm produced it, and a public key says what kind
//! of key it is. Accepting any combination is the bug class behind a long line of
//! breaks: an RSA key read as an elliptic curve point, a P-256 key verified with
//! an RSA scheme, a scheme from one protocol replayed into another that hands the
//! same bytes to a different primitive. So the matching here is by *kind* and is
//! the first thing each entry point does:
//!
//! - `rsa_pss_rsae_sha256` and `sha256WithRSAEncryption` need an **RSA** key;
//! - `ecdsa_secp256r1_sha256` needs a **P-256** key.
//!
//! A key that is neither is [`VerifyError::UnsupportedKeyType`], and a key of the
//! wrong kind is [`VerifyError::SchemeKeyMismatch`], which names both sides so a
//! diagnostic says which scheme met which key.
//!
//! # The one place PKCS#1 v1.5 is allowed, and the one place it is not
//!
//! TLS 1.3 removed the PKCS#1 v1.5 schemes from the handshake — RFC 8446 §4.2.3
//! keeps PSS and the ECDSA schemes for `signature_algorithms` — while RFC 5280
//! chains are *full* of `sha256WithRSAEncryption`, because that is what
//! certificates are signed with. So [`verify_handshake_signature`] accepts exactly
//! the two suites this crate implements (0x0804 and 0x0403), and
//! [`verify_certificate_signature`] additionally accepts PKCS#1 v1.5.
//!
//! That asymmetry is enforced by construction rather than by a comment:
//! [`Suite::from_handshake_scheme`] maps only the two implemented handshake
//! schemes, and its test sweeps **all 65536** code points to show that nothing
//! else is ever mapped. The PKCS#1 v1.5 code points are deliberately not
//! enumerated anywhere — there is no memorized registry value here to get wrong,
//! and a scheme is never accepted by falling through to a default.
//!
//! # What is here now, and what is not
//!
//! Everything this module was waiting for exists elsewhere in the crate: chain building
//! with the structural rules, the validity window and revocation are in [`crate::chain`]
//! (over [`crate::crl`]), host name matching and the leaf's purpose are in
//! [`crate::leaf`], both call this module for the arithmetic, and
//! [`crate::verifier::PinnedVerifier`] is the type that calls all of them in the order a
//! handshake needs — so the whole certificate decision is reachable, with
//! [`crate::RejectAll`] still there as the verifier a caller who has decided nothing gets.
//!
//! What is *not* here is still the trust decision's *inputs*: which anchors a caller trusts
//! and where they come from is the caller's own decision, and there is no root store in this
//! crate to inherit by accident. There is no OCSP, no stapling, and nothing that fetches a CRL
//! — a list is bytes the caller supplies, and keeping one fresh is a job for something that
//! talks to the network. Each open item is in `docs/RESUME.md` in the order it can be proven,
//! and none of it may be reachable from a URL before it exists.

use crate::ecdsa::{verify_ecdsa_secp256r1_sha256, EcdsaError};
use crate::handshake::{SIG_ECDSA_SECP256R1_SHA256, SIG_RSA_PSS_RSAE_SHA256};
use crate::rsa::{RsaError, RsaPublicKey};
use crate::x509::{Certificate, PublicKey, SignatureAlgorithm};

/// A signature scheme this crate can verify, paired with the kind of key it needs.
///
/// The pairing is the point: a `Suite` value is what a caller gets *after* a
/// scheme has been matched against a key kind, so the dispatch below cannot be
/// reached with a mismatched pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Suite {
    /// `rsa_pss_rsae_sha256`: an RSA key, RSASSA-PSS with MGF1-SHA-256 and a
    /// 32-byte salt. The scheme TLS 1.3 requires for an RSA handshake signature.
    RsaPssSha256,
    /// `sha256WithRSAEncryption`: an RSA key, RSASSA-PKCS1-v1_5. **Certificates
    /// only** — see the module docs, and [`Suite::from_handshake_scheme`], which
    /// cannot produce this value.
    RsaPkcs1Sha256,
    /// `ecdsa_secp256r1_sha256`: a P-256 key.
    EcdsaSecp256r1Sha256,
}

impl Suite {
    /// The scheme's name, for diagnostics and for the error a mismatch produces.
    pub fn name(self) -> &'static str {
        match self {
            Self::RsaPssSha256 => "rsa_pss_rsae_sha256",
            Self::RsaPkcs1Sha256 => "sha256WithRSAEncryption",
            Self::EcdsaSecp256r1Sha256 => "ecdsa_secp256r1_sha256",
        }
    }

    /// The suite a `CertificateVerify` `signature_scheme` names, or `None` for a
    /// scheme that must not be used in a handshake.
    ///
    /// Exactly two values map, and both come from [`crate::handshake`]'s constants
    /// rather than from literals here, so the code points live in one place. Every
    /// other value — the PKCS#1 v1.5 schemes, the SHA-384 and SHA-512 suites, the
    /// curves this crate does not implement, the legacy SHA-1 pairs, and the 65530
    /// unassigned code points — returns `None`, which the caller turns into a
    /// refusal. There is no default arm that accepts anything.
    pub fn from_handshake_scheme(scheme: u16) -> Option<Self> {
        match scheme {
            SIG_RSA_PSS_RSAE_SHA256 => Some(Self::RsaPssSha256),
            SIG_ECDSA_SECP256R1_SHA256 => Some(Self::EcdsaSecp256r1Sha256),
            _ => None,
        }
    }

    /// The suite a certificate's own `signatureAlgorithm` names.
    ///
    /// Two of the three shapes in [`SignatureAlgorithm`] map directly. The third,
    /// RSASSA-PSS, carries its parameters, and the salt length is the one that has
    /// to be checked here: TLS 1.3 requires a salt as long as the digest for the
    /// *handshake* schemes, and RFC 4055 says a certificate's PSS salt should be
    /// the hash length too — so a certificate that names any other salt length is
    /// refused rather than verified with whatever salt it asked for.
    pub fn from_certificate_algorithm(
        algorithm: &SignatureAlgorithm<'_>,
    ) -> Result<Self, VerifyError> {
        match algorithm {
            SignatureAlgorithm::RsaPkcs1Sha256 => Ok(Self::RsaPkcs1Sha256),
            SignatureAlgorithm::RsaPssSha256 { salt_len } => {
                if *salt_len != crate::rsa::SHA256_LEN as u32 {
                    return Err(VerifyError::PssSaltLength {
                        salt_len: *salt_len,
                    });
                }
                Ok(Self::RsaPssSha256)
            }
            SignatureAlgorithm::EcdsaSha256 => Ok(Self::EcdsaSecp256r1Sha256),
            SignatureAlgorithm::Unsupported(oid) => Err(
                VerifyError::UnsupportedSignatureAlgorithm(oid.as_bytes().to_vec()),
            ),
        }
    }
}

/// Why a signature, a scheme or a key was refused.
///
/// As everywhere else in this crate, every variant is fatal: there is no tolerant
/// mode, and each one names the rule that failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyError {
    /// The scheme cannot be used with this kind of key. Both sides are named so the
    /// diagnostic says which scheme met which key.
    SchemeKeyMismatch {
        /// The scheme's name, as [`Suite::name`] reports it.
        suite: &'static str,
        /// The kind of key it was given.
        key: &'static str,
    },
    /// The certificate's key is a type this crate cannot verify with, named by its
    /// OID contents.
    UnsupportedKeyType(Vec<u8>),
    /// The certificate is signed with an algorithm this crate does not implement,
    /// named by its OID contents.
    UnsupportedSignatureAlgorithm(Vec<u8>),
    /// A `signature_scheme` that must not appear in a handshake. The wire value is
    /// carried verbatim: it is public, and the diagnostic is more useful with the
    /// number than with a guess at a name.
    UnsupportedScheme(u16),
    /// A PSS certificate whose salt length is not the digest length.
    PssSaltLength {
        /// The salt length from the `AlgorithmIdentifier`.
        salt_len: u32,
    },
    /// The RSA part of the check failed.
    Rsa(RsaError),
    /// The P-256 part of the check failed.
    Ecdsa(EcdsaError),
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchemeKeyMismatch { suite, key } => {
                write!(f, "{suite} cannot be verified with {key}")
            }
            Self::UnsupportedKeyType(oid) => {
                write!(
                    f,
                    "the key's algorithm is not one this crate verifies ({oid:02x?})"
                )
            }
            Self::UnsupportedSignatureAlgorithm(oid) => write!(
                f,
                "the signature algorithm is not one this crate verifies ({oid:02x?})"
            ),
            Self::UnsupportedScheme(scheme) => write!(
                f,
                "signature scheme {scheme:#06x} is not one this client offers"
            ),
            Self::PssSaltLength { salt_len } => write!(
                f,
                "PSS names a {salt_len}-byte salt, and only a 32-byte one is accepted"
            ),
            Self::Rsa(why) => write!(f, "the RSA check failed: {why}"),
            Self::Ecdsa(why) => write!(f, "the P-256 check failed: {why}"),
        }
    }
}

impl std::error::Error for VerifyError {}

impl From<RsaError> for VerifyError {
    fn from(error: RsaError) -> Self {
        Self::Rsa(error)
    }
}

impl From<EcdsaError> for VerifyError {
    fn from(error: EcdsaError) -> Self {
        Self::Ecdsa(error)
    }
}

/// Verifies the signature on `certificate` with the public key of the certificate
/// that issued it: one link of a chain, checked.
///
/// The scheme comes from the certificate itself
/// ([`Certificate::signature_algorithm`]), the message is its
/// [`Certificate::tbs`] — the bytes the signature covers, which is asserted
/// elsewhere to be exactly what a signature over this certificate signs — and the
/// key is the issuer's, so a caller that has already linked issuer to subject by
/// name can check the link here.
///
/// Nothing here decides *whether* `issuer_key` is the right key to trust; that is
/// chain building and the root store, and they are not written yet. What this
/// guarantees is narrower and provable: **if** the issuer's key is the key that
/// signed this certificate, then this certificate was signed by it.
pub fn verify_certificate_signature(
    certificate: &Certificate<'_>,
    issuer_key: &PublicKey<'_>,
) -> Result<(), VerifyError> {
    verify_tbs_signature(
        certificate.signature_algorithm(),
        certificate.tbs(),
        certificate.signature(),
        issuer_key,
    )
}

/// Verifies a signature over the `tbs` bytes of an X.509 structure.
///
/// This is the shared half of every certificate-shaped signature check: a CRL is
/// signed exactly the way a certificate is — the same `AlgorithmIdentifier`, the same
/// `tbs`-covers-everything-but-the-signature rule, the same three suites — so both
/// callers go through here rather than each picking a suite for itself. Two copies of
/// "which scheme may verify which key" is two chances for one of them to be the
/// lenient one.
///
/// `tbs` must be the original bytes of the structure, never a re-encoding: the whole
/// point of the check is that the signature covers the fields that were *read*.
pub fn verify_tbs_signature(
    algorithm: &SignatureAlgorithm<'_>,
    tbs: &[u8],
    signature: &[u8],
    issuer_key: &PublicKey<'_>,
) -> Result<(), VerifyError> {
    let suite = Suite::from_certificate_algorithm(algorithm)?;
    let message = tbs;
    match suite {
        Suite::RsaPkcs1Sha256 => rsa_key(issuer_key, suite)?
            .verify_pkcs1v15_sha256(message, signature)
            .map_err(VerifyError::from),
        Suite::RsaPssSha256 => rsa_key(issuer_key, suite)?
            .verify_pss_sha256(message, signature)
            .map_err(VerifyError::from),
        Suite::EcdsaSecp256r1Sha256 => {
            // The curve is checked where the key is parsed: `x509` yields an `Ec`
            // point only for `prime256v1`, and refuses a compressed one, so this
            // slice is always a 65-byte uncompressed P-256 point.
            let point = ec_point(issuer_key, suite)?;
            verify_ecdsa_secp256r1_sha256(point, message, signature).map_err(VerifyError::from)
        }
    }
}

/// Verifies a `CertificateVerify` signature.
///
/// `scheme` is the `signature_scheme` from the message, `key` is the **leaf's**
/// public key, `signed_data` is the RFC 8446 §4.4.3 construction the state machine
/// already builds (64 spaces, the context string, a zero byte, the transcript hash
/// through `Certificate`), and `signature` is the signature bytes.
///
/// A scheme this crate does not implement, and every scheme RFC 8446 removed from
/// the handshake, is [`VerifyError::UnsupportedScheme`] with the wire value — so a
/// peer cannot get a handshake signature accepted with PKCS#1 v1.5 even though
/// [`verify_certificate_signature`] accepts it for certificates.
pub fn verify_handshake_signature(
    scheme: u16,
    key: &PublicKey<'_>,
    signed_data: &[u8],
    signature: &[u8],
) -> Result<(), VerifyError> {
    let suite =
        Suite::from_handshake_scheme(scheme).ok_or(VerifyError::UnsupportedScheme(scheme))?;
    match suite {
        // Unreachable, and asserted to be by the sweep test: no code point maps to
        // PKCS#1 v1.5. Written as an arm rather than an `unreachable!()` so that if
        // a future edit ever did map one, the refusal happens here instead of the
        // signature being checked with a scheme the handshake does not allow.
        Suite::RsaPkcs1Sha256 => Err(VerifyError::UnsupportedScheme(scheme)),
        Suite::RsaPssSha256 => rsa_key(key, suite)?
            .verify_pss_sha256(signed_data, signature)
            .map_err(VerifyError::from),
        Suite::EcdsaSecp256r1Sha256 => {
            let point = ec_point(key, suite)?;
            verify_ecdsa_secp256r1_sha256(point, signed_data, signature).map_err(VerifyError::from)
        }
    }
}

/// The RSA key, or the error that says why this key cannot be used for this suite.
fn rsa_key(key: &PublicKey<'_>, suite: Suite) -> Result<RsaPublicKey, VerifyError> {
    match key {
        // The exponent is a `u64` by the time it reaches here; `from_be_bytes`
        // reads it big-endian and strips leading zeros, which is how a certificate
        // writes it.
        PublicKey::Rsa { modulus, exponent } => Ok(RsaPublicKey::from_be_bytes(
            modulus,
            &exponent.to_be_bytes(),
        )?),
        PublicKey::Ec { .. } => Err(VerifyError::SchemeKeyMismatch {
            suite: suite.name(),
            key: "a P-256 key",
        }),
        PublicKey::Unsupported(oid) => {
            Err(VerifyError::UnsupportedKeyType(oid.as_bytes().to_vec()))
        }
    }
}

/// The SEC 1 point of a P-256 key, or the error that says why this key cannot be
/// used for this suite.
fn ec_point<'a>(key: &PublicKey<'a>, suite: Suite) -> Result<&'a [u8], VerifyError> {
    match key {
        PublicKey::Ec { point } => Ok(point),
        PublicKey::Rsa { .. } => Err(VerifyError::SchemeKeyMismatch {
            suite: suite.name(),
            key: "an RSA key",
        }),
        PublicKey::Unsupported(oid) => {
            Err(VerifyError::UnsupportedKeyType(oid.as_bytes().to_vec()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_two_implemented_suites_can_come_from_a_handshake() {
        // Every code point, not a list of the interesting ones. This is what
        // enforces RFC 8446 §4.2.3's removal of PKCS#1 v1.5 from the handshake
        // *without* naming a single registry value: whatever 0x0401 and its
        // siblings are spelled, none of them maps to a suite, because only two
        // values map at all and both are the crate's own constants.
        let mut mapped = Vec::new();
        for scheme in 0u16..=u16::MAX {
            if let Some(suite) = Suite::from_handshake_scheme(scheme) {
                mapped.push((scheme, suite));
            }
        }
        // The sweep is ordered by code point, so this is a set comparison rather
        // than a sequence one: what matters is that these two and nothing else.
        assert_eq!(
            mapped.len(),
            2,
            "exactly two schemes may come from a handshake: {mapped:?}"
        );
        assert!(mapped.contains(&(SIG_RSA_PSS_RSAE_SHA256, Suite::RsaPssSha256)));
        assert!(mapped.contains(&(SIG_ECDSA_SECP256R1_SHA256, Suite::EcdsaSecp256r1Sha256)));
        // And the value PKCS#1 v1.5 would be is refused, whatever it is spelled:
        // the sweep above is the general statement, this is the specific one that
        // would catch a future edit adding a third mapping.
        assert!(!mapped
            .iter()
            .any(|(_, suite)| *suite == Suite::RsaPkcs1Sha256));
    }

    #[test]
    fn a_certificates_algorithm_maps_to_its_suite_and_its_salt_is_checked() {
        assert_eq!(
            Suite::from_certificate_algorithm(&SignatureAlgorithm::RsaPkcs1Sha256),
            Ok(Suite::RsaPkcs1Sha256)
        );
        assert_eq!(
            Suite::from_certificate_algorithm(&SignatureAlgorithm::RsaPssSha256 { salt_len: 32 }),
            Ok(Suite::RsaPssSha256)
        );
        assert_eq!(
            Suite::from_certificate_algorithm(&SignatureAlgorithm::EcdsaSha256),
            Ok(Suite::EcdsaSecp256r1Sha256)
        );
        // A PSS certificate asking for a shorter salt is refused rather than
        // verified with the salt it asked for, which is the difference between
        // policy and tolerance.
        assert_eq!(
            Suite::from_certificate_algorithm(&SignatureAlgorithm::RsaPssSha256 { salt_len: 20 }),
            Err(VerifyError::PssSaltLength { salt_len: 20 })
        );
    }

    #[test]
    fn the_names_are_the_ones_the_protocol_uses() {
        // `.name()` is what a mismatch diagnostic reports, so it is worth pinning:
        // a wrong name here sends a reader looking for the wrong thing.
        assert_eq!(Suite::RsaPssSha256.name(), "rsa_pss_rsae_sha256");
        assert_eq!(Suite::EcdsaSecp256r1Sha256.name(), "ecdsa_secp256r1_sha256");
        assert_eq!(Suite::RsaPkcs1Sha256.name(), "sha256WithRSAEncryption");
        let mismatch = VerifyError::SchemeKeyMismatch {
            suite: Suite::EcdsaSecp256r1Sha256.name(),
            key: "an RSA key",
        };
        assert_eq!(
            mismatch.to_string(),
            "ecdsa_secp256r1_sha256 cannot be verified with an RSA key"
        );
    }
}
