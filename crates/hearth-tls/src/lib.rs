//! Hand-rolled TLS 1.3 primitives for the Hearth model downloader.
//!
//! **No dependencies, `std` only, by operator direction.** The downloader has to
//! reach `https://` on Linux, macOS and Windows on machines that may have no
//! `curl`, and the project takes no crypto crates, so the arithmetic lives
//! here. Nothing in this crate touches the network or the filesystem: it is
//! byte-in/byte-out arithmetic with published test vectors, which is the part
//! that can be proven correct in isolation.
//!
//! # Verification policy
//!
//! Every primitive is checked against its specification's own vectors
//! (FIPS 180-4, RFC 4231, RFC 5869, FIPS-197, SP 800-38D) rather than against
//! our own output, and wherever an independent implementation exists on the
//! development machine those vectors are re-derived from it before being
//! embedded (OpenSSL via the Python `cryptography` package for AES-GCM and
//! X25519). A constant copied out of memory is a guess wearing a test's
//! clothing; two independent implementations agreeing is evidence. When
//! the handshake lands, certificate verification must be **fatal and never
//! downgradeable**: a failed chain, a wrong signature or an unexpected
//! parameter must abort the connection. There must be no "insecure" switch
//! that a user or a download URL can reach.
//!
//! # What is here, and what is not yet
//!
//! Implemented and vector-tested:
//! - [`sha256`] — SHA-256, HMAC-SHA256 (the transcript hash and the HMAC base)
//! - [`hkdf`] — HKDF-Extract/Expand plus the RFC 8446 `HKDF-Expand-Label` and
//!   `Derive-Secret` helpers the key schedule is built from
//! - [`aes`] — AES-128 block cipher (forward direction only; GCM needs no
//!   inverse cipher), with the S-box derived at compile time instead of
//!   transcribed
//! - [`gcm`] — AES-128-GCM, the mandatory TLS 1.3 record cipher, with a
//!   branchless and table-free GHASH and a constant-time tag compare
//! - [`mod@x25519`] — X25519 key exchange, with the small-order-point check that
//!   RFC 7748 §6.1 requires, made unskippable by having no unchecked API for
//!   protocol code to reach for
//!
//! - [`record`] — the record layer: AEAD framing, the nonce derived from the
//!   sequence number, the encrypted header as additional data, and the TLS 1.3
//!   record-type-in-inner-plaintext trick. Pinned by RFC 8448's published
//!   protected record, byte for byte.
//!
//! On x86_64, AES-NI and PCLMULQDQ are used when the CPU reports them
//! (`crate::aesni`, `crate::ghash_x86`); the software implementations stay in
//! place as the reference the accelerated paths are tested against, and as what
//! runs everywhere else. `Aes128Gcm::accelerated()` reports which paths an
//! instance actually took.
//!
//! - [`keyschedule`] — the TLS 1.3 key schedule, pinned to RFC 8448's published
//!   early/handshake/master secrets and handshake traffic keys
//! - [`handshake`] — handshake message framing, the ClientHello we send, the
//!   ServerHello we accept (strictly, and without panicking on hostility), and
//!   the running transcript hash
//! - [`client`] — the 1-RTT client state machine: ServerHello through the
//!   `EncryptedExtensions`/`Certificate`/`CertificateVerify`/`Finished` flight,
//!   the fatal Finished MAC check, our own Finished, `KeyUpdate` in both
//!   directions and `close_notify`. It takes bytes in and hands bytes back, so
//!   it drives a transport the caller owns, and it cannot be built without a
//!   [`CertificateVerifier`]. The verifier is handed every input it needs — the
//!   chain, the signed data, the name we asked for and the caller's clock
//!   reading ([`UnixTime`]) — so a verification decision is a function of its
//!   arguments rather than of a clock the verifier reads itself, and the
//!   certificate slice's expiry tests cannot rot as the fixtures age.
//! - [`rsa`] — RSA signature verification: `rsa_pss_rsae_sha256` with MGF1-SHA-256
//!   and a 32-byte salt, over the same `signed_data` the state machine already
//!   rebuilds. Nothing calls it yet (the public key still has to come out of a
//!   certificate), which is why the only verifier the crate ships remains
//!   [`RejectAll`].
//! - [`der`] — reading DER (X.690) strictly: definite lengths only, minimal
//!   encodings only, no trailing bytes, no high-tag-number form. Every point where
//!   two readers could disagree about what a certificate says is a refusal here,
//!   and each refusal has a test.
//! - [`x509`] — a certificate's fields, read over [`der`]: the bytes a signature
//!   covers, both names as their own DER for chain linkage, the validity window as
//!   [`UnixTime`]s, the public key, and the extensions that constrain it. A
//!   critical extension it does not understand is a rejection, as RFC 5280 §4.2
//!   requires, and the `signatureAlgorithm` inside the signed part must be
//!   byte-identical to the one outside it.
//!
//! - [`p256`] — the P-256 curve: the field, the group, and the two scalar
//!   multiplications ECDSA needs. The constants come from OpenSSL's explicit
//!   parameters and are checked in-tree (`n * G` is the identity, the generator is
//!   on the curve), so a wrong constant fails the suite rather than a handshake.
//! - [`ecdsa`] — `ecdsa_secp256r1_sha256`, over [`der`] for the signature's
//!   encoding and [`p256`] for the equation. Like the RSA module it is a
//!   primitive that nothing calls yet.
//!
//! Still to come:
//! 1. **The certificate *decision*.** Both halves of signature verification now
//!    exist — RSA-PSS and ECDSA, each pinned to vectors from an independent
//!    implementation — and a certificate parses, so every ingredient is present.
//!    What does not exist is the code that *decides*: chain building to a pinned
//!    root, name matching, the validity window check against [`UnixTime`], a
//!    minimum key size, and the key-to-scheme matching that stops a P-256 key
//!    being verified with an RSA scheme. Its inputs are settled: the chain, the
//!    signature and the signed data come from the handshake
//!    ([`CertificateVerifyInput`]), the name from our own ClientHello, the instant
//!    from [`ClientConfig`], and the fields from [`x509`]. Until that exists the
//!    only verifier here is [`RejectAll`], which can only refuse, so no `https://`
//!    path may be enabled on this crate's account.
//! 2. **The transport** — sockets, and the read/write loop that drives it. That
//!    belongs in the downloader rather than here, where it would drag I/O into
//!    code that is provable in isolation.

/// True if `HEARTH_TLS_NO_HW` is set, which forces every primitive onto its
/// software path.
///
/// It exists so a benchmark can compare the two paths on one binary (this
/// repository's rule is to interleave before believing a delta) and so CI can
/// exercise the portable code on machines that have the instructions. It can
/// only ever select the *slower* implementation — it cannot weaken
/// verification, skip a check or accept a bad tag — and it is deliberately not
/// reachable from a URL or from the record layer.
pub(crate) fn hardware_disabled() -> bool {
    std::env::var_os("HEARTH_TLS_NO_HW").is_some()
}

pub mod aes;
pub mod chain;
pub mod client;
pub mod der;
pub mod ecdsa;
pub mod gcm;
pub mod handshake;
pub mod hkdf;
pub mod keyschedule;
pub mod leaf;
pub mod p256;
pub mod record;
pub mod rsa;
pub mod sha256;
pub mod verify;
pub mod x25519;
pub mod x509;

pub(crate) mod util;

#[cfg(target_arch = "x86_64")]
pub(crate) mod aesni;
#[cfg(target_arch = "x86_64")]
pub(crate) mod ghash_x86;

pub use aes::Aes128;
pub use chain::{verify_certificate_list, verify_chain, ChainError, Policy, Position};
pub use client::{
    CertificateVerifier, CertificateVerifyInput, ClientConfig, ClientError, ClientHandshake, Out,
    RejectAll, UnixTime,
};
pub use der::{oid_from_dotted, Class, Der, DerError, Element, Oid, Tag};
pub use gcm::{Aes128Gcm, GcmError, NONCE_LEN, TAG_LEN};
pub use handshake::{
    build_client_hello, parse_server_hello, ClientHelloParams, HandshakeError, HandshakeReader,
    Message, ServerHello, Transcript,
};
pub use hkdf::{derive_secret, derive_secret32, hkdf_expand, hkdf_expand_label, hkdf_extract};
pub use keyschedule::{finished_key, finished_verify_data, verify_finished, TrafficSecrets};
pub use leaf::{dns_name_matches, verify_leaf, verify_server_auth, verify_server_name, LeafError};
pub use record::{Record, RecordError, RecordLayer, TrafficKeys};
pub use rsa::{RsaError, RsaPublicKey};
pub use sha256::{hmac_sha256, hmac_sha256_parts, Sha256};
pub use verify::{verify_certificate_signature, verify_handshake_signature, Suite, VerifyError};
pub use x25519::{diffie_hellman, public_key, x25519, SharedSecretError, BASEPOINT};
pub use x509::{
    Certificate, GeneralName, KeyUsage, Name, PublicKey, SignatureAlgorithm, X509Error,
};
