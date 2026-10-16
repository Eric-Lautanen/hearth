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
//! - [`x25519`] — X25519 key exchange, with the small-order-point check that
//!   RFC 7748 §6.1 requires, made unskippable by having no unchecked API for
//!   protocol code to reach for
//!
//! On x86_64, AES-NI and PCLMULQDQ are used when the CPU reports them
//! (`crate::aesni`, `crate::ghash_x86`); the software implementations stay in
//! place as the reference the accelerated paths are tested against, and as what
//! runs everywhere else. `Aes128Gcm::accelerated()` reports which paths an
//! instance actually took.
//!
//! Still to come, in the order the handshake needs them:
//! 1. **The record layer** — AEAD framing, nonce construction from the
//!    sequence number, and the TLS 1.3 record-type-in-inner-plaintext trick.
//! 4. **The handshake** — ClientHello/ServerHello, key schedule, transcript
//!    hash, Finished MAC verification (a wrong MAC must abort, not warn).
//! 5. **Certificate verification** — the largest and most dangerous piece:
//!    X.509 chain building, ECDSA/RSA signature checks and a pinned root
//!    store. Deliberately last, and deliberately not attempted until the
//!    transport below it is solid.

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
pub mod gcm;
pub mod hkdf;
pub mod sha256;
pub mod x25519;

#[cfg(target_arch = "x86_64")]
pub(crate) mod aesni;
#[cfg(target_arch = "x86_64")]
pub(crate) mod ghash_x86;

pub use aes::Aes128;
pub use gcm::{Aes128Gcm, GcmError, NONCE_LEN, TAG_LEN};
pub use hkdf::{derive_secret, derive_secret32, hkdf_expand, hkdf_expand_label, hkdf_extract};
pub use sha256::{hmac_sha256, hmac_sha256_parts, Sha256};
pub use x25519::{diffie_hellman, public_key, x25519, SharedSecretError, BASEPOINT};
