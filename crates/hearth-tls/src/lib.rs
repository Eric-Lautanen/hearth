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
//! (FIPS 180-4, RFC 4231, RFC 5869) rather than against our own output. When
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
//!
//! Still to come, in the order the handshake needs them:
//! 1. **AES-128-GCM** — the TLS 1.3 record cipher (`AES-128-GCM` is the
//!    mandatory suite; `AES-256-GCM` and ChaCha20-Poly1305 follow).
//! 2. **X25519** — key exchange, RFC 7748 vectors.
//! 3. **The record layer** — AEAD framing, nonce construction from the
//!    sequence number, and the TLS 1.3 record-type-in-inner-plaintext trick.
//! 4. **The handshake** — ClientHello/ServerHello, key schedule, transcript
//!    hash, Finished MAC verification (a wrong MAC must abort, not warn).
//! 5. **Certificate verification** — the largest and most dangerous piece:
//!    X.509 chain building, ECDSA/RSA signature checks and a pinned root
//!    store. Deliberately last, and deliberately not attempted until the
//!    transport below it is solid.

pub mod hkdf;
pub mod sha256;

pub use hkdf::{derive_secret, derive_secret32, hkdf_expand, hkdf_expand_label, hkdf_extract};
pub use sha256::{hmac_sha256, hmac_sha256_parts, Sha256};
