//! `ecdsa_secp256r1_sha256` verification, against signatures OpenSSL produced.
//!
//! The vectors in `data/ecdsa_vectors.rs` are not hand-written. Each one was
//! signed by OpenSSL over a key derived from a fixed scalar, then checked by an
//! independent verifier written in plain Python integers (no OpenSSL), then
//! accepted by OpenSSL itself — all in `scripts/tls_p256_vectors.py`, which is in
//! the repository so the claim is checkable rather than believed. What this file
//! does is drive those bytes through the real interface.
//!
//! The negatives come in two kinds, and they are checked differently on purpose.
//! A **cryptographic** negative — a flipped bit in `r`, a scalar out of range —
//! has to come back as `VerificationFailed` or `ScalarOutOfRange`, because the
//! encoding was fine and the equation was not. An **encoding** negative — a
//! redundant leading zero, a negative integer, trailing bytes — has to be refused
//! *as an encoding*, and each one is pinned to the rule that must fire rather than
//! to a boolean, because a boolean cannot tell "refused this signature" from
//! "refused this byte string". A redundant leading zero is the case that makes the
//! distinction concrete: it is legal BER and names the same integer, so a lenient
//! reader accepts a signature DER does not have. (Measured, rather than assumed:
//! the generator reports any negative a lenient reader accepts, and in this run
//! OpenSSL's decoder refused every one of them too — so what these pin is which
//! rule fires, not a disagreement in outcomes.)
//!
//! The vectors are a **pin**, not something that reproduces byte for byte: ECDSA
//! signatures use a random nonce, so regenerating the file produces different
//! signature bytes and sometimes a different length, which is why the shape checks
//! below describe the set that is in the tree. Regenerating is safe — the
//! expectations come from the signatures rather than the other way round — but the
//! shape assertions would need re-deriving, and a generation with no 70-byte
//! signature would fail the second one as written.

use hearth_tls::der::DerError;
use hearth_tls::ecdsa::{verify_ecdsa_secp256r1_sha256, EcdsaError};
use hearth_tls::sha256::from_hex;

include!("data/ecdsa_vectors.rs");

fn hex(text: &str) -> Vec<u8> {
    from_hex(text).unwrap_or_else(|| panic!("not hex: {text}"))
}

fn verify(message: &str, signature: &str) -> Result<(), EcdsaError> {
    verify_ecdsa_secp256r1_sha256(&hex(PUBLIC_KEY_SEC1), &hex(message), &hex(signature))
}

#[test]
fn every_openssl_vector_verifies() {
    let key = hex(PUBLIC_KEY_SEC1);
    assert_eq!(key.len(), 65, "the key is one SEC 1 uncompressed point");
    assert_eq!(key[0], 0x04);
    assert_eq!(VECTORS.len(), 6);
    for (index, (message, signature)) in VECTORS.iter().enumerate() {
        verify_ecdsa_secp256r1_sha256(&key, &hex(message), &hex(signature))
            .unwrap_or_else(|error| panic!("vector {index} was refused: {error}"));
    }
}

#[test]
fn the_vectors_cover_the_shapes_that_matter() {
    // The messages include the empty one, a single zero byte, and a thousand
    // bytes, so the hash is exercised below, at and above one block.
    assert_eq!(VECTORS[0].0, "", "the first vector signs the empty message");
    assert_eq!(VECTORS[1].0, "00");
    assert_eq!(VECTORS[5].0.len() / 2, 1000);
    // One vector is exactly the RFC 8446 §4.4.3 construction a TLS server signs:
    // 64 spaces, the context string, a zero byte, then the transcript hash. That
    // is the message this code will really be handed.
    let signed_data = hex(VECTORS[4].0);
    assert_eq!(signed_data.len(), 64 + 33 + 1 + 32);
    assert_eq!(&signed_data[..64], &[b' '; 64]);
    assert_eq!(&signed_data[64..97], b"TLS 1.3, server CertificateVerify");
    assert_eq!(signed_data[97], 0);
    // And the integer encodings differ across vectors: some `r` or `s` needs the
    // leading zero byte DER requires for a high bit, some does not, so both paths
    // through the integer reader are covered by real signatures rather than by
    // constructed ones.
    let lengths: Vec<usize> = VECTORS.iter().map(|(_, s)| hex(s).len()).collect();
    assert!(
        lengths.contains(&70) && lengths.iter().any(|&len| len >= 71),
        "signature lengths were {lengths:?}"
    );
}

#[test]
fn a_signature_does_not_verify_under_a_key_that_did_not_sign_it() {
    let message = MESSAGE;
    let signature = VECTORS[0].1;
    // The control first: it verifies under the right key, so the failure below is
    // about the key and not about the vector.
    verify(message, signature).expect("the vector verifies under its own key");
    assert_eq!(
        verify_ecdsa_secp256r1_sha256(&hex(OTHER_PUBLIC_KEY_SEC1), &hex(message), &hex(signature)),
        Err(EcdsaError::VerificationFailed)
    );
}

#[test]
fn a_changed_message_does_not_verify() {
    let signature = hex(VECTORS[0].1);
    let key = hex(PUBLIC_KEY_SEC1);
    // The first vector signs the empty message, so a single byte is a change.
    for message in [vec![0u8], vec![b' '], vec![0u8; 2]] {
        assert_eq!(
            verify_ecdsa_secp256r1_sha256(&key, &message, &signature),
            Err(EcdsaError::VerificationFailed),
            "message {message:?}"
        );
    }
}

#[test]
fn the_cryptographic_negatives_fail_for_the_right_reason() {
    for (name, signature) in NEGATIVES {
        let expected = match *name {
            "TAMPERED_R" | "TAMPERED_S" => EcdsaError::VerificationFailed,
            "R_ZERO" => EcdsaError::ScalarOutOfRange("r"),
            "S_ZERO" | "S_EQUALS_N" | "S_ABOVE_N" => EcdsaError::ScalarOutOfRange("s"),
            _ => continue,
        };
        assert_eq!(
            verify(MESSAGE, signature),
            Err(expected),
            "{name} should have been refused"
        );
    }
}

#[test]
fn the_encoding_negatives_are_refused_as_encodings() {
    for (name, signature) in NEGATIVES {
        let error = verify(MESSAGE, signature).expect_err("an encoding negative must not verify");
        let refused_as_expected = match *name {
            "R_NEGATIVE" => matches!(error, EcdsaError::Signature(DerError::Invalid(_))),
            "R_REDUNDANT_ZERO" => {
                matches!(error, EcdsaError::Signature(DerError::NonMinimalLength(_)))
            }
            "TRAILING_BYTE" | "THREE_INTEGERS" => {
                matches!(error, EcdsaError::Signature(DerError::TrailingData(_)))
            }
            "SET_NOT_SEQUENCE" | "NOT_A_SEQUENCE" => {
                matches!(error, EcdsaError::Signature(DerError::UnexpectedTag { .. }))
            }
            "TRUNCATED" => matches!(error, EcdsaError::Signature(DerError::Truncated(_))),
            "EMPTY" => matches!(error, EcdsaError::Signature(DerError::Missing(_))),
            // The raw `r || s` form: whatever the first byte happens to look like,
            // it is not a DER signature and must not be read as one.
            "RAW_RS" => matches!(error, EcdsaError::Signature(_)),
            _ => continue,
        };
        assert!(refused_as_expected, "{name} produced {error:?}");
    }
}

#[test]
fn a_high_s_is_accepted_because_nothing_requires_low_s() {
    // `(r, n - s)` is the other valid signature over the same message. Rejecting
    // it is a Bitcoin consensus rule; X9.62 and TLS 1.3 do not have one, so a
    // server that produced it would be within its rights and this verifier must
    // not invent a reason to refuse it.
    verify(MESSAGE, S_NEGATED).expect("the negated s verifies");
}

/// Times one verification of the TLS-shaped vector, which is what a connection
/// pays per certificate.
///
/// Ignored by default so the suite stays quick, and here rather than in `src`
/// because the vector it measures lives in `data/`. It exists because a number in
/// a document should have been measured:
/// `cargo test --release -p hearth-tls --test ecdsa_vectors -- --ignored --nocapture`.
#[test]
#[ignore = "timing measurement"]
fn measure_a_verification() {
    let key = hex(PUBLIC_KEY_SEC1);
    let message = hex(VECTORS[4].0);
    let signature = hex(VECTORS[4].1);
    // Warm the caches the way a first TLS handshake would not, then time the rest.
    verify_ecdsa_secp256r1_sha256(&key, &message, &signature).expect("verifies");
    let rounds = 200u32;
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        verify_ecdsa_secp256r1_sha256(&key, &message, &signature).expect("verifies");
    }
    println!(
        "ecdsa_secp256r1_sha256: {:?} per verification ({rounds} rounds)",
        start.elapsed() / rounds
    );
}

#[test]
fn an_unusable_key_is_refused_before_the_signature_is_looked_at() {
    // The generator's key with one bit of y flipped: the signature is a real one,
    // so the only thing that can catch this is the key parser.
    let mut key = hex(PUBLIC_KEY_SEC1);
    let last = key.len() - 1;
    key[last] ^= 1;
    assert_eq!(
        verify_ecdsa_secp256r1_sha256(&key, &hex(MESSAGE), &hex(VECTORS[0].1)),
        Err(EcdsaError::PublicKey(
            hearth_tls::p256::P256Error::NotOnCurve
        ))
    );
}
