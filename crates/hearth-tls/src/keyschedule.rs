//! The TLS 1.3 key schedule (RFC 8446 §7.1).
//!
//! ```text
//!              0
//!              |
//!              v
//!    PSK ->  HKDF-Extract  = Early Secret
//!              |
//!              +---- Derive-Secret(., "derived", "") -> salt
//!              v
//! (EC)DHE -> HKDF-Extract  = Handshake Secret
//!              |
//!   Derive-Secret(., "c hs traffic", ClientHello...ServerHello) -> client HS secret
//!   Derive-Secret(., "s hs traffic", ClientHello...ServerHello) -> server HS secret
//!              |
//!              +---- Derive-Secret(., "derived", "") -> salt
//!              v
//!        0 -> HKDF-Extract  = Master Secret
//!              |
//!   Derive-Secret(., "c ap traffic", ClientHello...server Finished) -> client AP secret
//!   Derive-Secret(., "s ap traffic", ClientHello...server Finished) -> server AP secret
//! ```
//!
//! Two details that are easy to get wrong and are pinned by published vectors in
//! `tests`: the two `Derive-Secret(., "derived", "")` steps hash the **empty
//! string** (not the empty transcript, and not the current transcript), and the
//! traffic secrets are keyed by the transcript hash *up to that point in the
//! handshake*, so the same handshake secret yields different secrets at
//! different moments.
//!
//! There is no PSK in a 1-RTT handshake without resumption, which means the
//! Early Secret is `HKDF-Extract(0, 0)` — a constant. It is computed anyway
//! rather than hardcoded, because a resumption slice will need the real thing.
//!
//! The `resumption_master_secret` and the exporter keys are not derived here:
//! nothing consumes them yet, and an unused derivation is an untested one.

use crate::hkdf::{hkdf_expand, hkdf_expand_label, hkdf_extract, HASH_LEN};
use crate::sha256::{hmac_sha256, Sha256};
use crate::util::constant_time_eq;

/// Length of the Finished MAC, equal to `Hash.length` for SHA-256.
pub const FINISHED_LEN: usize = HASH_LEN;

/// The outputs of the schedule that a 1-RTT handshake needs.
#[derive(Clone)]
pub struct TrafficSecrets {
    /// `c hs traffic` or `c ap traffic`: the client writes with this.
    pub client: [u8; HASH_LEN],
    /// `s hs traffic` or `s ap traffic`: the server writes with this.
    pub server: [u8; HASH_LEN],
}

impl TrafficSecrets {
    /// Derives both directions from one schedule node and one transcript hash.
    ///
    /// `prefix` is `"hs"` for the handshake secrets or `"ap"` for the
    /// application secrets; the labels are built from it, so the two call sites
    /// cannot drift in spelling.
    pub fn derive(
        schedule_node: &[u8; HASH_LEN],
        prefix: &str,
        transcript_hash: &[u8; HASH_LEN],
    ) -> Result<Self, String> {
        let client_label = format!("c {prefix} traffic");
        let server_label = format!("s {prefix} traffic");
        Ok(Self {
            client: derive_secret(schedule_node, &client_label, transcript_hash)?,
            server: derive_secret(schedule_node, &server_label, transcript_hash)?,
        })
    }
}

/// `Hash("")` — the transcript hash before any message, used by the two
/// `Derive-Secret(., "derived", "")` steps.
pub fn empty_hash() -> [u8; HASH_LEN] {
    Sha256::new().finalize()
}

/// `HKDF-Extract(salt, IKM)`.
///
/// The salt is always a previous secret here; RFC 5869's "no salt" case is
/// `HASH_LEN` zero bytes, which is what the early secret uses.
pub fn extract(salt: &[u8], ikm: &[u8]) -> [u8; HASH_LEN] {
    hkdf_extract(salt, ikm)
}

/// `Derive-Secret(secret, label, transcript_hash)` (a thin wrapper so callers
/// read like the RFC).
pub fn derive_secret(
    secret: &[u8; HASH_LEN],
    label: &str,
    transcript_hash: &[u8; HASH_LEN],
) -> Result<[u8; HASH_LEN], String> {
    let mut out = [0u8; HASH_LEN];
    hkdf_expand_label(secret, label, transcript_hash, &mut out)?;
    Ok(out)
}

/// The Early Secret, from a pre-shared key or from nothing.
///
/// Without a PSK the input keying material is `HASH_LEN` zero bytes, so this is
/// `HKDF-Extract(0, 0)` — the constant RFC 8448 prints as
/// `33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a`.
pub fn early_secret(psk: Option<&[u8]>) -> [u8; HASH_LEN] {
    let zeros = [0u8; HASH_LEN];
    extract(&zeros, psk.unwrap_or(&zeros))
}

/// `Derive-Secret(early, "derived", "")` — the salt for the handshake extract.
pub fn derived_from_early(early: &[u8; HASH_LEN]) -> Result<[u8; HASH_LEN], String> {
    derive_secret(early, "derived", &empty_hash())
}

/// The Handshake Secret: `HKDF-Extract(Derive-Secret(early, "derived", ""), (EC)DHE)`.
///
/// `ecdhe` is the X25519 shared secret, which is not hashed before use — it is
/// the extract's input keying material exactly as X25519 outputs it.
pub fn handshake_secret(early: &[u8; HASH_LEN], ecdhe: &[u8]) -> Result<[u8; HASH_LEN], String> {
    Ok(extract(&derived_from_early(early)?, ecdhe))
}

/// The Master Secret (with no PSK): `HKDF-Extract(Derive-Secret(hs, "derived", ""), 0)`.
pub fn master_secret(handshake: &[u8; HASH_LEN]) -> Result<[u8; HASH_LEN], String> {
    let derived = derive_secret(handshake, "derived", &empty_hash())?;
    let zeros = [0u8; HASH_LEN];
    Ok(extract(&derived, &zeros))
}

/// `finished_key = HKDF-Expand-Label(traffic_secret, "finished", "", Hash.length)`.
pub fn finished_key(traffic_secret: &[u8; HASH_LEN]) -> Result<[u8; HASH_LEN], String> {
    let mut out = [0u8; HASH_LEN];
    hkdf_expand_label(traffic_secret, "finished", &[], &mut out)?;
    Ok(out)
}

/// The Finished `verify_data`: `HMAC(finished_key, transcript_hash)`.
///
/// The input is the transcript hash, not the transcript — RFC 8446 §4.4.4 is
/// explicit, and using the raw transcript instead is a classic mistake that
/// still "works" between two implementations of the same bug.
pub fn finished_verify_data(
    finished_key: &[u8; HASH_LEN],
    transcript_hash: &[u8; HASH_LEN],
) -> [u8; FINISHED_LEN] {
    hmac_sha256(finished_key, transcript_hash)
}

/// Checks a received Finished MAC.
///
/// Returns `Ok(())` only on an exact match, compared in constant time. A
/// mismatch is the one point in the handshake where the peer has proven it does
/// not hold the keys, so there is no "close enough" branch and no way to
/// continue: a caller that ignores an `Err` here has no session, only bytes.
pub fn verify_finished(
    finished_key: &[u8; HASH_LEN],
    transcript_hash: &[u8; HASH_LEN],
    received: &[u8],
) -> Result<(), String> {
    let expected = finished_verify_data(finished_key, transcript_hash);
    if constant_time_eq(&expected, received) {
        Ok(())
    } else {
        Err("Finished MAC does not match: the peer does not hold the handshake keys".to_string())
    }
}

/// Rekeys a traffic secret to the next generation (RFC 8446 §7.2):
/// `new = Derive-Secret(current, "traffic upd", "")`.
///
/// The context is the empty string, which is what makes `KeyUpdate` cheap: no
/// transcript is involved, so the new secret depends only on the old one.
pub fn update_traffic_secret(current: &[u8; HASH_LEN]) -> Result<[u8; HASH_LEN], String> {
    derive_secret(current, "traffic upd", &empty_hash())
}

/// `HKDF-Expand` re-exported for callers that need a raw expansion (the
/// exporter interface will); kept here so the schedule module owns every use of
/// the KDF.
pub fn expand(prk: &[u8], info: &[u8], out: &mut [u8]) -> Result<(), String> {
    hkdf_expand(prk, info, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x25519;

    /// Hex to bytes, for the 12- and 16-byte values as well as the 32-byte ones.
    fn hx(s: &str) -> Vec<u8> {
        let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(cleaned.len().is_multiple_of(2), "odd hex length");
        (0..cleaned.len() / 2)
            .map(|i| u8::from_str_radix(&cleaned[i * 2..i * 2 + 2], 16).expect("hex"))
            .collect()
    }

    fn h(s: &str) -> [u8; 32] {
        hx(s).try_into().expect("a 32-byte value")
    }

    /// RFC 8448 §3, the Simple 1-RTT Handshake. Every constant below was
    /// re-derived locally from the RFC's inputs *before* being embedded — the
    /// chain from the early secret to the write keys reproduces from
    /// `HKDF-Extract`/`HKDF-Expand-Label` alone, and the (EC)DHE input is itself
    /// checked in `published_ecdh_matches_our_x25519`.
    const ECDHE: &str = "8bd4054fb55b9d63fdfbacf9f04b9f0d35e6d63f537563efd46272900f89492d";
    /// `Hash(ClientHello...ServerHello)` from the same trace.
    const TRANSCRIPT_AFTER_SERVER_HELLO: &str =
        "860c06edc07858ee8e78f0e7428c58edd6b43f2ca3e6e95f02ed063cf0e1cad8";

    #[test]
    fn early_secret_matches_rfc8448() {
        let early = early_secret(None);
        assert_eq!(
            early,
            h("33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a")
        );
        assert_eq!(
            derived_from_early(&early).unwrap(),
            h("6f2615a108c702c5678f54fc9dbab69716c076189c48250cebeac3576c3611ba")
        );
    }

    #[test]
    fn handshake_and_master_secrets_match_rfc8448() {
        let early = early_secret(None);
        let hs = handshake_secret(&early, &h(ECDHE)).unwrap();
        assert_eq!(
            hs,
            h("1dc826e93606aa6fdc0aadc12f741b01046aa6b99f691ed221a9f0ca043fbeac")
        );
        assert_eq!(
            derive_secret(&hs, "derived", &empty_hash()).unwrap(),
            h("43de77e0c77713859a944db9db2590b53190a65b3ee2e4f12dd7a0bb7ce254b4")
        );
        assert_eq!(
            master_secret(&hs).unwrap(),
            h("18df06843d13a08bf2a449844c5f8a478001bc4d4c627984d5a41da8d0402919")
        );
    }

    #[test]
    fn handshake_traffic_secrets_match_rfc8448() {
        let early = early_secret(None);
        let hs = handshake_secret(&early, &h(ECDHE)).unwrap();
        let secrets = TrafficSecrets::derive(&hs, "hs", &h(TRANSCRIPT_AFTER_SERVER_HELLO)).unwrap();
        assert_eq!(
            secrets.client,
            h("b3eddb126e067f35a780b3abf45e2d8f3b1a950738f52e9600746a0e27a55a21")
        );
        assert_eq!(
            secrets.server,
            h("b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38")
        );

        // The write keys and IVs are the RFC's "key expanded"/"iv expanded".
        let server_keys = crate::record::TrafficKeys::from_secret(&secrets.server).unwrap();
        let client_keys = crate::record::TrafficKeys::from_secret(&secrets.client).unwrap();
        assert_eq!(
            server_keys.key()[..],
            hx("3fce516009c21727d0f2e4e86ee403bc")[..]
        );
        assert_eq!(server_keys.iv()[..], hx("5d313eb2671276ee13000b30")[..]);
        assert_eq!(
            client_keys.key()[..],
            hx("dbfaa693d1762c5b666af5d950258d01")[..]
        );
        assert_eq!(client_keys.iv()[..], hx("5bd3c71b836e0b76bb73265f")[..]);
    }

    #[test]
    fn finished_keys_match_rfc8448() {
        let early = early_secret(None);
        let hs = handshake_secret(&early, &h(ECDHE)).unwrap();
        let secrets = TrafficSecrets::derive(&hs, "hs", &h(TRANSCRIPT_AFTER_SERVER_HELLO)).unwrap();
        assert_eq!(
            finished_key(&secrets.server).unwrap(),
            h("008d3b66f816ea559f96b537e885c31fc068bf492c652f01f288a1d8cdc19fc8")
        );
        assert_eq!(
            finished_key(&secrets.client).unwrap(),
            h("b80ad01015fb2f0bd65ff7d4da5d6bf83f84821d1f87fdc7d3c75b5a7b42d9c4")
        );
    }

    /// The three published values must be consistent with each other under our
    /// own X25519: the RFC's client private key times its server public key must
    /// equal the RFC's (EC)DHE shared secret. This is a cross-check of values
    /// against an implementation that shares no code with the RFC's arithmetic.
    #[test]
    fn published_ecdh_matches_our_x25519() {
        let client_private = h("49af42ba7f7994852d713ef2784bcbcaa7911de26adc5642cb634540e7ea5005");
        let server_public = h("c9828876112095fe66762bdbf7c672e156d6cc253b833df1dd69b1b04e751f0f");
        let shared = x25519::diffie_hellman(&client_private, &server_public).expect("x25519");
        assert_eq!(shared, h(ECDHE));
    }

    /// The Finished computation, and that a single wrong bit anywhere in either
    /// input is rejected.
    #[test]
    fn finished_mac_verification_is_exact_and_fatal() {
        let keys = h("008d3b66f816ea559f96b537e885c31fc068bf492c652f01f288a1d8cdc19fc8");
        let transcript = h(TRANSCRIPT_AFTER_SERVER_HELLO);
        let mac = finished_verify_data(&keys, &transcript);

        assert_eq!(verify_finished(&keys, &transcript, &mac), Ok(()));

        // Wrong length, empty, truncated and each single-bit flip must fail.
        assert!(verify_finished(&keys, &transcript, &[]).is_err());
        assert!(verify_finished(&keys, &transcript, &mac[..31]).is_err());
        let mut long = mac.to_vec();
        long.push(0);
        assert!(verify_finished(&keys, &transcript, &long).is_err());
        for byte in 0..mac.len() {
            for bit in 0..8 {
                let mut bad = mac;
                bad[byte] ^= 1 << bit;
                assert!(
                    verify_finished(&keys, &transcript, &bad).is_err(),
                    "MAC byte {byte} bit {bit} was accepted"
                );
            }
        }
        // A wrong transcript hash is a different MAC.
        let mut wrong_transcript = transcript;
        wrong_transcript[0] ^= 1;
        assert!(verify_finished(
            &keys,
            &transcript,
            &finished_verify_data(&keys, &wrong_transcript)
        )
        .is_err());
    }

    /// The two `"derived"` steps use `Hash("")`, not the running transcript.
    #[test]
    fn derived_steps_hash_the_empty_string() {
        let empty = empty_hash();
        assert_eq!(
            empty,
            h("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        );
        // Using a non-empty context must change the result, which is what makes
        // the previous test meaningful.
        let early = early_secret(None);
        let with_empty = derive_secret(&early, "derived", &empty).unwrap();
        let with_other = derive_secret(&early, "derived", &h(ECDHE)).unwrap();
        assert_ne!(with_empty, with_other);
    }

    /// A PSK must change the early secret: without this, a resumption
    /// implementation could ignore its PSK and nothing would fail.
    #[test]
    fn psk_changes_the_early_secret() {
        let no_psk = early_secret(None);
        let zeros = [0u8; HASH_LEN];
        assert_eq!(no_psk, early_secret(Some(&zeros)), "zero PSK equals no PSK");
        let psk = h(ECDHE);
        assert_ne!(early_secret(Some(&psk)), no_psk);
    }

    /// Key update must be deterministic, depend only on the current secret, and
    /// never produce the same secret twice in a row.
    #[test]
    fn key_update_is_deterministic_and_changes_the_secret() {
        let secret = h("b67b7d690cc16c4e75e54213cb2d37b4e9c912bcded9105d42befd59d391ad38");
        let next = update_traffic_secret(&secret).unwrap();
        assert_eq!(update_traffic_secret(&secret).unwrap(), next);
        assert_ne!(next, secret);

        let next2 = update_traffic_secret(&next).unwrap();
        assert_ne!(next2, next);
        assert_ne!(next2, secret);
    }
}
