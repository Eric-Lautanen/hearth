//! HKDF (RFC 5869) and the TLS 1.3 key-schedule helper (RFC 8446 §7.1).
//!
//! TLS 1.3 derives every secret through `HKDF-Expand-Label`, so this is the
//! single place the handshake's secret arithmetic lives. Checked against the
//! RFC 5869 vectors in `tests` below.

use crate::sha256::{hmac_sha256, hmac_sha256_parts};

/// SHA-256 output length, the `Hash.length` of the TLS 1.3 suite we use.
pub const HASH_LEN: usize = 32;

/// Largest `HKDF-Expand` output: 255 blocks of `Hash.length` (RFC 5869 §2.3).
pub const MAX_EXPAND: usize = 255 * HASH_LEN;

/// The prefix TLS puts in front of every label.
const LABEL_PREFIX: &str = "tls13 ";

/// HKDF-Extract: `PRK = HMAC-Hash(salt, IKM)`.
///
/// An absent salt is defined as `Hash.length` zero bytes (RFC 5869 §2.2); HMAC
/// with a zero-length key pads to the block size, which is the same thing, so
/// `&[]` and `&[0u8; 32]` are equivalent here.
pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; HASH_LEN] {
    hmac_sha256(salt, ikm)
}

/// HKDF-Expand (RFC 5869 §2.3): fills `out` from `prk` and `info`.
///
/// Errors only when `out` is longer than [`MAX_EXPAND`], which no real caller
/// can hit — it is a guard against a silent truncation, not a runtime case.
pub fn hkdf_expand(prk: &[u8], info: &[u8], out: &mut [u8]) -> Result<(), String> {
    if out.len() > MAX_EXPAND {
        return Err(format!(
            "hkdf_expand: {} bytes requested, max is {} (255 blocks)",
            out.len(),
            MAX_EXPAND
        ));
    }

    // T(1) = HMAC(prk, info || 1), T(i) = HMAC(prk, T(i-1) || info || i).
    let mut t = [0u8; HASH_LEN];
    let mut counter: u8 = 1;
    let mut written = 0usize;
    while written < out.len() {
        let info_len = info.len();
        let digest = if written == 0 {
            hmac_sha256_parts(prk, &[info, &[counter]])
        } else {
            hmac_sha256_parts(prk, &[&t[..], &info[..info_len], &[counter]])
        };
        t = digest;
        let take = (out.len() - written).min(HASH_LEN);
        out[written..written + take].copy_from_slice(&t[..take]);
        written += take;
        // Wrapping: at exactly MAX_EXPAND the loop exits on the next test, so
        // this overflow is unreachable, but a panic here would be a silent
        // correctness cliff in release builds.
        counter = counter.wrapping_add(1);
    }
    Ok(())
}

/// The RFC 8446 `HkdfLabel` structure, serialised.
///
/// ```text
/// struct {
///     uint16 length;
///     opaque label<7..255> = "tls13 " + Label;
///     opaque context<0..255> = Context;
/// } HkdfLabel;
/// ```
fn hkdf_label(label: &str, context: &[u8], length: u16) -> Result<Vec<u8>, String> {
    let full = LABEL_PREFIX.len() + label.len();
    if full > 255 {
        return Err(format!("hkdf label too long: {} bytes", full));
    }
    if context.len() > 255 {
        return Err(format!("hkdf context too long: {} bytes", context.len()));
    }
    let mut out = Vec::with_capacity(2 + 1 + full + 1 + context.len());
    out.extend_from_slice(&length.to_be_bytes());
    out.push(full as u8);
    out.extend_from_slice(LABEL_PREFIX.as_bytes());
    out.extend_from_slice(label.as_bytes());
    out.push(context.len() as u8);
    out.extend_from_slice(context);
    Ok(out)
}

/// `HKDF-Expand-Label(secret, label, context, length)` from RFC 8446 §7.1.
///
/// The length is taken from `out`, so the caller cannot ask for one size and
/// read another — a mismatch between the two is what makes a key schedule
/// silently diverge.
pub fn hkdf_expand_label(
    secret: &[u8],
    label: &str,
    context: &[u8],
    out: &mut [u8],
) -> Result<(), String> {
    if out.len() > u16::MAX as usize {
        return Err(format!(
            "hkdf_expand_label: {} bytes cannot be encoded",
            out.len()
        ));
    }
    let info = hkdf_label(label, context, out.len() as u16)?;
    hkdf_expand(secret, &info, out)
}

/// `Derive-Secret(secret, label, messages)` from RFC 8446 §7.1.
///
/// `messages` is the transcript hash, not the transcript itself.
pub fn derive_secret(
    secret: &[u8],
    label: &str,
    transcript_hash: &[u8],
    out: &mut [u8],
) -> Result<(), String> {
    hkdf_expand_label(secret, label, transcript_hash, out)
}

/// Convenience for the fixed-size secrets (`[u8; 32]`) the schedule passes
/// around.
pub fn derive_secret32(
    secret: &[u8],
    label: &str,
    transcript_hash: &[u8],
) -> Result<[u8; HASH_LEN], String> {
    let mut out = [0u8; HASH_LEN];
    derive_secret(secret, label, transcript_hash, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::{hmac_sha256, to_hex};

    fn unhex(s: &str) -> Vec<u8> {
        crate::sha256::from_hex(s).expect("test vector is valid hex")
    }

    /// RFC 5869 appendix A.1: basic test case.
    #[test]
    fn rfc5869_case1() {
        let ikm = unhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");
        let salt = unhex("000102030405060708090a0b0c");
        let info = unhex("f0f1f2f3f4f5f6f7f8f9");

        let prk = hkdf_extract(&salt, &ikm);
        assert_eq!(
            to_hex(&prk),
            "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
        );

        let mut okm = [0u8; 42];
        hkdf_expand(&prk, &info, &mut okm).unwrap();
        assert_eq!(
            to_hex(&okm),
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
        );
    }

    /// RFC 5869 appendix A.2: longer inputs, output longer than one block.
    #[test]
    fn rfc5869_case2() {
        let ikm: Vec<u8> = (0..80u8).collect();
        let salt: Vec<u8> = (0x60..0xb0u8).collect();
        let info: Vec<u8> = (0xb0..=0xffu8).collect();

        let prk = hkdf_extract(&salt, &ikm);
        assert_eq!(
            to_hex(&prk),
            "06a6b88c5853361a06104c9ceb35b45cef760014904671014a193f40c15fc244"
        );

        let mut okm = [0u8; 82];
        hkdf_expand(&prk, &info, &mut okm).unwrap();
        assert_eq!(
            to_hex(&okm),
            "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c\
             59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71\
             cc30c58179ec3e87c14c01d5c1f3434f1d87"
        );
    }

    /// RFC 5869 appendix A.3: zero-length salt and info.
    #[test]
    fn rfc5869_case3() {
        let ikm = unhex("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b");

        let prk = hkdf_extract(&[], &ikm);
        assert_eq!(
            to_hex(&prk),
            "19ef24a32c717b167f33a91d6f648bdf96596776afdb6377ac434c1c293ccb04"
        );

        // An absent salt is the same as 32 zero bytes (RFC 5869 §2.2).
        assert_eq!(prk, hkdf_extract(&[0u8; 32], &ikm));

        let mut okm = [0u8; 42];
        hkdf_expand(&prk, &[], &mut okm).unwrap();
        assert_eq!(
            to_hex(&okm),
            "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d9d201395faa4b61a96c8"
        );
    }

    /// The label framing must be exactly `length || "tls13 " + label ||
    /// context`, because getting it wrong still produces a self-consistent but
    /// wrong key schedule. Compared here against the structure written out by
    /// hand, and against plain HKDF-Expand over those bytes.
    #[test]
    fn expand_label_framing() {
        let secret = [0x11u8; 32];
        let context = unhex("dea5ee2a5b9b3b1abc1e7e7e8f6a4f3c");

        let mut got = [0u8; 32];
        hkdf_expand_label(&secret, "c hs traffic", &context, &mut got).unwrap();

        // struct HkdfLabel { uint16 length; opaque label<7..255>; opaque context<0..255>; }
        let mut expected_info = vec![0x00, 0x20]; // length = 32
        let label = "tls13 c hs traffic";
        expected_info.push(label.len() as u8);
        expected_info.extend_from_slice(label.as_bytes());
        expected_info.push(context.len() as u8);
        expected_info.extend_from_slice(&context);

        let mut expected = [0u8; 32];
        hkdf_expand(&secret, &expected_info, &mut expected).unwrap();
        assert_eq!(got, expected);

        // "tls13 " must be present and must have its space.
        assert!(expected_info.starts_with(&[0x00, 0x20, 0x12]));
        assert_eq!(&expected_info[3..9], b"tls13 ");

        // The length field is the *output* length, so a different output size
        // must produce a different label and therefore a different secret.
        let mut short = [0u8; 16];
        hkdf_expand_label(&secret, "c hs traffic", &context, &mut short).unwrap();
        assert_ne!(&got[..16], &short[..]);
    }

    /// Every input to the derivation must actually matter: a changed secret,
    /// label or context changes the output, and none of them is ignored.
    #[test]
    fn expand_label_inputs_all_matter() {
        let secret = [0x22u8; 32];
        let ctx = [0x33u8; 32];
        let mut base = [0u8; 32];
        hkdf_expand_label(&secret, "s ap traffic", &ctx, &mut base).unwrap();

        let mut other_secret = [0u8; 32];
        hkdf_expand_label(&[0x22u8; 31], "s ap traffic", &ctx, &mut other_secret).unwrap();
        assert_ne!(base, other_secret);

        let mut other_label = [0u8; 32];
        hkdf_expand_label(&secret, "c ap traffic", &ctx, &mut other_label).unwrap();
        assert_ne!(base, other_label);

        let mut other_ctx = [0u8; 32];
        hkdf_expand_label(&secret, "s ap traffic", &[], &mut other_ctx).unwrap();
        assert_ne!(base, other_ctx);

        // Repeatability.
        let mut again = [0u8; 32];
        hkdf_expand_label(&secret, "s ap traffic", &ctx, &mut again).unwrap();
        assert_eq!(base, again);
    }

    /// Derive-Secret is Expand-Label with the transcript hash as context.
    #[test]
    fn derive_secret_matches_expand_label() {
        let secret = [0x44u8; 32];
        let th = crate::sha256::Sha256::digest(b"transcript");

        let a = derive_secret32(&secret, "derived", &th).unwrap();
        let mut b = [0u8; 32];
        hkdf_expand_label(&secret, "derived", &th, &mut b).unwrap();
        assert_eq!(a, b);
    }

    /// An output longer than 255 blocks is refused rather than truncated.
    #[test]
    fn expand_refuses_too_much_output() {
        let prk = hkdf_extract(b"salt", b"ikm");
        let mut too_much = vec![0u8; MAX_EXPAND + 1];
        let err = hkdf_expand(&prk, b"info", &mut too_much).unwrap_err();
        assert!(err.contains("255 blocks"), "unexpected error: {err}");

        // Exactly the maximum is allowed and must fill every byte.
        let mut max = vec![0u8; MAX_EXPAND];
        hkdf_expand(&prk, b"info", &mut max).unwrap();
        assert_eq!(max.len(), 255 * 32);
        assert!(max.iter().any(|b| *b != 0));

        // A single-byte request still works (partial final block).
        let mut one = [0u8; 1];
        hkdf_expand(&prk, b"info", &mut one).unwrap();
        let mut block = [0u8; 32];
        hkdf_expand(&prk, b"info", &mut block).unwrap();
        assert_eq!(one[0], block[0]);
    }

    /// Empty info and a multi-block output must be the plain HMAC chain.
    #[test]
    fn expand_multi_block_matches_definition() {
        let prk = [0x77u8; 32];
        let mut out = [0u8; 70];
        hkdf_expand(&prk, b"info", &mut out).unwrap();

        let t1 = hmac_sha256(&prk, b"info\x01");
        let mut second = Vec::from(&t1[..]);
        second.extend_from_slice(b"info\x02");
        let t2 = hmac_sha256(&prk, &second);
        let mut third = Vec::from(&t2[..]);
        third.extend_from_slice(b"info\x03");
        let t3 = hmac_sha256(&prk, &third);

        assert_eq!(&out[..32], &t1[..]);
        assert_eq!(&out[32..64], &t2[..]);
        assert_eq!(&out[64..], &t3[..6]);
    }
}
