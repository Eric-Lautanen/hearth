//! AES-128-GCM (NIST SP 800-38D) — the mandatory TLS 1.3 record cipher.
//!
//! Structure is the standard one: `GCTR` for confidentiality, `GHASH` for
//! integrity, and the tag as `E(K, J0) xor GHASH(...)`. Only a 96-bit nonce is
//! accepted, because that is the one TLS 1.3 uses and the one whose `J0` is
//! simply `nonce || 0x00000001`; the general (non-96-bit) `J0` construction is
//! dead code here and would only be a place to get it wrong.
//!
//! # Constant time
//!
//! `GHASH`'s carry-less multiply is done as a bit loop with masks and no
//! lookups: no table means no cache-timing side channel, and no
//! data-dependent branch means no branch-prediction one. The tag comparison is
//! a full-length XOR/OR accumulate rather than `==`, so a wrong tag cannot be
//! located byte by byte by timing.
//!
//! The cost is speed: 128 iterations per 16-byte block. That is deliberately
//! accepted in this slice — a correct baseline first — with hardware
//! acceleration (PCLMULQDQ for GHASH, AES-NI for the block function,
//! runtime-dispatched like `hearth-quant`'s kernels) as its own measured slice.
//! See `docs/BUG_TRACKER.md` for the software throughput figure.
//!
//! # Nonce uniqueness
//!
//! Reusing a `(key, nonce)` pair under GCM is catastrophic: it reveals the XOR
//! of two plaintexts and, worse, allows forging tags. This module cannot detect
//! reuse; the record layer must derive the nonce from the sequence number, as
//! RFC 8446 §5.3 requires.

use crate::aes::Aes128;
use std::fmt;

/// Authentication tag length (SP 800-38D §5.2.1.2): the full 128 bits, never a
/// truncated tag, because TLS 1.3 always uses the full tag.
pub const TAG_LEN: usize = 16;

/// Nonce length; TLS 1.3 derives a 12-byte nonce from the record sequence.
pub const NONCE_LEN: usize = 12;

/// Why authentication failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcmError {
    /// The tag did not match. The ciphertext must be treated as forged: no
    /// partial plaintext is returned, and the caller must not retry with a
    /// truncated tag.
    TagMismatch,
}

impl fmt::Display for GcmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TagMismatch => write!(f, "GCM authentication tag mismatch"),
        }
    }
}

impl std::error::Error for GcmError {}

/// AES-128-GCM with a precomputed key schedule and hash subkey.
#[derive(Clone)]
pub struct Aes128Gcm {
    aes: Aes128,
    /// `H = E(K, 0^128)`, the GHASH subkey (SP 800-38D §6.3).
    h: u128,
    /// Whether the PCLMULQDQ carry-less multiply is available on this CPU.
    ///
    /// Set once here from the runtime probe so the hot loop does not branch on
    /// a CPU feature per block, and so this field is the single justification
    /// for the `unsafe` call in [`Self::ghash`].
    pclmul: bool,
}

impl Aes128Gcm {
    /// Builds the cipher and derives the hash subkey.
    pub fn new(key: &[u8; 16]) -> Self {
        let aes = Aes128::new(key);
        let mut zero_block = [0u8; 16];
        aes.encrypt_block(&mut zero_block);
        Self {
            aes,
            h: u128::from_be_bytes(zero_block),
            pclmul: pclmul_available(),
        }
    }

    /// Which implementations this instance ended up using: `(aes_ni, pclmul)`.
    ///
    /// Exposed for the same reason `hearth-quant` reports its selected kernel —
    /// a dispatch that cannot be observed is a dispatch that cannot be checked.
    /// Expect `(false, false)` on non-x86_64 targets and on x86_64 CPUs older
    /// than Westmere.
    pub fn accelerated(&self) -> (bool, bool) {
        (self.aes.uses_hardware(), self.pclmul)
    }

    /// Encrypts and authenticates, returning `(ciphertext, tag)`.
    ///
    /// The ciphertext is the same length as the plaintext; the tag is always
    /// [`TAG_LEN`] bytes and is returned separately because TLS 1.3 carries it
    /// in the record's trailer.
    pub fn seal(
        &self,
        nonce: &[u8; NONCE_LEN],
        aad: &[u8],
        plaintext: &[u8],
    ) -> (Vec<u8>, [u8; TAG_LEN]) {
        let j0 = j0_from_nonce(nonce);
        let mut ciphertext = vec![0u8; plaintext.len()];
        self.gctr(inc32(j0), plaintext, &mut ciphertext);
        let tag = self.tag(j0, self.ghash(aad, &ciphertext));
        (ciphertext, tag)
    }

    /// Verifies the tag and, only if it matches, decrypts.
    ///
    /// Verification happens first and its failure is total: a forged record
    /// yields [`GcmError::TagMismatch`] and no plaintext at all. There is no
    /// API here that returns unauthenticated data, by design.
    pub fn open(
        &self,
        nonce: &[u8; NONCE_LEN],
        aad: &[u8],
        ciphertext: &[u8],
        tag: &[u8; TAG_LEN],
    ) -> Result<Vec<u8>, GcmError> {
        let j0 = j0_from_nonce(nonce);
        let expected = self.tag(j0, self.ghash(aad, ciphertext));
        if !constant_time_eq(&expected, tag) {
            return Err(GcmError::TagMismatch);
        }
        let mut plaintext = vec![0u8; ciphertext.len()];
        self.gctr(inc32(j0), ciphertext, &mut plaintext);
        Ok(plaintext)
    }

    /// The tag for an already-computed ciphertext. Exposed because the record
    /// layer will want to authenticate in place rather than allocate twice.
    pub fn tag_of(&self, nonce: &[u8; NONCE_LEN], aad: &[u8], ciphertext: &[u8]) -> [u8; TAG_LEN] {
        self.tag(j0_from_nonce(nonce), self.ghash(aad, ciphertext))
    }

    /// `GCTR` with an arbitrary starting counter (SP 800-38D §6.5).
    fn gctr(&self, counter: u128, input: &[u8], out: &mut [u8]) {
        for (i, (chunk, out_chunk)) in input.chunks(16).zip(out.chunks_mut(16)).enumerate() {
            let mut keystream = inc32_n(counter, i as u32).to_be_bytes();
            self.aes.encrypt_block(&mut keystream);
            for (j, (src, dst)) in chunk.iter().zip(out_chunk.iter_mut()).enumerate() {
                *dst = *src ^ keystream[j];
            }
        }
    }

    /// `GHASH` over `aad || pad || ciphertext || pad || len(aad) || len(ct)`
    /// (SP 800-38D §6.4).
    fn ghash(&self, aad: &[u8], ciphertext: &[u8]) -> u128 {
        if self.pclmul {
            // SAFETY: `pclmul` is only ever set from the runtime CPU probe in
            // `new`, and `ghash_x86::multiply` requires exactly that feature.
            self.ghash_with(aad, ciphertext, |x, y| unsafe {
                crate::ghash_x86::multiply(x, y)
            })
        } else {
            self.ghash_with(aad, ciphertext, multiply_software)
        }
    }

    /// The GHASH structure, independent of which multiply is used, so the two
    /// paths cannot drift apart in padding, ordering or the length block.
    fn ghash_with<M: Fn(u128, u128) -> u128>(
        &self,
        aad: &[u8],
        ciphertext: &[u8],
        multiply: M,
    ) -> u128 {
        let mut y = 0u128;
        for chunk in aad.chunks(16) {
            y = ghash_absorb(y, chunk, self.h, &multiply);
        }
        for chunk in ciphertext.chunks(16) {
            y = ghash_absorb(y, chunk, self.h, &multiply);
        }
        let aad_bits = (aad.len() as u64) << 3;
        let ct_bits = (ciphertext.len() as u64) << 3;
        let length_block = ((aad_bits as u128) << 64) | ct_bits as u128;
        ghash_absorb(y, &length_block.to_be_bytes(), self.h, &multiply)
    }

    /// `E(K, J0) xor S` — the tag (SP 800-38D §7.1 step 5/6).
    fn tag(&self, j0: u128, s: u128) -> [u8; TAG_LEN] {
        let mut mask = j0.to_be_bytes();
        self.aes.encrypt_block(&mut mask);
        let s_bytes = s.to_be_bytes();
        let mut tag = [0u8; TAG_LEN];
        for (t, (m, s)) in tag.iter_mut().zip(mask.iter().zip(s_bytes.iter())) {
            *t = *m ^ *s;
        }
        tag
    }
}

/// The initial counter block for a 96-bit nonce: `nonce || 0x00000001`.
#[inline]
fn j0_from_nonce(nonce: &[u8; NONCE_LEN]) -> u128 {
    let mut block = [0u8; 16];
    block[..NONCE_LEN].copy_from_slice(nonce);
    block[15] = 1;
    u128::from_be_bytes(block)
}

/// `inc32`: increment only the low 32 bits, leaving the rest alone (§6.2).
#[inline]
fn inc32(counter: u128) -> u128 {
    inc32_n(counter, 1)
}

#[inline]
fn inc32_n(counter: u128, n: u32) -> u128 {
    let low = (counter as u32).wrapping_add(n);
    (counter & !0xffff_ffffu128) | low as u128
}

/// Zero-pads `chunk` to a block, XORs it into the accumulator, multiplies by `h`.
#[inline]
fn ghash_absorb(y: u128, chunk: &[u8], h: u128, multiply: &dyn Fn(u128, u128) -> u128) -> u128 {
    let mut block = [0u8; 16];
    block[..chunk.len()].copy_from_slice(chunk);
    multiply(y ^ u128::from_be_bytes(block), h)
}

/// Whether the x86_64 carry-less multiply is usable, or `false` elsewhere.
#[inline]
fn pclmul_available() -> bool {
    if crate::hardware_disabled() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        crate::ghash_x86::available()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Carry-less multiply in GF(2^128) with the GCM bit order (Algorithm 1 of
/// SP 800-38D), written so that every step is a mask instead of a branch.
///
/// `R = 11100001 || 0^120` is the reduction polynomial's low word: when the
/// shifted-out bit is set, folding it back means XOR-ing `R` into the top byte.
#[inline]
pub(crate) fn multiply_software(x: u128, y: u128) -> u128 {
    let mut z = 0u128;
    let mut v = y;
    let mut i = 0u32;
    while i < 128 {
        let bit = (x >> (127 - i)) & 1;
        z ^= v & bit.wrapping_neg();
        let lsb = v & 1;
        v >>= 1;
        v ^= lsb.wrapping_neg() & (0xe1u128 << 120);
        i += 1;
    }
    z
}

/// Constant-time equality: no early exit, no length-dependent memory access
/// pattern beyond the equal-length case.
#[inline]
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(s: &str) -> Vec<u8> {
        let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(cleaned.len().is_multiple_of(2), "odd hex length");
        (0..cleaned.len() / 2)
            .map(|i| u8::from_str_radix(&cleaned[i * 2..i * 2 + 2], 16).expect("hex"))
            .collect()
    }

    fn key(s: &str) -> [u8; 16] {
        h(s).try_into().expect("16-byte key")
    }

    fn nonce(s: &str) -> [u8; NONCE_LEN] {
        h(s).try_into().expect("12-byte nonce")
    }

    /// Every vector below was produced by the independent `cryptography`
    /// (OpenSSL) implementation, and its decryption and tag-tamper rejection
    /// were confirmed there too — see `target/buffy-ab/gcm_vectors.py`. Cases
    /// 1-4 are also the published NIST/GCM-spec AES-128 examples.
    #[test]
    fn nist_case_1_empty_plaintext() {
        let gcm = Aes128Gcm::new(&key("00000000000000000000000000000000"));
        let (ct, tag) = gcm.seal(&nonce("000000000000000000000000"), b"", b"");
        assert!(ct.is_empty());
        assert_eq!(tag[..], h("58e2fccefa7e3061367f1d57a4e7455a")[..]);
    }

    #[test]
    fn nist_case_2_one_zero_block() {
        let gcm = Aes128Gcm::new(&key("00000000000000000000000000000000"));
        let (ct, tag) = gcm.seal(&nonce("000000000000000000000000"), b"", &[0u8; 16]);
        assert_eq!(ct[..], h("0388dace60b6a392f328c2b971b2fe78")[..]);
        assert_eq!(tag[..], h("ab6e47d42cec13bdf53a67b21257bddf")[..]);
    }

    #[test]
    fn nist_case_3_full_block_aligned() {
        let gcm = Aes128Gcm::new(&key("feffe9928665731c6d6a8f9467308308"));
        let n = nonce("cafebabefacedbaddecaf888");
        let pt = h(concat!(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72",
            "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b391aafd255"
        ));
        let (ct, tag) = gcm.seal(&n, b"", &pt);
        assert_eq!(
            ct[..],
            h(concat!(
                "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e",
                "21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091473f5985"
            ))[..]
        );
        assert_eq!(tag[..], h("4d5c2af327cd64a62cf35abd2ba6fab4")[..]);
        assert_eq!(gcm.open(&n, b"", &ct, &tag).unwrap(), pt);
    }

    #[test]
    fn nist_case_4_with_aad() {
        let gcm = Aes128Gcm::new(&key("feffe9928665731c6d6a8f9467308308"));
        let n = nonce("cafebabefacedbaddecaf888");
        let aad = h("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        let pt = h(concat!(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72",
            "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"
        ));
        let (ct, tag) = gcm.seal(&n, &aad, &pt);
        assert_eq!(
            ct[..],
            h(concat!(
                "42831ec2217774244b7221b784d0d49ce3aa212f2c02a4e035c17e2329aca12e",
                "21d514b25466931c7d8f6a5aac84aa051ba30b396a0aac973d58e091"
            ))[..]
        );
        assert_eq!(tag[..], h("5bc94fbc3221a5db94fae95ae7121a47")[..]);
        assert_eq!(gcm.open(&n, &aad, &ct, &tag).unwrap(), pt);
    }

    /// Odd lengths, where GHASH's zero-padding and the length block earn their
    /// keep. A padding bug shows up here and nowhere else.
    #[test]
    fn odd_length_plaintext_and_aad() {
        let gcm = Aes128Gcm::new(&key("00000000000000000000000000000000"));
        let n = nonce("000000000000000000000000");

        let (ct, tag) = gcm.seal(&n, b"\x00", b"\x00");
        assert_eq!(ct[..], h("03")[..]);
        assert_eq!(tag[..], h("bd556d10e79a00111847bc2966a58fc8")[..]);

        let aad = h("000102030405060708090a0b0c0d0e");
        let pt = h("000102030405060708090a0b0c0d0e0f10");
        let (ct, tag) = gcm.seal(&n, &aad, &pt);
        assert_eq!(ct[..], h("0389d8cd64b3a595fb21c8b27dbff077e7")[..]);
        assert_eq!(tag[..], h("7370aa8d6e7a9fc0642d76446d202b00")[..]);
        assert_eq!(gcm.open(&n, &aad, &ct, &tag).unwrap(), pt);
    }

    /// 1000 bytes through both GHASH passes: enough blocks that a
    /// per-block counter or accumulator mistake cannot match the tag.
    #[test]
    fn long_message_matches_the_oracle_tag() {
        let gcm = Aes128Gcm::new(&key("2b7e151628aed2a6abf7158809cf4f3c"));
        let n = nonce("000000000000004a00000000");
        let aad: Vec<u8> = (0..1000u32).map(|i| (i % 250) as u8).collect();
        let pt: Vec<u8> = (0..1000u32).map(|i| ((i * 7) % 256) as u8).collect();
        let (ct, tag) = gcm.seal(&n, &aad, &pt);
        assert_eq!(ct.len(), 1000);
        assert_eq!(tag[..], h("6817c9901e1ee567b413fad92fc97efd")[..]);
        assert_eq!(gcm.open(&n, &aad, &ct, &tag).unwrap(), pt);
    }

    /// A flipped bit anywhere in the tag must fail, not warn or truncate.
    #[test]
    fn tampered_tag_is_rejected() {
        let gcm = Aes128Gcm::new(&key("feffe9928665731c6d6a8f9467308308"));
        let n = nonce("cafebabefacedbaddecaf888");
        let pt = h("d9313225f88406e5a55909c5aff5269a");
        let (ct, tag) = gcm.seal(&n, b"", &pt);

        for byte in 0..TAG_LEN {
            for bit in 0..8 {
                let mut bad = tag;
                bad[byte] ^= 1 << bit;
                assert_eq!(
                    gcm.open(&n, b"", &ct, &bad),
                    Err(GcmError::TagMismatch),
                    "tag byte {} bit {} was accepted",
                    byte,
                    bit
                );
            }
        }
    }

    /// Tampering with the ciphertext, the AAD or the nonce must also fail: the
    /// tag covers all three.
    #[test]
    fn tampered_ciphertext_aad_and_nonce_are_rejected() {
        let gcm = Aes128Gcm::new(&key("feffe9928665731c6d6a8f9467308308"));
        let n = nonce("cafebabefacedbaddecaf888");
        let aad = h("feedfacedeadbeeffeedfacedeadbeefabaddad2");
        let pt = h("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72");
        let (ct, tag) = gcm.seal(&n, &aad, &pt);

        let mut bad_ct = ct.clone();
        bad_ct[0] ^= 1;
        assert_eq!(
            gcm.open(&n, &aad, &bad_ct, &tag),
            Err(GcmError::TagMismatch)
        );

        let mut bad_aad = aad.clone();
        bad_aad[0] ^= 1;
        assert_eq!(
            gcm.open(&n, &bad_aad, &ct, &tag),
            Err(GcmError::TagMismatch)
        );

        let mut bad_nonce = n;
        bad_nonce[11] ^= 1;
        assert_eq!(
            gcm.open(&bad_nonce, &aad, &ct, &tag),
            Err(GcmError::TagMismatch)
        );

        // Truncating the ciphertext must not authenticate either.
        assert_eq!(
            gcm.open(&n, &aad, &ct[..ct.len() - 1], &tag),
            Err(GcmError::TagMismatch)
        );
    }

    /// A one-bit difference in the key must produce a different tag; this is
    /// the cheapest guard against a key schedule that ignores part of the key.
    #[test]
    fn tag_depends_on_every_key_byte() {
        let n = nonce("cafebabefacedbaddecaf888");
        let pt = b"hearth";
        let base = Aes128Gcm::new(&key("feffe9928665731c6d6a8f9467308308"));
        let (_, base_tag) = base.seal(&n, b"", pt);

        for byte in 0..16 {
            for bit in 0..8 {
                let mut k = key("feffe9928665731c6d6a8f9467308308");
                k[byte] ^= 1 << bit;
                let (_, tag) = Aes128Gcm::new(&k).seal(&n, b"", pt);
                assert_ne!(tag, base_tag, "key byte {} bit {}", byte, bit);
            }
        }
    }

    #[test]
    fn inc32_wraps_only_the_low_word() {
        let counter = u128::from_be_bytes(*b"ABCDEFGHIJKL\xff\xff\xff\xff");
        assert_eq!(
            inc32(counter).to_be_bytes()[12..],
            [0x00, 0x00, 0x00, 0x00],
            "low word must wrap to zero"
        );
        assert_eq!(&inc32(counter).to_be_bytes()[..12], b"ABCDEFGHIJKL");
    }
}
