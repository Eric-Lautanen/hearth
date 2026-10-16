//! SHA-256 (FIPS 180-4) and HMAC-SHA256 (RFC 2104).
//!
//! Hand-rolled on purpose: the workspace takes no crypto dependency, and this
//! must stay `std`-only so `hearth-ui` keeps building on Linux, macOS and
//! Windows without a C toolchain. Everything here is checked against published
//! known-answer vectors in `tests` below — SHA-256 against FIPS 180-4, HMAC
//! against RFC 4231.
//!
//! The API is streaming because the TLS 1.3 transcript hash is updated
//! incrementally as handshake messages are sent and received.

/// Round constants (FIPS 180-4 §4.2.2).
const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Initial hash value (FIPS 180-4 §5.3.3).
const INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Incremental SHA-256.
#[derive(Clone)]
pub struct Sha256 {
    state: [u32; 8],
    block: [u8; 64],
    filled: usize,
    bytes: u64,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Block size in bytes.
    pub const BLOCK: usize = 64;
    /// Digest size in bytes.
    pub const OUTPUT: usize = 32;

    /// A fresh hasher.
    pub fn new() -> Self {
        Self {
            state: INIT,
            block: [0u8; 64],
            filled: 0,
            bytes: 0,
        }
    }

    /// Absorb `data`. May be called any number of times.
    pub fn update(&mut self, mut data: &[u8]) {
        self.bytes = self.bytes.wrapping_add(data.len() as u64);

        // Top up a partially filled block first.
        if self.filled > 0 {
            let take = (Self::BLOCK - self.filled).min(data.len());
            self.block[self.filled..self.filled + take].copy_from_slice(&data[..take]);
            self.filled += take;
            data = &data[take..];
            if self.filled == Self::BLOCK {
                let block = self.block;
                self.compress(&block);
                self.filled = 0;
            }
        }

        // Whole blocks straight through.
        while data.len() >= Self::BLOCK {
            let (head, tail) = data.split_at(Self::BLOCK);
            let mut block = [0u8; 64];
            block.copy_from_slice(head);
            self.compress(&block);
            data = tail;
        }

        // Keep the remainder for next time.
        if !data.is_empty() {
            self.block[..data.len()].copy_from_slice(data);
            self.filled = data.len();
        }
    }

    /// Finish and return the 32-byte digest.
    pub fn finalize(mut self) -> [u8; 32] {
        let bits = self.bytes.wrapping_mul(8);
        // Padding: 0x80, then zeros to 56 mod 64, then the length in bits.
        self.update(&[0x80]);
        while self.filled != 56 {
            self.update(&[0]);
        }
        self.update(&bits.to_be_bytes());

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    /// One-shot convenience.
    pub fn digest(data: &[u8]) -> [u8; 32] {
        let mut h = Self::new();
        h.update(data);
        h.finalize()
    }

    fn compress(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                block[i * 4],
                block[i * 4 + 1],
                block[i * 4 + 2],
                block[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let mut h = self.state;
        for i in 0..64 {
            let s1 = h[4].rotate_right(6) ^ h[4].rotate_right(11) ^ h[4].rotate_right(25);
            let ch = (h[4] & h[5]) ^ (!h[4] & h[6]);
            let t1 = h[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = h[0].rotate_right(2) ^ h[0].rotate_right(13) ^ h[0].rotate_right(22);
            let maj = (h[0] & h[1]) ^ (h[0] & h[2]) ^ (h[1] & h[2]);
            let t2 = s0.wrapping_add(maj);

            h[7] = h[6];
            h[6] = h[5];
            h[5] = h[4];
            h[4] = h[3].wrapping_add(t1);
            h[3] = h[2];
            h[2] = h[1];
            h[1] = h[0];
            h[0] = t1.wrapping_add(t2);
        }

        for (state, word) in self.state.iter_mut().zip(h.iter()) {
            *state = state.wrapping_add(*word);
        }
    }
}

/// HMAC-SHA256 over several pieces, without concatenating them first.
///
/// The transcript hash and the HKDF info strings are naturally made of parts,
/// and this keeps them off the heap.
pub fn hmac_sha256_parts(key: &[u8], parts: &[&[u8]]) -> [u8; 32] {
    // RFC 2104: keys longer than the block size are hashed first; shorter keys
    // are zero-padded to the block size (which is what the copies do here).
    let mut k = [0u8; 64];
    if key.len() > Sha256::BLOCK {
        k[..32].copy_from_slice(&Sha256::digest(key));
    } else {
        k[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= k[i];
        opad[i] ^= k[i];
    }

    let mut inner = Sha256::new();
    inner.update(&ipad);
    for part in parts {
        inner.update(part);
    }
    let inner = inner.finalize();

    let mut outer = Sha256::new();
    outer.update(&opad);
    outer.update(&inner);
    outer.finalize()
}

/// HMAC-SHA256 of a single message.
pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    hmac_sha256_parts(key, &[msg])
}

/// Lowercase hex, for tests and handshake debugging.
pub fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(char::from_digit((b >> 4) as u32, 16).unwrap_or('?'));
        s.push(char::from_digit((b & 0xf) as u32, 16).unwrap_or('?'));
    }
    s
}

/// Decode lowercase/uppercase hex; `None` on an odd length or a bad nibble.
pub fn from_hex(text: &str) -> Option<Vec<u8>> {
    let t = text.trim();
    if !t.len().is_multiple_of(2) {
        return None;
    }
    let bytes = t.as_bytes();
    let mut out = Vec::with_capacity(t.len() / 2);
    let nibble = |c: u8| -> Option<u8> {
        match c {
            b'0'..=b'9' => Some(c - b'0'),
            b'a'..=b'f' => Some(c - b'a' + 10),
            b'A'..=b'F' => Some(c - b'A' + 10),
            _ => None,
        }
    };
    for pair in bytes.chunks(2) {
        out.push((nibble(pair[0])? << 4) | nibble(pair[1])?);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        to_hex(bytes)
    }

    /// FIPS 180-4 / NIST example vectors.
    #[test]
    fn sha256_known_answers() {
        // Empty message.
        assert_eq!(
            hex(&Sha256::digest(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // "abc"
        assert_eq!(
            hex(&Sha256::digest(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // The 448-bit two-block message from FIPS 180-4 appendix B.2.
        assert_eq!(
            hex(&Sha256::digest(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    /// A digest that spans many blocks, plus a boundary case at 55/56/64 bytes
    /// where the length field either fits in the final block or forces another.
    #[test]
    fn sha256_block_boundaries() {
        let long = vec![b'a'; 1_000_000];
        assert_eq!(
            hex(&Sha256::digest(&long)),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );

        // 55 bytes + padding + length fits in one block; 56 does not.
        for n in [55usize, 56, 63, 64, 65, 119, 120] {
            let data = vec![b'x'; n];
            let mut streamed = Sha256::new();
            streamed.update(&data);
            assert_eq!(
                streamed.finalize(),
                Sha256::digest(&data),
                "one-shot and streamed disagree at {n} bytes"
            );
        }
    }

    /// Streaming in awkward chunks must equal the one-shot digest.
    #[test]
    fn streaming_matches_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        for chunk in [1usize, 3, 31, 63, 64, 65, 127, 999, 1000, 1001] {
            let mut h = Sha256::new();
            for part in data.chunks(chunk) {
                h.update(part);
            }
            assert_eq!(h.finalize(), Sha256::digest(&data), "chunk size {chunk}");
        }
    }

    /// RFC 4231 HMAC-SHA256 test cases.
    #[test]
    fn hmac_known_answers() {
        // 1. 20-byte key, short data.
        assert_eq!(
            hex(&hmac_sha256(&[0x0b; 20], b"Hi There")),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        // 2. ASCII key.
        assert_eq!(
            hex(&hmac_sha256(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        // 3. 20-byte 0xaa key, 50 bytes of 0xdd.
        assert_eq!(
            hex(&hmac_sha256(&[0xaa; 20], &[0xdd; 50])),
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"
        );
        // 4. 25-byte counting key, 50 bytes of 0xcd.
        let key: Vec<u8> = (1..=25u8).collect();
        assert_eq!(
            hex(&hmac_sha256(&key, &[0xcd; 50])),
            "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b"
        );
        // 6. 131-byte key: longer than the block size, so it is hashed first.
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
        // 7. Larger than block size for both key and data.
        assert_eq!(
            hex(&hmac_sha256(
                &[0xaa; 131],
                b"This is a test using a larger than block-size key and a larger than block-size data. The key needs to be hashed before being used by the HMAC algorithm."
            )),
            "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2"
        );
    }

    /// A 64-byte key is *not* hashed (the `> 64` boundary), and 65 is.
    #[test]
    fn hmac_key_length_boundary() {
        let k64 = [0x01u8; 64];
        let k65 = [0x01u8; 65];
        assert_ne!(hmac_sha256(&k64, b"x"), hmac_sha256(&k65, b"x"));
        // A 64-byte key must be the same as the equivalent padded shorter key.
        let mut padded = [0u8; 64];
        padded[..32].copy_from_slice(&[0x01u8; 32]);
        assert_eq!(hmac_sha256(&[0x01u8; 32], b"x"), hmac_sha256(&padded, b"x"));
    }

    /// Splitting the message must not change the MAC (this is what the
    /// transcript hash and HKDF rely on).
    #[test]
    fn hmac_parts_match_single_buffer() {
        let key = [0x42u8; 32];
        let a = b"the quick brown ";
        let b = b"fox jumps over ";
        let c = b"the lazy dog";
        let joined = [&a[..], &b[..], &c[..]].concat();
        assert_eq!(
            hmac_sha256_parts(&key, &[a, b, c]),
            hmac_sha256(&key, &joined)
        );
        // Empty parts are allowed and must not perturb the result.
        assert_eq!(
            hmac_sha256_parts(&key, &[&[], a, &[], b, c, &[]]),
            hmac_sha256(&key, &joined)
        );
    }

    #[test]
    fn hex_round_trip() {
        let raw = [0x00u8, 0x0f, 0xf0, 0xff, 0x5a];
        assert_eq!(to_hex(&raw), "000ff0ff5a");
        assert_eq!(from_hex("000FF0ff5a").unwrap(), raw);
        assert_eq!(from_hex("abc"), None);
        assert_eq!(from_hex("zz"), None);
    }
}
