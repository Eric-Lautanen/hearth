//! TLS 1.3 record protection (RFC 8446 §5).
//!
//! This is the layer between the handshake and the socket: it takes a content
//! type and a payload and produces the bytes that go on the wire, or takes wire
//! bytes and produces the payload. Three things here are easy to get subtly
//! wrong, and all three are pinned by a published fixture below rather than by
//! inspection:
//!
//! 1. **The nonce** (§5.3) is the static IV XORed with the 64-bit record
//!    sequence number, serialised big-endian and left-padded with four zero
//!    bytes. Reusing a `(key, nonce)` pair under GCM is catastrophic, so the
//!    sequence number is owned by this struct, incremented exactly when a record
//!    is sealed or opened, and never resettable from outside. The primitive
//!    cannot detect reuse; this is where reuse is prevented.
//! 2. **The additional data** is the record *header*, in its encrypted form
//!    (`0x17 0x03 0x03 <len>`), where `len` counts the ciphertext plus tag. That
//!    means the header cannot be built until the plaintext length is known, and
//!    on receipt the declared length must equal the actual one before it is
//!    trusted as AAD.
//! 3. **The inner plaintext** is `content || content_type || zero padding`. TLS
//!    1.3 hides the real content type inside the encrypted record, so the outer
//!    type is always `application_data`; on the way back the trailing zeros are
//!    padding and the last non-zero byte is the content type.
//!
//! [`RecordLayer`] is one direction of one connection: a fresh instance starts at
//! sequence 0, and the handshake installs two of them (one per direction) when it
//! derives the application traffic secrets.
//!
//! Not implemented here, deliberately: the `key_update` message (RFC 8446 §4.6.3)
//! and the alert protocol. Both belong with the handshake, and
//! [`TrafficKeys::from_secret`] is the single place a rekey will plug in.

use crate::gcm::{Aes128Gcm, GcmError, NONCE_LEN, TAG_LEN};
use crate::hkdf::hkdf_expand_label;

/// Maximum plaintext in one record (RFC 8446 §5.1): `2^14`.
pub const MAX_PLAINTEXT: usize = 1 << 14;

/// Maximum ciphertext in one record: `2^14 + 256`, to allow for expansion.
pub const MAX_CIPHERTEXT: usize = (1 << 14) + 256;

/// Bytes of record header before the protected content.
pub const HEADER_LEN: usize = 5;

/// The legacy record version carried by every TLS 1.3 encrypted record
/// (§5.1 — it is fixed at `0x0303` for compatibility).
pub const LEGACY_VERSION: [u8; 2] = [0x03, 0x03];

/// The only outer content type a protected record may have (§5.2).
pub const CONTENT_TYPE_APPLICATION_DATA: u8 = 23;

/// `change_cipher_spec`, still sent for middlebox compatibility.
pub const CONTENT_TYPE_CHANGE_CIPHER_SPEC: u8 = 20;
/// `alert`.
pub const CONTENT_TYPE_ALERT: u8 = 21;
/// `handshake`, which is what the real content type of handshake messages is
/// once they are encrypted.
pub const CONTENT_TYPE_HANDSHAKE: u8 = 22;

/// Traffic keys for one direction of one connection (RFC 8446 §7.3).
#[derive(Clone)]
pub struct TrafficKeys {
    key: [u8; 16],
    iv: [u8; NONCE_LEN],
}

impl TrafficKeys {
    /// Builds keys from material that is already known — used by tests and by a
    /// caller that received them out of band.
    pub fn new(key: [u8; 16], iv: [u8; NONCE_LEN]) -> Self {
        Self { key, iv }
    }

    /// The record protection key, as derived. Exposed so the key schedule can be
    /// checked against published vectors; nothing in the crate reads it back
    /// out.
    pub fn key(&self) -> &[u8; 16] {
        &self.key
    }

    /// The static IV, which the sequence number is XORed into per record.
    pub fn iv(&self) -> &[u8; NONCE_LEN] {
        &self.iv
    }

    /// Derives the key and IV from a traffic secret, as RFC 8446 §7.3 specifies:
    /// `key = HKDF-Expand-Label(secret, "key", "", 16)` and
    /// `iv = HKDF-Expand-Label(secret, "iv", "", 12)`.
    ///
    /// This is the seam a `key_update` will use, since a rekey is the same
    /// derivation from an updated secret.
    pub fn from_secret(secret: &[u8]) -> Result<Self, String> {
        let mut key = [0u8; 16];
        hkdf_expand_label(secret, "key", &[], &mut key)?;
        let mut iv = [0u8; NONCE_LEN];
        hkdf_expand_label(secret, "iv", &[], &mut iv)?;
        Ok(Self { key, iv })
    }
}

/// A decrypted record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    /// The content type that was hidden inside the record.
    pub content_type: u8,
    /// The payload, with the content-type byte and any padding removed.
    pub payload: Vec<u8>,
}

/// Why a record was rejected.
///
/// Every variant is fatal for the connection: TLS has no recoverable record
/// error, and a caller must never interpret one of these as "skip this record".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
    /// The plaintext exceeds [`MAX_PLAINTEXT`].
    PlaintextTooLong(usize),
    /// Fewer bytes than a header plus a content type plus a tag.
    RecordTooShort(usize),
    /// More bytes than the maximum ciphertext.
    RecordTooLong(usize),
    /// The header's length field disagrees with the bytes actually delivered.
    DeclaredLengthMismatch { declared: usize, actual: usize },
    /// The outer content type was not `application_data`. An unencrypted
    /// `change_cipher_spec` cannot reach here because this layer only exists
    /// after the keys are installed.
    UnexpectedOuterType(u8),
    /// The legacy version was not `0x0303`.
    UnexpectedVersion([u8; 2]),
    /// The sequence number would wrap, which would reuse a GCM nonce. RFC 8446
    /// §5.3 requires the connection to be closed before this happens.
    SequenceExhausted,
    /// The tag did not verify. The ciphertext is forged or the sequence numbers
    /// have diverged.
    AuthenticationFailed,
    /// The inner plaintext was all zero bytes, so there was no content type.
    NoContentType,
    /// The inner content type is not one of the four in §5.4.
    UnknownContentType(u8),
}

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PlaintextTooLong(n) => write!(f, "record plaintext {n} bytes exceeds the limit"),
            Self::RecordTooShort(n) => write!(f, "record {n} bytes is too short to be protected"),
            Self::RecordTooLong(n) => write!(f, "record {n} bytes exceeds the ciphertext limit"),
            Self::DeclaredLengthMismatch { declared, actual } => {
                write!(
                    f,
                    "record header declares {declared} bytes, {actual} delivered"
                )
            }
            Self::UnexpectedOuterType(t) => {
                write!(
                    f,
                    "protected record has outer content type {t}, expected 23"
                )
            }
            Self::UnexpectedVersion(v) => {
                write!(f, "record legacy version {v:02x?} is not 0x0303")
            }
            Self::SequenceExhausted => {
                write!(f, "record sequence number exhausted: rekey required")
            }
            Self::AuthenticationFailed => write!(f, "record failed authentication"),
            Self::NoContentType => write!(f, "record contained padding but no content type"),
            Self::UnknownContentType(t) => write!(f, "record has invalid content type {t}"),
        }
    }
}

impl std::error::Error for RecordError {}

impl From<GcmError> for RecordError {
    fn from(_: GcmError) -> Self {
        Self::AuthenticationFailed
    }
}

/// One direction of a connection's record protection.
pub struct RecordLayer {
    gcm: Aes128Gcm,
    iv: [u8; NONCE_LEN],
    sequence: u64,
}

impl RecordLayer {
    /// Installs traffic keys and starts at sequence number 0.
    pub fn new(keys: &TrafficKeys) -> Self {
        Self {
            gcm: Aes128Gcm::new(&keys.key),
            iv: keys.iv,
            sequence: 0,
        }
    }

    /// The sequence number the next record will use. Exposed for diagnostics
    /// only; there is deliberately no setter.
    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Whether this instance is using hardware AES/GHASH (`AES-NI`, `PCLMULQDQ`).
    pub fn accelerated(&self) -> (bool, bool) {
        self.gcm.accelerated()
    }

    /// Protects one record with no padding.
    pub fn seal(&mut self, content_type: u8, plaintext: &[u8]) -> Result<Vec<u8>, RecordError> {
        self.seal_padded(content_type, plaintext, 0)
    }

    /// Protects one record, appending `padding` zero bytes after the content
    /// type.
    ///
    /// Padding is optional and exists to blunt traffic analysis; the receiver
    /// strips it without being told it was there, so this only ever changes the
    /// record's length.
    pub fn seal_padded(
        &mut self,
        content_type: u8,
        plaintext: &[u8],
        padding: usize,
    ) -> Result<Vec<u8>, RecordError> {
        if plaintext.len() > MAX_PLAINTEXT {
            return Err(RecordError::PlaintextTooLong(plaintext.len()));
        }
        let inner_len = plaintext.len() + 1 + padding;
        let protected_len = inner_len + TAG_LEN;
        if protected_len > MAX_CIPHERTEXT {
            return Err(RecordError::RecordTooLong(protected_len));
        }
        if self.sequence == u64::MAX {
            return Err(RecordError::SequenceExhausted);
        }

        let mut inner = Vec::with_capacity(inner_len);
        inner.extend_from_slice(plaintext);
        inner.push(content_type);
        inner.resize(inner_len, 0);

        // The header is only knowable now, and it is the additional data.
        let header = header_for(protected_len);
        let nonce = nonce_for(&self.iv, self.sequence);
        let (ciphertext, tag) = self.gcm.seal(&nonce, &header, &inner);

        let mut out = Vec::with_capacity(HEADER_LEN + protected_len);
        out.extend_from_slice(&header);
        out.extend_from_slice(&ciphertext);
        out.extend_from_slice(&tag);
        self.sequence += 1;
        Ok(out)
    }

    /// Unprotects one record.
    ///
    /// Anything that is not exactly one well-formed, authenticated record is an
    /// error: there is no partial success and no plaintext on failure.
    pub fn open(&mut self, record: &[u8]) -> Result<Record, RecordError> {
        if record.len() < HEADER_LEN + 1 + TAG_LEN {
            return Err(RecordError::RecordTooShort(record.len()));
        }
        if record.len() > HEADER_LEN + MAX_CIPHERTEXT {
            return Err(RecordError::RecordTooLong(record.len()));
        }

        let header = &record[..HEADER_LEN];
        if header[0] != CONTENT_TYPE_APPLICATION_DATA {
            return Err(RecordError::UnexpectedOuterType(header[0]));
        }
        let version = [header[1], header[2]];
        if version != LEGACY_VERSION {
            return Err(RecordError::UnexpectedVersion(version));
        }
        let declared = u16::from_be_bytes([header[3], header[4]]) as usize;
        let actual = record.len() - HEADER_LEN;
        if declared != actual {
            return Err(RecordError::DeclaredLengthMismatch { declared, actual });
        }
        if self.sequence == u64::MAX {
            return Err(RecordError::SequenceExhausted);
        }

        let body_end = record.len() - TAG_LEN;
        let ciphertext = &record[HEADER_LEN..body_end];
        let tag: [u8; TAG_LEN] = record[body_end..]
            .try_into()
            .expect("exactly TAG_LEN bytes");

        let nonce = nonce_for(&self.iv, self.sequence);
        let inner = self
            .gcm
            .open(&nonce, header, ciphertext, &tag)
            .map_err(RecordError::from)?;

        // Padding is zero bytes before nothing; the last non-zero byte is the
        // content type. An all-zero inner plaintext has no content type.
        let end = inner
            .iter()
            .rposition(|&b| b != 0)
            .ok_or(RecordError::NoContentType)?;
        let content_type = inner[end];
        if !is_valid_content_type(content_type) {
            return Err(RecordError::UnknownContentType(content_type));
        }

        self.sequence += 1;
        Ok(Record {
            content_type,
            payload: inner[..end].to_vec(),
        })
    }
}

/// The record header for a given protected length.
#[inline]
fn header_for(protected_len: usize) -> [u8; HEADER_LEN] {
    let len = protected_len as u16;
    [
        CONTENT_TYPE_APPLICATION_DATA,
        LEGACY_VERSION[0],
        LEGACY_VERSION[1],
        (len >> 8) as u8,
        (len & 0xff) as u8,
    ]
}

/// The per-record nonce (RFC 8446 §5.3): the static IV XORed with the sequence
/// number as an 8-byte big-endian value left-padded with four zero bytes.
#[inline]
fn nonce_for(iv: &[u8; NONCE_LEN], sequence: u64) -> [u8; NONCE_LEN] {
    let mut nonce = *iv;
    let seq = sequence.to_be_bytes();
    // The padding is the first four bytes, which the sequence number never
    // reaches, so those bytes stay equal to the IV's.
    for (n, s) in nonce[4..].iter_mut().zip(seq.iter()) {
        *n ^= *s;
    }
    nonce
}

/// The four content types TLS 1.3 defines (RFC 8446 §5.4).
#[inline]
fn is_valid_content_type(t: u8) -> bool {
    matches!(
        t,
        CONTENT_TYPE_CHANGE_CIPHER_SPEC
            | CONTENT_TYPE_ALERT
            | CONTENT_TYPE_HANDSHAKE
            | CONTENT_TYPE_APPLICATION_DATA
    )
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

    fn b16(s: &str) -> [u8; 16] {
        h(s).try_into().expect("16 bytes")
    }

    fn b12(s: &str) -> [u8; NONCE_LEN] {
        h(s).try_into().expect("12 bytes")
    }

    /// RFC 8448 §3's client application traffic secret.
    const CLIENT_APP_SECRET: &str =
        "9e40646ce79a7f9dc05af8889bce6552875afa0b06df0087f792ebb7c17504a5";

    /// RFC 8448 §3's published protected application_data record: the header,
    /// 51 bytes of ciphertext and the 16-byte tag, for the 50-byte payload
    /// `00 01 02 … 31` at sequence number 0.
    const RFC8448_RECORD: &str = "17030300 43 \
        a23f7054b62c94d0affafe8228ba55cbefacea42f914aa66bcab3f2b9819a8a5b46b395bd54a9a20\
        441e2b62974e1f5a6292a2 \
        977014bd1e3deae63aeebb21694915e4";

    fn rfc8448_payload() -> Vec<u8> {
        (0u8..50).collect()
    }

    /// The published key and IV, derived in-crate from the published secret.
    /// This is the test that validates the `tls13 key` / `tls13 iv` labels and
    /// the empty context in `hkdf_expand_label`.
    #[test]
    fn derives_rfc8448_application_keys() {
        let keys = TrafficKeys::from_secret(&h(CLIENT_APP_SECRET)).expect("derive");
        assert_eq!(keys.key[..], h("17422dda596ed5d9acd890e3c63f5051")[..]);
        assert_eq!(keys.iv[..], h("5b78923dee08579033e523d9")[..]);
    }

    /// The whole record, byte for byte: this pins the nonce construction, the
    /// header-as-additional-data rule, the inner content-type byte and the
    /// ciphertext||tag layout all at once. If any one of them were wrong, the
    /// bytes would not match.
    #[test]
    fn seals_the_rfc8448_application_record() {
        let keys = TrafficKeys::from_secret(&h(CLIENT_APP_SECRET)).expect("derive");
        let mut layer = RecordLayer::new(&keys);
        assert_eq!(layer.sequence(), 0);

        let record = layer
            .seal(CONTENT_TYPE_APPLICATION_DATA, &rfc8448_payload())
            .expect("seal");
        assert_eq!(record[..], h(RFC8448_RECORD)[..]);
        assert_eq!(layer.sequence(), 1);
        assert_eq!(record.len(), HEADER_LEN + 51 + TAG_LEN);
    }

    /// And the same published record opens.
    #[test]
    fn opens_the_rfc8448_application_record() {
        let keys = TrafficKeys::from_secret(&h(CLIENT_APP_SECRET)).expect("derive");
        let mut layer = RecordLayer::new(&keys);
        let opened = layer.open(&h(RFC8448_RECORD)).expect("open");
        assert_eq!(opened.content_type, CONTENT_TYPE_APPLICATION_DATA);
        assert_eq!(opened.payload, rfc8448_payload());
        assert_eq!(layer.sequence(), 1);
    }

    /// Round-trip across every content type, several lengths including the
    /// maximum, and with padding.
    #[test]
    fn round_trips_all_content_types_and_lengths() {
        let keys = TrafficKeys::new(
            b16("00112233445566778899aabbccddeeff"),
            b12("0f0e0d0c0b0a090807060504"),
        );
        // One pair of layers for the whole sweep, so each record gets its own
        // sequence number. Building fresh layers per iteration would reuse
        // (key, nonce) across different plaintexts, which is the exact thing
        // this layer exists to prevent.
        let mut sender = RecordLayer::new(&keys);
        let mut receiver = RecordLayer::new(&keys);
        for content_type in [
            CONTENT_TYPE_CHANGE_CIPHER_SPEC,
            CONTENT_TYPE_ALERT,
            CONTENT_TYPE_HANDSHAKE,
            CONTENT_TYPE_APPLICATION_DATA,
        ] {
            for len in [0usize, 1, 15, 16, 17, 255, 1000, MAX_PLAINTEXT] {
                let payload: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
                let record = sender.seal(content_type, &payload).expect("seal");
                let opened = receiver.open(&record).expect("open");
                assert_eq!(opened.content_type, content_type, "len {len}");
                assert_eq!(opened.payload, payload, "len {len}");
            }
        }
    }

    #[test]
    fn padding_is_stripped_without_being_declared() {
        let keys = TrafficKeys::new(
            b16("00112233445566778899aabbccddeeff"),
            b12("0f0e0d0c0b0a090807060504"),
        );
        let mut sender = RecordLayer::new(&keys);
        let mut receiver = RecordLayer::new(&keys);

        let record = sender
            .seal_padded(CONTENT_TYPE_HANDSHAKE, b"hello", 64)
            .expect("seal");
        assert_eq!(record.len(), HEADER_LEN + 5 + 1 + 64 + TAG_LEN);
        let opened = receiver.open(&record).expect("open");
        assert_eq!(opened.content_type, CONTENT_TYPE_HANDSHAKE);
        assert_eq!(opened.payload, b"hello");
    }

    /// The nonce must depend on the sequence number: a record sealed at
    /// sequence 1 must not verify against a receiver still expecting 0. Without
    /// this, nonce reuse would go unnoticed.
    #[test]
    fn sequence_number_is_part_of_the_nonce() {
        let keys = TrafficKeys::new(
            b16("00112233445566778899aabbccddeeff"),
            b12("0f0e0d0c0b0a090807060504"),
        );
        let mut sender = RecordLayer::new(&keys);
        let mut receiver = RecordLayer::new(&keys);

        let first = sender
            .seal(CONTENT_TYPE_APPLICATION_DATA, b"one")
            .expect("seal");
        let second = sender
            .seal(CONTENT_TYPE_APPLICATION_DATA, b"two")
            .expect("seal");
        assert_ne!(first, second);

        // Skipping the first record desynchronises the sequence numbers.
        assert_eq!(
            receiver.open(&second),
            Err(RecordError::AuthenticationFailed)
        );
        assert_eq!(receiver.open(&first).expect("first").payload, b"one");
        assert_eq!(receiver.open(&second).expect("second").payload, b"two");
    }

    /// Sealing the same plaintext twice must produce different records, because
    /// the sequence number differs. Identical output would mean a reused nonce.
    #[test]
    fn repeated_plaintext_never_repeats_ciphertext() {
        let keys = TrafficKeys::new(
            b16("00112233445566778899aabbccddeeff"),
            b12("0f0e0d0c0b0a090807060504"),
        );
        let mut layer = RecordLayer::new(&keys);
        let a = layer
            .seal(CONTENT_TYPE_APPLICATION_DATA, b"same")
            .expect("seal");
        let b = layer
            .seal(CONTENT_TYPE_APPLICATION_DATA, b"same")
            .expect("seal");
        assert_ne!(a, b);
        // Only the counter block differs, so the first ciphertext byte differs.
        assert_ne!(a[HEADER_LEN], b[HEADER_LEN]);
    }

    /// Every field is authenticated, and each tamper is fatal.
    #[test]
    fn tampering_any_field_is_rejected() {
        let keys = TrafficKeys::new(
            b16("00112233445566778899aabbccddeeff"),
            b12("0f0e0d0c0b0a090807060504"),
        );

        let seal = |keys: &TrafficKeys| {
            let mut layer = RecordLayer::new(keys);
            layer
                .seal(CONTENT_TYPE_APPLICATION_DATA, b"authenticated")
                .expect("seal")
        };
        let good = seal(&keys);
        // "authenticated" is 13 bytes, so the protected length is 14 + 16 = 30
        // and the header carries 0x001e. The lowest length byte is the one to
        // move: the high byte is already zero, so bumping it would be a no-op.
        let declared = 30usize;
        assert_eq!((good[3], good[4]), (0x00, 0x1e));
        assert_eq!(good.len(), HEADER_LEN + declared);

        let cases: Vec<(&str, RecordError)> = vec![
            ("ciphertext", RecordError::AuthenticationFailed),
            ("tag", RecordError::AuthenticationFailed),
            ("outer type", RecordError::UnexpectedOuterType(0x16)),
            (
                "length field",
                RecordError::DeclaredLengthMismatch {
                    declared: declared + 1,
                    actual: declared,
                },
            ),
            ("version", RecordError::UnexpectedVersion([0x03, 0x01])),
        ];

        for (which, expected) in cases {
            let mut bad = good.clone();
            match which {
                "ciphertext" => bad[HEADER_LEN] ^= 0x01,
                "tag" => *bad.last_mut().expect("non-empty") ^= 0x01,
                "outer type" => bad[0] = 0x16,
                "length field" => bad[4] = (declared + 1) as u8,
                "version" => bad[2] = 0x01,
                _ => unreachable!(),
            }
            let mut receiver = RecordLayer::new(&keys);
            assert_eq!(receiver.open(&bad), Err(expected), "tampered {which}");
            // A failed record must not consume a sequence number.
            assert_eq!(
                receiver.sequence(),
                0,
                "tampered {which} advanced the sequence"
            );
        }
    }

    /// Truncation and over-length are rejected before any crypto runs, and the
    /// limits are the ones the RFC states.
    #[test]
    fn length_limits_are_enforced() {
        let keys = TrafficKeys::new(
            b16("00112233445566778899aabbccddeeff"),
            b12("0f0e0d0c0b0a090807060504"),
        );
        let mut layer = RecordLayer::new(&keys);

        // One byte over the plaintext limit.
        let too_long = vec![0u8; MAX_PLAINTEXT + 1];
        assert_eq!(
            layer.seal(CONTENT_TYPE_APPLICATION_DATA, &too_long),
            Err(RecordError::PlaintextTooLong(MAX_PLAINTEXT + 1))
        );
        // Exactly at the limit is fine.
        assert!(layer
            .seal(CONTENT_TYPE_APPLICATION_DATA, &vec![0u8; MAX_PLAINTEXT])
            .is_ok());

        assert_eq!(
            layer.open(&[0u8; HEADER_LEN + 1 + TAG_LEN - 1]),
            Err(RecordError::RecordTooShort(HEADER_LEN + TAG_LEN))
        );
        assert_eq!(
            layer.open(&vec![0u8; HEADER_LEN + MAX_CIPHERTEXT + 1]),
            Err(RecordError::RecordTooLong(HEADER_LEN + MAX_CIPHERTEXT + 1))
        );
    }

    /// A content type of zero is not a content type: with an empty payload the
    /// receiver sees only padding and says so, and with a payload it sees an
    /// invalid type rather than silently treating padding as data.
    #[test]
    fn a_zero_content_type_is_not_a_content_type() {
        let keys = TrafficKeys::new(
            b16("00112233445566778899aabbccddeeff"),
            b12("0f0e0d0c0b0a090807060504"),
        );
        let mut sender = RecordLayer::new(&keys);
        let mut receiver = RecordLayer::new(&keys);

        let empty = sender.seal(0, b"").expect("seal");
        assert_eq!(receiver.open(&empty), Err(RecordError::NoContentType));

        // Fresh layers for the second case: the record above authenticated but
        // carried no content type, which is fatal, and a fatal record does not
        // advance the sequence number (so this receiver still expects 0 while
        // the sender above had already moved on to 1).
        let mut sender = RecordLayer::new(&keys);
        let mut receiver = RecordLayer::new(&keys);
        let with_payload = sender.seal(0, b"data").expect("seal");
        // Trailing zeros are padding, and the last non-zero byte is "a".
        assert_eq!(
            receiver.open(&with_payload),
            Err(RecordError::UnknownContentType(b'a'))
        );
    }

    /// The nonce construction itself, against the definition rather than
    /// against the code that uses it.
    #[test]
    fn nonce_is_the_iv_xor_the_sequence_number() {
        let iv = b12("000102030405060708090a0b");
        // Sequence 0 leaves the IV alone.
        assert_eq!(nonce_for(&iv, 0), iv);
        // Sequence 1 changes only the last byte.
        assert_eq!(nonce_for(&iv, 1)[..11], iv[..11]);
        assert_eq!(nonce_for(&iv, 1)[11], 0x0b ^ 0x01);
        // A sequence number reaching into the fifth byte cannot touch the
        // first four, which stay equal to the IV's.
        let big = nonce_for(&iv, 0xdead_beef_cafe_0001);
        assert_eq!(big[..4], iv[..4]);
        // The sequence number's 8 bytes XOR onto the IV's last 8, in order.
        assert_eq!(
            big[4..],
            [
                0x04 ^ 0xde,
                0x05 ^ 0xad,
                0x06 ^ 0xbe,
                0x07 ^ 0xef,
                0x08 ^ 0xca,
                0x09 ^ 0xfe,
                // The 7th byte of 0xdead_beef_cafe_0001 is zero, so the IV's
                // byte survives unchanged.
                0x0a,
                0x0b ^ 0x01,
            ]
        );
    }

    /// The header must be the encrypted form of the record, since that is what
    /// both sides authenticate.
    #[test]
    fn header_is_the_authenticated_form() {
        assert_eq!(header_for(67), [0x17, 0x03, 0x03, 0x00, 0x43]);
        assert_eq!(header_for(0x1234), [0x17, 0x03, 0x03, 0x12, 0x34]);
    }
}
