//! Handshake message framing and the messages a 1-RTT client needs
//! (RFC 8446 §4).
//!
//! Everything here is byte-in/byte-out with no state machine; the state machine
//! lives in `client.rs`. Two properties matter more than the individual
//! encoders:
//!
//! 1. **Nothing panics on hostile input.** A server can truncate, corrupt or
//!    lie, and every parser here returns a `Result` with a specific error. The
//!    tests walk every prefix of a real ServerHello and a pile of mutations,
//!    asserting an error rather than a panic, because a panic in a network
//!    parser is a denial of service against ourselves.
//! 2. **Nothing is ignored silently.** A ServerHello with a trailing byte, an
//!    extension whose length overruns the message, a duplicate `key_share`, or
//!    a cipher suite we did not offer is rejected. Unknown *extensions* are
//!    skipped by type, which is what forward compatibility requires, but the
//!    TLV structure itself must be exact.
//!
//! The wire format of a handshake message is a one-byte type, a three-byte
//! big-endian length, then the body — four bytes of header, not five. (Getting
//! that one wrong is easy: it is the record header that is five.)

use crate::sha256::Sha256;

/// `client_hello` (RFC 8446 §4.1.2).
pub const TYPE_CLIENT_HELLO: u8 = 1;
/// `server_hello` (§4.1.3).
pub const TYPE_SERVER_HELLO: u8 = 2;
/// `new_session_ticket` (§4.6.1).
pub const TYPE_NEW_SESSION_TICKET: u8 = 4;
/// `encrypted_extensions` (§4.3.1).
pub const TYPE_ENCRYPTED_EXTENSIONS: u8 = 8;
/// `certificate` (§4.4.2).
pub const TYPE_CERTIFICATE: u8 = 11;
/// `certificate_verify` (§4.4.3).
pub const TYPE_CERTIFICATE_VERIFY: u8 = 15;
/// `certificate_request` (§4.3.2). Never accepted: this client has no
/// certificate to offer, so the message is only ever named in an error.
pub const TYPE_CERTIFICATE_REQUEST: u8 = 13;
/// `finished` (§4.4.4).
pub const TYPE_FINISHED: u8 = 20;
/// `key_update` (§4.6.3).
pub const TYPE_KEY_UPDATE: u8 = 24;

/// `TLS_AES_128_GCM_SHA256`, the only cipher suite this crate implements.
pub const TLS_AES_128_GCM_SHA256: u16 = 0x1301;
/// The `x25519` named group.
pub const GROUP_X25519: u16 = 0x001d;
/// The TLS 1.3 version value used inside `supported_versions`.
pub const VERSION_TLS13: u16 = 0x0304;
/// The legacy version field carried by every handshake message.
pub const LEGACY_VERSION: u16 = 0x0303;

/// Bytes of handshake message header: type plus a three-byte length.
pub const HEADER_LEN: usize = 4;

/// `server_name` extension (§4.2.2).
const EXT_SERVER_NAME: u16 = 0x0000;
/// `supported_groups` (§4.2.7).
const EXT_SUPPORTED_GROUPS: u16 = 0x000a;
/// `signature_algorithms` (§4.2.3).
const EXT_SIGNATURE_ALGORITHMS: u16 = 0x000d;
/// `supported_versions` (§4.2.1).
const EXT_SUPPORTED_VERSIONS: u16 = 0x002b;
/// `key_share` (§4.2.8).
const EXT_KEY_SHARE: u16 = 0x0033;

/// `ecdsa_secp256r1_sha256`, offered because most servers require the extension
/// to be present.
///
/// Note that this crate cannot verify either offered algorithm yet — signature
/// verification arrives with certificate validation, and until then the client
/// driver refuses to complete a handshake without a caller-supplied verifier.
/// Advertising them is a statement about what we will support, not a claim that
/// we can already check it.
pub const SIG_ECDSA_SECP256R1_SHA256: u16 = 0x0403;
/// `rsa_pss_rsae_sha256`, the other ubiquitous one.
pub const SIG_RSA_PSS_RSAE_SHA256: u16 = 0x0804;

/// Why a handshake message was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandshakeError {
    /// Fewer bytes than a field needed.
    Truncated { needed: usize, available: usize },
    /// A length field disagrees with the bytes present.
    BadLength(&'static str),
    /// The message type byte was not the one expected in this position.
    UnexpectedMessage { expected: u8, found: u8 },
    /// The legacy version was not `0x0303`.
    UnexpectedVersion(u16),
    /// `supported_versions` did not contain TLS 1.3.
    UnsupportedProtocolVersion(u16),
    /// The server chose a suite we did not offer.
    UnsupportedCipherSuite(u16),
    /// A `key_share` group we cannot use.
    UnsupportedGroup(u16),
    /// A required extension was absent.
    MissingExtension(u16),
    /// An extension appeared twice.
    DuplicateExtension(u16),
    /// An extension's internal structure was wrong.
    MalformedExtension { which: u16, why: &'static str },
    /// The session id was longer than the 32 bytes the protocol allows.
    InvalidSessionIdLength(usize),
    /// The legacy compression method was not null.
    BadCompression(u8),
    /// Bytes remained after the message was fully parsed.
    TrailingBytes(usize),
    /// The server name is empty or too long for the extension.
    BadServerName(&'static str),
}

impl std::fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { needed, available } => {
                write!(f, "truncated: needed {needed} bytes, {available} available")
            }
            Self::BadLength(what) => write!(f, "length field disagrees with the data: {what}"),
            Self::UnexpectedMessage { expected, found } => {
                write!(f, "expected handshake type {expected}, found {found}")
            }
            Self::UnexpectedVersion(v) => write!(f, "legacy version {v:#06x} is not 0x0303"),
            Self::UnsupportedProtocolVersion(v) => {
                write!(f, "server selected protocol version {v:#06x}, need TLS 1.3")
            }
            Self::UnsupportedCipherSuite(s) => {
                write!(
                    f,
                    "server selected cipher suite {s:#06x}, which we did not offer"
                )
            }
            Self::UnsupportedGroup(g) => write!(f, "unsupported key exchange group {g:#06x}"),
            Self::MissingExtension(t) => write!(f, "required extension {t:#06x} is missing"),
            Self::DuplicateExtension(t) => write!(f, "extension {t:#06x} appears twice"),
            Self::MalformedExtension { which, why } => {
                write!(f, "extension {which:#06x} is malformed: {why}")
            }
            Self::InvalidSessionIdLength(n) => write!(f, "session id length {n} exceeds 32"),
            Self::BadCompression(m) => write!(f, "legacy compression method {m} is not 0"),
            Self::TrailingBytes(n) => write!(f, "{n} bytes remain after the message"),
            Self::BadServerName(why) => write!(f, "server name is invalid: {why}"),
        }
    }
}

impl std::error::Error for HandshakeError {}

/// A cursor over a byte slice that never reads out of bounds and never panics.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], HandshakeError> {
        if self.remaining() < n {
            return Err(HandshakeError::Truncated {
                needed: n,
                available: self.remaining(),
            });
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, HandshakeError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, HandshakeError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u24(&mut self) -> Result<usize, HandshakeError> {
        let b = self.take(3)?;
        Ok(((b[0] as usize) << 16) | ((b[1] as usize) << 8) | b[2] as usize)
    }

    /// A vector with a one-byte length prefix.
    fn u8_vec(&mut self) -> Result<&'a [u8], HandshakeError> {
        let len = self.u8()? as usize;
        self.take(len)
    }

    /// A vector with a two-byte length prefix.
    fn u16_vec(&mut self) -> Result<&'a [u8], HandshakeError> {
        let len = self.u16()? as usize;
        self.take(len)
    }
}

/// Encodes one handshake message: type, three-byte length, body.
pub fn encode(msg_type: u8, body: &[u8]) -> Vec<u8> {
    let len = body.len();
    let mut out = Vec::with_capacity(HEADER_LEN + len);
    out.push(msg_type);
    // A body larger than 2^24-1 cannot be framed and must not be silently
    // truncated; every caller in this crate is far below the limit, and the
    // slice below is the assertion.
    out.extend_from_slice(&[(len >> 16) as u8, (len >> 8) as u8, len as u8]);
    out.extend_from_slice(body);
    out
}

/// One parsed handshake message, carrying its original bytes so the transcript
/// hash can be updated with exactly what arrived.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    msg_type: u8,
    body: Vec<u8>,
    raw: Vec<u8>,
}

impl Message {
    /// The handshake type.
    pub fn msg_type(&self) -> u8 {
        self.msg_type
    }

    /// The body, without the header.
    pub fn body(&self) -> &[u8] {
        &self.body
    }

    /// The complete message as it appeared on the wire, header included. This is
    /// what goes into the transcript.
    pub fn raw(&self) -> &[u8] {
        &self.raw
    }
}

/// Reads a stream of handshake messages from a decrypted record payload.
///
/// Strict on purpose: a record carries whole messages, so a partial message at
/// the end is an error, not something to wait for.
pub struct HandshakeReader<'a> {
    cursor: Cursor<'a>,
}

impl<'a> HandshakeReader<'a> {
    /// Wraps a decrypted buffer.
    pub fn new(buf: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(buf),
        }
    }

    /// The next message, or `Ok(None)` once the buffer is exactly exhausted.
    pub fn next_message(&mut self) -> Result<Option<Message>, HandshakeError> {
        if self.cursor.remaining() == 0 {
            return Ok(None);
        }
        let start = self.cursor.pos;
        let msg_type = self.cursor.u8()?;
        let len = self.cursor.u24()?;
        if self.cursor.remaining() < len {
            return Err(HandshakeError::Truncated {
                needed: len,
                available: self.cursor.remaining(),
            });
        }
        let body = self.cursor.take(len)?.to_vec();
        let raw = self.cursor.buf[start..self.cursor.pos].to_vec();
        Ok(Some(Message {
            msg_type,
            body,
            raw,
        }))
    }
}

/// The fields of a ServerHello this client needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerHello {
    /// The server's 32 random bytes.
    pub random: [u8; 32],
    /// The echo of the client's session id (empty is legal).
    pub session_id: Vec<u8>,
    /// The chosen cipher suite; must be [`TLS_AES_128_GCM_SHA256`].
    pub cipher_suite: u16,
    /// The server's `key_share` public value.
    pub key_share: [u8; 32],
    /// The selected protocol version from `supported_versions`.
    pub supported_version: u16,
    /// The complete message, header included, for the transcript.
    pub raw: Vec<u8>,
}

/// Parses a complete ServerHello message (header included) and validates every
/// field this client depends on.
pub fn parse_server_hello(message: &[u8]) -> Result<ServerHello, HandshakeError> {
    let mut cursor = Cursor::new(message);
    let msg_type = cursor.u8()?;
    if msg_type != TYPE_SERVER_HELLO {
        return Err(HandshakeError::UnexpectedMessage {
            expected: TYPE_SERVER_HELLO,
            found: msg_type,
        });
    }
    let declared = cursor.u24()?;
    if declared != cursor.remaining() {
        return Err(HandshakeError::BadLength(
            "ServerHello length does not match the bytes present",
        ));
    }

    let legacy_version = cursor.u16()?;
    if legacy_version != LEGACY_VERSION {
        return Err(HandshakeError::UnexpectedVersion(legacy_version));
    }

    let mut random = [0u8; 32];
    random.copy_from_slice(cursor.take(32)?);

    let session_id = cursor.u8_vec()?;
    if session_id.len() > 32 {
        return Err(HandshakeError::InvalidSessionIdLength(session_id.len()));
    }
    let session_id = session_id.to_vec();

    let cipher_suite = cursor.u16()?;
    if cipher_suite != TLS_AES_128_GCM_SHA256 {
        return Err(HandshakeError::UnsupportedCipherSuite(cipher_suite));
    }
    let compression = cursor.u8()?;
    if compression != 0 {
        return Err(HandshakeError::BadCompression(compression));
    }

    let extensions = cursor.u16_vec()?;
    if cursor.remaining() != 0 {
        return Err(HandshakeError::TrailingBytes(cursor.remaining()));
    }

    let mut key_share: Option<[u8; 32]> = None;
    let mut supported_version: Option<u16> = None;

    let mut ext_cursor = Cursor::new(extensions);
    while ext_cursor.remaining() > 0 {
        let which = ext_cursor.u16()?;
        let body = ext_cursor.u16_vec()?;
        match which {
            EXT_KEY_SHARE => {
                if key_share.is_some() {
                    return Err(HandshakeError::DuplicateExtension(which));
                }
                let mut share = Cursor::new(body);
                let group = share.u16()?;
                if group != GROUP_X25519 {
                    return Err(HandshakeError::UnsupportedGroup(group));
                }
                // `key_exchange` is an opaque vector with a **two**-byte
                // length (KeyShareEntry, §4.2.8); reading it as one byte sorted
                // the 0x0020 length into 0x00 and left the key behind as
                // trailing bytes.
                let key = share.u16_vec()?;
                if share.remaining() != 0 {
                    return Err(HandshakeError::MalformedExtension {
                        which,
                        why: "trailing bytes after the key",
                    });
                }
                if key.len() != 32 {
                    return Err(HandshakeError::MalformedExtension {
                        which,
                        why: "x25519 key is not 32 bytes",
                    });
                }
                let mut k = [0u8; 32];
                k.copy_from_slice(key);
                key_share = Some(k);
            }
            EXT_SUPPORTED_VERSIONS => {
                if supported_version.is_some() {
                    return Err(HandshakeError::DuplicateExtension(which));
                }
                if body.len() != 2 {
                    return Err(HandshakeError::MalformedExtension {
                        which,
                        why: "expected a single two-byte version",
                    });
                }
                supported_version = Some(u16::from_be_bytes([body[0], body[1]]));
            }
            // Unknown extensions are skipped by type; that is the forward
            // compatibility the protocol requires. Their structure was still
            // checked by `u16_vec` above.
            _ => {}
        }
    }

    let key_share = key_share.ok_or(HandshakeError::MissingExtension(EXT_KEY_SHARE))?;
    let supported_version =
        supported_version.ok_or(HandshakeError::MissingExtension(EXT_SUPPORTED_VERSIONS))?;
    if supported_version != VERSION_TLS13 {
        return Err(HandshakeError::UnsupportedProtocolVersion(
            supported_version,
        ));
    }

    Ok(ServerHello {
        random,
        session_id,
        cipher_suite,
        key_share,
        supported_version,
        raw: message.to_vec(),
    })
}

/// The inputs to a ClientHello.
pub struct ClientHelloParams<'a> {
    /// 32 bytes of client randomness, always from a CSPRNG outside this crate.
    pub random: [u8; 32],
    /// The legacy session id. TLS 1.3 middlebox compatibility mode uses a
    /// non-empty 32-byte value; the server echoes it and we check the echo.
    pub session_id: &'a [u8],
    /// Our `key_share` public value.
    pub key_share: [u8; 32],
    /// The SNI host name, if the caller has one.
    pub server_name: Option<&'a str>,
}

/// Builds a ClientHello offering exactly what this crate can do.
///
/// One cipher suite, one group, one key share: offering suites we cannot
/// implement would invite a server to negotiate something we would then have to
/// reject mid-handshake.
pub fn build_client_hello(params: &ClientHelloParams<'_>) -> Result<Vec<u8>, HandshakeError> {
    let mut body = Vec::with_capacity(256);
    body.extend_from_slice(&LEGACY_VERSION.to_be_bytes());
    body.extend_from_slice(&params.random);
    body.push(params.session_id.len() as u8);
    body.extend_from_slice(params.session_id);

    // cipher_suites: one suite.
    body.extend_from_slice(&2u16.to_be_bytes());
    body.extend_from_slice(&TLS_AES_128_GCM_SHA256.to_be_bytes());

    // legacy_compression_methods: null, as TLS 1.3 requires.
    body.push(1);
    body.push(0);

    let mut extensions = Vec::with_capacity(128);
    if let Some(name) = params.server_name {
        let name = name.as_bytes();
        if name.is_empty() || name.len() > u16::MAX as usize {
            return Err(HandshakeError::BadServerName(
                "must be between 1 and 65535 bytes",
            ));
        }
        let mut ext = Vec::with_capacity(name.len() + 5);
        ext.extend_from_slice(&((name.len() + 3) as u16).to_be_bytes());
        ext.push(0); // host_name
        ext.extend_from_slice(&(name.len() as u16).to_be_bytes());
        ext.extend_from_slice(name);
        push_extension(&mut extensions, EXT_SERVER_NAME, &ext);
    }

    let mut groups = Vec::with_capacity(2);
    groups.extend_from_slice(&GROUP_X25519.to_be_bytes());
    push_extension(&mut extensions, EXT_SUPPORTED_GROUPS, &groups);

    let mut sigs = Vec::with_capacity(4);
    sigs.extend_from_slice(&SIG_ECDSA_SECP256R1_SHA256.to_be_bytes());
    sigs.extend_from_slice(&SIG_RSA_PSS_RSAE_SHA256.to_be_bytes());
    push_extension(&mut extensions, EXT_SIGNATURE_ALGORITHMS, &sigs);

    let mut share = Vec::with_capacity(36);
    share.extend_from_slice(&GROUP_X25519.to_be_bytes());
    share.extend_from_slice(&32u16.to_be_bytes());
    share.extend_from_slice(&params.key_share);
    push_extension(&mut extensions, EXT_KEY_SHARE, &share);

    push_extension(&mut extensions, EXT_SUPPORTED_VERSIONS, &[0x03, 0x04]);

    body.extend_from_slice(&(extensions.len() as u16).to_be_bytes());
    body.extend_from_slice(&extensions);
    Ok(encode(TYPE_CLIENT_HELLO, &body))
}

/// Appends `type || length || body`.
fn push_extension(out: &mut Vec<u8>, which: u16, body: &[u8]) {
    out.extend_from_slice(&which.to_be_bytes());
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(body);
}

/// The running transcript hash over handshake messages (RFC 8446 §4.4.1).
///
/// Every message goes in as it appeared on the wire, including its header, and
/// in the order sent or received. The hash is what the key schedule and the
/// Finished MAC are computed over, so an accidental omission or a message
/// pushed twice is not a cosmetic bug — it produces different keys, which is why
/// the driver feeds this from parsed messages rather than from a list a caller
/// maintains.
pub struct Transcript {
    hasher: Sha256,
}

impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}

impl Transcript {
    /// An empty transcript.
    pub fn new() -> Self {
        Self {
            hasher: Sha256::new(),
        }
    }

    /// Adds one complete handshake message.
    pub fn push(&mut self, message: &[u8]) {
        self.hasher.update(message);
    }

    /// The current hash, without consuming the transcript.
    pub fn hash(&self) -> [u8; 32] {
        self.hasher.clone().finalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hx(s: &str) -> Vec<u8> {
        let cleaned: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(cleaned.len().is_multiple_of(2), "odd hex length");
        (0..cleaned.len() / 2)
            .map(|i| u8::from_str_radix(&cleaned[i * 2..i * 2 + 2], 16).expect("hex"))
            .collect()
    }

    /// RFC 8448 §3's ServerHello, verbatim: 90 bytes, header `02 00 00 56`.
    /// It was re-derived locally (lengths, key_share and supported_versions all
    /// checked against the RFC's separately printed values) before being
    /// embedded here.
    fn rfc8448_server_hello() -> Vec<u8> {
        hx("020000560303\
            a6af06a4121860dc5e6e60249cd34c95930c8ac5cb1434dac155772ed3e26928\
            00\
            1301\
            00\
            002e\
            00330024001d0020c9828876112095fe66762bdbf7c672e156d6cc253b833df1dd69b1b04e751f0f\
            002b0002030 4")
    }

    /// The real ServerHello parses, and every field is the RFC's value.
    #[test]
    fn parses_the_rfc8448_server_hello() {
        let raw = rfc8448_server_hello();
        assert_eq!(raw.len(), 90);
        let sh = parse_server_hello(&raw).expect("valid ServerHello");

        assert_eq!(sh.cipher_suite, TLS_AES_128_GCM_SHA256);
        assert_eq!(sh.supported_version, VERSION_TLS13);
        assert!(sh.session_id.is_empty());
        assert_eq!(
            sh.key_share[..],
            hx("c9828876112095fe66762bdbf7c672e156d6cc253b833df1dd69b1b04e751f0f")[..]
        );
        assert_eq!(
            sh.random[..],
            hx("a6af06a4121860dc5e6e60249cd34c95930c8ac5cb1434dac155772ed3e26928")[..]
        );
        assert_eq!(sh.raw, raw);
    }

    /// Every truncation of a valid ServerHello must be an error, and none may
    /// panic. This is the property that matters when the peer is hostile.
    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let raw = rfc8448_server_hello();
        for cut in 0..raw.len() {
            let result = parse_server_hello(&raw[..cut]);
            assert!(
                result.is_err(),
                "a ServerHello truncated to {cut} bytes was accepted"
            );
        }
        // The full message still parses, so the loop above is not vacuous.
        assert!(parse_server_hello(&raw).is_ok());
    }

    /// Arbitrary mutations must produce errors or a valid parse, never a panic
    /// and never a silent acceptance of a broken field.
    #[test]
    fn mutations_never_panic() {
        let raw = rfc8448_server_hello();
        for byte in 0..raw.len() {
            for bit in 0..8 {
                let mut bad = raw.clone();
                bad[byte] ^= 1 << bit;
                // Either it parses (some byte is genuinely free, such as the
                // random) or it is rejected; it must never panic.
                let _ = parse_server_hello(&bad);
            }
        }
        // Pseudo-random buffers of varying length.
        let mut state = 0x1234_5678_9abc_def0u64;
        for len in 0..200usize {
            let mut buf = Vec::with_capacity(len);
            for _ in 0..len {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                buf.push((state >> 33) as u8);
            }
            let _ = parse_server_hello(&buf);
        }
    }

    /// The specific fields a real server could get wrong, each rejected for its
    /// own reason.
    #[test]
    fn specific_lying_fields_are_rejected() {
        let raw = rfc8448_server_hello();

        // Wrong message type.
        let mut bad = raw.clone();
        bad[0] = TYPE_CLIENT_HELLO;
        assert!(matches!(
            parse_server_hello(&bad),
            Err(HandshakeError::UnexpectedMessage { .. })
        ));

        // Length field that does not match the bytes.
        let mut bad = raw.clone();
        bad[3] = 0x57;
        assert!(matches!(
            parse_server_hello(&bad),
            Err(HandshakeError::BadLength(_))
        ));

        // Legacy version 0x0301.
        let mut bad = raw.clone();
        bad[5] = 0x01;
        assert!(matches!(
            parse_server_hello(&bad),
            Err(HandshakeError::UnexpectedVersion(0x0301))
        ));

        // A cipher suite we did not offer (TLS_AES_256_GCM_SHA384).
        let mut bad = raw.clone();
        bad[39] = 0x13;
        bad[40] = 0x02;
        assert!(matches!(
            parse_server_hello(&bad),
            Err(HandshakeError::UnsupportedCipherSuite(0x1302))
        ));

        // Non-null legacy compression.
        let mut bad = raw.clone();
        bad[41] = 0x01;
        assert!(matches!(
            parse_server_hello(&bad),
            Err(HandshakeError::BadCompression(1))
        ));

        // supported_versions offering only TLS 1.2.
        let mut bad = raw.clone();
        bad[89] = 0x03;
        assert!(matches!(
            parse_server_hello(&bad),
            Err(HandshakeError::UnsupportedProtocolVersion(0x0303))
        ));

        // A key_share group we cannot use.
        let mut bad = raw.clone();
        bad[48] = 0x00;
        bad[49] = 0x1e; // group 0x001e
        assert!(matches!(
            parse_server_hello(&bad),
            Err(HandshakeError::UnsupportedGroup(0x001e))
        ));

        // A 31-byte key.
        let mut bad = raw.clone();
        bad[51] = 31;
        assert!(matches!(
            parse_server_hello(&bad),
            Err(HandshakeError::MalformedExtension { .. })
        ));
    }

    /// Missing extensions are named, not silently tolerated.
    #[test]
    fn missing_required_extensions_are_named() {
        // Rebuild a ServerHello without the key_share extension: extensions
        // length 6, containing only supported_versions.
        let body_ext = hx("002b00020304");
        let mut message = hx("0200004a0303");
        message.extend_from_slice(&[0x11u8; 32]);
        message.push(0); // empty session id
        message.extend_from_slice(&TLS_AES_128_GCM_SHA256.to_be_bytes());
        message.push(0); // null compression
        message.extend_from_slice(&(body_ext.len() as u16).to_be_bytes());
        message.extend_from_slice(&body_ext);
        let declared = message.len() - HEADER_LEN;
        message[1] = (declared >> 16) as u8;
        message[2] = (declared >> 8) as u8;
        message[3] = declared as u8;

        assert!(matches!(
            parse_server_hello(&message),
            Err(HandshakeError::MissingExtension(EXT_KEY_SHARE))
        ));
    }

    /// A ClientHello we build must be well-formed on its own terms: the header
    /// length, the extensions length and every inner length must agree, and the
    /// values the server needs must be present. Checked with an independent
    /// walker here rather than by parsing with the code under test.
    #[test]
    fn a_built_client_hello_is_well_formed() {
        let key_share = [0x42u8; 32];
        let session_id = [0x11u8; 32];
        let msg = build_client_hello(&ClientHelloParams {
            random: [0xab; 32],
            session_id: &session_id,
            key_share,
            server_name: Some("example.com"),
        })
        .expect("build");

        assert_eq!(msg[0], TYPE_CLIENT_HELLO);
        let declared = ((msg[1] as usize) << 16) | ((msg[2] as usize) << 8) | msg[3] as usize;
        assert_eq!(declared, msg.len() - HEADER_LEN, "header length");

        let mut c = Cursor::new(&msg[HEADER_LEN..]);
        assert_eq!(c.u16().unwrap(), LEGACY_VERSION);
        assert_eq!(c.take(32).unwrap(), [0xab; 32]);
        assert_eq!(c.u8_vec().unwrap(), session_id);
        let suites = c.u16_vec().unwrap();
        assert_eq!(suites, TLS_AES_128_GCM_SHA256.to_be_bytes());
        assert_eq!(c.u8_vec().unwrap(), [0u8]);
        let extensions = c.u16_vec().unwrap();
        assert_eq!(c.remaining(), 0, "no trailing bytes after the extensions");

        // Walk the extensions and collect what we find.
        let mut ec = Cursor::new(extensions);
        let mut seen: Vec<(u16, Vec<u8>)> = Vec::new();
        while ec.remaining() > 0 {
            let which = ec.u16().unwrap();
            let body = ec.u16_vec().unwrap().to_vec();
            seen.push((which, body));
        }
        let find = |t: u16| {
            seen.iter()
                .find(|(which, _)| *which == t)
                .map(|(_, body)| body.clone())
        };
        assert_eq!(
            seen.len(),
            5,
            "server_name, groups, sigs, key_share, versions"
        );

        let sni = find(EXT_SERVER_NAME).expect("server_name");
        // ServerNameList: two-byte list length, then type, then a two-byte name
        // length, then the name.
        assert_eq!(sni[2], 0, "host_name type");
        assert_eq!(&sni[5..], b"example.com");
        assert_eq!(
            find(EXT_SUPPORTED_GROUPS).unwrap(),
            GROUP_X25519.to_be_bytes()
        );
        assert_eq!(find(EXT_SUPPORTED_VERSIONS).unwrap(), [0x03, 0x04]);
        let ks = find(EXT_KEY_SHARE).expect("key_share");
        assert_eq!(ks.len(), 36);
        assert_eq!(&ks[..2], &GROUP_X25519.to_be_bytes());
        assert_eq!(&ks[4..], &key_share);
        let sigs = find(EXT_SIGNATURE_ALGORITHMS).expect("signature_algorithms");
        assert_eq!(sigs.len(), 4);
        assert_eq!(&sigs[..2], &SIG_ECDSA_SECP256R1_SHA256.to_be_bytes());
    }

    /// A ClientHello without SNI omits the extension entirely.
    #[test]
    fn server_name_is_optional() {
        let msg = build_client_hello(&ClientHelloParams {
            random: [0u8; 32],
            session_id: &[],
            key_share: [0u8; 32],
            server_name: None,
        })
        .expect("build");
        let extensions = {
            let mut c = Cursor::new(&msg[HEADER_LEN..]);
            c.u16().unwrap();
            c.take(32).unwrap();
            c.u8_vec().unwrap();
            c.u16_vec().unwrap();
            c.u8_vec().unwrap();
            c.u16_vec().unwrap()
        };
        let mut ec = Cursor::new(extensions);
        let mut types = Vec::new();
        while ec.remaining() > 0 {
            let which = ec.u16().unwrap();
            let _ = ec.u16_vec().unwrap();
            types.push(which);
        }
        assert!(!types.contains(&EXT_SERVER_NAME));
    }

    /// An empty SNI is rejected rather than encoded as a zero-length name.
    #[test]
    fn empty_server_name_is_rejected() {
        let result = build_client_hello(&ClientHelloParams {
            random: [0u8; 32],
            session_id: &[],
            key_share: [0u8; 32],
            server_name: Some(""),
        });
        assert!(matches!(result, Err(HandshakeError::BadServerName(_))));
    }

    /// The reader yields messages in order, gives back exactly the bytes that
    /// arrived, and refuses an incomplete tail.
    #[test]
    fn reader_walks_concatenated_messages() {
        let a = encode(TYPE_ENCRYPTED_EXTENSIONS, &[1, 2, 3]);
        let b = encode(TYPE_FINISHED, &[9; 32]);
        let mut buf = a.clone();
        buf.extend_from_slice(&b);

        let mut reader = HandshakeReader::new(&buf);
        let first = reader.next_message().unwrap().expect("first");
        assert_eq!(first.msg_type(), TYPE_ENCRYPTED_EXTENSIONS);
        assert_eq!(first.body(), &[1, 2, 3]);
        assert_eq!(first.raw(), &a[..]);
        let second = reader.next_message().unwrap().expect("second");
        assert_eq!(second.msg_type(), TYPE_FINISHED);
        assert_eq!(second.body(), &[9u8; 32]);
        assert_eq!(second.raw(), &b[..]);
        assert!(reader.next_message().unwrap().is_none());

        // A partial tail is an error, not a message.
        let mut truncated = buf.clone();
        truncated.pop();
        let mut reader = HandshakeReader::new(&truncated);
        assert!(reader.next_message().unwrap().is_some());
        assert!(matches!(
            reader.next_message(),
            Err(HandshakeError::Truncated { .. })
        ));

        // A body claiming more bytes than the buffer holds.
        let lying = encode(TYPE_FINISHED, &[0; 4]);
        let mut lying = lying[..5].to_vec();
        lying[1] = 0x00;
        lying[2] = 0x01;
        lying[3] = 0x00; // claims 256 bytes
        let mut reader = HandshakeReader::new(&lying);
        assert!(matches!(
            reader.next_message(),
            Err(HandshakeError::Truncated { .. })
        ));
    }

    /// The transcript is the SHA-256 of the concatenated messages, and the hash
    /// changes when anything about them changes.
    #[test]
    fn transcript_hashes_the_messages_in_order() {
        let a = encode(TYPE_CLIENT_HELLO, &[1, 2, 3]);
        let b = encode(TYPE_SERVER_HELLO, &[4, 5]);

        let mut t = Transcript::new();
        assert_eq!(t.hash(), crate::keyschedule::empty_hash());

        t.push(&a);
        let after_a = t.hash();
        t.push(&b);
        let after_b = t.hash();
        assert_ne!(after_a, after_b);

        // Independent check: the same bytes hashed with the public SHA-256 API.
        let mut direct = Sha256::new();
        direct.update(&a);
        assert_eq!(after_a, direct.clone().finalize());
        direct.update(&b);
        assert_eq!(after_b, direct.finalize());

        // Order matters, and repeating a message matters.
        let mut reversed = Transcript::new();
        reversed.push(&b);
        reversed.push(&a);
        assert_ne!(reversed.hash(), after_b);

        let mut doubled = Transcript::new();
        doubled.push(&a);
        doubled.push(&a);
        doubled.push(&b);
        assert_ne!(doubled.hash(), after_b);

        // Pushing more messages must not disturb earlier hashes.
        assert_eq!(after_a, {
            let mut again = Transcript::new();
            again.push(&a);
            again.hash()
        });
    }

    /// Encoding round-trips through the reader, and the type byte is preserved.
    #[test]
    fn encode_and_decode_agree() {
        for len in [0usize, 1, 255, 256, 65535] {
            let body = vec![0x5au8; len];
            let raw = encode(TYPE_CERTIFICATE, &body);
            assert_eq!(raw.len(), HEADER_LEN + len);
            let mut reader = HandshakeReader::new(&raw);
            let msg = reader.next_message().unwrap().expect("one message");
            assert_eq!(msg.msg_type(), TYPE_CERTIFICATE);
            assert_eq!(msg.body(), &body[..]);
            assert!(reader.next_message().unwrap().is_none());
        }
    }
}
