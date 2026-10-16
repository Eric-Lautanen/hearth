//! The TLS 1.3 client state machine (RFC 8446 §4).
//!
//! This is a 1-RTT client for `TLS_AES_128_GCM_SHA256` over
//! [`GROUP_X25519`](crate::handshake::GROUP_X25519): it sends a ClientHello,
//! validates the server's ServerHello, derives the handshake keys, checks the
//! server's certificate through a caller-supplied
//! [`CertificateVerifier`], verifies the server's Finished MAC, and then sends
//! its own Finished and switches to the application keys.
//!
//! # No I/O, and no way to reach the network
//!
//! Nothing here touches a socket, a clock or a filesystem. The caller feeds raw
//! bytes in ([`ClientHandshake::feed`]) and writes the bytes it hands back
//! ([`Out::Send`]). That is what keeps the state machine testable against
//! RFC 8448's recorded flights, and it is why the transport, the CSPRNG and the
//! clock live outside this crate: the caller supplies the randomness and the
//! time ([`ClientConfig::validation_time`]), so there is no entropy source here
//! to be silently broken and no `SystemTime::now()` to make a validity check
//! depend on the day the test suite runs.
//!
//! # Certificate verification cannot be skipped
//!
//! A `ClientHandshake` cannot be constructed without a [`CertificateVerifier`],
//! and this crate ships exactly one implementation: [`RejectAll`], which can
//! only ever refuse. There is no "insecure" flag, no environment variable and
//! no default that accepts a chain — a caller that wants a usable connection has
//! to write, and therefore own, the verification step. The X.509 work itself is
//! a later slice; until it exists, this state machine can complete a handshake
//! only against a verifier that a test provided.
//!
//! **A verifier has to be a named type, and that is deliberate.** The trait is
//! not implemented for closures, so the shorthand that would turn an accept-all
//! into one line at any call site —
//! `ClientHandshake::new(config, |_| Ok(()))` — does not exist. A type named
//! `AcceptTestChain` in a test is visible in review; a closure that happens to
//! return `Ok` is not, and the difference matters more than the handful of lines
//! it costs. [`CertificateVerifyInput`] carries everything the decision needs,
//! including the name the ClientHello asked for, so a verifier never has to
//! capture state to be usable.
//!
//! That property is checkable, so it is checked rather than asserted in prose:
//! the block below must not compile.
//!
//! ```compile_fail
//! use hearth_tls::client::{CertificateVerifyInput, CertificateVerifier};
//!
//! struct RefusesToDecide;
//! impl CertificateVerifier for RefusesToDecide {
//!     fn verify_certificate(&self, _: &CertificateVerifyInput<'_>) -> Result<(), String> {
//!         Err("no".to_string())
//!     }
//! }
//! fn needs_a_verifier<V: CertificateVerifier>(_: V) {}
//!
//! needs_a_verifier(RefusesToDecide); // a named type is fine
//! // The shortest "accept everything" is a closure — and it is rejected:
//! needs_a_verifier(|_: &CertificateVerifyInput<'_>| -> Result<(), String> { Ok(()) });
//! ```
//!
//! # Every error is fatal
//!
//! TLS has no recoverable protocol error, so the first failure puts the client
//! into a failed state: `feed` returns [`ClientError::Failed`] from then on and
//! no keys are installed. In particular a Finished MAC mismatch ends the
//! connection — there is no branch that continues with unauthenticated bytes,
//! and [`verify_finished`] compares in
//! constant time.
//!
//! # What is not here
//!
//! Client certificates, `CertificateRequest` (a server asking for one is
//! rejected rather than ignored), session resumption and PSKs, 0-RTT early
//! data, ALPN, and post-handshake authentication. `KeyUpdate` *is* implemented,
//! in both directions. Extensions in `EncryptedExtensions` other than the empty
//! `server_name` acknowledgment are refused, as RFC 8446 §4.3.1 requires of a
//! client that did not offer them.

use std::time::SystemTime;

use crate::handshake::{
    self, HandshakeError, HandshakeReader, Message, ServerHello, Transcript,
    SIG_ECDSA_SECP256R1_SHA256, SIG_RSA_PSS_RSAE_SHA256, TYPE_CERTIFICATE,
    TYPE_CERTIFICATE_REQUEST, TYPE_CERTIFICATE_VERIFY, TYPE_ENCRYPTED_EXTENSIONS, TYPE_FINISHED,
    TYPE_KEY_UPDATE, TYPE_SERVER_HELLO,
};
use crate::hkdf::HASH_LEN;
use crate::keyschedule::{
    early_secret, finished_key, finished_verify_data, handshake_secret, master_secret,
    update_traffic_secret, verify_finished, TrafficSecrets, FINISHED_LEN,
};
use crate::record::{
    self, RecordError, RecordLayer, TrafficKeys, CONTENT_TYPE_ALERT, CONTENT_TYPE_APPLICATION_DATA,
    CONTENT_TYPE_CHANGE_CIPHER_SPEC, CONTENT_TYPE_HANDSHAKE, LEGACY_VERSION,
};
use crate::x25519;

/// The record header length, which is not a constant in `record` because it is
/// part of the wire format rather than a property of the cipher.
const RECORD_HEADER_LEN: usize = 5;

/// `change_cipher_spec`, the record type RFC 8446 §5 says to ignore.
const CONTENT_TYPE_CCS: u8 = 20;

/// Length of a `Hash` output for this suite; the size of a Finished MAC.
const HASH: usize = FINISHED_LEN;

/// Why the handshake stopped.
///
/// Every variant is fatal: see the module docs. The wrapped errors carry the
/// specific reason; the rest are what the state machine adds on top of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientError {
    /// A handshake message was malformed, or was not one we accept.
    Handshake(HandshakeError),
    /// A record failed to authenticate or was not framed as TLS requires.
    Record(RecordError),
    /// An HKDF derivation failed.
    KeySchedule(String),
    /// The server's `session_id` echo did not match the one we sent.
    SessionIdMismatch,
    /// A handshake message arrived where a different one was expected.
    UnexpectedMessage {
        /// What this state of the handshake required.
        expected: &'static str,
        /// The type byte that arrived instead.
        found: u8,
    },
    /// A record arrived with a content type that is not legal at this point.
    UnexpectedContentType(u8),
    /// Application data arrived before the handshake finished.
    UnexpectedAppData,
    /// The server sent a fatal alert, or a warning we do not implement.
    ServerAlert {
        /// The alert level byte.
        level: u8,
        /// The alert description byte.
        description: u8,
    },
    /// The connection is closing: `close_notify` has been sent and received.
    ConnectionClosed,
    /// `EncryptedExtensions` carried an extension we did not offer.
    UnsupportedExtension(u16),
    /// A message's internal structure was wrong.
    Malformed(&'static str),
    /// The `certificate_list` was empty, which is only legal with a PSK.
    EmptyCertificateChain,
    /// The server chose a signature algorithm we did not offer.
    UnsupportedSignatureScheme(u16),
    /// The caller's verifier refused the chain or the signature.
    VerifierRejected(String),
    /// The server's key share was a low-order point (RFC 7748 §6.1).
    LowOrderServerKey,
    /// This client never sent a ClientHello, so there is nothing to continue.
    NotStarted,
    /// [`ClientHandshake::start`] was called twice.
    AlreadyStarted,
    /// The handshake is not finished, so no application data may be written.
    NotConnected,
    /// A previous error already ended this connection.
    Failed,
    /// A feature this client does not implement was requested by the server.
    Unsupported(&'static str),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Handshake(e) => write!(f, "handshake message rejected: {e}"),
            Self::Record(e) => write!(f, "record rejected: {e}"),
            Self::KeySchedule(e) => write!(f, "key schedule failed: {e}"),
            Self::SessionIdMismatch => write!(f, "server did not echo our session id"),
            Self::UnexpectedMessage { expected, found } => {
                write!(f, "expected {expected}, got handshake type {found}")
            }
            Self::UnexpectedContentType(t) => write!(f, "unexpected record content type {t}"),
            Self::UnexpectedAppData => {
                write!(f, "application data arrived before the handshake finished")
            }
            Self::ServerAlert { level, description } => {
                write!(
                    f,
                    "server sent alert level {level}, description {description}"
                )
            }
            Self::ConnectionClosed => write!(f, "connection closed by close_notify"),
            Self::UnsupportedExtension(t) => {
                write!(f, "server sent extension {t:#06x}, which we did not offer")
            }
            Self::Malformed(why) => write!(f, "malformed message: {why}"),
            Self::EmptyCertificateChain => write!(f, "server sent an empty certificate chain"),
            Self::UnsupportedSignatureScheme(s) => {
                write!(f, "server signed with {s:#06x}, which we did not offer")
            }
            Self::VerifierRejected(why) => write!(f, "certificate verification failed: {why}"),
            Self::LowOrderServerKey => write!(f, "server key share is a low-order point"),
            Self::NotStarted => write!(f, "the ClientHello has not been sent"),
            Self::AlreadyStarted => write!(f, "the ClientHello was already sent"),
            Self::NotConnected => write!(f, "the handshake has not finished"),
            Self::Failed => write!(f, "the connection is dead: an earlier error ended it"),
            Self::Unsupported(what) => {
                write!(f, "server requested {what}, which is not implemented")
            }
        }
    }
}

impl std::error::Error for ClientError {}

impl From<HandshakeError> for ClientError {
    fn from(e: HandshakeError) -> Self {
        Self::Handshake(e)
    }
}

/// Every `String` error in this crate comes out of an HKDF derivation
/// (`hkdf_expand_label`, `TrafficKeys::from_secret`), which is why this maps
/// straight onto [`ClientError::KeySchedule`] rather than needing its own
/// variant.
impl From<String> for ClientError {
    fn from(e: String) -> Self {
        Self::KeySchedule(e)
    }
}

impl From<RecordError> for ClientError {
    fn from(e: RecordError) -> Self {
        Self::Record(e)
    }
}

impl From<x25519::SharedSecretError> for ClientError {
    fn from(_: x25519::SharedSecretError) -> Self {
        // The only failure `diffie_hellman` reports is the small-order point
        // check, which is exactly what must stop the handshake here.
        Self::LowOrderServerKey
    }
}

/// What a [`CertificateVerifier`] is given to make its decision.
///
/// The signature is checked by the caller because only the caller knows what it
/// trusts; this crate has no root store and no business having one. `signed_data`
/// is constructed by the state machine exactly as RFC 8446 §4.4.3 specifies —
/// 64 spaces, the context string, a zero byte, then the transcript hash through
/// the `Certificate` message — so a verifier never has to reimplement that and
/// cannot get it subtly wrong.
pub struct CertificateVerifyInput<'a> {
    /// The instant this connection is considered to begin, verbatim from
    /// [`ClientConfig::validation_time`].
    ///
    /// It is carried here rather than read by the verifier so that a validity
    /// check is a pure function of its inputs: a verifier that called
    /// `SystemTime::now()` itself could not be given an expired or
    /// not-yet-valid chain in a test without those tests expiring with the
    /// fixtures they use.
    pub validation_time: UnixTime,
    /// The `certificate_list` from the `Certificate` message: the concatenation
    /// of `opaque certificate<1..2^24-1>` entries, so each certificate is a
    /// three-byte big-endian length followed by DER. Verified to frame exactly
    /// the message before it reaches here.
    pub certificate_list: &'a [u8],
    /// The signature scheme from `CertificateVerify`, already checked to be one
    /// we offered.
    pub signature_scheme: u16,
    /// The signature bytes.
    pub signature: &'a [u8],
    /// Exactly the bytes the signature must cover.
    pub signed_data: &'a [u8],
    /// The SNI name the ClientHello asked for, if any. This is one of the two
    /// inputs a verifier cannot derive from the messages it is handed — it is in
    /// the ClientHello *we* sent, not in anything the server says — so it is
    /// carried here rather than expected to be captured in the verifier: the
    /// client already knows it, and a name checked against a captured copy is a
    /// name that can silently drift from the one on the wire.
    pub server_name: Option<&'a [u8]>,
}

/// Decides whether the server's certificate chain and its signature should be
/// trusted.
///
/// Returning `Err` is fatal to the handshake, and it must be: this is the one
/// decision in the protocol that is about *who* the peer is rather than whether
/// the bytes are intact.
///
/// Implement it on a named type: there is deliberately no `CertificateVerifier`
/// implementation for closures, so `|_| Ok(())` cannot be passed at a call site.
/// A closure is how an accept-all becomes an unremarkable one-liner in a diff,
/// and it also forces the verifier to capture things like the expected host name
/// that [`CertificateVerifyInput`] already provides.
pub trait CertificateVerifier {
    /// Checks the chain in `input.certificate_list` and the signature in
    /// `input.signature` over `input.signed_data`.
    ///
    /// The chain, the leaf's name for the server (`input.server_name`), the
    /// validity window against `input.validation_time`, and the signature itself
    /// are all the verifier's to check. Every one of those inputs is supplied by
    /// the handshake rather than captured by the verifier, so returning `Ok(())`
    /// claims exactly what the verifier was handed and nothing about state it
    /// kept on the side.
    fn verify_certificate(&self, input: &CertificateVerifyInput<'_>) -> Result<(), String>;
}

/// An instant on the wall clock, as whole seconds since the Unix epoch.
///
/// This is the currency of X.509 validity: `notBefore` and `notAfter` are
/// second-resolution times, so a verifier compares against seconds rather than
/// against a [`SystemTime`], whose conversion to seconds is fallible and whose
/// resolution is not what the comparison is defined in terms of.
///
/// Reading the clock is the caller's business, not this crate's: a
/// [`UnixTime`] reaches the state machine through
/// [`ClientConfig::validation_time`], so a test can ask for any instant in
/// either direction from now and expect the same answer forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnixTime {
    /// Seconds since 1970-01-01T00:00:00Z. Negative before the epoch, which the
    /// sign is kept for so that a subtraction against a validity bound cannot
    /// wrap.
    pub secs: i64,
}

impl UnixTime {
    /// An instant `secs` seconds after the epoch.
    pub const fn from_secs(secs: i64) -> Self {
        Self { secs }
    }

    /// 1970-01-01T00:00:00Z.
    pub const fn epoch() -> Self {
        Self { secs: 0 }
    }

    /// The instant a [`SystemTime`] denotes, saturating rather than wrapping on
    /// a clock set absurdly far from the epoch.
    pub fn from_system_time(time: SystemTime) -> Self {
        match time.duration_since(SystemTime::UNIX_EPOCH) {
            Ok(since) => Self {
                secs: i64::try_from(since.as_secs()).unwrap_or(i64::MAX),
            },
            Err(before) => Self {
                secs: i64::try_from(before.duration().as_secs())
                    .map_or(i64::MIN, |seconds| -seconds),
            },
        }
    }

    /// Whether `self` falls outside the inclusive `[not_before, not_after]`
    /// window, which is the check every X.509 verifier makes. `None` for
    /// `not_before` means "no lower bound".
    ///
    /// It lives here, next to the type, so that the one comparison the whole
    /// certificate slice turns on is written once and can be tested directly.
    pub fn outside(self, not_before: Option<Self>, not_after: Self) -> bool {
        not_before.is_some_and(|start| self < start) || self > not_after
    }
}

impl From<SystemTime> for UnixTime {
    fn from(time: SystemTime) -> Self {
        Self::from_system_time(time)
    }
}

impl std::fmt::Display for UnixTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}s since the Unix epoch", self.secs)
    }
}

/// The only verifier this crate provides, and it can only refuse.
///
/// It exists so a caller can be explicit about not verifying yet (a test, or a
/// benchmark of the transport) and be *visible* about it. Nothing here can be
/// convinced to accept a chain, which is why there is no corresponding
/// `AcceptAll`.
pub struct RejectAll;

impl CertificateVerifier for RejectAll {
    fn verify_certificate(&self, _input: &CertificateVerifyInput<'_>) -> Result<(), String> {
        Err("no certificate verifier is installed (RejectAll rejects everything)".to_string())
    }
}

/// The inputs the caller owns: randomness, time, and the name to ask for.
pub struct ClientConfig<'a> {
    /// The SNI host name. `None` sends no `server_name` extension.
    pub server_name: Option<&'a str>,
    /// The instant this connection is considered to begin, which is the "now"
    /// a [`CertificateVerifier`] checks the chain's validity window against. It
    /// reaches the verifier as [`CertificateVerifyInput::validation_time`].
    ///
    /// Reading the clock is the caller's job — `SystemTime::now().into()` at the
    /// point the connection is opened — and that keeps the state machine free of
    /// a clock exactly as it is free of a socket and a CSPRNG. A connection that
    /// takes a week or a year would want a fresher reading, but a TLS handshake
    /// is seconds; the same instant is what the transcript and the keys are
    /// bound to.
    pub validation_time: UnixTime,
    /// 32 bytes from a CSPRNG, the `Random` in the ClientHello.
    pub random: [u8; 32],
    /// The legacy session id. A non-empty value puts us in middlebox
    /// compatibility mode (RFC 8446 §D.4), which also sends a dummy
    /// `change_cipher_spec` after the ClientHello; 32 bytes is what real clients
    /// send. At most 32 bytes.
    pub session_id: Vec<u8>,
    /// Our X25519 private key for this connection. Must come from a CSPRNG, must
    /// be fresh per connection, and must be discarded when the handshake ends.
    pub private_key: [u8; 32],
}

/// Something the caller must act on.
///
/// Every `Send` is already a complete record: the caller writes the bytes to the
/// socket as they are. Events appear in the order the server's bytes produced
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Out {
    /// Bytes to write to the transport.
    Send(Vec<u8>),
    /// The handshake finished and application keys are installed. Anything
    /// received after this point arrives as [`Out::Application`], and
    /// [`ClientHandshake::write`] may be used for the other direction.
    Connected,
    /// Decrypted application data from the peer.
    Application(Vec<u8>),
    /// The peer sent `close_notify`. The transport can be closed once our own
    /// `close_notify` has been sent.
    Closed,
}

/// Where the handshake is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Start,
    WaitServerHello,
    WaitEncryptedExtensions,
    WaitCertificate,
    WaitCertificateVerify,
    WaitFinished,
    Connected,
    Closed,
    Failed,
}

impl Phase {
    /// The next message this state must receive, for error messages and the
    /// check itself.
    fn expected(self) -> Option<(&'static str, u8)> {
        match self {
            Self::WaitServerHello => Some(("ServerHello", TYPE_SERVER_HELLO)),
            Self::WaitEncryptedExtensions => {
                Some(("EncryptedExtensions", TYPE_ENCRYPTED_EXTENSIONS))
            }
            Self::WaitCertificate => Some(("Certificate", TYPE_CERTIFICATE)),
            Self::WaitCertificateVerify => Some(("CertificateVerify", TYPE_CERTIFICATE_VERIFY)),
            Self::WaitFinished => Some(("Finished", TYPE_FINISHED)),
            _ => None,
        }
    }
}

/// A TLS 1.3 client handshake.
///
/// See the module documentation for the shape of a session: `start`, then
/// `feed` until [`Out::Connected`], then `write`/`feed`.
pub struct ClientHandshake<V> {
    server_name: Option<String>,
    /// The caller's clock reading, handed to the verifier untouched.
    validation_time: UnixTime,
    session_id: Vec<u8>,
    private_key: [u8; 32],
    random: [u8; 32],
    verifier: V,
    phase: Phase,
    started: bool,
    /// Raw bytes received, not yet framed.
    inbuf: Vec<u8>,
    /// Decrypted handshake message bytes, which may hold a partial message: a
    /// handshake message is not required to fit in one record, so a truncated
    /// tail here means "wait for more records" rather than "malformed".
    hs_pending: Vec<u8>,
    transcript: Transcript,
    /// The Handshake Secret, which the master secret and the application
    /// secrets are derived from.
    hs_secret: [u8; HASH_LEN],
    /// The client handshake traffic secret, needed for our own Finished and kept
    /// until then.
    client_hs_secret: [u8; HASH_LEN],
    /// The server handshake traffic secret, needed for its Finished.
    server_hs_secret: [u8; HASH_LEN],
    /// The application traffic secrets, kept so a `KeyUpdate` can rotate them.
    client_app_secret: [u8; HASH_LEN],
    server_app_secret: [u8; HASH_LEN],
    /// The certificate chain from the Certificate message, held until
    /// CertificateVerify arrives to be verified with it.
    certificate_list: Vec<u8>,
    send_hs: Option<RecordLayer>,
    recv_hs: Option<RecordLayer>,
    send_app: Option<RecordLayer>,
    recv_app: Option<RecordLayer>,
    /// Sent and received `close_notify` flags, so the caller can tell when the
    /// connection is finished in both directions.
    sent_close: bool,
    got_close: bool,
}

impl<V: CertificateVerifier> ClientHandshake<V> {
    /// Creates a client, validating the configuration. Nothing is sent yet.
    pub fn new(config: ClientConfig<'_>, verifier: V) -> Result<Self, ClientError> {
        if config.session_id.len() > 32 {
            return Err(ClientError::Malformed("session id longer than 32 bytes"));
        }
        if let Some(name) = config.server_name {
            if name.is_empty() || name.len() > u16::MAX as usize {
                return Err(ClientError::Malformed(
                    "server name must be between 1 and 65535 bytes",
                ));
            }
        }
        Ok(Self {
            server_name: config.server_name.map(str::to_string),
            validation_time: config.validation_time,
            session_id: config.session_id,
            private_key: config.private_key,
            random: config.random,
            verifier,
            phase: Phase::Start,
            started: false,
            inbuf: Vec::new(),
            hs_pending: Vec::new(),
            transcript: Transcript::new(),
            hs_secret: [0u8; HASH_LEN],
            client_hs_secret: [0u8; HASH_LEN],
            server_hs_secret: [0u8; HASH_LEN],
            client_app_secret: [0u8; HASH_LEN],
            server_app_secret: [0u8; HASH_LEN],
            certificate_list: Vec::new(),
            send_hs: None,
            recv_hs: None,
            send_app: None,
            recv_app: None,
            sent_close: false,
            got_close: false,
        })
    }

    /// The phase, for diagnostics.
    pub fn is_connected(&self) -> bool {
        self.phase == Phase::Connected
    }

    /// Whether the record layers ended up on the hardware AES/GHASH paths.
    /// `None` before any keys exist.
    pub fn accelerated(&self) -> Option<(bool, bool)> {
        self.send_app
            .as_ref()
            .or(self.send_hs.as_ref())
            .map(RecordLayer::accelerated)
    }

    /// Builds and returns the first flight: the ClientHello record, and the
    /// dummy `change_cipher_spec` when a session id was sent.
    ///
    /// The ClientHello goes into the transcript here, before any byte of the
    /// server's reply is processed, because the transcript is defined over the
    /// messages as sent.
    pub fn start(&mut self) -> Result<Vec<u8>, ClientError> {
        if self.started {
            return Err(ClientError::AlreadyStarted);
        }
        let key_share = x25519::public_key(&self.private_key);
        let client_hello = handshake::build_client_hello(&handshake::ClientHelloParams {
            random: self.random,
            session_id: &self.session_id,
            key_share,
            server_name: self.server_name.as_deref(),
        })?;
        self.started = true;
        self.phase = Phase::WaitServerHello;
        self.transcript.push(&client_hello);

        let mut out = plaintext_record(CONTENT_TYPE_HANDSHAKE, &client_hello);
        if !self.session_id.is_empty() {
            // RFC 8446 §D.4: middlebox compatibility mode. Not a handshake
            // message, so it never enters the transcript.
            out.extend_from_slice(&plaintext_record(CONTENT_TYPE_CHANGE_CIPHER_SPEC, &[1]));
        }
        Ok(out)
    }

    /// Feeds raw bytes received from the server and returns what to do next.
    ///
    /// Partial records and partial handshake messages are buffered, so the
    /// caller can pass whatever a `read` returned. There is no bound on the
    /// buffer's growth beyond what the caller's own reads produce.
    pub fn feed(&mut self, data: &[u8]) -> Result<Vec<Out>, ClientError> {
        if self.phase == Phase::Failed {
            return Err(ClientError::Failed);
        }
        if !self.started {
            return Err(ClientError::NotStarted);
        }
        self.inbuf.extend_from_slice(data);
        match self.pump() {
            Ok(out) => Ok(out),
            Err(e) => {
                // Fatal, and sticky: a connection that produced one of these has
                // no valid keys and must not be used for anything else.
                self.phase = Phase::Failed;
                Err(e)
            }
        }
    }

    /// Encrypts application data. Only valid once [`Out::Connected`] arrived.
    pub fn write(&mut self, data: &[u8]) -> Result<Vec<u8>, ClientError> {
        if self.phase == Phase::Failed {
            return Err(ClientError::Failed);
        }
        if self.sent_close {
            return Err(ClientError::ConnectionClosed);
        }
        let layer = self.send_app.as_mut().ok_or(ClientError::NotConnected)?;
        let record = layer.seal(CONTENT_TYPE_APPLICATION_DATA, data)?;
        Ok(record)
    }

    /// Sends `close_notify` and returns the record bytes.
    ///
    /// The connection can be closed after this even if the peer never answers;
    /// without it, a truncation attack is indistinguishable from an orderly end
    /// of stream (RFC 8446 §6.1).
    pub fn close_notify(&mut self) -> Result<Vec<u8>, ClientError> {
        if self.phase == Phase::Failed {
            return Err(ClientError::Failed);
        }
        let layer = self
            .send_app
            .as_mut()
            .or(self.send_hs.as_mut())
            .ok_or(ClientError::NotConnected)?;
        let record = layer.seal(CONTENT_TYPE_ALERT, &[1, 0])?;
        self.sent_close = true;
        Ok(record)
    }

    /// Consumes one record and acts on it.
    fn handle_record(
        &mut self,
        outer_type: u8,
        record: &[u8],
        out: &mut Vec<Out>,
    ) -> Result<(), ClientError> {
        let body = &record[RECORD_HEADER_LEN..];
        match outer_type {
            // RFC 8446 §5: an unencrypted change_cipher_spec may arrive at any
            // time in compatibility mode and must be ignored. The transcript
            // does not see it.
            CONTENT_TYPE_CCS => {
                if body != [1] {
                    return Err(ClientError::Malformed(
                        "change_cipher_spec body is not a single 0x01",
                    ));
                }
                Ok(())
            }
            CONTENT_TYPE_ALERT => {
                if self.phase != Phase::WaitServerHello {
                    // Past the ServerHello everything is encrypted, so a
                    // plaintext alert is a protocol violation rather than a
                    // reason to close.
                    return Err(ClientError::UnexpectedContentType(CONTENT_TYPE_ALERT));
                }
                self.handle_alert(body, out)
            }
            CONTENT_TYPE_HANDSHAKE => {
                if self.phase != Phase::WaitServerHello {
                    return Err(ClientError::UnexpectedContentType(CONTENT_TYPE_HANDSHAKE));
                }
                let mut reader = HandshakeReader::new(body);
                let msg = reader
                    .next_message()?
                    .ok_or(ClientError::Malformed("empty handshake record"))?;
                if reader.next_message()?.is_some() {
                    // Nothing else may be sent in the clear: the flight after the
                    // ServerHello is encrypted, and there is no other plaintext
                    // handshake message in TLS 1.3 for a client to accept.
                    return Err(ClientError::Malformed(
                        "more than one message in the plaintext handshake record",
                    ));
                }
                self.on_message(msg, out)
            }
            CONTENT_TYPE_APPLICATION_DATA => {
                let layer = self.recv_layer()?;
                let opened = layer.open(record)?;
                match opened.content_type {
                    CONTENT_TYPE_HANDSHAKE => {
                        self.hs_pending.extend_from_slice(&opened.payload);
                        self.consume_handshake(out)
                    }
                    CONTENT_TYPE_ALERT => self.handle_alert(&opened.payload, out),
                    CONTENT_TYPE_APPLICATION_DATA => {
                        if self.phase != Phase::Connected {
                            return Err(ClientError::UnexpectedAppData);
                        }
                        out.push(Out::Application(opened.payload));
                        Ok(())
                    }
                    other => Err(ClientError::UnexpectedContentType(other)),
                }
            }
            other => Err(ClientError::UnexpectedContentType(other)),
        }
    }

    /// The layer that decrypts whatever the server sends next.
    fn recv_layer(&mut self) -> Result<&mut RecordLayer, ClientError> {
        self.recv_app
            .as_mut()
            .or(self.recv_hs.as_mut())
            .ok_or(ClientError::NotStarted)
    }

    /// Parses every complete handshake message out of `hs_pending`, keeping a
    /// partial tail for the next record.
    fn consume_handshake(&mut self, out: &mut Vec<Out>) -> Result<(), ClientError> {
        loop {
            if self.hs_pending.is_empty() {
                return Ok(());
            }
            let parsed = HandshakeReader::new(&self.hs_pending).next_message();
            match parsed {
                Ok(Some(msg)) => {
                    let consumed = msg.raw().len();
                    self.hs_pending.drain(..consumed);
                    self.on_message(msg, out)?;
                }
                // A message split across records: wait for the rest. The bound
                // on the wait is the caller's read loop, not this parser.
                Err(HandshakeError::Truncated { .. }) => return Ok(()),
                Ok(None) => return Ok(()),
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Handles one complete handshake message, in whatever state we are in.
    fn on_message(&mut self, msg: Message, out: &mut Vec<Out>) -> Result<(), ClientError> {
        match self.phase {
            Phase::WaitServerHello => {
                self.expect(TYPE_SERVER_HELLO, msg.msg_type())?;
                self.on_server_hello(&msg)
            }
            Phase::WaitEncryptedExtensions => {
                self.expect(TYPE_ENCRYPTED_EXTENSIONS, msg.msg_type())?;
                self.transcript.push(msg.raw());
                self.check_encrypted_extensions(msg.body())?;
                self.phase = Phase::WaitCertificate;
                Ok(())
            }
            Phase::WaitCertificate => {
                match msg.msg_type() {
                    TYPE_CERTIFICATE => {
                        self.transcript.push(msg.raw());
                        self.certificate_list = parse_certificate(msg.body())?;
                        self.phase = Phase::WaitCertificateVerify;
                        Ok(())
                    }
                    // A server that wants to authenticate *us* is a feature we do
                    // not implement; continuing as if it had not asked would be
                    // worse than stopping.
                    TYPE_CERTIFICATE_REQUEST => {
                        Err(ClientError::Unsupported("client authentication"))
                    }
                    found => Err(ClientError::UnexpectedMessage {
                        expected: "Certificate",
                        found,
                    }),
                }
            }
            Phase::WaitCertificateVerify => {
                self.expect(TYPE_CERTIFICATE_VERIFY, msg.msg_type())?;
                self.on_certificate_verify(&msg)?;
                self.phase = Phase::WaitFinished;
                Ok(())
            }
            Phase::WaitFinished => {
                self.expect(TYPE_FINISHED, msg.msg_type())?;
                self.on_server_finished(&msg, out)
            }
            Phase::Connected => {
                if msg.msg_type() == TYPE_KEY_UPDATE {
                    self.on_key_update(&msg, out)
                } else {
                    // Everything else in the post-handshake phase
                    // (NewSessionTicket, CertificateRequest, ...) is either
                    // irrelevant to us or unimplemented, and either way must not
                    // be silently swallowed.
                    Err(ClientError::Unsupported("a post-handshake message"))
                }
            }
            Phase::Start | Phase::Closed | Phase::Failed => Err(ClientError::UnexpectedMessage {
                expected: "nothing",
                found: msg.msg_type(),
            }),
        }
    }

    /// Checks the type byte for the current phase.
    fn expect(&self, want: u8, found: u8) -> Result<(), ClientError> {
        if found == want {
            return Ok(());
        }
        let expected = self
            .phase
            .expected()
            .map(|(name, _)| name)
            .unwrap_or("no message");
        Err(ClientError::UnexpectedMessage { expected, found })
    }

    /// Validates and consumes the ServerHello, then derives the handshake keys.
    fn on_server_hello(&mut self, msg: &Message) -> Result<(), ClientError> {
        let sh: ServerHello = handshake::parse_server_hello(msg.raw())?;
        if sh.session_id != self.session_id {
            return Err(ClientError::SessionIdMismatch);
        }
        // The (EC)DHE secret. `diffie_hellman` enforces RFC 7748 §6.1's
        // all-zero-output check, so a server that sends a low-order point is
        // rejected here rather than contributing a known secret.
        let ecdhe = x25519::diffie_hellman(&self.private_key, &sh.key_share)?;

        self.transcript.push(msg.raw());
        let early = early_secret(None);
        self.hs_secret = handshake_secret(&early, &ecdhe).map_err(ClientError::KeySchedule)?;
        let secrets = self.handshake_secrets(&self.transcript.hash())?;
        self.client_hs_secret = secrets.client;
        self.server_hs_secret = secrets.server;
        self.send_hs = Some(RecordLayer::new(&TrafficKeys::from_secret(
            &self.client_hs_secret,
        )?));
        self.recv_hs = Some(RecordLayer::new(&TrafficKeys::from_secret(
            &self.server_hs_secret,
        )?));
        self.phase = Phase::WaitEncryptedExtensions;
        Ok(())
    }

    /// The handshake traffic secrets for a given transcript hash.
    fn handshake_secrets(
        &self,
        transcript_hash: &[u8; HASH_LEN],
    ) -> Result<TrafficSecrets, ClientError> {
        TrafficSecrets::derive(&self.hs_secret, "hs", transcript_hash)
            .map_err(ClientError::KeySchedule)
    }

    /// `EncryptedExtensions` must carry nothing we did not offer.
    fn check_encrypted_extensions(&self, body: &[u8]) -> Result<(), ClientError> {
        if body.len() < 2 {
            return Err(ClientError::Malformed("EncryptedExtensions has no length"));
        }
        let declared = u16::from_be_bytes([body[0], body[1]]) as usize;
        if declared != body.len() - 2 {
            return Err(ClientError::Malformed(
                "EncryptedExtensions length disagrees with the message",
            ));
        }
        let mut rest = &body[2..];
        while !rest.is_empty() {
            if rest.len() < 4 {
                return Err(ClientError::Malformed("truncated extension header"));
            }
            let which = u16::from_be_bytes([rest[0], rest[1]]);
            let len = u16::from_be_bytes([rest[2], rest[3]]) as usize;
            if rest.len() < 4 + len {
                return Err(ClientError::Malformed("truncated extension body"));
            }
            let value = &rest[4..4 + len];
            // The one extension a server may echo at us is the empty
            // `server_name` acknowledgment, and only because we sent SNI.
            let is_sni_ack =
                which == EXT_SERVER_NAME && value.is_empty() && self.server_name.is_some();
            if !is_sni_ack {
                return Err(ClientError::UnsupportedExtension(which));
            }
            rest = &rest[4 + len..];
        }
        Ok(())
    }

    /// Verifies the server's signature over the handshake transcript.
    fn on_certificate_verify(&mut self, msg: &Message) -> Result<(), ClientError> {
        let body = msg.body();
        if body.len() < 4 {
            return Err(ClientError::Malformed("CertificateVerify is too short"));
        }
        let scheme = u16::from_be_bytes([body[0], body[1]]);
        let sig_len = u16::from_be_bytes([body[2], body[3]]) as usize;
        if body.len() != 4 + sig_len {
            return Err(ClientError::Malformed(
                "CertificateVerify signature length disagrees with the message",
            ));
        }
        let signature = &body[4..];
        if scheme != SIG_ECDSA_SECP256R1_SHA256 && scheme != SIG_RSA_PSS_RSAE_SHA256 {
            return Err(ClientError::UnsupportedSignatureScheme(scheme));
        }

        // RFC 8446 §4.4.3: the signed content is 64 spaces, the context string,
        // a zero byte, then the transcript hash *through Certificate* — this
        // message is not in it.
        let mut signed_data = Vec::with_capacity(64 + 33 + 1 + HASH);
        signed_data.extend_from_slice(&[0x20; 64]);
        signed_data.extend_from_slice(b"TLS 1.3, server CertificateVerify");
        signed_data.push(0);
        signed_data.extend_from_slice(&self.transcript.hash());

        let input = CertificateVerifyInput {
            certificate_list: &self.certificate_list,
            signature_scheme: scheme,
            signature,
            signed_data: &signed_data,
            validation_time: self.validation_time,
            server_name: self.server_name.as_deref().map(str::as_bytes),
        };
        self.verifier
            .verify_certificate(&input)
            .map_err(ClientError::VerifierRejected)?;
        self.transcript.push(msg.raw());
        Ok(())
    }

    /// Verifies the server's Finished MAC and, on success, finishes our side.
    fn on_server_finished(&mut self, msg: &Message, out: &mut Vec<Out>) -> Result<(), ClientError> {
        if msg.body().len() != HASH {
            return Err(ClientError::Malformed("Finished is not one hash long"));
        }
        let key = finished_key(&self.server_hs_secret).map_err(ClientError::KeySchedule)?;
        verify_finished(&key, &self.transcript.hash(), msg.body())
            .map_err(ClientError::KeySchedule)?;
        self.transcript.push(msg.raw());

        // Application secrets are derived from the transcript through the
        // server's Finished, and our own Finished is computed over that same
        // hash (RFC 8446 §4.4.4 and §7.1).
        self.finish_from_transcript(out)
    }

    /// Derives the application secrets, sends our Finished, and switches keys.
    fn finish_from_transcript(&mut self, out: &mut Vec<Out>) -> Result<(), ClientError> {
        let transcript_hash = self.transcript.hash();
        let master = master_secret(&self.hs_secret).map_err(ClientError::KeySchedule)?;
        let secrets = TrafficSecrets::derive(&master, "ap", &transcript_hash)
            .map_err(ClientError::KeySchedule)?;
        self.client_app_secret = secrets.client;
        self.server_app_secret = secrets.server;

        // Our Finished uses the *handshake* client secret and the transcript
        // through the server's Finished.
        let key = finished_key(&self.client_hs_secret).map_err(ClientError::KeySchedule)?;
        let verify_data = finished_verify_data(&key, &transcript_hash);
        let message = handshake::encode(TYPE_FINISHED, &verify_data);
        self.transcript.push(&message);

        let send_hs = self.send_hs.as_mut().ok_or(ClientError::NotStarted)?;
        let record = send_hs.seal(CONTENT_TYPE_HANDSHAKE, &message)?;
        out.push(Out::Send(record));

        self.send_app = Some(RecordLayer::new(&TrafficKeys::from_secret(
            &self.client_app_secret,
        )?));
        self.recv_app = Some(RecordLayer::new(&TrafficKeys::from_secret(
            &self.server_app_secret,
        )?));
        self.phase = Phase::Connected;
        out.push(Out::Connected);
        Ok(())
    }

    /// Handles a `KeyUpdate`: the peer's sending keys are rekeyed immediately,
    /// and a request means ours must follow.
    fn on_key_update(&mut self, msg: &Message, out: &mut Vec<Out>) -> Result<(), ClientError> {
        if msg.body().len() != 1 {
            return Err(ClientError::Malformed("KeyUpdate is not one byte"));
        }
        let request_update = match msg.body()[0] {
            0 => false,
            1 => true,
            other => {
                return Err(ClientError::Malformed(if other == 255 {
                    "update_not_requested/update_requested are 0 and 1"
                } else {
                    "KeyUpdate request byte is not 0 or 1"
                }))
            }
        };
        self.transcript.push(msg.raw());
        // The record carrying this message was sealed with the old keys; the
        // new keys apply from the next record. Installing them here is exactly
        // that boundary.
        self.server_app_secret =
            update_traffic_secret(&self.server_app_secret).map_err(ClientError::KeySchedule)?;
        self.recv_app = Some(RecordLayer::new(&TrafficKeys::from_secret(
            &self.server_app_secret,
        )?));
        if request_update {
            let record = self.send_key_update(false)?;
            out.push(Out::Send(record));
        }
        Ok(())
    }

    /// Sends a `KeyUpdate`, rekeys this direction's sending keys, and returns the
    /// record to write.
    ///
    /// The new keys apply from the *next* record: the returned record is sealed
    /// with the old ones, which is what makes the transition unambiguous on the
    /// wire.
    pub fn key_update(&mut self) -> Result<Vec<u8>, ClientError> {
        if self.phase == Phase::Failed {
            return Err(ClientError::Failed);
        }
        if self.phase != Phase::Connected {
            return Err(ClientError::NotConnected);
        }
        self.send_key_update(true)
    }

    /// The one path that emits a `KeyUpdate`, so the rekey cannot be half-done.
    fn send_key_update(&mut self, request_update: bool) -> Result<Vec<u8>, ClientError> {
        let message = handshake::encode(TYPE_KEY_UPDATE, &[u8::from(request_update)]);
        let layer = self.send_app.as_mut().ok_or(ClientError::NotConnected)?;
        let record = layer.seal(CONTENT_TYPE_HANDSHAKE, &message)?;
        self.transcript.push(&message);
        self.client_app_secret =
            update_traffic_secret(&self.client_app_secret).map_err(ClientError::KeySchedule)?;
        self.send_app = Some(RecordLayer::new(&TrafficKeys::from_secret(
            &self.client_app_secret,
        )?));
        Ok(record)
    }

    /// Handles a decrypted alert. `close_notify` is an orderly shutdown; every
    /// other description ends the connection with an error.
    fn handle_alert(&mut self, body: &[u8], out: &mut Vec<Out>) -> Result<(), ClientError> {
        if body.len() != 2 {
            return Err(ClientError::Malformed("alert is not two bytes"));
        }
        let (level, description) = (body[0], body[1]);
        if description == 0 {
            self.got_close = true;
            self.phase = Phase::Closed;
            out.push(Out::Closed);
            return Ok(());
        }
        Err(ClientError::ServerAlert { level, description })
    }

    /// Frames and dispatches every complete record in `inbuf`.
    fn pump(&mut self) -> Result<Vec<Out>, ClientError> {
        let mut out = Vec::new();
        loop {
            if self.phase == Phase::Closed {
                break;
            }
            if self.inbuf.len() < RECORD_HEADER_LEN {
                break;
            }
            let outer_type = self.inbuf[0];
            let version = [self.inbuf[1], self.inbuf[2]];
            if version != LEGACY_VERSION {
                return Err(ClientError::Record(RecordError::UnexpectedVersion(version)));
            }
            let len = u16::from_be_bytes([self.inbuf[3], self.inbuf[4]]) as usize;
            if len > record::MAX_CIPHERTEXT {
                return Err(ClientError::Record(RecordError::RecordTooLong(len)));
            }
            if self.inbuf.len() < RECORD_HEADER_LEN + len {
                break; // a partial record; wait for the rest
            }
            let record: Vec<u8> = self.inbuf.drain(..RECORD_HEADER_LEN + len).collect();
            self.handle_record(outer_type, &record, &mut out)?;
        }
        Ok(out)
    }
}

/// The `server_name` extension type, used only to recognise the empty
/// acknowledgment in `EncryptedExtensions`.
const EXT_SERVER_NAME: u16 = 0x0000;

/// Builds one unprotected record.
fn plaintext_record(content_type: u8, body: &[u8]) -> Vec<u8> {
    let len = body.len() as u16;
    let mut out = Vec::with_capacity(RECORD_HEADER_LEN + body.len());
    out.push(content_type);
    out.extend_from_slice(&LEGACY_VERSION);
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(body);
    out
}

/// Validates the `Certificate` message's framing and returns its
/// `certificate_list`.
///
/// Strict on purpose: the list is a sequence of length-prefixed DER blobs, so a
/// list length that does not equal the sum of its entries means the message is
/// not what it claims to be. It is also worth knowing that RFC 8448's own
/// example message fails this check — it declares a 437-byte list around a
/// single 432-byte entry, leaving two bytes unaccounted for — which is why the
/// fixture test drives that trace through the record and transcript layer
/// rather than through here. Real servers emit a consistent list.
fn parse_certificate(body: &[u8]) -> Result<Vec<u8>, ClientError> {
    if body.is_empty() {
        return Err(ClientError::Malformed("Certificate has no context length"));
    }
    let ctx_len = body[0] as usize;
    if body.len() < 1 + ctx_len + 3 {
        return Err(ClientError::Malformed("Certificate is truncated"));
    }
    let context = &body[1..1 + ctx_len];
    if !context.is_empty() {
        // A non-empty context only appears in response to a CertificateRequest,
        // which we reject before reaching here.
        return Err(ClientError::Unsupported("client authentication"));
    }
    let list_start = 1 + ctx_len;
    let list_len = u32::from_be_bytes([
        0,
        body[list_start],
        body[list_start + 1],
        body[list_start + 2],
    ]) as usize;
    let list = &body[list_start + 3..];
    if list.len() != list_len {
        return Err(ClientError::Malformed(
            "certificate_list length disagrees with the message",
        ));
    }
    if list.is_empty() {
        return Err(ClientError::EmptyCertificateChain);
    }
    // Every entry is `opaque certificate<1..2^24-1>`, so a lone three-byte
    // header with no DER behind it is not an entry. The walk is `chain.rs`'s,
    // because the verifier applies the same rule to the same bytes and two copies
    // of a framing rule are two chances to disagree.
    match crate::chain::split_entries(list) {
        Ok(_) => Ok(list.to_vec()),
        Err(crate::chain::ListError::Empty) => Err(ClientError::EmptyCertificateChain),
        Err(crate::chain::ListError::EndsInsideLength { .. }) => Err(ClientError::Malformed(
            "certificate_list ends inside an entry's length",
        )),
        Err(crate::chain::ListError::ZeroLengthEntry { .. }) => Err(ClientError::Malformed(
            "certificate_list entry is malformed",
        )),
        Err(crate::chain::ListError::TruncatedEntry { .. }) => Err(ClientError::Malformed(
            "certificate_list entry is malformed",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handshake::{
        encode, HEADER_LEN as HS_HEADER, TYPE_CERTIFICATE_REQUEST, TYPE_CLIENT_HELLO,
        TYPE_NEW_SESSION_TICKET,
    };
    use crate::sha256::{hmac_sha256, Sha256};
    use std::sync::{Arc, Mutex};

    const CLIENT_PRIV: [u8; 32] = [0x11; 32];
    const SERVER_PRIV: [u8; 32] = [0x22; 32];
    const SERVER_RANDOM: [u8; 32] = [0x33; 32];
    const SESSION_ID: [u8; 32] = [0x5a; 32];
    const SNI: &str = "models.example";
    /// The instant every handshake in these tests claims to begin. It is a
    /// constant, and deliberately *not* `SystemTime::now()`: the certificate
    /// slice's expired/not-yet-valid tests must give the same answer in 2036 as
    /// they do today.
    const VALIDATION_TIME: UnixTime = UnixTime::from_secs(1_700_000_000);
    /// A stand-in for a DER certificate: this slice only has to *carry* a chain,
    /// never read one.
    const FAKE_CERTIFICATE: [u8; 40] = [0xca; 40];

    fn sha(parts: &[&[u8]]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        for part in parts {
            hasher.update(part);
        }
        hasher.finalize()
    }

    /// A verifier that trusts everything, and it is a *named type* on purpose:
    /// the trait is not implemented for closures precisely so that "accept it"
    /// reads as a type in the diff rather than as a stray `|_| Ok(())`. It lives
    /// only in these tests, which is why the crate itself ships no such thing.
    struct AcceptTestChain;

    impl CertificateVerifier for AcceptTestChain {
        fn verify_certificate(&self, _input: &CertificateVerifyInput<'_>) -> Result<(), String> {
            Ok(())
        }
    }

    /// A verifier that checks every field the state machine is supposed to hand
    /// over, against values the test computed on its own.
    struct ExpectingChain {
        through_certificate: [u8; 32],
        chain: Vec<u8>,
        validation_time: UnixTime,
    }

    impl CertificateVerifier for ExpectingChain {
        fn verify_certificate(&self, input: &CertificateVerifyInput<'_>) -> Result<(), String> {
            assert_eq!(input.signature_scheme, SIG_ECDSA_SECP256R1_SHA256);
            assert_eq!(input.signed_data.len(), 130, "64 + 33 + 1 + 32");
            assert_eq!(&input.signed_data[..64], &[0x20u8; 64][..], "64 spaces");
            assert_eq!(
                &input.signed_data[64..97],
                b"TLS 1.3, server CertificateVerify"
            );
            assert_eq!(input.signed_data[97], 0, "the separator");
            assert_eq!(
                &input.signed_data[98..],
                &self.through_certificate[..],
                "the transcript hash, hashed here from the messages we sent"
            );
            assert_eq!(
                input.server_name,
                Some(SNI.as_bytes()),
                "the name the ClientHello asked for reaches the verifier, so it never has to capture it"
            );
            assert_eq!(
                input.validation_time, self.validation_time,
                "the caller's clock reading reaches the verifier untouched"
            );
            assert_eq!(
                input.certificate_list,
                &self.chain[..],
                "the chain from Certificate"
            );
            // The fake server "signs" by hashing the signed data, so this holds
            // only if the bytes we were handed are the bytes it signed.
            let mut hasher = Sha256::new();
            hasher.update(input.signed_data);
            assert_eq!(input.signature, hasher.finalize());
            Ok(())
        }
    }

    /// A verifier that records every instant it is shown through a shared
    /// handle, so a test can read them after the verifier has moved into the
    /// client.
    #[derive(Clone)]
    struct RecordingTime(Arc<Mutex<Vec<UnixTime>>>);

    impl CertificateVerifier for RecordingTime {
        fn verify_certificate(&self, input: &CertificateVerifyInput<'_>) -> Result<(), String> {
            self.0
                .lock()
                .expect("the recording handle is not poisoned")
                .push(input.validation_time);
            Ok(())
        }
    }

    /// Runs an honest handshake whose configuration claims to begin at `time`,
    /// and returns every instant the verifier was shown.
    fn instants_seen(time: UnixTime) -> Vec<UnixTime> {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (_, from_waiting) = waiting_client(Some(SNI));
        let flight = honest_flight(&from_waiting);

        let mut client = ClientHandshake::new(
            ClientConfig {
                server_name: Some(SNI),
                validation_time: time,
                random: [0x44; 32],
                session_id: SESSION_ID.to_vec(),
                private_key: CLIENT_PRIV,
            },
            RecordingTime(Arc::clone(&seen)),
        )
        .expect("valid configuration");
        let again = client.start().expect("we build our own ClientHello");
        assert_eq!(
            from_waiting, again,
            "a fixed configuration builds one ClientHello"
        );

        let events = client.feed(&flight.bytes).expect("an honest flight");
        assert!(
            events.contains(&Out::Connected),
            "the handshake completed: {events:?}"
        );
        let seen = seen.lock().expect("the recording handle is not poisoned");
        assert_eq!(seen.len(), 1, "one CertificateVerify, one decision");
        seen.clone()
    }

    fn client<V: CertificateVerifier>(
        verifier: V,
        session_id: Vec<u8>,
        server_name: Option<&str>,
    ) -> (ClientHandshake<V>, Vec<u8>) {
        let mut client = ClientHandshake::new(
            ClientConfig {
                server_name,
                validation_time: VALIDATION_TIME,
                random: [0x44; 32],
                session_id,
                private_key: CLIENT_PRIV,
            },
            verifier,
        )
        .expect("valid configuration");
        let first = client.start().expect("we build our own ClientHello");
        (client, first)
    }

    /// Test-only encoders for the server's side of the handshake. They exist so
    /// the tests can lie in specific ways; the production parsers they exercise
    /// are the ones in `handshake` and above.
    fn server_hello_bytes(session_id: &[u8], key_share: &[u8; 32], suite: u16) -> Vec<u8> {
        let mut extensions = Vec::new();
        extensions.extend_from_slice(&0x002bu16.to_be_bytes());
        extensions.extend_from_slice(&2u16.to_be_bytes());
        extensions.extend_from_slice(&[0x03, 0x04]);
        extensions.extend_from_slice(&0x0033u16.to_be_bytes());
        extensions.extend_from_slice(&36u16.to_be_bytes());
        extensions.extend_from_slice(&0x001du16.to_be_bytes());
        extensions.extend_from_slice(&32u16.to_be_bytes());
        extensions.extend_from_slice(key_share);

        let mut body = Vec::new();
        body.extend_from_slice(&[0x03, 0x03]);
        body.extend_from_slice(&SERVER_RANDOM);
        body.push(session_id.len() as u8);
        body.extend_from_slice(session_id);
        body.extend_from_slice(&suite.to_be_bytes());
        body.push(0);
        body.extend_from_slice(&(extensions.len() as u16).to_be_bytes());
        body.extend_from_slice(&extensions);
        encode(TYPE_SERVER_HELLO, &body)
    }

    fn encrypted_extensions_bytes(extensions: &[(u16, Vec<u8>)]) -> Vec<u8> {
        let mut inner = Vec::new();
        for (which, value) in extensions {
            inner.extend_from_slice(&which.to_be_bytes());
            inner.extend_from_slice(&(value.len() as u16).to_be_bytes());
            inner.extend_from_slice(value);
        }
        let mut body = Vec::new();
        body.extend_from_slice(&(inner.len() as u16).to_be_bytes());
        body.extend_from_slice(&inner);
        encode(TYPE_ENCRYPTED_EXTENSIONS, &body)
    }

    fn certificate_bytes(list: &[u8]) -> Vec<u8> {
        let mut body = vec![0u8]; // empty certificate_request_context
        body.extend_from_slice(&[
            (list.len() >> 16) as u8,
            (list.len() >> 8) as u8,
            list.len() as u8,
        ]);
        body.extend_from_slice(list);
        encode(TYPE_CERTIFICATE, &body)
    }

    /// `opaque certificate<1..2^24-1>` entries, framed.
    fn certificate_list(entries: &[&[u8]]) -> Vec<u8> {
        let mut list = Vec::new();
        for entry in entries {
            list.extend_from_slice(&[
                (entry.len() >> 16) as u8,
                (entry.len() >> 8) as u8,
                entry.len() as u8,
            ]);
            list.extend_from_slice(entry);
        }
        list
    }

    /// The fake server's "signature": a hash of the signed data, so a verifier
    /// can check what it was handed without any real signature algorithm.
    fn certificate_verify_bytes(signed_data: &[u8]) -> Vec<u8> {
        let signature = sha(&[signed_data]);
        let mut body = Vec::new();
        body.extend_from_slice(&SIG_ECDSA_SECP256R1_SHA256.to_be_bytes());
        body.extend_from_slice(&(signature.len() as u16).to_be_bytes());
        body.extend_from_slice(&signature);
        encode(TYPE_CERTIFICATE_VERIFY, &body)
    }

    /// The server's half of the connection, enough to complete a handshake and
    /// to check what the client sends back.
    struct Server {
        client_hello: Vec<u8>,
        server_hello: Vec<u8>,
        /// The transcript as a plain concatenation, hashed on demand so nothing
        /// here shares bookkeeping with the client's `Transcript`.
        transcript: Vec<u8>,
        handshake_secret: [u8; 32],
        client_hs_secret: [u8; 32],
        server_hs_secret: [u8; 32],
        client_app_secret: [u8; 32],
        server_app_secret: [u8; 32],
        send_hs: RecordLayer,
        recv_hs: RecordLayer,
        send_app: Option<RecordLayer>,
        recv_app: Option<RecordLayer>,
    }

    impl Server {
        /// Parses the client's first flight and installs the handshake keys.
        fn accept(
            client_flight: &[u8],
            suite: u16,
            session_id_echo: Option<&[u8]>,
            key_share: [u8; 32],
        ) -> Self {
            assert_eq!(client_flight[0], CONTENT_TYPE_HANDSHAKE);
            let hello_len = u16::from_be_bytes([client_flight[3], client_flight[4]]) as usize;
            let client_hello = client_flight[5..5 + hello_len].to_vec();
            assert_eq!(client_hello[0], TYPE_CLIENT_HELLO);
            let rest = &client_flight[5 + hello_len..];
            if !rest.is_empty() {
                assert_eq!(
                    rest,
                    &plaintext_record(CONTENT_TYPE_CCS, &[1])[..],
                    "a non-empty session id means a compatibility-mode CCS follows"
                );
            }

            // `client_hello` still carries its four-byte handshake header (the
            // transcript needs it), so the body starts at 4: version, then 32
            // bytes of random, then the one-byte session id length.
            let session_id_at = HS_HEADER + 2 + 32;
            let session_id_len = client_hello[session_id_at] as usize;
            let session_id = &client_hello[session_id_at + 1..session_id_at + 1 + session_id_len];
            let echo = session_id_echo
                .map(<[u8]>::to_vec)
                .unwrap_or_else(|| session_id.to_vec());
            let server_hello = server_hello_bytes(&echo, &key_share, suite);

            let ecdhe = x25519::diffie_hellman(&SERVER_PRIV, &x25519::public_key(&CLIENT_PRIV))
                .expect("our own peer key is not low order");
            let handshake = handshake_secret(&early_secret(None), &ecdhe).expect("hkdf");

            let mut transcript = Vec::new();
            transcript.extend_from_slice(&client_hello);
            transcript.extend_from_slice(&server_hello);
            let secrets =
                TrafficSecrets::derive(&handshake, "hs", &sha(&[&transcript])).expect("hkdf");

            Self {
                client_hello,
                server_hello,
                transcript,
                handshake_secret: handshake,
                client_hs_secret: secrets.client,
                server_hs_secret: secrets.server,
                client_app_secret: [0u8; 32],
                server_app_secret: [0u8; 32],
                send_hs: RecordLayer::new(
                    &TrafficKeys::from_secret(&secrets.server).expect("hkdf"),
                ),
                recv_hs: RecordLayer::new(
                    &TrafficKeys::from_secret(&secrets.client).expect("hkdf"),
                ),
                send_app: None,
                recv_app: None,
            }
        }

        fn push(&mut self, message: &[u8]) {
            self.transcript.extend_from_slice(message);
        }

        fn transcript_hash(&self) -> [u8; 32] {
            sha(&[&self.transcript])
        }

        /// RFC 8446 §4.4.3's signed content, written out here from the
        /// specification rather than reused from the client.
        fn signed_data(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&[0x20u8; 64]);
            out.extend_from_slice(b"TLS 1.3, server CertificateVerify");
            out.push(0);
            out.extend_from_slice(&self.transcript_hash());
            out
        }

        /// RFC 8446 §4.4.4's server `verify_data`.
        fn finished_mac(&self) -> [u8; 32] {
            let key = finished_key(&self.server_hs_secret).expect("hkdf");
            hmac_sha256(&key, &self.transcript_hash())
        }

        fn seal_each(&mut self, messages: &[&[u8]]) -> Vec<u8> {
            let mut out = Vec::new();
            for message in messages {
                out.extend_from_slice(
                    &self
                        .send_hs
                        .seal(CONTENT_TYPE_HANDSHAKE, message)
                        .expect("seal"),
                );
            }
            out
        }

        fn seal_joined(&mut self, messages: &[&[u8]]) -> Vec<u8> {
            let mut body = Vec::new();
            for message in messages {
                body.extend_from_slice(message);
            }
            self.send_hs
                .seal(CONTENT_TYPE_HANDSHAKE, &body)
                .expect("seal")
        }

        /// Verifies the client's Finished and installs the application keys.
        fn finish(&mut self, client_bytes: &[u8]) -> Result<(), String> {
            let through_server_finished = self.transcript_hash();
            let master = master_secret(&self.handshake_secret)?;
            let app = TrafficSecrets::derive(&master, "ap", &through_server_finished)?;
            self.client_app_secret = app.client;
            self.server_app_secret = app.server;

            let record = self.recv_hs.open(client_bytes).map_err(|e| e.to_string())?;
            assert_eq!(record.content_type, CONTENT_TYPE_HANDSHAKE);
            let mut reader = HandshakeReader::new(&record.payload);
            let message = reader
                .next_message()
                .map_err(|e| e.to_string())?
                .ok_or("no message")?;
            assert_eq!(message.msg_type(), TYPE_FINISHED);
            let key = finished_key(&self.client_hs_secret)?;
            verify_finished(&key, &through_server_finished, message.body())?;
            self.transcript.extend_from_slice(message.raw());

            self.send_app = Some(RecordLayer::new(&TrafficKeys::from_secret(
                &self.server_app_secret,
            )?));
            self.recv_app = Some(RecordLayer::new(&TrafficKeys::from_secret(
                &self.client_app_secret,
            )?));
            Ok(())
        }

        fn app_write(&mut self, data: &[u8]) -> Vec<u8> {
            self.send_app
                .as_mut()
                .expect("app keys")
                .seal(CONTENT_TYPE_APPLICATION_DATA, data)
                .expect("seal")
        }

        fn app_read(&mut self, record: &[u8]) -> Vec<u8> {
            let opened = self
                .recv_app
                .as_mut()
                .expect("app keys")
                .open(record)
                .expect("open");
            assert_eq!(opened.content_type, CONTENT_TYPE_APPLICATION_DATA);
            opened.payload
        }

        fn alert(&mut self, level: u8, description: u8) -> Vec<u8> {
            self.send_app
                .as_mut()
                .expect("app keys")
                .seal(CONTENT_TYPE_ALERT, &[level, description])
                .expect("seal")
        }

        /// Consumes a `KeyUpdate` from the client and rekeys the receive side.
        fn read_key_update(&mut self, record: &[u8]) {
            let opened = self
                .recv_app
                .as_mut()
                .expect("app keys")
                .open(record)
                .expect("open");
            assert_eq!(opened.content_type, CONTENT_TYPE_HANDSHAKE);
            let mut reader = HandshakeReader::new(&opened.payload);
            let message = reader.next_message().expect("parse").expect("a message");
            assert_eq!(message.msg_type(), TYPE_KEY_UPDATE);
            self.transcript.extend_from_slice(message.raw());
            self.client_app_secret = update_traffic_secret(&self.client_app_secret).expect("hkdf");
            self.recv_app = Some(RecordLayer::new(
                &TrafficKeys::from_secret(&self.client_app_secret).expect("hkdf"),
            ));
        }

        fn key_update(&mut self, request: bool) -> Vec<u8> {
            let message = encode(TYPE_KEY_UPDATE, &[u8::from(request)]);
            let record = self
                .send_app
                .as_mut()
                .expect("app keys")
                .seal(CONTENT_TYPE_HANDSHAKE, &message)
                .expect("seal");
            self.transcript.extend_from_slice(&message);
            self.server_app_secret = update_traffic_secret(&self.server_app_secret).expect("hkdf");
            self.send_app = Some(RecordLayer::new(
                &TrafficKeys::from_secret(&self.server_app_secret).expect("hkdf"),
            ));
            record
        }
    }

    /// A complete, honest server flight, plus what the test needs to check the
    /// client's reaction to it.
    struct Flight {
        bytes: Vec<u8>,
        through_certificate: [u8; 32],
        chain: Vec<u8>,
        server: Server,
    }

    fn honest_flight(client_flight: &[u8]) -> Flight {
        let mut server = Server::accept(
            client_flight,
            0x1301,
            None,
            x25519::public_key(&SERVER_PRIV),
        );
        let mut bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &server.server_hello);

        // The empty `server_name` acknowledgment, which is what a real server
        // sends when it accepted the SNI we offered.
        let extensions = encrypted_extensions_bytes(&[(0x0000, Vec::new())]);
        server.push(&extensions);
        let chain = certificate_list(&[&FAKE_CERTIFICATE]);
        let certificate = certificate_bytes(&chain);
        server.push(&certificate);
        let through_certificate = sha(&[
            &server.client_hello,
            &server.server_hello,
            &extensions,
            &certificate,
        ]);

        let verify = certificate_verify_bytes(&server.signed_data());
        server.push(&verify);
        let finished = encode(TYPE_FINISHED, &server.finished_mac());
        server.push(&finished);

        // Two records, two messages each: the flight is not required to align
        // records with messages.
        bytes.extend_from_slice(&server.seal_joined(&[&extensions, &certificate]));
        bytes.extend_from_slice(&server.seal_joined(&[&verify, &finished]));

        Flight {
            bytes,
            through_certificate,
            chain,
            server,
        }
    }

    /// A client that has sent its ClientHello and is waiting for the
    /// ServerHello, with a verifier that raises no objection of its own.
    fn waiting_client(server_name: Option<&str>) -> (ClientHandshake<AcceptTestChain>, Vec<u8>) {
        client(AcceptTestChain, SESSION_ID.to_vec(), server_name)
    }

    /// A server that has read the ClientHello and installed its keys, but has
    /// not sent the ServerHello into the transcript yet.
    fn accepted(client_flight: &[u8]) -> Server {
        Server::accept(
            client_flight,
            0x1301,
            None,
            x25519::public_key(&SERVER_PRIV),
        )
    }

    /// How a flight should be built, so the negative tests can name the single
    /// thing they are breaking.
    #[derive(Clone)]
    struct Plan {
        /// `None` sends no EncryptedExtensions at all.
        ee: Option<Vec<(u16, Vec<u8>)>>,
        /// `None` sends no Certificate at all.
        certificate: Option<Vec<u8>>,
        certificate_request: bool,
        corrupt_finished: bool,
        /// Put every message in one record instead of one record each.
        one_record: bool,
    }

    impl Plan {
        fn honest() -> Self {
            Self {
                ee: Some(vec![(0x0000, Vec::new())]),
                certificate: Some(certificate_list(&[&FAKE_CERTIFICATE])),
                certificate_request: false,
                corrupt_finished: false,
                one_record: false,
            }
        }
    }

    /// Builds the ServerHello record and the encrypted flight the plan
    /// describes, pushing each message into the server's transcript as a real
    /// server would.
    fn flight_with(server: &mut Server, plan: &Plan) -> Vec<u8> {
        let mut bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &server.server_hello.clone());
        bytes.extend_from_slice(&flight_records(server, plan));
        bytes
    }

    /// Just the encrypted part of the flight, for tests that need to interleave
    /// something of their own.
    fn flight_records(server: &mut Server, plan: &Plan) -> Vec<u8> {
        let mut messages: Vec<Vec<u8>> = Vec::new();
        if let Some(extensions) = &plan.ee {
            let ee = encrypted_extensions_bytes(extensions);
            server.push(&ee);
            messages.push(ee);
        }
        if plan.certificate_request {
            let request = encode(TYPE_CERTIFICATE_REQUEST, &[0]);
            server.push(&request);
            messages.push(request);
        }
        if let Some(list) = &plan.certificate {
            let certificate = certificate_bytes(list);
            server.push(&certificate);
            messages.push(certificate);
        }
        let verify = certificate_verify_bytes(&server.signed_data());
        server.push(&verify);
        messages.push(verify);

        let mut verify_data = server.finished_mac();
        if plan.corrupt_finished {
            verify_data[0] ^= 0x01;
        }
        let finished = encode(TYPE_FINISHED, &verify_data);
        server.push(&finished);
        messages.push(finished);

        let refs: Vec<&[u8]> = messages.iter().map(Vec::as_slice).collect();
        if plan.one_record {
            server.seal_joined(&refs)
        } else {
            server.seal_each(&refs)
        }
    }

    /// Runs the handshake to `Connected` and returns the client, the server, and
    /// every event the client produced.
    ///
    /// The verifier can only be supplied at construction and it has to know what
    /// the flight will contain, so the ClientHello is built twice. That is
    /// itself a check: the second one must be byte-identical.
    fn connected() -> (ClientHandshake<ExpectingChain>, Server, Vec<Out>) {
        let (_, first) = waiting_client(Some(SNI));
        let mut flight = honest_flight(&first);
        let (mut client, again) = client(
            ExpectingChain {
                through_certificate: flight.through_certificate,
                chain: flight.chain.clone(),
                validation_time: VALIDATION_TIME,
            },
            SESSION_ID.to_vec(),
            Some(SNI),
        );
        assert_eq!(first, again, "a fixed configuration builds one ClientHello");

        let events = client
            .feed(&flight.bytes)
            .expect("an honest flight is accepted");
        let mut to_server = Vec::new();
        let mut saw_connected = false;
        for event in &events {
            match event {
                Out::Send(bytes) => to_server.extend_from_slice(bytes),
                Out::Connected => saw_connected = true,
                other => panic!("unexpected event {other:?}"),
            }
        }
        assert!(saw_connected, "the client must report Connected");
        flight
            .server
            .finish(&to_server)
            .expect("the client's Finished verifies");
        (client, flight.server, events)
    }

    #[test]
    fn the_first_flight_is_a_client_hello_and_a_compatibility_ccs() {
        let (_client, first) = waiting_client(Some(SNI));
        assert_eq!(first[0], CONTENT_TYPE_HANDSHAKE);
        let hello_len = u16::from_be_bytes([first[3], first[4]]) as usize;
        assert_eq!(first[5], TYPE_CLIENT_HELLO);
        assert_eq!(
            first.len(),
            5 + hello_len + 6,
            "the ClientHello and the CCS"
        );
        assert_eq!(
            &first[5 + hello_len..],
            &plaintext_record(CONTENT_TYPE_CCS, &[1])[..]
        );

        // An empty session id means no compatibility mode, so no CCS.
        let (_client, first) = client(AcceptTestChain, Vec::new(), Some(SNI));
        let hello_len = u16::from_be_bytes([first[3], first[4]]) as usize;
        assert_eq!(first.len(), 5 + hello_len);
    }

    #[test]
    fn a_full_handshake_reaches_connected() {
        let (mut client, mut server, _) = connected();
        assert!(client.is_connected());

        // Application data both ways, which only works if both sides derived the
        // same application secrets from the same transcript.
        let from_server = server.app_write(b"GET /models HTTP/1.1");
        let events = client.feed(&from_server).unwrap();
        assert_eq!(
            events,
            vec![Out::Application(b"GET /models HTTP/1.1".to_vec())]
        );
        let from_client = client.write(b"ok").unwrap();
        assert_eq!(server.app_read(&from_client), b"ok");
    }

    #[test]
    fn a_tampered_finished_is_fatal() {
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.corrupt_finished = true;
        let bytes = flight_with(&mut server, &plan);

        let error = client
            .feed(&bytes)
            .expect_err("a wrong Finished MAC must stop us");
        match &error {
            ClientError::KeySchedule(why) => assert!(why.contains("Finished MAC"), "{why}"),
            other => panic!("expected a Finished failure, got {other:?}"),
        }
        assert!(!client.is_connected());
        // And it is sticky: nothing else can be done with this connection.
        assert_eq!(client.feed(&[]), Err(ClientError::Failed));
        assert_eq!(client.write(b"data"), Err(ClientError::Failed));
    }

    #[test]
    fn a_verifier_that_refuses_ends_the_connection() {
        let mut client = ClientHandshake::new(
            ClientConfig {
                server_name: Some(SNI),
                validation_time: VALIDATION_TIME,
                random: [0x44; 32],
                session_id: SESSION_ID.to_vec(),
                private_key: CLIENT_PRIV,
            },
            RejectAll,
        )
        .expect("valid configuration");
        let first = client.start().expect("we build our own ClientHello");
        let mut server = accepted(&first);
        let bytes = flight_with(&mut server, &Plan::honest());
        let error = client
            .feed(&bytes)
            .expect_err("RejectAll rejects everything");
        assert!(
            matches!(error, ClientError::VerifierRejected(_)),
            "{error:?}"
        );
        assert!(!client.is_connected());
    }

    /// The instant the verifier is shown is the one the caller put in the
    /// configuration, never the time of day.
    ///
    /// Every instant here is one that a clock would not report: 1970, the second
    /// before it, and the year 2100. If the state machine read a clock this test
    /// would fail today, and — the part that matters — it would still fail the
    /// same way in 2040, because none of these values is "now" at any date.
    /// That is what makes the certificate slice's expired and not-yet-valid tests
    /// writable at all: they need an answer that does not age with the fixture.
    #[test]
    fn the_verifier_is_given_the_callers_instant_and_not_the_clock() {
        for time in [
            UnixTime::from_secs(-1),
            UnixTime::epoch(),
            UnixTime::from_secs(1_700_000_000),
            UnixTime::from_secs(4_102_444_800), // 2100-01-01T00:00:00Z
        ] {
            assert_eq!(instants_seen(time), vec![time], "{time}");
        }

        // Two configurations are two instants, so the value is threaded through
        // rather than defaulted to a constant.
        assert_ne!(
            instants_seen(UnixTime::epoch()),
            instants_seen(UnixTime::from_secs(1))
        );
    }

    #[test]
    fn messages_out_of_order_are_refused() {
        // Certificate with no EncryptedExtensions first.
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.ee = None;
        let bytes = flight_with(&mut server, &plan);
        assert_eq!(
            client.feed(&bytes),
            Err(ClientError::UnexpectedMessage {
                expected: "EncryptedExtensions",
                found: TYPE_CERTIFICATE,
            })
        );

        // Finished before CertificateVerify.
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.certificate = None;
        let bytes = flight_with(&mut server, &plan);
        assert_eq!(
            client.feed(&bytes),
            Err(ClientError::UnexpectedMessage {
                expected: "Certificate",
                found: TYPE_CERTIFICATE_VERIFY,
            })
        );
    }

    #[test]
    fn a_client_authentication_request_is_refused() {
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.certificate_request = true;
        let bytes = flight_with(&mut server, &plan);
        assert_eq!(
            client.feed(&bytes),
            Err(ClientError::Unsupported("client authentication"))
        );
    }

    #[test]
    fn certificate_chains_that_are_not_framed_correctly_are_refused() {
        // An empty chain is only legal with a PSK, which we do not offer.
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.certificate = Some(Vec::new());
        let bytes = flight_with(&mut server, &plan);
        let error = client
            .feed(&bytes)
            .expect_err("an empty chain is not acceptable");
        assert!(
            matches!(error, ClientError::EmptyCertificateChain),
            "{error:?}"
        );

        // A list that claims more bytes than it has, an entry length that runs
        // past the end, an entry that is a bare length header, and a zero-length
        // entry (which would let a chain be empty without saying so).
        for list in [
            vec![0u8, 0, 40, 0xaa],
            vec![0u8, 0, 3, 0, 0, 8, 1, 2, 3],
            vec![0u8, 0, 32],
            vec![0u8, 0, 0],
            vec![0u8, 0, 0, 0],
        ] {
            let (mut client, first) = waiting_client(Some(SNI));
            let mut server = accepted(&first);
            let mut plan = Plan::honest();
            plan.certificate = Some(list.clone());
            let bytes = flight_with(&mut server, &plan);
            let error = client.feed(&bytes).expect_err("bad framing is fatal");
            assert!(matches!(error, ClientError::Malformed(_)), "{error:?}");
        }
    }

    #[test]
    fn a_bad_server_hello_is_refused() {
        // A cipher suite we did not offer.
        let (mut client, first) = waiting_client(Some(SNI));
        let server = Server::accept(&first, 0x1302, None, x25519::public_key(&SERVER_PRIV));
        let bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &server.server_hello.clone());
        let error = client.feed(&bytes).expect_err("a suite we did not offer");
        assert!(matches!(error, ClientError::Handshake(_)), "{error:?}");

        // A session id echo that is not ours.
        let (mut client, first) = waiting_client(Some(SNI));
        let server = Server::accept(
            &first,
            0x1301,
            Some(&[0xde; 32]),
            x25519::public_key(&SERVER_PRIV),
        );
        let bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &server.server_hello.clone());
        assert_eq!(client.feed(&bytes), Err(ClientError::SessionIdMismatch));

        // A key share that is not a valid public key (RFC 7748 §6.1).
        let (mut client, first) = waiting_client(Some(SNI));
        let server = Server::accept(&first, 0x1301, None, [0u8; 32]);
        let bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &server.server_hello.clone());
        assert_eq!(client.feed(&bytes), Err(ClientError::LowOrderServerKey));

        // Two messages in the plaintext record: nothing else may follow the
        // ServerHello in the clear.
        let (mut client, first) = waiting_client(Some(SNI));
        let server = accepted(&first);
        let extra = encode(TYPE_FINISHED, &[0u8; 32]);
        let mut body = server.server_hello.clone();
        body.extend_from_slice(&extra);
        let bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &body);
        let error = client.feed(&bytes).expect_err("two plaintext messages");
        assert!(matches!(error, ClientError::Malformed(_)), "{error:?}");
    }

    #[test]
    fn encrypted_extension_rules_are_enforced() {
        // The empty SNI acknowledgment is accepted when we sent SNI...
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.ee = Some(vec![(0x0000, Vec::new())]);
        let bytes = flight_with(&mut server, &plan);
        let events = client.feed(&bytes).expect("the SNI ack is fine");
        assert!(events.contains(&Out::Connected));

        // ...but an extension we did not offer is not.
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.ee = Some(vec![(0x1234, vec![1, 2, 3])]);
        let bytes = flight_with(&mut server, &plan);
        assert_eq!(
            client.feed(&bytes),
            Err(ClientError::UnsupportedExtension(0x1234))
        );

        // And the same SNI acknowledgment is refused when we sent no SNI, since
        // then the server is answering an offer we never made.
        let (mut client, first) = waiting_client(None);
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.ee = Some(vec![(0x0000, Vec::new())]);
        let bytes = flight_with(&mut server, &plan);
        assert_eq!(
            client.feed(&bytes),
            Err(ClientError::UnsupportedExtension(0x0000))
        );
    }

    #[test]
    fn alerts_and_close_notify_are_reported() {
        // A plaintext fatal alert instead of a ServerHello.
        let (mut client, _) = waiting_client(Some(SNI));
        let bytes = plaintext_record(CONTENT_TYPE_ALERT, &[2, 40]);
        assert_eq!(
            client.feed(&bytes),
            Err(ClientError::ServerAlert {
                level: 2,
                description: 40
            })
        );

        // An encrypted alert after the handshake is fatal too.
        let (mut client, mut server, _) = connected();
        let record = server.alert(2, 51);
        assert_eq!(
            client.feed(&record),
            Err(ClientError::ServerAlert {
                level: 2,
                description: 51
            })
        );

        // close_notify is orderly, and ends the read side.
        let (mut client, mut server, _) = connected();
        let record = server.alert(1, 0);
        assert_eq!(client.feed(&record), Ok(vec![Out::Closed]));
        assert!(!client.is_connected());
        // Our own close_notify is still allowed, and then we are done writing.
        client.close_notify().expect("we can still say goodbye");
        assert!(client.write(b"late").is_err());
    }

    #[test]
    fn application_data_before_the_handshake_is_refused() {
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        // A well-formed, correctly encrypted record whose inner type is
        // application data, sent before the Finished.
        let record = server
            .send_hs
            .seal(CONTENT_TYPE_APPLICATION_DATA, b"early")
            .expect("seal");
        let mut bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &server.server_hello.clone());
        bytes.extend_from_slice(&record);
        assert_eq!(client.feed(&bytes), Err(ClientError::UnexpectedAppData));
    }

    #[test]
    fn hostile_record_framing_is_refused_without_panicking() {
        // A version that is not 0x0303.
        let (mut client, _) = waiting_client(Some(SNI));
        let bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &[0xff; 4]);
        let mut bad_version = bytes.clone();
        bad_version[1] = 0x02;
        assert_eq!(
            client.feed(&bad_version),
            Err(ClientError::Record(RecordError::UnexpectedVersion([2, 3])))
        );

        // A length beyond anything TLS allows.
        let (mut client, _) = waiting_client(Some(SNI));
        let mut oversized = plaintext_record(CONTENT_TYPE_HANDSHAKE, &[0xff; 4]);
        oversized[3] = 0xff;
        oversized[4] = 0xff;
        assert_eq!(
            client.feed(&oversized),
            Err(ClientError::Record(RecordError::RecordTooLong(0xffff)))
        );

        // A change_cipher_spec in the middle of the flight is ignored, as
        // RFC 8446 §5 requires.
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut bytes = plaintext_record(CONTENT_TYPE_HANDSHAKE, &server.server_hello.clone());
        bytes.extend_from_slice(&plaintext_record(CONTENT_TYPE_CCS, &[1]));
        bytes.extend_from_slice(&flight_records(&mut server, &Plan::honest()));
        let events = client.feed(&bytes).expect("an interleaved CCS is ignored");
        assert!(events.contains(&Out::Connected));
    }

    #[test]
    fn messages_may_span_records_and_arrive_one_byte_at_a_time() {
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let bytes = flight_with(&mut server, &Plan::honest());
        let mut connected = false;
        for byte in &bytes {
            for event in client
                .feed(std::slice::from_ref(byte))
                .expect("valid bytes")
            {
                if event == Out::Connected {
                    connected = true;
                }
            }
        }
        assert!(connected, "one byte at a time still finishes");

        // The whole flight in one record is equally fine.
        let (mut client, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let mut plan = Plan::honest();
        plan.one_record = true;
        let bytes = flight_with(&mut server, &plan);
        let events = client.feed(&bytes).expect("one record is fine");
        assert!(events.contains(&Out::Connected));
    }

    #[test]
    fn key_update_rekeys_both_directions() {
        let (mut client, mut server, _) = connected();

        // The server rekeys and asks us to.
        let update = server.key_update(true);
        let events = client.feed(&update).expect("KeyUpdate is accepted");
        let response: Vec<u8> = events
            .iter()
            .filter_map(|event| match event {
                Out::Send(bytes) => Some(bytes.clone()),
                _ => None,
            })
            .flatten()
            .collect();
        assert!(
            !response.is_empty(),
            "a request means we must answer with one"
        );
        // The server has to follow our answer, which is what proves both sides
        // moved to the same generation.
        server.read_key_update(&response);

        // The server's new keys are the ones we now listen for.
        let record = server.app_write(b"after update");
        assert_eq!(
            client.feed(&record),
            Ok(vec![Out::Application(b"after update".to_vec())])
        );

        // And our own update, after which the server can still read us. This
        // failing with `AuthenticationFailed` is the signature of a half-done
        // rekey: the record opens only if both directions advanced together.
        let ours = client.key_update().expect("we can request an update");
        server.read_key_update(&ours);
        let data = client.write(b"post-update").expect("write");
        assert_eq!(server.app_read(&data), b"post-update");

        // Nothing above had to assert that the old keys are gone: every record
        // in this test opens only if both ends advanced to the same generation
        // at the same boundary, so a half-done rekey fails as an AEAD error
        // rather than passing quietly. That is exactly the failure this test
        // first hit, on the server's receive side.
    }

    #[test]
    fn post_handshake_messages_we_do_not_implement_are_refused() {
        let (mut client, mut server, _) = connected();
        let ticket = encode(TYPE_NEW_SESSION_TICKET, &[0u8; 8]);
        let record = server
            .send_app
            .as_mut()
            .expect("app keys")
            .seal(CONTENT_TYPE_HANDSHAKE, &ticket)
            .expect("seal");
        assert_eq!(
            client.feed(&record),
            Err(ClientError::Unsupported("a post-handshake message"))
        );
    }

    #[test]
    fn write_before_the_handshake_finishes_is_refused() {
        let (mut client, _) = waiting_client(Some(SNI));
        assert_eq!(client.write(b"early"), Err(ClientError::NotConnected));
        assert_eq!(client.key_update(), Err(ClientError::NotConnected));
    }

    #[test]
    fn hostile_input_never_panics() {
        let (_, first) = waiting_client(Some(SNI));
        let mut server = accepted(&first);
        let honest = flight_with(&mut server, &Plan::honest());

        // Every truncation.
        for len in 0..=honest.len() {
            let (mut client, _) = waiting_client(Some(SNI));
            let _ = client.feed(&honest[..len]);
        }
        // Every single-bit change.
        for index in 0..honest.len() {
            for bit in [0x01u8, 0x80] {
                let mut candidate = honest.clone();
                candidate[index] ^= bit;
                let (mut client, _) = waiting_client(Some(SNI));
                let _ = client.feed(&candidate);
            }
        }
        // Pure noise, including bytes that look like record headers.
        for seed in 0..64u8 {
            let noise: Vec<u8> = (0..96u8).map(|i| i.wrapping_mul(7) ^ seed).collect();
            let (mut client, _) = waiting_client(Some(SNI));
            let _ = client.feed(&noise);
        }
    }

    #[test]
    fn the_client_hello_is_not_sent_twice_and_feed_needs_it() {
        let mut client = ClientHandshake::new(
            ClientConfig {
                server_name: None,
                validation_time: VALIDATION_TIME,
                random: [0x44; 32],
                session_id: Vec::new(),
                private_key: CLIENT_PRIV,
            },
            AcceptTestChain,
        )
        .unwrap();
        assert_eq!(client.feed(&[0u8; 5]), Err(ClientError::NotStarted));
        client.start().unwrap();
        assert_eq!(client.start(), Err(ClientError::AlreadyStarted));
    }

    #[test]
    fn a_bad_configuration_is_refused() {
        let config = |session_id: Vec<u8>, name: Option<&'static str>| ClientConfig {
            server_name: name,
            validation_time: VALIDATION_TIME,
            random: [0x44; 32],
            session_id,
            private_key: CLIENT_PRIV,
        };
        assert!(ClientHandshake::new(config(vec![0u8; 33], None), RejectAll).is_err());
        assert!(ClientHandshake::new(config(Vec::new(), Some("")), RejectAll).is_err());
        assert!(ClientHandshake::new(config(Vec::new(), None), RejectAll).is_ok());
    }

    /// The time arithmetic, which is pure, is tested exactly rather than only
    /// through a handshake.
    mod unix_time {
        use super::*;
        use std::time::Duration;

        #[test]
        fn a_system_time_converts_at_second_resolution_in_both_directions() {
            let epoch = SystemTime::UNIX_EPOCH;
            assert_eq!(UnixTime::from_system_time(epoch), UnixTime::epoch());
            assert_eq!(UnixTime::from(epoch), UnixTime::epoch());
            assert_eq!(
                UnixTime::from_system_time(epoch + Duration::from_secs(1_700_000_000)),
                UnixTime::from_secs(1_700_000_000)
            );
            // A clock set before 1970 must not panic or wrap.
            assert_eq!(
                UnixTime::from_system_time(epoch - Duration::from_secs(5)),
                UnixTime::from_secs(-5)
            );
            // Sub-second precision is truncated, not rounded: X.509 bounds are
            // second-resolution, so a bound of `T` is valid through the whole of
            // second `T` rather than only its first instants.
            assert_eq!(
                UnixTime::from_system_time(epoch + Duration::from_millis(1500)),
                UnixTime::from_secs(1)
            );
            assert_eq!(
                UnixTime::from_system_time(epoch + Duration::from_micros(999_999)),
                UnixTime::epoch()
            );
        }

        #[test]
        fn the_validity_window_is_inclusive_at_both_ends() {
            let not_before = UnixTime::from_secs(1_000);
            let not_after = UnixTime::from_secs(2_000);
            assert!(!UnixTime::from_secs(1_500).outside(Some(not_before), not_after));
            assert!(
                !not_before.outside(Some(not_before), not_after),
                "inclusive"
            );
            assert!(!not_after.outside(Some(not_before), not_after), "inclusive");
            assert!(UnixTime::from_secs(999).outside(Some(not_before), not_after));
            assert!(UnixTime::from_secs(2_001).outside(Some(not_before), not_after));
            // A missing lower bound leaves only the upper one in force.
            assert!(!UnixTime::epoch().outside(None, not_after));
            assert!(UnixTime::from_secs(2_001).outside(None, not_after));
        }

        #[test]
        fn display_names_the_unit_so_an_error_cannot_be_misread() {
            assert_eq!(
                UnixTime::from_secs(1_700_000_000).to_string(),
                "1700000000s since the Unix epoch"
            );
        }
    }
}
