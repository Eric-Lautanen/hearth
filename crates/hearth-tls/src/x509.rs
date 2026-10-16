//! Reading X.509 certificates (RFC 5280), strictly.
//!
//! The layer between [`crate::der`], which knows encodings, and the verifier,
//! which will know trust. It turns a certificate into the fields a decision needs:
//! the bytes a signature covers, the two names, the validity window, the public
//! key, and the extensions that constrain what the certificate may do.
//!
//! # What is checked, and what is left to the caller
//!
//! Checked here, because they are properties of the *encoding* or of RFC 5280's
//! profile:
//!
//! - The outer structure is exactly three elements with nothing left over, and the
//!   `signatureAlgorithm` inside the signed `TBSCertificate` is **byte-identical**
//!   to the one outside it. RFC 5280 requires that, and a certificate where they
//!   differ is the shape of several real vulnerabilities: a signature verified
//!   under one algorithm while the fields were read under another.
//! - A **critical** extension that this crate does not understand is a rejection,
//!   as RFC 5280 §4.2 requires. Unknown non-critical extensions are skipped, which
//!   is what makes certificates extensible.
//! - An extension appears at most once. Two `keyUsage` extensions would be two
//!   answers to one question.
//! - The serial number is positive and no more than 20 octets (§4.1.2.2), the two
//!   times are well-formed and in order, and `pathLenConstraint` appears only when
//!   `cA` is asserted (§4.2.1.9).
//! - The key is a supported shape: `rsaEncryption` with a positive modulus and
//!   exponent, or `ecPublicKey` on `prime256v1` in the uncompressed point format.
//!   Anything else parses as [`PublicKey::Unsupported`] rather than failing, so a
//!   chain can be reported as unsupported rather than malformed.
//!
//! **Not** checked here, deliberately: whether the chain is trusted, whether the
//! name matches, whether the validity window contains the connection's instant,
//! whether the leaf is a server certificate, and whether the key is big enough.
//! Those are decisions, they belong in the verifier, and keeping them out of the
//! parser is what lets this file be read as a description of the format.
//!
//! # The version question
//!
//! A `v1` certificate is parsed and its `version` reported as 1. TLS 1.3 requires
//! X.509v3 (§4.4.2.2), so the verifier refuses anything else; the parser's job is
//! to read the field, not to enforce a protocol rule that lives three layers up.

use crate::client::UnixTime;
use crate::der::{oids, Der, DerError, Element, Oid, Tag};

/// Why a certificate was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum X509Error {
    /// The encoding is not the DER this reader accepts.
    Der(DerError),
    /// A field the profile requires is absent.
    Missing(&'static str),
    /// A field is present but not the shape RFC 5280 gives it.
    Invalid(&'static str),
    /// A critical extension this crate does not implement (RFC 5280 §4.2).
    UnsupportedCriticalExtension(Vec<u8>),
    /// The certificate is well-formed but uses something this crate cannot read.
    Unsupported(&'static str),
}

impl std::fmt::Display for X509Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Der(error) => write!(f, "{error}"),
            Self::Missing(what) => write!(f, "the certificate has no {what}"),
            Self::Invalid(why) => write!(f, "invalid certificate: {why}"),
            Self::UnsupportedCriticalExtension(oid) => write!(
                f,
                "the certificate has a critical extension this client does not understand ({})",
                oid.iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            ),
            Self::Unsupported(what) => write!(f, "unsupported certificate: {what}"),
        }
    }
}

impl std::error::Error for X509Error {}

impl From<DerError> for X509Error {
    fn from(error: DerError) -> Self {
        Self::Der(error)
    }
}

/// The public key in a certificate, as far as this crate can read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublicKey<'a> {
    /// An RSA modulus and exponent, both big-endian, from `RSAPublicKey`.
    Rsa {
        /// The modulus, big-endian, with no leading zero.
        modulus: &'a [u8],
        /// The public exponent.
        exponent: u64,
    },
    /// A P-256 public key in the uncompressed point format (`0x04 || X || Y`).
    Ec {
        /// The point, 65 bytes when uncompressed.
        point: &'a [u8],
    },
    /// A key type this crate cannot verify with, kept so the caller can say so
    /// rather than claim the certificate was malformed.
    Unsupported(Oid<'a>),
}

/// The signature algorithm, normalized to the ones this crate implements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureAlgorithm<'a> {
    /// `sha256WithRSAEncryption`, the PKCS#1 v1.5 form most CAs sign with.
    RsaPkcs1Sha256,
    /// `rsa_pss_rsae_sha256`, with MGF1-SHA-256 and the salt length the
    /// `AlgorithmIdentifier` names.
    RsaPssSha256 {
        /// The salt length from the parameters, checked against the digest length
        /// by whoever verifies.
        salt_len: u32,
    },
    /// `ecdsa-with-SHA256`. The signature is a DER `SEQUENCE` of two integers.
    EcdsaSha256,
    /// A scheme this crate does not implement, named by its OID.
    Unsupported(Oid<'a>),
}

/// One choice of `GeneralName` (RFC 5280 §4.2.1.6).
///
/// The names are carried as the raw IA5 bytes rather than as `str`, because a name
/// is compared and matched, not displayed: turning bytes into a string first is
/// how an invalid UTF-8 name becomes a valid-looking one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeneralName<'a> {
    /// `dNSName`, which is what a TLS client checks.
    Dns(&'a [u8]),
    /// `rfc822Name`.
    Rfc822(&'a [u8]),
    /// `iPAddress`, four or sixteen raw bytes.
    IpAddress(&'a [u8]),
    /// `uniformResourceIdentifier`.
    Uri(&'a [u8]),
    /// Any other choice, including the ones this crate never needs.
    Other,
}

/// The `keyUsage` bits, numbered as the DER `BIT STRING` numbers them: bit 0 is
/// `digitalSignature`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyUsage(pub u16);

impl KeyUsage {
    /// bit 0 — `digitalSignature`, which a TLS server certificate needs.
    pub const DIGITAL_SIGNATURE: u16 = 1 << 0;
    /// bit 1 — `contentCommitment` (the old `nonRepudiation`).
    pub const CONTENT_COMMITMENT: u16 = 1 << 1;
    /// bit 2 — `keyEncipherment`.
    pub const KEY_ENCIPHERMENT: u16 = 1 << 2;
    /// bit 3 — `dataEncipherment`.
    pub const DATA_ENCIPHERMENT: u16 = 1 << 3;
    /// bit 4 — `keyAgreement`.
    pub const KEY_AGREEMENT: u16 = 1 << 4;
    /// bit 5 — `keyCertSign`, which only a CA may assert.
    pub const KEY_CERT_SIGN: u16 = 1 << 5;
    /// bit 6 — `cRLSign`.
    pub const CRL_SIGN: u16 = 1 << 6;
    /// bit 7 — `encipherOnly`, defined only when `keyAgreement` is set.
    pub const ENCIPHER_ONLY: u16 = 1 << 7;
    /// bit 8 — `decipherOnly`, defined only when `keyAgreement` is set.
    pub const DECIPHER_ONLY: u16 = 1 << 8;

    /// Whether every bit in `bits` is asserted.
    pub fn has(self, bits: u16) -> bool {
        self.0 & bits == bits
    }
}

/// A `Name`: an `RDNSequence`, kept as its own DER so that two names can be
/// compared the way a chain needs them to be — by their encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Name<'a> {
    der: &'a [u8],
}

impl<'a> Name<'a> {
    /// The name's own DER, without the enclosing structure. Two certificates are
    /// linked when the issuer of one is this and the subject of the other is byte
    /// for byte the same.
    pub fn as_der(&self) -> &'a [u8] {
        self.der
    }

    /// The last `commonName` in the name, if it has one.
    ///
    /// RFC 6125 §6.4.4 deprecated using it for host name checks, and this crate's
    /// verifier will not, but it is what a diagnostic wants to print and what
    /// `subjectAltName`-less certificates are named by.
    pub fn common_name(&self) -> Result<Option<&'a [u8]>, X509Error> {
        let mut reader = Der::new(self.der);
        let sequence = reader.take(Tag::SEQUENCE)?;
        reader.expect_end()?;

        let mut found = None;
        let mut rdns = sequence.reader()?;
        while !rdns.is_empty() {
            let rdn = rdns.take(Tag::SET)?;
            let mut attributes = rdn.reader()?;
            while !attributes.is_empty() {
                let attribute = attributes.take(Tag::SEQUENCE)?;
                let mut parts = attribute.reader()?;
                let oid = parts.take(Tag::OID)?.as_oid()?;
                let value = parts.next_element()?;
                parts.expect_end()?;
                if oid.as_bytes() == oids::COMMON_NAME {
                    found = Some(value.as_primitive()?);
                }
            }
        }
        Ok(found)
    }
}

/// A parsed X.509 certificate, borrowing from the DER it was read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Certificate<'a> {
    der: &'a [u8],
    tbs: &'a [u8],
    version: u8,
    serial: &'a [u8],
    issuer: Name<'a>,
    subject: Name<'a>,
    not_before: UnixTime,
    not_after: UnixTime,
    public_key: PublicKey<'a>,
    is_ca: bool,
    path_len: Option<u32>,
    key_usage: Option<KeyUsage>,
    extended_key_usage: Vec<Oid<'a>>,
    subject_alt_names: Vec<GeneralName<'a>>,
    signature_algorithm: SignatureAlgorithm<'a>,
    signature: &'a [u8],
}

impl<'a> Certificate<'a> {
    /// Reads a certificate.
    ///
    /// Every structural rule in the module documentation is applied here; a
    /// failure means the bytes are not the certificate they claim to be, never
    /// that a field was merely unusual.
    pub fn parse(der: &'a [u8]) -> Result<Self, X509Error> {
        let mut top = Der::new(der);
        let certificate = top.take(Tag::SEQUENCE)?;
        top.expect_end()?;

        let mut fields = certificate.reader()?;
        let tbs_element = fields.take(Tag::SEQUENCE)?;
        let tbs = tbs_element.whole();
        let signature_algorithm_element = fields.take(Tag::SEQUENCE)?;
        let signature = fields.take(Tag::BIT_STRING)?.as_bit_string()?;
        fields.expect_end()?;

        let mut body = tbs_element.reader()?;

        // version [0] EXPLICIT Version DEFAULT v1: absent means v1.
        let version = match body.take_if(Tag::context(0))? {
            None => 1,
            Some(explicit) => {
                let mut inner = explicit.reader()?;
                let value = inner.take(Tag::INTEGER)?.as_u64()?;
                inner.expect_end()?;
                match value {
                    0 => 1,
                    1 => 2,
                    2 => 3,
                    _ => return Err(X509Error::Invalid("the version is above 3")),
                }
            }
        };

        let serial_element = body.take(Tag::INTEGER)?;
        let serial = serial_element.as_integer_bytes()?;
        // RFC 5280 §4.1.2.2: a positive integer of at most 20 octets. The DER
        // reader has already removed the sign pad and refused a negative value.
        if serial.len() > 20 {
            return Err(X509Error::Invalid(
                "the serial number is longer than 20 octets",
            ));
        }

        // The algorithm inside the signed part must be the same as the one
        // outside it, byte for byte (RFC 5280 §4.1.1.2).
        let tbs_algorithm = body.take(Tag::SEQUENCE)?;
        if tbs_algorithm.whole() != signature_algorithm_element.whole() {
            return Err(X509Error::Invalid(
                "the signature algorithm inside the certificate differs from the one outside it",
            ));
        }
        let signature_algorithm = parse_signature_algorithm(&signature_algorithm_element)?;

        let issuer_element = body.take(Tag::SEQUENCE)?;
        let issuer = Name {
            der: issuer_element.whole(),
        };
        // Parsing the name here would only catch the malformed cases the CN lookup
        // catches later, and a name that is never looked up should not be a reason
        // to reject a certificate whose *signature* is what matters. The chain
        // compares the DER; the leaf's own name is read through `common_name`.

        let validity = body.take(Tag::SEQUENCE)?;
        let (not_before, not_after) = parse_validity(&validity)?;

        let subject_element = body.take(Tag::SEQUENCE)?;
        let subject = Name {
            der: subject_element.whole(),
        };

        let spki = body.take(Tag::SEQUENCE)?;
        let public_key = parse_public_key(&spki)?;

        // issuerUniqueID [1] and subjectUniqueID [2] are v2/v3 leftovers this
        // crate has no use for: skipped if present, refused if malformed.
        while let Some(element) = body.take_if(Tag::context_primitive(1))? {
            let _ = element.as_primitive()?;
        }
        while let Some(element) = body.take_if(Tag::context_primitive(2))? {
            let _ = element.as_primitive()?;
        }

        let mut extensions = Extensions::default();
        if let Some(explicit) = body.take_if(Tag::context(3))? {
            let mut inner = explicit.reader()?;
            let list = inner.take(Tag::SEQUENCE)?;
            inner.expect_end()?;
            extensions = Extensions::parse(&list)?;
        }
        body.expect_end()?;

        if !extensions.seen.is_empty() && version != 3 {
            return Err(X509Error::Invalid(
                "extensions are present but the version is not v3",
            ));
        }
        if not_before > not_after {
            return Err(X509Error::Invalid("the validity window is reversed"));
        }

        Ok(Self {
            der,
            tbs,
            version,
            serial,
            issuer,
            subject,
            not_before,
            not_after,
            public_key,
            is_ca: extensions.is_ca,
            path_len: extensions.path_len,
            key_usage: extensions.key_usage,
            extended_key_usage: extensions.extended_key_usage,
            subject_alt_names: extensions.subject_alt_names,
            signature_algorithm,
            signature,
        })
    }

    /// The whole certificate, as it was read.
    pub fn der(&self) -> &'a [u8] {
        self.der
    }

    /// The `TBSCertificate`: exactly the bytes its signature covers.
    ///
    /// Handed over as the original bytes rather than re-encoded, so "the bytes
    /// that were signed" and "the fields that were read" cannot be made to differ.
    pub fn tbs(&self) -> &'a [u8] {
        self.tbs
    }

    /// The version: 1, 2 or 3.
    pub fn version(&self) -> u8 {
        self.version
    }

    /// The serial number's magnitude, big-endian, no leading zero.
    pub fn serial(&self) -> &'a [u8] {
        self.serial
    }

    /// The issuing name, as its own DER.
    pub fn issuer(&self) -> Name<'a> {
        self.issuer
    }

    /// The subject name, as its own DER.
    pub fn subject(&self) -> Name<'a> {
        self.subject
    }

    /// The start of the validity window.
    pub fn not_before(&self) -> UnixTime {
        self.not_before
    }

    /// The end of the validity window, inclusive.
    pub fn not_after(&self) -> UnixTime {
        self.not_after
    }

    /// The subject's public key.
    pub fn public_key(&self) -> &PublicKey<'a> {
        &self.public_key
    }

    /// Whether `basicConstraints` asserts `cA`.
    pub fn is_ca(&self) -> bool {
        self.is_ca
    }

    /// `pathLenConstraint`, when the certificate has one.
    pub fn path_len(&self) -> Option<u32> {
        self.path_len
    }

    /// `keyUsage`, when the certificate has one.
    pub fn key_usage(&self) -> Option<KeyUsage> {
        self.key_usage
    }

    /// Every `extKeyUsage` purpose.
    pub fn extended_key_usage(&self) -> &[Oid<'a>] {
        &self.extended_key_usage
    }

    /// Every `subjectAltName`, in the order they appear.
    pub fn subject_alt_names(&self) -> &[GeneralName<'a>] {
        &self.subject_alt_names
    }

    /// The signature algorithm.
    pub fn signature_algorithm(&self) -> &SignatureAlgorithm<'a> {
        &self.signature_algorithm
    }

    /// The signature bytes, from the `BIT STRING`.
    pub fn signature(&self) -> &'a [u8] {
        self.signature
    }
}

/// The extensions this crate understands, plus the bookkeeping that makes an
/// unknown critical one fatal.
#[derive(Default, Debug)]
struct Extensions<'a> {
    seen: Vec<Oid<'a>>,
    is_ca: bool,
    path_len: Option<u32>,
    key_usage: Option<KeyUsage>,
    extended_key_usage: Vec<Oid<'a>>,
    subject_alt_names: Vec<GeneralName<'a>>,
}

impl<'a> Extensions<'a> {
    fn parse(list: &Element<'a>) -> Result<Self, X509Error> {
        let mut result = Self::default();
        let mut reader = list.reader()?;
        while !reader.is_empty() {
            let extension = reader.take(Tag::SEQUENCE)?;
            let mut fields = extension.reader()?;
            let oid = fields.take(Tag::OID)?.as_oid()?;
            let critical = match fields.take_if(Tag::BOOLEAN)? {
                Some(value) => value.as_boolean()?,
                // DEFAULT FALSE: absent means not critical.
                None => false,
            };
            let value = fields.take(Tag::OCTET_STRING)?.as_primitive()?;
            fields.expect_end()?;

            if result.seen.contains(&oid) {
                return Err(X509Error::Invalid("an extension appears more than once"));
            }
            result.seen.push(oid);

            match oid.as_bytes() {
                oids::BASIC_CONSTRAINTS => {
                    let (is_ca, path_len) = basic_constraints(value)?;
                    result.is_ca = is_ca;
                    result.path_len = path_len;
                }
                oids::KEY_USAGE => result.key_usage = Some(key_usage(value)?),
                oids::EXT_KEY_USAGE => result.extended_key_usage = extended_key_usage(value)?,
                oids::SUBJECT_ALT_NAME => result.subject_alt_names = subject_alt_names(value)?,
                _ => {
                    // RFC 5280 §4.2: "A certificate-using system MUST reject the
                    // certificate if it encounters a critical extension it does
                    // not recognize." Non-critical ones are what makes the format
                    // extensible, so they are skipped.
                    if critical {
                        return Err(X509Error::UnsupportedCriticalExtension(
                            oid.as_bytes().to_vec(),
                        ));
                    }
                }
            }
        }
        Ok(result)
    }
}

/// `BasicConstraints ::= SEQUENCE { cA BOOLEAN DEFAULT FALSE, pathLenConstraint INTEGER OPTIONAL }`.
fn basic_constraints(value: &[u8]) -> Result<(bool, Option<u32>), X509Error> {
    let mut reader = Der::new(value);
    let sequence = reader.take(Tag::SEQUENCE)?;
    reader.expect_end()?;
    let mut fields = sequence.reader()?;
    let is_ca = match fields.take_if(Tag::BOOLEAN)? {
        Some(element) => element.as_boolean()?,
        None => false,
    };
    let path_len = match fields.take_if(Tag::INTEGER)? {
        Some(element) => {
            if !is_ca {
                // RFC 5280 §4.2.1.9: "CAs MUST NOT include the pathLenConstraint
                // field unless the cA boolean is asserted and the key usage
                // extension, if present, asserts the keyCertSign bit."
                return Err(X509Error::Invalid(
                    "pathLenConstraint is present without cA",
                ));
            }
            Some(element.as_u64()? as u32)
        }
        None => None,
    };
    fields.expect_end()?;
    Ok((is_ca, path_len))
}

/// `KeyUsage ::= BIT STRING`, at most nine bits.
fn key_usage(value: &[u8]) -> Result<KeyUsage, X509Error> {
    let mut reader = Der::new(value);
    let bits = reader.take(Tag::BIT_STRING)?;
    reader.expect_end()?;
    let contents = bits.contents();
    let (&unused, bytes) = contents
        .split_first()
        .ok_or(X509Error::Missing("the keyUsage bit string"))?;
    if unused > 7 {
        return Err(X509Error::Invalid(
            "the keyUsage unused-bit count is above 7",
        ));
    }
    if bytes.len() > 2 {
        return Err(X509Error::Invalid("keyUsage has more than nine bits"));
    }
    // X.509 numbers these bits from the most significant bit of the first content
    // octet, so the mapping is not a shift of the octet but a bit-by-bit walk.
    // Getting it wrong is silent: every value is still "a mask", just against the
    // wrong bit, and a `keyCertSign` check would then read `digitalSignature`.
    let mut mask = 0u16;
    for (index, &byte) in bytes.iter().enumerate() {
        for bit in 0..8u16 {
            if byte & (0x80u8 >> bit) != 0 {
                mask |= 1 << (index as u16 * 8 + bit);
            }
        }
    }
    Ok(KeyUsage(mask))
}

/// `ExtKeyUsageSyntax ::= SEQUENCE OF KeyPurposeId`.
fn extended_key_usage<'a>(value: &'a [u8]) -> Result<Vec<Oid<'a>>, X509Error> {
    let mut reader = Der::new(value);
    let sequence = reader.take(Tag::SEQUENCE)?;
    reader.expect_end()?;
    let mut purposes = Vec::new();
    let mut fields = sequence.reader()?;
    while !fields.is_empty() {
        purposes.push(fields.take(Tag::OID)?.as_oid()?);
    }
    if purposes.is_empty() {
        return Err(X509Error::Invalid("extKeyUsage is empty"));
    }
    Ok(purposes)
}

/// `SubjectAltName ::= GeneralNames`, a sequence of context-tagged choices.
fn subject_alt_names<'a>(value: &'a [u8]) -> Result<Vec<GeneralName<'a>>, X509Error> {
    let mut reader = Der::new(value);
    let sequence = reader.take(Tag::SEQUENCE)?;
    reader.expect_end()?;
    let mut names = Vec::new();
    let mut fields = sequence.reader()?;
    while !fields.is_empty() {
        let element = fields.next_element()?;
        let name = match element.tag().number() {
            1 if element.tag().class() == crate::der::Class::Context => {
                GeneralName::Rfc822(element.as_primitive()?)
            }
            2 if element.tag().class() == crate::der::Class::Context => {
                GeneralName::Dns(element.as_primitive()?)
            }
            6 if element.tag().class() == crate::der::Class::Context => {
                GeneralName::Uri(element.as_primitive()?)
            }
            7 if element.tag().class() == crate::der::Class::Context => {
                let address = element.as_primitive()?;
                if address.len() != 4 && address.len() != 16 {
                    return Err(X509Error::Invalid("an iPAddress is four or sixteen bytes"));
                }
                GeneralName::IpAddress(address)
            }
            _ => GeneralName::Other,
        };
        names.push(name);
    }
    if names.is_empty() {
        return Err(X509Error::Invalid("subjectAltName is empty"));
    }
    Ok(names)
}

/// `Validity ::= SEQUENCE { notBefore Time, notAfter Time }`.
fn parse_validity(validity: &Element<'_>) -> Result<(UnixTime, UnixTime), X509Error> {
    let mut fields = validity.reader()?;
    let not_before = parse_time(&fields.next_element()?)?;
    let not_after = parse_time(&fields.next_element()?)?;
    fields.expect_end()?;
    Ok((not_before, not_after))
}

/// A `Time`: `UTCTime` or `GeneralizedTime`, both required to be UTC.
///
/// RFC 5280 §4.1.2.5 fixes the forms: `YYMMDDHHMMSSZ` for years 1950-2049 and
/// `YYYYMMDDHHMMSSZ` outside that, always with seconds and always `Z`. No offsets,
/// no fractional seconds, no omitted seconds — a time that could be read two ways
/// is a time two implementations can disagree about, which is how a certificate
/// that has expired for one verifier is valid for another.
fn parse_time(element: &Element<'_>) -> Result<UnixTime, X509Error> {
    let text = element.as_primitive()?;
    let (year, rest) = match element.tag() {
        Tag::UTC_TIME => {
            if text.len() != 13 || text[12] != b'Z' {
                return Err(X509Error::Invalid("a UTCTime is exactly YYMMDDHHMMSSZ"));
            }
            // RFC 5280 §4.1.2.5.1: "Where YY is greater than or equal to 50, the
            // year SHALL be interpreted as 19YY; and where YY is less than 50,
            // the year SHALL be interpreted as 20YY."
            let yy = two_digits(text[0], text[1])? as i64;
            let year = if yy >= 50 { 1900 + yy } else { 2000 + yy };
            (year, &text[2..])
        }
        Tag::GENERALIZED_TIME => {
            if text.len() != 15 || text[14] != b'Z' {
                return Err(X509Error::Invalid("a GeneralizedTime is YYYYMMDDHHMMSSZ"));
            }
            let mut year = 0i64;
            for &digit in &text[..4] {
                year = year * 10 + (digit.wrapping_sub(b'0')) as i64;
            }
            if !text[..4].iter().all(u8::is_ascii_digit) {
                return Err(X509Error::Invalid("the year has a non-digit"));
            }
            (year, &text[4..])
        }
        _ => {
            return Err(X509Error::Invalid(
                "a validity time is UTCTime or GeneralizedTime",
            ))
        }
    };

    let month = two_digits(rest[0], rest[1])? as i64;
    let day = two_digits(rest[2], rest[3])? as i64;
    let hour = two_digits(rest[4], rest[5])? as i64;
    let minute = two_digits(rest[6], rest[7])? as i64;
    let second = two_digits(rest[8], rest[9])? as i64;

    if !(1..=12).contains(&month) {
        return Err(X509Error::Invalid("the month is not 1 to 12"));
    }
    if !(1..=days_in_month(year, month)).contains(&day) {
        return Err(X509Error::Invalid("the day is not in the month"));
    }
    if hour > 23 {
        return Err(X509Error::Invalid("the hour is above 23"));
    }
    if minute > 59 {
        return Err(X509Error::Invalid("the minute is above 59"));
    }
    if second > 59 {
        // Minutes with 60 seconds are leap seconds, which X.509 does not encode;
        // accepting one would make a certificate valid for an extra second in
        // some readers and not others.
        return Err(X509Error::Invalid("the second is above 59"));
    }

    Ok(UnixTime::from_secs(
        days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second,
    ))
}

fn two_digits(high: u8, low: u8) -> Result<u32, X509Error> {
    if !high.is_ascii_digit() || !low.is_ascii_digit() {
        return Err(X509Error::Invalid("a time field has a non-digit"));
    }
    Ok(((high - b'0') as u32) * 10 + (low - b'0') as u32)
}

fn is_leap(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap(year) => 29,
        2 => 28,
        _ => 0,
    }
}

/// Days from 1970-01-01 to a proleptic Gregorian date, which is all the calendar
/// arithmetic a certificate needs.
///
/// The month is shifted so the year starts in March, which puts the leap day at
/// the end of the year and makes the whole thing a single expression.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let year_of_era = year - era * 400; // 0..=399
    let day_of_year = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// `AlgorithmIdentifier ::= SEQUENCE { algorithm OID, parameters ANY OPTIONAL }`.
fn parse_signature_algorithm<'a>(
    element: &Element<'a>,
) -> Result<SignatureAlgorithm<'a>, X509Error> {
    let mut fields = element.reader()?;
    let oid = fields.take(Tag::OID)?.as_oid()?;
    match oid.as_bytes() {
        oids::SHA256_WITH_RSA => {
            // The parameters are NULL, and some encoders omit them; anything else
            // means an algorithm this crate has not read the specification of.
            match fields.next_element() {
                Err(DerError::Missing(_)) => {}
                Ok(parameters) => parameters.expect(Tag::NULL)?,
                Err(error) => return Err(error.into()),
            }
            fields.expect_end()?;
            Ok(SignatureAlgorithm::RsaPkcs1Sha256)
        }
        oids::ECDSA_WITH_SHA256 => {
            // RFC 5758 §3.2: the parameters MUST be absent.
            if fields.next_element().is_ok() {
                return Err(X509Error::Invalid(
                    "ecdsa-with-SHA256 must have no parameters",
                ));
            }
            Ok(SignatureAlgorithm::EcdsaSha256)
        }
        oids::RSASSA_PSS => {
            let salt_len = parse_pss_parameters(&mut fields)?;
            fields.expect_end()?;
            Ok(SignatureAlgorithm::RsaPssSha256 { salt_len })
        }
        _ => Ok(SignatureAlgorithm::Unsupported(oid)),
    }
}

/// `RSASSA-PSS-params`: hash, mask generation function, salt length, trailer.
///
/// Only SHA-256 with MGF1-SHA-256 is read, because that is the only combination
/// `rsa_pss_rsae_sha256` may name. A parameter set that says anything else is
/// refused: accepting one would mean verifying a signature with a scheme the
/// caller did not ask for.
fn parse_pss_parameters(fields: &mut Der<'_>) -> Result<u32, X509Error> {
    let parameters = fields.take(Tag::SEQUENCE)?;
    let mut reader = parameters.reader()?;
    let mut salt_len = None;

    // [0] hashAlgorithm, [1] maskGenAlgorithm, [2] saltLength, [3] trailerField.
    // All are DEFAULTed, so any may be absent; the defaults are SHA-1 and 20,
    // which this crate does not implement, so an absent hash is a rejection
    // rather than a silent SHA-1.
    if let Some(hash) = reader.take_if(Tag::context(0))? {
        let mut inner = hash.reader()?;
        let element = inner.take(Tag::SEQUENCE)?;
        inner.expect_end()?;
        let mut algorithm = element.reader()?;
        let oid = algorithm.take(Tag::OID)?.as_oid()?;
        match algorithm.next_element() {
            Err(DerError::Missing(_)) => {}
            Ok(parameters) => parameters.expect(Tag::NULL)?,
            Err(error) => return Err(error.into()),
        }
        if oid.as_bytes() != oids::SHA256 {
            return Err(X509Error::Unsupported("a PSS hash other than SHA-256"));
        }
        algorithm.expect_end()?;
    } else {
        // The default hash is SHA-1, which this crate does not implement, so an
        // absent hashAlgorithm is a rejection rather than a silent SHA-1.
        return Err(X509Error::Unsupported(
            "PSS without an explicit SHA-256 hash",
        ));
    }

    if let Some(mgf) = reader.take_if(Tag::context(1))? {
        let mut inner = mgf.reader()?;
        let element = inner.take(Tag::SEQUENCE)?;
        inner.expect_end()?;
        let mut algorithm = element.reader()?;
        let oid = algorithm.take(Tag::OID)?.as_oid()?;
        if oid.as_bytes() != oids::MGF1 {
            return Err(X509Error::Unsupported("a PSS mask other than MGF1"));
        }
        let parameters = algorithm.take(Tag::SEQUENCE)?;
        algorithm.expect_end()?;
        let mut mgf_hash = parameters.reader()?;
        let hash_oid = mgf_hash.take(Tag::OID)?.as_oid()?;
        match mgf_hash.next_element() {
            Err(DerError::Missing(_)) => {}
            Ok(parameters) => parameters.expect(Tag::NULL)?,
            Err(error) => return Err(error.into()),
        }
        mgf_hash.expect_end()?;
        if hash_oid.as_bytes() != oids::SHA256 {
            return Err(X509Error::Unsupported("an MGF1 hash other than SHA-256"));
        }
    } else {
        return Err(X509Error::Unsupported(
            "PSS without an explicit MGF1-SHA-256",
        ));
    }

    if let Some(salt) = reader.take_if(Tag::context(2))? {
        let mut inner = salt.reader()?;
        let value = inner.take(Tag::INTEGER)?.as_u64()?;
        inner.expect_end()?;
        salt_len = Some(u32::try_from(value).unwrap_or(u32::MAX));
    }

    // trailerField [3] must be the default (1) or absent.
    if let Some(trailer) = reader.take_if(Tag::context(3))? {
        let mut inner = trailer.reader()?;
        if inner.take(Tag::INTEGER)?.as_u64()? != 1 {
            return Err(X509Error::Unsupported("a PSS trailer other than 1"));
        }
        inner.expect_end()?;
    }
    reader.expect_end()?;

    // Every parameter that has a SHA-1 default has already been refused, so what
    // is left is the salt length, which has no default this crate would accept.
    salt_len.ok_or(X509Error::Unsupported(
        "PSS without an explicit salt length",
    ))
}

/// `SubjectPublicKeyInfo ::= SEQUENCE { algorithm, subjectPublicKey BIT STRING }`.
fn parse_public_key<'a>(spki: &Element<'a>) -> Result<PublicKey<'a>, X509Error> {
    let mut fields = spki.reader()?;
    let algorithm = fields.take(Tag::SEQUENCE)?;
    let key_bits = fields.take(Tag::BIT_STRING)?.as_bit_string()?;
    fields.expect_end()?;

    let mut algorithm_fields = algorithm.reader()?;
    let oid = algorithm_fields.take(Tag::OID)?.as_oid()?;
    match oid.as_bytes() {
        oids::RSA_ENCRYPTION => {
            // `parameters` is NULL for rsaEncryption (RFC 3279 §2.3.1).
            match algorithm_fields.next_element() {
                Err(DerError::Missing(_)) => {}
                Ok(parameters) => parameters.expect(Tag::NULL)?,
                Err(error) => return Err(error.into()),
            }
            algorithm_fields.expect_end()?;

            let mut key = Der::new(key_bits);
            let rsa = key.take(Tag::SEQUENCE)?;
            key.expect_end()?;
            let mut numbers = rsa.reader()?;
            let modulus_element = numbers.take(Tag::INTEGER)?;
            if modulus_element.as_integer_bytes()?.is_empty() {
                return Err(X509Error::Invalid("the RSA modulus is zero"));
            }
            let modulus = modulus_element.as_integer_bytes()?;
            let exponent = numbers.take(Tag::INTEGER)?.as_u64()?;
            numbers.expect_end()?;
            if exponent < 3 || exponent & 1 == 0 {
                return Err(X509Error::Invalid(
                    "the RSA exponent is not odd and at least 3",
                ));
            }
            Ok(PublicKey::Rsa { modulus, exponent })
        }
        oids::EC_PUBLIC_KEY => {
            // `parameters` is a namedCurve OID; TLS 1.3 requires P-256 support and
            // that is all this crate offers.
            let curve = algorithm_fields.take(Tag::OID)?.as_oid()?;
            algorithm_fields.expect_end()?;
            if curve.as_bytes() != oids::PRIME256V1 {
                return Err(X509Error::Unsupported(
                    "an elliptic curve other than prime256v1",
                ));
            }
            match key_bits {
                [0x04, ..] if key_bits.len() == 65 => Ok(PublicKey::Ec { point: key_bits }),
                [0x04, ..] => Err(X509Error::Invalid(
                    "a P-256 point is 65 bytes in the uncompressed format",
                )),
                _ => Err(X509Error::Unsupported("a compressed elliptic curve point")),
            }
        }
        _ => Ok(PublicKey::Unsupported(oid)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epoch(text: &str) -> i64 {
        text.parse().expect("a number")
    }

    #[test]
    fn the_calendar_matches_known_dates() {
        // Spot values against the Unix epoch, including two leap days and the
        // year 2000, which is a leap year and 1900, which is not.
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(1969, 12, 31), -1);
        assert_eq!(days_from_civil(2000, 1, 1), 10957);
        assert_eq!(days_from_civil(2000, 2, 29), 11016);
        // 1900 is not a leap year: 28 February and 1 March are one day apart.
        assert_eq!(days_from_civil(1900, 2, 28), -25509);
        assert_eq!(days_from_civil(1900, 3, 1), -25508);
        assert_eq!(days_from_civil(2024, 2, 29), 19782);
        assert_eq!(days_from_civil(2038, 1, 19), 24855);
        assert!(is_leap(2000) && is_leap(2024) && !is_leap(1900) && !is_leap(2100));
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2026, 7), 31);
    }

    fn time(tag: Tag, text: &[u8]) -> Element<'_> {
        // A DER element with the given tag, built by hand: the tests below are
        // about what the time parser does with the contents.
        let mut der = vec![0x18, text.len() as u8];
        der[0] = match tag {
            Tag::UTC_TIME => 0x17,
            _ => 0x18,
        };
        der.extend_from_slice(text);
        let boxed = der.leak();
        Der::new(boxed).next_element().expect("well framed")
    }

    #[test]
    fn utctime_uses_the_1950_pivot() {
        // The pivot is the single most dangerous line in this file: "50" is 1950
        // and "49" is 2049, and getting it backwards turns an expired certificate
        // into a valid one nine decades long.
        assert_eq!(
            parse_time(&time(Tag::UTC_TIME, b"490101000000Z")).unwrap(),
            UnixTime::from_secs(epoch("2493072000"))
        );
        assert_eq!(
            parse_time(&time(Tag::UTC_TIME, b"500101000000Z")).unwrap(),
            UnixTime::from_secs(epoch("-631152000"))
        );
        assert_eq!(
            parse_time(&time(Tag::UTC_TIME, b"700101000000Z")).unwrap(),
            UnixTime::epoch()
        );
        assert_eq!(
            parse_time(&time(Tag::UTC_TIME, b"260101000000Z")).unwrap(),
            UnixTime::from_secs(epoch("1767225600"))
        );
        // And an offset or an omitted seconds field is refused rather than
        // guessed at.
        for bad in [
            &b"2601010000Z"[..],
            b"260101000000",
            b"260101000000+0000",
            b"260101000000z",
            b"26010100000aZ",
            b"991301000000Z",
            b"260101250000Z",
            b"260101006000Z",
            b"260230000000Z",
        ] {
            assert!(
                parse_time(&time(Tag::UTC_TIME, bad)).is_err(),
                "{:?} was accepted",
                core::str::from_utf8(bad).unwrap_or("(binary)")
            );
        }
    }

    #[test]
    fn generalized_time_has_four_digit_years_and_no_pivot() {
        assert_eq!(
            parse_time(&time(Tag::GENERALIZED_TIME, b"20500601000000Z")).unwrap(),
            UnixTime::from_secs(epoch("2537654400"))
        );
        assert_eq!(
            parse_time(&time(Tag::GENERALIZED_TIME, b"19500601000000Z")).unwrap(),
            UnixTime::from_secs(epoch("-618105600"))
        );
        // Leap day accepted, and the day after a leap day in a non-leap year
        // refused.
        assert!(parse_time(&time(Tag::GENERALIZED_TIME, b"20240229120000Z")).is_ok());
        assert!(parse_time(&time(Tag::GENERALIZED_TIME, b"20230229120000Z")).is_err());
        // Fractional seconds and missing seconds are not this format.
        assert!(parse_time(&time(Tag::GENERALIZED_TIME, b"20500601000000.5Z")).is_err());
        assert!(parse_time(&time(Tag::GENERALIZED_TIME, b"205006010000Z")).is_err());
    }

    fn extension_value(oid: &[u8], value: &[u8]) -> Vec<u8> {
        // Extension ::= SEQUENCE { extnID OID, extnValue OCTET STRING }
        let mut body = vec![0x06, oid.len() as u8];
        body.extend_from_slice(oid);
        body.extend_from_slice(&[0x04, value.len() as u8]);
        body.extend_from_slice(value);
        let mut out = vec![0x30, body.len() as u8];
        out.extend_from_slice(&body);
        out
    }

    #[test]
    fn basic_constraints_reads_the_ca_flag_and_the_path_length() {
        // cA TRUE with pathLen 0: 30 06 01 01 ff 02 01 00
        assert_eq!(
            basic_constraints(&[0x30, 0x06, 0x01, 0x01, 0xff, 0x02, 0x01, 0x00]).unwrap(),
            (true, Some(0))
        );
        // cA TRUE with no path length.
        assert_eq!(
            basic_constraints(&[0x30, 0x03, 0x01, 0x01, 0xff]).unwrap(),
            (true, None)
        );
        // An empty sequence is cA FALSE, which is what "critical basicConstraints
        // with no cA" means in a leaf.
        assert_eq!(basic_constraints(&[0x30, 0x00]).unwrap(), (false, None));
        // cA explicitly FALSE.
        assert_eq!(
            basic_constraints(&[0x30, 0x03, 0x01, 0x01, 0x00]).unwrap(),
            (false, None)
        );
        // A path length without cA is forbidden.
        assert!(matches!(
            basic_constraints(&[0x30, 0x06, 0x01, 0x01, 0x00, 0x02, 0x01, 0x00]),
            Err(X509Error::Invalid(
                "pathLenConstraint is present without cA"
            ))
        ));
    }

    #[test]
    fn key_usage_bits_are_numbered_from_the_top_of_the_first_octet() {
        // digitalSignature: bit 0, so the unused-bit count is 7 and the octet is
        // 0x80.
        assert_eq!(key_usage(&[0x03, 0x02, 0x07, 0x80]).unwrap().0, 0x001);
        // keyCertSign and cRLSign: bits 5 and 6.
        assert_eq!(key_usage(&[0x03, 0x02, 0x01, 0x06]).unwrap().0, 0x060);
        // The full server-usage set from the fixtures: digitalSignature and
        // keyEncipherment are bits 0 and 2.
        let usage = key_usage(&[0x03, 0x02, 0x05, 0xa0]).unwrap();
        assert!(usage.has(KeyUsage::DIGITAL_SIGNATURE));
        assert!(usage.has(KeyUsage::KEY_ENCIPHERMENT));
        assert!(!usage.has(KeyUsage::KEY_CERT_SIGN));
        // Nine bits needs two octets; ten is too many.
        assert!(key_usage(&[0x03, 0x03, 0x07, 0x80, 0x80]).is_ok());
        assert!(key_usage(&[0x03, 0x04, 0x06, 0x80, 0x80, 0x80]).is_err());
    }

    #[test]
    fn subject_alt_names_keep_the_names_and_skip_the_rest() {
        // [2] dNSName "a.example", [1] rfc822Name "x@a.example", [7] iPAddress
        // 127.0.0.1, then a [6] URI.
        let mut names = Vec::new();
        for (tag, value) in [
            (0x82u8, &b"a.example"[..]),
            (0x81, b"x@a.example"),
            (0x87, &[127, 0, 0, 1]),
            (0x86, b"https://a.example/"),
        ] {
            names.push(tag);
            names.push(value.len() as u8);
            names.extend_from_slice(value);
        }
        let mut sequence = vec![0x30, names.len() as u8];
        sequence.extend_from_slice(&names);
        let parsed = subject_alt_names(&sequence).unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0], GeneralName::Dns(b"a.example"));
        assert_eq!(parsed[1], GeneralName::Rfc822(b"x@a.example"));
        assert_eq!(parsed[2], GeneralName::IpAddress(&[127, 0, 0, 1]));
        assert_eq!(parsed[3], GeneralName::Uri(b"https://a.example/"));

        // An iPAddress that is not four or sixteen bytes is refused rather than
        // compared as a name.
        let mut bad = vec![0x30, 0x03, 0x87, 0x01, 0x7f];
        assert!(subject_alt_names(&bad).is_err());
        // An empty sequence is not a usable subjectAltName.
        bad = vec![0x30, 0x00];
        assert!(subject_alt_names(&bad).is_err());
    }

    #[test]
    fn an_unknown_extension_is_skipped_but_a_critical_one_is_fatal() {
        let known = extension_value(oids::KEY_USAGE, &[0x03, 0x02, 0x07, 0x80]);
        let unknown_oid = &[0x2b, 0x06, 0x01, 0x04, 0x01, 0x86, 0x8d, 0x1f, 0x01];
        let unknown = {
            let mut value = vec![0x06, unknown_oid.len() as u8];
            value.extend_from_slice(unknown_oid);
            value.extend_from_slice(&[0x04, 0x03, 0x04, 0x01, 0x61]);
            let mut out = vec![0x30, value.len() as u8];
            out.extend_from_slice(&value);
            out
        };
        let critical_unknown = {
            // The same extension with `critical BOOLEAN TRUE` added.
            let mut value = vec![0x06, unknown_oid.len() as u8];
            value.extend_from_slice(unknown_oid);
            value.extend_from_slice(&[0x01, 0x01, 0xff]);
            value.extend_from_slice(&[0x04, 0x03, 0x04, 0x01, 0x61]);
            let mut out = vec![0x30, value.len() as u8];
            out.extend_from_slice(&value);
            out
        };

        let list = |body: &[u8]| {
            let mut der = vec![0x30, body.len() as u8];
            der.extend_from_slice(body);
            // Leaked so the borrow outlives the call, which is fine in a test.
            let der = der.leak();
            let element = Der::new(der).next_element().unwrap();
            Extensions::parse(&element)
        };

        let parsed = list(&[known.clone(), unknown.clone()].concat()).unwrap();
        assert_eq!(parsed.key_usage.unwrap().0, 0x001, "the known one is read");
        assert_eq!(parsed.seen.len(), 2, "the unknown one is noted and skipped");

        let error = list(&[known.clone(), critical_unknown].concat()).unwrap_err();
        assert!(matches!(error, X509Error::UnsupportedCriticalExtension(_)));
        assert!(error.to_string().contains("99999") || error.to_string().contains("2b06010401"));

        // The same extension twice is two answers to one question.
        let error = list(&[known.clone(), known].concat()).unwrap_err();
        assert_eq!(
            error,
            X509Error::Invalid("an extension appears more than once")
        );
    }
}
