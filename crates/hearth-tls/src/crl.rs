//! Reading certificate revocation lists (RFC 5280 §5), strictly.
//!
//! The counterpart of [`crate::x509`] for the other structure a verifier reads: a
//! CRL says which certificates an issuer has revoked, and it is the only
//! revocation information this crate reads — there is no OCSP and no stapling.
//!
//! # What this module is, and what it is not
//!
//! It **reads** a CRL: the issuer name, the two update times, the revoked serial
//! numbers, and the bytes a signature covers. It has no opinion about trust — a CRL
//! is accepted here if it is the encoding it claims to be, whether or not it was
//! signed by anything — and it does not decide when a CRL applies. Both of those
//! are the verifier's business, in [`crate::chain`].
//!
//! Nothing here fetches anything. A CRL arrives as DER from the caller, exactly
//! like a trust anchor, because this crate does no I/O: fetching a CRL over HTTP and
//! re-fetching it before `nextUpdate` is the caller's job, and so is deciding what
//! to do when a refresh fails.
//!
//! # The rules, and why each one is here
//!
//! - The outer structure is exactly three elements with nothing left over, and the
//!   signature algorithm inside the signed `TBSCertList` is **byte-identical** to the
//!   one outside it (§5.1.1.2, the same requirement certificates carry) — a CRL where
//!   the two differ is a signature checked under one algorithm while the fields were
//!   read under another.
//! - A **critical** extension this crate does not understand is a rejection, as §4.2
//!   and §5.2 require, on the CRL itself and on each revoked entry. This is what
//!   refuses a `deltaCRLIndicator`: a delta CRL lists only the changes since a base
//!   CRL, so reading it as a complete one would report a certificate as *not* revoked
//!   that the base revokes. OpenSSL 3.5.5 accepts it in that position; this crate is
//!   deliberately the stricter side.
//! - An entry carrying `certificateIssuer` is refused by name rather than read past,
//!   whatever its criticality: that extension is how an *indirect* CRL says an entry
//!   belongs to a different issuer, and honoring it means deciding which issuer a CRL
//!   is about entry by entry. Not implemented, so not accepted.
//! - `version` is v1 when absent and must be v2 (the INTEGER 1) when present; a v1 CRL
//!   that carries extensions is refused, because §5.1.2.1 forbids exactly that
//!   combination. Both formats are read, and the v1 one has no extensions in which to
//!   carry a delta indicator or an indirect-CRL statement.
//! - `nextUpdate` is optional (§5.1.2.5) and its absence means "no stated expiry", not
//!   "expired": a CRL without one is read, and the verifier says what it does with it.
//! - Each entry's serial number is read the way an `INTEGER` is: positive, minimally
//!   encoded, at most 20 octets like §4.1.2.2 caps a certificate's, because an entry
//!   exists to name a certificate's serial number and a longer one can name none.
//! - A signature is the `BIT STRING`'s contents with its unused-bits octet accounted
//!   for, so what a caller hands to the verifier is the signature and not the
//!   signature plus a length byte.

use crate::client::UnixTime;
use crate::der::{Der, DerError, Element, Tag};
use crate::x509::{parse_signature_algorithm, parse_time, Name, SignatureAlgorithm, X509Error};

/// Why a CRL was refused.
///
/// Separate from [`X509Error`] only because the messages differ — a CRL is not a
/// certificate, and a diagnostic that says "the certificate has no nextUpdate" about a
/// CRL sends a reader looking in the wrong place. The rules are the same ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrlError {
    /// The encoding is not the DER this reader accepts.
    Der(DerError),
    /// A field the profile requires is absent.
    Missing(&'static str),
    /// A field is present but not the shape RFC 5280 gives it.
    Invalid(&'static str),
    /// A critical extension this crate does not implement (RFC 5280 §5.2).
    UnsupportedCriticalExtension(Vec<u8>),
    /// The CRL is well-formed but uses something this crate cannot read.
    Unsupported(&'static str),
}

impl std::fmt::Display for CrlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Der(error) => write!(f, "{error}"),
            Self::Missing(what) => write!(f, "the CRL has no {what}"),
            Self::Invalid(why) => write!(f, "invalid CRL: {why}"),
            Self::UnsupportedCriticalExtension(oid) => write!(
                f,
                "the CRL has a critical extension this client does not understand ({})",
                oid.iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
            ),
            Self::Unsupported(what) => write!(f, "unsupported CRL: {what}"),
        }
    }
}

impl std::error::Error for CrlError {}

impl From<DerError> for CrlError {
    fn from(error: DerError) -> Self {
        Self::Der(error)
    }
}

impl From<X509Error> for CrlError {
    // Only two things are shared with `x509` — the signature `AlgorithmIdentifier` and
    // the reading of a `Time` — and only their refusals can arrive here. Mapping every
    // variant preserves the message rather than losing it.
    fn from(error: X509Error) -> Self {
        match error {
            X509Error::Der(error) => Self::Der(error),
            X509Error::Missing(what) => Self::Missing(what),
            X509Error::Invalid(why) => Self::Invalid(why),
            X509Error::UnsupportedCriticalExtension(oid) => Self::UnsupportedCriticalExtension(oid),
            X509Error::Unsupported(what) => Self::Unsupported(what),
        }
    }
}

/// `certificateIssuer`, the entry extension that says an entry is about a certificate
/// issued by *another* CA (RFC 5280 §5.3.3).
///
/// Named so that it can be refused whether or not it is marked critical: honoring it
/// means resolving the issuer named inside the entry rather than the CRL's own issuer.
pub const CERTIFICATE_ISSUER: &[u8] = &[0x55, 0x1d, 0x1d];

/// One `revokedCertificates` entry: a serial number and when it was revoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevokedCertificate<'a> {
    serial: &'a [u8],
    revocation_date: UnixTime,
}

impl<'a> RevokedCertificate<'a> {
    /// The revoked certificate's serial number, as its magnitude: big-endian, with no
    /// leading zero. Comparable with [`crate::x509::Certificate::serial`], which is
    /// read the same way for the same reason.
    pub fn serial(&self) -> &'a [u8] {
        self.serial
    }

    /// The entry's `revocationDate`.
    pub fn revocation_date(&self) -> UnixTime {
        self.revocation_date
    }
}

/// A parsed CRL, borrowing from the DER it was read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CertificateRevocationList<'a> {
    der: &'a [u8],
    tbs: &'a [u8],
    version: u8,
    issuer: Name<'a>,
    this_update: UnixTime,
    next_update: Option<UnixTime>,
    revoked: Vec<RevokedCertificate<'a>>,
    signature_algorithm: SignatureAlgorithm<'a>,
    signature: &'a [u8],
}

impl<'a> CertificateRevocationList<'a> {
    /// Reads a CRL.
    ///
    /// Every rule in the module documentation is applied here. A failure means the bytes
    /// are not the CRL they claim to be — never that the CRL was merely unusual, and
    /// never that it was untrusted.
    pub fn parse(der: &'a [u8]) -> Result<Self, CrlError> {
        let mut top = Der::new(der);
        let list = top.take(Tag::SEQUENCE)?;
        top.expect_end()?;

        let mut fields = list.reader()?;
        let tbs_element = fields.take(Tag::SEQUENCE)?;
        let tbs = tbs_element.whole();
        let signature_algorithm_element = fields.take(Tag::SEQUENCE)?;
        let signature = fields.take(Tag::BIT_STRING)?.as_bit_string()?;
        fields.expect_end()?;

        let mut body = tbs_element.reader()?;

        // version Version DEFAULT v1: absent means v1, and §5.1.2.1 writes v2 as the
        // INTEGER 1.
        //
        // It is a plain `INTEGER`, **not** the `[0] EXPLICIT` a certificate wraps its
        // version in: `TBSCertList` starts with this field, so a bare INTEGER at that
        // position can be nothing else, while a certificate's `TBSCertificate` starts
        // with a serial number — another INTEGER — and needs the tag to tell them apart.
        // Reading the certificate's shape here refused every real v2 CRL, and did not
        // show up until a CRL built by OpenSSL was put through this parser instead of
        // one built by the same mistake (see `docs/BUG_TRACKER.md`).
        let version = match body.take_if(Tag::INTEGER)? {
            None => 1,
            Some(element) => match element.as_u64()? {
                0 => 1,
                1 => 2,
                _ => return Err(CrlError::Invalid("the version is neither v1 nor v2")),
            },
        };

        // The algorithm inside the signed part must be the one outside it, byte for byte,
        // exactly as in a certificate (§5.1.1.2).
        let tbs_algorithm = body.take(Tag::SEQUENCE)?;
        if tbs_algorithm.whole() != signature_algorithm_element.whole() {
            return Err(CrlError::Invalid(
                "the signature algorithm inside the CRL differs from the one outside it",
            ));
        }
        let signature_algorithm = parse_signature_algorithm(&signature_algorithm_element)?;

        let issuer_element = body.take(Tag::SEQUENCE)?;
        let issuer = Name::from_der(issuer_element.whole());

        let this_update = parse_time(&body.next_element()?)?;
        // nextUpdate is OPTIONAL. Its absence means the CRL states no expiry.
        let next_update = match body.take_if(Tag::UTC_TIME)? {
            Some(element) => Some(parse_time(&element)?),
            None => match body.take_if(Tag::GENERALIZED_TIME)? {
                Some(element) => Some(parse_time(&element)?),
                None => None,
            },
        };
        if let Some(next) = next_update {
            if next < this_update {
                return Err(CrlError::Invalid("nextUpdate is before thisUpdate"));
            }
        }

        // revokedCertificates SEQUENCE OF SEQUENCE OPTIONAL: absent means nothing is
        // revoked, which is the common case rather than an error.
        let mut revoked = Vec::new();
        if let Some(entries) = body.take_if(Tag::SEQUENCE)? {
            let mut reader = entries.reader()?;
            while !reader.is_empty() {
                revoked.push(parse_entry(&mut reader)?);
            }
        }

        // crlExtensions [0] EXPLICIT Extensions OPTIONAL.
        if let Some(explicit) = body.take_if(Tag::context(0))? {
            if version == 1 {
                // §5.1.2.1: "Conforming CRL issuers MUST NOT use the v1 CRL format if
                // ... extensions are present in the CRL."
                return Err(CrlError::Invalid("a v1 CRL carries extensions"));
            }
            let mut inner = explicit.reader()?;
            let list = inner.take(Tag::SEQUENCE)?;
            inner.expect_end()?;
            check_extensions(&list, false)?;
        }
        body.expect_end()?;

        Ok(Self {
            der,
            tbs,
            version,
            issuer,
            this_update,
            next_update,
            revoked,
            signature_algorithm,
            signature,
        })
    }

    /// The whole CRL, as it was read.
    pub fn der(&self) -> &'a [u8] {
        self.der
    }

    /// The `TBSCertList`: exactly the bytes its signature covers.
    ///
    /// The original bytes rather than a re-encoding, so "what was signed" and "what was
    /// read" cannot be made to differ.
    pub fn tbs(&self) -> &'a [u8] {
        self.tbs
    }

    /// The version: 1 or 2.
    pub fn version(&self) -> u8 {
        self.version
    }

    /// The issuing name, as its own DER. A CRL applies to a certificate when this is
    /// byte for byte that certificate's *issuer* name.
    pub fn issuer(&self) -> Name<'a> {
        self.issuer
    }

    /// `thisUpdate`: when this CRL's information was published.
    pub fn this_update(&self) -> UnixTime {
        self.this_update
    }

    /// `nextUpdate`, when the CRL states one: the instant after which it is stale and
    /// must be replaced.
    pub fn next_update(&self) -> Option<UnixTime> {
        self.next_update
    }

    /// Every revoked entry, in the order the CRL lists them.
    pub fn revoked(&self) -> &[RevokedCertificate<'a>] {
        &self.revoked
    }

    /// Whether `serial` is listed as revoked.
    ///
    /// `serial` is compared as the magnitude bytes both sides are read into, so a
    /// certificate whose serial needed a leading zero inside its `INTEGER` matches an
    /// entry carrying the same number: the pad belongs to the encoding, not the number.
    pub fn is_revoked(&self, serial: &[u8]) -> bool {
        self.revoked.iter().any(|entry| entry.serial == serial)
    }

    /// The signature algorithm, which the parser has already required to match the one
    /// inside the signed part.
    pub fn signature_algorithm(&self) -> &SignatureAlgorithm<'a> {
        &self.signature_algorithm
    }

    /// The signature bytes, from the `BIT STRING`.
    pub fn signature(&self) -> &'a [u8] {
        self.signature
    }
}

/// One entry of `revokedCertificates`.
///
/// `revocationDate` is read — it is required, and a caller may want it for a diagnostic
/// — but not acted on: whether a certificate was revoked *after* the instant being
/// checked is a question about time travel, and a listed serial is what a verifier
/// treats as revoked.
fn parse_entry<'a>(reader: &mut Der<'a>) -> Result<RevokedCertificate<'a>, CrlError> {
    let entry = reader.take(Tag::SEQUENCE)?;
    let mut fields = entry.reader()?;
    let serial_element = fields.take(Tag::INTEGER)?;
    let serial = serial_element.as_integer_bytes()?;
    // A certificate's serial is at most 20 octets (§4.1.2.2) and an entry exists to name
    // one, so a longer entry can name no certificate this crate would accept.
    if serial.len() > 20 {
        return Err(CrlError::Invalid(
            "a revoked entry names a serial number longer than 20 octets",
        ));
    }
    let revocation_date = parse_time(&fields.next_element()?)?;
    if let Some(extensions) = fields.take_if(Tag::SEQUENCE)? {
        check_extensions(&extensions, true)?;
    }
    fields.expect_end()?;
    Ok(RevokedCertificate {
        serial,
        revocation_date,
    })
}

/// Walks an extension list, refusing what this crate must not ignore.
///
/// Three rules, all from RFC 5280: an extension appears at most once, a **critical**
/// extension that is not understood is fatal (§4.2 for certificates, §5.2 for CRLs), and
/// on an entry `certificateIssuer` is fatal whether or not it is critical — see
/// [`CERTIFICATE_ISSUER`]. `entry` selects which of the two lists this is, because the
/// by-name refusal is the entry's alone.
fn check_extensions(list: &Element<'_>, entry: bool) -> Result<(), CrlError> {
    let mut reader = list.reader()?;
    let mut seen: Vec<&[u8]> = Vec::new();
    while !reader.is_empty() {
        let extension = reader.take(Tag::SEQUENCE)?;
        let mut fields = extension.reader()?;
        let oid = fields.take(Tag::OID)?.as_oid()?;
        let critical = match fields.take_if(Tag::BOOLEAN)? {
            Some(value) => value.as_boolean()?,
            // DEFAULT FALSE: absent means not critical.
            None => false,
        };
        // Read so that a malformed value is refused here rather than later: an extension
        // whose contents are not an OCTET STRING is not an extension.
        let _ = fields.take(Tag::OCTET_STRING)?.as_primitive()?;
        fields.expect_end()?;

        if seen.iter().any(|seen| *seen == oid.as_bytes()) {
            return Err(CrlError::Invalid("an extension appears more than once"));
        }
        seen.push(oid.as_bytes());

        if entry && oid.as_bytes() == CERTIFICATE_ISSUER {
            return Err(CrlError::Unsupported(
                "an entry naming another certificate issuer (an indirect CRL)",
            ));
        }
        if critical {
            return Err(CrlError::UnsupportedCriticalExtension(
                oid.as_bytes().to_vec(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DER `SEQUENCE` around `contents`, with a minimal length.
    fn sequence(contents: &[u8]) -> Vec<u8> {
        let mut out = vec![0x30];
        if contents.len() < 0x80 {
            out.push(contents.len() as u8);
        } else {
            out.push(0x81);
            out.push(contents.len() as u8);
        }
        out.extend_from_slice(contents);
        out
    }

    fn integer(contents: &[u8]) -> Vec<u8> {
        let mut out = vec![0x02, contents.len() as u8];
        out.extend_from_slice(contents);
        out
    }

    fn utc_time(text: &str) -> Vec<u8> {
        let mut out = vec![0x17, text.len() as u8];
        out.extend_from_slice(text.as_bytes());
        out
    }

    /// `sha256WithRSAEncryption` with `NULL` parameters: the bytes a CA signs with.
    fn algorithm() -> Vec<u8> {
        let mut oid = vec![0x06, 0x09];
        oid.extend_from_slice(&[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b]);
        oid.extend_from_slice(&[0x05, 0x00]);
        sequence(&oid)
    }

    /// A minimal issuer name: `SEQUENCE { SET { SEQUENCE { OID, PrintableString } } }`.
    fn issuer_name(text: &str) -> Vec<u8> {
        let mut printable = vec![0x13, text.len() as u8];
        printable.extend_from_slice(text.as_bytes());
        let mut attribute = vec![0x06, 0x03, 0x55, 0x04, 0x03];
        attribute.extend_from_slice(&printable);
        let attribute = sequence(&attribute);
        let mut set = vec![0x31, attribute.len() as u8];
        set.extend_from_slice(&attribute);
        sequence(&set)
    }

    /// One entry: `SEQUENCE { serial INTEGER, revocationDate UTCTime }`, plus
    /// `extensions` verbatim when given.
    fn entry_with(serial: &[u8], date: &str, extensions: Option<&[u8]>) -> Vec<u8> {
        let mut body = integer(serial);
        body.extend_from_slice(&utc_time(date));
        if let Some(extensions) = extensions {
            body.extend_from_slice(extensions);
        }
        sequence(&body)
    }

    fn entry(serial: &[u8], date: &str) -> Vec<u8> {
        entry_with(serial, date, None)
    }

    /// The `revokedCertificates` element around `entries`.
    fn revoked(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut contents = Vec::new();
        for entry in entries {
            contents.extend_from_slice(entry);
        }
        sequence(&contents)
    }

    /// A whole CRL around a hand-built `TBSCertList`.
    ///
    /// `version` is the version INTEGER's contents (`None` leaves the field out
    /// entirely, which means v1), `tbs_fields` are the TBS's fields before the
    /// algorithm, and `tail` is whatever follows the two update times — the revoked
    /// list, extensions, both or neither. The signature bytes are opaque to the parser
    /// and never checked here.
    fn crl_der(version: Option<&[u8]>, tbs_fields: &[Vec<u8>], tail: &[u8]) -> Vec<u8> {
        let mut tbs = Vec::new();
        if let Some(version) = version {
            tbs.extend_from_slice(&integer(version));
        }
        for field in tbs_fields {
            tbs.extend_from_slice(field);
        }
        tbs.extend_from_slice(tail);
        let tbs = sequence(&tbs);
        let mut body = tbs;
        body.extend_from_slice(&algorithm());
        body.extend_from_slice(&[0x03, 0x05, 0x00, 0xaa, 0xbb, 0xcc, 0xdd]);
        sequence(&body)
    }

    /// The two update times, which almost every case below wants unchanged.
    fn window() -> Vec<u8> {
        let mut out = utc_time("250101000000Z");
        out.extend_from_slice(&utc_time("270101000000Z"));
        out
    }

    /// A CRL with no revoked entries, both update times and a v2 version.
    fn simple() -> Vec<u8> {
        let mut tbs_fields = algorithm();
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        crl_der(Some(&[0x01]), &[tbs_fields], &window())
    }

    /// A CRL whose TBS holds `entries` as its revoked list.
    fn with_entries(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut tbs_fields = algorithm();
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        let mut tail = window();
        tail.extend_from_slice(&revoked(entries));
        crl_der(Some(&[0x01]), &[tbs_fields], &tail)
    }

    /// A CRL whose `crlExtensions` element is the extension list `extensions`.
    fn with_crl_extensions(version: Option<&[u8]>, extensions: Vec<u8>) -> Vec<u8> {
        let mut tbs_fields = algorithm();
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        let mut tail = window();
        tail.push(0xa0);
        tail.push(extensions.len() as u8);
        tail.extend_from_slice(&extensions);
        crl_der(version, &[tbs_fields], &tail)
    }

    /// One extension: `SEQUENCE { OID, [critical BOOLEAN], OCTET STRING }`.
    fn extension(oid: &[u8], critical: bool, value: &[u8]) -> Vec<u8> {
        let mut body = vec![0x06, oid.len() as u8];
        body.extend_from_slice(oid);
        if critical {
            body.extend_from_slice(&[0x01, 0x01, 0xff]);
        }
        body.push(0x04);
        body.push(value.len() as u8);
        body.extend_from_slice(value);
        sequence(&body)
    }

    #[test]
    fn a_well_formed_crl_reads_into_its_fields() {
        let der = simple();
        let parsed = CertificateRevocationList::parse(&der).expect("a CRL");
        assert_eq!(parsed.der(), &der[..]);
        assert_eq!(parsed.version(), 2);
        // The TBS is a subslice of the input — the bytes the signature covers — and not a
        // re-encoding of the fields.
        assert_eq!(parsed.tbs(), &der[2..der.len() - algorithm().len() - 7]);
        assert_eq!(parsed.this_update().secs, 1_735_689_600);
        assert_eq!(
            parsed.next_update().map(|time| time.secs),
            Some(1_798_761_600)
        );
        assert!(parsed.revoked().is_empty());
        assert_eq!(
            parsed
                .issuer()
                .common_name()
                .expect("a name")
                .map(<[u8]>::to_vec),
            Some(b"Test CA".to_vec())
        );
    }

    #[test]
    fn entries_are_read_with_their_serials_and_dates() {
        // The second serial is the number 0x8000, whose `INTEGER` needs a leading zero
        // octet so that it is not read as negative. The pad belongs to the encoding and
        // not to the number, which is what `serial()` removing it is for.
        let der = with_entries(&[
            entry(&[0x01], "250102000000Z"),
            entry(&[0x00, 0x80, 0x00], "250103000000Z"),
        ]);
        let parsed = CertificateRevocationList::parse(&der).expect("a CRL");
        assert_eq!(parsed.revoked().len(), 2);
        assert_eq!(parsed.revoked()[0].serial(), &[0x01]);
        assert_eq!(parsed.revoked()[0].revocation_date().secs, 1_735_776_000);
        assert_eq!(parsed.revoked()[1].serial(), &[0x80, 0x00]);
        assert!(parsed.is_revoked(&[0x01]));
        assert!(parsed.is_revoked(&[0x80, 0x00]));
        // A serial that is a prefix of a revoked one is a different number.
        assert!(!parsed.is_revoked(&[0x80]));
        assert!(!parsed.is_revoked(&[0x02]));
        assert!(!parsed.is_revoked(&[]));
    }

    #[test]
    fn a_v1_crl_has_no_version_field_and_may_not_carry_extensions() {
        let mut tbs_fields = algorithm();
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        let der = crl_der(None, &[tbs_fields], &window());
        let parsed = CertificateRevocationList::parse(&der).expect("a v1 CRL");
        assert_eq!(parsed.version(), 1);
        assert_eq!(parsed.this_update().secs, 1_735_689_600);

        // §5.1.2.1: a v1 CRL may not carry extensions, so this combination is refused
        // however well-formed the extension itself is.
        let crl_number = extension(&[0x55, 0x1d, 0x14], false, &[0x01]);
        let der = with_crl_extensions(None, sequence(&crl_number));
        assert_eq!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Invalid("a v1 CRL carries extensions"))
        );
        // The same extensions on a v2 CRL are read, which is what makes the rule above
        // about the version rather than about the extensions.
        let der = with_crl_extensions(Some(&[0x01]), sequence(&crl_number));
        assert!(CertificateRevocationList::parse(&der).is_ok());
    }

    #[test]
    fn a_crl_without_next_update_states_no_expiry() {
        let mut tbs_fields = algorithm();
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        let der = crl_der(Some(&[0x01]), &[tbs_fields], &utc_time("250101000000Z"));
        let parsed = CertificateRevocationList::parse(&der).expect("a CRL");
        assert_eq!(parsed.next_update(), None);
        assert_eq!(parsed.this_update().secs, 1_735_689_600);
    }

    #[test]
    fn the_two_signature_algorithms_must_agree_byte_for_byte() {
        // The TBS's own algorithm replaced, with the outer one left alone: the fields are
        // the same and only the algorithm inside the signed part moved.
        let mut inner = algorithm();
        inner[2] = 0x0a; // id-RSASSA-PSS instead of sha256WithRSAEncryption
        let mut tbs_fields = inner;
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        let der = crl_der(Some(&[0x01]), &[tbs_fields], &window());
        assert_eq!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Invalid(
                "the signature algorithm inside the CRL differs from the one outside it"
            ))
        );
    }

    #[test]
    fn the_version_is_a_plain_integer_and_must_be_v1_or_v2() {
        // Present as v2.
        let mut tbs_fields = algorithm();
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        let der = crl_der(Some(&[0x01]), &[tbs_fields], &window());
        assert_eq!(
            CertificateRevocationList::parse(&der).map(|crl| crl.version()),
            Ok(2)
        );
        // Present as v1(0), which the ASN.1 allows even though the profile requires the
        // field to be absent for v1 — reading it is not a leniency that changes anything.
        let mut tbs_fields = algorithm();
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        let der = crl_der(Some(&[0x00]), &[tbs_fields], &window());
        assert_eq!(
            CertificateRevocationList::parse(&der).map(|crl| crl.version()),
            Ok(1)
        );
        for value in [0x02u8, 0x7f] {
            let mut tbs_fields = algorithm();
            tbs_fields.extend_from_slice(&issuer_name("Test CA"));
            let der = crl_der(Some(&[value]), &[tbs_fields], &window());
            assert_eq!(
                CertificateRevocationList::parse(&der),
                Err(CrlError::Invalid("the version is neither v1 nor v2")),
                "version {value}"
            );
        }
        // And the certificate's `[0] EXPLICIT` form is *not* what a CRL uses: a CRL
        // wrapped that way is refused here for the same reason it is unreadable to
        // OpenSSL, which is the bug this rule was written to fix.
        let mut tbs = vec![0xa0, 0x03, 0x02, 0x01, 0x01];
        tbs.extend_from_slice(&algorithm());
        tbs.extend_from_slice(&issuer_name("Test CA"));
        tbs.extend_from_slice(&window());
        let tbs = sequence(&tbs);
        let der = sequence(
            &[
                &tbs[..],
                &algorithm()[..],
                &[0x03, 0x05, 0x00, 0xaa, 0xbb, 0xcc, 0xdd][..],
            ]
            .concat(),
        );
        assert!(matches!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Der(_))
        ));
    }

    #[test]
    fn a_reversed_window_is_refused() {
        let mut tbs_fields = algorithm();
        tbs_fields.extend_from_slice(&issuer_name("Test CA"));
        let mut tail = utc_time("270101000000Z");
        tail.extend_from_slice(&utc_time("250101000000Z"));
        let der = crl_der(Some(&[0x01]), &[tbs_fields], &tail);
        assert_eq!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Invalid("nextUpdate is before thisUpdate"))
        );
    }

    #[test]
    fn a_redundant_serial_encoding_is_refused_rather_than_normalized() {
        // 0x00 0x01 is a non-minimal INTEGER: the same number as 0x01, and two encodings
        // of one serial number is exactly what a revocation list must not contain.
        let der = with_entries(&[entry(&[0x00, 0x01], "250102000000Z")]);
        assert!(matches!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Der(DerError::NonMinimalLength(_)))
        ));
    }

    #[test]
    fn a_negative_or_oversized_serial_is_refused() {
        let der = with_entries(&[entry(&[0xff], "250102000000Z")]);
        assert!(matches!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Der(DerError::Invalid(_)))
        ));

        let long = [0x7fu8; 21];
        let der = with_entries(&[entry(&long, "250102000000Z")]);
        assert_eq!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Invalid(
                "a revoked entry names a serial number longer than 20 octets"
            ))
        );
    }

    #[test]
    fn an_entry_naming_another_issuer_is_refused_whether_it_is_critical_or_not() {
        // certificateIssuer is critical in a conforming CRL. Both forms are refused, and
        // the non-critical one is the case a critical-only rule would have read past.
        for critical in [true, false] {
            let extensions = sequence(&extension(CERTIFICATE_ISSUER, critical, &[0x30, 0x00]));
            let der = with_entries(&[entry_with(&[0x01], "250102000000Z", Some(&extensions))]);
            assert_eq!(
                CertificateRevocationList::parse(&der),
                Err(CrlError::Unsupported(
                    "an entry naming another certificate issuer (an indirect CRL)"
                )),
                "critical={critical}"
            );
        }
    }

    #[test]
    fn an_unknown_critical_extension_is_fatal_on_the_crl_and_on_an_entry() {
        // On the CRL: deltaCRLIndicator, which must not stand in for a complete CRL.
        let extensions = sequence(&extension(&[0x55, 0x1d, 0x1b], true, &[0x02, 0x01, 0x07]));
        let der = with_crl_extensions(Some(&[0x01]), extensions);
        assert_eq!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::UnsupportedCriticalExtension(vec![
                0x55, 0x1d, 0x1b
            ]))
        );

        // The same idea on one entry, with an OID nobody has assigned.
        let extensions = sequence(&extension(&[0x2a, 0x03, 0x04], true, &[0x05, 0x00]));
        let der = with_entries(&[entry_with(&[0x01], "250102000000Z", Some(&extensions))]);
        assert_eq!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::UnsupportedCriticalExtension(vec![
                0x2a, 0x03, 0x04
            ]))
        );

        // And a non-critical entry extension is read past: a revoke *reason* changes no
        // decision a verifier makes.
        let extensions = sequence(&extension(&[0x55, 0x1d, 0x15], false, &[0x0a, 0x01, 0x04]));
        let der = with_entries(&[entry_with(&[0x01], "250102000000Z", Some(&extensions))]);
        assert!(CertificateRevocationList::parse(&der).is_ok());
    }

    #[test]
    fn an_extension_appears_at_most_once() {
        let mut duplicate = extension(&[0x55, 0x1d, 0x14], false, &[0x01]);
        duplicate.extend_from_slice(&extension(&[0x55, 0x1d, 0x14], false, &[0x02]));
        let extensions = sequence(&duplicate);
        let der = with_crl_extensions(Some(&[0x01]), extensions);
        assert_eq!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Invalid("an extension appears more than once"))
        );
    }

    #[test]
    fn trailing_bytes_and_a_truncated_input_are_refused() {
        let mut der = simple();
        der.push(0x00);
        assert!(matches!(
            CertificateRevocationList::parse(&der),
            Err(CrlError::Der(DerError::TrailingData(1)))
        ));
        // Half a CRL is not a CRL: the outer length promises bytes that are not there.
        let der = simple();
        assert!(matches!(
            CertificateRevocationList::parse(&der[..der.len() - 1]),
            Err(CrlError::Der(DerError::Truncated(_)))
        ));
        assert!(matches!(
            CertificateRevocationList::parse(&[]),
            Err(CrlError::Der(_))
        ));
    }
}
