//! Reading DER (X.690), strictly.
//!
//! Enough ASN.1 to read an X.509 certificate and nothing more, and deliberately
//! intolerant: DER exists precisely so that one value has exactly one encoding, and
//! every place this reader could accept a second one is a place where two
//! implementations can disagree about what a certificate says. Certificate
//! parsing has a long history of bugs of exactly that shape — a signature that
//! covers different bytes than the fields that were read, a length that means one
//! thing to the verifier and another to the consumer — so this module refuses
//! rather than repairs, and each refusal has a test.
//!
//! What is refused, and why:
//!
//! - **Indefinite lengths** (`0x80`). Forbidden by DER, and the reason is the
//!   reason this module exists: `30 80 … 00 00` lets an attacker choose where a
//!   sequence ends, so two readers can disagree.
//! - **Non-minimal lengths.** A long-form length whose first octet is zero
//!   (`81 05` instead of `05`) is well-formed BER and has a second encoding for
//!   the same value.
//! - **High-tag-number form** (tag octets with the low five bits set). No field
//!   this crate needs has a tag number of 31 or more, and refusing the form
//!   outright is cheaper to audit than supporting it. An `Option` and a
//!   `Context` field can both be encoded with it, which is how a field can be made
//!   to appear *not* to be there.
//! - **Truncated contents**, **any length octets that run past the end**, and
//!   **trailing bytes** after the fields that were read. A certificate with
//!   unobserved bytes is a certificate whose signature may cover something else.
//! - **Non-minimal or negative `INTEGER`s.** A serial number or an RSA modulus
//!   that arrives with a redundant leading zero is ambiguous in the same way; a
//!   negative one is not a value X.509 uses.
//! - **`BOOLEAN` values other than `0x00` and `0xFF`.** BER allows `0x01` for
//!   true and DER does not, which makes `basicConstraints` a place to disagree
//!   about whether a certificate is a CA.
//! - **`BIT STRING`s with unused bits**, for the uses here: a signature or a key
//!   has no partial trailing bits, and the count is a byte an attacker controls.
//!
//! What it does *not* do: no `SET OF` ordering check, no UTF-8 validation of
//! strings, no schema. The X.509 layer owns meaning; this layer owns the encoding.

/// An ASN.1 tag class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// The universal class: `SEQUENCE`, `INTEGER`, and the rest of X.680's built-ins.
    Universal,
    /// An application-defined type; nothing in a certificate uses one.
    Application,
    /// A context-specific type: how X.509 numbers the fields of a `SEQUENCE` and
    /// how extensions are tagged.
    Context,
    /// A private type; nothing in a certificate uses one.
    Private,
}

/// A DER tag: a class, whether the value is constructed, and the tag number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tag {
    class: Class,
    constructed: bool,
    number: u8,
}

impl Tag {
    /// `SEQUENCE`, the workhorse of X.509: always constructed.
    pub const SEQUENCE: Self = Self::built_in(16, true);
    /// `SET`, always constructed.
    pub const SET: Self = Self::built_in(17, true);
    /// `INTEGER`, always primitive.
    pub const INTEGER: Self = Self::built_in(2, false);
    /// `BIT STRING`, always primitive: the inner type is the caller's business.
    pub const BIT_STRING: Self = Self::built_in(3, false);
    /// `OCTET STRING`, always primitive.
    pub const OCTET_STRING: Self = Self::built_in(4, false);
    /// `NULL`, always primitive and always empty.
    pub const NULL: Self = Self::built_in(5, false);
    /// `OBJECT IDENTIFIER`, always primitive.
    pub const OID: Self = Self::built_in(6, false);
    /// `BOOLEAN`, always primitive.
    pub const BOOLEAN: Self = Self::built_in(1, false);
    /// `UTF8String`, always primitive.
    pub const UTF8_STRING: Self = Self::built_in(12, false);
    /// `PrintableString`, always primitive.
    pub const PRINTABLE_STRING: Self = Self::built_in(19, false);
    /// `IA5String`, always primitive.
    pub const IA5_STRING: Self = Self::built_in(22, false);
    /// `UTCTime`, always primitive.
    pub const UTC_TIME: Self = Self::built_in(23, false);
    /// `GeneralizedTime`, always primitive.
    pub const GENERALIZED_TIME: Self = Self::built_in(24, false);

    const fn built_in(number: u8, constructed: bool) -> Self {
        Self {
            class: Class::Universal,
            constructed,
            number,
        }
    }

    /// A context-specific constructed tag: `[n]` around a nested sequence.
    pub const fn context(number: u8) -> Self {
        Self {
            class: Class::Context,
            constructed: true,
            number,
        }
    }

    /// A context-specific primitive tag: `[n]` around a bare value, which is how
    /// `GeneralName` is encoded inside `subjectAltName`.
    pub const fn context_primitive(number: u8) -> Self {
        Self {
            class: Class::Context,
            constructed: false,
            number,
        }
    }

    /// A tag with explicit fields, for the odd case a test needs.
    pub const fn new(class: Class, constructed: bool, number: u8) -> Self {
        Self {
            class,
            constructed,
            number,
        }
    }

    /// The tag's class.
    pub fn class(self) -> Class {
        self.class
    }

    /// Whether the value is constructed (contains further elements).
    pub fn is_constructed(self) -> bool {
        self.constructed
    }

    /// The tag number within its class.
    pub fn number(self) -> u8 {
        self.number
    }

    /// The universal tag for `number`, constructed as X.680 defines it. Only for
    /// tests and diagnostics; the named constants above are what the parser uses.
    fn from_byte(byte: u8) -> Result<Self, DerError> {
        // High-tag-number form: legal in X.690, refused here (see the module
        // docs). One byte of decision is cheaper to audit than a multi-octet one,
        // and refusing it early means the length below is never reached.
        if byte & 0x1f == 0x1f {
            return Err(DerError::HighTagNumber);
        }
        let class = match byte >> 6 {
            0 => Class::Universal,
            1 => Class::Application,
            2 => Class::Context,
            _ => Class::Private,
        };
        Ok(Self {
            class,
            constructed: byte & 0x20 != 0,
            number: byte & 0x1f,
        })
    }
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let class = match self.class {
            Class::Universal => "universal",
            Class::Application => "application",
            Class::Context => "context",
            Class::Private => "private",
        };
        write!(
            f,
            "{class} {}{}",
            if self.constructed { "constructed " } else { "" },
            self.number
        )
    }
}

/// Why a byte string is not the DER this crate expects.
///
/// Every variant is fatal: this reader has no "best effort" mode, because a
/// best-effort parse of a certificate is a parse that can disagree with the
/// signature that covers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DerError {
    /// The input ended in the middle of an element.
    Truncated(&'static str),
    /// Indefinite length, which DER forbids.
    IndefiniteLength,
    /// A length that has a shorter encoding, or a reserved one.
    NonMinimalLength(&'static str),
    /// High-tag-number form, which this reader does not support.
    HighTagNumber,
    /// An element's tag is not the one the caller required.
    UnexpectedTag {
        /// What the caller asked for.
        expected: Tag,
        /// What was there.
        found: Tag,
    },
    /// An element was required and the input was empty.
    Missing(&'static str),
    /// Bytes left over after the fields that were read.
    TrailingData(usize),
    /// A primitive element was asked for a reader over its contents.
    NotConstructed,
    /// A constructed element was used where the contents themselves were wanted.
    Constructed(&'static str),
    /// The value's contents are not a legal instance of its type.
    Invalid(&'static str),
}

impl std::fmt::Display for DerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated(what) => write!(f, "DER is truncated inside {what}"),
            Self::IndefiniteLength => write!(f, "DER uses an indefinite length"),
            Self::NonMinimalLength(what) => write!(f, "DER length is not minimal ({what})"),
            Self::HighTagNumber => write!(f, "DER uses high-tag-number form"),
            Self::UnexpectedTag { expected, found } => {
                write!(f, "expected a {expected} tag, found {found}")
            }
            Self::Missing(what) => write!(f, "missing {what}"),
            Self::TrailingData(count) => write!(f, "{count} bytes left after the last field"),
            Self::NotConstructed => write!(f, "the element is primitive, so it has no children"),
            Self::Constructed(what) => write!(f, "{what} is constructed, not a value"),
            Self::Invalid(why) => write!(f, "invalid DER value: {why}"),
        }
    }
}

impl std::error::Error for DerError {}

/// One DER element: its tag and its contents.
///
/// Kept as a borrowed slice of the input, so `whole()` — the complete encoding —
/// is exactly the bytes the signature covers, with no re-encoding step that could
/// differ from the original.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Element<'a> {
    tag: Tag,
    /// The complete encoding, borrowed from the input.
    whole: &'a [u8],
    /// How many of those bytes are the tag plus the length.
    header: usize,
}

impl<'a> Element<'a> {
    /// This element's tag.
    pub fn tag(&self) -> Tag {
        self.tag
    }

    /// The contents, without the header. No tag check: use the `as_*` methods.
    pub fn contents(&self) -> &'a [u8] {
        &self.whole[self.header..]
    }

    /// The complete encoding — header, length and contents — which is what a
    /// signature over this element covers.
    ///
    /// It is a slice of the input rather than a re-encoding, so "the bytes that
    /// were signed" and "the bytes that were read" cannot differ.
    pub fn whole(&self) -> &'a [u8] {
        self.whole
    }

    /// A reader over the contents of a constructed element.
    pub fn reader(&self) -> Result<Der<'a>, DerError> {
        if !self.tag.constructed {
            return Err(DerError::NotConstructed);
        }
        Ok(Der::new(self.contents()))
    }

    /// The raw contents of a primitive element, for types this module does not
    /// decode: strings, times, and whatever the caller knows the shape of.
    pub fn as_primitive(&self) -> Result<&'a [u8], DerError> {
        if self.tag.constructed {
            return Err(DerError::Constructed("the value"));
        }
        Ok(self.contents())
    }

    /// An `INTEGER`'s magnitude, as big-endian bytes with any leading zero
    /// removed. Only non-negative values, only minimal encodings.
    pub fn as_integer_bytes(&self) -> Result<&'a [u8], DerError> {
        self.expect(Tag::INTEGER)?;
        let bytes = self.contents();
        match bytes {
            [] => Err(DerError::Invalid("an INTEGER needs at least one byte")),
            [0x00] => Ok(&bytes[1..]), // zero, and the only legal encoding of it
            [first, ..] if first & 0x80 != 0 => Err(DerError::Invalid("the INTEGER is negative")),
            [0x00, second, ..] if second & 0x80 == 0 => Err(DerError::NonMinimalLength(
                "the INTEGER has a redundant leading zero",
            )),
            [0x00, rest @ ..] => Ok(rest),
            _ => Ok(bytes),
        }
    }

    /// An `INTEGER` that has to fit in a `u64`: a version number, a key usage bit
    /// position, a path length.
    pub fn as_u64(&self) -> Result<u64, DerError> {
        let bytes = self.as_integer_bytes()?;
        if bytes.len() > 8 {
            return Err(DerError::Invalid("the INTEGER does not fit in 64 bits"));
        }
        Ok(bytes.iter().fold(0u64, |acc, &byte| acc << 8 | byte as u64))
    }

    /// An `OBJECT IDENTIFIER`, with its encoding checked.
    pub fn as_oid(&self) -> Result<Oid<'a>, DerError> {
        self.expect(Tag::OID)?;
        Oid::check(self.contents())?;
        Ok(Oid(self.contents()))
    }

    /// A `BIT STRING`'s contents, requiring zero unused bits.
    ///
    /// Both users of this — a public key and a signature — are whole octets; a
    /// non-zero count is a byte the peer controls and this crate has no use for,
    /// so it is refused rather than masked off.
    pub fn as_bit_string(&self) -> Result<&'a [u8], DerError> {
        self.expect(Tag::BIT_STRING)?;
        match self.contents() {
            [] => Err(DerError::Invalid(
                "a BIT STRING needs its unused-bits count",
            )),
            [0, rest @ ..] => Ok(rest),
            [unused, ..] if *unused <= 7 => Err(DerError::Invalid(
                "the BIT STRING has unused bits, which no signature or key here has",
            )),
            _ => Err(DerError::Invalid(
                "the BIT STRING's unused-bit count is above 7",
            )),
        }
    }

    /// A `BOOLEAN`, requiring one of the two DER encodings.
    pub fn as_boolean(&self) -> Result<bool, DerError> {
        self.expect(Tag::BOOLEAN)?;
        match self.contents() {
            [0x00] => Ok(false),
            [0xff] => Ok(true),
            [_] => Err(DerError::Invalid(
                "a DER BOOLEAN is 0x00 or 0xFF, not any non-zero byte",
            )),
            _ => Err(DerError::Invalid("a BOOLEAN is one byte")),
        }
    }

    /// An `INTEGER`-free check that the element is the tag the caller wanted.
    pub fn expect(&self, tag: Tag) -> Result<(), DerError> {
        if self.tag == tag {
            Ok(())
        } else {
            Err(DerError::UnexpectedTag {
                expected: tag,
                found: self.tag,
            })
        }
    }
}

/// A reader over a sequence of DER elements.
///
/// Borrowed from the input, so reading never copies and `Element::whole` always
/// hands back the original bytes.
#[derive(Debug, Clone, Copy)]
pub struct Der<'a> {
    rest: &'a [u8],
}

impl<'a> Der<'a> {
    /// A reader over a byte string that is expected to be a sequence of elements.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { rest: bytes }
    }

    /// Whether every element has been consumed.
    pub fn is_empty(&self) -> bool {
        self.rest.is_empty()
    }

    /// What has not been read yet.
    pub fn rest(&self) -> &'a [u8] {
        self.rest
    }

    /// The next element's tag, without consuming it. `None` at the end.
    pub fn peek_tag(&self) -> Result<Option<Tag>, DerError> {
        match self.rest.first() {
            None => Ok(None),
            Some(&byte) => Ok(Some(Tag::from_byte(byte)?)),
        }
    }

    /// Consumes and returns the next element.
    pub fn next_element(&mut self) -> Result<Element<'a>, DerError> {
        let (element, consumed) = parse_element(self.rest)?;
        self.rest = &self.rest[consumed..];
        Ok(element)
    }

    /// Consumes the next element, which must have `tag`.
    ///
    /// A tag mismatch leaves the reader untouched: the element is parsed and
    /// checked before anything is consumed. The error is fatal to the caller
    /// either way — this reader never guesses — but a reader that is still
    /// pointing at the same element makes the failure report the right thing and
    /// lets a caller that is choosing between shapes try the next one.
    pub fn take(&mut self, tag: Tag) -> Result<Element<'a>, DerError> {
        let (element, consumed) = parse_element(self.rest)?;
        element.expect(tag)?;
        self.rest = &self.rest[consumed..];
        Ok(element)
    }

    /// Consumes the next element if it has `tag`, and leaves it alone otherwise.
    pub fn take_if(&mut self, tag: Tag) -> Result<Option<Element<'a>>, DerError> {
        match self.peek_tag()? {
            Some(found) if found == tag => Ok(Some(self.take(tag)?)),
            _ => Ok(None),
        }
    }

    /// Requires that nothing is left to read.
    pub fn expect_end(&self) -> Result<(), DerError> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(DerError::TrailingData(self.rest.len()))
        }
    }
}

/// Parses one element, returning it and how many bytes it occupied.
fn parse_element(bytes: &[u8]) -> Result<(Element<'_>, usize), DerError> {
    let (&first, after_tag) = bytes.split_first().ok_or(DerError::Missing("an element"))?;
    let tag = Tag::from_byte(first)?;

    let (&first_length, after_first_length) = after_tag
        .split_first()
        .ok_or(DerError::Truncated("the length"))?;
    let (length, header_len) = if first_length < 0x80 {
        (first_length as usize, 2)
    } else {
        match first_length {
            0x80 => return Err(DerError::IndefiniteLength),
            0xff => {
                return Err(DerError::NonMinimalLength(
                    "0xFF is a reserved length octet",
                ))
            }
            _ => {
                let count = (first_length & 0x7f) as usize;
                // Every size that appears in a certificate fits in four bytes; a
                // longer length field is refused rather than computed, so a length
                // can never overflow the arithmetic below.
                if count > 4 {
                    return Err(DerError::NonMinimalLength(
                        "a length field wider than four bytes",
                    ));
                }
                if after_first_length.len() < count {
                    return Err(DerError::Truncated("the length"));
                }
                if after_first_length[0] == 0 {
                    return Err(DerError::NonMinimalLength("the long form could be shorter"));
                }
                let mut length = 0usize;
                for &byte in &after_first_length[..count] {
                    length = (length << 8) | byte as usize;
                }
                (length, 2 + count)
            }
        }
    };

    let contents_at = header_len;
    let end = contents_at
        .checked_add(length)
        .ok_or(DerError::Truncated("the length"))?;
    if bytes.len() < end {
        return Err(DerError::Truncated("the contents"));
    }
    Ok((
        Element {
            tag,
            whole: &bytes[..end],
            header: header_len,
        },
        end,
    ))
}

/// An `OBJECT IDENTIFIER`, borrowed from the input.
///
/// Compared by encoding, which is what DER is for: two OIDs are the same when
/// their bytes are, and no parsing to arcs is needed to decide that.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Oid<'a>(&'a [u8]);

impl<'a> Oid<'a> {
    /// The OID's contents, without its tag and length.
    pub fn as_bytes(&self) -> &'a [u8] {
        self.0
    }

    /// Checks an OID's encoding: at least the two-arc first byte, then
    /// sub-identifiers that are minimal (no `0x80` continuation byte, which
    /// encodes a leading zero) and that actually terminate.
    fn check(bytes: &'a [u8]) -> Result<(), DerError> {
        let (&first, rest) = bytes
            .split_first()
            .ok_or(DerError::Invalid("a zero-length OID"))?;
        if first / 40 > 2 {
            return Err(DerError::Invalid(
                "the first sub-identifier is not 0, 1 or 2 times 40",
            ));
        }
        let mut index = 0;
        while index < rest.len() {
            if rest[index] == 0x80 {
                return Err(DerError::Invalid(
                    "a sub-identifier has a redundant leading 0x80",
                ));
            }
            // A sub-identifier is a base-128 value whose last byte has its high
            // bit clear.
            let mut seen_end = rest[index] & 0x80 == 0;
            while !seen_end {
                index += 1;
                if index >= rest.len() {
                    return Err(DerError::Invalid("a sub-identifier is unterminated"));
                }
                seen_end = rest[index] & 0x80 == 0;
            }
            index += 1;
        }
        Ok(())
    }
}

/// Encodes a dotted OID (`"2.5.29.19"`) into DER contents, for building fixtures
/// and for checking the constants below against something readable.
///
/// This is for tests and constants, not for parsing: it panics on anything it
/// cannot encode, which is the right behaviour for a literal and the wrong one for
/// input.
pub fn oid_from_dotted(dotted: &str) -> Vec<u8> {
    let arcs: Vec<u64> = dotted
        .split('.')
        .map(|arc| arc.parse().expect("numeric OID arcs"))
        .collect();
    assert!(arcs.len() >= 2, "an OID needs at least two arcs");
    assert!(arcs[0] <= 2, "the first arc is 0, 1 or 2");
    assert!(
        arcs[0] < 2 || arcs[1] < 40,
        "under arc 2 the second arc is below 40"
    );
    let mut out = Vec::new();
    encode_base128(arcs[0] * 40 + arcs[1], &mut out);
    for &arc in &arcs[2..] {
        encode_base128(arc, &mut out);
    }
    out
}

fn encode_base128(mut value: u64, out: &mut Vec<u8>) {
    let mut stack = [0u8; 10];
    let mut len = 0;
    loop {
        stack[len] = (value & 0x7f) as u8;
        len += 1;
        value >>= 7;
        if value == 0 {
            break;
        }
    }
    for index in (0..len).rev() {
        out.push(stack[index] | if index == 0 { 0 } else { 0x80 });
    }
}

/// Object identifiers this crate has to name, as their DER contents (no tag, no
/// length). Each one is checked against its dotted form by a test, so a wrong byte
/// here fails the suite rather than a handshake.
pub mod oids {
    /// `rsaEncryption`, the `SubjectPublicKeyInfo` algorithm of an RSA key.
    pub const RSA_ENCRYPTION: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];
    /// `id-RSASSA-PSS`, the TLS 1.3 signature algorithm for RSA keys.
    pub const RSASSA_PSS: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a];
    /// `sha256WithRSAEncryption`, what most CAs sign certificates with.
    pub const SHA256_WITH_RSA: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
    /// `mgf1`, the only mask generation function a PSS parameter may name here.
    pub const MGF1: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x08];
    /// `id-sha256`, the only hash these signatures use so far.
    pub const SHA256: &[u8] = &[0x60, 0x86, 0x48, 0x01, 0x65, 0x03, 0x04, 0x02, 0x01];
    /// `id-ecPublicKey`, the algorithm of a P-256 key.
    pub const EC_PUBLIC_KEY: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
    /// `prime256v1` (alias `secp256r1`), the curve TLS 1.3 requires support for.
    pub const PRIME256V1: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
    /// `ecdsa-with-SHA256`.
    pub const ECDSA_WITH_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
    /// `basicConstraints`, the extension that says whether a certificate is a CA.
    pub const BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1d, 0x13];
    /// `keyUsage`.
    pub const KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x0f];
    /// `extKeyUsage`.
    pub const EXT_KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x25];
    /// `subjectAltName`, where a downloader finds the names it may connect to.
    pub const SUBJECT_ALT_NAME: &[u8] = &[0x55, 0x1d, 0x11];
    /// `serverAuth`, the required purpose for a TLS server certificate.
    pub const SERVER_AUTH: &[u8] = &[0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01];
    /// `commonName`, the only `RDN` this crate reads.
    pub const COMMON_NAME: &[u8] = &[0x55, 0x04, 0x03];
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The DER of a `SEQUENCE` wrapping `contents`, with a minimal length.
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

    fn integer(value: u8) -> Vec<u8> {
        vec![0x02, 0x01, value]
    }

    /// One table row: a complete encoding and what reading it should produce.
    type Case<'a> = (&'a [u8], Result<&'a [u8], DerError>);

    #[test]
    fn a_sequence_of_integers_reads_back() {
        let body = [integer(1), integer(0x7f), integer(0xff)].concat();
        let der = sequence(&body);
        let outer = Der::new(&der).next_element().expect("the sequence");
        assert_eq!(outer.tag(), Tag::SEQUENCE);
        let mut reader = outer.reader().expect("constructed");
        assert_eq!(reader.take(Tag::INTEGER).unwrap().as_u64().unwrap(), 1);
        assert_eq!(reader.take(Tag::INTEGER).unwrap().as_u64().unwrap(), 127);
        // 0xFF is negative, so it is refused rather than read as 255.
        assert_eq!(
            reader.take(Tag::INTEGER).unwrap().as_u64(),
            Err(DerError::Invalid("the INTEGER is negative"))
        );
        reader.expect_end().unwrap();
    }

    #[test]
    fn integers_are_read_as_magnitudes_and_non_minimal_ones_are_refused() {
        let cases: &[Case<'_>] = &[
            (&[0x00], Ok(&[][..])),
            (&[0x01], Ok(&[0x01][..])),
            (&[0x7f], Ok(&[0x7f][..])),
            // 128 is positive in one byte with the high bit set: it needs the pad.
            (&[0x00, 0x80], Ok(&[0x80][..])),
            (&[0x01, 0x00], Ok(&[0x01, 0x00][..])),
            // A redundant leading zero, and a negative value.
            (
                &[0x00, 0x01],
                Err(DerError::NonMinimalLength(
                    "the INTEGER has a redundant leading zero",
                )),
            ),
            (&[0xff], Err(DerError::Invalid("the INTEGER is negative"))),
            (
                &[0xff, 0x7f],
                Err(DerError::Invalid("the INTEGER is negative")),
            ),
            (
                &[],
                Err(DerError::Invalid("an INTEGER needs at least one byte")),
            ),
        ];
        for (contents, expected) in cases {
            let mut der = vec![0x02u8];
            der.push(contents.len() as u8);
            der.extend_from_slice(contents);
            let element = Der::new(&der).next_element().expect("well framed");
            assert_eq!(
                element.as_integer_bytes(),
                *expected,
                "contents {contents:02x?}"
            );
        }
    }

    #[test]
    fn a_long_form_length_is_read_and_a_redundant_one_is_refused() {
        let mut payload = vec![0x04, 0x81, 0x80];
        payload.extend(std::iter::repeat_n(0xaa, 0x80));
        let element = Der::new(&payload).next_element().expect("0x81 0x80");
        assert_eq!(element.contents().len(), 0x80);
        assert_eq!(element.contents()[0], 0xaa);
        assert_eq!(
            element.whole().len(),
            0x83,
            "one tag, two length octets, 128"
        );

        // 0x81 0x00 is the same length written with a redundant octet.
        assert_eq!(
            Der::new(&[0x04, 0x81, 0x00]).next_element(),
            Err(DerError::NonMinimalLength("the long form could be shorter"))
        );
        // Indefinite, and the reserved 0xFF.
        assert_eq!(
            Der::new(&[0x30, 0x80, 0x00, 0x00]).next_element(),
            Err(DerError::IndefiniteLength)
        );
        assert!(matches!(
            Der::new(&[0x30, 0xff]).next_element(),
            Err(DerError::NonMinimalLength(_))
        ));
        // A length field wider than four bytes is refused rather than computed.
        assert!(matches!(
            Der::new(&[0x30, 0x85, 1, 2, 3, 4, 5]).next_element(),
            Err(DerError::NonMinimalLength(_))
        ));
    }

    #[test]
    fn truncation_is_caught_at_every_level() {
        // A header with no length byte.
        assert_eq!(
            Der::new(&[0x30]).next_element(),
            Err(DerError::Truncated("the length"))
        );
        // A length field that promises more octets than are present.
        assert_eq!(
            Der::new(&[0x30, 0x82, 0x01]).next_element(),
            Err(DerError::Truncated("the length"))
        );
        // Contents that run past the end.
        assert_eq!(
            Der::new(&[0x04, 0x05, 0x01, 0x02]).next_element(),
            Err(DerError::Truncated("the contents"))
        );
        // Nothing at all.
        assert_eq!(
            Der::new(&[]).next_element(),
            Err(DerError::Missing("an element"))
        );
    }

    #[test]
    fn trailing_bytes_are_reported_rather_than_ignored() {
        let der = sequence(&integer(1));
        let mut reader = Der::new(&der);
        reader.take(Tag::SEQUENCE).unwrap();
        assert!(reader.is_empty());
        reader.expect_end().unwrap();

        let mut padded = der.clone();
        padded.push(0x00);
        let mut reader = Der::new(&padded);
        reader.take(Tag::SEQUENCE).unwrap();
        assert_eq!(reader.expect_end(), Err(DerError::TrailingData(1)));
    }

    #[test]
    fn high_tag_numbers_are_refused() {
        // 0x9f would introduce a multi-octet tag number.
        assert_eq!(
            Der::new(&[0x9f, 0x20, 0x00]).next_element(),
            Err(DerError::HighTagNumber)
        );
        // A high-tag-number form is also how a field can be hidden, so the refusal
        // happens before any length is read.
        assert_eq!(
            Der::new(&[0x3f]).next_element(),
            Err(DerError::HighTagNumber)
        );
    }

    #[test]
    fn a_tag_is_the_one_that_was_asked_for() {
        let der = integer(1);
        let mut reader = Der::new(&der);
        assert_eq!(
            reader.take(Tag::OCTET_STRING),
            Err(DerError::UnexpectedTag {
                expected: Tag::OCTET_STRING,
                found: Tag::INTEGER
            })
        );
        // The failed take consumed nothing, so the caller can still read it.
        assert_eq!(reader.take(Tag::INTEGER).unwrap().as_u64().unwrap(), 1);
    }

    #[test]
    fn booleans_must_use_a_der_encoding() {
        let cases: &[(&[u8], Result<bool, DerError>)] = &[
            (&[0x01, 0x01, 0x00], Ok(false)),
            (&[0x01, 0x01, 0xff], Ok(true)),
            (
                &[0x01, 0x01, 0x01],
                Err(DerError::Invalid(
                    "a DER BOOLEAN is 0x00 or 0xFF, not any non-zero byte",
                )),
            ),
            (
                &[0x01, 0x02, 0x00, 0x00],
                Err(DerError::Invalid("a BOOLEAN is one byte")),
            ),
            (
                &[0x01, 0x00],
                Err(DerError::Invalid("a BOOLEAN is one byte")),
            ),
        ];
        for (der, expected) in cases {
            let element = Der::new(der).next_element().expect("well framed");
            assert_eq!(element.as_boolean(), *expected, "{der:02x?}");
        }
    }

    #[test]
    fn bit_strings_require_zero_unused_bits() {
        let cases: &[Case<'_>] = &[
            (&[0x03, 0x03, 0x00, 0xaa, 0xbb], Ok(&[0xaa, 0xbb][..])),
            (
                &[0x03, 0x03, 0x03, 0xaa, 0xbb],
                Err(DerError::Invalid(
                    "the BIT STRING has unused bits, which no signature or key here has",
                )),
            ),
            (
                &[0x03, 0x00],
                Err(DerError::Invalid(
                    "a BIT STRING needs its unused-bits count",
                )),
            ),
        ];
        for (der, expected) in cases {
            let element = Der::new(der).next_element().expect("well framed");
            assert_eq!(element.as_bit_string(), *expected, "{der:02x?}");
        }
    }

    #[test]
    fn a_primitive_element_has_no_children_and_a_constructed_one_is_not_a_value() {
        let der = integer(1);
        let element = Der::new(&der).next_element().unwrap();
        assert_eq!(element.reader().unwrap_err(), DerError::NotConstructed);
        let der = sequence(&integer(1));
        let element = Der::new(&der).next_element().unwrap();
        assert_eq!(
            element.as_primitive().unwrap_err(),
            DerError::Constructed("the value")
        );
    }

    #[test]
    fn an_element_remembers_its_own_bytes() {
        let der = sequence(&integer(1));
        let element = Der::new(&der).next_element().unwrap();
        assert_eq!(element.whole(), &der[..], "header, length and contents");
        // A nested element's `whole` is a slice of the same input, so a signature
        // over it covers the original bytes rather than a re-encoding.
        let mut reader = element.reader().unwrap();
        let inner = reader.next_element().unwrap();
        assert_eq!(inner.whole(), &[0x02, 0x01, 0x01]);
    }

    #[test]
    fn object_identifiers_are_checked_for_minimal_encodings() {
        let good = oid_from_dotted("1.2.840.113549.1.1.1");
        let mut der = vec![0x06, good.len() as u8];
        der.extend_from_slice(&good);
        let element = Der::new(&der).next_element().unwrap();
        assert_eq!(element.as_oid().unwrap().as_bytes(), &good[..]);

        // A sub-identifier with a redundant leading 0x80: the same arc, two
        // encodings, which is exactly what DER forbids.
        let mut bad = vec![0x06, 0x04, 0x2a, 0x80, 0x01, 0x01];
        let element = Der::new(&bad).next_element().unwrap();
        assert!(matches!(element.as_oid(), Err(DerError::Invalid(_))));
        // Unterminated sub-identifier: the last byte has its continuation bit set.
        bad = vec![0x06, 0x02, 0x2a, 0x81];
        let element = Der::new(&bad).next_element().unwrap();
        assert!(matches!(element.as_oid(), Err(DerError::Invalid(_))));
        // Empty.
        let element = Der::new(&[0x06, 0x00]).next_element().unwrap();
        assert!(matches!(element.as_oid(), Err(DerError::Invalid(_))));
    }

    #[test]
    fn the_named_oids_are_the_ones_they_claim_to_be() {
        // The encoding is derived from the dotted form here, so a wrong byte in
        // the constants above cannot survive the suite.
        for (dotted, encoded) in [
            ("1.2.840.113549.1.1.1", oids::RSA_ENCRYPTION),
            ("1.2.840.113549.1.1.10", oids::RSASSA_PSS),
            ("1.2.840.113549.1.1.11", oids::SHA256_WITH_RSA),
            ("1.2.840.113549.1.1.8", oids::MGF1),
            ("2.16.840.1.101.3.4.2.1", oids::SHA256),
            ("1.2.840.10045.2.1", oids::EC_PUBLIC_KEY),
            ("1.2.840.10045.3.1.7", oids::PRIME256V1),
            ("1.2.840.10045.4.3.2", oids::ECDSA_WITH_SHA256),
            ("2.5.29.19", oids::BASIC_CONSTRAINTS),
            ("2.5.29.15", oids::KEY_USAGE),
            ("2.5.29.37", oids::EXT_KEY_USAGE),
            ("2.5.29.17", oids::SUBJECT_ALT_NAME),
            ("1.3.6.1.5.5.7.3.1", oids::SERVER_AUTH),
            ("2.5.4.3", oids::COMMON_NAME),
        ] {
            assert_eq!(
                oid_from_dotted(dotted),
                encoded.to_vec(),
                "the constant for {dotted} is wrong"
            );
        }
        // And the encoding helper round-trips the arcs, so the check above is
        // checking something: 2.5.29.19 is 55 1d 13.
        assert_eq!(oid_from_dotted("2.5.29.19"), vec![0x55, 0x1d, 0x13]);
    }

    #[test]
    fn the_explicit_tag_constructors_are_the_tags_they_say() {
        assert_eq!(Tag::SEQUENCE.class(), Class::Universal);
        assert_eq!(Tag::SEQUENCE.number(), 16);
        assert!(Tag::SEQUENCE.is_constructed());
        assert!(!Tag::INTEGER.is_constructed());
        assert_eq!(Tag::context(0).class(), Class::Context);
        assert!(Tag::context(0).is_constructed());
        assert!(!Tag::context_primitive(2).is_constructed());
        assert_eq!(Tag::context(2).number(), 2);
        // A context tag is 0xa0 for [0] constructed, 0x82 for [2] primitive.
        let der = vec![0xa0, 0x00];
        assert_eq!(
            Der::new(&der).next_element().unwrap().tag(),
            Tag::context(0)
        );
        let der = vec![0x82, 0x00];
        assert_eq!(
            Der::new(&der).next_element().unwrap().tag(),
            Tag::context_primitive(2)
        );
    }

    #[test]
    fn take_if_leaves_an_element_alone_when_the_tag_differs() {
        let body = [integer(1), integer(2)].concat();
        let der = sequence(&body);
        let outer = Der::new(&der).next_element().unwrap();
        let mut reader = outer.reader().unwrap();
        assert!(reader.take_if(Tag::BOOLEAN).unwrap().is_none());
        assert!(reader.peek_tag().unwrap() == Some(Tag::INTEGER));
        assert_eq!(
            reader
                .take_if(Tag::INTEGER)
                .unwrap()
                .unwrap()
                .as_u64()
                .unwrap(),
            1
        );
        assert_eq!(reader.next_element().unwrap().as_u64().unwrap(), 2);
    }
}
