//! The leaf's own claims: which server it is for, and what it may be used for.
//!
//! Two checks that are about the leaf rather than about the path, and they are the
//! last ones between this crate and an `https://` URL: a chain that verifies
//! ([`crate::chain`]) says a certificate is authentic, and these say it is the
//! *right* certificate.
//!
//! # Host names
//!
//! - **`subjectAltName` is the only place a name is looked for.** The `commonName`
//!   fallback was deprecated by RFC 6125 §6.4.4 and is a live hazard: a name that
//!   appears only in a free-text field is not a constraint on the certificate. A
//!   certificate with a `commonName` and no `dNSName` therefore matches **nothing**,
//!   which the fixture `LEAF_EC` exists to prove.
//! - **Only `dNSName` entries are names.** `rfc822Name`, `uniformResourceIdentifier`
//!   and `iPAddress` are carried by [`crate::x509`] but never compared against a
//!   server name: `admin@models.example` is not a host and must not match one.
//! - **Comparison is ASCII case-insensitive**, and one trailing dot is ignored on
//!   both sides, because `example.com.` and `example.com` are the same absolute
//!   name. Nothing else is normalised: no whitespace is trimmed, no unicode case
//!   folding is done, and a name with a NUL or any other byte outside
//!   `a-z 0-9 - . _` simply never matches a usable server name.
//! - **A wildcard is one whole leftmost label, or nothing.** The pattern must be
//!   exactly `*.` followed by at least two labels (`*.com` is refused, which is the
//!   only public-suffix protection available without shipping a public suffix
//!   list), it must match at least one label of the server name, and no `*` may
//!   appear anywhere else — so neither `a.*.example` nor `foo*bar.example` is a
//!   pattern at all, and a server name containing `*` is never matched.
//! - **An IP literal is refused rather than matched.** A `dNSName` cannot be an
//!   address, so matching `192.0.2.1` against names would be matching it against
//!   the wrong kind of thing. A client that wants to connect to an address must
//!   check `iPAddress` entries, which this crate does not do, because the downloader
//!   only ever fetches by name.
//!
//! # Purpose
//!
//! `extendedKeyUsage`, when present, must contain `serverAuth`: a certificate the CA
//! issued for `clientAuth` only is not a server certificate. When the extension is
//! **absent**, any purpose is allowed (RFC 5280 §4.2.1.12).
//!
//! `keyUsage`, when present, must contain `digitalSignature`, because the leaf's own
//! key is what signs `CertificateVerify`. A certificate whose usage is
//! `keyEncipherment` alone is for RSA key transport, which TLS 1.3 removed
//! entirely, so it can never complete a handshake here.
//!
//! Every check is fatal on failure, as everywhere else in this crate.

use std::fmt;

use crate::der::oids;
use crate::x509::{Certificate, GeneralName, KeyUsage};

/// Why a leaf certificate was refused for the connection being made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeafError {
    /// No server name was offered, so no name can be checked. A verifier without a
    /// name to check is a verifier that cannot tell one server from another.
    NoServerName,
    /// The server name is not a usable host name: empty, too long, or containing a
    /// byte no host name may contain.
    MalformedServerName {
        /// The name as it was given.
        server_name: Vec<u8>,
    },
    /// The server name is an IP literal, and an address is not something a
    /// `dNSName` can match.
    IpLiteral {
        /// The literal as it was given.
        server_name: Vec<u8>,
    },
    /// The certificate has no `dNSName` at all, so there is nothing to match.
    NoDnsNames,
    /// Every `dNSName` in the certificate was compared and none matched.
    NoNameMatch {
        /// The name that was being looked for.
        server_name: Vec<u8>,
    },
    /// `extendedKeyUsage` is present and does not include `serverAuth`.
    MissingServerAuth {
        /// The purposes the certificate does name, each as the OID's DER contents.
        purposes: Vec<Vec<u8>>,
    },
    /// `keyUsage` is present and does not include `digitalSignature`.
    NoDigitalSignature {
        /// The `keyUsage` bits as read, for a diagnostic that says what it does
        /// name rather than only what it does not.
        key_usage: u16,
    },
}

/// Renders bytes for a diagnostic, lossily, because these are *our* bytes: a server
/// name comes from the URL we were asked to fetch, not from the peer.
fn shown(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

impl fmt::Display for LeafError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoServerName => write!(
                f,
                "no server name was offered, so the certificate's names cannot be checked"
            ),
            Self::MalformedServerName { server_name } => write!(
                f,
                "{:?} is not a usable host name, so it cannot be checked against a certificate",
                shown(server_name)
            ),
            Self::IpLiteral { server_name } => write!(
                f,
                "{:?} is an IP address, and a dNSName can never match one",
                shown(server_name)
            ),
            Self::NoDnsNames => write!(
                f,
                "the certificate has no dNSName, so it names no server at all (a commonName is \
                 not a name for this purpose)"
            ),
            Self::NoNameMatch { server_name } => write!(
                f,
                "no dNSName in the certificate matches {:?}",
                shown(server_name)
            ),
            Self::MissingServerAuth { purposes } => write!(
                f,
                "the certificate's extendedKeyUsage does not include serverAuth, only {}",
                purposes
                    .iter()
                    .map(|oid| oid
                        .iter()
                        .map(|byte| format!("{byte:02x}"))
                        .collect::<String>())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::NoDigitalSignature { key_usage } => write!(
                f,
                "the certificate's keyUsage ({key_usage:#05x}) does not include digitalSignature, \
                 which signing a CertificateVerify requires"
            ),
        }
    }
}

impl std::error::Error for LeafError {}

/// A name lower-cased with one trailing dot removed, which is the form both sides
/// of a comparison are put in.
fn normalize(name: &[u8]) -> Vec<u8> {
    let name = name.strip_suffix(b".").unwrap_or(name);
    name.iter().map(u8::to_ascii_lowercase).collect()
}

/// Whether a `dNSName` pattern matches a server name.
///
/// This is the whole name-matching rule as a pure function of two byte strings,
/// which is why it is public: the wildcard policy is where a name matcher is
/// usually wrong, and here it can be tested exhaustively without a certificate.
/// The rules are the ones in the module docs; anything not matching them is `false`
/// rather than an error, because a pattern that does not match is simply not the
/// name being looked for.
///
/// Both sides are normalised first (ASCII lower case, one trailing dot removed),
/// which allocates. That is deliberate: this runs once per connection, and the
/// alternative is an in-place comparison whose off-by-one would be a security bug.
pub fn dns_name_matches(pattern: &[u8], server_name: &[u8]) -> bool {
    let pattern = normalize(pattern);
    let name = normalize(server_name);
    if pattern.is_empty() || name.is_empty() {
        return false;
    }
    // A server name is never a pattern.
    if name.contains(&b'*') {
        return false;
    }
    let suffix = match pattern.strip_prefix(b"*.") {
        // No wildcard: the two names must simply be equal.
        None => return pattern == name,
        Some(suffix) => suffix,
    };
    // One wildcard, in one place: anywhere else in the pattern and it is not a
    // pattern this crate recognises.
    if suffix.contains(&b'*') {
        return false;
    }
    // `*.com` would match a registrable name under a public suffix, so the pattern
    // must leave at least two labels. Without a public suffix list this is the
    // strongest rule available, and it is stated as a heuristic rather than as a
    // guarantee.
    if suffix.split(|byte| *byte == b'.').count() < 2 {
        return false;
    }
    if suffix.len() >= name.len() || !name.ends_with(suffix) {
        return false;
    }
    // Exactly one whole label in front of the suffix: `a.cdn.example` matches, and
    // `b.a.cdn.example` does not.
    let prefix = &name[..name.len() - suffix.len()];
    matches!(prefix.strip_suffix(b"."), Some(label) if !label.is_empty() && !label.contains(&b'.'))
}

/// Whether a name is a shape this crate will compare with anything at all.
///
/// Refusing is deliberate: a server name with a NUL, a slash or a non-ASCII byte is
/// a bug in the caller — an IDN must be passed in its A-label form, which is also
/// what RFC 6066 requires SNI to carry — and a silent non-match would turn that bug
/// into "the certificate is wrong".
fn server_name_is_usable(name: &[u8]) -> bool {
    !name.is_empty()
        && name.len() <= 253
        && !name.starts_with(b".")
        && name.iter().all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_uppercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'.' | b'-' | b'_')
        })
}

/// Whether a name is address-shaped rather than a host name.
///
/// Full address syntax is not validated, and does not need to be: the only thing
/// this decides is whether a name is refused as "an address, not a name" or as "not
/// a host name", and both refuse. What it must not do is call something an address
/// that is plainly a host name with a mistake in it — `models.example:443` is a
/// host name with a port, not a literal, and saying so is more useful.
fn is_ip_literal(name: &[u8]) -> bool {
    // IPv4: four dot-separated runs of at most three digits.
    let parts: Vec<&[u8]> = name.split(|byte| *byte == b'.').collect();
    if parts.len() == 4
        && parts
            .iter()
            .all(|part| !part.is_empty() && part.len() <= 3 && part.iter().all(u8::is_ascii_digit))
    {
        return true;
    }
    // IPv6: two or more colons and nothing but hex digits, colons and dots — which is
    // what an address is made of and what a host name never is.
    name.iter().filter(|byte| **byte == b':').count() >= 2
        && name
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() || matches!(byte, b':' | b'.'))
}

/// Checks that the certificate names the server being connected to.
///
/// `server_name` is the SNI name the `ClientHello` carried, which
/// [`crate::client::CertificateVerifyInput`] already hands a verifier. `None` is
/// refused: a connection with no name is a connection whose certificate cannot be
/// tied to the server.
pub fn verify_server_name(
    certificate: &Certificate<'_>,
    server_name: Option<&[u8]>,
) -> Result<(), LeafError> {
    let Some(server_name) = server_name else {
        return Err(LeafError::NoServerName);
    };
    // The address check comes first, and the order is deliberate: a colon is not a
    // byte a host name may contain, so an IPv6 literal would otherwise be reported as
    // a malformed name — refusing it either way, but with the diagnosis that hides
    // the actual mistake. Both checks refuse; this order says *why*.
    if is_ip_literal(server_name) {
        return Err(LeafError::IpLiteral {
            server_name: server_name.to_vec(),
        });
    }
    if !server_name_is_usable(server_name) {
        return Err(LeafError::MalformedServerName {
            server_name: server_name.to_vec(),
        });
    }
    let mut compared = 0usize;
    for name in certificate.subject_alt_names() {
        // Only `dNSName`. Every other choice is skipped rather than rejected: a
        // `rfc822Name` is not a wrong name, it is not a name at all for this check.
        if let GeneralName::Dns(pattern) = name {
            compared += 1;
            if dns_name_matches(pattern, server_name) {
                return Ok(());
            }
        }
    }
    if compared == 0 {
        Err(LeafError::NoDnsNames)
    } else {
        Err(LeafError::NoNameMatch {
            server_name: server_name.to_vec(),
        })
    }
}

/// Checks that the certificate is allowed to sign a `CertificateVerify` as a
/// server: `extendedKeyUsage` including `serverAuth` and `keyUsage` including
/// `digitalSignature`, each when the extension is present.
pub fn verify_server_auth(certificate: &Certificate<'_>) -> Result<(), LeafError> {
    if let Some(usage) = certificate.key_usage() {
        if !usage.has(KeyUsage::DIGITAL_SIGNATURE) {
            return Err(LeafError::NoDigitalSignature { key_usage: usage.0 });
        }
    }
    let purposes = certificate.extended_key_usage();
    // An absent `extKeyUsage` allows any purpose; the parser refuses an empty one,
    // so an empty slice here means the extension was not there.
    if !purposes.is_empty()
        && !purposes
            .iter()
            .any(|oid| oid.as_bytes() == oids::SERVER_AUTH)
    {
        return Err(LeafError::MissingServerAuth {
            purposes: purposes.iter().map(|oid| oid.as_bytes().to_vec()).collect(),
        });
    }
    Ok(())
}

/// Both leaf checks, in one call: the certificate is for this server, and it may be
/// used as a server certificate.
///
/// Purpose is checked before the name because it is the cheaper test and because
/// "this is not a server certificate" is the more fundamental thing to be told.
pub fn verify_leaf(
    certificate: &Certificate<'_>,
    server_name: Option<&[u8]>,
) -> Result<(), LeafError> {
    verify_server_auth(certificate)?;
    verify_server_name(certificate, server_name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_exact_name_matches_and_nothing_else_does() {
        assert!(dns_name_matches(b"models.example", b"models.example"));
        assert!(!dns_name_matches(b"models.example", b"models.example.org"));
        assert!(!dns_name_matches(b"models.example", b"odels.example"));
        assert!(!dns_name_matches(b"models.example", b"evilmodels.example"));
        assert!(!dns_name_matches(b"models.example", b"models.example.evil"));
    }

    #[test]
    fn case_and_one_trailing_dot_are_the_only_normalisation() {
        assert!(dns_name_matches(b"Models.Example", b"models.example"));
        assert!(dns_name_matches(b"models.example", b"MODELS.EXAMPLE"));
        assert!(dns_name_matches(b"models.example.", b"models.example"));
        assert!(dns_name_matches(b"models.example", b"models.example."));
        assert!(dns_name_matches(b"*.cdn.example", b"A.CDN.Example."));
        // Two trailing dots is not a name, so it does not match a one-dot pattern.
        assert!(!dns_name_matches(b"models.example", b"models.example.."));
        // Whitespace is not trimmed, and a leading dot is not stripped.
        assert!(!dns_name_matches(b"models.example", b" models.example"));
        assert!(!dns_name_matches(b"models.example", b".models.example"));
    }

    #[test]
    fn a_wildcard_covers_exactly_one_whole_leftmost_label() {
        assert!(dns_name_matches(b"*.cdn.example", b"a.cdn.example"));
        assert!(dns_name_matches(b"*.cdn.example", b"a-b.cdn.example"));
        // Not a bare domain: `*.example` must not be satisfied by `example`.
        assert!(!dns_name_matches(b"*.cdn.example", b"cdn.example"));
        // Not two labels.
        assert!(!dns_name_matches(b"*.cdn.example", b"a.b.cdn.example"));
        // Not a partial label.
        assert!(!dns_name_matches(b"*.cdn.example", b"acdn.example"));
        assert!(!dns_name_matches(b"*.cdn.example", b"a.evil.example"));
        // Not the empty label.
        assert!(!dns_name_matches(b"*.cdn.example", b".cdn.example"));
    }

    #[test]
    fn a_wildcard_that_is_not_one_leading_label_is_not_a_pattern() {
        // A bare `*` is not a name.
        assert!(!dns_name_matches(b"*", b"models.example"));
        assert!(!dns_name_matches(b"*.", b"models.example"));
        // A `*` anywhere but the whole leftmost label.
        assert!(!dns_name_matches(b"a.*.example", b"a.b.example"));
        assert!(!dns_name_matches(b"models*.example", b"models1.example"));
        assert!(!dns_name_matches(b"*models.example", b"xmodels.example"));
        assert!(!dns_name_matches(b"*.*.example", b"a.b.example"));
        // And a server name is never itself a pattern.
        assert!(!dns_name_matches(b"*.example", b"*.example"));
    }

    #[test]
    fn a_wildcard_that_would_match_a_public_suffix_is_refused() {
        // The strongest rule available without a public suffix list: a wildcard must
        // leave at least two labels, so these match nothing at all — not even a name
        // that would otherwise be under them.
        assert!(!dns_name_matches(b"*.com", b"example.com"));
        assert!(!dns_name_matches(b"*.example", b"models.example"));
        // Whereas three labels leave two, which is allowed.
        assert!(dns_name_matches(b"*.example.com", b"models.example.com"));
    }

    #[test]
    fn bytes_that_are_not_host_name_bytes_never_match() {
        // A NUL is not a separator that ends a name; it is a byte that differs.
        assert!(!dns_name_matches(b"models.example", b"models.example\0"));
        assert!(!dns_name_matches(b"models.example\0", b"models.example"));
        assert!(!dns_name_matches(
            b"models.example",
            b"models.example\0.evil"
        ));
        // Nor is a unicode case fold done: the A-label form is the only form.
        assert!(!dns_name_matches(
            b"models.example",
            "mödels.example".as_bytes()
        ));
        assert!(!dns_name_matches(
            "mödels.example".as_bytes(),
            b"models.example"
        ));
        // Empty on either side is not a match.
        assert!(!dns_name_matches(b"", b""));
        assert!(!dns_name_matches(b"models.example", b""));
        assert!(!dns_name_matches(b"", b"models.example"));
    }

    #[test]
    fn a_server_name_is_checked_before_any_pattern_is_looked_at() {
        assert!(server_name_is_usable(b"models.example"));
        assert!(server_name_is_usable(b"a-b.c_d.example"));
        assert!(server_name_is_usable(b"1.2.3.4"));
        // Not usable: empty, oversized, a leading dot, a `*`, a NUL, a slash, a
        // colon, whitespace, and anything outside ASCII.
        assert!(!server_name_is_usable(b""));
        assert!(!server_name_is_usable(&vec![b'a'; 254]));
        assert!(!server_name_is_usable(b".models.example"));
        assert!(!server_name_is_usable(b"*.example"));
        assert!(!server_name_is_usable(b"models\0.example"));
        assert!(!server_name_is_usable(b"models/../example"));
        assert!(!server_name_is_usable(b"models.example:443"));
        assert!(!server_name_is_usable(b"models example"));
        assert!(!server_name_is_usable("mödels.example".as_bytes()));
    }

    #[test]
    fn an_address_is_recognised_as_an_address() {
        assert!(is_ip_literal(b"192.0.2.1"));
        assert!(is_ip_literal(b"1.1.1.1"));
        // Still address-shaped, and so still refused rather than matched as a name.
        assert!(is_ip_literal(b"999.999.999.999"));
        assert!(is_ip_literal(b"2001:db8::1"));
        assert!(is_ip_literal(b"::1"));
        assert!(is_ip_literal(b"fe80::1"));
        assert!(is_ip_literal(b"::ffff:192.0.2.1"));
        // Not addresses.
        assert!(!is_ip_literal(b"models.example"));
        assert!(!is_ip_literal(b"1.2.3"));
        assert!(!is_ip_literal(b"1.2.3.4.5"));
        assert!(!is_ip_literal(b"1.2.3.example"));
        assert!(!is_ip_literal(b"1234.1.1.1"));
        // A host name with a port is a mistake in the caller, not a literal: the
        // letters are not hex digits, so this is not address-shaped.
        assert!(!is_ip_literal(b"models.example:443"));
        assert!(!is_ip_literal(b"models.example:"));
    }

    #[test]
    fn an_error_says_what_it_was_that_failed() {
        assert_eq!(
            LeafError::IpLiteral {
                server_name: b"1.2.3.4".to_vec()
            }
            .to_string(),
            "\"1.2.3.4\" is an IP address, and a dNSName can never match one"
        );
        assert_eq!(
            LeafError::NoNameMatch {
                server_name: b"models.example".to_vec()
            }
            .to_string(),
            "no dNSName in the certificate matches \"models.example\""
        );
        assert!(LeafError::NoDnsNames.to_string().contains("commonName"));
        assert!(LeafError::NoDigitalSignature { key_usage: 0x0020 }
            .to_string()
            .contains("digitalSignature"));
        assert!(LeafError::MissingServerAuth {
            purposes: vec![vec![0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x02]]
        }
        .to_string()
        .contains("serverAuth"));
    }
}
