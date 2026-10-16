//! Certificate parsing against certificates OpenSSL also parses.
//!
//! Twenty-seven real certificates — a root CA, an intermediate, twelve leaves, four
//! self-signed signers (a shadow root and two anchors that are wrong about nothing but
//! their own window and their own key size), two mirrored sub-hierarchies whose CAs
//! state different `pathLenConstraint`s, a certificate its own issuer marked non-CA which
//! signs one anyway, and a **key rollover** (a CA re-issued under the same distinguished
//! name with a new key, the only certificates here that carry key identifiers) — generated
//! by `scripts/tls_x509_fixtures.py`, with every
//! expectation in `data/x509_fixtures.rs` read out of OpenSSL's X.509 parser
//! rather than out of this crate. The test is therefore a comparison of two
//! implementations of RFC 5280, not a record of our own output.
//!
//! The set is chosen to reach the branches that a single "normal" certificate
//! would leave untested:
//!
//! - a signature algorithm with parameters (`RSASSA-PSS`, whose salt and mask are
//!   named inside the `AlgorithmIdentifier`) and one without (`ecdsa-with-SHA256`,
//!   which must have none);
//! - both `Time` encodings: `UTCTime` for the 2020s and `GeneralizedTime` past
//!   2050, where the 1950 pivot the two-digit form needs does not apply;
//! - an RSA `SubjectPublicKeyInfo` and a P-256 one;
//! - a `subjectAltName` holding four different `GeneralName` choices, of which
//!   only the two `dNSName`s are host names;
//! - an unknown extension that must be skipped, and a **critical** unknown
//!   extension that RFC 5280 §4.2 requires to be fatal;
//! - a serial number whose DER needs a leading zero octet (0xFF is negative in one
//!   octet and positive in two);
//! - `basicConstraints` with a path length and without, and a certificate with no
//!   extensions at all.
//!
//! The same certificates are the paths the chain tests at the end of this file
//! build: a real root, a real intermediate and two real leaves. They cover all
//! three signature suites *as links*, both validity bounds exactly, and every
//! structural rule with its own negative, which is why the chain suite lives here
//! rather than in a file of its own — a second test binary would have to redefine
//! the `Expectation` type the generated table is written in.

use hearth_tls::ecdsa::{verify_ecdsa_secp256r1_sha256, EcdsaError};
use hearth_tls::{
    verify_certificate_list, verify_certificate_signature, verify_chain, verify_leaf,
    verify_server_auth, verify_server_name, ChainError, LeafError, Policy, Position, RsaPublicKey,
    Suite, UnixTime, VerifyError,
};

/// `clientAuth` (RFC 5280 §4.2.1.12), the purpose `LEAF_CLIENT_AUTH` names instead of
/// `serverAuth`. Not in `der::oids` because nothing in the crate compares against it —
/// it is here as the *other* side of the check.
const CLIENT_AUTH_OID: &[u8] = &[0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x02];
use hearth_tls::{
    Certificate, Der, GeneralName, KeyUsage, PublicKey, SignatureAlgorithm, Tag, X509Error,
};

include!("data/x509_fixtures.rs");

/// What OpenSSL says about one fixture. The generated table fills these in.
#[derive(Debug)]
struct Expectation {
    der_len: usize,
    /// The certificate's `signatureValue`, counted from the DER rather than assumed:
    /// an RSA signature is the *issuer's* modulus length and one fixture is signed by
    /// a 1024-bit key, so "256" would be an assumption about the fixture set.
    signature_len: usize,
    version: u8,
    subject_cn: &'static str,
    /// The `commonName` of the issuer, so a test can find a certificate's issuer
    /// among the fixtures without a hand-written table that a new fixture can be
    /// left out of.
    issuer_cn: &'static str,
    serial_hex: &'static str,
    not_before: i64,
    not_after: i64,
    is_ca: bool,
    /// Whether `basicConstraints` is present at all. Distinct from `is_ca`: the anchor
    /// rule refuses a stated `cA: FALSE` and allows the absence.
    has_basic_constraints: bool,
    path_len: Option<u32>,
    key_usage: Option<u16>,
    san_dns: &'static [&'static str],
    signature_algorithm: &'static str,
    key: Key,
}

/// One path, with what OpenSSL's own `verify` said about it.
///
/// The paths live in the generated table rather than in the test so that both
/// implementations are asked about exactly the same thing, and the verdicts are recorded
/// rather than taken at test time: the file is a reviewed pin, and a test that shells out
/// to a tool is a test that quietly skips when the tool is absent.
#[derive(Debug)]
struct VerifyCase {
    name: &'static str,
    /// `chain[0]` is the leaf; the rest are the certificates a server would send, in order.
    chain: &'static [&'static str],
    anchors: &'static [&'static str],
    /// The instant both implementations are asked about.
    attime: i64,
    openssl_ok: bool,
    openssl_error: Option<i32>,
    openssl_depth: Option<u32>,
    /// Why a disagreement is expected, or `None` when the two have to agree.
    divergence: Option<&'static str>,
}

/// The public key fields OpenSSL reports, ready to compare.
#[derive(Debug)]
enum Key {
    Rsa {
        modulus: &'static str,
        exponent: u64,
    },
    Ec {
        point: &'static str,
    },
}

fn hex(text: &str) -> Vec<u8> {
    let cleaned: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(cleaned.len().is_multiple_of(2), "odd hex length");
    (0..cleaned.len() / 2)
        .map(|i| u8::from_str_radix(&cleaned[i * 2..i * 2 + 2], 16).expect("hex"))
        .collect()
}

fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn expectation(label: &str) -> &'static Expectation {
    &EXPECTATIONS
        .iter()
        .find(|(name, _)| *name == label)
        .unwrap_or_else(|| panic!("no fixture named {label}"))
        .1
}

/// The generated hex constants, by fixture name.
fn der_hex(label: &str) -> &'static str {
    match label {
        "ROOT" => ROOT_DER_HEX,
        "INTERMEDIATE" => INTERMEDIATE_DER_HEX,
        "LEAF_RSA" => LEAF_RSA_DER_HEX,
        "LEAF_EC" => LEAF_EC_DER_HEX,
        "LEAF_CLIENT_AUTH" => LEAF_CLIENT_AUTH_DER_HEX,
        "LEAF_NO_DIGITAL_SIGNATURE" => LEAF_NO_DIGITAL_SIGNATURE_DER_HEX,
        "SELF_SIGNED_PSS" => SELF_SIGNED_PSS_DER_HEX,
        "SELF_SIGNED_EC" => SELF_SIGNED_EC_DER_HEX,
        "LEAF_UNKNOWN_CRITICAL" => LEAF_UNKNOWN_CRITICAL_DER_HEX,
        "ROOT_SHADOW" => ROOT_SHADOW_DER_HEX,
        "TOP_CA" => TOP_CA_DER_HEX,
        "TOP_CA_STRICT" => TOP_CA_STRICT_DER_HEX,
        "SUB_CA" => SUB_CA_DER_HEX,
        "LEAF_SUB" => LEAF_SUB_DER_HEX,
        "CA_NO_KEY_CERT_SIGN" => CA_NO_KEY_CERT_SIGN_DER_HEX,
        "LEAF_OF_NO_SIGN_CA" => LEAF_OF_NO_SIGN_CA_DER_HEX,
        "SUB_CA_STRICT" => SUB_CA_STRICT_DER_HEX,
        "LEAF_SUB_STRICT" => LEAF_SUB_STRICT_DER_HEX,
        "EXPIRED_ANCHOR" => EXPIRED_ANCHOR_DER_HEX,
        "LEAF_OF_EXPIRED_ANCHOR" => LEAF_OF_EXPIRED_ANCHOR_DER_HEX,
        "WEAK_ANCHOR" => WEAK_ANCHOR_DER_HEX,
        "LEAF_OF_WEAK_ANCHOR" => LEAF_OF_WEAK_ANCHOR_DER_HEX,
        "NON_CA_SIGNER" => NON_CA_SIGNER_DER_HEX,
        "LEAF_OF_NON_CA_SIGNER" => LEAF_OF_NON_CA_SIGNER_DER_HEX,
        "ROLLOVER_CA_A" => ROLLOVER_CA_A_DER_HEX,
        "ROLLOVER_CA_B" => ROLLOVER_CA_B_DER_HEX,
        "LEAF_OF_ROLLOVER_CA_B" => LEAF_OF_ROLLOVER_CA_B_DER_HEX,
        other => panic!("no fixture named {other}"),
    }
}

/// Every fixture, decoded, in the order the generated table lists them.
fn all_fixtures() -> Vec<(&'static str, Vec<u8>)> {
    EXPECTATIONS
        .iter()
        .map(|(name, _)| (*name, hex(der_hex(name))))
        .collect()
}

#[test]
fn every_fixture_parses_to_what_openssl_says_it_says() {
    for (name, der) in all_fixtures() {
        // The one fixture that must *not* parse: its unknown extension is marked
        // critical, which RFC 5280 §4.2 makes fatal, so its expectations are the
        // fields OpenSSL still reports rather than a parse this crate should reach.
        if name == "LEAF_UNKNOWN_CRITICAL" {
            assert!(matches!(
                Certificate::parse(&der),
                Err(X509Error::UnsupportedCriticalExtension(_))
            ));
            continue;
        }
        let want = expectation(name);
        assert_eq!(der.len(), want.der_len, "{name}: the fixture's own length");
        let certificate = Certificate::parse(&der)
            .unwrap_or_else(|error| panic!("{name} did not parse: {error}"));

        // The signature's bytes, not just its algorithm: a `BIT STRING` has an unused-bits
        // octet in front of its value, and reading that octet as part of the value would
        // leave a signature one byte too long for every fixture in the set.
        assert_eq!(
            certificate.signature().len(),
            want.signature_len,
            "{name}: signatureValue"
        );
        assert_eq!(certificate.version(), want.version, "{name}: version");
        assert_eq!(
            certificate
                .subject()
                .common_name()
                .expect("a readable subject"),
            Some(want.subject_cn.as_bytes()),
            "{name}: subject commonName"
        );
        assert_eq!(
            to_hex(certificate.serial()),
            want.serial_hex,
            "{name}: serial"
        );
        assert_eq!(
            certificate.not_before().secs,
            want.not_before,
            "{name}: notBefore"
        );
        assert_eq!(
            certificate.not_after().secs,
            want.not_after,
            "{name}: notAfter"
        );
        assert_eq!(
            certificate.is_ca(),
            want.is_ca,
            "{name}: basicConstraints cA"
        );
        assert_eq!(
            certificate.has_basic_constraints(),
            want.has_basic_constraints,
            "{name}: basicConstraints present"
        );
        assert_eq!(certificate.path_len(), want.path_len, "{name}: pathLen");
        assert_eq!(
            certificate.key_usage().map(|usage| usage.0),
            want.key_usage,
            "{name}: keyUsage"
        );

        let dns: Vec<&str> = certificate
            .subject_alt_names()
            .iter()
            .filter_map(|entry| match entry {
                GeneralName::Dns(name) => Some(core::str::from_utf8(name).expect("a DNS name")),
                _ => None,
            })
            .collect();
        assert_eq!(dns, want.san_dns, "{name}: the dNSNames");

        // The signature algorithm, compared through the same OID text OpenSSL
        // printed rather than through our own variant names alone: a wrong
        // mapping between an OID and a variant is what this catches.
        match (want.signature_algorithm, certificate.signature_algorithm()) {
            ("1.2.840.113549.1.1.11", SignatureAlgorithm::RsaPkcs1Sha256) => {}
            ("1.2.840.113549.1.1.10", SignatureAlgorithm::RsaPssSha256 { salt_len: 32 }) => {}
            ("1.2.840.10045.4.3.2", SignatureAlgorithm::EcdsaSha256) => {}
            (oid, algorithm) => panic!("{name}: OpenSSL says {oid}, we read {algorithm:?}"),
        }

        match (certificate.public_key(), &want.key) {
            (
                PublicKey::Rsa { modulus, exponent },
                Key::Rsa {
                    modulus: want_modulus,
                    exponent: want_exponent,
                },
            ) => {
                assert_eq!(to_hex(modulus), *want_modulus, "{name}: RSA modulus");
                assert_eq!(exponent, want_exponent, "{name}: RSA exponent");
            }
            (PublicKey::Ec { point }, Key::Ec { point: want_point }) => {
                assert_eq!(to_hex(point), *want_point, "{name}: EC point");
            }
            (ours, theirs) => panic!("{name}: OpenSSL says {theirs:?}, we read {ours:?}"),
        }
    }
}

#[test]
fn a_chain_links_by_the_der_of_its_names() {
    // The linkage a chain builder uses: the issuer of one certificate is the
    // subject of the next, compared as the DER of the `Name` rather than as
    // rendered text. Two names can print the same and encode differently, and it
    // is the encoding that the signature covers.
    // A `Certificate` borrows the DER it was parsed from, so the buffers are bound
    // here rather than parsed inside a closure.
    let root_der = hex(der_hex("ROOT"));
    let intermediate_der = hex(der_hex("INTERMEDIATE"));
    let leaf_der = hex(der_hex("LEAF_RSA"));
    let pss_der = hex(der_hex("SELF_SIGNED_PSS"));

    let root = Certificate::parse(&root_der).expect("the root");
    let intermediate = Certificate::parse(&intermediate_der).expect("the intermediate");
    let leaf = Certificate::parse(&leaf_der).expect("the leaf");

    assert_eq!(leaf.issuer().as_der(), intermediate.subject().as_der());
    assert_eq!(intermediate.issuer().as_der(), root.subject().as_der());
    // The names are different, so the equality above is checking something rather
    // than comparing two empty sequences.
    assert_ne!(leaf.subject().as_der(), intermediate.subject().as_der());
    assert_ne!(root.subject().as_der(), intermediate.subject().as_der());

    // And the self-signed ones point at themselves.
    let pss = Certificate::parse(&pss_der).expect("the PSS signer");
    assert_eq!(pss.issuer().as_der(), pss.subject().as_der());
}

#[test]
fn the_bytes_a_signature_covers_are_the_tbs_and_nothing_else() {
    for (name, der) in all_fixtures() {
        // LEAF_UNKNOWN_CRITICAL is the one fixture this crate refuses, by design.
        if name == "LEAF_UNKNOWN_CRITICAL" {
            continue;
        }
        let certificate =
            Certificate::parse(&der).unwrap_or_else(|error| panic!("{name}: {error}"));
        // The bytes a signature covers are the TBS element, and the property that
        // matters is stronger than "the fields agree": the TBS must be a *slice of
        // the input*, so that what gets signed is what was parsed rather than a
        // re-encoding that could differ from it.
        let outer = Der::new(&der)
            .next_element()
            .expect("the certificate's own SEQUENCE");
        assert_eq!(outer.tag(), Tag::SEQUENCE);
        let header = outer.whole().len() - outer.contents().len();
        let offset = certificate.tbs().as_ptr() as usize - der.as_ptr() as usize;
        assert_eq!(
            offset, header,
            "{name}: the TBS is the first element inside the certificate's header"
        );
        assert_eq!(
            &der[offset..offset + certificate.tbs().len()],
            certificate.tbs(),
            "{name}: the TBS is the original bytes"
        );
        assert!(
            der.len() > certificate.tbs().len(),
            "{name}: a certificate is more than its TBS"
        );
        // The signature's shape follows the **issuer's** key, not the subject's:
        // LEAF_EC has a P-256 key and a 256-byte RSA signature, because the CA that
        // signed it holds an RSA-2048 key. The length itself comes from the table —
        // OpenSSL's own count — because it is the issuer's modulus length and the set
        // is no longer uniformly 2048 bits; an ECDSA signature is a DER `SEQUENCE` of
        // two integers, which is short and of varying length.
        let signature_len = expectation(name).signature_len;
        match certificate.signature_algorithm() {
            SignatureAlgorithm::EcdsaSha256 => {
                assert!(
                    certificate.signature().len() < 80,
                    "{name}: an ECDSA signature is two short integers"
                );
                assert_eq!(certificate.signature()[0], 0x30, "{name}: a DER SEQUENCE");
            }
            SignatureAlgorithm::RsaPkcs1Sha256 | SignatureAlgorithm::RsaPssSha256 { .. } => {
                assert_eq!(
                    certificate.signature().len(),
                    signature_len,
                    "{name}: an RSA signature is the issuer's modulus length"
                );
                // The modulus is a whole number of bytes, so an RSA signature is one
                // of 128, 256 or 512 bytes here — never an arbitrary count.
                assert!(signature_len.is_power_of_two(), "{name}: {signature_len}");
            }
            SignatureAlgorithm::Unsupported(oid) => {
                panic!("{name}: unsupported signature algorithm {oid:?}")
            }
        }
    }
}

#[test]
fn the_leafs_subject_alt_name_keeps_every_choice_and_only_the_dns_ones_are_names() {
    let der = hex(der_hex("LEAF_RSA"));
    let certificate = Certificate::parse(&der).expect("the leaf parses");
    let names = certificate.subject_alt_names();
    assert_eq!(names.len(), 4, "the fixture's four GeneralNames");
    assert_eq!(names[0], GeneralName::Dns(b"models.example"));
    assert_eq!(names[1], GeneralName::Dns(b"*.cdn.example"));
    // An rfc822Name and a URI are carried as what they are, not folded into the
    // DNS list: a verifier that treated a mail address as a host name would be
    // accepting a name the CA did not vouch for as a host.
    assert_eq!(names[2], GeneralName::Rfc822(b"admin@models.example"));
    assert_eq!(names[3], GeneralName::Uri(b"https://models.example/"));

    // The wildcard is carried verbatim. Whether it matches is the verifier's
    // decision, and it is a decision that must be made explicitly.
    assert!(names
        .iter()
        .any(|entry| matches!(entry, GeneralName::Dns(d) if *d == b"*.cdn.example")));
}

#[test]
fn the_key_usage_bits_are_the_ones_openssl_reports() {
    let der = hex(der_hex("LEAF_RSA"));
    let certificate = Certificate::parse(&der).expect("the leaf parses");
    let usage = certificate.key_usage().expect("the leaf has keyUsage");
    assert!(usage.has(KeyUsage::DIGITAL_SIGNATURE));
    assert!(usage.has(KeyUsage::KEY_ENCIPHERMENT));
    assert!(!usage.has(KeyUsage::KEY_CERT_SIGN), "a leaf may not sign");
    assert!(!usage.has(KeyUsage::CRL_SIGN));
    assert_eq!(usage.0, 0b101, "digitalSignature and keyEncipherment");

    // And a CA is the other way round: it may sign and it may not use the key for
    // TLS itself.
    let der = hex(der_hex("INTERMEDIATE"));
    let ca_usage = Certificate::parse(&der)
        .expect("the intermediate")
        .key_usage();
    let ca_usage = ca_usage.expect("the intermediate has keyUsage");
    assert!(ca_usage.has(KeyUsage::KEY_CERT_SIGN));
    assert!(ca_usage.has(KeyUsage::CRL_SIGN));
    assert!(!ca_usage.has(KeyUsage::DIGITAL_SIGNATURE));
}

#[test]
fn the_extended_key_usage_names_server_auth() {
    let der = hex(der_hex("LEAF_RSA"));
    let certificate = Certificate::parse(&der).expect("the leaf parses");
    let purposes = certificate.extended_key_usage();
    assert_eq!(purposes.len(), 1);
    assert_eq!(purposes[0].as_bytes(), hearth_tls::der::oids::SERVER_AUTH);
}

#[test]
fn an_unknown_critical_extension_is_a_rejection() {
    let der = hex(der_hex("LEAF_UNKNOWN_CRITICAL"));
    // The same certificate would parse if the extension were not critical: the
    // difference is one BOOLEAN, and it is the difference between a client that
    // understands the certificate and one that must not use it.
    let error = Certificate::parse(&der).expect_err("a critical extension we do not know");
    match error {
        X509Error::UnsupportedCriticalExtension(oid) => {
            assert_eq!(to_hex(&oid), "2b06010401868d1f02");
        }
        other => panic!("expected the critical-extension error, got {other}"),
    }
}

#[test]
fn a_certificate_whose_two_algorithm_identifiers_differ_is_rejected() {
    // RFC 5280 §4.1.1.2 requires the `signatureAlgorithm` inside the signed part
    // and the one outside it to be the same. A certificate where they differ is
    // the shape of an algorithm-confusion attack: the signature checked under one
    // scheme while the fields were read under another. It is also exactly the kind
    // of thing a parser that reads only one of the two would never notice, so it
    // gets a test that constructs the mismatch.
    let mut der = hex(der_hex("LEAF_RSA"));
    let mut algorithm = vec![0x30, 0x0d, 0x06, 0x09];
    algorithm.extend_from_slice(hearth_tls::der::oids::SHA256_WITH_RSA);
    algorithm.extend_from_slice(&[0x05, 0x00]);
    let occurrences: Vec<usize> = der
        .windows(algorithm.len())
        .enumerate()
        .filter(|(_, window)| *window == &algorithm[..])
        .map(|(index, _)| index)
        .collect();
    assert_eq!(occurrences.len(), 2, "one inside the TBS, one outside");
    assert!(Certificate::parse(&der).is_ok(), "the fixture parses first");

    // Change the inner one's NULL parameters to a non-NULL value: same length, so
    // every offset stays put and only the algorithm differs.
    let inner = occurrences[0];
    der[inner + algorithm.len() - 1] = 0x01;
    let error = Certificate::parse(&der).expect_err("the two must agree");
    assert!(
        matches!(error, X509Error::Invalid(message) if message.contains("differs")),
        "{error}"
    );
}

#[test]
fn a_truncated_or_padded_certificate_is_rejected() {
    let der = hex(der_hex("LEAF_RSA"));
    assert!(Certificate::parse(&der).is_ok());

    // Cut at every length from a little past the header to nearly the end: each
    // one must fail rather than parse to something plausible. This is the cheap
    // version of "no prefix of a certificate is a certificate".
    for length in [8, 16, 64, 200, der.len() - 1] {
        assert!(
            Certificate::parse(&der[..length]).is_err(),
            "a {length}-byte prefix was accepted"
        );
    }
    // Trailing bytes are not a certificate either.
    let mut padded = der.clone();
    padded.push(0x00);
    assert!(Certificate::parse(&padded).is_err());
    // Nor is the body without its header.
    assert!(Certificate::parse(&der[4..]).is_err());
}

#[test]
fn the_self_signed_ec_fixture_verifies_with_the_key_inside_it() {
    // The whole reading half in one test: DER framing, the X.509 fields, the
    // P-256 point parser and the ECDSA equation, over a certificate and a
    // signature that OpenSSL produced. Nothing here compares this crate with
    // itself.
    let der = hex(der_hex("SELF_SIGNED_EC"));
    let certificate = Certificate::parse(&der).expect("parses");
    assert!(matches!(
        certificate.signature_algorithm(),
        SignatureAlgorithm::EcdsaSha256
    ));
    let PublicKey::Ec { point } = certificate.public_key() else {
        panic!("the fixture's key is a P-256 point");
    };
    verify_ecdsa_secp256r1_sha256(point, certificate.tbs(), certificate.signature())
        .expect("the certificate verifies with its own key");

    // The bytes the signature covers are the bytes that were read, which is what
    // makes the check above mean anything: change one byte of the serial number,
    // change nothing else, and the same signature must fail. A verifier that
    // re-encoded the fields it had parsed, instead of signing over the input,
    // could pass the first half of this test and fail here.
    let serial = certificate.serial();
    let offset = serial.as_ptr() as usize - der.as_ptr() as usize;
    let mut tampered = der.clone();
    tampered[offset + serial.len() - 1] ^= 1;
    let changed = Certificate::parse(&tampered).expect("a different serial is still a certificate");
    assert_ne!(changed.serial(), serial, "the serial did change");
    let PublicKey::Ec { point } = changed.public_key() else {
        panic!("still a P-256 point");
    };
    assert_eq!(
        verify_ecdsa_secp256r1_sha256(point, changed.tbs(), changed.signature()),
        Err(EcdsaError::VerificationFailed)
    );

    // The RSA-signed fixtures are verified in
    // `every_fixtures_signature_verifies_with_the_key_of_its_issuer`, which needs
    // the PKCS#1 v1.5 or PSS check behind `verify_certificate_signature`; this test
    // is the ECDSA path end to end. Note which rule applies where: TLS 1.3 refuses
    // PKCS#1 v1.5 as a *handshake* signature scheme (§4.2.3), while nearly every
    // certificate in the world is signed with it — that is the one place the older
    // scheme is still accepted, and the distinction is why the two live in
    // different modules.
}

/// Parses a certificate per label, keeping the buffers alive for the returned
/// certificates to borrow.
fn certificates_of<'a>(ders: &'a [Vec<u8>]) -> Vec<Certificate<'a>> {
    ders.iter()
        .map(|der| Certificate::parse(der).expect("the fixture parses"))
        .collect()
}

#[test]
fn every_fixtures_signature_verifies_with_the_key_of_its_issuer() {
    // The chain, link by link, checked in the only way that means anything: the
    // signature on each certificate verified with the public key of the
    // certificate that issued it — over the `tbsCertificate` bytes that were read,
    // with the suite the certificate names. All sixteen fixtures and three suites:
    // fourteen PKCS#1 v1.5 (the scheme TLS 1.3 removed from the handshake and which
    // every certificate still uses), one RSA-PSS, one ECDSA.
    //
    // Every fixture is covered because the loop runs over `EXPECTATIONS` and finds
    // each certificate's issuer through the `issuer_cn` the generator records. The
    // hand-written table this replaced could silently omit a new fixture; this
    // cannot.
    //
    // `LEAF_UNKNOWN_CRITICAL` is deliberately absent: a critical extension this
    // crate does not understand is a rejection before any signature is looked at,
    // which `an_unknown_critical_extension_is_a_rejection` asserts on its own.
    let ders: Vec<(&str, Vec<u8>)> = EXPECTATIONS
        .iter()
        .map(|(name, _)| (*name, hex(der_hex(name))))
        .collect();
    let parsed: Vec<(&str, Option<Certificate>)> = ders
        .iter()
        .map(|(name, der)| (*name, Certificate::parse(der).ok()))
        .collect();

    for (name, certificate) in &parsed {
        let Some(certificate) = certificate else {
            assert_eq!(*name, "LEAF_UNKNOWN_CRITICAL", "{name} did not parse");
            continue;
        };
        let want = expectation(name);
        // The issuer is found by name *and* by signature, because either half alone
        // is wrong: the name alone is ambiguous — `ROOT_SHADOW` carries the root's
        // subject name with a key of its own, which is what makes "a name match is
        // not trust" testable — and a signature alone would not check that the
        // certificate names its issuer.
        let mut issued_by = 0usize;
        for (candidate_name, candidate) in &parsed {
            let Some(candidate) = candidate else { continue };
            if expectation(candidate_name).subject_cn != want.issuer_cn {
                continue;
            }
            if verify_certificate_signature(certificate, candidate.public_key()).is_ok() {
                assert_eq!(
                    certificate.issuer().as_der(),
                    candidate.subject().as_der(),
                    "{name} verifies with {candidate_name}'s key but does not name it as issuer"
                );
                issued_by += 1;
            }
        }
        assert_eq!(issued_by, 1, "{name}: {issued_by} fixtures issued it");
    }

    // The control that makes the loop mean something: a key of the right *kind* but
    // the wrong identity must fail, so the loop cannot be passing on key-type
    // matching alone. `LEAF_RSA` was signed by the intermediate, not the root.
    //
    // Which rule fires is not pinned, and pinning it would be pinning an accident:
    // the wrong key's public-key operation produces a block that is effectively
    // random, so it can fail the block type, the padding, the `DigestInfo` or the
    // hash depending on the key — all of them are the same refusal to a caller.
    let pair = ders_for(&["ROOT", "LEAF_RSA"]);
    let certificates = certificates_of(&pair);
    let wrong_issuer = verify_certificate_signature(&certificates[1], certificates[0].public_key());
    assert!(
        matches!(wrong_issuer, Err(VerifyError::Rsa(_))),
        "the root's key should fail the RSA check on the leaf: {wrong_issuer:?}"
    );
}

#[test]
fn a_scheme_and_a_key_that_cannot_go_together_are_refused() {
    // Each pair below is one real certificate and one real key from the fixtures.
    // Nothing is wrong with either on its own; what is wrong is the combination,
    // and the combination is checked before any arithmetic — which is what the
    // error says, rather than a signature failure that would look like a bad
    // certificate.
    let ders: Vec<Vec<u8>> = [
        "ROOT",
        "LEAF_RSA",
        "LEAF_EC",
        "SELF_SIGNED_PSS",
        "SELF_SIGNED_EC",
    ]
    .iter()
    .map(|label| hex(der_hex(label)))
    .collect();
    let certificates = certificates_of(&ders);
    let (root, leaf_rsa, leaf_ec, pss, ec) = (
        &certificates[0],
        &certificates[1],
        &certificates[2],
        &certificates[3],
        &certificates[4],
    );

    // `LEAF_EC` has a P-256 key and an **RSA signature**, because its issuer is
    // RSA: the scheme comes from the issuer's key type, not the subject's, which is
    // exactly the confusion this check exists to prevent. So its own EC key must
    // not verify it...
    assert_eq!(
        verify_certificate_signature(leaf_ec, leaf_ec.public_key()),
        Err(VerifyError::SchemeKeyMismatch {
            suite: "sha256WithRSAEncryption",
            key: "a P-256 key",
        })
    );
    // ... and `ROOT`'s RSA key must not verify an ECDSA-signed certificate with an
    // RSA suite either: the reverse direction, where a scheme is asked to run on a
    // key that cannot produce it.
    assert_eq!(
        verify_certificate_signature(ec, pss.public_key()),
        Err(VerifyError::SchemeKeyMismatch {
            suite: "ecdsa_secp256r1_sha256",
            key: "an RSA key",
        })
    );
    // An RSA-signed certificate with a P-256 key is the same refusal from the other
    // entry: `roota` key type on an RSA suite.
    assert_eq!(
        verify_certificate_signature(leaf_rsa, leaf_ec.public_key()),
        Err(VerifyError::SchemeKeyMismatch {
            suite: "sha256WithRSAEncryption",
            key: "a P-256 key",
        })
    );
    // A PSS-signed certificate with a P-256 key.
    assert_eq!(
        verify_certificate_signature(pss, ec.public_key()),
        Err(VerifyError::SchemeKeyMismatch {
            suite: "rsa_pss_rsae_sha256",
            key: "a P-256 key",
        })
    );
    // And the RSA-PSS fixture with the RSA key that is the wrong *identity* still
    // fails on the signature rather than on the pairing, so the pairing check is
    // not standing in for verification.
    let wrong_identity = verify_certificate_signature(pss, root.public_key());
    assert!(
        matches!(wrong_identity, Err(VerifyError::Rsa(_))),
        "a wrong RSA key of the right kind must fail the RSA check: {wrong_identity:?}"
    );
    // The suite each fixture will be checked with, for the record: these are the
    // three schemes a real chain uses, and they are asserted here so a change to
    // the mapping has to change this test too.
    assert_eq!(
        Suite::from_certificate_algorithm(root.signature_algorithm()),
        Ok(Suite::RsaPkcs1Sha256)
    );
    assert_eq!(
        Suite::from_certificate_algorithm(ec.signature_algorithm()),
        Ok(Suite::EcdsaSecp256r1Sha256)
    );
    assert_eq!(
        Suite::from_certificate_algorithm(pss.signature_algorithm()),
        Ok(Suite::RsaPssSha256)
    );
}

/// The DER of one fixture, by label.
fn fixture(label: &str) -> Vec<u8> {
    hex(der_hex(label))
}

/// Frames DERs as a `certificate_list`, the way a `Certificate` message carries
/// them: three bytes of length, then exactly that much DER, for each entry.
fn certificate_list(ders: &[Vec<u8>]) -> Vec<u8> {
    let mut out = Vec::new();
    for der in ders {
        out.extend_from_slice(&(der.len() as u32).to_be_bytes()[1..]);
        out.extend_from_slice(der);
    }
    out
}

/// The instant `delta` seconds after `secs`, for the boundary cases.
fn plus(secs: i64, delta: i64) -> UnixTime {
    UnixTime::from_secs(secs + delta)
}

/// The DER buffers of the named fixtures, as an owned `Vec` a caller binds so
/// that `certificates_of` can hand back certificates that borrow it.
fn ders_for(labels: &[&str]) -> Vec<Vec<u8>> {
    labels.iter().map(|label| fixture(label)).collect()
}

// ---------------------------------------------------------------------------
// Chain building, over the same certificates
//
// Every call site in `chain.rs` has a negative test here. Two of them took fixtures
// of their own — the *anchor's* validity window and the *anchor's* key size, which no
// other certificate can reach, because every other fixture anchor is inside its window
// and at least 2048 bits at any instant a chain of its own is still valid.
// ---------------------------------------------------------------------------

#[test]
fn the_fixture_chain_verifies_from_the_leaf_to_the_root() {
    let chain_ders = [fixture("LEAF_RSA"), fixture("INTERMEDIATE")];
    let anchor_ders = [fixture("ROOT")];
    let chain = certificates_of(&chain_ders);
    let anchors = certificates_of(&anchor_ders);
    // The leaf's own lower bound is inside the intermediate's window and the
    // root's as well, which the assertions below require by verifying the whole
    // path at that instant rather than at a transcribed date.
    let time = chain[0].not_before();

    assert_eq!(
        verify_chain(&chain, &anchors, time, &Policy::default()),
        Ok(()),
        "leaf -> intermediate -> root"
    );

    // The same path with the root included in the presented chain: a server may
    // send it, and the anchor is then the last certificate rather than the one
    // outside it.
    let with_root_ders = [
        fixture("LEAF_RSA"),
        fixture("INTERMEDIATE"),
        fixture("ROOT"),
    ];
    let with_root = certificates_of(&with_root_ders);
    assert_eq!(
        verify_chain(&with_root, &anchors, time, &Policy::default()),
        Ok(())
    );

    // And an anchor need not be a root: pinning the intermediate is a legitimate
    // thing for a caller to do, and the path is then one certificate long.
    let leaf_only_ders = [fixture("LEAF_RSA")];
    let leaf_only = certificates_of(&leaf_only_ders);
    let intermediate_ders = ders_for(&["INTERMEDIATE"]);
    let intermediate = certificates_of(&intermediate_ders);
    assert_eq!(
        verify_chain(&leaf_only, &intermediate, time, &Policy::default()),
        Ok(())
    );
}

#[test]
fn every_suite_verifies_as_a_chain_link() {
    // The three suites this crate implements, each as a *link* rather than as a
    // bare signature. PKCS#1 v1.5 is what `sha256WithRSAEncryption` gives, and it
    // is what both links of the chain above are signed with; RSA-PSS and ECDSA are
    // the two self-signed fixtures, which pin themselves as their own anchors. A
    // chain builder that handled only one suite would fail the others here.
    for label in ["SELF_SIGNED_PSS", "SELF_SIGNED_EC"] {
        let ders = [fixture(label)];
        let certificate = certificates_of(&ders);
        let time = certificate[0].not_before();
        assert_eq!(
            verify_chain(&certificate, &certificate, time, &Policy::default()),
            Ok(()),
            "{label} pinned as its own anchor, verified with its own key"
        );
    }
}

#[test]
fn the_validity_window_is_inclusive_at_both_ends() {
    let chain_ders = [fixture("LEAF_RSA"), fixture("INTERMEDIATE")];
    let anchor_ders = [fixture("ROOT")];
    let chain = certificates_of(&chain_ders);
    let anchors = certificates_of(&anchor_ders);
    let policy = Policy::default();
    let (not_before, not_after) = (chain[0].not_before().secs, chain[0].not_after().secs);
    assert!(not_before < not_after, "the fixture has a window at all");

    // Both bounds are inside the window, inclusively — RFC 5280's rule, and the
    // difference between accepting and refusing a certificate at its own boundary.
    for secs in [not_before, not_after] {
        assert_eq!(
            verify_chain(&chain, &anchors, UnixTime::from_secs(secs), &policy),
            Ok(()),
            "{secs} is inside the leaf's window"
        );
    }
    // One second either side is not, and the error says which end was crossed.
    assert_eq!(
        verify_chain(&chain, &anchors, plus(not_before, -1), &policy),
        Err(ChainError::NotYetValid(Position::Chain(0)))
    );
    assert_eq!(
        verify_chain(&chain, &anchors, plus(not_after, 1), &policy),
        Err(ChainError::Expired(Position::Chain(0)))
    );
}

#[test]
fn an_issuers_expired_window_is_fatal_even_when_the_leaf_is_valid() {
    // The EC leaf is valid from 2050, ten years after the intermediate that signed
    // it stopped being valid, so the two windows do not overlap at all. That makes
    // one property checkable that no single certificate can show: an issuer's own
    // window is enforced, and the refusal names the issuer rather than the leaf.
    let chain_ders = [fixture("LEAF_EC"), fixture("INTERMEDIATE")];
    let anchor_ders = [fixture("ROOT")];
    let chain = certificates_of(&chain_ders);
    let anchors = certificates_of(&anchor_ders);
    let leaf_start = chain[0].not_before();
    assert!(
        leaf_start > chain[1].not_after(),
        "the fixture's overlap is empty, which is the case being asserted"
    );

    assert_eq!(
        verify_chain(&chain, &anchors, leaf_start, &Policy::default()),
        Err(ChainError::Expired(Position::Chain(1))),
        "the leaf is valid at this instant and its issuer is not"
    );
}

#[test]
fn a_leaf_that_claims_to_be_a_ca_is_refused() {
    // `INTERMEDIATE` is a CA certificate, so presenting it as the server's own is
    // exactly the confusion this refuses. Nothing else about it is wrong: it is
    // valid, its key is the right size, and it anchors in the root.
    let chain_ders = [fixture("INTERMEDIATE")];
    let anchor_ders = [fixture("ROOT")];
    let chain = certificates_of(&chain_ders);
    let anchors = certificates_of(&anchor_ders);

    assert_eq!(
        verify_chain(&chain, &anchors, chain[0].not_before(), &Policy::default()),
        Err(ChainError::CaUsedAsLeaf)
    );
}

#[test]
fn an_issuer_that_is_not_a_ca_is_refused() {
    // A leaf certificate has no business signing another certificate, whatever its
    // name says. `LEAF_EC` is not a CA, so a chain that puts it in an issuer
    // position is refused for that, before any name is compared.
    let chain_ders = [fixture("LEAF_RSA"), fixture("LEAF_EC")];
    let chain = certificates_of(&chain_ders);
    let anchor_ders = ders_for(&["ROOT"]);
    let anchors = certificates_of(&anchor_ders);

    assert_eq!(
        verify_chain(&chain, &anchors, chain[0].not_before(), &Policy::default()),
        Err(ChainError::NotACa(Position::Chain(1)))
    );
}

#[test]
fn a_chain_that_omits_its_intermediate_is_broken() {
    // The common server mistake: send the leaf and the root and leave out the
    // intermediate. The root is a CA, so the only thing wrong is the link, and the
    // refusal says which one.
    let chain_ders = [fixture("LEAF_RSA"), fixture("ROOT")];
    let chain = certificates_of(&chain_ders);
    let anchor_ders = ders_for(&["ROOT"]);
    let anchors = certificates_of(&anchor_ders);

    assert_eq!(
        verify_chain(&chain, &anchors, chain[0].not_before(), &Policy::default()),
        Err(ChainError::BrokenLink { index: 0 })
    );
}

#[test]
fn a_path_that_reaches_no_anchor_is_refused() {
    // Everything about this path is fine except that its top certificate was
    // signed by a root the caller does not trust. The refusal carries the DER of
    // the name that could not be resolved, so a diagnostic can say which anchor is
    // missing.
    let chain_ders = [fixture("LEAF_RSA"), fixture("INTERMEDIATE")];
    let anchor_ders = [fixture("SELF_SIGNED_EC")];
    let chain = certificates_of(&chain_ders);
    let anchors = certificates_of(&anchor_ders);
    let root_ders = ders_for(&["ROOT"]);
    let root = certificates_of(&root_ders);

    assert_eq!(
        verify_chain(&chain, &anchors, chain[0].not_before(), &Policy::default()),
        Err(ChainError::Unanchored {
            issuer: root[0].subject().as_der().to_vec()
        })
    );
}

#[test]
fn the_policy_caps_the_length_and_the_key_size() {
    let chain_ders = [fixture("LEAF_RSA"), fixture("INTERMEDIATE")];
    let anchor_ders = [fixture("ROOT")];
    let chain = certificates_of(&chain_ders);
    let anchors = certificates_of(&anchor_ders);
    let time = chain[0].not_before();

    // A cap of one certificate refuses a two-certificate chain, and says both
    // numbers.
    let capped = Policy {
        max_chain_len: 1,
        ..Policy::default()
    };
    assert_eq!(
        verify_chain(&chain, &anchors, time, &capped),
        Err(ChainError::ChainTooLong { len: 2, max: 1 })
    );

    // The key-size floor applies to the *leaf's* key, not only to issuers': a leaf
    // whose key is weaker than the policy allows is refused even though its own
    // signature is fine.
    let strict = Policy {
        min_rsa_bits: 4096,
        ..Policy::default()
    };
    let leaf_only_ders = ders_for(&["LEAF_RSA"]);
    let leaf_only = certificates_of(&leaf_only_ders);
    let intermediate_ders = ders_for(&["INTERMEDIATE"]);
    let intermediate = certificates_of(&intermediate_ders);
    assert_eq!(
        verify_chain(&leaf_only, &intermediate, time, &strict),
        Err(ChainError::KeyTooSmall {
            at: Position::Chain(0),
            bits: 2048,
            min_bits: 4096,
        })
    );
    // The default policy is inside every fixture's key size, which is what lets
    // the tests above run at all.
    assert_eq!(
        verify_chain(&chain, &anchors, time, &Policy::default()),
        Ok(())
    );
}

#[test]
fn a_path_len_constraint_is_enforced() {
    // `pathLenConstraint` counts the non-self-issued CAs *below* the one that stated
    // it, and both halves of the check are reachable here: a CA inside the chain, and
    // the anchor outside it.
    //
    // The legal path first. `SUB_CA` allows none below it and has none; `TOP_CA`
    // allows one and has one. The leaf is below both and does not count, which is the
    // part of the rule an off-by-one gets wrong.
    let chain_ders = [fixture("LEAF_SUB"), fixture("SUB_CA")];
    let chain = certificates_of(&chain_ders);
    let top_ders = ders_for(&["TOP_CA"]);
    let top = certificates_of(&top_ders);
    let time = chain[0].not_before();
    assert_eq!(chain[1].path_len(), Some(0));
    assert_eq!(top[0].path_len(), Some(1));
    assert_eq!(
        verify_chain(&chain, &top, time, &Policy::default()),
        Ok(()),
        "one intermediate below a CA that allows one"
    );

    // One intermediate too many, refused where it stands in the chain rather than at
    // the anchor. The strict subtree is a mirror of the one above — the same two
    // subject names and the same key usage, with the top CA's limit at zero instead of
    // one — and `TOP_CA`'s key really did sign everything under it, so the links and
    // signatures are checked here rather than assumed. With the limit removed this path
    // would verify: it is refused for the limit and for nothing else.
    let over_ders = [
        fixture("LEAF_SUB_STRICT"),
        fixture("SUB_CA_STRICT"),
        fixture("TOP_CA_STRICT"),
    ];
    let over = certificates_of(&over_ders);
    assert_eq!(over[2].path_len(), Some(0));
    assert_eq!(
        over[2].subject().as_der(),
        top[0].subject().as_der(),
        "one name, two limits"
    );
    for pair in over.windows(2) {
        assert_eq!(pair[0].issuer().as_der(), pair[1].subject().as_der());
        assert_eq!(
            verify_certificate_signature(&pair[0], pair[1].public_key()),
            Ok(()),
            "each link was signed by the key that follows it"
        );
    }
    assert_eq!(
        verify_chain(&over, &top, time, &Policy::default()),
        Err(ChainError::PathLengthExceeded {
            ca: Position::Chain(2),
            allowed: 0,
            found: 1,
        })
    );

    // And the anchor's own limit, which is the same check at `Position::Anchor`: the
    // root allows one intermediate below it and this path has two. Everything else
    // about the path is valid — the intermediate's own limit is respected, the names
    // link, the signatures verify — so the refusal comes from the anchor alone.
    let deep_ders = [fixture("LEAF_SUB"), fixture("SUB_CA"), fixture("TOP_CA")];
    let deep = certificates_of(&deep_ders);
    let root_ders = ders_for(&["ROOT"]);
    let root = certificates_of(&root_ders);
    assert_eq!(root[0].path_len(), Some(1));
    assert_eq!(
        verify_certificate_signature(&deep[2], root[0].public_key()),
        Ok(()),
        "the anchor did issue the top of the path; only its limit refuses it"
    );
    assert_eq!(
        verify_chain(&deep, &root, time, &Policy::default()),
        Err(ChainError::PathLengthExceeded {
            ca: Position::Anchor(0),
            allowed: 1,
            found: 2,
        })
    );
}

#[test]
fn an_issuer_that_does_not_claim_key_cert_sign_is_refused() {
    // `CA_NO_KEY_CERT_SIGN` says it is a CA — `basicConstraints` is `CA:TRUE` with
    // `pathLenConstraint: 0` — and then its `keyUsage` names `digitalSignature` and
    // `cRLSign` and not `keyCertSign`. RFC 5280 §4.2.1.9 makes that key unusable for
    // signing certificates whatever `basicConstraints` claims, so the two extensions
    // disagree and the path is refused for it.
    let missing = KeyUsage::DIGITAL_SIGNATURE | KeyUsage::CRL_SIGN;

    // In the chain, where the certificate is the leaf's own issuer: the names link
    // and the signature verifies, and the usage is the only defect.
    let chain_ders = [
        fixture("LEAF_OF_NO_SIGN_CA"),
        fixture("CA_NO_KEY_CERT_SIGN"),
    ];
    let chain = certificates_of(&chain_ders);
    assert!(chain[1].is_ca(), "the fixture is a CA, which is the point");
    assert_eq!(
        verify_certificate_signature(&chain[0], chain[1].public_key()),
        Ok(()),
        "the link itself is sound"
    );
    let root_ders = ders_for(&["ROOT"]);
    let root = certificates_of(&root_ders);
    assert_eq!(
        verify_chain(&chain, &root, chain[0].not_before(), &Policy::default()),
        Err(ChainError::MissingKeyCertSign {
            at: Position::Chain(1),
            key_usage: missing,
        })
    );

    // And as the anchor, where the same check runs too — and *before* the signature
    // it would otherwise pass, so being pinned as trusted is not an exemption from
    // claiming the right to sign certificates.
    let leaf_ders = ders_for(&["LEAF_OF_NO_SIGN_CA"]);
    let leaf = certificates_of(&leaf_ders);
    let ca_ders = ders_for(&["CA_NO_KEY_CERT_SIGN"]);
    let ca = certificates_of(&ca_ders);
    assert_eq!(
        verify_certificate_signature(&leaf[0], ca[0].public_key()),
        Ok(()),
        "the anchor really did sign this leaf"
    );
    assert_eq!(
        verify_chain(&leaf, &ca, leaf[0].not_before(), &Policy::default()),
        Err(ChainError::MissingKeyCertSign {
            at: Position::Anchor(0),
            key_usage: missing,
        })
    );
}

#[test]
fn the_anchors_own_window_and_key_size_are_checked() {
    // The last two call sites, and the reason they needed fixtures of their own: an
    // anchor is only checked against its own window and its own key size when the path
    // is otherwise sound, so each one here has a leaf it really did sign. The signature
    // assertions below are what make that a check rather than a claim.
    let expired_leaf_ders = ders_for(&["LEAF_OF_EXPIRED_ANCHOR"]);
    let expired_leaf = certificates_of(&expired_leaf_ders);
    let expired_anchor_ders = ders_for(&["EXPIRED_ANCHOR"]);
    let expired_anchor = certificates_of(&expired_anchor_ders);
    let leaf_time = expired_leaf[0].not_before();
    assert!(
        leaf_time > expired_anchor[0].not_after(),
        "the two windows do not overlap, which is the case being asserted"
    );
    assert_eq!(
        verify_certificate_signature(&expired_leaf[0], expired_anchor[0].public_key()),
        Ok(()),
        "the anchor really did sign this leaf"
    );
    assert_eq!(
        verify_chain(
            &expired_leaf,
            &expired_anchor,
            leaf_time,
            &Policy::default()
        ),
        Err(ChainError::Expired(Position::Anchor(0))),
        "the leaf is valid at this instant and its anchor is not"
    );

    // The same shape with a key below the floor. A 1024-bit CA is well-formed and
    // `rsa.rs` accepts the key — RFC 8448's own fixture is one — so the refusal has to
    // come from the policy, which is why the floor lives in `chain.rs` and not there.
    let weak_leaf_ders = ders_for(&["LEAF_OF_WEAK_ANCHOR"]);
    let weak_leaf = certificates_of(&weak_leaf_ders);
    let weak_anchor_ders = ders_for(&["WEAK_ANCHOR"]);
    let weak_anchor = certificates_of(&weak_anchor_ders);
    // The key really is 1024 bits — read the way the chain builder reads it, so this is
    // the same number the policy compares against.
    match weak_anchor[0].public_key() {
        PublicKey::Rsa { modulus, exponent } => {
            let key = RsaPublicKey::from_be_bytes(modulus, &exponent.to_be_bytes())
                .expect("the fixture's own key reads");
            assert_eq!(key.key_bits(), 1024);
        }
        other => panic!("the fixture is RSA, not {other:?}"),
    }
    assert!(weak_anchor[0].is_ca(), "it is an ordinary CA, only weak");
    assert_eq!(
        verify_certificate_signature(&weak_leaf[0], weak_anchor[0].public_key()),
        Ok(()),
        "the anchor really did sign this leaf"
    );
    assert_eq!(
        verify_chain(
            &weak_leaf,
            &weak_anchor,
            weak_leaf[0].not_before(),
            &Policy::default()
        ),
        Err(ChainError::KeyTooSmall {
            at: Position::Anchor(0),
            bits: 1024,
            min_bits: 2048,
        })
    );
}

#[test]
fn an_anchor_that_states_it_is_not_a_ca_is_refused() {
    // An anchor's key verifies a signature on *another* certificate, which is the act of a
    // CA — the same argument as the `keyCertSign` check beside it — so an anchor that says
    // `cA: FALSE` is refused for the contradiction.
    let leaf_ders = ders_for(&["LEAF_OF_NON_CA_SIGNER"]);
    let leaf = certificates_of(&leaf_ders);
    let signer_ders = ders_for(&["NON_CA_SIGNER"]);
    let signer = certificates_of(&signer_ders);
    assert!(
        signer[0].has_basic_constraints(),
        "the fixture states it is not a CA, which is the case being asserted"
    );
    assert!(!signer[0].is_ca());
    assert!(
        signer[0]
            .key_usage()
            .expect("a keyUsage")
            .has(KeyUsage::KEY_CERT_SIGN),
        "and it claims the right to sign certificates, so its two claims disagree"
    );
    assert_eq!(
        verify_certificate_signature(&leaf[0], signer[0].public_key()),
        Ok(()),
        "the anchor really did sign this leaf, so the path is otherwise sound"
    );
    assert_eq!(
        verify_chain(&leaf, &signer, leaf[0].not_before(), &Policy::default()),
        Err(ChainError::NotACa(Position::Anchor(0)))
    );

    // The same certificate in the chain, where an *absent* `basicConstraints` is refused
    // too: a link that arrives from the wire has to state that it may issue certificates.
    // Same fixture, same rule, a different position — and a different reason for the same
    // error, which is what the two assertions together pin down.
    let link_ders = ders_for(&["LEAF_OF_NON_CA_SIGNER", "NON_CA_SIGNER"]);
    let link = certificates_of(&link_ders);
    let root_ders = ders_for(&["ROOT"]);
    let root = certificates_of(&root_ders);
    assert_eq!(
        verify_chain(&link, &root, link[0].not_before(), &Policy::default()),
        Err(ChainError::NotACa(Position::Chain(1)))
    );
}

#[test]
fn pinning_a_certificate_is_an_identity_and_not_an_issuance() {
    // The rule above refuses a stated `cA: FALSE` because the anchor is signing off on
    // someone else's certificate. Pinning is the other case: the caller says "this exact
    // certificate is the one I mean", and the certificate proves its own identity rather
    // than claiming a right to issue. Two shapes of that are accepted, and between them
    // they are what keeps the rule from drifting into "an anchor must have
    // `basicConstraints`, and it must say `cA: TRUE`" — which would break pinning a bare
    // certificate, or a self-signed server certificate, to fix nothing.
    //
    // Stated `cA: FALSE`, pinned as its own anchor: accepted. `SELF_SIGNED_PSS` is exactly
    // that certificate, and OpenSSL accepts it too — while refusing a *non-CA* used as an
    // issuer, which is the case the test above asserts.
    let pss_ders = ders_for(&["SELF_SIGNED_PSS"]);
    let pss = certificates_of(&pss_ders);
    assert!(
        pss[0].has_basic_constraints() && !pss[0].is_ca(),
        "the fixture states cA: FALSE, which is the case being asserted"
    );
    assert_eq!(
        verify_chain(&pss, &pss, pss[0].not_before(), &Policy::default()),
        Ok(()),
        "an identity pin is not an issuance claim"
    );

    // No `basicConstraints` at all: accepted. `SELF_SIGNED_EC` carries no extensions
    // whatsoever, so it is the whole of that case.
    let bare_ders = ders_for(&["SELF_SIGNED_EC"]);
    let bare = certificates_of(&bare_ders);
    assert!(!bare[0].has_basic_constraints());
    assert_eq!(
        verify_chain(&bare, &bare, bare[0].not_before(), &Policy::default()),
        Ok(()),
        "a pin with no basicConstraints is not refused for the absence"
    );

    // And the exemption cannot be walked out of: the same stated-`cA: FALSE` certificate
    // is refused the moment it is asked to authenticate anything but itself, in the chain
    // or at the anchor — the two assertions in the test above are what say so.
}

// ---------------------------------------------------------------------------
// A rule that only a key rollover can reach
// ---------------------------------------------------------------------------

#[test]
fn a_self_issued_ca_does_not_spend_its_parents_allowance() {
    // RFC 5280 §4.2.1.9 excludes self-issued certificates from a `pathLenConstraint` count,
    // and "self-issued" is a statement about the **names**: `ROLLOVER_CA_B` was signed by
    // `ROLLOVER_CA_A`'s key and carries `ROLLOVER_CA_A`'s distinguished name, so it is
    // self-issued even though it is not self-signed. That is what makes a key rollover
    // legal at all — `ROLLOVER_CA_A` allows no intermediates below it, and the rollover is
    // not one.
    //
    // Every other fixture here is non-self-issued, so without this path the branch that
    // does the excluding is never exercised: it is the one rule in `chain.rs` whose
    // behaviour no earlier fixture could reach.
    let ders = ders_for(&["LEAF_OF_ROLLOVER_CA_B", "ROLLOVER_CA_B", "ROLLOVER_CA_A"]);
    let chain = certificates_of(&ders);
    let root_ders = ders_for(&["ROOT"]);
    let root = certificates_of(&root_ders);
    assert_eq!(
        chain[2].path_len(),
        Some(0),
        "the constrained CA allows none"
    );
    assert_eq!(
        chain[1].subject().as_der(),
        chain[1].issuer().as_der(),
        "the same DN in both fields, which is what makes it self-issued"
    );
    assert_ne!(
        chain[1].public_key(),
        chain[2].public_key(),
        "and it is not self-signed: the rollover really is a new key"
    );
    assert_eq!(
        verify_chain(&chain, &root, chain[0].not_before(), &Policy::default()),
        Ok(()),
        "legal only because the self-issued certificate does not count against CA_A"
    );

    // The contrast, so that the assertion above is about self-issuance and not about
    // permissiveness: a second CA that is *not* self-issued does spend the allowance, and
    // the sibling CA that allows none refuses it. Same shape, one property different.
    let strict_ders = ders_for(&["LEAF_SUB_STRICT", "SUB_CA_STRICT"]);
    let strict = certificates_of(&strict_ders);
    let strict_anchor_ders = ders_for(&["TOP_CA_STRICT"]);
    let strict_anchor = certificates_of(&strict_anchor_ders);
    assert_ne!(
        strict[1].subject().as_der(),
        strict[1].issuer().as_der(),
        "this CA is not self-issued"
    );
    assert_eq!(
        verify_chain(
            &strict,
            &strict_anchor,
            strict[0].not_before(),
            &Policy::default()
        ),
        Err(ChainError::PathLengthExceeded {
            ca: Position::Anchor(0),
            allowed: 0,
            found: 1,
        })
    );
}

// ---------------------------------------------------------------------------
// The oracle: the same paths, put to a second implementation
// ---------------------------------------------------------------------------

#[test]
fn every_path_is_verified_the_same_way_the_openssl_cli_verifies_it() {
    // Everything above compares our reading of a certificate against OpenSSL's reading of
    // the same bytes, which leaves the *policy* — which rules run, and what each one means
    // — checked only against our own understanding of RFC 5280. This asks a second
    // implementation of path validation about the same paths at the same instants.
    //
    // The recorded verdicts come from `openssl verify -partial_chain -auth_level 0` (the
    // version is in the generated table). `-partial_chain` is the closest equivalent to
    // this crate's rule that an anchor need not be self-signed; `-auth_level 0` keeps
    // OpenSSL's separate security-level policy out of a comparison about structure. So a
    // disagreement is either a bug here or a written-down divergence, and a divergence is
    // only ever allowed in the safe direction: OpenSSL accepting where this crate refuses.
    // This crate being the more permissive side is a failure, not a footnote.
    //
    // Only the *verdict* is compared. Which certificate gets blamed is not comparable:
    // OpenSSL numbers depths from the leaf and counts the anchor when it is a chain
    // element, and it reports the highest expired certificate while this crate reports the
    // first one it reaches — two refusals for the same expired path, four depths apart.    // Every disagreement is collected and reported together: a test that stops at the
    // first one turns adding a path into one run per surprise, and the *set* of
    // disagreements is what a reviewer needs to see.
    let mut disagreements: Vec<String> = Vec::new();
    for case in VERIFY_CASES {
        let chain_ders: Vec<Vec<u8>> = case.chain.iter().map(|label| fixture(label)).collect();
        let anchor_ders: Vec<Vec<u8>> = case.anchors.iter().map(|label| fixture(label)).collect();
        let chain = certificates_of(&chain_ders);
        let anchors = certificates_of(&anchor_ders);
        let ours = verify_chain(
            &chain,
            &anchors,
            UnixTime::from_secs(case.attime),
            &Policy::default(),
        );

        match case.divergence {
            Some(reason) => {
                assert!(
                    !reason.is_empty(),
                    "{}: a divergence needs a reason",
                    case.name
                );
                if ours.is_ok() {
                    disagreements.push(format!(
                        "{}: recorded as a divergence ({reason}) but this crate accepts it, so the \
                         divergence has gone stale",
                        case.name
                    ));
                }
                if !case.openssl_ok {
                    disagreements.push(format!(
                        "{}: recorded as a divergence, but OpenSSL refused this path as well \
                         ({:?} at depth {:?}), so the two do not actually disagree",
                        case.name, case.openssl_error, case.openssl_depth
                    ));
                }
            }
            None => {
                if ours.is_ok() != case.openssl_ok {
                    disagreements.push(format!(
                        "{}: this crate says {ours:?}, OpenSSL says ok={} (error {:?} at depth \
                         {:?})",
                        case.name, case.openssl_ok, case.openssl_error, case.openssl_depth
                    ));
                }
            }
        }
    }
    assert!(
        disagreements.is_empty(),
        "{} of {} paths disagree:\n{}",
        disagreements.len(),
        VERIFY_CASES.len(),
        disagreements.join("\n")
    );

    // A table that only asked about paths everyone accepts would prove nothing, so both
    // answers have to be in it, and the refusals have to include ones that come from a
    // *rule* rather than from an issuer simply being missing.
    assert!(
        VERIFY_CASES.iter().any(|case| case.openssl_ok),
        "the table needs at least one path OpenSSL accepts"
    );
    assert!(
        VERIFY_CASES.iter().filter(|case| !case.openssl_ok).count() >= 5,
        "the table needs a real set of refusals, not one"
    );
    assert!(
        VERIFY_CASES
            .iter()
            // 25 is "path length constraint exceeded" and 79 "invalid CA certificate":
            // the two OpenSSL codes that mean a structural rule refused the path, which is
            // the half of the policy this crate is actually in the business of deciding.
            .any(|case| matches!(case.openssl_error, Some(25 | 79))),
        "the table needs a refusal by a structural rule, not only by a missing issuer"
    );
}

#[test]
fn a_certificate_that_does_not_parse_stops_the_chain() {
    // Through the whole wire path: framing, then parsing, then the chain. The
    // fixture is the one with an unknown **critical** extension, which RFC 5280
    // §4.2 requires a reader to treat as fatal, and it fails before any structure
    // is considered.
    let list = certificate_list(&[fixture("LEAF_UNKNOWN_CRITICAL")]);
    let anchor_ders = ders_for(&["ROOT"]);
    let anchors = certificates_of(&anchor_ders);

    match verify_certificate_list(&list, &anchors, UnixTime::from_secs(0), &Policy::default()) {
        Err(ChainError::Certificate { index, error }) => {
            assert_eq!(index, 0);
            assert!(
                matches!(error, X509Error::UnsupportedCriticalExtension(..)),
                "the fixture's defect is the critical extension, got {error:?}"
            );
        }
        other => panic!("expected a parse failure, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The leaf's own claims: which server it is for, and what it may be used for
// ---------------------------------------------------------------------------

#[test]
fn the_leaves_names_match_the_server_and_nothing_else() {
    let der = fixture("LEAF_RSA");
    let certificate = Certificate::parse(&der).expect("the leaf parses");

    // Its four `subjectAltName` choices include two `dNSName`s, one of which is a
    // wildcard: `models.example` and `*.cdn.example` (the other two are an
    // `rfc822Name` and a URI, which `the_leafs_subject_alt_name_keeps_every_choice_...`
    // already asserts are kept but are not names).
    assert_eq!(
        verify_server_name(&certificate, Some(b"models.example")),
        Ok(())
    );
    // Case, and one trailing dot, are the only normalisation.
    assert_eq!(
        verify_server_name(&certificate, Some(b"MODELS.Example")),
        Ok(())
    );
    assert_eq!(
        verify_server_name(&certificate, Some(b"models.example.")),
        Ok(())
    );
    // The wildcard covers exactly one whole label.
    assert_eq!(
        verify_server_name(&certificate, Some(b"a.cdn.example")),
        Ok(())
    );
    assert_eq!(
        verify_server_name(&certificate, Some(b"A.CDN.EXAMPLE")),
        Ok(())
    );

    // And the refusals: a different name, the wildcard's own suffix as a bare
    // domain, and one label too many.
    for name in [
        &b"evil.example"[..],
        &b"cdn.example"[..],
        &b"a.b.cdn.example"[..],
        &b"models.example.evil"[..],
        &b"xmodels.example"[..],
    ] {
        assert_eq!(
            verify_server_name(&certificate, Some(name)),
            Err(LeafError::NoNameMatch {
                server_name: name.to_vec()
            }),
            "{name:?} is not one of the leaf's names"
        );
    }
}

#[test]
fn a_common_name_is_not_a_name() {
    // `LEAF_EC` has a `commonName` of `ec.example` and **no** `subjectAltName` at
    // all. The subject's own name is not a constraint on the certificate, and a
    // matcher that fell back to it would accept this — so it must refuse, and say
    // that there was nothing to match rather than that the name was wrong.
    let der = fixture("LEAF_EC");
    let certificate = Certificate::parse(&der).expect("the fixture parses");
    assert_eq!(
        certificate.subject().common_name(),
        Ok(Some(&b"ec.example"[..])),
        "the certificate does have that commonName"
    );
    assert!(certificate.subject_alt_names().is_empty());

    for name in [&b"ec.example"[..], &b"models.example"[..]] {
        assert_eq!(
            verify_server_name(&certificate, Some(name)),
            Err(LeafError::NoDnsNames)
        );
    }
}

#[test]
fn a_server_name_that_cannot_be_compared_is_refused() {
    let der = fixture("LEAF_RSA");
    let certificate = Certificate::parse(&der).expect("the leaf parses");

    // No name offered at all: a connection with no name is a connection whose
    // certificate cannot be tied to a server. `CertificateVerifyInput` carries it as
    // an `Option`, and `None` is not a pass.
    assert_eq!(
        verify_server_name(&certificate, None),
        Err(LeafError::NoServerName)
    );
    // An address is not matched against names: `dNSName` cannot be an address, so
    // this is refused rather than compared.
    for name in [&b"192.0.2.1"[..], &b"::1"[..], &b"2001:db8::1"[..]] {
        assert_eq!(
            verify_server_name(&certificate, Some(name)),
            Err(LeafError::IpLiteral {
                server_name: name.to_vec()
            }),
            "{name:?}"
        );
    }
    // And names that are not host names at all, each of which is a caller bug rather
    // than a certificate problem: a NUL, a slash, a port, whitespace, an IDN in its
    // non-ASCII form, and the empty name.
    for name in [
        &b"models.example\0.evil"[..],
        &b"models/../example"[..],
        &b"models.example:443"[..],
        &b" models.example"[..],
        &b"*.example"[..],
        &b""[..],
        // `mödels.example` in UTF-8: an IDN must reach here as an A-label.
        &b"m\xc3\xb6dels.example"[..],
    ] {
        assert_eq!(
            verify_server_name(&certificate, Some(name)),
            Err(LeafError::MalformedServerName {
                server_name: name.to_vec()
            }),
            "{name:?}"
        );
    }
}

#[test]
fn the_purpose_checks_are_what_the_certificate_says() {
    // A server certificate: `keyUsage` with `digitalSignature` and an
    // `extendedKeyUsage` naming `serverAuth`.
    let leaf = fixture("LEAF_RSA");
    let leaf = Certificate::parse(&leaf).expect("the leaf parses");
    assert_eq!(verify_server_auth(&leaf), Ok(()));

    // `LEAF_EC` has neither extension, and RFC 5280 §4.2.1.12 makes an absent
    // `extKeyUsage` mean any purpose is acceptable. This is about the leaf alone;
    // its *chain* has no valid instant, which the chain tests assert separately.
    let ec = fixture("LEAF_EC");
    let ec = Certificate::parse(&ec).expect("the fixture parses");
    assert!(ec.extended_key_usage().is_empty() && ec.key_usage().is_none());
    assert_eq!(verify_server_auth(&ec), Ok(()));

    // The negative that needed a fixture: a perfectly good chain whose purpose is
    // `clientAuth` alone.
    let client = fixture("LEAF_CLIENT_AUTH");
    let client = Certificate::parse(&client).expect("the fixture parses");
    match verify_server_auth(&client) {
        Err(LeafError::MissingServerAuth { purposes }) => {
            assert_eq!(purposes.len(), 1);
            assert_eq!(purposes[0], CLIENT_AUTH_OID.to_vec());
        }
        other => panic!("expected a purpose refusal, got {other:?}"),
    }

    // And the other one: `keyUsage` that allows encipherment but not signing, which
    // TLS 1.3 has no use for at all.
    let no_sign = fixture("LEAF_NO_DIGITAL_SIGNATURE");
    let no_sign = Certificate::parse(&no_sign).expect("the fixture parses");
    assert_eq!(
        verify_server_auth(&no_sign),
        Err(LeafError::NoDigitalSignature {
            key_usage: KeyUsage::KEY_ENCIPHERMENT
        })
    );
}

#[test]
fn the_leaf_check_is_both_halves_in_one_call() {
    let leaf = fixture("LEAF_RSA");
    let leaf = Certificate::parse(&leaf).expect("the leaf parses");
    assert_eq!(verify_leaf(&leaf, Some(b"models.example")), Ok(()));
    // The name is wrong on that one, and the purpose on the other: each refusal is
    // the one that belongs to the fixture.
    assert_eq!(
        verify_leaf(&leaf, Some(b"evil.example")),
        Err(LeafError::NoNameMatch {
            server_name: b"evil.example".to_vec()
        })
    );
    let client = fixture("LEAF_CLIENT_AUTH");
    let client = Certificate::parse(&client).expect("the fixture parses");
    // Its name *does* match, so a purpose failure here is the purpose alone.
    assert_eq!(verify_server_name(&client, Some(b"client.example")), Ok(()));
    assert!(matches!(
        verify_leaf(&client, Some(b"client.example")),
        Err(LeafError::MissingServerAuth { .. })
    ));
}

#[test]
fn a_name_match_alone_does_not_anchor_a_chain() {
    // `ROOT_SHADOW` is a self-signed CA carrying the *root's subject name* with a
    // key of its own, which is the only way to make this checkable. The chain's top
    // certificate was signed by the root, so an anchor matched by name alone would
    // accept it; the key check is what refuses it. This is the case the module docs
    // call out, and the one a name-only matcher would have got wrong.
    let chain_ders = [fixture("LEAF_RSA"), fixture("INTERMEDIATE")];
    let shadow_ders = [fixture("ROOT_SHADOW")];
    let chain = certificates_of(&chain_ders);
    let shadow = certificates_of(&shadow_ders);
    let root_ders = ders_for(&["ROOT"]);
    let root = certificates_of(&root_ders);

    // The names really are the same and the keys really are not, so the refusal
    // below cannot be about the name.
    assert_eq!(shadow[0].subject().as_der(), root[0].subject().as_der());
    assert_ne!(shadow[0].public_key(), root[0].public_key());

    let time = chain[0].not_before();
    let refused = verify_chain(&chain, &shadow, time, &Policy::default());
    assert!(
        matches!(
            refused,
            Err(ChainError::Signature {
                certificate: Position::Chain(1),
                ..
            })
        ),
        "the anchor's name matches and its key does not: {refused:?}"
    );

    // A candidate that fails does not end the search: with both anchors present the
    // real one is found, in the order they were given.
    let both_ders = ders_for(&["ROOT_SHADOW", "ROOT"]);
    let both = certificates_of(&both_ders);
    assert_eq!(
        verify_chain(&chain, &both, time, &Policy::default()),
        Ok(())
    );
}

#[test]
fn the_wire_path_frames_before_it_verifies_anything() {
    let leaf = fixture("LEAF_RSA");
    let intermediate = fixture("INTERMEDIATE");
    let (leaf_len, intermediate_len) = (leaf.len(), intermediate.len());
    let anchor_ders = [fixture("ROOT")];
    let anchors = certificates_of(&anchor_ders);
    let leaf_ders = [leaf.clone()];
    let time = certificates_of(&leaf_ders)[0].not_before();
    let policy = Policy::default();

    // A list that frames correctly verifies, so the negatives below are about the
    // framing rather than about a chain that was wrong anyway.
    let good_list = certificate_list(&[leaf.clone(), intermediate]);
    assert_eq!(
        verify_certificate_list(&good_list, &anchors, time, &policy),
        Ok(())
    );

    // Nothing at all is not a chain.
    assert_eq!(
        verify_certificate_list(&[], &anchors, time, &policy),
        Err(ChainError::EmptyList)
    );
    // Dropping the list's final byte truncates the *last entry's body* — the
    // prefix is intact and still claims the full length — so the refusal names
    // that entry's offset rather than the tail.
    let short_body = &good_list[..good_list.len() - 1];
    assert_eq!(
        verify_certificate_list(short_body, &anchors, time, &policy),
        Err(ChainError::TruncatedEntry {
            offset: 3 + leaf_len,
            declared: intermediate_len,
            remaining: intermediate_len - 1,
        })
    );
    // Cutting two bytes past the first entry leaves too little for the second
    // entry's length prefix, which is the other failure: the list ends *inside* a
    // length.
    let inside_prefix = &good_list[..3 + leaf_len + 2];
    assert_eq!(
        verify_certificate_list(inside_prefix, &anchors, time, &policy),
        Err(ChainError::EndsInsideLength { remaining: 2 })
    );
    // An entry that declares more bytes than are there, as the single entry of a
    // one-certificate list.
    let one_leaf = std::slice::from_ref(&leaf);
    let mut truncated_entry = certificate_list(one_leaf);
    truncated_entry.pop();
    assert_eq!(
        verify_certificate_list(&truncated_entry, &anchors, time, &policy),
        Err(ChainError::TruncatedEntry {
            offset: 0,
            declared: leaf_len,
            remaining: leaf_len - 1,
        })
    );
    // An entry that declares a length of zero, which RFC 8446's
    // `certificate<1..2^24-1>` does not allow.
    let zero_length = [0x00, 0x00, 0x00];
    assert_eq!(
        verify_certificate_list(&zero_length, &anchors, time, &policy),
        Err(ChainError::ZeroLengthEntry { offset: 0 })
    );
    // One good entry followed by a bad one, so the offset in the refusal is not
    // the zero that a bug could report for everything.
    let mut second_bad = certificate_list(one_leaf);
    second_bad.extend_from_slice(&[0x00, 0x00, 0x09, 0x01]);
    assert_eq!(
        verify_certificate_list(&second_bad, &anchors, time, &policy),
        Err(ChainError::TruncatedEntry {
            offset: leaf_len + 3,
            declared: 9,
            remaining: 1,
        })
    );
    // And a list whose entry frames but is not a certificate at all: the parse is
    // what refuses it.
    let not_a_certificate = certificate_list(&[vec![0x30, 0x03, 0x01, 0x01, 0x00]]);
    assert!(matches!(
        verify_certificate_list(&not_a_certificate, &anchors, time, &policy),
        Err(ChainError::Certificate { index: 0, .. })
    ));
}
