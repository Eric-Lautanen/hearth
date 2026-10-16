#!/usr/bin/env python3
"""Generate the X.509 fixtures for `crates/hearth-tls/tests/x509_fixtures.rs`.

The expectations printed here come from OpenSSL through the `cryptography`
package's own X.509 parser — subject, issuer, validity, key, basicConstraints,
keyUsage, subjectAltName and the signature algorithm — so the test that consumes
them is a differential test against a real implementation rather than a snapshot
of our own output.

The keys are derived from a fixed seed with `sympy` primes, so the *keys* are the
same on every run and four of the six fixtures regenerate byte for byte (their RSA
signatures are deterministic). Two do not: `SELF_SIGNED_PSS` is RSA-PSS, whose salt
is random, and `SELF_SIGNED_EC` is ECDSA, whose nonce is random — so their signature
bytes differ on every run, and the ECDSA signature's *length* can differ too, which
moves that certificate's whole DER length.

**Every fixture has its own key, and that is load-bearing.** Two certificates that
share a key cannot tell a verifier from a key lookup: the PSS fixture once reused
the root's key, and as a result "a certificate verified with the wrong identity
fails" was not a check the fixture set could make — it passed for the wrong reason.
Seeds are allocated one per identity for that reason; do not alias one key into two
certificates.

**`ROOT_SHADOW` exists to make one specific check possible.** It is a self-signed
CA with the *same subject name* as `ROOT` and a different key, which is the only
way to test that a chain builder anchors on a key rather than on a name: a chain
whose top certificate was signed by `ROOT` must be refused when `ROOT_SHADOW` is
the only anchor, because the name matches and the signature does not. Without it,
that path — the one a name-only matcher would accept — has no negative test.

Every field the test asserts is read out of the certificate, so a regenerated file
is equivalent, not just similar: regenerating and swapping the file in was measured
to keep all twelve fixture tests green, including the ECDSA one whose `der_len`
changed. The copy in `crates/hearth-tls/tests/data/` is the reviewed pin; regenerate
only deliberately.

Nothing here is a secret: these keys exist to sign test fixtures and must never be
used for anything else.

Requires `pip install cryptography sympy`. Run from the repository root:

    python scripts/tls_x509_fixtures.py
"""
import datetime
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile

import sympy
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateNumbers,
    RSAPublicNumbers,
)
from cryptography.x509.oid import NameOID, ObjectIdentifier

UTC = datetime.timezone.utc


def rsa_key(seed, bits=2048):
    """A reproducible RSA key: two `sympy` primes from a seeded RNG."""
    rng = random.Random(seed)
    half = bits // 2
    for _ in range(200):
        p = int(sympy.nextprime(rng.getrandbits(half) | (1 << (half - 1)) | 1))
        q = int(sympy.nextprime(rng.getrandbits(half) | (1 << (half - 1)) | 1))
        if p == q or (p * q).bit_length() != bits:
            continue
        e = 65537
        d = pow(e, -1, int(sympy.lcm(p - 1, q - 1)))
        return RSAPrivateNumbers(
            p=p,
            q=q,
            d=d,
            dmp1=d % (p - 1),
            dmq1=d % (q - 1),
            iqmp=pow(q, -1, p),
            public_numbers=RSAPublicNumbers(e, p * q),
        ).private_key()
    raise AssertionError("no %d-bit key from seed %d" % (bits, seed))


def ec_key(seed):
    """A reproducible P-256 key from a seeded scalar."""
    rng = random.Random(seed)
    order = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
    return ec.derive_private_key(rng.randrange(1, order), ec.SECP256R1())


def der_hex(data, indent="    "):
    text = data.hex()
    lines = [text[i : i + 32] for i in range(0, len(text), 32)]
    return "\\\n" + "".join("%s%s \\\n" % (indent, line) for line in lines[:-1]) + indent + lines[-1]


def read_tlv(buf, index):
    """`(tag, body_start, body_len, next_index)` for the TLV at `index`."""
    tag = buf[index]
    index += 1
    first = buf[index]
    if first < 0x80:
        size = first
        index += 1
    else:
        count = first & 0x7F
        size = int.from_bytes(buf[index + 1 : index + 1 + count], "big")
        index += 1 + count
    return tag, index, size, index + size


def signature_length(der):
    """The byte length of the certificate's `signatureValue`, read from the DER.

    This is not a number the test could derive from the fixture set: an RSA signature
    is the *issuer's* modulus length, and one fixture here is signed by a 1024-bit key,
    so "256" would be an assumption about the set rather than about the format. The
    one octet of a `BIT STRING` that names its unused bits is not part of the value.
    """
    tag, start, size, _ = read_tlv(der, 0)
    assert tag == 0x30, "a certificate is a SEQUENCE"
    body = der[start : start + size]
    _, _, _, after_tbs = read_tlv(body, 0)
    _, _, _, after_algorithm = read_tlv(body, after_tbs)
    tag, _, size, _ = read_tlv(body, after_algorithm)
    assert tag == 0x03, "the signature is a BIT STRING"
    return size - 1


def openssl_version():
    """The OpenSSL CLI's version line, recorded beside the verdicts it produced."""
    if shutil.which("openssl") is None:
        raise SystemExit(
            "openssl is not on PATH. The path table below records verdicts taken from\n"
            "the OpenSSL CLI; the committed data file is the pin, and it stays valid\n"
            "without the tool. Install OpenSSL only to regenerate deliberately."
        )
    return subprocess.run(
        ["openssl", "version"], capture_output=True, text=True, check=True
    ).stdout.strip()


def openssl_verdict(by_label, chain, anchors, attime):
    """`(ok, error, depth)` from the OpenSSL CLI for one path.

    Three flags are deliberate. `-attime` puts both implementations at the instant the
    test uses, because these fixtures were built around particular dates on purpose.
    `-partial_chain` is the closest equivalent to this crate's rule that an anchor need
    not be self-signed: pinning an intermediate is explicit here and needs a flag there.
    `-auth_level 0` keeps OpenSSL's separate security-level policy out of the
    comparison — this crate has a key-size floor of its own, and that difference is
    recorded as a divergence rather than buried in a flag.

    `-untrusted` is omitted when there is nothing between the leaf and the anchor:
    passing an empty file makes openssl refuse to run at all, with a message that looks
    like a verdict and is not one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ca = os.path.join(tmp, "ca.pem")
        leaf = os.path.join(tmp, "leaf.pem")
        write_pem(by_label, ca, anchors)
        write_pem(by_label, leaf, [chain[0]])
        command = [
            "openssl", "verify",
            "-CAfile", ca,
            "-partial_chain",
            "-attime", str(attime),
            "-auth_level", "0",
        ]
        if chain[1:]:
            untrusted = os.path.join(tmp, "chain.pem")
            write_pem(by_label, untrusted, chain[1:])
            command += ["-untrusted", untrusted]
        result = subprocess.run(command + [leaf], capture_output=True, text=True)

    found = re.search(r"error (\d+) at (\d+) depth lookup", result.stdout + result.stderr)
    return (
        result.returncode == 0,
        int(found.group(1)) if found else None,
        int(found.group(2)) if found else None,
    )


def write_pem(by_label, path, labels):
    with open(path, "wb") as handle:
        for label in labels:
            handle.write(by_label[label].public_bytes(serialization.Encoding.PEM))


def verify_cases():
    """The paths both implementations are asked about.

    `(name, chain, anchors, instant-from, note)`, where `chain[0]` is the leaf and the
    instant comes from that certificate's own `notBefore` — the same instant the tests
    use, since the fixtures were built around particular dates deliberately.

    `note` is set only where the two are *expected* to disagree, and every such case has
    to be one where OpenSSL accepts and this crate refuses: being the more permissive
    side is not an excusable disagreement, and the test enforces that.
    """
    return [
        ("leaf-and-intermediate-to-root", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], "LEAF_RSA", None),
        ("intermediate-pinned-as-the-anchor", ["LEAF_SUB", "SUB_CA"], ["TOP_CA"], "LEAF_SUB", None),
        ("path-deeper-than-the-root-allows", ["LEAF_SUB", "SUB_CA", "TOP_CA"], ["ROOT"], "LEAF_SUB", None),
        ("issuer-without-keyCertSign", ["LEAF_OF_NO_SIGN_CA", "CA_NO_KEY_CERT_SIGN"], ["ROOT"], "LEAF_OF_NO_SIGN_CA", None),
        ("anchor-stating-ca-false", ["LEAF_OF_NON_CA_SIGNER"], ["NON_CA_SIGNER"], "LEAF_OF_NON_CA_SIGNER", None),
        ("ca-false-certificate-as-a-link", ["LEAF_OF_NON_CA_SIGNER", "NON_CA_SIGNER"], ["ROOT"], "LEAF_OF_NON_CA_SIGNER", None),
        ("self-signed-ca-false-pinned-as-itself", ["SELF_SIGNED_PSS"], ["SELF_SIGNED_PSS"], "SELF_SIGNED_PSS", None),
        ("no-basic-constraints-pinned-as-itself", ["SELF_SIGNED_EC"], ["SELF_SIGNED_EC"], "SELF_SIGNED_EC", None),
        ("root-the-caller-does-not-trust", ["LEAF_RSA", "INTERMEDIATE"], ["SELF_SIGNED_EC"], "LEAF_RSA", None),
        ("anchor-whose-window-closed", ["LEAF_OF_EXPIRED_ANCHOR"], ["EXPIRED_ANCHOR"], "LEAF_OF_EXPIRED_ANCHOR", None),
        ("issuer-expired-under-a-valid-leaf", ["LEAF_EC", "INTERMEDIATE"], ["ROOT"], "LEAF_EC", None),
        ("broken-link-missing-intermediate", ["LEAF_RSA", "ROOT"], ["ROOT"], "LEAF_RSA", None),
        ("strict-sibling-CA-as-the-anchor", ["LEAF_SUB_STRICT", "SUB_CA_STRICT"], ["TOP_CA_STRICT"], "LEAF_SUB_STRICT", None),
        (
            "ca-certificate-as-the-server",
            ["INTERMEDIATE"],
            ["ROOT"],
            "INTERMEDIATE",
            "this crate refuses a CA certificate as the server's own (CaUsedAsLeaf); "
            "OpenSSL's default purpose does not, and a TLS server certificate may not be a CA",
        ),
        (
            "anchor-below-the-key-size-floor",
            ["LEAF_OF_WEAK_ANCHOR"],
            ["WEAK_ANCHOR"],
            "LEAF_OF_WEAK_ANCHOR",
            "this crate's Policy::min_rsa_bits is 2048; OpenSSL at -auth_level 0 has no floor",
        ),
    ]


def key_usage_mask(usage):
    """The DER bit numbers: 0 digitalSignature .. 8 decipherOnly, bit 0 first."""
    names = [
        "digital_signature",
        "content_commitment",
        "key_encipherment",
        "data_encipherment",
        "key_agreement",
        "key_cert_sign",
        "crl_sign",
        "encipher_only",
        "decipher_only",
    ]
    mask = 0
    for index, name in enumerate(names):
        # Bits 7 and 8 are defined only when keyAgreement is set; OpenSSL raises
        # rather than reporting them otherwise, and so would any reader that
        # invented a value for them.
        if index >= 7 and not usage.key_agreement:
            continue
        if getattr(usage, name):
            mask |= 1 << index
    return mask


def expectations(cert):
    """Everything the Rust test asserts, read out of OpenSSL's parser."""
    public = cert.public_key()
    if isinstance(public, rsa.RSAPublicKey):
        numbers = public.public_numbers()
        # The key's own byte length, not a fixed 256: one fixture's anchor holds a
        # 1024-bit key on purpose, and padding its modulus to 2048 bits would compare
        # it against bytes that are not in the certificate.
        width = (numbers.n.bit_length() + 7) // 8
        key = "Key::Rsa { modulus: \"%s\", exponent: %d }" % (
            numbers.n.to_bytes(width, "big").hex(),
            numbers.e,
        )
    else:
        point = public.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        key = "Key::Ec { point: \"%s\" }" % point.hex()

    names = []
    try:
        extension = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        names = extension.value.get_values_for_type(x509.DNSName)
    except x509.ExtensionNotFound:
        pass

    usage = None
    try:
        usage = cert.extensions.get_extension_for_class(x509.KeyUsage).value
    except x509.ExtensionNotFound:
        pass

    ca = False
    path_len = None
    has_basic_constraints = False
    for extension in cert.extensions:
        if isinstance(extension.value, x509.BasicConstraints):
            ca = extension.value.ca
            path_len = extension.value.path_length
            has_basic_constraints = True

    cn = [a.value for a in cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)]
    issuer_cn = [a.value for a in cert.issuer.get_attributes_for_oid(NameOID.COMMON_NAME)]
    return {
        "subject_cn": cn[0] if cn else "",
        "issuer_cn": issuer_cn[0] if issuer_cn else "",
        "serial": "%02x" % cert.serial_number,
        "not_before": int(cert.not_valid_before_utc.timestamp()),
        "not_after": int(cert.not_valid_after_utc.timestamp()),
        "is_ca": ca,
        "has_basic_constraints": has_basic_constraints,
        "path_len": path_len,
        "key_usage": key_usage_mask(usage) if usage else None,
        "san_dns": names,
        "signature_algorithm": cert.signature_algorithm_oid.dotted_string,
        "key": key,
    }


def main():
    root_key = rsa_key(20261016)
    intermediate_key = rsa_key(20261017)
    leaf_key = rsa_key(20261018)
    ec_leaf_key = ec_key(20261019)
    # Its own key, not the root's: see the note in the module docstring. A
    # self-signed certificate sharing another fixture's key makes the
    # "right scheme, wrong identity" check untestable.
    pss_key = rsa_key(20261020)
    # A second CA with the *same subject name* as the root and its own key: see the
    # note in the module docstring. Nothing is issued by it, which is the point —
    # it exists to be an anchor whose name matches and whose key does not.
    shadow_root_key = rsa_key(20261021)
    client_auth_key = rsa_key(20261022)
    no_sign_key = rsa_key(20261023)
    sub_ca_key = rsa_key(20261024)
    top_ca_key = rsa_key(20261025)
    top_ca_strict_key = rsa_key(20261026)
    sub_ca_strict_key = rsa_key(20261030)
    leaf_sub_strict_key = rsa_key(20261031)
    leaf_sub_key = rsa_key(20261027)
    no_key_cert_sign_key = rsa_key(20261028)
    leaf_of_no_sign_key = rsa_key(20261029)
    expired_anchor_key = rsa_key(20261101)
    weak_anchor_key = rsa_key(20261102, bits=1024)
    leaf_of_expired_key = rsa_key(20261103)
    leaf_of_weak_key = rsa_key(20261104)
    non_ca_signer_key = rsa_key(20261201)
    leaf_of_non_ca_key = rsa_key(20261202)

    def usage(**flags):
        """A `KeyUsage` with only the named bits set."""
        values = dict(
            digital_signature=False,
            content_commitment=False,
            key_encipherment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False,
        )
        values.update(flags)
        return x509.KeyUsage(**values)

    def name(common):
        return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common)])

    root = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Root CA"))
        .issuer_name(name("Hearth Test Root CA"))
        .public_key(root_key.public_key())
        .serial_number(0x01)
        .not_valid_before(datetime.datetime(2020, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2040, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(root_key, hashes.SHA256())
    )

    intermediate = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Intermediate CA"))
        .issuer_name(root.subject)
        .public_key(intermediate_key.public_key())
        .serial_number(0x02)
        .not_valid_before(datetime.datetime(2021, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2039, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(root_key, hashes.SHA256())
    )

    # A leaf with a serial whose DER needs a leading zero octet (0x00FF is
    # positive in two octets and negative in one), a mixed subjectAltName, an
    # unknown extension, and its whole validity in UTCTime.
    leaf_rsa = (
        x509.CertificateBuilder()
        .subject_name(name("models.example"))
        .issuer_name(intermediate.subject)
        .public_key(leaf_key.public_key())
        .serial_number(0xFF)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("models.example"),
                    x509.DNSName("*.cdn.example"),
                    x509.RFC822Name("admin@models.example"),
                    x509.UniformResourceIdentifier("https://models.example/"),
                ]
            ),
            critical=False,
        )
        .add_extension(
            # An unknown, non-critical extension: a reader must skip it rather than
            # fail, and must not let it change anything.
            x509.UnrecognizedExtension(ObjectIdentifier("1.3.6.1.4.1.99999.1"), b"\x04\x03abc"),
            critical=False,
        )
        .sign(intermediate_key, hashes.SHA256())
    )

    # P-256, ecdsa-with-SHA256, no subjectAltName at all, and dates past 2050 so
    # the validity is GeneralizedTime rather than UTCTime.
    leaf_ec = (
        x509.CertificateBuilder()
        .subject_name(name("ec.example"))
        .issuer_name(intermediate.subject)
        .public_key(ec_leaf_key.public_key())
        .serial_number(0x03)
        .not_valid_before(datetime.datetime(2050, 6, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2060, 6, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(intermediate_key, hashes.SHA256())
    )

    # The same subject name as the root, a different key, and self-signed: an
    # anchor that matches the root by name and not by key.
    root_shadow = (
        x509.CertificateBuilder()
        .subject_name(root.subject)
        .issuer_name(root.subject)
        .public_key(shadow_root_key.public_key())
        .serial_number(0x07)
        .not_valid_before(datetime.datetime(2020, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2040, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(shadow_root_key, hashes.SHA256())
    )

    # A self-signed RSA certificate whose *signature* is RSASSA-PSS, so the
    # `AlgorithmIdentifier` carries hash, MGF and salt parameters rather than
    # being an OID with no parameters at all.
    self_signed_pss = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test PSS Signer"))
        .issuer_name(name("Hearth Test PSS Signer"))
        .public_key(pss_key.public_key())
        .serial_number(0x04)
        .not_valid_before(datetime.datetime(2026, 3, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2036, 3, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(
            pss_key,
            hashes.SHA256(),
            rsa_padding=padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
        )
    )

    # A self-signed P-256 certificate with no extensions at all: the signature is
    # ecdsa-with-SHA256, and there is no basicConstraints to read.
    self_signed_ec = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test EC Signer"))
        .issuer_name(name("Hearth Test EC Signer"))
        .public_key(ec_leaf_key.public_key())
        .serial_number(0x05)
        .not_valid_before(datetime.datetime(2026, 3, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2036, 3, 1, tzinfo=UTC))
        .sign(ec_leaf_key, hashes.SHA256())
    )

    # A leaf the intermediate issued for **client** authentication only. The chain is
    # perfectly good and the name is right; the purpose is what makes it not a server
    # certificate, and the check that refuses it needs exactly this certificate.
    leaf_client_auth = (
        x509.CertificateBuilder()
        .subject_name(name("client.example"))
        .issuer_name(intermediate.subject)
        .public_key(client_auth_key.public_key())
        .serial_number(0x08)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("client.example")]), critical=False
        )
        .sign(intermediate_key, hashes.SHA256())
    )

    # A leaf whose `keyUsage` allows encipherment but not signing. TLS 1.3 removed key
    # transport entirely, so a certificate like this can never complete a handshake —
    # and the rule that refuses it is only provable with a certificate that carries it.
    leaf_no_sign = (
        x509.CertificateBuilder()
        .subject_name(name("no-sign.example"))
        .issuer_name(intermediate.subject)
        .public_key(no_sign_key.public_key())
        .serial_number(0x09)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("no-sign.example")]), critical=False
        )
        .sign(intermediate_key, hashes.SHA256())
    )

    # A two-level sub-hierarchy, for `pathLenConstraint`. The root allows one
    # intermediate below it; `TOP_CA` allows one more, and `TOP_CA_STRICT` allows
    # none — the same subject name and a different key, so the two can be swapped
    # into one chain and only the limit differs. `SUB_CA` allows none either, so a
    # path through it is legal only while nothing else is below it.
    top_ca = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Top CA"))
        .issuer_name(root.subject)
        .public_key(top_ca_key.public_key())
        .serial_number(0x0B)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
        .sign(root_key, hashes.SHA256())
    )

    # Issued *by* `TOP_CA`, sharing its subject name and carrying a key of its own.
    # That is what lets a test put this one or the real one third in the same chain
    # and have the only difference be the limit: the names link and the signatures
    # verify either way, because `TOP_CA` signed both `SUB_CA` and this one.
    top_ca_strict = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Top CA"))
        .issuer_name(top_ca.subject)
        .public_key(top_ca_strict_key.public_key())
        .serial_number(0x0C)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
        .sign(top_ca_key, hashes.SHA256())
    )

    sub_ca = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Sub CA"))
        .issuer_name(top_ca.subject)
        .public_key(sub_ca_key.public_key())
        .serial_number(0x0A)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
        .sign(top_ca_key, hashes.SHA256())
    )

    leaf_sub = (
        x509.CertificateBuilder()
        .subject_name(name("sub.example"))
        .issuer_name(sub_ca.subject)
        .public_key(leaf_sub_key.public_key())
        .serial_number(0x0D)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("sub.example")]), critical=False)
        .sign(sub_ca_key, hashes.SHA256())
    )

    # The same two levels again under `TOP_CA_STRICT`, with one level's limit changed
    # and nothing else: where the tree above has a CA allowing one intermediate below
    # it, this one has a CA allowing none. Both subtrees are otherwise identical, so a
    # chain built from one and then the other differs only in which limit applies.
    sub_ca_strict = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Sub CA"))
        .issuer_name(top_ca_strict.subject)
        .public_key(sub_ca_strict_key.public_key())
        .serial_number(0x10)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
        .sign(top_ca_strict_key, hashes.SHA256())
    )

    leaf_sub_strict = (
        x509.CertificateBuilder()
        .subject_name(name("sub-strict.example"))
        .issuer_name(sub_ca_strict.subject)
        .public_key(leaf_sub_strict_key.public_key())
        .serial_number(0x11)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("sub-strict.example")]), critical=False
        )
        .sign(sub_ca_strict_key, hashes.SHA256())
    )

    # A CA that says it is one and then does not claim the right to sign
    # certificates: `basicConstraints` CA is true, `keyUsage` is present without
    # `keyCertSign`. RFC 5280 §4.2.1.9 makes that fatal to the path, and only a
    # certificate like this can prove the check.
    no_key_cert_sign_ca = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test No-Sign CA"))
        .issuer_name(root.subject)
        .public_key(no_key_cert_sign_key.public_key())
        .serial_number(0x0E)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(digital_signature=True, crl_sign=True), critical=True)
        .sign(root_key, hashes.SHA256())
    )

    leaf_of_no_sign = (
        x509.CertificateBuilder()
        .subject_name(name("no-sign-ca.example"))
        .issuer_name(no_key_cert_sign_ca.subject)
        .public_key(leaf_of_no_sign_key.public_key())
        .serial_number(0x0F)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("no-sign-ca.example")]), critical=False
        )
        .sign(no_key_cert_sign_key, hashes.SHA256())
    )

    # Two anchors whose *own* properties are the only thing wrong with the path, each
    # with a leaf it really did sign — the anchor's window and its key size are the last
    # two call sites no other fixture can reach.
    #
    # The first stopped being valid in 2020 while the leaf it signed is valid from 2026,
    # so the leaf is inside its own window at the instant of the check and the issuer is
    # not. The second is a real CA with a real 1024-bit key: nothing about it is
    # malformed, it is simply below the policy's floor, which is the case `rsa.rs`
    # deliberately accepts and `Policy::min_rsa_bits` refuses.
    expired_anchor = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Expired Anchor"))
        .issuer_name(name("Hearth Test Expired Anchor"))
        .public_key(expired_anchor_key.public_key())
        .serial_number(0x20)
        .not_valid_before(datetime.datetime(2015, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2020, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
        .sign(expired_anchor_key, hashes.SHA256())
    )

    leaf_of_expired_anchor = (
        x509.CertificateBuilder()
        .subject_name(name("expired-anchor.example"))
        .issuer_name(expired_anchor.subject)
        .public_key(leaf_of_expired_key.public_key())
        .serial_number(0x21)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("expired-anchor.example")]), critical=False
        )
        .sign(expired_anchor_key, hashes.SHA256())
    )

    weak_anchor = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Weak Anchor"))
        .issuer_name(name("Hearth Test Weak Anchor"))
        .public_key(weak_anchor_key.public_key())
        .serial_number(0x22)
        .not_valid_before(datetime.datetime(2020, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2040, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
        .sign(weak_anchor_key, hashes.SHA256())
    )

    leaf_of_weak_anchor = (
        x509.CertificateBuilder()
        .subject_name(name("weak-anchor.example"))
        .issuer_name(weak_anchor.subject)
        .public_key(leaf_of_weak_key.public_key())
        .serial_number(0x23)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("weak-anchor.example")]), critical=False
        )
        .sign(weak_anchor_key, hashes.SHA256())
    )

    # A certificate whose issuer marked it **not** a CA, which nevertheless claims
    # `keyCertSign` and signs a certificate of its own. As an anchor it is refused for
    # that contradiction; in a chain it is refused as an issuer. `SELF_SIGNED_EC`, which
    # carries no extensions at all, is the control in the other direction: a pin with no
    # `basicConstraints` to contradict it is accepted.
    non_ca_signer = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Non-CA Signer"))
        .issuer_name(root.subject)
        .public_key(non_ca_signer_key.public_key())
        .serial_number(0x30)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True, key_cert_sign=True), critical=True)
        .sign(root_key, hashes.SHA256())
    )

    leaf_of_non_ca_signer = (
        x509.CertificateBuilder()
        .subject_name(name("non-ca-signer.example"))
        .issuer_name(non_ca_signer.subject)
        .public_key(leaf_of_non_ca_key.public_key())
        .serial_number(0x31)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("non-ca-signer.example")]), critical=False
        )
        .sign(non_ca_signer_key, hashes.SHA256())
    )

    # A leaf whose unknown extension is marked **critical**. RFC 5280 requires a
    # reader that does not understand a critical extension to reject the
    # certificate, so this one must not parse.
    leaf_unknown_critical = (
        x509.CertificateBuilder()
        .subject_name(name("critical.example"))
        .issuer_name(intermediate.subject)
        .public_key(leaf_key.public_key())
        .serial_number(0x06)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.UnrecognizedExtension(ObjectIdentifier("1.3.6.1.4.1.99999.2"), b"\x04\x03xyz"),
            critical=True,
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("critical.example")]), critical=False
        )
        .sign(intermediate_key, hashes.SHA256())
    )

    fixtures = [
        ("ROOT", root),
        ("INTERMEDIATE", intermediate),
        ("LEAF_RSA", leaf_rsa),
        ("LEAF_EC", leaf_ec),
        ("LEAF_CLIENT_AUTH", leaf_client_auth),
        ("LEAF_NO_DIGITAL_SIGNATURE", leaf_no_sign),
        ("SELF_SIGNED_PSS", self_signed_pss),
        ("SELF_SIGNED_EC", self_signed_ec),
        ("LEAF_UNKNOWN_CRITICAL", leaf_unknown_critical),
        ("ROOT_SHADOW", root_shadow),
        ("TOP_CA", top_ca),
        ("TOP_CA_STRICT", top_ca_strict),
        ("SUB_CA", sub_ca),
        ("LEAF_SUB", leaf_sub),
        ("CA_NO_KEY_CERT_SIGN", no_key_cert_sign_ca),
        ("LEAF_OF_NO_SIGN_CA", leaf_of_no_sign),
        ("SUB_CA_STRICT", sub_ca_strict),
        ("LEAF_SUB_STRICT", leaf_sub_strict),
        ("EXPIRED_ANCHOR", expired_anchor),
        ("LEAF_OF_EXPIRED_ANCHOR", leaf_of_expired_anchor),
        ("WEAK_ANCHOR", weak_anchor),
        ("LEAF_OF_WEAK_ANCHOR", leaf_of_weak_anchor),
        ("NON_CA_SIGNER", non_ca_signer),
        ("LEAF_OF_NON_CA_SIGNER", leaf_of_non_ca_signer),
    ]

    # `//` rather than `//!`: this file is pulled in with `include!`, so an inner
    # doc comment would land in the middle of the including module.
    out = []
    out.append("// X.509 fixtures and the values OpenSSL's parser reads out of them.")
    out.append("//")
    out.append("// Generated by scripts/tls_x509_fixtures.py — do not edit by hand.")
    out.append("//")
    out.append("// The keys come from a fixed seed, so four of these fixtures regenerate byte for")
    out.append("// byte. Two do not: the RSA-PSS one (random salt) and the ECDSA one (random")
    out.append("// nonce) carry a fresh signature on every run, which can also move the ECDSA")
    out.append("// certificate's total length. Every field asserted below is read out of the")
    out.append("// certificate, so a regenerated file passes the same tests — but this file is")
    out.append("// the reviewed pin, so regenerate only deliberately.")
    out.append("")
    for label, cert in fixtures:
        der = cert.public_bytes(serialization.Encoding.DER)
        out.append('const %s_DER_HEX: &str = "%s";\n' % (label, der_hex(der)))
    out.append("/// What OpenSSL's parser says about each fixture above.")
    out.append("const EXPECTATIONS: &[(&str, Expectation)] = &[")
    for label, cert in fixtures:
        der = cert.public_bytes(serialization.Encoding.DER)
        want = expectations(cert)
        names = ", ".join('"%s"' % n for n in want["san_dns"])
        out.append("    (")
        out.append('        "%s",' % label)
        out.append("        Expectation {")
        out.append("            der_len: %d," % len(der))
        out.append("            signature_len: %d," % signature_length(der))
        out.append("            version: %d," % (cert.version.value + 1))
        out.append('            subject_cn: "%s",' % want["subject_cn"])
        out.append('            issuer_cn: "%s",' % want["issuer_cn"])
        out.append('            serial_hex: "%s",' % want["serial"])
        out.append("            not_before: %d," % want["not_before"])
        out.append("            not_after: %d," % want["not_after"])
        out.append("            is_ca: %s," % ("true" if want["is_ca"] else "false"))
        out.append(
            "            has_basic_constraints: %s,"
            % ("true" if want["has_basic_constraints"] else "false")
        )
        out.append(
            "            path_len: %s,"
            % ("None" if want["path_len"] is None else "Some(%d)" % want["path_len"])
        )
        out.append(
            "            key_usage: %s,"
            % ("None" if want["key_usage"] is None else "Some(0x%03x)" % want["key_usage"])
        )
        out.append("            san_dns: &[%s]," % names)
        out.append('            signature_algorithm: "%s",' % want["signature_algorithm"])
        out.append("            key: %s," % want["key"])
        out.append("        },")
        out.append("    ),")
    out.append("];")
    out.append("")

    out.append("/// What the OpenSSL CLI says about each of these paths.")
    out.append("///")
    out.append("// %s" % openssl_version())
    out.append("// `openssl verify -partial_chain -auth_level 0 -attime <instant>`")
    out.append("///")
    out.append("/// `divergence` is set only where the two are known to disagree, and only ever")
    out.append("/// where OpenSSL accepts: this crate may be stricter than OpenSSL, never laxer.")
    out.append("const VERIFY_CASES: &[VerifyCase] = &[")
    by_label = dict(fixtures)
    for name, chain, anchors, instant_from, note in verify_cases():
        attime = int(by_label[instant_from].not_valid_before_utc.timestamp())
        ok, code, depth = openssl_verdict(by_label, chain, anchors, attime)
        out.append("    VerifyCase {")
        out.append('        name: "%s",' % name)
        out.append("        chain: &[%s]," % ", ".join('"%s"' % label for label in chain))
        out.append("        anchors: &[%s]," % ", ".join('"%s"' % label for label in anchors))
        out.append("        attime: %d," % attime)
        out.append("        openssl_ok: %s," % ("true" if ok else "false"))
        out.append(
            "        openssl_error: %s," % ("None" if code is None else "Some(%d)" % code)
        )
        out.append(
            "        openssl_depth: %s," % ("None" if depth is None else "Some(%d)" % depth)
        )
        out.append(
            "        divergence: %s,"
            % ("None" if note is None else 'Some("%s")' % note.replace('"', '\\"'))
        )
        out.append("    },")
    out.append("];")
    out.append("")

    destination = sys.argv[1] if len(sys.argv) > 1 else "crates/hearth-tls/tests/data/x509_fixtures.rs"
    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(out))

    print("wrote %s: %d fixtures, %d lines" % (destination, len(fixtures), len(out)))
    print(
        "linkage: leaf->intermediate %s, intermediate->root %s"
        % (leaf_rsa.issuer == intermediate.subject, intermediate.issuer == root.subject),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
