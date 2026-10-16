#!/usr/bin/env python3
"""Generate the X.509 fixtures for `crates/hearth-tls/tests/x509_fixtures.rs`.

The expectations printed here come from OpenSSL through the `cryptography`
package's own X.509 parser — subject, issuer, validity, key, basicConstraints,
keyUsage, subjectAltName and the signature algorithm — so the test that consumes
them is a differential test against a real implementation rather than a snapshot
of our own output.

The keys are derived from a fixed seed with `sympy` primes, so the *keys* are the
same on every run and every fixture regenerates byte for byte except two:
`SELF_SIGNED_PSS` is RSA-PSS, whose salt is random, and `SELF_SIGNED_EC` is ECDSA,
whose nonce is random — so their signature bytes differ on every run, and the ECDSA
signature's *length* can differ too, which moves that certificate's whole DER
length. The `CV_*` signature vectors below are RSA-PSS and ECDSA for the same
reason, and are verified by the OpenSSL CLI before being written down.

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
import hashlib
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


def der_tlv(tag, body):
    """One DER element, with a minimal length."""
    if len(body) < 0x80:
        return bytes([tag, len(body)]) + body
    encoded = len(body).to_bytes((len(body).bit_length() + 7) // 8, "big")
    return bytes([tag, 0x80 | len(encoded)]) + encoded + body


def der_sequence(*parts):
    return der_tlv(0x30, b"".join(parts))


def der_integer(value):
    raw = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
    if raw[0] & 0x80:
        raw = b"\x00" + raw
    return der_tlv(0x02, raw)


def der_utc_time(moment):
    return der_tlv(0x17, moment.strftime("%y%m%d%H%M%SZ").encode())


def build_crl(issuer_der, signing_key, serials, *, last, next_, version=2, extensions=()):
    """A CRL written out by hand.

    The library cannot produce the shapes that matter here — v1 (no version field and no
    extensions), a CRL with no `nextUpdate`, and one carrying an extension of our own
    choosing — so the bytes are built directly. `issuer_der` is the issuer's own `Name`
    encoding as `Name.public_bytes()` gives it, which is the same encoding a certificate
    carries, and `extensions` are `(oid, value, critical)` triples.

    Everything written down here is checked afterwards by [`crl_facts_from_openssl`]: a
    hand-built structure whose two readers disagree is a fixture bug, and it raises
    rather than being pinned.
    """
    algorithm = der_sequence(
        der_tlv(0x06, bytes([0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x0B])),
        der_tlv(0x05, b""),
    )
    fields = []
    if version == 2:
        # A plain INTEGER, not the `[0] EXPLICIT` a *certificate* wraps its version in.
        # The tagged form is unreadable to OpenSSL and is the shape this generator first
        # wrote, which is how the same mistake survived in the Rust parser: two readings
        # of the format that agreed with each other and with nothing else.
        fields.append(der_integer(1))
    fields.append(algorithm)
    fields.append(issuer_der)
    fields.append(der_utc_time(last))
    if next_ is not None:
        fields.append(der_utc_time(next_))
    if serials:
        entries = b"".join(
            der_sequence(der_integer(serial), der_utc_time(last)) for serial in serials
        )
        fields.append(der_sequence(entries))
    if extensions:
        listed = b"".join(
            der_sequence(
                der_tlv(0x06, oid)
                + (b"\x01\x01\xff" if critical else b"")
                + der_tlv(0x04, value)
            )
            for oid, value, critical in extensions
        )
        fields.append(der_tlv(0xA0, der_sequence(listed)))
    tbs = der_sequence(*fields)
    signature = signing_key.sign(tbs, padding.PKCS1v15(), hashes.SHA256())
    return der_sequence(tbs, algorithm, der_tlv(0x03, b"\x00" + signature))


def crl_facts_from_openssl(der):
    """What the OpenSSL CLI reads out of a CRL, or a refusal.

    A CRL the tool cannot *load* is not a CRL the tool refuses, and the difference has
    already cost this project one wrong conclusion, so an unreadable CRL raises here
    instead of producing a verdict.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "one.der")
        with open(path, "wb") as handle:
            handle.write(der)
        result = subprocess.run(
            ["openssl", "crl", "-in", path, "-inform", "DER", "-noout", "-text"],
            capture_output=True,
            text=True,
        )
        text = result.stdout + result.stderr
        if result.returncode != 0:
            raise RuntimeError("openssl cannot read this CRL: %s" % text.strip()[:200])
    # The CLI prints the serial number's DER contents, so a serial that needed a leading
    # zero octet to stay positive arrives with it. The Rust side reads magnitudes, so what
    # is recorded is the magnitude: colons out, lower case, leading zeros off.
    # `strip` first: a revoked entry's serial is printed indented, and a `startswith` on
    # the raw line silently reads *no* serials out of every CRL with entries in it —
    # which is exactly what the first version of this did, on a list that does have one.
    serials = [
        line.split(":", 1)[1].strip().replace(":", "").lower().lstrip("0").rjust(2, "0")
        for line in text.splitlines()
        if line.strip().startswith("Serial Number:")
    ]
    return {
        "version": 2 if "Version 2" in text else 1,
        "serials": serials,
        "this_update": _openssl_moment(text, "Last Update:"),
        "next_update": _openssl_moment(text, "Next Update:"),
    }


def _openssl_moment(text, prefix):
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            value = line.split(":", 1)[1].strip()
            if value == "NONE":
                return None
            parsed = datetime.datetime.strptime(value, "%b %d %H:%M:%S %Y GMT")
            return int(parsed.replace(tzinfo=UTC).timestamp())
    raise RuntimeError("no %s line in the CRL text" % prefix)


def openssl_crl_verdict(by_label, crl_ders, chain, anchors, attime, crl_check_all):
    """`(ok, error, depth)` from the OpenSSL CLI for one path, checked with CRLs.

    The same three flags as [`openssl_verdict`] and one more decision: `-crl_check_all`
    asks about every certificate in the path, which is what this crate's `Revocation::
    Required` means, while `-crl_check` asks about the leaf alone. Both are used, because
    the gap between them is one of the recorded divergences.
    """
    with tempfile.TemporaryDirectory() as tmp:
        ca = os.path.join(tmp, "ca.pem")
        leaf = os.path.join(tmp, "leaf.pem")
        untrusted = os.path.join(tmp, "chain.pem")
        crl_file = os.path.join(tmp, "crls.pem")
        write_pem(by_label, ca, anchors)
        write_pem(by_label, leaf, [chain[0]])
        with open(crl_file, "wb") as handle:
            handle.write(b"".join(crl_ders))
        command = [
            "openssl", "verify",
            "-CAfile", ca,
            "-CRLfile", crl_file,
            "-crl_check_all" if crl_check_all else "-crl_check",
            "-partial_chain",
            "-attime", str(attime),
            "-auth_level", "0",
        ]
        if chain[1:]:
            write_pem(by_label, untrusted, chain[1:])
            command += ["-untrusted", untrusted]
        result = subprocess.run(command + [leaf], capture_output=True, text=True)

    found = re.search(r"error (\d+) at (\d+) depth lookup", result.stdout + result.stderr)
    return (
        result.returncode == 0,
        int(found.group(1)) if found else None,
        int(found.group(2)) if found else None,
    )


def crl_cases():
    """The paths both implementations are asked about, with revocation turned on.

    `(name, chain, anchors, crls, instant-from, crl_check_all, diverges)`, where `crls`
    names fixtures from [`crl_fixtures`]. `diverges` is set only where a disagreement is
    expected, and the test enforces the direction: OpenSSL may accept while this crate
    refuses, never the reverse.
    """
    return [
        ("crl-clean-path", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_CLEAN", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-leaf-revoked", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_REVOKES_LEAF", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-intermediate-revoked", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_CLEAN", "CRL_BY_ROOT_REVOKES_INTERMEDIATE"], "LEAF_RSA", True, None),
        ("crl-anchor-revoked-by-its-own-list", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_CLEAN", "CRL_BY_ROOT", "CRL_BY_ROOT_REVOKES_ROOT"], "LEAF_RSA", True, None),
        ("crl-missing-for-the-leaf", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-missing-for-the-intermediate", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_CLEAN"], "LEAF_RSA", True, None),
        (
            "crl-missing-for-the-intermediate-leaf-only-check",
            ["LEAF_RSA", "INTERMEDIATE"],
            ["ROOT"],
            ["CRL_CLEAN"],
            "LEAF_RSA",
            False,
            "OpenSSL's `-crl_check` asks only about the leaf, so a revoked intermediate "
            "goes unnoticed there; this crate checks every certificate in the path, so a "
            "CRL for the intermediate is required rather than optional",
        ),
        ("crl-expired", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_EXPIRED", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-not-yet-valid", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_NOT_YET_VALID", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-without-next-update", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_NO_NEXT_UPDATE", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-v1", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_V1", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-v1-revokes-the-leaf", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_V1_REVOKES_LEAF", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        (
            "crl-unknown-critical-extension",
            ["LEAF_RSA", "INTERMEDIATE"],
            ["ROOT"],
            ["CRL_UNKNOWN_CRITICAL", "CRL_BY_ROOT"],
            "LEAF_RSA",
            True,
            "both refuse, at different layers: OpenSSL loads the CRL and returns error 36, "
            "while this crate refuses to read it at all (RFC 5280 §5.2 makes an unknown "
            "critical extension fatal)",
        ),
        (
            "crl-delta-indicator",
            ["LEAF_RSA", "INTERMEDIATE"],
            ["ROOT"],
            ["CRL_DELTA", "CRL_BY_ROOT"],
            "LEAF_RSA",
            True,
            "a delta CRL lists only what changed since a base CRL, so using it as a complete "
            "one reports as unrevoked what the base revokes; OpenSSL 3.5.5 accepts it in this "
            "position and this crate refuses to read it",
        ),
        ("crl-signed-by-the-wrong-key", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_WRONG_KEY", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-stale-revoking-beside-fresh-clean", ["LEAF_RSA", "INTERMEDIATE"], ["ROOT"], ["CRL_REVOKES_LEAF_STALE", "CRL_CLEAN", "CRL_BY_ROOT"], "LEAF_RSA", True, None),
        ("crl-issuer-without-crl-sign", ["LEAF_OF_NO_CRL_SIGN_CA", "CA_NO_CRL_SIGN"], ["ROOT"], ["CRL_BY_NO_CRL_SIGN", "CRL_BY_ROOT"], "LEAF_OF_NO_CRL_SIGN_CA", True, None),
    ]


def check_crl_facts(label, der, expected=None):
    """The OpenSSL CLI's reading of a CRL, checked against what was written down.

    Two readers of the same bytes: the CLI, which is independent, and — for the
    hand-built CRLs alone — the facts their own construction claims. A disagreement is a
    fixture bug and raises here rather than being pinned into the table.
    """
    try:
        facts = crl_facts_from_openssl(der)
    except RuntimeError as error:
        raise RuntimeError("%s: %s" % (label, error))
    for key, wanted in (expected or {}).items():
        if facts[key] != wanted:
            raise RuntimeError(
                "%s: openssl reads %s=%s, but the fixture claims %s"
                % (label, key, facts[key], wanted)
            )
    return facts


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
        ("ec-leaf-and-intermediate-to-root", ["LEAF_EC_VALID", "INTERMEDIATE"], ["ROOT"], "LEAF_EC_VALID", None),
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
        ("root-included-in-the-chain", ["LEAF_RSA", "INTERMEDIATE", "ROOT"], ["ROOT"], "LEAF_RSA", None),
        (
            "key-rollover-self-issued-intermediate",
            ["LEAF_OF_ROLLOVER_CA_B", "ROLLOVER_CA_B", "ROLLOVER_CA_A"],
            ["ROOT"],
            "LEAF_OF_ROLLOVER_CA_B",
            None,
        ),
        (
            "leaf-whose-key-identifier-matches",
            ["LEAF_WITH_MATCHING_AKI", "AKI_CA"],
            ["ROOT"],
            "LEAF_WITH_MATCHING_AKI",
            None,
        ),
        (
            "leaf-whose-key-identifier-points-nowhere",
            ["LEAF_WITH_MISMATCHED_AKI", "AKI_CA"],
            ["ROOT"],
            "LEAF_WITH_MISMATCHED_AKI",
            None,
        ),
        (
            # The anchor shares the last certificate's distinguished name without being it,
            # which is the branch of the anchor-side count nothing else reaches.
            "rollover-pinned-at-the-newer-ca",
            ["LEAF_OF_ROLLOVER_CA_B", "ROLLOVER_CA_B"],
            ["ROLLOVER_CA_A"],
            "LEAF_OF_ROLLOVER_CA_B",
            None,
        ),
        (
            "chain-out-of-order",
            ["LEAF_RSA", "ROOT", "INTERMEDIATE"],
            ["ROOT"],
            "LEAF_RSA",
            "this crate builds exactly the path it is given, and RFC 8446 §4.4.2 requires each "
            "certificate to certify the one before it; OpenSSL selects a path out of the pool instead",
        ),
        (
            "chain-with-the-leaf-twice",
            ["LEAF_RSA", "LEAF_RSA", "INTERMEDIATE"],
            ["ROOT"],
            "LEAF_RSA",
            "a duplicate makes the presented list non-conforming, so this crate refuses it rather "
            "than picking certificates out of it; OpenSSL ignores what it does not need",
        ),
        (
            "chain-with-an-unrelated-certificate",
            ["LEAF_RSA", "INTERMEDIATE", "SELF_SIGNED_EC"],
            ["ROOT"],
            "LEAF_RSA",
            "same shape: an extra certificate is not part of the path, and this crate refuses the "
            "list rather than selecting from it",
        ),
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

    # The key identifiers, read out of OpenSSL's own parser: the two extensions the
    # chain builder compares, and the only ones whose fields are context-tagged.
    subject_key_id = None
    authority_key_id = None
    for extension in cert.extensions:
        if isinstance(extension.value, x509.SubjectKeyIdentifier):
            subject_key_id = extension.value.digest.hex()
        elif isinstance(extension.value, x509.AuthorityKeyIdentifier):
            identifier = extension.value.key_identifier
            authority_key_id = identifier.hex() if identifier else None

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
        "subject_key_id": subject_key_id,
        "authority_key_id": authority_key_id,
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
    # Two more identities for the revocation slice: a CA that may issue certificates but
    # may not publish revocation lists, and a leaf it signed.
    no_crl_sign_ca_key = rsa_key(20261501)
    leaf_of_no_crl_sign_key = rsa_key(20261502)
    # An EC leaf that is *actually usable*: its window sits inside its issuer's, unlike
    # `LEAF_EC`, whose window starts after its issuer's has closed. The whole verifier needs
    # one chain per signature suite, or a verifier that only ever worked for RSA would pass.
    leaf_ec_valid_key = ec_key(20261601)
    aki_ca_key = rsa_key(20261401)
    leaf_with_matching_aki_key = rsa_key(20261402)
    leaf_with_mismatched_aki_key = rsa_key(20261403)
    rollover_ca_a_key = rsa_key(20261301)
    rollover_ca_b_key = rsa_key(20261302)
    leaf_of_rollover_key = rsa_key(20261303)

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

    # A **key rollover**: a CA that re-issues itself under the same distinguished name with
    # a new key, and a leaf under the new one. RFC 5280 calls such a certificate
    # *self-issued* — the same DN in the subject and issuer fields, whatever key signed it —
    # and §4.2.1.9 excludes self-issued certificates from a `pathLenConstraint` count, which
    # is what makes a rollover possible at all. No other fixture here is self-issued except
    # the roots, so without this pair the branch that does the excluding never excludes
    # anything.
    #
    # These three, and only these three, carry `subjectKeyIdentifier` and
    # `authorityKeyIdentifier`. That is not decoration: a rollover is *ambiguous by subject
    # name alone*, and without the key identifiers a verifier that matches by name picks the
    # wrong certificate — OpenSSL does, and reports either a signature failure (error 7) or
    # a self-signed certificate in the chain (error 19) on paths that are in fact sound. The
    # rest of the set omits them deliberately: it keeps those certificates minimal, and it
    # is what makes `openssl verify -x509_strict` refuse every one of them.
    def key_identifiers(builder, subject_key, issuer_key):
        return (
            builder
            .add_extension(
                x509.SubjectKeyIdentifier.from_public_key(subject_key.public_key()),
                critical=False,
            )
            .add_extension(
                x509.AuthorityKeyIdentifier.from_issuer_public_key(issuer_key.public_key()),
                critical=False,
            )
        )

    rollover_ca_a = key_identifiers(
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Rollover CA"))
        .issuer_name(root.subject)
        .public_key(rollover_ca_a_key.public_key())
        .serial_number(0x40)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True),
        rollover_ca_a_key,
        root_key,
    ).sign(root_key, hashes.SHA256())

    rollover_ca_b = key_identifiers(
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test Rollover CA"))
        .issuer_name(rollover_ca_a.subject)
        .public_key(rollover_ca_b_key.public_key())
        .serial_number(0x41)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True),
        rollover_ca_b_key,
        rollover_ca_a_key,
    ).sign(rollover_ca_a_key, hashes.SHA256())

    leaf_of_rollover_ca_b = key_identifiers(
        x509.CertificateBuilder()
        .subject_name(name("rollover.example"))
        .issuer_name(rollover_ca_b.subject)
        .public_key(leaf_of_rollover_key.public_key())
        .serial_number(0x42)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("rollover.example")]), critical=False
        ),
        leaf_of_rollover_key,
        rollover_ca_b_key,
    ).sign(rollover_ca_b_key, hashes.SHA256())

    # Whether a child's `authorityKeyIdentifier` has to agree with its issuer's
    # `subjectKeyIdentifier`. `AKI_CA` names its own key; one leaf points at it, the other
    # points at twenty bytes nothing owns. The second is the shape a rollover makes
    # dangerous: two certificates with one distinguished name, and the key identifier as the
    # only thing that says which one signed you.
    aki_ca = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test AKI CA"))
        .issuer_name(root.subject)
        .public_key(aki_ca_key.public_key())
        .serial_number(0x50)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True, crl_sign=True), critical=True)
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(aki_ca_key.public_key()), critical=False
        )
        .sign(root_key, hashes.SHA256())
    )

    def aki_leaf(cn, key, serial, identifier):
        return (
            x509.CertificateBuilder()
            .subject_name(name(cn))
            .issuer_name(aki_ca.subject)
            .public_key(key.public_key())
            .serial_number(serial)
            .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
            .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(usage(digital_signature=True), critical=True)
            .add_extension(
                x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
            )
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(cn)]), critical=False)
            .add_extension(
                x509.AuthorityKeyIdentifier(
                    key_identifier=identifier, authority_cert_issuer=None, authority_cert_serial_number=None
                ),
                critical=False,
            )
            .sign(aki_ca_key, hashes.SHA256())
        )

    # The identifier the CA actually has, and twenty bytes of nothing.
    aki_of_aki_ca = x509.SubjectKeyIdentifier.from_public_key(aki_ca_key.public_key()).digest
    nobody_key_id = bytes.fromhex("deadbeef" * 5)
    leaf_with_matching_aki = aki_leaf("aki-match.example", leaf_with_matching_aki_key, 0x51, aki_of_aki_ca)
    leaf_with_mismatched_aki = aki_leaf(
        "aki-mismatch.example", leaf_with_mismatched_aki_key, 0x52, nobody_key_id
    )

    # A CA that may sign certificates but may **not** sign revocation lists: the
    # `keyUsage` bit that matters for a CRL is `cRLSign`, and this is the fixture that
    # makes it a rule rather than a comment. It is an ordinary CA in every other way, and
    # its leaf really is signed by it.
    no_crl_sign_ca = (
        x509.CertificateBuilder()
        .subject_name(name("Hearth Test No-CRL-Sign CA"))
        .issuer_name(root.subject)
        .public_key(no_crl_sign_ca_key.public_key())
        .serial_number(0x53)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(usage(key_cert_sign=True), critical=True)
        .sign(root_key, hashes.SHA256())
    )
    leaf_of_no_crl_sign_ca = (
        x509.CertificateBuilder()
        .subject_name(name("no-crl-sign.example"))
        .issuer_name(no_crl_sign_ca.subject)
        .public_key(leaf_of_no_crl_sign_key.public_key())
        .serial_number(0x54)
        .not_valid_before(datetime.datetime(2026, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2027, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("no-crl-sign.example")]), critical=False
        )
        .sign(no_crl_sign_ca_key, hashes.SHA256())
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

    # ---- the revocation lists --------------------------------------------------
    #
    # Built last because they are about the certificates above. The windows are chosen
    # around the instant the CRL cases use (LEAF_RSA's `notBefore`, 2026-01-01Z): a
    # "current" list covers it, an expired one closed before it, and a future one opens
    # after it.
    window_start = datetime.datetime(2025, 1, 1, tzinfo=UTC)
    window_end = datetime.datetime(2027, 1, 1, tzinfo=UTC)
    stale_start = datetime.datetime(2019, 1, 1, tzinfo=UTC)
    stale_end = datetime.datetime(2021, 1, 1, tzinfo=UTC)
    later_start = datetime.datetime(2025, 6, 1, tzinfo=UTC)
    future_start = datetime.datetime(2030, 1, 1, tzinfo=UTC)
    leaf_serial = 0xFF
    intermediate_serial = 0x02
    root_serial = 0x01

    def signed_crl(issuer_cert, signing_key, serials, *, last, next_, number):
        builder = (
            x509.CertificateRevocationListBuilder()
            .issuer_name(issuer_cert.subject)
            .last_update(last)
            .next_update(next_)
            .add_extension(x509.CRLNumber(number), critical=False)
        )
        for serial in serials:
            builder = builder.add_revoked_certificate(
                x509.RevokedCertificateBuilder()
                .serial_number(serial)
                .revocation_date(last)
                .build()
            )
        return builder.sign(signing_key, hashes.SHA256())

    # The two the hierarchy's own CAs publish, and the four shapes of "which instant is
    # this list about": current, superseded, expired, and not yet in force.
    crl_clean = signed_crl(intermediate, intermediate_key, [], last=window_start, next_=window_end, number=1)
    crl_by_root = signed_crl(root, root_key, [], last=window_start, next_=window_end, number=1)
    crl_revokes_leaf = signed_crl(intermediate, intermediate_key, [leaf_serial], last=later_start, next_=window_end, number=2)
    crl_revokes_leaf_stale = signed_crl(intermediate, intermediate_key, [leaf_serial], last=stale_start, next_=stale_end, number=1)
    crl_expired = signed_crl(intermediate, intermediate_key, [], last=stale_start, next_=stale_end, number=3)
    crl_not_yet_valid = signed_crl(intermediate, intermediate_key, [], last=future_start, next_=datetime.datetime(2031, 1, 1, tzinfo=UTC), number=4)
    crl_by_root_revokes_intermediate = signed_crl(root, root_key, [intermediate_serial], last=later_start, next_=window_end, number=2)
    crl_by_root_revokes_root = signed_crl(root, root_key, [root_serial], last=later_start, next_=window_end, number=3)
    crl_by_no_crl_sign = signed_crl(no_crl_sign_ca, no_crl_sign_ca_key, [], last=window_start, next_=window_end, number=1)
    # The right issuer name signed by the wrong key: a same-name impostor's list.
    crl_wrong_key = signed_crl(intermediate, root_key, [], last=window_start, next_=window_end, number=5)
    # Two lists published at the *same* instant, with the same CRL number, that disagree
    # about the same certificate: nothing in either says which is the later one, so there is
    # no honest way to pick between them. This pair exists for the rule that refuses instead
    # of guessing. OpenSSL is not asked about it — either answer it gave would be a choice,
    # not a rule.
    crl_conflict_clean = signed_crl(intermediate, intermediate_key, [], last=window_start, next_=window_end, number=2)
    crl_conflict_revokes_leaf = signed_crl(intermediate, intermediate_key, [leaf_serial], last=window_start, next_=window_end, number=2)

    # Hand-built, because the library cannot produce a v1 CRL, a CRL with no
    # `nextUpdate`, or one carrying an extension of our choosing. Their facts are given
    # below and checked against the OpenSSL CLI's reading of the same bytes, so a
    # hand-built structure whose two readers disagree stops the generator.
    intermediate_name = intermediate.subject.public_bytes()
    unknown_oid = bytes([0x2A, 0x03, 0x04])
    delta_oid = bytes([0x55, 0x1D, 0x1B])
    crl_no_next_update = build_crl(
        intermediate_name, intermediate_key, [], last=window_start, next_=None
    )
    crl_v1 = build_crl(intermediate_name, intermediate_key, [], last=window_start, next_=window_end, version=1)
    crl_v1_revokes_leaf = build_crl(
        intermediate_name, intermediate_key, [leaf_serial], last=window_start, next_=window_end, version=1
    )
    crl_unknown_critical = build_crl(
        intermediate_name,
        intermediate_key,
        [],
        last=window_start,
        next_=window_end,
        extensions=[(unknown_oid, b"\x05\x00", True)],
    )
    crl_delta = build_crl(
        intermediate_name,
        intermediate_key,
        [],
        last=window_start,
        next_=window_end,
        extensions=[(delta_oid, b"\x02\x01\x07", True)],
    )

    # P-256, a window inside the intermediate's, and a name of its own: the chain an
    # end-to-end verification of an `ecdsa_secp256r1_sha256` CertificateVerify needs.
    leaf_ec_valid = (
        x509.CertificateBuilder()
        .subject_name(name("ec-signer.example"))
        .issuer_name(intermediate.subject)
        .public_key(leaf_ec_valid_key.public_key())
        .serial_number(0x55)
        .not_valid_before(datetime.datetime(2030, 1, 1, tzinfo=UTC))
        .not_valid_after(datetime.datetime(2035, 1, 1, tzinfo=UTC))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(usage(digital_signature=True), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([x509.oid.ExtendedKeyUsageOID.SERVER_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("ec-signer.example")]), critical=False
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
        ("ROLLOVER_CA_A", rollover_ca_a),
        ("ROLLOVER_CA_B", rollover_ca_b),
        ("LEAF_OF_ROLLOVER_CA_B", leaf_of_rollover_ca_b),
        ("AKI_CA", aki_ca),
        ("LEAF_WITH_MATCHING_AKI", leaf_with_matching_aki),
        ("LEAF_WITH_MISMATCHED_AKI", leaf_with_mismatched_aki),
        ("CA_NO_CRL_SIGN", no_crl_sign_ca),
        ("LEAF_OF_NO_CRL_SIGN_CA", leaf_of_no_crl_sign_ca),
        ("LEAF_EC_VALID", leaf_ec_valid),
    ]

    # ---- the CertificateVerify vectors -----------------------------------------
    #
    # RFC 8446 §4.4.3's construction — 64 spaces, the context string, a zero byte, then the
    # transcript hash through `Certificate` — signed with the fixture leaves' own keys, so a
    # whole `CertificateVerifier` can be driven from a test. The hash is a fixed 32 bytes and
    # not a transcript of anything: what these pin is the *arithmetic and the dispatch*, and
    # the real transcript is already pinned by RFC 8448's own recorded flight. Each signature
    # is verified by the OpenSSL CLI before it is written down here, so a vector this crate
    # accepts is one a second implementation also accepts.
    transcript_hash = hashlib.sha256(b"hearth fixture transcript hash").digest()
    signed_data = b"\x20" * 64 + b"TLS 1.3, server CertificateVerify" + b"\x00" + transcript_hash
    cv_rsa_signature = leaf_key.sign(
        signed_data,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
        hashes.SHA256(),
    )
    cv_ecdsa_signature = leaf_ec_valid_key.sign(signed_data, ec.ECDSA(hashes.SHA256()))
    cv_rsa_forged = bytearray(cv_rsa_signature)
    cv_rsa_forged[-1] ^= 0x01

    def check_signature_with_openssl(public_key, signature, *, pss):
        """Whether the OpenSSL CLI verifies this signature over `signed_data`."""
        with tempfile.TemporaryDirectory() as tmp:
            key_path = os.path.join(tmp, "leaf.pem")
            data_path = os.path.join(tmp, "data.bin")
            sig_path = os.path.join(tmp, "sig.bin")
            with open(key_path, "wb") as handle:
                handle.write(
                    public_key.public_bytes(
                        serialization.Encoding.PEM,
                        serialization.PublicFormat.SubjectPublicKeyInfo,
                    )
                )
            with open(data_path, "wb") as handle:
                handle.write(signed_data)
            with open(sig_path, "wb") as handle:
                handle.write(bytes(signature))
            command = ["openssl", "dgst", "-sha256", "-verify", key_path]
            if pss:
                command += [
                    "-sigopt", "rsa_padding_mode:pss",
                    "-sigopt", "rsa_pss_saltlen:32",
                    "-sigopt", "rsa_mgf1_md:sha256",
                ]
            command += ["-signature", sig_path, data_path]
            result = subprocess.run(command, capture_output=True, text=True)
            return result.returncode == 0 and "Verified OK" in result.stdout

    if not check_signature_with_openssl(leaf_key.public_key(), cv_rsa_signature, pss=True):
        raise RuntimeError("the RSA CertificateVerify vector does not verify under OpenSSL")
    if not check_signature_with_openssl(
        leaf_ec_valid_key.public_key(), cv_ecdsa_signature, pss=False
    ):
        raise RuntimeError("the ECDSA CertificateVerify vector does not verify under OpenSSL")
    if check_signature_with_openssl(leaf_key.public_key(), cv_rsa_forged, pss=True):
        raise RuntimeError("the forged CertificateVerify vector verifies under OpenSSL")

    # The vectors the Rust tests consume: a name, the fixture whose private key signed (so
    # the test looks the public key up instead of restating it), the scheme code as RFC 8446
    # spells it (0x0804 rsa_pss_rsae_sha256, 0x0403 ecdsa_secp256r1_sha256), the constant
    # holding the signature, and whether a verifier must accept it. The PSS salt is 32
    # bytes because RFC 8446 §4.2.3 requires the salt to be as long as the hash.
    cv_vectors = [
        ("rsa-pss-leaf-signed", "LEAF_RSA", 0x0804, "CV_RSA_SIGNATURE", True),
        ("rsa-pss-leaf-forged", "LEAF_RSA", 0x0804, "CV_RSA_SIGNATURE_FORGED", False),
        ("ecdsa-leaf-signed", "LEAF_EC_VALID", 0x0403, "CV_ECDSA_SIGNATURE", True),
    ]

    # The CRLs, with the facts the OpenSSL CLI reads out of them. The hand-built ones
    # carry an expected reading, by construction, and it is checked rather than assumed.
    crl_fixtures = [
        ("CRL_CLEAN", crl_clean.public_bytes(serialization.Encoding.DER), None),
        ("CRL_BY_ROOT", crl_by_root.public_bytes(serialization.Encoding.DER), None),
        ("CRL_REVOKES_LEAF", crl_revokes_leaf.public_bytes(serialization.Encoding.DER), None),
        ("CRL_REVOKES_LEAF_STALE", crl_revokes_leaf_stale.public_bytes(serialization.Encoding.DER), None),
        ("CRL_EXPIRED", crl_expired.public_bytes(serialization.Encoding.DER), None),
        ("CRL_NOT_YET_VALID", crl_not_yet_valid.public_bytes(serialization.Encoding.DER), None),
        ("CRL_BY_ROOT_REVOKES_INTERMEDIATE", crl_by_root_revokes_intermediate.public_bytes(serialization.Encoding.DER), None),
        ("CRL_BY_ROOT_REVOKES_ROOT", crl_by_root_revokes_root.public_bytes(serialization.Encoding.DER), None),
        ("CRL_BY_NO_CRL_SIGN", crl_by_no_crl_sign.public_bytes(serialization.Encoding.DER), None),
        ("CRL_WRONG_KEY", crl_wrong_key.public_bytes(serialization.Encoding.DER), None),
        ("CRL_CONFLICT_CLEAN", crl_conflict_clean.public_bytes(serialization.Encoding.DER), None),
        ("CRL_CONFLICT_REVOKES_LEAF", crl_conflict_revokes_leaf.public_bytes(serialization.Encoding.DER), None),
        (
            "CRL_NO_NEXT_UPDATE",
            crl_no_next_update,
            {"version": 2, "serials": [], "this_update": int(window_start.timestamp()), "next_update": None},
        ),
        (
            "CRL_V1",
            crl_v1,
            {"version": 1, "serials": [], "this_update": int(window_start.timestamp()), "next_update": int(window_end.timestamp())},
        ),
        (
            "CRL_V1_REVOKES_LEAF",
            crl_v1_revokes_leaf,
            {"version": 1, "serials": ["ff"], "this_update": int(window_start.timestamp()), "next_update": int(window_end.timestamp())},
        ),
        (
            "CRL_UNKNOWN_CRITICAL",
            crl_unknown_critical,
            {"version": 2, "serials": [], "this_update": int(window_start.timestamp()), "next_update": int(window_end.timestamp())},
        ),
        (
            "CRL_DELTA",
            crl_delta,
            {"version": 2, "serials": [], "this_update": int(window_start.timestamp()), "next_update": int(window_end.timestamp())},
        ),
    ]
    crls_checked = [
        (label, der, check_crl_facts(label, der, expected)) for label, der, expected in crl_fixtures
    ]

    # `//` rather than `//!`: this file is pulled in with `include!`, so an inner
    # doc comment would land in the middle of the including module.
    out = []
    out.append("// X.509 fixtures and the values OpenSSL's parser reads out of them.")
    out.append("//")
    out.append("// Generated by scripts/tls_x509_fixtures.py — do not edit by hand.")
    out.append("//")
    out.append("// The keys come from a fixed seed, so every fixture here regenerates byte for byte")
    out.append("// except two: the RSA-PSS one (random salt) and the ECDSA one (random nonce),")
    out.append("// whose signatures — and, for ECDSA, DER length — change on every run. The")
    out.append("// CertificateVerify signature vectors are the same kind of thing. Every field")
    out.append("// asserted below is read out of the certificate, so a regenerated file passes the")
    out.append("// same tests — but this file is the reviewed pin, so regenerate only deliberately.")
    out.append("")
    for label, cert in fixtures:
        der = cert.public_bytes(serialization.Encoding.DER)
        out.append('const %s_DER_HEX: &str = "%s";\n' % (label, der_hex(der)))
    out.append("/// The DER of every certificate above, by name.")
    out.append("///")
    out.append("/// A table rather than a `match` in the test: a hand-written lookup is one more")
    out.append("/// place a new fixture can be left out of, and the failure it produces (\"no fixture")
    out.append("/// named X\") is a test author's problem rather than a fact about the certificate.")
    out.append("const DER_HEX: &[(&str, &str)] = &[")
    for label, _ in fixtures:
        out.append('    ("%s", %s_DER_HEX),' % (label, label))
    out.append("];")
    out.append("")
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
        out.append(
            "            subject_key_id: %s,"
            % (
                "None"
                if want["subject_key_id"] is None
                else 'Some("%s")' % want["subject_key_id"]
            )
        )
        out.append(
            "            authority_key_id: %s,"
            % (
                "None"
                if want["authority_key_id"] is None
                else 'Some("%s")' % want["authority_key_id"]
            )
        )
        out.append('            signature_algorithm: "%s",' % want["signature_algorithm"])
        out.append("            key: %s," % want["key"])
        out.append("        },")
        out.append("    ),")
    out.append("];")
    out.append("")

    out.append("/// Every revocation list, and what the OpenSSL CLI reads out of it.")
    out.append("///")
    out.append("/// Two of these carry extensions this crate refuses (an unknown critical one and a")
    out.append("/// deltaCRLIndicator), so their rows are the fields OpenSSL reports rather than a")
    out.append("/// parse this crate should reach. `CRL_NO_NEXT_UPDATE`, `CRL_V1`,")
    out.append("/// `CRL_V1_REVOKES_LEAF`, `CRL_UNKNOWN_CRITICAL` and `CRL_DELTA` are written out by")
    out.append("/// hand in the generator, because the library cannot build those shapes; the facts")
    out.append("/// written beside them are the ones they were built with, and the generator checks")
    out.append("/// them against the CLI's reading rather than trusting them.")
    out.append("const CRL_EXPECTATIONS: &[(&str, CrlExpectation)] = &[")
    for label, der, facts in crls_checked:
        serials = ", ".join('"%s"' % serial for serial in facts["serials"])
        out.append("    (")
        out.append('        "%s",' % label)
        out.append("        CrlExpectation {")
        out.append("            der_len: %d," % len(der))
        out.append("            version: %d," % facts["version"])
        out.append("            this_update: %d," % facts["this_update"])
        out.append(
            "            next_update: %s,"
            % ("None" if facts["next_update"] is None else "Some(%d)" % facts["next_update"])
        )
        out.append("            revoked_serials: &[%s]," % serials)
        out.append("        },")
        out.append("    ),")
    out.append("];")
    out.append("")
    out.append("/// The CRLs themselves, by fixture name.")
    out.append("const CRL_DER_HEX: &[(&str, &str)] = &[")
    for label, der, _ in crls_checked:
        out.append('    ("%s", "%s"),' % (label, der_hex(der)))
    out.append("];")
    out.append("")

    out.append("/// The bytes an RFC 8446 §4.4.3 `CertificateVerify` signature covers.")
    out.append("///")
    out.append("/// 64 spaces, the context string, a zero byte, then a 32-byte hash. The hash is a")
    out.append("/// fixed value rather than a transcript of anything: what the vectors below pin is")
    out.append("/// the verification arithmetic and the dispatch to the right key type, and a real")
    out.append("/// transcript is already pinned by the RFC 8448 flight. The *shape* is what a test")
    out.append("/// asserts here, so a generator change that dropped the context string is visible.")
    out.append('const CV_SIGNED_DATA_HEX: &str = "%s";' % signed_data.hex())
    out.append("")
    out.append("/// A `CertificateVerify` signature, verified by the OpenSSL CLI before it was")
    out.append("/// written down here. RSA-PSS and ECDSA both draw fresh randomness, so these bytes")
    out.append("/// differ on every regeneration — the tests assert that they verify, not that they")
    out.append("/// match anything.")
    out.append('const CV_RSA_SIGNATURE_HEX: &str = "%s";' % cv_rsa_signature.hex())
    out.append('const CV_ECDSA_SIGNATURE_HEX: &str = "%s";' % cv_ecdsa_signature.hex())
    out.append("")
    out.append("/// The same PSS signature with its last byte flipped: the bytes are a valid")
    out.append("/// length and the scheme is one we offered, so only the arithmetic can refuse it.")
    out.append('const CV_RSA_SIGNATURE_FORGED_HEX: &str = "%s";' % bytes(cv_rsa_forged).hex())
    out.append("")
    out.append("/// Signatures over [`CV_SIGNED_DATA_HEX`], and which of them must verify.")
    out.append("///")
    out.append("/// `leaf` names the fixture whose private key made the signature, so the test")
    out.append("/// looks up the public key it has to be checked against rather than restating it.")
    out.append("const CV_VECTORS: &[CvVector] = &[")
    for name, label, scheme, signature, okay in cv_vectors:
        out.append("    CvVector {")
        out.append('        name: "%s",' % name)
        out.append('        leaf: "%s",' % label)
        out.append("        scheme: 0x%04x," % scheme)
        out.append("        signature_hex: %s_HEX," % signature)
        out.append("        okay: %s," % ("true" if okay else "false"))
        out.append("    },")
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

    out.append("/// The same paths, asked of the OpenSSL CLI with revocation checking turned on.")
    out.append("///")
    out.append("// `openssl verify -crl_check_all -CRLfile <crls> -partial_chain -auth_level 0`")
    out.append("///")
    out.append("/// `crl_check_all` is false for the one case asked with `-crl_check` instead, which")
    out.append("/// is the flag that leaves every issuer in the path unchecked.")
    out.append("const CRL_CASES: &[CrlCase] = &[")
    by_crl_label = {label: der for label, der, _ in crls_checked}
    for name, chain, anchors, crls, instant_from, crl_check_all, note in crl_cases():
        attime = int(by_label[instant_from].not_valid_before_utc.timestamp())
        ok, code, depth = openssl_crl_verdict(
            by_label,
            [by_crl_label[label] for label in crls],
            chain,
            anchors,
            attime,
            crl_check_all,
        )
        out.append("    CrlCase {")
        out.append('        name: "%s",' % name)
        out.append("        chain: &[%s]," % ", ".join('"%s"' % label for label in chain))
        out.append("        anchors: &[%s]," % ", ".join('"%s"' % label for label in anchors))
        out.append("        crls: &[%s]," % ", ".join('"%s"' % label for label in crls))
        out.append("        attime: %d," % attime)
        out.append("        crl_check_all: %s," % ("true" if crl_check_all else "false"))
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
