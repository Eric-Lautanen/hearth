#!/usr/bin/env python3
"""Re-derive, from RFC 8448's trace alone, the values `tests/rfc8448_flight.rs` embeds.

This is the provenance for that test's RSA constants, and it exists so the claim
"every constant is re-derived from an independent implementation before being
embedded" can be *checked* rather than believed. It uses nothing from this
repository except the RFC's own hex, which it reads out of the test file:

  * `cryptography` (OpenSSL) for AES-GCM, for parsing the DER certificate, and for
    verifying the signature with its own PSS implementation;
  * `hashlib`/`hmac` for the HKDF-Expand-Label that derives the record key and IV.

It prints, and the test asserts, the same values:

  * the 128-byte RSA modulus, which the test also requires to appear verbatim
    inside the trace's 432-byte certificate;
  * `Hash(ClientHello..Certificate)`, which is the transcript hash the signature
    covers and which this crate computes independently;
  * `SHA-256(signed_data)` over the 130 bytes of RFC 8446 section 4.4.3;
  * the EMSA-PSS structure, so the checks in `rsa.rs` are known to be reached.

Requires `pip install cryptography`. Run from the repository root:

    python scripts/tls_rfc8448_rsa.py
"""
import hashlib
import hmac
import os
import re
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.x509 import load_der_x509_certificate

TEST_FILE = os.path.join("crates", "hearth-tls", "tests", "rfc8448_flight.rs")


def main():
    if not os.path.exists(TEST_FILE):
        sys.exit("run this from the repository root (%s not found)" % TEST_FILE)
    text = open(TEST_FILE, encoding="utf-8").read()

    def const(name):
        match = re.search(r'const %s: &str = "\\\n(.*?)";' % name, text, re.S)
        if not match:
            sys.exit("no constant %s in %s" % (name, TEST_FILE))
        return bytes(int(h, 16) for h in re.findall(r"[0-9a-f]{2}", match.group(1)))

    def hkdf_expand(prk, info, length):
        out, block, counter = b"", b"", 1
        while len(out) < length:
            block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
            out += block
            counter += 1
        return out[:length]

    def expand_label(secret, name, length):
        full = b"tls13 " + name
        info = length.to_bytes(2, "big") + bytes([len(full)]) + full + b"\x00"
        return hkdf_expand(secret, info, length)

    def mgf1(seed, length):
        out = b""
        for i in range((length + 31) // 32):
            out += hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        return out[:length]

    # Open the recorded server flight with OpenSSL's AES-GCM.
    secret = const("SERVER_HANDSHAKE_TRAFFIC_HEX")
    record = const("SERVER_FLIGHT_RECORD_HEX")
    assert record[0] == 0x17 and record[1:3] == b"\x03\x03", record[:5]
    assert int.from_bytes(record[3:5], "big") == len(record) - 5
    iv = expand_label(secret, b"iv", 12)
    nonce = bytes(a ^ b for a, b in zip(iv, (0).to_bytes(12, "big")))
    plain = AESGCM(expand_label(secret, b"key", 16)).decrypt(nonce, record[5:], record[:5])
    assert plain[-1] == 22, "the inner content type is handshake"
    payload = plain[:-1]
    print("opened the recorded flight: %d bytes of handshake messages" % len(payload))

    messages, off = [], 0
    while off < len(payload):
        length = int.from_bytes(payload[off + 1 : off + 4], "big")
        messages.append(payload[off : off + 4 + length])
        off += 4 + length
    assert [m[0] for m in messages] == [8, 11, 15, 20]

    # The certificate, framed as RFC 8446 section 4.4.2 requires.
    body = messages[1][4:]
    assert body[0] == 0, "an empty certificate_request_context"
    entry_len = int.from_bytes(body[4:7], "big")
    certificate = body[7 : 7 + entry_len]
    parsed = load_der_x509_certificate(certificate)
    numbers = parsed.public_key().public_numbers()
    modulus = numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8, "big")
    assert modulus in certificate, "the modulus must appear verbatim in the certificate"
    print("certificate: %d bytes, subject %s" % (len(certificate), parsed.subject.rfc4514_string()))
    print("key: %d bits, e = %d" % (parsed.public_key().key_size, numbers.e))
    print("validity: %s .. %s" % (parsed.not_valid_before_utc, parsed.not_valid_after_utc))
    print("  (both are in the past: this fixture cannot be checked against a clock)")

    # The 130 bytes of RFC 8446 section 4.4.3, rebuilt from the trace.
    hello = const("CLIENT_HELLO_HEX") + const("SERVER_HELLO_HEX")
    through_certificate = hashlib.sha256(hello + messages[0] + messages[1]).digest()
    signed = b"\x20" * 64 + b"TLS 1.3, server CertificateVerify" + b"\x00" + through_certificate
    assert len(signed) == 130

    verify = messages[2][4:]
    signature = verify[4:]
    assert verify[:2] == b"\x08\x04", verify[:2]
    parsed.public_key().verify(
        signature,
        signed,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32),
        hashes.SHA256(),
    )
    print("OpenSSL verifies the trace's own signature with a 32-byte salt")

    # And the structure our decoder walks, so a failure in the crate says which
    # step agreed and which did not.
    mod_bits = numbers.n.bit_length()
    em_len = (mod_bits - 1 + 7) // 8
    em = pow(int.from_bytes(signature, "big"), numbers.e, numbers.n).to_bytes(em_len, "big")
    db_len = em_len - 32 - 1
    masked_db, h = em[:db_len], em[db_len : db_len + 32]
    db = bytearray(a ^ b for a, b in zip(masked_db, mgf1(h, db_len)))
    unused = 8 * em_len - (mod_bits - 1)
    db[0] &= 0xFF >> unused
    db = bytes(db)
    separator = db_len - 32 - 1
    assert set(db[: separator - 1 + 1]) == {0} and db[separator] == 0x01
    print("encoding: modBits %d, emLen %d, k %d, %d unused bit(s)" % (mod_bits, em_len, len(numbers.n.to_bytes((mod_bits + 7) // 8, "big")), unused))
    print("          PS %d zeros, separator 0x01 at %d, salt %d bytes" % (separator, separator, 32))
    assert em[-1] == 0xBC
    assert h == hashlib.sha256(b"\x00" * 8 + hashlib.sha256(signed).digest() + db[separator + 1 :]).digest()

    print()
    print("values embedded in tests/rfc8448_flight.rs:")
    print("SERVER_MODULUS_HEX            = " + modulus.hex())
    print("HASH_THROUGH_CERTIFICATE_HEX  = " + through_certificate.hex())
    print("SIGNED_DATA_HASH_HEX          = " + hashlib.sha256(signed).hexdigest())
    print("SIGNATURE (from the trace)    = " + signature.hex())


if __name__ == "__main__":
    main()
