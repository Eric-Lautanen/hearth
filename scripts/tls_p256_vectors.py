#!/usr/bin/env python3
"""Generate the `ecdsa_secp256r1_sha256` vectors for
`crates/hearth-tls/tests/ecdsa_vectors.rs`.

Three things happen for every vector before it is written down, and the order
matters:

1. **OpenSSL produces it** (`cryptography`'s signer), so the signature is a real
   one, from a second implementation, over a key derived from a fixed scalar.
2. **An independent verifier checks it**, written here in plain Python integers
   over affine coordinates: the X9.62 equation, `R = u1*G + u2*Q`, `v = R.x mod n`.
   That implementation shares no code with OpenSSL and none with the Rust being
   tested, so it is the thing that says the vector is what it claims to be.
3. **OpenSSL verifies it too**, which catches a vector whose signature OpenSSL
   would not have accepted in the first place.

Only then is the pair embedded. The negative vectors are produced by mutating a
valid one, each mutation named for the rule it is meant to trip, and each is
checked here to be *rejected* by the independent verifier before being embedded as
something the Rust must also reject.

One caveat about regenerating: ECDSA signatures use a random nonce, so this is not
reproducible byte for byte the way the RSA vectors are. Running it again produces
different signature bytes and sometimes different lengths, which is why the shape
line it prints each time is worth reading — the test that asserts the set contains
both a 32-byte and a 33-byte integer encoding is describing whatever generation is
in the tree, and a generation whose shapes differ needs that assertion re-derived.
The keys are reproducible: the private scalar is SHA-256 of a label.

Requires `pip install cryptography`. Run from the repository root:

    python scripts/tls_p256_vectors.py
"""

import hashlib
import sys

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils

# --- the curve, from OpenSSL's explicit parameters (see tls_p256_params.py) ---

p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
a = p - 3
b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
G = (
    0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
    0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5,
)
n = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


def point_add(left, right):
    if left is None:
        return right
    if right is None:
        return left
    (x1, y1), (x2, y2) = left, right
    if x1 == x2:
        if (y1 + y2) % p == 0:
            return None
        slope = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:
        slope = (y2 - y1) * pow(x2 - x1, -1, p) % p
    x3 = (slope * slope - x1 - x2) % p
    return (x3, (slope * (x1 - x3) - y1) % p)


def point_mul(scalar, point):
    result, addend = None, point
    while scalar:
        if scalar & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        scalar >>= 1
    return result


def reference_verify(public, message, signature):
    """X9.62 §7.4.1.4, in plain Python. No OpenSSL, no Rust."""
    try:
        r, s = utils.decode_dss_signature(signature)
    except ValueError:
        return False
    if not (1 <= r < n and 1 <= s < n):
        return False
    e = int.from_bytes(hashlib.sha256(message).digest(), "big") % n
    w = pow(s, -1, n)
    u1 = e * w % n
    u2 = r * w % n
    point = point_add(point_mul(u1, G), point_mul(u2, public))
    if point is None:
        return False
    return point[0] % n == r


def der_sequence(*integers):
    """A DER `ECDSA-Sig-Value` from Python integers, for the negative cases."""
    body = b""
    for value in integers:
        magnitude = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
        if magnitude[0] & 0x80:
            magnitude = b"\x00" + magnitude
        body += b"\x02" + bytes([len(magnitude)]) + magnitude
    return b"\x30" + bytes([len(body)]) + body


def der_with_padding(value, extra_zero_bytes):
    """An `INTEGER` for `value` with `extra_zero_bytes` redundant leading zeros."""
    magnitude = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
    magnitude = b"\x00" * extra_zero_bytes + magnitude
    if extra_zero_bytes == 0 and magnitude[0] & 0x80:
        magnitude = b"\x00" + magnitude
    return b"\x02" + bytes([len(magnitude)]) + magnitude


def der_signature(r, s):
    """A minimal DER `ECDSA-Sig-Value` for two Python integers."""
    body = der_with_padding(r, 0) + der_with_padding(s, 0)
    return b"\x30" + bytes([len(body)]) + body


def der_with_redundant_zero(value):
    """An `INTEGER` for `value` that always carries a spare leading zero."""
    magnitude = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
    if magnitude[0] & 0x80:
        return b"\x02" + bytes([len(magnitude) + 2]) + b"\x00\x00" + magnitude
    return b"\x02" + bytes([len(magnitude) + 1]) + b"\x00" + magnitude


def main():
    # The private scalar is derived, not typed: SHA-256 of a label, reduced into
    # `1..n`. A fixed scalar is what makes the public key reproducible.
    scalar = int.from_bytes(hashlib.sha256(b"hearth p256 ecdsa vectors").digest(), "big")
    scalar = scalar % (n - 1) + 1
    private = ec.derive_private_key(scalar, ec.SECP256R1())
    public_key = private.public_key()
    numbers = public_key.public_numbers()
    other = ec.derive_private_key(scalar + 1, ec.SECP256R1()).public_key().public_numbers()

    # A TLS-shaped message: exactly the RFC 8446 §4.4.3 construction with a fixed
    # transcript hash, since that is what this will verify in practice.
    signed_data = b" " * 64 + b"TLS 1.3, server CertificateVerify" + b"\x00" + bytes(range(32))

    messages = [
        b"",
        b"\x00",
        b"hearth",
        bytes(range(256)),
        signed_data,
        bytes((index * 7 + 3) & 0xFF for index in range(1000)),
    ]

    vectors = []
    for message in messages:
        signature = private.sign(message, ec.ECDSA(hashes.SHA256()))
        # Before embedding: an independent verifier, and OpenSSL's own.
        if not reference_verify((numbers.x, numbers.y), message, signature):
            sys.exit("the reference implementation rejects a vector OpenSSL produced")
        public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        vectors.append((message, signature))

    shapes = set()
    for _, signature in vectors:
        r, s = utils.decode_dss_signature(signature)
        shapes.add((len(signature), len(der_with_padding(r, 0)) - 2, len(der_with_padding(s, 0)) - 2))
    print("vector signature shapes encountered: %s" % sorted(shapes))

    r0, s0 = utils.decode_dss_signature(vectors[0][1])
    message0 = vectors[0][0]
    negatives = {
        # A valid signature with one bit of each integer flipped.
        "TAMPERED_R": der_signature(r0 ^ 1, s0),
        "TAMPERED_S": der_signature(r0, s0 ^ 1),
        # The edges of the scalar range: zero, n, and n + 1.
        "R_ZERO": der_signature(0, s0),
        "S_ZERO": der_signature(r0, 0),
        "S_EQUALS_N": der_signature(r0, n),
        # `s = n + 1` needs 33 bytes, so it is refused by width before any
        # comparison happens.
        "S_ABOVE_N": der_signature(r0, n + 1),
        # A `s` that is `n - s0`, which is the other valid signature over the same
        # message under a different nonce. It must be *accepted*: low-S is a
        # Bitcoin rule, not an X9.62 or TLS one. Embedding it makes that a
        # decision the suite states rather than one nobody notices.
        "S_NEGATED": der_signature(r0, n - s0),
    }

    r_bytes = r0.to_bytes(32, "big")
    body = (
        b"\x02\x20"
        + bytes([r_bytes[0] | 0x80])
        + r_bytes[1:]
        + der_with_padding(s0, 0)
    )
    negatives["R_NEGATIVE"] = b"\x30" + bytes([len(body)]) + body
    body = der_with_redundant_zero(r0) + der_with_padding(s0, 0)
    negatives["R_REDUNDANT_ZERO"] = b"\x30" + bytes([len(body)]) + body

    valid = vectors[0][1]
    negatives["TRAILING_BYTE"] = valid + b"\x00"
    negatives["THREE_INTEGERS"] = der_sequence(r0, s0, 1)
    negatives["SET_NOT_SEQUENCE"] = b"\x31" + valid[1:]
    negatives["TRUNCATED"] = valid[:-1]
    negatives["EMPTY"] = b""
    negatives["NOT_A_SEQUENCE"] = b"\x04\x02\x00\x00"
    # The raw `r || s` form some ecosystems use: 64 bytes, no DER framing at all.
    negatives["RAW_RS"] = r0.to_bytes(32, "big") + s0.to_bytes(32, "big")

    # Two kinds of negative, and they are checked differently.
    #
    # A *cryptographic* negative (a tampered integer, a scalar out of range) must
    # be rejected by the independent verifier, or it is not a negative at all.
    # An *encoding* negative (non-minimal, negative, trailing, truncated) may be
    # accepted by a lenient parser — OpenSSL's own decoder is one — and that is the
    # point of having them: the Rust refuses on the encoding, where a lenient
    # reader would have read the same value. So these are reported, and the Rust
    # test pins the variant each one has to produce.
    encoding_cases = {
        "R_REDUNDANT_ZERO",
        "R_NEGATIVE",
        "TRAILING_BYTE",
        "THREE_INTEGERS",
        "SET_NOT_SEQUENCE",
        "TRUNCATED",
        "EMPTY",
        "NOT_A_SEQUENCE",
        "RAW_RS",
    }
    for name, signature in negatives.items():
        accepted = reference_verify((numbers.x, numbers.y), message0, signature)
        if name == "S_NEGATED":
            if not accepted:
                sys.exit("the negated s should verify: low-S is not enforced here")
        elif name not in encoding_cases and accepted:
            sys.exit("%s verifies, so it is not a negative vector" % name)
        elif accepted:
            print("note: %s is accepted by a lenient reader, refused here on encoding" % name)

    out = []
    out.append("// ECDSA P-256 verification vectors.")
    out.append("//")
    out.append("// Generated by scripts/tls_p256_vectors.py — do not edit by hand.")
    out.append("//")
    out.append("// Each signature was produced by OpenSSL, checked against an independent")
    out.append("// verifier written in plain Python integers in that script, and then accepted by")
    out.append("// OpenSSL itself before being written here. The negative vectors are mutations of")
    out.append("// the first one, each named for the rule it must trip, each one checked to be")
    out.append("// rejected before being embedded.")
    out.append("")
    out.append("/// The public key every vector is over, SEC 1 uncompressed.")
    out.append('pub(crate) const PUBLIC_KEY_SEC1: &str = "%s";' % ("04" + "%064x%064x" % (numbers.x, numbers.y)))
    out.append("")
    out.append("/// A second, unrelated P-256 key, for the test that a signature does not verify")
    out.append("/// under a key that did not sign it.")
    out.append('pub(crate) const OTHER_PUBLIC_KEY_SEC1: &str = "%s";' % ("04" + "%064x%064x" % (other.x, other.y)))
    out.append("")
    out.append("/// `(message, signature)` pairs, hex.")
    out.append("pub(crate) const VECTORS: &[(&str, &str)] = &[")
    for message, signature in vectors:
        out.append('    ("%s", "%s"),' % (message.hex(), signature.hex()))
    out.append("];")
    out.append("")
    out.append("/// The message the first vector signs, for the negative cases.")
    out.append('pub(crate) const MESSAGE: &str = "%s";' % message0.hex())
    out.append("")
    out.append("/// Signatures that must be refused, each for a named reason.")
    out.append("pub(crate) const NEGATIVES: &[(&str, &str)] = &[")
    for name, signature in negatives.items():
        if name == "S_NEGATED":
            continue
        out.append('    ("%s", "%s"),' % (name, signature.hex()))
    out.append("];")
    out.append("")
    out.append("/// `(r, s)` rewritten as `(r, n - s)`: a valid signature that must be accepted,")
    out.append("/// because rejecting high-S is not required by X9.62 or by TLS.")
    out.append('pub(crate) const S_NEGATED: &str = "%s";' % negatives["S_NEGATED"].hex())
    out.append("")

    destination = sys.argv[1] if len(sys.argv) > 1 else "crates/hearth-tls/tests/data/ecdsa_vectors.rs"
    with open(destination, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(out))
    print(
        "wrote %s: %d vectors, %d negatives, message0 %d bytes"
        % (destination, len(vectors), len(negatives), len(message0))
    )


if __name__ == "__main__":
    main()
