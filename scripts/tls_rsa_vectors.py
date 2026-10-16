#!/usr/bin/env python3
"""Regenerate the constants in `crates/hearth-tls/tests/rsa_vectors.rs`.

The provenance for that file. Everything it produces comes from an implementation
that is not ours, which is the point:

  * the modular-exponentiation vectors (`s^e mod n`) come from CPython's own
    arbitrary-precision `pow`;
  * the PSS vectors come from OpenSSL through `cryptography`, which picks the
    salt, masks the encoding and assembles `EM` itself;
  * the primes for the 1025-bit key come from `sympy.randprime`, because
    OpenSSL's key generator always sets the top bit and so never produces a
    modulus whose width is 1 mod 8 — the case where `emLen` is one byte shorter
    than `k`, which a 2048-bit key cannot exercise.

**The modpow vectors are reproducible exactly** (the RNG is seeded). **The PSS
vectors are not**: OpenSSL seeds itself from the OS, so re-running this prints a
different — equally valid — key and signature. A diff there is expected and is
not evidence of a regression; what matters is that the crate accepts the values it
was given and refuses everything else.

Requires `pip install cryptography sympy`. Run from the repository root:

    python scripts/tls_rsa_vectors.py
"""
import random
import sys

import sympy
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateNumbers, RSAPublicNumbers

random.seed(20261016)


def print_constant(name, value, width=32):
    text = value.hex()
    print('const %s_HEX: &str = "\\' % name)
    for i in range(0, len(text), width):
        print("    %s \\" % text[i : i + width])
    print('";')


def odd_modulus(bits):
    """A random odd modulus of exactly `bits` bits, which is all RSAVP1 needs.

    It is not a product of two primes — the public-key operation cannot tell, and
    pretending otherwise would mean factoring, which is the point of RSA.
    """
    for _ in range(100):
        candidate = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if candidate % 3 != 0:
            return candidate
    raise AssertionError("no modulus in 100 tries")


def emit_modpow(label, bits, exponent):
    modulus = odd_modulus(bits)
    signature = random.randrange(2, modulus)
    result = pow(signature, exponent, modulus)
    em_len = ((modulus.bit_length() - 1) + 7) // 8
    assert result < 1 << (8 * em_len), "the result must fit emLen"
    for suffix, value, size in (
        ("N", modulus, (bits + 7) // 8),
        ("S", signature, (bits + 7) // 8),
        ("M", result, em_len),
    ):
        print_constant("%s_%s" % (label, suffix), value.to_bytes(size, "big"))
    print(
        "# %s: bits=%d limbs=%d exponent=%d em_len=%d\n"
        % (label, bits, (bits + 63) // 64, exponent, em_len)
    )


def generated(bits):
    for _ in range(20):
        key = rsa.generate_private_key(public_exponent=65537, key_size=bits)
        if key.public_key().public_numbers().n.bit_length() == bits:
            return key
    raise AssertionError("OpenSSL would not give a %d-bit modulus" % bits)


def emit_pss(label, key, message, salt_length, bits):
    numbers = key.public_key().public_numbers()
    assert numbers.n.bit_length() == bits, numbers.n.bit_length()
    signature = key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=salt_length),
        hashes.SHA256(),
    )
    n_bytes = (bits + 7) // 8
    em_len = ((bits - 1) + 7) // 8
    print_constant(label + "_N", numbers.n.to_bytes(n_bytes, "big"))
    print_constant(label + "_SIG", signature)
    print(
        "# %s: bits=%d k=%d em_len=%d salt=%d message=%r\n"
        % (label, bits, n_bytes, em_len, salt_length, message)
    )


def odd_sized_key(bits):
    """A modulus of `bits` bits from two primes, for widths OpenSSL will not make."""
    # Two primes of ceil(bits / 2) bits multiply to `bits` or `bits + 1` bits. For
    # 1025 that is two 513-bit primes and a product of 1025 or 1026, of which only
    # the 1025-bit case is wanted -- which is exactly the width an OpenSSL key
    # cannot have, since its generator always sets the top bit.
    half = (bits + 1) // 2
    for _ in range(40):
        p = int(sympy.randprime(1 << (half - 1), 1 << half))
        q = int(sympy.randprime(1 << (half - 1), 1 << half))
        if p == q or (p * q).bit_length() != bits:
            continue
        if p < q:
            p, q = q, p
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
    raise AssertionError("no %d-bit modulus in 40 tries" % bits)


def main():
    # RSAVP1 at the limb counts that matter: 4, 8, 9, 16, 24, 32.
    emit_modpow("RSAVP1_256", 256, 65537)
    emit_modpow("RSAVP1_512", 512, 65537)
    emit_modpow("RSAVP1_576", 576, 65537)
    emit_modpow("RSAVP1_1024", 1024, 65537)
    emit_modpow("RSAVP1_1536", 1536, 3)
    emit_modpow("RSAVP1_2048", 2048, 65537)

    # PSS: emLen == k, then emLen == k - 1, then a salt TLS 1.3 forbids.
    emit_pss(
        "PSS_2048",
        generated(2048),
        b"hearth tls 1.3 certificate verify, 2048-bit key",
        32,
        2048,
    )
    emit_pss(
        "PSS_1025",
        odd_sized_key(1025),
        b"hearth tls 1.3 certificate verify, 1025-bit key",
        32,
        1025,
    )
    emit_pss(
        "PSS_2048_SHORT_SALT",
        generated(2048),
        b"hearth tls 1.3 certificate verify, salt 20",
        20,
        2048,
    )
    print("# 24 constants. Copy the block above into tests/rsa_vectors.rs,", file=sys.stderr)
    print("# merging by name; the comments record what each vector is.", file=sys.stderr)


if __name__ == "__main__":
    main()
