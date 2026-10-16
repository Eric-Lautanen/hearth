#!/usr/bin/env python3
"""Derive the P-256 (secp256r1 / prime256v1) constants for `hearth-tls/src/p256.rs`.

Nothing here is copied from a specification or from memory. The parameters are
read out of OpenSSL's own explicit-parameter dump — a second implementation — and
then *checked* as a group before they are printed as Rust:

  * `p` and `n` are prime (sympy, which shares no code with OpenSSL),
  * `a == p - 3`, which is how SEC2 defines this curve's `a`,
  * the generator `G` satisfies `y^2 == x^3 + a*x + b (mod p)`, which is the one
    equation a transcription error in `b`, `Gx` or `Gy` would break,
  * `n * G` is the identity — this is what pins the *order* against the
    *generator*, so a wrong `n` cannot pass,
  * `n` is in the Hasse interval `(p + 1 - 2*sqrt(p), p + 1 + 2*sqrt(p))`, the
    only check that says `n` is this curve's group order rather than some other
    prime of the right width,
  * `n != p`, so a reduction modulo one is not a reduction modulo the other, and
  * `p < 2n` and `n > 2^255`, which is what lets the Rust side reduce a 256-bit
    value with a single conditional subtraction.

The point multiplication used for the `n * G` check is plain Python integer
arithmetic over affine coordinates: the slowest and most obviously correct form,
deliberately independent of the Rust code it is checking.

Requires `pip install sympy` and an `openssl` on PATH. Run from the repository
root:

    python scripts/tls_p256_params.py
"""

import math
import re
import subprocess
import sys

import sympy


def openssl_params():
    """The explicit curve parameters, as OpenSSL prints them."""
    text = subprocess.run(
        ["openssl", "ecparam", "-name", "prime256v1", "-param_enc", "explicit", "-text", "-noout"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    values = {}
    # Walk the lines, tracking which field we are inside, and concatenate the
    # indented hex lines that follow its label.
    hex_lines = re.compile(r"^\s+([0-9a-f:]+)$")
    current = None
    for raw_line in text.splitlines():
        if raw_line.startswith("Prime:"):
            current = "p"
        elif re.match(r"^A:\s*$", raw_line):
            current = "a"
        elif raw_line.startswith("B:"):
            current = "b"
        elif raw_line.startswith("Generator"):
            current = "g"
        elif raw_line.startswith("Order:"):
            current = "n"
        elif raw_line.startswith("Cofactor:"):
            current = None
            values["h"] = int(re.search(r"([0-9]+)", raw_line).group(1))
        elif raw_line.startswith("Seed:"):
            current = None
        elif current is not None:
            match = hex_lines.match(raw_line)
            if match:
                values[current] = values.get(current, "") + match.group(1).replace(":", "")
    missing = {"p", "a", "b", "g", "n"} - values.keys()
    if missing:
        sys.exit("could not parse %s out of OpenSSL's output:\n%s" % (missing, text))
    point = bytes.fromhex(values["g"])
    assert point[0] == 0x04, "the generator should be an uncompressed point"
    return {
        "p": int(values["p"], 16),
        "a": int(values["a"], 16),
        "b": int(values["b"], 16),
        "gx": int.from_bytes(point[1:33], "big"),
        "gy": int.from_bytes(point[33:65], "big"),
        "n": int(values["n"], 16),
        "h": values["h"],
    }


# --- the independent reference used to check the group ----------------------
#
# Affine coordinates, one modular inverse per addition. Slow and obvious.


def point_add(curve, left, right):
    if left is None:
        return right
    if right is None:
        return left
    p, a = curve["p"], curve["a"]
    (x1, y1), (x2, y2) = left, right
    if x1 == x2:
        if (y1 + y2) % p == 0:
            return None
        slope = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:
        slope = (y2 - y1) * pow(x2 - x1, -1, p) % p
    x3 = (slope * slope - x1 - x2) % p
    return (x3, (slope * (x1 - x3) - y1) % p)


def point_mul(curve, scalar, point):
    """Double-and-add, most significant bit first."""
    result = None
    addend = point
    while scalar:
        if scalar & 1:
            result = point_add(curve, result, addend)
        addend = point_add(curve, addend, addend)
        scalar >>= 1
    return result


def on_curve(curve, point):
    if point is None:
        return True
    x, y = point
    p = curve["p"]
    return (y * y - (x * x * x + curve["a"] * x + curve["b"])) % p == 0


def limbs(value):
    """The 32-byte big-endian value as four little-endian 64-bit limbs."""
    return [value >> (64 * index) & (2**64 - 1) for index in range(4)]


def rust_array(name, value):
    rendered = ", ".join("0x%016x" % limb for limb in limbs(value))
    return "const %s: [u64; 4] = [%s];" % (name, rendered)


def check(curve):
    g = (curve["gx"], curve["gy"])
    n, p = curve["n"], curve["p"]

    assert curve["a"] == p - 3, "a is p - 3 on this curve"
    assert curve["h"] == 1, "the cofactor is 1, so n is the order of every point"
    assert on_curve(curve, g), "the generator is not on the curve"
    assert point_mul(curve, n, g) is None, "n * G is not the identity"
    # n must be the actual group order: by Hasse it lies within 2*sqrt(p) of
    # p + 1, and n being prime with n * G == identity then makes it *the* order.
    bound = 2 * math.isqrt(p) + 2
    assert abs(n - (p + 1)) <= bound, "n is not in the Hasse interval for this p"
    assert n != p, "the field characteristic and the group order must differ"
    assert p < 2 * n, "p < 2n, which is what makes a single conditional subtraction enough"
    assert n > 2**255, "n is wider than 255 bits, so n < 2^256 for every scalar"
    assert sympy.isprime(n), "n is prime"
    assert sympy.isprime(p), "p is prime"
    return g


def main():
    curve = openssl_params()
    g = check(curve)
    p, n = curve["p"], curve["n"]

    print("OpenSSL reports, and this script checked:")
    print("  p  = %064x  (prime, %d bits)" % (p, p.bit_length()))
    print("  a  = p - 3")
    print("  b  = %064x" % curve["b"])
    print("  Gx = %064x" % curve["gx"])
    print("  Gy = %064x" % curve["gy"])
    print("  n  = %064x  (prime, %d bits)" % (n, n.bit_length()))
    print("  G is on the curve, n * G is the identity, n is in Hasse(p), cofactor 1")
    print()
    print("// The field characteristic, the curve coefficient and the group order,")
    print("// little-endian 64-bit limbs. Derived from OpenSSL's explicit parameter")
    print("// dump by scripts/tls_p256_params.py, which checks them as a group before")
    print("// printing them: see that script for what is asserted and why.")
    print(rust_array("P", p))
    print(rust_array("B", curve["b"]))
    print(rust_array("GX", curve["gx"]))
    print(rust_array("GY", curve["gy"]))
    print(rust_array("N", n))
    print()
    print("// p - 2 and n - 2, the exponents of the Fermat inverses.")
    print(rust_array("P_MINUS_2", p - 2))
    print(rust_array("N_MINUS_2", n - 2))
    print()
    print("// -m^-1 mod 2^64 for both moduli, and 2^512 mod m: the two constants")
    print("// Montgomery multiplication needs. The Rust side derives these from the")
    print("// modulus itself, so their values here are only for the test that checks")
    print("// the derivation.")
    for name, modulus in (("P", p), ("N", n)):
        inv = pow(-modulus % 2**64, -1, 2**64)
        assert modulus & 1 == 1
        print("// %s_N0INV = 0x%016x" % (name, inv))
        print("// %s_R2 = %s" % (name, ", ".join("0x%016x" % limb for limb in limbs(pow(2, 512, modulus)))))
    assert g is not None


if __name__ == "__main__":
    main()
