#!/usr/bin/env python3
"""Dependency-free GGUF metadata + tensor inspector.

Reads only the header/metadata/tensor-info section (never the tensor data), so it
is instant even on multi-GB files. Used to verify that a new Prism GGUF carries
the architecture keys and quantization types Hearth needs.

Usage:  python scripts/gguf_info.py models/Foo.gguf [--tensors] [--meta KEY]
"""
import sys
import struct

DTYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4",
    41: "Q1_0", 42: "Q2_0", 43: "Q1_0_G128", 142: "PQ2_0", 143: "PTQ1_0",
}

# types whose block layout we know, as (elements_per_block, bytes_per_block)
BLOCK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24),
    "Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144),
    "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "Q1_0": (128, 18), "Q1_0_G128": (128, 18), "Q2_0": (128, 34),
    "PQ2_0": (128, 34), "PTQ1_0": (128, 28), "I8": (1, 1), "I16": (1, 2),
    "I32": (1, 4), "I64": (1, 8),
}

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, \
    T_U64, T_I64, T_F64 = range(13)


def read(r, fmt, n=None):
    size = struct.calcsize(fmt)
    data = r.read(size * (n or 1))
    if len(data) != size * (n or 1):
        raise EOFError("truncated")
    out = struct.unpack("<" + fmt * (n or 1), data)
    return out if n else out[0]


def read_str(r):
    n = read(r, "Q")
    return r.read(n).decode("utf-8", "replace")


def read_value(r, t):
    if t == T_U8:
        return read(r, "B")
    if t == T_I8:
        return read(r, "b")
    if t == T_U16:
        return read(r, "H")
    if t == T_I16:
        return read(r, "h")
    if t == T_U32:
        return read(r, "I")
    if t == T_I32:
        return read(r, "i")
    if t == T_F32:
        return read(r, "f")
    if t == T_BOOL:
        return bool(read(r, "B"))
    if t == T_STR:
        return read_str(r)
    if t == T_U64:
        return read(r, "Q")
    if t == T_I64:
        return read(r, "q")
    if t == T_F64:
        return read(r, "d")
    if t == T_ARR:
        et = read(r, "I")
        n = read(r, "Q")
        # Long arrays of strings (the vocab) are summarized, not materialized.
        if et == T_STR and n > 8:
            vals = [read_str(r) for _ in range(n)]
            return f"<string[{n}]> {vals[:4]} ..."
        if et in (T_U32, T_I32, T_F32) and n > 16:
            vals = [read_value(r, et) for _ in range(n)]
            return f"<type{et}[{n}]> {vals[:8]} ..."
        return [read_value(r, et) for _ in range(n)]
    raise ValueError(f"unknown gguf value type {t}")


def block_bytes(dtype, shape):
    """Byte size of a tensor with `shape` (nearest dim first) in `dtype`."""
    name = DTYPES.get(dtype, str(dtype))
    if name not in BLOCK:
        return None
    per, bpb = BLOCK[name]
    if name in ("F32", "F16", "BF16", "I8", "I16", "I32", "I64"):
        n = 1
        for d in shape:
            n *= d
        return n * bpb
    inner = shape[-1]
    if inner % per:
        return None
    rows = 1
    for d in shape[:-1]:
        rows *= d
    return rows * (inner // per) * bpb


def main():
    path = sys.argv[1]
    show_tensors = "--tensors" in sys.argv
    only = None
    if "--meta" in sys.argv:
        only = sys.argv[sys.argv.index("--meta") + 1]

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            sys.exit(f"not a GGUF file (magic {magic!r})")
        version = read(f, "I")
        n_tensors = read(f, "Q")
        n_kv = read(f, "Q")
        print(f"# {path}")
        print(f"gguf_version={version} tensors={n_tensors} kv={n_kv}")

        meta = {}
        for _ in range(n_kv):
            key = read_str(f)
            t = read(f, "I")
            val = read_value(f, t)
            meta[key] = val
            if only is None or only in key:
                if isinstance(val, str) and len(val) > 220:
                    print(f"{key} = <str len {len(val)}> {val[:220]!r}...")
                else:
                    print(f"{key} = {val}")

        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd = read(f, "I")
            dims = [read(f, "Q") for _ in range(nd)]
            dtype = read(f, "I")
            off = read(f, "Q")
            tensors.append((name, dims, dtype, off))

        if show_tensors:
            by_dtype = {}
            total = 0
            for name, dims, dtype, off in tensors:
                dn = DTYPES.get(dtype, f"type{dtype}")
                b = block_bytes(dtype, dims)
                by_dtype[dn] = by_dtype.get(dn, 0) + (b or 0)
                total += b or 0
                if not name.startswith("blk."):
                    print(f"tensor {name} {dn} {dims} bytes={b}")
            print("--- dtype totals ---")
            for dn, b in sorted(by_dtype.items(), key=lambda kv: -kv[1]):
                print(f"{dn:10s} {b/1e6:10.1f} MB  {100.0*b/max(total,1):5.1f}%")
            print(f"TOTAL      {total/1e6:10.1f} MB")


if __name__ == "__main__":
    main()
