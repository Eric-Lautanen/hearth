"""Convert Gemlite INT2 state_dict.pt → Hearth Q2_0 binary format.

Gemlite format (129-element group, INT2 packed, transposed):
  W_q:      [in_dim//4, out_dim] uint8  — 4 input values packed per byte
  scales:   [in_dim//128, out_dim] f32   — one scale per 128-input group
  zeros:    [in_dim//128, out_dim] f32   — zero points (unused for ternary)
  orig_shape: [out_dim, in_dim]          — original PyTorch weight shape

Q2_0 format (128-element block, row-major):
  Per row: for each block of 128 input elements:
    [u8×2] f16 scale
    [u8×32] packed INT2 (4 values/byte, 32 bytes = 128 values)

Mapping: Gemlite groups 128 input elements → Q2_0 blocks of 128 consecutive elements.
Conversion: transpose W_q and scales to row-major Q2_0 layout.

Usage: python scripts/convert_bonsai_image.py
"""

import os, struct, json
import numpy as np
import torch

MODEL_DIR = os.path.expandvars(
    r"%USERPROFILE%\AppData\Roaming\hearth\models\Bonsai-Image-Ternary-4B-Gemlite\transformer-gemlite-int2"
)
OUT_PATH = os.path.expandvars(
    r"%USERPROFILE%\AppData\Roaming\hearth\models\Bonsai-Image-4B-Q2_0.bin"
)

Q2_0_BLOCK = 34
GROUP_SIZE = 128
BYTES_PER_GROUP = 32  # 128 values / 4 per byte


def load_state_dict():
    path = os.path.join(MODEL_DIR, "state_dict.pt")
    print(f"Loading {path} ({os.path.getsize(path)/1e9:.2f} GB)...")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    print(f"  {len(sd)} tensor entries")
    return sd


def reshape_for_orig(t, orig_shape):
    """Gemlite stores some tensors with extra padding. Reshape if needed."""
    expected = orig_shape[0] * orig_shape[1]
    if t.numel() == expected:
        return t.reshape(orig_shape[0], orig_shape[1]).numpy()
    # Padded to next multiple
    return t.numpy()


def convert_one_tensor(name, W_q, scales, orig_shape, f, counter):
    """Convert one Gemlite-packed tensor to Q2_0 and write to file."""
    out_dim, in_dim = int(orig_shape[0]), int(orig_shape[1])
    num_blocks = in_dim // GROUP_SIZE

    # W_q: [in_dim//4, out_dim] uint8
    # scales: [in_dim//128, out_dim] f32
    w_packed = W_q.numpy()       # [in_dim//4, out_dim]
    s = scales.numpy()           # [in_dim//128, out_dim]

    q2_data = bytearray()
    total_elements = 0

    for row in range(out_dim):
        for blk in range(num_blocks):
            # Q2_0 block: 2-byte f16 scale + 32-byte packed
            block = bytearray(Q2_0_BLOCK)

            # Scale: convert f32 → f16
            scale_f32 = float(s[blk, row])
            scale_f16 = struct.pack("<e", scale_f32)
            block[0:2] = scale_f16

            # Packed values: transpose one column slice from W_q
            # W_q[blk*32 : blk*32+32, row] → 32 bytes
            col_start = blk * BYTES_PER_GROUP
            col_end = col_start + BYTES_PER_GROUP
            block[2:34] = w_packed[col_start:col_end, row].tobytes()

            q2_data.extend(block)
            total_elements += GROUP_SIZE

    expected_elements = out_dim * in_dim
    assert total_elements == expected_elements, \
        f"{name}: expected {expected_elements} elements, got {total_elements}"

    expected_bytes = out_dim * num_blocks * Q2_0_BLOCK
    assert len(q2_data) == expected_bytes, \
        f"{name}: expected {expected_bytes} Q2_0 bytes, got {len(q2_data)}"

    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<H", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<QQ", out_dim, in_dim))
    f.write(struct.pack("<I", len(q2_data)))
    f.write(q2_data)

    mb = len(q2_data) / 1e6
    print(f"  [{counter:3d}] Q2_0  {name:55s}  [{out_dim:6d} × {in_dim:6d}]  {mb:7.2f} MB  {num_blocks * out_dim} blocks")
    return len(q2_data)


def convert_skipped(name, tensor, f, counter):
    """Write a non-quantized (BF16/FP32) tensor."""
    if tensor.ndim < 2:
        return 0  # skip 1D tensors (biases, norms)

    out_dim, in_dim = int(tensor.shape[0]), int(tensor.shape[1])

    if tensor.dtype == torch.bfloat16:
        # Store raw BF16 bytes as uint16
        raw = tensor.view(torch.uint16).numpy()
        align = raw.nbytes
        data = raw.tobytes()
        dtype = 1
    elif tensor.dtype == torch.float32:
        dtype = 0  # FP32
        data = tensor.numpy().tobytes()
    elif tensor.dtype == torch.uint8:
        return 0  # skip packed tensors
    else:
        dtype = 0
        data = tensor.float().numpy().tobytes()

    name_bytes = name.encode("utf-8")
    f.write(struct.pack("<H", len(name_bytes)))
    f.write(name_bytes)
    f.write(struct.pack("<QQ", out_dim, in_dim))
    f.write(struct.pack("<B", dtype))
    f.write(struct.pack("<I", len(data)))
    f.write(data)

    print(f"  [{counter:3d}] SKIP  {name:55s}  [{out_dim:6d} × {in_dim:6d}]  {len(data)/1e6:7.2f} MB  {tensor.dtype}")
    return len(data)


def main():
    with open(os.path.join(MODEL_DIR, "quantization_config.json")) as fh:
        qcfg = json.load(fh)

    quant_fqns = qcfg["quantized_fqns"]
    quant_set = set(quant_fqns)
    print(f"Quantized layers: {len(quant_fqns)}")

    sd = load_state_dict()

    # Build name → (W_q, scales, orig_shape) map for quantized tensors
    quant_tensors = {}
    for fqn in quant_fqns:
        w_key = f"{fqn}.W_q"
        s_key = f"{fqn}.scales"
        o_key = f"{fqn}.orig_shape"
        if w_key in sd and s_key in sd and o_key in sd:
            quant_tensors[fqn] = (sd[w_key], sd[s_key], sd[o_key])
        else:
            print(f"  WARNING: missing keys for {fqn}")

    # Build skipped tensor list
    skipped = {}
    weight_names = {k.replace(".weight", "") for k in sd if k.endswith(".weight")}
    skipped_names = sorted(weight_names - quant_set)
    for name in skipped_names:
        wk = f"{name}.weight"
        if wk in sd and sd[wk].ndim >= 2:
            skipped[name] = sd[wk]

    print(f"\nWriting {OUT_PATH}...")
    with open(OUT_PATH, "wb") as f:
        # Header
        f.write(struct.pack("<I", 0x30513248))  # magic
        f.write(struct.pack("<I", len(quant_tensors)))

        total_q2 = 0
        for i, (name, (wq, sc, og)) in enumerate(quant_tensors.items(), 1):
            total_q2 += convert_one_tensor(name, wq, sc, og, f, i)

        f.write(struct.pack("<I", len(skipped)))
        total_skip = 0
        for i, (name, tensor) in enumerate(skipped.items(), len(quant_tensors) + 1):
            total_skip += convert_skipped(name, tensor, f, i)

        f.write(struct.pack("<I", 0x30513248))  # end magic
        file_size = f.tell()

    print(f"\n{'='*70}")
    print(f"Output: {OUT_PATH}")
    print(f"  Quantized: {len(quant_tensors)} tensors, {total_q2/1e6:.1f} MB")
    print(f"  Skipped:   {len(skipped)} tensors, {total_skip/1e6:.1f} MB")
    print(f"  Total:     {file_size/1e6:.1f} MB")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
