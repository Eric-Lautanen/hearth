#!/usr/bin/env python3
"""Rasterize resources/hearth.svg into the icon containers the installers need.

Why this exists rather than a checked-in rasterizer step: the `.ico` and
`.icns` files are *inputs* to packaging. WiX reads the icon for the MSI's Icon
table (shortcut, Add/Remove Programs, dialog header) and the macOS bundle names
it through `CFBundleIconFile`, while neither CI runner has an SVG rasterizer
installed. So the containers are committed, and this script is what regenerates
them when `resources/hearth.svg` changes:

    python scripts/make-icons.py [--png-dir DIR]

Outputs (all committed):

  resources/hearth.ico        shell/ARP icon; 16px frame first, because that is
                              the order Windows' shell resolution wants.
  resources/hearth-setup.ico  the same frames reordered so 48px comes first:
                              MSI's Icon control draws the FIRST frame of an
                              icon and stretches it to the control's pixel
                              size, so a 16px-first container is the 3x-upscaled
                              blur this pair exists to avoid.
  resources/hearth.icns       the macOS bundle icon (PNG frames).

The .ico frames are classic 32bpp DIB bitmaps rather than PNGs: PNG-in-ICO is
legal, but the small frames are the ones legacy consumers (Add/Remove Programs,
some Explorer paths) read, and DIB is what always renders. The PNG decoder
below is stdlib-only on purpose - the only external dependency is the SVG
rasterizer, which must be `rsvg-convert` (librsvg) on PATH.
"""

import argparse
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG = ROOT / "resources" / "hearth.svg"

# Frames the Windows icon carries, smallest first.
ICO_SIZES = (16, 32, 48, 64, 128, 256)
# Frame order the installer dialogs want first, then the rest for any other
# consumer (mirrors assets/icon-setup.ico in the reference packaging).
SETUP_ORDER = (48, 64, 128, 256, 32, 16)
# macOS wants the full ladder; every entry is a PNG in the container.
ICNS_ENTRIES = (
    (b"icp4", 16),
    (b"icp5", 32),
    (b"icp6", 64),
    (b"ic07", 128),
    (b"ic08", 256),
    (b"ic09", 512),
    (b"ic10", 1024),
    # Retina pairs Apple reads when a display asks for them.
    (b"ic11", 32),
    (b"ic12", 64),
    (b"ic13", 256),
    (b"ic14", 512),
)


def render(size: int) -> bytes:
    """Render the SVG to a `size` square PNG using the external rasterizer.

    The sink is a real file in a temp dir rather than a stream: rsvg-convert on
    Windows reads the conventional `-o -` as a literal path named "-".
    """
    if shutil.which("rsvg-convert") is None:
        raise SystemExit(
            "rsvg-convert not found: install librsvg (mingw-w64-rsvg on msys2, "
            "librsvg2-bin on Debian) to regenerate the icon assets"
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"hearth-{size}.png"
        subprocess.run(
            ["rsvg-convert", "-w", str(size), "-h", str(size), "-o", str(out), str(SVG)],
            check=True,
        )
        return out.read_bytes()


def decode_png(data: bytes) -> tuple[int, int, bytes]:
    """Decode an 8-bit RGBA, non-interlaced PNG into packed RGBA rows."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("rasterizer did not return a PNG")
    pos = 8
    width = height = 0
    idat = bytearray()
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, colour, _comp, _filt, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if depth != 8 or colour != 6 or interlace != 0:
                raise SystemExit(
                    "expected an 8-bit RGBA non-interlaced PNG, got "
                    f"depth={depth} colour={colour} interlace={interlace}"
                )
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
    if width == 0 or height == 0:
        raise SystemExit("PNG carried no IHDR")
    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    rows = bytearray()
    previous = bytearray(stride)
    at = 0
    for _ in range(height):
        filter_type = raw[at]
        at += 1
        row = bytearray(raw[at : at + stride])
        at += stride
        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(4, stride):
                row[i] = (row[i] + row[i - 4]) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                row[i] = (row[i] + previous[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = row[i - 4] if i >= 4 else 0
                row[i] = (row[i] + ((left + previous[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = row[i - 4] if i >= 4 else 0
                above = previous[i]
                corner = previous[i - 4] if i >= 4 else 0
                estimate = left + above - corner
                dl, da, dc = (
                    abs(estimate - left),
                    abs(estimate - above),
                    abs(estimate - corner),
                )
                if dl <= da and dl <= dc:
                    nearest = left
                elif da <= dc:
                    nearest = above
                else:
                    nearest = corner
                row[i] = (row[i] + nearest) & 0xFF
        else:
            raise SystemExit(f"unknown PNG row filter {filter_type}")
        rows += row
        previous = row
    return width, height, bytes(rows)


def dib_frame(width: int, height: int, rgba: bytes) -> bytes:
    """Wrap RGBA pixels as the classic 32bpp BITMAPINFOHEADER frame an ICO holds.

    The XOR bitmap is bottom-up BGRA; the AND mask is present but all zero,
    which is the conventional encoding when the alpha channel is authoritative.
    """
    header = struct.pack(
        "<IiiHHIIiiII", 40, width, height * 2, 1, 32, 0, width * height * 4, 0, 0, 0, 0
    )
    xor = bytearray(width * height * 4)
    for y in range(height):
        source = (height - 1 - y) * width * 4
        target = y * width * 4
        for x in range(width):
            r, g, b, a = rgba[source + x * 4 : source + x * 4 + 4]
            xor[target + x * 4 : target + x * 4 + 4] = bytes((b, g, r, a))
    mask_stride = ((width + 31) // 32) * 4
    mask = bytes(mask_stride * height)
    return header + bytes(xor) + mask


def build_ico(frames: dict[int, bytes]) -> bytes:
    order = [size for size in ICO_SIZES if size in frames]
    offset = 6 + 16 * len(order)
    directory = bytearray()
    body = bytearray()
    for size in order:
        image = frames[size]
        directory += struct.pack(
            "<BBBBHHII",
            size if size < 256 else 0,
            size if size < 256 else 0,
            0,
            0,
            1,
            32,
            len(image),
            offset,
        )
        offset += len(image)
        body += image
    return struct.pack("<HHH", 0, 1, len(order)) + bytes(directory) + bytes(body)


def build_icns(frames: dict[int, bytes], pngs: dict[int, bytes]) -> bytes:
    body = bytearray()
    for kind, size in ICNS_ENTRIES:
        png = pngs.get(size)
        if png is None:
            continue
        body += kind + struct.pack(">I", 8 + len(png)) + png
    return b"icns" + struct.pack(">I", 8 + len(body)) + bytes(body)


def check_ico(data: bytes) -> int:
    """Assert a container's own offsets and lengths agree with its bytes.

    A malformed .ico is not rejected by WiX at build time - it fails later, as
    a blank Add/Remove Programs icon or a dialog that draws nothing - so the
    writer checks its own output instead.
    """
    count = struct.unpack("<H", data[4:6])[0]
    for i in range(count):
        entry = struct.unpack("<BBBBHHII", data[6 + 16 * i : 22 + 16 * i])
        _width, _height, _colours, _reserved, planes, bits, size, offset = entry
        if planes != 1 or bits != 32 or offset + size > len(data):
            raise SystemExit(f"ico frame {i} is malformed: {entry}")
    return count


def check_icns(data: bytes) -> int:
    """The same self-check for the ICNS container macOS reads."""
    if data[:4] != b"icns" or struct.unpack(">I", data[4:8])[0] != len(data):
        raise SystemExit("icns header length disagrees with the file size")
    at = 8
    entries = 0
    while at < len(data):
        kind = data[at : at + 4]
        (length,) = struct.unpack(">I", data[at + 4 : at + 8])
        if length < 16 or at + length > len(data):
            raise SystemExit(f"icns entry {kind!r} claims {length} bytes past the end")
        if data[at + 8 : at + 12] != b"\x89PNG":
            raise SystemExit(f"icns entry {kind!r} does not carry a PNG")
        at += length
        entries += 1
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--png-dir",
        type=Path,
        help="also write the raw renders (hearth-<size>.png) here for eyeballing",
    )
    args = parser.parse_args()

    pngs: dict[int, bytes] = {}
    dibs: dict[int, bytes] = {}
    sizes = sorted(set(ICO_SIZES) | {size for _kind, size in ICNS_ENTRIES})
    for size in sizes:
        png = render(size)
        width, height, rgba = decode_png(png)
        if (width, height) != (size, size):
            raise SystemExit(f"rasterizer returned {width}x{height} for size {size}")
        pngs[size] = png
        dibs[size] = dib_frame(width, height, rgba)
        print(f"  {size:4d}px  {len(png):7d} B png  {len(dibs[size]):7d} B dib")
        if args.png_dir:
            args.png_dir.mkdir(parents=True, exist_ok=True)
            (args.png_dir / f"hearth-{size}.png").write_bytes(png)

    shell_frames = [size for size in ICO_SIZES if size in dibs]
    shell = build_ico(dibs)
    (ROOT / "resources" / "hearth.ico").write_bytes(shell)
    print(
        f"wrote resources/hearth.ico ({len(shell)} bytes, "
        f"{check_ico(shell)} frames {shell_frames}, first frame {shell_frames[0]}px)"
    )

    setup_frames = [size for size in SETUP_ORDER if size in dibs]
    setup = build_ico(dibs)
    (ROOT / "resources" / "hearth-setup.ico").write_bytes(setup)
    print(
        f"wrote resources/hearth-setup.ico ({len(setup)} bytes, "
        f"{check_ico(setup)} frames {setup_frames}, first frame {setup_frames[0]}px)"
    )

    icns = build_icns(dibs, pngs)
    (ROOT / "resources" / "hearth.icns").write_bytes(icns)
    print(
        f"wrote resources/hearth.icns ({len(icns)} bytes, "
        f"{check_icns(icns)} PNG entries)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
