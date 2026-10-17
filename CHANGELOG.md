# Changelog

Releases are built from a tag by `.github/workflows/release.yml`, which produces
one installer or archive per platform (Windows zip + MSI, Linux tarball, macOS
`.app`) and publishes them with `SHA256SUMS`. `0.x` means the CLI flags and the
HTTP API can still move between minor versions.

## [0.2.0] - 2026-10-17

The release machinery, the Windows installer, and a hardening pass over the
quant kernels, the TLS client and the local HTTP surface.

### Added

- **Windows MSI installer** (`packaging/windows/hearth.wxs`, WiX 7): per-user, so
  it installs to `%LOCALAPPDATA%\Programs\Hearth` and uninstalls with no UAC
  prompt, with a two-page wizard whose checkboxes choose a Desktop shortcut and
  whether Hearth opens when setup finishes. Carries a Start-menu folder, an
  Add/Remove Programs entry with icon and publisher URL, and an upgrade identity
  that replaces a newer install rather than stacking beside it.
- **Release workflow** (`.github/workflows/release.yml`): five runners
  (Windows, Linux x86_64 and arm64, macOS x86_64 and arm64) build, smoke-test
  the binary on each, package, sign when signing secrets exist, and publish to a
  tagged GitHub release together with `SHA256SUMS`. A `workflow_dispatch` run
  builds and uploads exactly the same artifacts and stops before publishing.
- **macOS bundle and Linux archive**: `scripts/package-macos.sh` assembles
  `Hearth.app` (version-stamped `Info.plist`, `.icns`, `plutil`-checked); the
  Linux tarball ships `hearth.desktop` and the icon beside the binaries.
- **Packaging verification**: `scripts/msi-query.ps1` prints the MSI tables and
  runs fourteen checks against them, exiting non-zero when one fails; the
  packaging README records the install/uninstall matrix with measured results.
- **Icons**: `resources/hearth.ico`, `hearth-setup.ico` and `hearth.icns`,
  generated from `resources/hearth.svg` by `scripts/make-icons.py`.
- **Hand-rolled TLS**: DER and X.509 reading, chain building, name matching, leaf
  purpose, `basicConstraints`/`pathLenConstraint`/`keyCertSign`, P-256/ECDSA and
  RSA (PSS and PKCS#1 v1.5), caller-supplied CRLs, and key rollover — each rule
  proven by a test that removes it.
- **Local HTTP hardening**: a loopback host gate, a foreign-origin block, and an
  opt-in API token.

### Changed

- `hearth-ui` is a **GUI-subsystem binary on Windows** and gives every child it
  spawns (`CREATE_NO_WINDOW`) no console window: neither it nor the engine
  service flashes a window, while a Terminal launch still shows logs.
- x86_64 release builds pin `x86-64-v3`, and every `#[target_feature]` kernel
  body declares the features it uses, so dispatched kernels inline instead of
  running as per-instruction calls.
- Q2_0 dispatches the AVX2 LUT kernel first; the Q1_0 VNNI kernel is dispatched
  at all (it had only ever been declared); the dense-prefill worker count is
  machine-derived and shareable with the tests.
- Version bumped to 0.2.0 across the workspace and the macOS `Info.plist`, so
  artifacts and the installer can no longer disagree about what they are.

### Fixed

- A kernel that reached for `fma`/`f16c` without declaring them ran ~5.7x slower
  on the 1.7B Q1_0 dot path while the dispatch banner still reported `avx2`.
- Every `#[allow(...)]` in the workspace, each by fixing its cause.
- The MSI's `models/` note installed beside the executables instead of in
  `models/` (a `File` lands in its component's directory).
- The batched-prefill "cliff" reported earlier was mostly ambient machine load,
  not code; the measurement and the doc were corrected.
- Setups built without a modern installer version silently failed at launch
  conditions on this machine; the floor is now declared and enforced.

### Performance

Measured one model at a time on the Ryzen 7 8840HS test box, 50 tokens, temp 0:
1.7B Q1_0 46.7, 1.7B Q2_0 23.6, 4B Q1_0 23.6, 4B Q2_0 12.2, 8B Q1_0 13.0,
8B Q2_0 6.2 tok/s. `docs/BUG_TRACKER.md` has the before/after for each change,
including the ones that were reverted.

## [0.1.0] - 2026-10-16

First tagged build: the Bonsai model family running on CPU (`Q1_0`, `Q2_0`), the
web chat UI with its model downloader, the terminal client, and the workspace
breakdown into `hearth-core`, `hearth-gguf`, `hearth-quant`, `hearth-tokenizer`,
`hearth-sampler`, `hearth-llm`, `hearth-compute`, `hearth-diffusion` and
`hearth-ui`.

[0.2.0]: https://github.com/Eric-Lautanen/hearth/compare/v0.1.0...v0.2.0
