# Signing the release artifacts

The release workflow signs what it can and publishes the rest. **Nothing here is
required**: with no secrets set, the Windows executables and MSI are unsigned
and the macOS bundle is unsigned, the workflow says so in a `::notice::`, and
the release is exactly the one a fork would produce. With them set, the
platforms stop showing an unknown-publisher prompt.

Signing happens *before* packaging - the executables are signed first so the MSI
embeds signed copies, and the MSI is signed after WiX writes it.

## Windows (Authenticode)

| Secret | What it is |
|---|---|
| `WINDOWS_CERT_PFX_B64` | Base64 of a code-signing `.pfx` (EV or OV, includes the private key) |
| `WINDOWS_CERT_PASSWORD` | The `.pfx` password |

Without both, the Windows steps exit before touching anything.

## macOS (Developer ID + notarization)

| Secret | What it is |
|---|---|
| `MACOS_CERT_P12_B64` | Base64 of a Developer ID Application `.p12` |
| `MACOS_CERT_PASSWORD` | The `.p12` password |
| `MACOS_SIGN_IDENTITY` | e.g. `Developer ID Application: Name (TEAMID)` |
| `APPLE_ID` | Apple ID used for notarization |
| `APPLE_APP_PASSWORD` | An app-specific password for that Apple ID |
| `APPLE_TEAM_ID` | The 10-character team ID |

The bundle is signed with the hardened runtime (notarytool rejects it otherwise)
and a timestamp, notarized, stapled, and then asked the question Gatekeeper
asks: `spctl --assess`. Notarization is skipped, with a notice, when `APPLE_ID`
or `APPLE_APP_PASSWORD` is unset - the bundle is still signed, so it opens after
one right-click, Open.

## Verifying by hand

```powershell
# Windows: the signature and who it belongs to
Get-AuthenticodeSignature target/dist/hearth-v0.2.0-windows-x86_64.msi | Format-List
```

```sh
# macOS: signature, staple and Gatekeeper
codesign --verify --deep --strict --verbose=2 Hearth.app
xcrun stapler validate Hearth.app
spctl --assess --type execute --verbose=2 Hearth.app
```
