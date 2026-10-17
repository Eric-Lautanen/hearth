<#
.SYNOPSIS
  Stage the Windows payload that both Windows release artifacts are built from.

.DESCRIPTION
  One layout, two artifacts: the MSI's source directory (`packaging/windows/
  hearth.wxs`, whose PayloadDir points here) and the portable zip are the same
  bytes in the same arrangement, so a user who mixes them sees no difference
  beyond where the files landed.

  The layout is the app's own contract:

    hearth-ui.exe        the whole product: serves the UI and opens the browser
    hearth-chat-cli.exe  must sit beside hearth-ui.exe - the two-file layout
    models\README.txt    creates the folder the app resolves models from
    README.md            usage, including the terminal client
    LICENSE-MIT          license texts, so a copied folder stays self-describing
    LICENSE-APACHE

  curl is deliberately not shipped: Windows 10 1803 and later carry curl.exe in
  System32, and the app reports what to do if it cannot find one.

.EXAMPLE
  ./scripts/package-win.ps1 -Version 0.2.0 -Zip
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$PayloadDir = 'target/payload',
    [string]$OutDir = 'target/dist',
    [switch]$Zip
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$release = Join-Path $root 'target/release'
$payload = Join-Path $root $PayloadDir

if (Test-Path -LiteralPath $payload) {
    Remove-Item -LiteralPath $payload -Recurse -Force
}
New-Item -ItemType Directory -Force -Path (Join-Path $payload 'models') | Out-Null

foreach ($name in 'hearth-ui.exe', 'hearth-chat-cli.exe') {
    $source = Join-Path $release $name
    if (-not (Test-Path -LiteralPath $source)) {
        throw "missing $source - build it first: cargo build --release"
    }
    Copy-Item -LiteralPath $source -Destination $payload
}

foreach ($name in 'README.md', 'LICENSE-MIT', 'LICENSE-APACHE') {
    $source = Join-Path $root $name
    if (-not (Test-Path -LiteralPath $source)) {
        throw "missing $source"
    }
    Copy-Item -LiteralPath $source -Destination $payload
}

# The models note goes in as a file rather than being written here so the zip,
# the MSI and the Linux tarball all carry the same text. Its .txt name keeps the
# app's .gguf-only model scan from seeing it, and its presence is what makes the
# directory survive `Compress-Archive` (which drops empty directories).
$modelsNote = Join-Path $root 'resources/models-README.txt'
if (-not (Test-Path -LiteralPath $modelsNote)) {
    throw "missing $modelsNote"
}
Copy-Item -LiteralPath $modelsNote -Destination (Join-Path $payload 'models/README.txt')

Write-Host "payload staged in $payload"
Get-ChildItem -LiteralPath $payload -Recurse -File |
    ForEach-Object {
        $relative = $_.FullName.Substring($payload.Length).TrimStart('\', '/')
        "  {0,-28} {1,10:N0} bytes" -f $relative, $_.Length
    }

if ($Zip) {
    $out = Join-Path $root $OutDir
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    # Not $zip: PowerShell variable names are case-insensitive, so that would
    # overwrite the [switch]$Zip parameter with a string.
    $zipPath = Join-Path $out "hearth-v$Version-windows-x86_64.zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive -Path (Join-Path $payload '*') -DestinationPath $zipPath
    Write-Host "wrote $zipPath ($((Get-Item -LiteralPath $zipPath).Length) bytes)"
}
