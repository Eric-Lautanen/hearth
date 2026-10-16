#!/bin/sh
# Assemble a double-clickable Hearth.app from a release build (macOS only).
#
# Usage: ./scripts/package-macos.sh [app-name]   (default: Hearth.app)
#
# Expects: target/release/hearth-ui, target/release/hearth-chat-cli
# Produces: ./Hearth.app with both binaries, an empty models/ dir, README
# and icon. First launch on a fresh machine needs right-click -> Open until
# the binary is signed (Gatekeeper).
set -eu

APP="${1:-Hearth.app}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for bin in hearth-ui hearth-chat-cli; do
  if [ ! -x "$ROOT/target/release/$bin" ]; then
    echo "missing: cargo build --release -p hearth-ui (and the workspace) first" >&2
    exit 1
  fi
done

rm -rf "$ROOT/$APP"
mkdir -p "$ROOT/$APP/Contents/MacOS" "$ROOT/$APP/Contents/Resources"

cp "$ROOT/target/release/hearth-ui" "$ROOT/target/release/hearth-chat-cli" \
   "$ROOT/$APP/Contents/MacOS/"
chmod +x "$ROOT/$APP/Contents/MacOS/"*
cp "$ROOT/resources/Info.plist" "$ROOT/$APP/Contents/"
cp "$ROOT/resources/hearth.svg" "$ROOT/$APP/Contents/Resources/" 2>/dev/null || true
cp "$ROOT/README.md" "$ROOT/$APP/Contents/Resources/"
mkdir -p "$ROOT/$APP/Contents/Resources/models"

echo "built $ROOT/$APP"
