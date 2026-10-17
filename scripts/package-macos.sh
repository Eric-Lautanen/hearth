#!/bin/sh
# Assemble a double-clickable Hearth.app from a release build (macOS only).
#
# Usage: ./scripts/package-macos.sh [app-path] [version]
#
#   app-path  Where to write the bundle: `Hearth.app`, `target/Hearth.app` or
#             an absolute path. Default: `Hearth.app` in the repo root.
#   version   The version the bundle reports. Default: whatever the committed
#             Info.plist says; the release workflow passes the tag's version
#             ("0.2.0") so a bundle is never stamped with a stale number.
#
# Expects: target/release/hearth-ui, target/release/hearth-chat-cli
# Produces: a bundle holding both binaries, an empty `models/` dir, the README
# and licenses, the icon and Info.plist. First launch on a fresh machine needs
# right-click -> Open until the binaries are signed (Gatekeeper).
set -eu

APP_ARG="${1:-Hearth.app}"
VERSION="${2:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

case "$APP_ARG" in
  /*) APP="$APP_ARG" ;;
  *) APP="$ROOT/$APP_ARG" ;;
esac

for bin in hearth-ui hearth-chat-cli; do
  if [ ! -x "$ROOT/target/release/$bin" ]; then
    echo "missing: cargo build --release -p hearth-ui (and the workspace) first" >&2
    exit 1
  fi
done

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$ROOT/target/release/hearth-ui" "$ROOT/target/release/hearth-chat-cli" \
   "$APP/Contents/MacOS/"
chmod +x "$APP/Contents/MacOS/"*

# Stamp the two version keys - and only those - by rewriting the *line after*
# each key. `sed -i` is not portable between BSD and GNU, so the result is
# written to the bundle rather than edited in place in the repo.
if [ -n "$VERSION" ]; then
  sed -e '/<key>CFBundleVersion<\/key>/{n;s|<string>.*</string>|<string>'"$VERSION"'</string>|;}' \
      -e '/<key>CFBundleShortVersionString<\/key>/{n;s|<string>.*</string>|<string>'"$VERSION"'</string>|;}' \
      "$ROOT/resources/Info.plist" > "$APP/Contents/Info.plist"
else
  cp "$ROOT/resources/Info.plist" "$APP/Contents/"
fi

# The .icns is committed (scripts/make-icons.py regenerates it from the SVG when
# the artwork changes); CFBundleIconFile names it, so without this copy Finder
# and the Dock fall back to the generic application icon.
cp "$ROOT/resources/hearth.icns" "$APP/Contents/Resources/"
cp "$ROOT/resources/hearth.svg" "$APP/Contents/Resources/"
cp "$ROOT/README.md" "$ROOT/LICENSE-MIT" "$ROOT/LICENSE-APACHE" \
   "$APP/Contents/Resources/"
mkdir -p "$APP/Contents/Resources/models"

# The stamp above is a sed edit, so the plist is validated on the platform that
# reads it rather than assumed good.
if command -v plutil >/dev/null 2>&1; then
  plutil -lint "$APP/Contents/Info.plist"
fi

echo "built $APP"
