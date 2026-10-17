//! Embedded frontend assets for hearth-ui. Compiled into the binary; the
//! server never touches the disk to serve the UI.

pub const INDEX_HTML: &str = include_str!("index.html");

/// The app icon, taken from the same file the desktop entry and the macOS
/// bundle use, so the browser tab and the installed app cannot drift apart.
/// The path is relative to this file: four levels up to the repository root.
pub const FAVICON_SVG: &[u8] = include_bytes!("../../../../resources/hearth.svg");
