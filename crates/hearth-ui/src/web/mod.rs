//! Embedded frontend assets for hearth-ui. Compiled into the binary; the
//! server never touches the disk to serve the UI.

pub const INDEX_HTML: &str = include_str!("index.html");
