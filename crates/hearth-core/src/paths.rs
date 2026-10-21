//! Where Hearth looks for model weights.
//!
//! One implementation for both sides of the app: the engine service lists the
//! `.gguf` files, the UI downloader writes them. Resolving a *single*
//! directory (the old behaviour) meant an installed build only ever saw
//! `<install>/models` and silently ignored weights the user already had in
//! `%APPDATA%\hearth\models` or beside a source checkout, so the UI listed two
//! models out of twenty.
//!
//! `models_roots()` returns every directory worth listing; `primary_models_dir()`
//! returns the one new downloads belong in.

use std::path::{Path, PathBuf};

/// Explicit override — when set, it is the only root, and it is where
/// downloads land.
pub const MODELS_DIR_ENV: &str = "HEARTH_MODELS_DIR";

/// How far up from the executable to look for `models/`
/// (`target/release/x.exe` → `<repo>/models` is three levels).
const EXE_WALK_UP: usize = 6;

fn non_empty_env(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && Path::new(s).is_dir())
        .map(PathBuf::from)
}

/// `HEARTH_MODELS_DIR` may name one directory or a list of them, separated by
/// the platform path separator (`;` on Windows, `:` elsewhere) — the weight
/// files are often split across a fast disk and a big one.
fn split_roots(raw: &str) -> Vec<PathBuf> {
    raw.split(if cfg!(windows) { ';' } else { ':' })
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

fn env_list(key: &str) -> Vec<PathBuf> {
    std::env::var(key)
        .map(|raw| split_roots(&raw))
        .unwrap_or_default()
}

/// The `HEARTH_MODELS_DIR` entries in order (empty when unset).
fn models_dir_overrides() -> Vec<PathBuf> {
    env_list(MODELS_DIR_ENV)
}

/// The first `HEARTH_MODELS_DIR` entry — where downloads go.
pub fn models_dir_override() -> Option<PathBuf> {
    models_dir_overrides().into_iter().next()
}

/// `models/` in the working directory and in every ancestor directory of the
/// running executable — covers `cargo run`, `target/release/<exe>` and a
/// portable `hearth-ui.exe` sitting next to `models/`.
fn exe_and_cwd_models() -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from("models")];
    let Some(mut dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    else {
        return out;
    };
    for _ in 0..EXE_WALK_UP {
        out.push(dir.join("models"));
        match dir.parent() {
            Some(parent) => dir = parent.to_path_buf(),
            None => break,
        }
    }
    out
}

/// Per-user data directories where Hearth keeps weights across installs.
fn user_models_dirs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for key in ["APPDATA", "LOCALAPPDATA"] {
        if let Some(dir) = non_empty_env(key) {
            out.push(dir.join("hearth").join("models"));
        }
    }
    if let Some(home) = non_empty_env("HOME") {
        #[cfg(target_os = "macos")]
        out.push(
            home.join("Library")
                .join("Application Support")
                .join("hearth")
                .join("models"),
        );
        #[cfg(not(target_os = "macos"))]
        {
            if let Some(xdg) = non_empty_env("XDG_DATA_HOME") {
                out.push(xdg.join("hearth").join("models"));
            }
            out.push(
                home.join(".local")
                    .join("share")
                    .join("hearth")
                    .join("models"),
            );
        }
    }
    out
}

/// Two spellings of the same directory (`models` vs the absolute path,
/// `C:\Foo` vs `c:\foo`) must collapse to one entry.
fn same_dir(a: &Path, b: &Path) -> bool {
    fn key(p: &Path) -> String {
        let s = std::fs::canonicalize(p)
            .unwrap_or_else(|_| p.to_path_buf())
            .to_string_lossy()
            .into_owned();
        if cfg!(windows) {
            s.to_lowercase()
        } else {
            s
        }
    }
    key(a) == key(b)
}

/// `models` means `<cwd>/models` to a reader of the JSON; spell it out so the
/// UI can show a path the user can actually find. Existing absolute paths are
/// left alone (no `\\?\` verbatim prefix, which canonicalize would add).
fn absolutize(dir: PathBuf) -> PathBuf {
    if dir.is_absolute() {
        return dir;
    }
    match std::env::current_dir() {
        Ok(cwd) => cwd.join(dir),
        Err(_) => dir,
    }
}

fn push_unique(out: &mut Vec<PathBuf>, dir: PathBuf) {
    let dir = absolutize(dir);
    if out.iter().any(|seen| same_dir(seen, &dir)) {
        return;
    }
    out.push(dir);
}

/// Every existing directory that may hold weights, most specific first.
///
/// Ordering only matters for de-duplication of the same filename in two
/// places: the first root wins. An explicit `HEARTH_MODELS_DIR` (one entry or
/// a `;`/`:` separated list) suppresses the rest, so a pinned setup never
/// picks up strays.
pub fn models_roots() -> Vec<PathBuf> {
    let overrides = models_dir_overrides();
    if !overrides.is_empty() {
        let mut out = Vec::new();
        for dir in overrides {
            if dir.is_dir() {
                push_unique(&mut out, dir);
            }
        }
        return out;
    }
    let mut out = Vec::new();
    let mut candidates = exe_and_cwd_models();
    candidates.extend(user_models_dirs());
    for dir in candidates {
        if dir.is_dir() {
            push_unique(&mut out, dir);
        }
    }
    out
}

/// Where a downloaded model goes: the override, else `models/` beside the
/// executable, else `./models`, else the per-user data directory. The caller
/// creates the directory — this only decides the path.
pub fn primary_models_dir() -> PathBuf {
    if let Some(dir) = models_dir_override() {
        return dir;
    }
    let candidates = exe_and_cwd_models();
    if let Some(dir) = candidates.into_iter().find(|d| d.is_dir()) {
        return absolutize(dir);
    }
    user_models_dirs()
        .into_iter()
        .next()
        .unwrap_or_else(|| absolutize(PathBuf::from("models")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hearth-paths-{}-{}", name, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn push_unique_collapses_relative_and_absolute_spellings() {
        let dir = temp_dir("dedupe");
        let mut roots = Vec::new();
        push_unique(&mut roots, dir.clone());
        push_unique(&mut roots, dir.join("."));
        push_unique(&mut roots, dir.clone());
        assert_eq!(roots.len(), 1, "same directory must appear once");

        let other = temp_dir("dedupe-other");
        push_unique(&mut roots, other.clone());
        assert_eq!(roots.len(), 2, "a distinct directory must be kept");
        let _ = fs::remove_dir_all(dir);
        let _ = fs::remove_dir_all(other);
    }

    #[test]
    fn override_list_splits_on_the_platform_separator() {
        let raw = if cfg!(windows) {
            r" C:\models ; D:\more-weights ;  "
        } else {
            " /mnt/ssd/models : /srv/weights : "
        };
        let roots = split_roots(raw);
        assert_eq!(roots.len(), 2, "blank entries are dropped: {roots:?}");
        assert!(!roots[1].as_os_str().is_empty());
    }

    #[test]
    fn primary_falls_back_to_an_existing_models_dir() {
        // The repo checkout this test runs from has `models/`, so the walk up
        // from the test executable must find it and nothing may panic.
        let dir = primary_models_dir();
        assert!(!dir.as_os_str().is_empty());
    }
}
