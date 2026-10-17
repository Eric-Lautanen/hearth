# Installed-app UX plan — double-click to chat (Win/Mac/Linux)

> Goal, per operator: on a user's computer, clicking the app launches the
> server and opens the default browser on the chat; closing the window stops
> the server. This doc is the plan. Status of each piece is marked.

## What already works (no changes needed)

- **Auto-open browser**: `hearth-ui` opens the default browser at launch
  (`cmd /C start` on Windows, `open` on macOS, `xdg-open` on Linux),
  `--no-browser` opts out. Verified on Windows this session.
- **Port conflicts**: binds the requested port, scans 20 ahead if taken, and
  opens the *chosen* URL, so a second copy Just Works.
- **Close-the-tab stops the server**: every tab says hello, heartbeats every
  15 s, and sends a goodbye beacon on close. With no tabs for ~25 s (and no
  generation or download running) the server exits. Reloads re-register in
  ~1 s, well inside the grace. `HEARTH_NO_AUTO_SHUTDOWN=1` opts out.
- **Two-file layout**: `hearth-ui` spawns `hearth-chat-cli` sitting next to
  it (`hearth-chat-cli.exe` on Windows, no suffix elsewhere). The installer
  must ship **both** files side by side — the engine is intentionally a
  separate process (a model crash takes the service, not the UI).

## What to build: packaging per OS

All three ship the same two binaries plus an empty `models/` dir (the
downloader fills it) and the README. No admin rights needed anywhere.

### Windows — zip **and MSI**, both shipping

- `hearth-v0.2.0-windows-x86_64.zip`: `hearth-ui.exe`, `hearth-chat-cli.exe`,
  `models/README.txt`, `README.md`, both licenses. Unzip anywhere and
  double-click `hearth-ui.exe`.
- `hearth-v0.2.0-windows-x86_64.msi` (`packaging/windows/hearth.wxs`): the same
  payload, per-user (`%LOCALAPPDATA%\Programs\Hearth`) with no UAC prompt, a
  Start-menu folder, an optional Desktop shortcut, an Add/Remove Programs entry,
  and a launch-on-finish checkbox. Built and verified end to end on Windows 11 —
  `packaging/windows/README.md` lists the checks and the measured results.
- The console window is gone, because option (b) turned out to be cheap.
  `hearth-ui.exe` is a GUI-subsystem binary and every child it spawns — the
  engine service, `curl`, the browser opener — is created with
  `CREATE_NO_WINDOW`. Verified: `MainWindowHandle` is 0 for both
  `hearth-ui.exe` and `hearth-chat-cli.exe` while the server answers. The reason
  option (a) came first was the fear of silent failures, and that does not
  materialise: stderr still reaches a redirected or Terminal launch, and
  `cargo test -p hearth-ui` is unaffected.
- Per-user install without elevation is what makes the MSI the same promise as
  the zip, so the two can be mixed without the user noticing.

### macOS — `.app` bundle via script

- `scripts/package-macos.sh` assembles `Hearth.app`:
  `Contents/MacOS/{hearth-ui,hearth-chat-cli}` (both `chmod +x`),
  `Contents/Resources/{models/,README.md,icon.icns}`, and `Info.plist`
  (`CFBundleExecutable=hearth-ui`, `LSMinimumSystemVersion=13.0`,
  `NSHighResolutionCapable`). Double-click runs with no Terminal window;
  stdout goes to Console.app, so prefix all server logs with `[hearth-ui]`
  (already done) for findability.
- Unsigned builds trigger Gatekeeper on first run — document the
  right-click → Open flow until signing exists. No code change.

### Linux — tarball + `.desktop`

- `hearth-linux.tar.gz`: `hearth-ui`, `hearth-chat-cli`, `models/`,
  `README.md`, `hearth.desktop`:
  `[Desktop Entry]`, `Type=Application`, `Name=Hearth Chat`,
  `Exec=%k/hearth-ui` (repo-relative so it works from anywhere),
  `Terminal=false`, `Icon=` (ship an SVG, reference by absolute path after
  install to `~/.local/share/icons`), `Categories=Utility;`.
- Install = unpack to `~/hearth` or `/opt/hearth`, copy the `.desktop` to
  `~/.local/share/applications`. xdg-open already covers every major browser.

## Edge cases, decided

- **Browser stays open, our tab closed** → server exits (pagehide fires per
  tab). Correct: the chat, not the browser, owns the lifetime.
- **Two tabs open, one closes** → server stays (per-tab ids + heartbeats).
- **Browser crash / killed** → heartbeats stop, ~25 s grace, server exits.
  No orphan processes by design; verify on each OS during packaging.
- **Generation or download in flight when the last tab closes** → shutdown
  waits (both block the reaper); partial downloads resume next launch.
- **Second launch while one runs** → port scan picks 8900+, opens its own
  tab. Two servers is acceptable v1; a single-instance lock file is a
  later improvement, not a requirement.
- **No models on first run** → hero screen points at the downloads drawer;
  the downloader targets the install-local `models/` via `HEARTH_MODELS_DIR`
  override when needed.

## Work list (in order)

1. `scripts/package-macos.sh` + `Info.plist` template. ✅ written
   (`scripts/package-macos.sh`, `resources/Info.plist`, `resources/hearth.icns`);
   the build is exercised by the release workflow, the launch needs a Mac.
2. `hearth.desktop` + icon SVG + Linux install notes. ✅ written
   (`resources/hearth.desktop`, `resources/hearth.svg`, packaged into the Linux
   tarball by the release workflow), needs a Linux box to verify the launch.
3. Windows zip layout doc (one paragraph in README) + console-vs-GUI call.
   ✅ **Verified 2026-10-16**: both release exes copied to a foreign dir,
   launched with `HEARTH_MODELS_DIR` set — service spawned, `/api/models`
   listed all 14, clean shutdown, no repo paths involved.
4. Per-OS verify matrix: launch → tab → close tab → process gone, for each.
   ✅ **Windows done 2026-10-17**, for both shapes: MSI install → run from
   `%LOCALAPPDATA%\Programs\Hearth` → `/`, `/api/models`, `/api/downloads` all
   200 → no visible window → uninstall leaves nothing (or leaves `models\` when
   it holds a download). macOS and Linux stay CI-only until a box is available.
5. Windows MSI (this replaced "future: Inno Setup"). ✅ **Done 2026-10-17**:
   WiX 7, `packaging/windows/hearth.wxs`. WiX carries the upgrade identity, the
   ARP row and the per-user story that Inno Setup would have needed hand-rolled.
6. Later: **signing/notarization** (the workflow already has the hooks; they
   no-op without secrets), single-instance lock, a Linux `install.sh`.
