# Windows packaging

Two supported shapes, built from the same payload:

* `hearth.wxs` - a per-user **MSI installer**. Double-click installs through a
  two-page wizard: a welcome page whose two checkboxes decide whether a Desktop
  shortcut appears and whether Hearth opens when setup finishes, then a
  "Setup complete" page whose Finish button closes the wizard. Both boxes are
  ticked by default, each owns a feature (`DesktopShortcutFeature`, or the
  `LAUNCH_APP` property that the completion page's launch action reads), and a
  silent install runs no UI and lands on the same defaults.
* The **portable zip**, staged by `scripts/package-win.ps1`: unzip anywhere and
  run `hearth-ui.exe`.

There is no portable install/uninstall flag to run, because Hearth claims no
file associations and writes nothing outside its own folder - the zip and the
MSI leave the machine in the same state apart from where the files sit.

Both are per-user (`Scope="perUser"` plus `ALLUSERS=2` / `MSIINSTALLPERUSER=1`),
so a standard user installs everything inside their own profile
(`%LOCALAPPDATA%\Programs\Hearth`) with no UAC prompt, and uninstalls it again
with no prompt. The Add/Remove Programs row is registered machine-wide under
`HKLM` even so - that is a property of `ALLUSERS=2`, not of this file, and it is
what the uninstall test above pins down. An existing MSI install is
major-upgraded in place: the same version replaces itself, and a genuinely newer
install is detected and reported rather than downgraded.

## Building it locally

```powershell
# 1. release binaries (the workspace, because the UI spawns the CLI)
cargo build --release

# 2. the payload: both exes, models/README.txt, README, licenses; -Zip also
#    writes target/dist/hearth-v0.2.0-windows-x86_64.zip
./scripts/package-win.ps1 -Version 0.2.0 -Zip

# 3. the installer (WiX 7 as a dotnet tool; the UI extension carries the
#    dialogs' stock pages)
dotnet tool install --global wix --version 7.0.0
& "$env:USERPROFILE\.dotnet\tools\wix.exe" eula accept wix7
& "$env:USERPROFILE\.dotnet\tools\wix.exe" extension add -g WixToolset.UI.wixext
& "$env:USERPROFILE\.dotnet\tools\wix.exe" build `
    -acceptEula wix7 -ext WixToolset.UI.wixext -arch x64 `
    -d Version=0.2.0 `
    -d PayloadDir=target/payload `
    -d Icon=resources/hearth.ico `
    -d DialogIcon=resources/hearth-setup.ico `
    -o target/dist/hearth-v0.2.0-windows-x86_64.msi `
    packaging/windows/hearth.wxs
```

Build defines: `Version` (numeric, three or four fields - the release workflow
strips a tag's `v` and any `-suffix`), `PayloadDir`, `Icon` and `DialogIcon`.

The icon pair exists because MSI's Icon control draws the *first* frame of an
icon stretched to the control's pixel size: `hearth-setup.ico` leads with its
48px frame (a 32x32-dialog-unit control at 100% scaling) while `hearth.ico`
leads with 16px, the order the shell's icon resolution wants. Both come from
`resources/hearth.svg` via `python scripts/make-icons.py`, which also writes the
macOS `.icns`; regenerate them whenever the artwork changes.

## Verifying a build

```powershell
# 1. tables: shortcuts, features, install layout, installer choices, dialogs.
#    Fourteen checks, and it exits non-zero if any of them fails, so it is
#    usable as a gate.
./scripts/msi-query.ps1 target/dist/hearth-v0.2.0-windows-x86_64.msi

# 2. for what that API cannot read back - the RemoveFolder table rejects every
#    query spelling - ask WiX itself. Decompiled authoring is what proves the
#    three On="uninstall" folder removals are really in the package.
& "$env:USERPROFILE\.dotnet\tools\wix.exe" msi decompile -acceptEula wix7 `
    -o target/dist/hearth-decompiled.wxs target/dist/hearth-v0.2.0-windows-x86_64.msi

# 3. a real install and uninstall, with the log to read if it fails. /qn runs no
#    UI at all, so both defaults are what a scripted install gets.
msiexec /i target/dist/hearth-v0.2.0-windows-x86_64.msi /qn /l*v target/msi-install.log
& "$env:LOCALAPPDATA\Programs\Hearth\hearth-ui.exe" --no-browser --port 8899
msiexec /x target/dist/hearth-v0.2.0-windows-x86_64.msi /qn /l*v target/msi-uninstall.log
```

What a correct install leaves, each of these checked on Windows 11 before a
release:

* `%LOCALAPPDATA%\Programs\Hearth\` with both executables, `models\README.txt`,
  the README and both licenses. The note has to sit *in* `models\`: a `File`
  installs into its component's directory, so a component parented to
  `INSTALLFOLDER` put it beside the executables (observed, and why the note has
  a `<Directory>` of its own).
* `Hearth Chat` and `Uninstall Hearth` in a `Hearth` Start-menu folder, plus the
  Desktop shortcut unless it was unticked.
* One Add/Remove Programs row named Hearth, carrying its version and
  manufacturer.
* Nothing visible on screen while it serves: `hearth-ui.exe` is a GUI-subsystem
  binary and every child it spawns gets `CREATE_NO_WINDOW`, so neither it nor
  `hearth-chat-cli.exe` owns a window (`Get-Process hearth-ui, hearth-chat-cli`
  reports `MainWindowHandle` 0 for both) while `/`, `/api/models` and
  `/api/downloads` all answer 200.
* An uninstall that leaves nothing - except a downloaded model. Put a file in
  `models\`, uninstall, and the file and its folder are still there
  (`RemoveFolder` removes empty directories only), which is what a reinstall
  then finds. Uninstall without one and `Programs\Hearth` itself is gone along
  with the Start-menu folder, the Desktop shortcut and the Add/Remove Programs
  row.
* Exactly one Add/Remove Programs row after installing a *second build* of the
  same version - the case `<MajorUpgrade>` gets wrong, because its VersionMax is
  exclusive and a republished 0.2.0 then matches nothing.

Two things about this machine surprise at first sight, both measured rather than
assumed: the install log reports `VersionNT` 603 and `WindowsBuild` 9600 (the
editor's shipped installer logs exactly the same), which is why there is no
Windows-version launch condition; and the Add/Remove Programs row lands under
`HKLM` while the payload, shortcuts and log-classified per-user install stay in
the installing account's profile. Neither install nor uninstall prompts for
elevation.

## Not in the installer

* **Models.** They are 0.5 GB and up and live on Hugging Face, so the app's own
  Models panel downloads them (needs `curl.exe`, present since Windows 10 1803).
* **The icon inside the executables.** The MSI's shortcuts and Add/Remove
  Programs entry carry the icon; `hearth-ui.exe` itself still shows the generic
  one in Explorer, because embedding an icon needs `rc.exe`/`windres` or a
  hand-built `.res` linked from `build.rs`. Tracked, not done.
* **A PATH entry.** Hearth installs for one account and is started from its
  shortcut; `hearth-chat-cli.exe` is in the install directory for anyone who
  wants it in a terminal.
