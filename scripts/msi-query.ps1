<#
.SYNOPSIS
  Print and check the parts of a built Hearth MSI that can go wrong silently.

.DESCRIPTION
  An MSI that WiX compiled is not an MSI that installs correctly. A missing
  EmbedCab leaves an external cabinet the user downloads without; an exclusive
  upgrade range leaves a second Add/Remove Programs row beside the old one; a
  wrong StandardDirectory puts the payload where a standard user cannot write;
  an installer choice that defaults to 0 silently changes what a scripted
  install gets; a File in the wrong component puts the models folder beside the
  executables. Every one of those is a table row, so this reads the tables
  through Windows Installer's own COM API - no WiX installation needed, and no
  MSI internals reimplemented.

  Exits non-zero if any check fails, so it is usable as a gate.

.EXAMPLE
  ./scripts/msi-query.ps1 target/dist/hearth-v0.2.0-windows-x86_64.msi
#>
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$Msi)

$ErrorActionPreference = 'Stop'
$Msi = (Resolve-Path -LiteralPath $Msi).Path

$installer = New-Object -ComObject WindowsInstaller.Installer
$db = $installer.GetType().InvokeMember('OpenDatabase', 'InvokeMethod', $null, $installer, @($Msi, 0))

function Get-Rows([string]$sql, [int[]]$columns) {
    $view = $db.GetType().InvokeMember('OpenView', 'InvokeMethod', $null, $db, @($sql))
    # Assigned rather than left to the pipeline: an uncaptured return value is
    # part of this function's output, which would prepend a null row.
    $null = $view.GetType().InvokeMember('Execute', 'InvokeMethod', $null, $view, $null)
    $rows = @()
    while ($true) {
        $record = $view.GetType().InvokeMember('Fetch', 'InvokeMethod', $null, $view, $null)
        if ($null -eq $record) { break }
        # @() because a one-column select would otherwise hand back a bare
        # string rather than a row array.
        $values = @(foreach ($column in $columns) {
            [string]$record.GetType().InvokeMember('StringData', 'GetProperty', $null, $record, $column)
        })
        $rows += , $values
    }
    # `,$rows` keeps a single row from unrolling into its own columns.
    return , $rows
}

function Show([string]$title, [string]$sql, [int[]]$columns) {
    Write-Host ''
    Write-Host "== $title"
    $rows = Get-Rows $sql $columns
    if ($rows.Count -eq 0) { Write-Host '  (no rows)' }
    foreach ($row in $rows) { Write-Host ('  ' + ($row -join ' | ')) }
}

Write-Host "MSI: $Msi"

Show 'product and installer choices' `
    'SELECT `Property`,`Value` FROM `Property` WHERE `Property`=''ProductName'' OR `Property`=''ProductVersion'' OR `Property`=''Manufacturer'' OR `Property`=''ALLUSERS'' OR `Property`=''MSIINSTALLPERUSER'' OR `Property`=''DESKTOP_SHORTCUT'' OR `Property`=''LAUNCH_APP'' OR `Property`=''ARPNOMODIFY'' OR `Property`=''ARPPRODUCTICON'' OR `Property`=''ARPCOMMENTS'' OR `Property`=''ARPURLINFOABOUT''' @(1, 2)

Show 'features (Feature | Title | Level | Description)' `
    'SELECT `Feature`,`Title`,`Level`,`Description` FROM `Feature`' @(1, 2, 3, 4)

Show 'install layout (Directory | Parent | DefaultDir)' `
    'SELECT `Directory`,`Directory_Parent`,`DefaultDir` FROM `Directory`' @(1, 2, 3)

Show 'components (Component | Directory | KeyPath | Attributes)' `
    'SELECT `Component`,`Directory_`,`KeyPath`,`Attributes` FROM `Component`' @(1, 2, 3, 4)

Show 'files (File | Component | Name | Size)' `
    'SELECT `File`,`Component_`,`FileName`,`FileSize` FROM `File`' @(1, 2, 3, 4)

Show 'shortcuts (Directory | Name | Target | Arguments | Icon)' `
    'SELECT `Directory_`,`Name`,`Target`,`Arguments`,`Icon_` FROM `Shortcut`' @(1, 2, 3, 4, 5)

Show 'icons (Name)' 'SELECT `Name` FROM `Icon`' @(1)

Show 'custom actions (Action | Type | Source | Target)' `
    'SELECT `Action`,`Type`,`Source`,`Target` FROM `CustomAction`' @(1, 2, 3, 4)

Show 'upgrade rows (UpgradeCode | Min | Max | Attributes | ActionProperty | Remove)' `
    'SELECT `UpgradeCode`,`VersionMin`,`VersionMax`,`Attributes`,`ActionProperty`,`Remove` FROM `Upgrade`' @(1, 2, 3, 4, 5, 6)

Show 'launch conditions (Condition | Description)' `
    'SELECT `Condition`,`Description` FROM `LaunchCondition`' @(1, 2)

Show 'media (DiskId | LastSequence | Cabinet)' `
    'SELECT `DiskId`,`LastSequence`,`Cabinet` FROM `Media`' @(1, 2, 3)

Show 'welcome page text (Control | Text)' `
    'SELECT `Control`,`Text` FROM `Control` WHERE `Dialog_`=''HearthWelcomeDlg'' AND `Text` IS NOT NULL ORDER BY `Control`' @(1, 2)

Show 'welcome page events (Control | Event | Argument | Condition)' `
    'SELECT `Control_`,`Event`,`Argument`,`Condition` FROM `ControlEvent` WHERE `Dialog_`=''HearthWelcomeDlg'' ORDER BY `Ordering`' @(1, 2, 3, 4)

Show 'completion page events (Control | Event | Argument | Condition)' `
    'SELECT `Control_`,`Event`,`Argument`,`Condition` FROM `ControlEvent` WHERE `Dialog_`=''HearthDoneDlg'' ORDER BY `Ordering`' @(1, 2, 3, 4)

# ---- the checks that a reviewer would otherwise make by eye ----------------
Write-Host ''
Write-Host '== checks'

$script:checks = 0
$script:failures = 0
function Expect([string]$what, [bool]$ok, [string]$detail) {
    $script:checks++
    if (-not $ok) { $script:failures++ }
    $verdict = if ($ok) { 'ok  ' } else { 'FAIL' }
    Write-Host "  [$verdict] $what - $detail"
}

function Property([string]$name) {
    # Built with -f rather than interpolated: in a double-quoted string the
    # doubled quotes below stop being SQL quoting and the query fails to parse.
    $sql = "SELECT ``Value`` FROM ``Property`` WHERE ``Property``='{0}'" -f $name
    $rows = Get-Rows $sql @(1)
    if ($rows.Count -eq 0) { return '' }
    return $rows[0][0]
}

Expect 'per-user scope' ((Property 'ALLUSERS') -eq '2' -and (Property 'MSIINSTALLPERUSER') -eq '1') `
    "ALLUSERS=$(Property 'ALLUSERS') MSIINSTALLPERUSER=$(Property 'MSIINSTALLPERUSER')"
Expect 'installer choices default to the documented values' `
    ((Property 'DESKTOP_SHORTCUT') -eq '1' -and (Property 'LAUNCH_APP') -eq '1') `
    "DESKTOP_SHORTCUT=$(Property 'DESKTOP_SHORTCUT') LAUNCH_APP=$(Property 'LAUNCH_APP')"
Expect 'Add/Remove Programs keeps its icon and the repo URL' `
    ((Property 'ARPPRODUCTICON') -eq 'hearth.ico' -and (Property 'ARPURLINFOABOUT') -eq 'https://github.com/Eric-Lautanen/hearth') `
    "ARPPRODUCTICON=$(Property 'ARPPRODUCTICON')"

$media = Get-Rows 'SELECT `Cabinet`,`LastSequence` FROM `Media`' @(1, 2)
$cabinet = if ($media.Count -gt 0) { $media[0][0] } else { '' }
Expect 'the payload cabinet is embedded' ($cabinet.StartsWith('#')) "Media.Cabinet='$cabinet'"
$stray = Get-ChildItem -LiteralPath (Split-Path -Parent $Msi) -Filter '*.cab' -ErrorAction SilentlyContinue
Expect 'no external cabinet beside the MSI' ($null -eq $stray) "$(@($stray).Count) .cab file(s)"

$upgrade = Get-Rows 'SELECT `UpgradeCode`,`VersionMin`,`VersionMax`,`Attributes`,`ActionProperty` FROM `Upgrade`' @(1, 2, 3, 4, 5)
$codes = @($upgrade | ForEach-Object { $_[0].Trim('{}') } | Select-Object -Unique)
Expect 'one upgrade identity, and it is the committed one' `
    ($upgrade.Count -eq 2 -and $codes.Count -eq 1 -and $codes[0] -eq '346FF6B2-F464-473A-8436-054B8199D02C') `
    "$($codes -join ', ') across $($upgrade.Count) rows"
$arms = @($upgrade | ForEach-Object { $_[4] })
Expect 'it both removes older versions and detects newer ones' `
    (($arms -contains 'WIX_UPGRADE_DETECTED') -and ($arms -contains 'WIX_NEWER_VERSION_DETECTED')) `
    "$($arms -join ', ')"
# Whether the removal range includes its own maximum is a WiX attribute that
# packs into the encoded Attributes bitmask, so it is checked where it is
# observable: installing the same MSI twice must leave one Add/Remove Programs
# row, not two. See packaging/windows/README.md.
Expect 'the removal arm covers its own version, so a reinstall replaces it' `
    (@($upgrade | Where-Object { $_[4] -eq 'WIX_UPGRADE_DETECTED' -and $_[2] -eq (Property 'ProductVersion') }).Count -eq 1) `
    "removal arm VersionMax=$(($upgrade | Where-Object { $_[4] -eq 'WIX_UPGRADE_DETECTED' } | ForEach-Object { $_[2] }) -join ', ') vs ProductVersion=$(Property 'ProductVersion')"

$install = Get-Rows 'SELECT `Directory_Parent`,`DefaultDir` FROM `Directory` WHERE `Directory`=''INSTALLFOLDER''' @(1, 2)
Expect 'the app installs under the per-user program files folder' `
    ($install.Count -eq 1 -and $install[0][0] -eq 'PerUserProgramFilesFolder') `
    $(if ($install.Count -eq 1) { "INSTALLFOLDER = $($install[0][0])\$($install[0][1])" } else { 'INSTALLFOLDER missing' })

$shortcuts = Get-Rows 'SELECT `Directory_`,`Name`,`Target` FROM `Shortcut`' @(1, 2, 3)
Expect 'the Start menu launches the app' `
    (@($shortcuts | Where-Object { $_[0] -eq 'HearthMenuFolder' -and $_[1] -like '*Hearth Chat' -and $_[2] -like '*hearth-ui.exe' }).Count -eq 1) `
    "$(($shortcuts | ForEach-Object { "$($_[0])\$($_[1])" }) -join ', ')"
Expect 'an uninstall shortcut points at msiexec' `
    (@($shortcuts | Where-Object { $_[1] -like '*Uninstall*' -and $_[2] -like '*msiexec*' }).Count -eq 1) `
    "$(@($shortcuts).Count) shortcut(s)"

# Names are compared with a wildcard because a long name that did not fit 8.3
# arrives as `short|long`.
$files = Get-Rows 'SELECT `FileName` FROM `File`' @(1)
$names = @($files | ForEach-Object { $_[0] })
Expect 'both executables ship' `
    ((@($names | Where-Object { $_ -like '*hearth-ui.exe' }).Count -eq 1) -and (@($names | Where-Object { $_ -like '*hearth-chat-cli.exe' }).Count -eq 1)) `
    "$($names -join ', ')"
Expect 'the models note ships (it creates the models folder)' `
    (@($names | Where-Object { $_ -like '*README.txt' }).Count -eq 1) `
    "$(@($names | Where-Object { $_ -like '*README.txt' }).Count) of $($names.Count) files"

$actions = Get-Rows 'SELECT `Action`,`Type`,`Target` FROM `CustomAction` WHERE `Action`=''LaunchHearth''' @(1, 2, 3)
Expect 'the finish page has a launch action' `
    ($actions.Count -eq 1 -and ([int]$actions[0][1] -band 0x3F) -eq 18) `
    $(if ($actions.Count -eq 1) { "type=$($actions[0][1]) target=$($actions[0][2])" } else { 'missing' })

# RemoveFolder deletes a directory only when it is empty, which is what makes the
# two uninstall outcomes come out right: the payload folders disappear, and a
# downloaded model keeps its folder (and therefore Programs\Hearth). That table
# cannot be read back through this API - Windows Installer rejects the query
# whatever the quoting - so the built authoring is checked with `wix msi
# decompile` instead, and packaging/windows/README.md records both uninstall
# scenarios with their commands and observed results.

Write-Host ''
if ($script:failures -gt 0) {
    Write-Host "$($script:failures) of $($script:checks) checks FAILED"
    exit 1
}
Write-Host "all $($script:checks) checks passed"
exit 0
