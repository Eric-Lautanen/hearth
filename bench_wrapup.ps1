param(
    [int]$GenToks = 200,
    [string]$OutDir = ".\bench_results"
)

# Hearth vs llama.cpp-prism — 5-pass benchmark per model
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$StartTime = Get-Date

$Models = @(
    @{Name="1.7B-Q1_0";  File="Bonsai-1.7B-Q1_0.gguf"},
    @{Name="1.7B-Q2_0";  File="Ternary-Bonsai-1.7B-Q2_0.gguf"},
    @{Name="4B-Q1_0";    File="Bonsai-4B.gguf"},
    @{Name="4B-Q2_0";    File="Ternary-Bonsai-4B-Q2_0.gguf"},
    @{Name="8B-Q1_0";    File="Bonsai-8B.gguf"},
    @{Name="8B-Q2_0";    File="Ternary-Bonsai-8B-Q2_0.gguf"}
)

$ModelDir = "$env:USERPROFILE\AppData\Roaming\hearth\models"
$RefExe   = "$env:TEMP\llama.cpp-prism\build\bin\Release\llama-cli.exe"
$HearthExe= ".\target\release\hearth-chat-cli.exe"
$Prompt   = "Hello"
$NumPasses = 5

# Engine warmup: run a full decode pass to get CPU to boost before measurement
# Windows 11 frequency scaling takes 30-40 tokens to reach full boost.
# A throwaway 200-token run as warmup ensures measurement passes are at steady state.
function Warmup-CPU {
    Write-Host "  Warming up CPU ..." -NoNewline
    & $HearthExe $ModelPath --temp 0 --max-tokens 200 --prompt $Prompt --prompt-raw *>$null
    Write-Host " done"
}

function Run-Hearth {
    $tmp = [System.IO.Path]::GetTempFileName()
    & $HearthExe $ModelPath --temp 0 --max-tokens $GenToks --prompt $Prompt --prompt-raw *>$tmp
    $txt = Get-Content $tmp -Raw
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue

    $toks = 0; $ms = 0; $tps = 0.0
    if ($txt -match "\[(\d+) tokens, (\d+) ms, ([\d.]+) tok/s\]") {
        $toks = [int]$Matches[1]
        $ms   = [int]$Matches[2]
        $tps  = [double]$Matches[3]
    }
    return @{Toks=$toks; Ms=$ms; Tps=$tps}
}

function Run-Ref {
    $tmp = [System.IO.Path]::GetTempFileName()
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $RefExe
    $psi.Arguments = "-m `"$ModelPath`" --temp 0 -n $GenToks -p `"$Prompt`" --no-display-prompt"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardInput = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.StandardInput.Close()
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit(300000)
    ($stdout + $stderr) | Out-File $tmp -Encoding UTF8
    $txt = Get-Content $tmp -Raw
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue

    $genTps = 0.0
    if ($txt -match "Generation:\s*([\d.]+)\s*t/s") {
        $genTps = [double]$Matches[1]
    }
    return @{Tps=$genTps}
}

$AllResults = @()

foreach ($mod in $Models) {
    $script:ModelPath = "$ModelDir\$($mod.File)"
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host " MODEL: $($mod.Name)  (200 tok x $NumPasses passes)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Warmup CPU with one full decode pass
    Warmup-CPU

    $hPasses = @()
    $rPasses = @()

    Write-Host "  Hearth (our engine):" -ForegroundColor Green
    for ($i = 1; $i -le $NumPasses; $i++) {
        $h = Run-Hearth
        $hPasses += $h
        Write-Host ("    pass {0}: {1} tok/s  ({2} tokens in {3}ms)" -f $i, ("{0:N1}" -f $h.Tps), $h.Toks, $h.Ms)
    }

    Write-Host "  Ref (llama.cpp-prism):" -ForegroundColor Magenta
    for ($i = 1; $i -le $NumPasses; $i++) {
        $r = Run-Ref
        $rPasses += $r
        Write-Host ("    pass {0}: {1} tok/s" -f $i, ("{0:N1}" -f $r.Tps))
    }

    $hAvg = ($hPasses | Measure-Object -Property Tps -Average).Average
    $rAvg = ($rPasses | Measure-Object -Property Tps -Average).Average
    $speedup = if ($rAvg -gt 0) { $hAvg / $rAvg } else { 0 }

    Write-Host ("  AVERAGE: Hearth={0:N1} tok/s  Ref={1:N1} tok/s  Speedup={2:N2}x" -f $hAvg, $rAvg, $speedup) -ForegroundColor Yellow

    $AllResults += [PSCustomObject]@{
        Model     = $mod.Name
        HearthAvg = "{0:N1}" -f $hAvg
        RefAvg    = "{0:N1}" -f $rAvg
        Speedup   = "{0:N2}x" -f $speedup
        HPasses   = ($hPasses | ForEach-Object { "{0:N1}" -f $_.Tps }) -join ", "
        RPasses   = ($rPasses | ForEach-Object { "{0:N1}" -f $_.Tps }) -join ", "
    }
}

# Summary
$Elapsed = (Get-Date) - $StartTime
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " COMPLETE in $($Elapsed.TotalMinutes.ToString('0.0')) min" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
$AllResults | Format-Table Model, HearthAvg, RefAvg, Speedup -AutoSize

# Save results
$csvPath = "$OutDir\benchmark_results.csv"
$AllResults | Export-Csv -Path $csvPath -NoTypeInformation
$AllResults | Format-Table -AutoSize | Out-File "$OutDir\benchmark_table.txt"
Write-Host "Results saved to $csvPath" -ForegroundColor Green
