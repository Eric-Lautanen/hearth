param(
    [int]$DecodeToks = 100,
    [string]$OutDir = ".\bench_results"
)

# Hearth vs llama.cpp-prism — 5-pass benchmark per model at different prompt lengths
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

# Prompts at increasing lengths to exercise prefill + decode across context sizes
$Prompts = @(
    @{Label="1tok";   Prompt="Hello"},
    @{Label="5tok";   Prompt="The quick brown fox jumps"},
    @{Label="10tok";  Prompt="Hello world this is a test prompt for benchmarking"},
    @{Label="20tok";  Prompt="Large language models are neural networks trained on vast text corpora to understand and generate human-like text for various applications"},
    @{Label="50tok";  Prompt="Transformer-based language models use self-attention mechanisms to process sequential data. They have revolutionized natural language processing with applications in translation, summarization, and code generation. Inference requires managing key-value caches across multiple layers. Attention patterns determine how tokens interact. Memory bandwidth is the primary bottleneck during decoding."}
)

# Engine warmup: run a full decode pass to get CPU to boost before measurement
function Warmup-CPU($ModelPath) {
    Write-Host "  Warming up CPU ..." -NoNewline
    & $HearthExe $ModelPath --temp 0 --max-tokens 200 --prompt "Hello" --prompt-raw *>$null
    Write-Host " done"
}

function Run-Hearth($Prmpt) {
    $tmp = [System.IO.Path]::GetTempFileName()
    & $HearthExe $ModelPath --temp 0 --max-tokens $DecodeToks --prompt $Prmpt --prompt-raw *>$tmp
    $txt = Get-Content $tmp -Raw
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue

    $toks = 0; $ms = 0; $tps = 0.0
    if ($txt -match "\[(\d+) tokens, (\d+) ms, ([\d.]+) tok/s\]") {
        $toks = [int]$Matches[1]
        $ms   = [int]$Matches[2]
        $tps  = [double]$Matches[3]
    }
    $pfTok = 0; $pfMs = 0; $pfMsPerTok = 0.0
    if ($txt -match "\[prefill\] (\d+) tokens in (\d+)ms \(([\d.]+)ms/tok\)") {
        $pfTok = [int]$Matches[1]
        $pfMs  = [int]$Matches[2]
        $pfMsPerTok = [double]$Matches[3]
    }
    return [PSCustomObject]@{Toks=$toks; Ms=$ms; Tps=$tps; PfTok=$pfTok; PfMs=$pfMs; PfMsPerTok=$pfMsPerTok}
}

function Run-Ref($Prmpt) {
    $tmp = [System.IO.Path]::GetTempFileName()
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $RefExe
    $psi.Arguments = "-m `"$ModelPath`" --temp 0 -n $DecodeToks -p `"$Prmpt`" --no-display-prompt"
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardInput = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.StandardInput.Close()
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit(300000)
    $elapsedMs = $sw.ElapsedMilliseconds
    ($stdout + $stderr) | Out-File $tmp -Encoding UTF8
    $txt = Get-Content $tmp -Raw
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue

    $genTps = 0.0; $promptTps = 0.0
    if ($txt -match "Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s") {
        $promptTps = [double]$Matches[1]
        $genTps = [double]$Matches[2]
    }
    return [PSCustomObject]@{Tps=$genTps; Ms=$elapsedMs; PromptTps=$promptTps}
}

$AllResults = @()

foreach ($mod in $Models) {
    $ModelPath = "$ModelDir\$($mod.File)"
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host " MODEL: $($mod.Name)  ($DecodeToks tok decode x $($Prompts.Count) prompts)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    Warmup-CPU $ModelPath

    $hPasses = @()
    $rPasses = @()

    Write-Host "  Hearth (our engine):" -ForegroundColor Green
    foreach ($pr in $Prompts) {
        $h = Run-Hearth $pr.Prompt
        $hPasses += $h
        $hPfStr = if ($h.PfTok -gt 0) { "$($h.PfTok)tok $($h.PfMs)ms ($('{0:N1}' -f $h.PfMsPerTok)ms/tok)" } else { "N/A" }
        Write-Host ("    {0,-6} dec={1} tok/s  pf={2}  ({3} tok/{4}ms)" -f $pr.Label, ("{0:N1}" -f $h.Tps), $hPfStr, $h.Toks, $h.Ms)
    }

    Write-Host "  Ref (llama.cpp-prism):" -ForegroundColor Magenta
    foreach ($pr in $Prompts) {
        $r = Run-Ref $pr.Prompt
        $rPasses += $r
        Write-Host ("    {0,-6} dec={1} tok/s  pf={2} t/s  ({3}ms)" -f $pr.Label, ("{0:N1}" -f $r.Tps), ("{0:N1}" -f $r.PromptTps), $r.Ms)
    }

    $hDecAvg = ($hPasses | Measure-Object -Property Tps -Average).Average
    $hPfVals = ($hPasses | Where-Object { $_.PfMsPerTok -gt 0 } | ForEach-Object { $_.PfMsPerTok })
    $hPfAvg  = if ($hPfVals.Count -gt 0) { ($hPfVals | Measure-Object -Average).Average } else { 0 }
    $rDecAvg = ($rPasses | Measure-Object -Property Tps -Average).Average
    $rPfAvg  = ($rPasses | Measure-Object -Property PromptTps -Average).Average
    $decSpeedup = if ($rDecAvg -gt 0) { $hDecAvg / $rDecAvg } else { 0 }

    $hPfStr = if ($hPfAvg -gt 0) { "{0:N1}ms/tok" -f $hPfAvg } else { "N/A" }
    Write-Host ("  AVERAGE: Hearth dec={0:N1} tok/s pf={1} | Ref dec={2:N1} tok/s pf={3:N1}t/s | Dec speedup={4:N2}x" -f $hDecAvg, $hPfStr, $rDecAvg, $rPfAvg, $decSpeedup) -ForegroundColor Yellow

    $AllResults += [PSCustomObject]@{
        Model      = $mod.Name
        DecodeToks = $DecodeToks
        HearthDec  = "{0:N1}" -f $hDecAvg
        HearthPf   = $hPfStr
        RefDec     = "{0:N1}" -f $rDecAvg
        RefPf      = "{0:N1}t/s" -f $rPfAvg
        DecSpeedup = "{0:N2}x" -f $decSpeedup
        HPasses    = ($hPasses | ForEach-Object { "{0:N1}" -f $_.Tps }) -join ", "
        RPasses    = ($rPasses | ForEach-Object { "{0:N1}" -f $_.Tps }) -join ", "
    }
}

$Elapsed = (Get-Date) - $StartTime
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " COMPLETE in $($Elapsed.TotalMinutes.ToString('0.0')) min" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
$AllResults | Format-Table Model, DecodeToks, HearthDec, HearthPf, RefDec, RefPf, DecSpeedup -AutoSize

$csvPath = "$OutDir\benchmark_results.csv"
$AllResults | Export-Csv -Path $csvPath -NoTypeInformation
$AllResults | Format-Table -AutoSize | Out-File "$OutDir\benchmark_table.txt"
Write-Host "Results saved to $csvPath" -ForegroundColor Green
