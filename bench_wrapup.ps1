param(
    [int]$GenToks = 20,
    [string]$OutDir = ".\bench_results"
)

# Hearth vs llama.cpp-prism — comprehensive wrap-up benchmark
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

$Prompts = @(
    @{Label="001tok"; Prompt="Hello"},
    @{Label="005tok"; Prompt="The quick brown fox jumps over"},
    @{Label="010tok"; Prompt="Hello world this is a test prompt for benchmarking language models"},
    @{Label="020tok"; Prompt="Large language models are neural networks trained on vast text corpora to understand and generate human-like text for various applications"},
    @{Label="030tok"; Prompt="Transformer-based language models use self-attention mechanisms to process sequential data. They have revolutionized natural language processing with applications in translation, summarization, and code generation. Inference requires managing key-value caches."},
    @{Label="050tok"; Prompt="Quantization reduces the memory footprint of large language models by storing weights in lower precision formats like 1-bit or 2-bit integers instead of 32-bit floats. This enables running models on consumer hardware with limited RAM, though some accuracy is traded for speed. Different quantization schemes offer various tradeoffs between compression ratio and output quality."},
    @{Label="100tok"; Prompt="The performance of CPU-based LLM inference depends on several factors including memory bandwidth, SIMD vectorization, and thread utilization. Modern CPUs with AVX-512 instructions can process multiple data elements simultaneously. The key operations during inference are matrix-vector multiplications between quantized weights and activation vectors. These operations are memory-bandwidth bound because the model weights must be read from DRAM for each token generated. Techniques like weight quantization reduce the data transferred, while thread pools and cache-friendly data layouts maximize throughput. Prefill processing differs fundamentally from decode: prefill is compute-bound as multiple tokens are processed in parallel, while decode is memory-bandwidth bound generating one token at a time using the growing key-value cache."},
    @{Label="200tok"; Prompt="Optimizing LLM inference on CPU requires different strategies for the prefill and decode phases. During prefill, the model processes all prompt tokens simultaneously through each transformer layer. This involves computing query, key, and value projections for every token via matrix multiplication against weight matrices. The attention mechanism then computes scores between all pairs of tokens in the prompt, which has quadratic complexity in the sequence length. For long prompts, attention becomes the dominant cost. Optimizations for prefill include parallelizing batch operations across thread pool workers, fusing consecutive operations like layer normalization and quantization to reduce memory traffic, and using SIMD vector instructions for element-wise operations.\n\nDuring decode, each new token is generated one at a time, processing it through all layers while reading from the cached key-value tensors of previous tokens. This phase is fundamentally memory-bandwidth bound: each generated token requires reading the entire model weights from DRAM, plus the growing key-value cache. Techniques like weight quantization directly reduce the memory bottleneck by packing more weights per byte. Custom thread pools with spin-loop synchronization avoid OS syscall overhead. Cache-friendly tiling of weight rows keeps data in L2 cache longer."},
    @{Label="500tok"; Prompt="CPU inference for large language models has become increasingly practical due to advances in quantization techniques and optimized compute kernels. The fundamental challenge is that modern LLMs have billions of parameters requiring gigabytes of storage, far exceeding typical CPU cache sizes. Each inference step must stream weights from main memory, making memory bandwidth the primary bottleneck. Quantized formats like Q1_0 and Q2_0 reduce this by representing weights in 1-bit or 2-bit formats, packing many weights into each memory transaction.\n\nThe prefill phase processes the entire input prompt in parallel, computing hidden states for each token position through each transformer layer. This involves matrix multiplications between the hidden states and weight matrices for QKV projections and feed-forward networks. Since all tokens share the same weight matrices, weights are read once and reused across the batch, shifting the bottleneck toward compute rather than memory. Attention in prefill requires computing pairwise interactions between all token positions, scaling as O(n^2) with prompt length. At short to medium prompt lengths, the matrix multiplications dominate. At very long prompts, attention becomes the primary cost.\n\nThe decode phase generates tokens one at a time. Each step processes one token through all layers while attending to the full KV cache of previous tokens. The weight matrices must be fully read from DRAM for each token, creating a hard memory-bandwidth bound. The KV cache also grows linearly with generation length, adding to memory pressure. Optimizing decode focuses on reducing per-token data movement: quantized weights, efficient dequantization kernels using SIMD instructions, multi-threaded row-parallel matrix multiplication, and software pipelining to overlap compute with memory access.\n\nThread pool design significantly impacts performance on multi-core systems. A spin-loop-based thread pool with atomic generation counters avoids operating system synchronization overhead while allowing workers to begin processing immediately when new work is dispatched. The number of threads must be carefully chosen: too few underutilizes cores, while too many causes SMT contention that degrades performance."},
    @{Label="1000tok"; Prompt="Quantization-aware kernel design for CPU-based LLM inference requires careful consideration of the target hardware's microarchitecture. Modern x86-64 processors from AMD and Intel implement various SIMD instruction sets including AVX2 and AVX-512, each with different throughput characteristics for integer and floating-point operations. The Q1_0 quantization format represents each weight as a single bit using sign-magnitude encoding, packing 128 weights plus a shared scale factor into 18-byte blocks. The dot product between Q1_0 weights and Q8_0 activations can be computed using specialized SIMD kernels that exploit instructions like vpdpbusd when available, or through bit-manipulation techniques using shuffle and mask operations on AVX2 hardware.\n\nThe key insight for efficient CPU inference is that the quantization format determines the compute kernel structure, which in turn determines how well the kernel maps to available SIMD hardware. Formats with bit-level packing require unpacking or lookup-table approaches that add instruction overhead but save DRAM bandwidth. The tradeoff: reduced memory traffic versus increased compute. For bandwidth-bound decode, formats with higher compression win even with slower kernels. For compute-bound prefill, formats that map cleanly to SIMD pipelines may perform better despite lower compression.\n\nMemory hierarchy optimization is critical. The Ryzen 8840HS used in this system has 32KB L1 data cache and 1MB L2 cache per core, plus 16MB shared L3 cache. Weight matrices for even small quantized models far exceed any cache level, so weights must stream from DRAM during decode. The L1 cache becomes important for activation data which are reused within a layer. The L2 cache can hold a row of quantized weights for each worker thread. Tile-based dispatch where each thread processes weight rows in L2-sized chunks improves cache utilization and prefetcher behavior.\n\nThread synchronization strategy dramatically affects performance at scale. The gen-counter approach uses a single atomic counter incremented by the main thread to signal new work. Worker threads spin-loop checking this counter, sleeping only after extended idle periods. This eliminates the per-dispatch syscall overhead of park/unpark mechanisms while adding minimal latency. Each worker computes its chunk of rows independently, writing results to a pre-arranged output buffer. The main thread waits for all workers by polling their individual done flags.\n\nQuantization format choice affects not just size but also compute characteristics. Q1_0 uses {-1, +1} values that enable a shuffle-based kernel using SIMD compare-to-mask operations to expand bits to signed bytes, then multiply-accumulate with activations. Q2_0 uses {-1, 0, +1, +2} values that require a 256-entry lookup table to convert packed 2-bit indices to i8 weight values. The LUT approach adds 1KB of L1 pressure but enables efficient SIMD gather operations. For Q1_0, the shuffle kernel avoids LUT entirely but uses more instructions per element. The relative performance depends on cache pressure: Q1_0's lack of LUT gives it an advantage at large model sizes where L1 is already strained by activation data.\n\nModel architecture also matters. The Qwen3-based Bonsai models use grouped-query attention with 8 KV heads and variable query heads, head dimension of 128, and YaRN rotary position embedding with 4x context extension. The feed-forward networks use SwiGLU activation with intermediate dimensions varying by model size. Layer normalization uses RMS norm applied before each sub-layer. The models include per-head QK normalization learned weights that normalize query and key vectors before computing attention scores. These architectural choices affect the compute-to-memory ratio and the effectiveness of different optimization strategies."}
)

$Results = @()

function Run-Hearth($ModelPath, $Prompt) {
    $tmp = [System.IO.Path]::GetTempFileName()
    & ".\target\release\hearth-chat-cli.exe" $ModelPath --temp 0 --max-tokens $GenToks --prompt "$Prompt" --prompt-raw *>$tmp
    $txt = Get-Content $tmp -Raw
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue

    $pfToks=0; $pfMs=0; $cpuUs=0
    if ($txt -match "\[prefill\] (\d+) tokens in (\d+)ms") { $pfToks=[int]$Matches[1]; $pfMs=[int]$Matches[2] }
    if ($txt -match "avg_cpu_overhead=(\d+)") { $cpuUs=[int]$Matches[1] }
    $decTps = if ($cpuUs -gt 0) { 1e6/$cpuUs } else { 0 }
    return @{PreT=$pfToks; PreMs=$pfMs; DecTps=$decTps}
}

function Run-Ref($ModelPath, $Prompt, $KnownToks) {
    # Close stdin to prevent interactive hang by providing empty input
    $tmp = [System.IO.Path]::GetTempFileName()
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $RefExe
    $psi.Arguments = "-m `"$ModelPath`" --temp 0 -n $GenToks -p `"$Prompt`""
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.RedirectStandardInput = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.StandardInput.Close()
    $stdout = $p.StandardOutput.ReadToEnd()
    $stderr = $p.StandardError.ReadToEnd()
    $p.WaitForExit(120000)
    ($stdout + $stderr) | Out-File $tmp -Encoding UTF8
    $txt = Get-Content $tmp -Raw
    Remove-Item $tmp -Force -ErrorAction SilentlyContinue

    $promptTps=0; $genTps=0
    if ($txt -match "Prompt:\s*([\d.]+)\s*t/s.*Generation:\s*([\d.]+)\s*t/s") {
        $promptTps=[double]$Matches[1]; $genTps=[double]$Matches[2]
    }
    $pfMs = if ($promptTps -gt 0 -and $KnownToks -gt 0) { [math]::Round($KnownToks/$promptTps*1000) } else { 0 }
    return @{PreMs=$pfMs; DecTps=$genTps}
}

foreach ($mod in $Models) {
    $modelPath = "$ModelDir\$($mod.File)"
    Write-Host "`n========================" -ForegroundColor Cyan
    Write-Host " MODEL: $($mod.Name)" -ForegroundColor Cyan
    Write-Host "========================" -ForegroundColor Cyan

    foreach ($pr in $Prompts) {
        Write-Host "  $($pr.Label) ..." -NoNewline

        $h = Run-Hearth $modelPath $pr.Prompt
        $r = Run-Ref $modelPath $pr.Prompt $h.PreT

        $pfSpd = if ($r.PreMs -gt 0 -and $h.PreMs -gt 0) { "{0:N2}x" -f ($r.PreMs/$h.PreMs) } else { "N/A" }
        $decSpd = if ($r.DecTps -gt 0 -and $h.DecTps -gt 0) { "{0:N2}x" -f ($h.DecTps/$r.DecTps) } else { "N/A" }

        Write-Host " toks=$($h.PreT)  H-pf=$($h.PreMs)ms  R-pf~$($r.PreMs)ms  $pfSpd  H-dec=$('{0:N1}' -f $h.DecTps)  R-dec=$('{0:N1}' -f $r.DecTps)  $decSpd" -ForegroundColor Green

        $Results += [PSCustomObject]@{
            Model = $mod.Name
            Prompt = $pr.Label
            Toks = $h.PreT
            HearthPfMs = $h.PreMs
            RefPfMs = $r.PreMs
            PfSpeedup = $pfSpd
            HearthDecTokS = "{0:N1}" -f $h.DecTps
            RefDecTokS = "{0:N1}" -f $r.DecTps
            DecSpeedup = $decSpd
        }
    }
}

# Print summary
$Elapsed = (Get-Date) - $StartTime
Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " COMPLETE in $($Elapsed.TotalMinutes.ToString('0.0')) min" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "`nINDIVIDUAL RESULTS" -ForegroundColor White
$Results | Format-Table -AutoSize

Write-Host "`nAVERAGE SPEEDUP PER MODEL" -ForegroundColor White
foreach ($mod in $Models) {
    $mr = $Results | Where-Object { $_.Model -eq $mod.Name }
    $pfV = $mr | ForEach-Object { if ($_.HearthPfMs -gt 0 -and $_.RefPfMs -gt 0) { $_.RefPfMs / $_.HearthPfMs } }
    $dcV = $mr | ForEach-Object { if ($_.HearthDecTokS -ne "0.0" -and $_.RefDecTokS -ne "0.0") { [double]$_.HearthDecTokS / [double]$_.RefDecTokS } }
    $aPf = if ($pfV.Count -gt 0) { ($pfV | Measure-Object -Average).Average } else { 0 }
    $aDc = if ($dcV.Count -gt 0) { ($dcV | Measure-Object -Average).Average } else { 0 }
    Write-Host ("  {0,-12} prefill: {1,6:N2}x   decode: {2,6:N2}x" -f $mod.Name, $aPf, $aDc)
}

# Save to project root
$csvPath = "$OutDir\benchmark_results.csv"
$Results | Export-Csv -Path $csvPath -NoTypeInformation
$Results | Format-Table -AutoSize | Out-File "$OutDir\benchmark_table.txt"
Write-Host "`nResults saved to:" -ForegroundColor Green
Write-Host "  $csvPath" -ForegroundColor Green
Write-Host "  $OutDir\benchmark_table.txt" -ForegroundColor Green
