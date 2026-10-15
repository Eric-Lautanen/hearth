@echo off
setlocal enabledelayedexpansion

set models=1.7B-Q1_0 1.7B-Q2_0 4B-Q1_0 4B-Q2_0 8B-Q1_0 8B-Q2_0
set files=Bonsai-1.7B-Q1_0.gguf Ternary-Bonsai-1.7B-Q2_0.gguf Bonsai-4B.gguf Ternary-Bonsai-4B-Q2_0.gguf Bonsai-8B.gguf Ternary-Bonsai-8B-Q2_0.gguf
set base=%USERPROFILE%\AppData\Roaming\hearth\models\
cd /d C:\github\hearth

set idx=0
for %%f in (%files%) do (
    call :setname %%f
)

REM Just run each model manually
echo === CPU: Bonsai-1.7B-Q1_0.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Bonsai-1.7B-Q1_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu"
echo === CPU: Ternary-Bonsai-1.7B-Q2_0.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Ternary-Bonsai-1.7B-Q2_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu"
echo === CPU: Bonsai-4B.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Bonsai-4B.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu"
echo === CPU: Ternary-Bonsai-4B-Q2_0.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Ternary-Bonsai-4B-Q2_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu"
echo === CPU: Bonsai-8B.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Bonsai-8B.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu"
echo === CPU: Ternary-Bonsai-8B-Q2_0.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Ternary-Bonsai-8B-Q2_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu"

echo.
echo ===============================
echo GPU BENCHMARKS (with --gpu)
echo ===============================

echo === GPU: Bonsai-1.7B-Q1_0.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Bonsai-1.7B-Q1_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw --gpu 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu\|use_gpu"
echo === GPU: Ternary-Bonsai-1.7B-Q2_0.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Ternary-Bonsai-1.7B-Q2_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw --gpu 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu\|use_gpu"
echo === GPU: Bonsai-4B.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Bonsai-4B.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw --gpu 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu\|use_gpu"
echo === GPU: Ternary-Bonsai-4B-Q2_0.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Ternary-Bonsai-4B-Q2_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw --gpu 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu\|use_gpu"
echo === GPU: Bonsai-8B.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Bonsai-8B.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw --gpu 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu\|use_gpu"
echo === GPU: Ternary-Bonsai-8B-Q2_0.gguf ===
.\target\release\hearth-chat-cli.exe "%base%Ternary-Bonsai-8B-Q2_0.gguf" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw --gpu 2>&1 | findstr /r "tokens.*ms.*tok\|prefill.*tokens\|avg_cpu\|use_gpu"
