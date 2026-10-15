@echo off
setlocal enabledelayedexpansion
set models=Bonsai-1.7B-Q1_0.gguf Ternary-Bonsai-1.7B-Q2_0.gguf Bonsai-4B.gguf Ternary-Bonsai-4B-Q2_0.gguf Bonsai-8B.gguf Ternary-Bonsai-8B-Q2_0.gguf
set base=%USERPROFILE%\AppData\Roaming\hearth\models\
cd /d C:\github\hearth
for %%m in (%models%) do (
    echo === CPU: %%m ===
    .\target\release\hearth-chat-cli.exe "%base%%%m" --temp 0 --max-tokens 50 --prompt "Hello" --prompt-raw 2>&1 | findstr "tokens prefill avg_cpu"
    echo.
)
