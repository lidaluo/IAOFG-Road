# 全量 City-scale test 评估（后台日志），睡醒后看 eval_outputs 下文件。
# 用法（在仓库根 E:\Code\IAOF-Graph）:
#   powershell -ExecutionPolicy Bypass -File scripts/run_test_eval_overnight.ps1
# 若未自动 conda activate，请先手动: conda activate road_extraction

$ErrorActionPreference = "Stop"
# 本文件在 scripts/ 下，仓库根为上一级
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root

$OutDir = Join-Path $Root "eval_outputs"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stdout = Join-Path $OutDir "test_eval_bg_stdout.log"
$stderr = Join-Path $OutDir "test_eval_bg_stderr.log"
$jsonl = Join-Path $OutDir "test_per_tile.jsonl"
$summary = Join-Path $OutDir "test_split_full_summary.json"

Write-Host "WorkingDirectory: $Root"
Write-Host "Logs: $stdout , $stderr"

$p = Start-Process -FilePath "python" -ArgumentList @(
    "scripts\evaluate_cityscale.py",
    "--config", "configs\cityscale_iaofpp.yaml",
    "--split", "test",
    "--no-refine",
    "--summary-json", "eval_outputs\test_split_full_summary.json",
    "--per-tile-jsonl", "eval_outputs\test_per_tile.jsonl"
) -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru

Write-Host "Started PID=$($p.Id). 查看进度: Get-Content $stdout -Tail 20 -Wait"
