# PE-MMNet v4 Team Training 启动脚本
# 兼容 PowerShell 5.1+ 和 PowerShell 7+

# 先放宽当前进程 ExecutionPolicy（不影响全局，仅本次）
try { Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force -ErrorAction SilentlyContinue } catch {}

$ErrorActionPreference = 'Stop'

$PROJECT_ROOT = $PSScriptRoot
$CONDA_PY = Join-Path $env:USERPROFILE '.conda\envs\pe_mmnet\python.exe'
$CMD_LOG = Join-Path $PROJECT_ROOT 'logs\cmd_run.log'

# 窗口标题
try { $host.UI.RawUI.WindowTitle = 'PE-MMNet v4 Team Training' } catch {}

# banner
Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '  PE-MMNet v4 Team Training  (conda env: pe_mmnet)'
Write-Host ('  Project: ' + $PROJECT_ROOT)
Write-Host ('  Python : ' + $CONDA_PY)
Write-Host ('  Log    : ' + $CMD_LOG)
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ''

# 解释器检查
if (-not (Test-Path $CONDA_PY)) {
    Write-Host ('[ERROR] conda python.exe not found: ' + $CONDA_PY) -ForegroundColor Red
    Read-Host 'Press Enter to exit'
    exit 1
}

# 解释器版本预览
$pyVer = & $CONDA_PY -c 'import sys; print(sys.version.split()[0])'
Write-Host ('[OK] Python ' + $pyVer) -ForegroundColor Green

# 启动前确认
Read-Host 'Press Enter to start training (Ctrl+C to cancel)'

# 确保日志目录存在
$logDir = Split-Path $CMD_LOG -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

# Python 环境变量
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

Set-Location $PROJECT_ROOT

Write-Host ''
Write-Host 'Starting team_train.py --auto ...' -ForegroundColor Cyan
Add-Content -Path $CMD_LOG -Value ('[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] Starting team_train.py --auto') -Encoding utf8
Write-Host ''

# 跑训练：控制台实时输出 + 同步写日志
$lines = & $CONDA_PY (Join-Path $PROJECT_ROOT 'team_train.py') --auto 2>&1
foreach ($line in $lines) {
    Write-Host $line
    Add-Content -Path $CMD_LOG -Value $line -Encoding utf8
}

$exitCode = $LASTEXITCODE

Write-Host ''
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ('  team_train.py exited with code ' + $exitCode)
Write-Host ('  See ' + $CMD_LOG + ' for full log')
Write-Host '============================================================' -ForegroundColor Cyan
Add-Content -Path $CMD_LOG -Value ('[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') + '] team_train.py exited with code ' + $exitCode) -Encoding utf8

Read-Host 'Press Enter to exit'
