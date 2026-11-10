param([switch]$OfflineTest, [switch]$OpenBrowser)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = Join-Path $taskRoot 'src'
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw '请先运行 scripts\setup.ps1 创建项目环境。' }
if (-not $OfflineTest) { & $taskPython -m agentic_kb start-ollama; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }
$taskArgs = @('-m','agentic_kb','serve')
if ($OfflineTest) { $taskArgs += '--offline-test' }
if ($OpenBrowser) { $taskArgs += '--open-browser' }
& $taskPython @taskArgs

