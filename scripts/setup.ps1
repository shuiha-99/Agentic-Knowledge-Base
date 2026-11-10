param([switch]$WithModels, [switch]$WithEvaluation)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$env:UV_CACHE_DIR = Join-Path $taskRoot 'runtime\cache\uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $taskRoot 'runtime\cache\python'
$env:PYTHONUTF8 = '1'
$taskBuildTemp = Join-Path $taskRoot 'work\build_tmp'
$taskPreviousTemp = $env:TEMP
$taskPreviousTmp = $env:TMP
$taskPreviousLinkMode = $env:UV_LINK_MODE
try {
    New-Item -ItemType Directory -Path $taskBuildTemp -Force | Out-Null
    $env:TEMP = $taskBuildTemp
    $env:TMP = $taskBuildTemp
    $env:UV_LINK_MODE = 'copy'
    $taskArgs = @('sync','--locked','--extra','dev','--no-editable','--default-index','https://pypi.org/simple')
    if ($WithModels) { $taskArgs += @('--extra','models') }
    & uv @taskArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if ($WithEvaluation) {
        & uv venv '.venv-ragas' --python '3.11'
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & uv pip sync '--python' '.venv-ragas\Scripts\python.exe' '--default-index' 'https://pypi.org/simple' 'docs\requirements-ragas.lock.txt'
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
    if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath '.env.example' -Destination '.env' }
    Write-Output '环境已准备完成。真实模型：python -m agentic_kb prepare-models；界面：scripts\start.ps1。'
}
finally {
    $env:TEMP = $taskPreviousTemp
    $env:TMP = $taskPreviousTmp
    $env:UV_LINK_MODE = $taskPreviousLinkMode
}
