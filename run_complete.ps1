param([switch]$ReportOnly)
$ErrorActionPreference = 'Stop'
$experimentPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $experimentPython)) { throw '请先按 README 安装本目录的 .venv 和依赖。' }
Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath 'results/completion.json')) {
        & ./run.ps1
        if ($LASTEXITCODE -ne 0) { throw '模块 A 未成功完成。' }
    }
    & $experimentPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw '方法测试失败。' }
    if (!$ReportOnly) {
        if (!(Test-Path -LiteralPath 'results/extensions/completion.json')) {
            & $experimentPython src/extensions.py
            if ($LASTEXITCODE -ne 0) { throw '模块 B/C 运行失败。可从完成的模拟种子检查点恢复。' }
        }
        if (!(Test-Path -LiteralPath 'results/extensions/feedback/diagnostic_completion.json')) {
            & $experimentPython src/feedback_diagnostic.py
            if ($LASTEXITCODE -ne 0) { throw '同规模反馈诊断运行失败。' }
        }
    }
    & $experimentPython src/complete_report.py
    if ($LASTEXITCODE -ne 0) { throw '完整报告生成失败。' }
    & $experimentPython src/verify_extensions.py
    if ($LASTEXITCODE -ne 0) { throw '新增实验独立核验失败。' }
} finally { Pop-Location }
