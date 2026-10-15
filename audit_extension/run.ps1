$ErrorActionPreference = 'Stop'
$experimentPython = Join-Path $PSScriptRoot '../.venv/Scripts/python.exe'
Push-Location $PSScriptRoot
try {
    & $experimentPython -X utf8 -m unittest test_methods -v
    if ($LASTEXITCODE -ne 0) { throw '方法测试失败' }
    if (!(Test-Path -LiteralPath 'results/completion.json')) {
        & $experimentPython -X utf8 -u run_experiment.py
        if ($LASTEXITCODE -ne 0) { throw '实验执行失败；保留检查点以便恢复' }
    }
    & $experimentPython -X utf8 verify.py
    if ($LASTEXITCODE -ne 0) { throw '独立核验失败' }
    & $experimentPython -X utf8 report.py
    if ($LASTEXITCODE -ne 0) { throw '报告生成失败' }
} finally { Pop-Location }
