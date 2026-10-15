param([switch]$CheckOnly)
$ErrorActionPreference = 'Stop'
$experimentRoot = $PSScriptRoot
$localPython = Join-Path $experimentRoot '.venv/Scripts/python.exe'
$existingPython = Join-Path (Split-Path $experimentRoot -Parent) 'loan_fairness_experiment/.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $localPython) { $experimentPython = $localPython }
elseif (Test-Path -LiteralPath $existingPython) { $experimentPython = $existingPython }
else { throw '请使用 Python 3.12 创建 .venv，并安装 requirements.txt 中的依赖。' }
Push-Location $experimentRoot
try {
    & $experimentPython src/experiment.py --check-data
    if ($LASTEXITCODE -ne 0) { throw '数据检查未通过。请查看 data/raw/README.md。' }
    if ($CheckOnly) { return }
    & $experimentPython -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) { throw '方法测试失败。' }
    & $experimentPython src/experiment.py
    if ($LASTEXITCODE -ne 0) { throw '实验运行失败；保留检查点，可修复后重新运行。' }
    & $experimentPython src/report.py
    if ($LASTEXITCODE -ne 0) { throw '报告生成失败。' }
    & $experimentPython src/verify.py
    if ($LASTEXITCODE -ne 0) { throw '独立结果核验失败。' }
} finally { Pop-Location }
