param(
    [ValidateSet('setup','run','report','verify','all')]
    [string]$Action = 'run'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$pythonV3 = Join-Path $PSScriptRoot '.venv-v3/Scripts/python.exe'
function Invoke-V3Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArgs)
    & $pythonV3 @PythonArgs
    if ($LASTEXITCODE -ne 0) { throw ('Experiment command failed: '+($PythonArgs -join ' ')) }
}
if ($Action -in @('setup','all')) {
    if (-not $env:HTTPS_PROXY) {
        $proxySettings = Get-ItemProperty -LiteralPath 'HKCU:/Software/Microsoft/Windows/CurrentVersion/Internet Settings' -ErrorAction SilentlyContinue
        if ($proxySettings.ProxyEnable -eq 1 -and $proxySettings.ProxyServer -match '^([\w.-]+):(\d+)$') {
            $env:HTTPS_PROXY = 'http://'+$proxySettings.ProxyServer
            $env:HTTP_PROXY = $env:HTTPS_PROXY
        }
    }
    if (-not (Test-Path -LiteralPath $pythonV3)) {
        if (Get-Command py -ErrorAction SilentlyContinue) { py -3.12 -m venv '.venv-v3' }
        elseif (Get-Command python -ErrorAction SilentlyContinue) {
            python -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
            if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 before setup' }
            python -m venv '.venv-v3'
        }
        else { throw 'Install Python 3.12 and add py or python to PATH before setup' }
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 virtual environment creation failed' }
    }
    Invoke-V3Python -m pip install 'torch==2.7.0+cu128' --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
    Invoke-V3Python -m pip install -r research_extension_v3/requirements.txt
    Invoke-V3Python -m research_extension_v3.setup_baselines
}
if (-not (Test-Path -LiteralPath $pythonV3)) { throw 'Run .\run_research_v3.ps1 setup first' }
if ($Action -in @('run','all')) {
    Invoke-V3Python -m research_extension_v3.prepare
    Invoke-V3Python -m pytest research_extension_v3/tests research_extension_v3/test_advanced_models.py -q
    Invoke-V3Python -m research_extension_v3.run --models xgboost,lightgbm,logistic,m2fgb,m2fgb_tuned,tabm --seeds 731,732,733
    Invoke-V3Python -m research_extension_v3.mechanisms
    Invoke-V3Python -m research_extension_v3.training_objective
    Invoke-V3Python -m research_extension_v3.fairlearn_business
}
if ($Action -in @('run','report','all')) {
    Invoke-V3Python -m research_extension_v3.provenance
    Invoke-V3Python -m research_extension_v3.run --aggregate
    Invoke-V3Python -m research_extension_v3.inference --run research_extension_v3/results/run_001 --repetitions 1000
    Invoke-V3Python -m research_extension_v3.report --run-dir research_extension_v3/results/run_001
}
if ($Action -in @('run','verify','all')) {
    Invoke-V3Python -m research_extension_v3.verify --run-dir research_extension_v3/results/run_001 --require-complete
}
