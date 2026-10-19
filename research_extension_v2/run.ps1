$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
Set-Location (Split-Path $PSScriptRoot -Parent)
$python = Join-Path (Get-Location) '.venv/Scripts/python.exe'
if (!(Test-Path $python)) { throw 'Create .venv and install requirements.txt first; see research_extension_v2/README.md.' }
& $python -m compileall -q research_extension_v2
if ($LASTEXITCODE -ne 0) { throw 'Static compilation failed' }
& $python -m unittest discover -s research_extension_v2/tests -v
if ($LASTEXITCODE -ne 0) { throw 'Method tests failed' }
if (!(Test-Path 'research_extension_v2/results/run_001/completion.json')) {
    & $python -m research_extension_v2.restore_data
    if ($LASTEXITCODE -ne 0) { throw 'Data recovery failed' }
    & $python -m research_extension_v2.run
    if ($LASTEXITCODE -ne 0) { throw 'Experiment failed; preserve failure evidence and choose a fresh output directory' }
}
& $python -m research_extension_v2.verify
if ($LASTEXITCODE -ne 0) { throw 'Independent verification failed' }
& $python -m research_extension_v2.report
if ($LASTEXITCODE -ne 0) { throw 'Report failed' }
