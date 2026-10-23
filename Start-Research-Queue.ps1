$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$pythonQueue = Join-Path $projectRoot '.venv-v3/Scripts/python.exe'
$queueDir = Join-Path $projectRoot 'research_extension_v3/continuation_queue'
New-Item -ItemType Directory -Path $queueDir -Force | Out-Null
$queueState = Join-Path $queueDir 'queue.json'
if (Test-Path -LiteralPath $queueState) {
    $state = Get-Content -LiteralPath $queueState -Raw | ConvertFrom-Json
    if ($state.status -eq 'COMPLETE') { Write-Output 'All queue tasks already complete.'; exit 0 }
    if ($state.worker_pid) {
        $existing = Get-CimInstance Win32_Process -Filter ('ProcessId='+$state.worker_pid) -ErrorAction SilentlyContinue
        if ($existing -and $existing.CommandLine -like '*research_extension_v3.background_queue*') {
            Write-Output ('Queue already running with PID '+$state.worker_pid);exit 0
        }
    }
}
$process = Start-Process -FilePath $pythonQueue -ArgumentList '-u','-X','utf8','-m','research_extension_v3.background_queue' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $queueDir 'worker.stdout.log') -RedirectStandardError (Join-Path $queueDir 'worker.stderr.log')
Write-Output ('Started background queue PID '+$process.Id)
