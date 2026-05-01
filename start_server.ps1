# Amlhere Server - Background Start Script
# Survives SSH session disconnection

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $ProjectDir "logs"
$PidFile = Join-Path $ProjectDir "server.pid"
$ErrorLogFile = Join-Path $LogDir ("server_error_" + (Get-Date -Format "yyyy-MM-dd") + ".log")
$OutLogFile = Join-Path $LogDir ("server_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

if (Test-Path $PidFile) {
    $oldPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($oldPid) {
        $proc = Get-Process -Id $oldPid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[!] Server already running (PID: $oldPid)"
            Write-Host "    Stop with: .\stop_server.ps1"
            exit 1
        }
    }
}

$envFile = Join-Path $ProjectDir ".env.local"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and (-not $line.StartsWith("#")) -and $line.Contains("=")) {
            $idx = $line.IndexOf("=")
            $key = $line.Substring(0, $idx).Trim()
            $val = $line.Substring($idx + 1).Trim()
            [System.Environment]::SetEnvironmentVariable($key, $val, "Process")
        }
    }
    Write-Host "[*] Loaded .env.local"
}

$process = Start-Process -FilePath "python" -ArgumentList "-m","uvicorn","main:app","--host","0.0.0.0","--port","8000" -WorkingDirectory $ProjectDir -WindowStyle Hidden -RedirectStandardOutput $OutLogFile -RedirectStandardError $ErrorLogFile -PassThru

$process.Id | Out-File -FilePath $PidFile -NoNewline

Write-Host ""
Write-Host "[OK] Server started!"
Write-Host "  PID  : $($process.Id)"
Write-Host "  URL  : http://0.0.0.0:8000"
Write-Host "  Log  : $OutLogFile"
Write-Host ""
Write-Host "[*] Server will keep running after SSH disconnect"
Write-Host "[*] Stop : .\stop_server.ps1"
Write-Host "[*] Status: .\status_server.ps1"
Write-Host ""
