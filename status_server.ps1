# Amlhere Server - Status Check Script

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ProjectDir "server.pid"
$LogDir = Join-Path $ProjectDir "logs"

Write-Host ""
Write-Host "=== Amlhere Server Status ==="
Write-Host ""

if (Test-Path $PidFile) {
    $serverPid = Get-Content $PidFile -ErrorAction SilentlyContinue
    if ($serverPid) {
        $proc = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "[OK] Server is RUNNING"
            Write-Host "  PID     : $serverPid"
            Write-Host "  Process : $($proc.ProcessName)"
            Write-Host "  Memory  : $([math]::Round($proc.WorkingSet64 / 1MB, 1)) MB"
            Write-Host "  CPU     : $($proc.TotalProcessorTime)"
            Write-Host "  Started : $($proc.StartTime)"
        } else {
            Write-Host "[!] PID file exists but process not found (PID: $serverPid)"
            Write-Host "    Server may have crashed"
        }
    }
} else {
    Write-Host "[!] Server is STOPPED (no PID file)"
}

Write-Host ""
Write-Host "--- Port 8000 ---"
$listening = netstat -ano | Select-String ":8000.*LISTENING"
if ($listening) {
    $listening | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "  No process listening on port 8000"
}

Write-Host ""
Write-Host "--- Health Check ---"
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8000/health" -TimeoutSec 3 -ErrorAction Stop
    Write-Host "  Status : $($response.status)" -ForegroundColor Green
    Write-Host "  Gemini : $($response.gemini_available)"
} catch {
    Write-Host "  Health check failed: Cannot connect to server"
}

Write-Host ""
Write-Host "--- Recent Logs (last 10 lines) ---"
$ts = Get-Date -Format "yyyy-MM-dd"
$errLog = Join-Path $LogDir ("server_error_" + $ts + ".log")
$outLog = Join-Path $LogDir ("server_" + $ts + ".log")
if (Test-Path $errLog) {
    Get-Content $errLog -Tail 10 | ForEach-Object { Write-Host "  $_" }
} elseif (Test-Path $outLog) {
    Get-Content $outLog -Tail 10 | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "  No log files found"
}

Write-Host ""
