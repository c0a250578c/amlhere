# Amlhere Server - Stop Script

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PidFile = Join-Path $ProjectDir "server.pid"

Write-Host ""

if (-not (Test-Path $PidFile)) {
    Write-Host "[!] PID file not found. Server may not be running."
    $listening = netstat -ano | Select-String ":8000.*LISTENING"
    if ($listening) {
        Write-Host "[*] Processes listening on port 8000:"
        $listening | ForEach-Object { Write-Host "    $_" }
        Write-Host "[*] Manual stop: Stop-Process -Id <PID> -Force"
    }
    exit 1
}

$serverPid = Get-Content $PidFile -ErrorAction SilentlyContinue
if (-not $serverPid) {
    Write-Host "[!] PID file is empty"
    Remove-Item $PidFile -ErrorAction SilentlyContinue
    exit 1
}

$proc = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
if ($proc) {
    Write-Host "[*] Stopping server... (PID: $serverPid)"
    Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $serverPid } | ForEach-Object {
        Write-Host "    Stopping child process: PID $($_.ProcessId)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    $check = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
    if ($check) {
        Write-Host "[!] Process still running. Force killing..."
        Stop-Process -Id $serverPid -Force
    }
    Write-Host "[OK] Server stopped"
} else {
    Write-Host "[*] Process (PID: $serverPid) already stopped"
}

Remove-Item $PidFile -ErrorAction SilentlyContinue
Write-Host ""
