# Starts the voice stack detached: speakwright on 8765 (ear + whisper on the
# GPU + TTS) and the cloned voice on 8766. Safe to run again: a port that is
# already listening is left alone. Logs land next to this script.
#
#   powershell -ExecutionPolicy Bypass -File D:\speakwright\start-voice.ps1
#
# Keep this file pure ASCII: PowerShell 5.1 misreads a BOM-less UTF-8 file.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Start-Process with redirected output gives the child a cp1252 stdout, and
# the first emoji in a log line would kill it. utf-8 everywhere.
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Listening($port) {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    return ($null -ne $c)
}

if (Listening 8765) {
    Write-Host "speakwright already listening on 8765"
} else {
    $env:VOICE_PORT = "8765"
    $env:WHISPER_DEVICE = "cuda"
    $env:WHISPER_MODEL_SIZE = "small.en"
    $env:DEFAULT_VOICE = "nate"
    Start-Process -FilePath "$root\.venv\Scripts\python.exe" -ArgumentList "app.py" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$root\speakwright.log" -RedirectStandardError "$root\speakwright.err.log"
    Write-Host "speakwright starting on 8765 (whisper small.en on cuda, default voice nate)"
}

if (Listening 8766) {
    Write-Host "voice clone already listening on 8766"
} else {
    Start-Process -FilePath "$root\.venv-clone\Scripts\python.exe" -ArgumentList "clone.py" -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput "$root\clone.log" -RedirectStandardError "$root\clone.err.log"
    Write-Host "voice clone starting on 8766 (first line loads the model, 30 to 60 s)"
}
