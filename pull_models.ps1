# Pull the chat and vision models used by Ruas AI.
$ErrorActionPreference = "Stop"

$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($ollama) {
    $ollamaPath = $ollama.Source
} else {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe",
        "$env:ProgramFiles\Ollama\ollama.exe"
    )
    $ollamaPath = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $ollamaPath) {
    Write-Host "Ollama was not found. Install it from https://ollama.com/download and run this again." -ForegroundColor Red
    exit 1
}

Write-Host "Using $ollamaPath" -ForegroundColor Cyan
Write-Host "Pulling llama3.2:3b (fast text model)..." -ForegroundColor Cyan
& $ollamaPath pull llama3.2:3b
Write-Host "Pulling qwen3-vl:8b (vision model for images and files)..." -ForegroundColor Cyan
& $ollamaPath pull qwen3-vl:8b

Write-Host ""
Write-Host "Done. Installed models:" -ForegroundColor Green
& $ollamaPath list
