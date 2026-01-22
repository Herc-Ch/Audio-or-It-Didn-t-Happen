# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

# Activate venv
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1

# Install Python dependencies from pyproject.toml
Write-Host "Installing Python packages..." -ForegroundColor Cyan
pip install --upgrade pip
pip install -e .

# Check/install pandoc
$pandocInstalled = Get-Command pandoc -ErrorAction SilentlyContinue
if (-not $pandocInstalled) {
    Write-Host "Pandoc not found. Installing via winget..." -ForegroundColor Yellow
    
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Error: winget not found. Please install Windows App Installer." -ForegroundColor Red
        exit 1
    }
    
    winget install --id JohnMacFarlane.Pandoc --accept-package-agreements --accept-source-agreements
    Write-Host "Pandoc installed. You may need to restart your terminal." -ForegroundColor Green
} else {
    Write-Host "✓ Pandoc is already installed" -ForegroundColor Green
}

# Check/install ffmpeg (required for pydub/audio processing)
$ffmpegInstalled = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpegInstalled) {
    Write-Host "ffmpeg not found. Installing via winget..." -ForegroundColor Yellow
    
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "Error: winget not found. Please install Windows App Installer." -ForegroundColor Red
        exit 1
    }
    
    winget install --id ffmpeg --accept-package-agreements --accept-source-agreements
    Write-Host "ffmpeg installed. You may need to restart your terminal." -ForegroundColor Green
} else {
    Write-Host "✓ ffmpeg is already installed" -ForegroundColor Green
}

Write-Host "`nSetup complete! Run: python main.py" -ForegroundColor Green
Write-Host "  For TTS: python main.py --tts" -ForegroundColor Cyan



