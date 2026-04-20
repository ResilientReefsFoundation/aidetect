@echo off
cd /d "%~dp0"
TITLE Reef AI - Dependency Installer

echo ==========================================
echo    REEF AI - INSTALLING DEPENDENCIES
echo ==========================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in your PATH.
    echo Please install Python 3.10+ from python.org
    pause
    exit /b
)

:: Check for Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in your PATH.
    echo Please install Node.js (LTS version) from nodejs.org
    pause
    exit /b
)

echo.
echo [1/4] Creating Python virtual environment...
if not exist ".venv" (
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists, skipping creation.
)

echo.
echo [2/4] Installing PyTorch with CUDA GPU support...
echo      (This is ~2-3 GB — please wait, do not close this window)
call .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo WARNING: CUDA PyTorch install failed. Trying CPU-only fallback...
    pip install torch torchvision
)

echo.
echo [3/4] Installing Python libraries...
pip install -r requirements.txt
pip install icrawler
pip install yt-dlp
if errorlevel 1 (
    echo ERROR: pip install failed. Check your internet connection.
    pause
    exit /b
)

echo.
echo [4/4] Installing web interface dependencies...
call npm install
if errorlevel 1 (
    echo ERROR: npm install failed. Check your internet connection.
    pause
    exit /b
)

echo.
echo ------------------------------------------
echo SUCCESS: All dependencies installed!
echo GPU (CUDA) support enabled for fast training.
echo.
echo You can now use 'run_reef_ai.bat' to start the app.
echo ------------------------------------------
echo.
pause
exit /b
