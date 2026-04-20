@echo off
cd /d "%~dp0"
TITLE Reef AI - Updater

echo ==========================================
echo    REEF AI - CHECKING FOR UPDATES
echo ==========================================
echo.

:: Check git is installed
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed.
    echo Download from: https://git-scm.com/download/win
    pause
    exit /b
)

echo Current version:
git log --oneline -1
echo.

echo Fetching latest version from GitHub...
git pull origin main
if errorlevel 1 (
    echo.
    echo ERROR: Update failed. Check your internet connection.
    echo If you have local changes, run: git stash
    pause
    exit /b
)

echo.
echo Checking for new dependencies...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate
    pip install -r requirements.txt -q
) else (
    pip install -r requirements.txt -q
)

call npm install --silent

echo.
echo ==========================================
echo  Update complete! 
echo  Run run_reef_ai.bat to start the app.
echo ==========================================
pause
