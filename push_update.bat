@echo off
cd /d "%~dp0"
TITLE Reef AI - Push Update to GitHub

:: Get current version from App.tsx
for /f "tokens=*" %%i in ('findstr /r "v[0-9]*\.[0-9]*" src\App.tsx') do (
    set LINE=%%i
    goto :found
)
:found

echo Pushing latest changes to GitHub...
git add .
git diff --cached --stat
echo.
set /p MSG="Enter commit message (or press Enter for auto): "
if "%MSG%"=="" (
    git commit -m "Auto update"
) else (
    git commit -m "%MSG%"
)
git push
echo.
echo Done! Changes are live on GitHub.
pause
