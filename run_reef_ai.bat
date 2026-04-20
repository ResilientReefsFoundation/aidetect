@echo off
echo.
echo ==========================================
echo    REEF AI - STARTING
echo ==========================================

cd /d "%~dp0"

echo [STEP 1] Checking for Node.js...
where node >nul 2>&1
if errorlevel 1 ( echo [ERROR] Node.js not found. Run setup_dependencies.bat first. & pause & exit /b )

echo [STEP 2] Checking files...
if not exist "app.py"       ( echo [ERROR] app.py MISSING!                                   & pause & exit /b )
if not exist "node_modules" ( echo [ERROR] node_modules MISSING - run setup_dependencies.bat & pause & exit /b )

echo [STEP 3] Checking GPU / Python environment...
if exist ".venv\Scripts\python.exe" (
    echo [OK] Virtual environment found.
    .venv\Scripts\python.exe -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'NONE'; print('[GPU]', name if gpu else 'NOT AVAILABLE - running on CPU (slow)')"
) else (
    echo [WARNING] No virtual environment found - using system Python.
    echo           Run setup_dependencies.bat to create it.
    python -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'NONE'; print('[GPU]', name if gpu else 'NOT AVAILABLE - running on CPU (slow)')" 2>nul || echo [WARNING] PyTorch not found
)

echo [STEP 4] Starting Python AI backend (minimised)...
if exist ".venv\Scripts\activate.bat" (
    powershell -WindowStyle Minimized -Command "Start-Process cmd -ArgumentList '/k', 'cd /d ""%~dp0"" && call .venv\Scripts\activate && python app.py || (echo. && echo [ERROR] Python crashed - see above && pause)' -WindowStyle Minimized"
) else (
    powershell -WindowStyle Minimized -Command "Start-Process cmd -ArgumentList '/k', 'cd /d ""%~dp0"" && python app.py || (echo. && echo [ERROR] Python crashed - see above && pause)' -WindowStyle Minimized"
)

echo [STEP 5] Starting web interface (minimised)...
powershell -WindowStyle Minimized -Command "Start-Process cmd -ArgumentList '/k', 'cd /d ""%~dp0"" && npm run dev' -WindowStyle Minimized"

echo.
echo Waiting for Python backend (up to 30 seconds)...

set TRIES=0
:WAIT_LOOP
set /a TRIES+=1
if %TRIES% GTR 15 (
    echo [WARNING] Could not confirm Python started - opening browser anyway.
    echo           Check the GPU Backend window in the taskbar for errors.
    goto OPEN_BROWSER
)
curl -s --max-time 2 http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo Waiting... %TRIES%/15
    timeout /t 2 /nobreak >nul
    goto WAIT_LOOP
)
echo [OK] Python backend is ready!

:OPEN_BROWSER
timeout /t 1 /nobreak >nul
start "" "http://localhost:3000"

echo.
echo ==========================================
echo  Reef AI is running!
echo  App:       http://localhost:3000
echo  Python AI: http://localhost:5000
echo.
echo  Both terminals are minimised in taskbar.
echo  Close the GPU Backend window to stop AI.
echo ==========================================
pause
