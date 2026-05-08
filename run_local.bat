@echo off
SETLOCAL EnableDelayedExpansion

:: Lock the root directory
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

echo ============================================================
echo   ReviewerAgent - Local Startup Script (v2.5)
echo   [CONTROLLER MODE]
echo ============================================================

:: 1. Check Ollama
echo [1/4] Checking Ollama API...
curl -s http://localhost:11434/api/tags > nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Ollama is not running. Please start Ollama and try again.
    pause
    exit /b 1
)
echo [OK] Ollama is running.

:: 2. Database
echo [2/4] Starting PostgreSQL database...
docker compose -f "%ROOT_DIR%docker-compose-db.yml" up -d
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to start Docker database. Ensure Docker Desktop is running.
    pause
    exit /b 1
)
echo [OK] Database is up.

:: 3. Start Backend
echo [3/4] Starting Backend (RA_BACKEND)...
set "BACKEND_DIR=%ROOT_DIR%backend"
set "ROOT_VENV=%BACKEND_DIR%\.venv"

if not exist "%BACKEND_DIR%" (
    echo [ERROR] Backend directory not found at %BACKEND_DIR%
    pause
    exit /b 1
)

:: Check if root venv exists
if not exist "%ROOT_VENV%\Scripts\activate.bat" (
    echo [ERROR] Root virtual environment not found at %ROOT_VENV%
    pause
    exit /b 1
)

:: Start in a new window with a specific title for tracking
start "RA_BACKEND" cmd /k "cd /d "%BACKEND_DIR%" && call "%ROOT_VENV%\Scripts\activate" && python main.py"

:: 4. Start Frontend
echo [4/4] Starting Frontend (RA_FRONTEND)...
set "FRONTEND_DIR=%ROOT_DIR%frontend"

if not exist "%FRONTEND_DIR%" (
    echo [ERROR] Frontend directory not found at %FRONTEND_DIR%
    pause
    exit /b 1
)

pushd "%FRONTEND_DIR%"
    set VITE_API_BASE_URL=http://localhost:8000
    set "PKG_MANAGER=npm"
    where pnpm >nul 2>nul && set "PKG_MANAGER=pnpm"
    
    if not exist node_modules (
        echo Installing node dependencies...
        call !PKG_MANAGER! install
    )
    :: Start in a new window with a specific title for tracking
    start "RA_FRONTEND" cmd /k "cd /d "%FRONTEND_DIR%" && !PKG_MANAGER! run dev"
popd

echo ============================================================
echo   SYSTEM ACTIVE!
echo   Services are running in separate windows.
echo ============================================================

timeout /t 5
exit
