@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :run

set "PYEXE="
where py >nul 2>nul
if %errorlevel%==0 (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
)
if not defined PYEXE (
    where python >nul 2>nul
    if %errorlevel%==0 (
        for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do set "PYEXE=%%P"
    )
)
if not defined PYEXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYEXE (
    for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do set "PYEXE=%%D\python.exe"
)

if not defined PYEXE goto :error

echo Creating virtual environment...
"%PYEXE%" -m venv .venv
if errorlevel 1 goto :error

echo Installing PyTorch (CUDA 12.1)...
".venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
if errorlevel 1 goto :error

echo Installing Transformers and safety model dependencies...
".venv\Scripts\python.exe" -m pip install transformers accelerate safetensors tokenizers
if errorlevel 1 goto :error

echo Installing application dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

:run
echo.
echo Starting Ruas AI at http://127.0.0.1:5000
echo Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" app.py
goto :eof

:error
echo.
echo Could not find Python 3.10+. Install it from https://python.org and try again.
pause
