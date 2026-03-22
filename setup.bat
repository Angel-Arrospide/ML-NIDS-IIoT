@echo off
setlocal enabledelayedexpansion

:: --- SECONDARY ENVS ---
for %%f in ("requirements\*.txt") do (
    set "filename=%%~nxf"
    set "name_no_ext=%%~nf"
    set "req_file=%%f"
    
    set "env_name=.venv_!name_no_ext!"
    call :setup_env "!env_name!" "-r !req_file!"
)

:: --- DEFAULT ENV ---
call :setup_env ".venv" "-e ."

echo --- DONE ---
goto :EOF

:: --- FUNCTION ---
:setup_env
set "ENV_DIR=%~1"
set "REQ_TARGET=%~2"

echo ======================================
echo Setting up environment: %ENV_DIR%
echo ======================================

python -m venv "%ENV_DIR%"
"%ENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip

"%ENV_DIR%\Scripts\pip.exe" install -e .

if not "%REQ_TARGET%"=="-e ." (
    "%ENV_DIR%\Scripts\pip.exe" install %REQ_TARGET%
)

"%ENV_DIR%\Scripts\python.exe" src\config.py

echo Successfully configured %ENV_DIR%!
echo.
exit /b
