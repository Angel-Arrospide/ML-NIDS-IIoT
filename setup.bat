:setup_env
set "ENV_DIR=.venv"
set "REQ_TARGET=-e ."

echo ======================================
echo Setting up environment: %ENV_DIR%
echo ======================================

python -m venv "%ENV_DIR%"
"%ENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip

:: legacy env must NOT install -e . — pyproject.toml requires numpy>=2.4
:: which conflicts with catboost/sktime. Install project without deps instead.
if "%ENV_DIR%"==".venv_legacy" (
    "%ENV_DIR%\Scripts\pip.exe" install --no-deps -e .
    "%ENV_DIR%\Scripts\pip.exe" install %REQ_TARGET%
) else (
    "%ENV_DIR%\Scripts\pip.exe" install -e .
    if not "%REQ_TARGET%"=="-e ." (
        "%ENV_DIR%\Scripts\pip.exe" install %REQ_TARGET%
    )
)

"%ENV_DIR%\Scripts\python.exe" src\config.py

echo Successfully configured %ENV_DIR%!
echo.
pause