#!/bin/bash

# --- FUNCTION ---
setup_env() {
    local ENV_DIR="$1"
    local REQ_TARGET="$2"

    echo "======================================"
    echo "Setting up environment: $ENV_DIR"
    echo "======================================"

    python3 -m venv "$ENV_DIR"
    "$ENV_DIR/bin/python" -m pip install --upgrade pip

    # legacy env: install project WITHOUT deps so pyproject.toml constraints
    # (numpy>=2.4, pandas>=3.0.2) don't conflict with catboost/sktime pins.
    if [ "$ENV_DIR" = ".venv_legacy" ]; then
        "$ENV_DIR/bin/pip" install --no-deps -e .
    else
        "$ENV_DIR/bin/pip" install -e .
    fi

    if [ "$REQ_TARGET" != "-e ." ]; then
        "$ENV_DIR/bin/pip" install $REQ_TARGET
    fi

    "$ENV_DIR/bin/python" src/config.py

    echo "Successfully configured $ENV_DIR!"
    echo ""
}

# --- SECONDARY ENVS ---
for req_file in requirements/*.txt; do
    [ -e "$req_file" ] || continue

    filename=$(basename -- "$req_file")   # filename with extension
    name_no_ext="${filename%.*}"          # strip extension

    env_name=".venv_${name_no_ext}"
    setup_env "$env_name" "-r $req_file"
done

# --- DEFAULT ENV ---
setup_env ".venv" "-e ."

echo "--- DONE ---"