#!/bin/bash

# --- FUNCTION ---
setup_env() {
    local ENV_DIR=$1
    local REQ_TARGET=$2

    echo "======================================"
    echo "Setting up environment: $ENV_DIR"
    echo "======================================"
    
    python3 -m venv $ENV_DIR
    ./$ENV_DIR/bin/python -m pip install --upgrade pip
    
    ./$ENV_DIR/bin/pip install -e .
    
    if [ "$REQ_TARGET" != "-e ." ]; then
        ./$ENV_DIR/bin/pip install $REQ_TARGET
    fi
    
    ./$ENV_DIR/bin/python src/config.py
    
    echo "Successfully configured $ENV_DIR!"
    echo ""
}

# --- SECONDARY ENVS ---
for req_file in requirements/*.txt; do
    [ -e "$req_file" ] || continue
    
    filename=$(basename -- "$req_file") # Get filename with extension
    name_no_ext="${filename%.*}" # Remove extension
    
    env_name=".venv_$name_no_ext"
    setup_env "$env_name" "-r $req_file"
done

# --- DEFAULT ENV ---
setup_env ".venv" "-e ."

# --------------------


echo "--- DONE ---"