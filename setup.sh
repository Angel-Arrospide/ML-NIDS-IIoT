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
    ./$ENV_DIR/bin/pip install $REQ_TARGET
    ./$ENV_DIR/bin/python src/config.py
    
    echo "Successfully configured $ENV_DIR!"
    echo ""
}

# --- DEFAULT ENV ---
setup_env ".venv" "-e ."

# --- OTHER ENVS ---
setup_env ".venv_resnet" "-r requirements/resnet.txt"


