# ML-NIDS-IIoT

## Descripción
`#ToDo`

## Configuración del Entorno
1. **Clonar el repositorio**:
    ```bash
    git clone git@github.com:Angel-Arrospide/ML-NIDS-IIoT.git
    cd ML-NIDS-IIoT
    ```
2. **Entorno Virtual e Instalación Editable**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # En Windows: .\.venv\Scripts\activate

    # Esto instala dependencias y registra la carpeta 'src'
    pip install -e .
    ```

3. **Configurar el Kernel para Jupyter**
    ```bash
    python -m ipykernel install --user --name=ml_nids_env --display-name "Python (ML-NIDS-IIoT)"
    ```

4. **Inicializar Proyecto**
    ```bash
    python3 src/config.py
    ```
