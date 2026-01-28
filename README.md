# ML-NIDS-IIoT

## Descripción
`#ToDo`

## Configuración del Entorno
1. Clonar el repositorio
```bash
git clone git@github.com:Angel-Arrospide/ML-NIDS-IIoT.git
cd ML-NIDS-IIoT
```
2. Crear y activar el entorno virtual
```bash
# Crear el entorno
python3 -m venv .venv

# Activar el entorno
# En Linux/macOS:
source .venv/bin/activate
# En Windows:
# .\.venv\Scripts\activate
```
3. Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Configurar el Kernel para Jupyter
```bash
python -m ipykernel install --user --name=ml_nids_env --display-name "Python (ML-NIDS-IIoT)"
```