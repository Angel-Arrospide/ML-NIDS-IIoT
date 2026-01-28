import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

# =============================================================================
#  CONFIGURACIONES
# =============================================================================

# Configuración visual de Rich
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "path": "underline blue"
})
console = Console(theme=custom_theme)

# Cargar BASE_PATH desde .env o definirlo si se ejecuta directamente
if __name__ == "__main__":
    BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
else:
    load_dotenv()
    BASE_PATH = os.getenv("PROJECT_BASE_PATH")
    if not BASE_PATH:
        BASE_PATH = None

# Definición de variables requeridas para el archivo .env
REQUIRED_VARS = {
    "PROJECT_BASE_PATH": BASE_PATH,
    #"KAGGLE_USERNAME": "tu_usuario",
    #"KAGGLE_KEY": "tu_key"
}

# Configuración de Datasets (Depende de BASE_PATH)
DATASETS = {
    "edge_iiot": {
        "url": "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
        "raw_path": os.path.join(BASE_PATH, "data", "raw", "edge_iiot") if BASE_PATH else "N/A",
        "processed_path": os.path.join(BASE_PATH, "data", "processed", "edge_iiot") if BASE_PATH else "N/A",
    },
}

# =============================================================================
#  FUNCIONES
# =============================================================================

def manage_env_variables(root_path):
    """Verifica, crea o repara el archivo .env según las necesidades."""
    env_path = os.path.join(root_path, ".env")
    current_vars = {}

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    current_vars[key] = value

    update_needed = False
    for key, def_val in REQUIRED_VARS.items():
        if key not in current_vars or (key == "PROJECT_BASE_PATH" and current_vars[key] != root_path):
            update_needed = True
            break

    if update_needed:
        console.print("[warning]Configuración de entorno incompleta. Sincronizando .env...[/warning]")
        with open(env_path, "w") as f:
            for key, default_value in REQUIRED_VARS.items():
                value = current_vars.get(key, default_value)
                if key == "PROJECT_BASE_PATH": value = root_path
                
                if value in ["tu_usuario", "tu_key"]:
                    f.write(f"# {key}={value}\n")
                else:
                    f.write(f"{key}={value}\n")
        
        console.print("[success]Archivo .env actualizado correctamente.[/success]")
        load_dotenv(override=True)
    else:
        console.print("[info]Variables de entorno verificadas.[/info]")

def init_project_structure():
    """Crea la jerarquía de carpetas necesaria basada en DATASETS."""
    if not BASE_PATH:
        console.print("[error][ERROR] No se puede inicializar la estructura sin PROJECT_BASE_PATH.[/error]")
        return

    console.print(Panel(f"[bold info]Inicializando Proyecto[/bold info]\n[path]{BASE_PATH}[/path]", expand=False))
    
    # Lista de carpetas base
    folders = [
        os.path.join(BASE_PATH, "data", "raw"),
        os.path.join(BASE_PATH, "data", "processed"),
        os.path.join(BASE_PATH, "models"),
    ]
    # Carpetas específicas por dataset
    for _, config in DATASETS.items():
        folders.extend([config["raw_path"], config["processed_path"]])

    # Creación física
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            console.print(f" [green]+[/green] Carpeta creada: [path]{folder}[/path]")
        else:
            console.print(f" [blue]>[/blue] Carpeta existente: [path]{folder}[/path]")
            
    console.print("\n[bold success]Estructura de directorios lista.[/bold success]\n")

# =============================================================================
#  EJECUCIÓN (MAIN)
# =============================================================================

if __name__ == "__main__":
    # 1. Asegurar que el entorno (.env) sea correcto
    manage_env_variables(BASE_PATH)
    
    # 2. Construir las carpetas del proyecto
    init_project_structure()