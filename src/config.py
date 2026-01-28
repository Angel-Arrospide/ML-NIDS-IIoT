import os
from pathlib import Path
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

# =============================================================================
#  CONFIGURACIONES VISUALES
# =============================================================================
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "path": "underline blue"
})
console = Console(theme=custom_theme)

# =============================================================================
#  GESTIÓN DE RUTAS (PATHLIB)
# =============================================================================
# Definimos la raíz como el padre de la carpeta 'src'
BASE_PATH = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_PATH / ".env"

# Cargar variables de entorno
load_dotenv(ENV_PATH)

# Configuración de Datasets 
DATASETS = {
    "edge_iiot": {
        "url": "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
        "raw_path": BASE_PATH / "data" / "raw" / "edge_iiot",
        "processed_path": BASE_PATH / "data" / "processed" / "edge_iiot",
    },
}

# =============================================================================
#  FUNCIONES
# =============================================================================

def setup_environment():
    """Asegura que el archivo .env tenga la ruta base correcta."""
    if not ENV_PATH.exists():
        ENV_PATH.touch()
        console.print("[warning].env no encontrado. Creando uno nuevo...[/warning]")

    # Actualizar la ruta base en el .env de forma automática
    set_key(str(ENV_PATH), "PROJECT_BASE_PATH", str(BASE_PATH))
    console.print(f"[info]Variable PROJECT_BASE_PATH sincronizada en:[/info] [path]{ENV_PATH}[/path]")

def init_project_structure():
    """Crea la jerarquía de carpetas necesaria."""
    console.print(Panel(f"[bold info]Inicializando Estructura[/bold info]\nRaíz: [path]{BASE_PATH}[/path]", expand=False))
    
    # Carpetas generales y específicas
    folders = [
        BASE_PATH / "data" / "raw",
        BASE_PATH / "data" / "processed",
        BASE_PATH / "models",
    ]
    for ds in DATASETS.values():
        folders.extend([ds["raw_path"], ds["processed_path"]])

    for folder in folders:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            console.print(f" [green]+[/green] Creada: [path]{folder.relative_to(BASE_PATH)}[/path]")
        else:
            console.print(f" [blue]>[/blue] Existe: [path]{folder.relative_to(BASE_PATH)}[/path]")

if __name__ == "__main__":
    setup_environment()
    init_project_structure()
    console.print("\n[bold success]¡Configuración completada con éxito![/bold success]\n")