import os
from pathlib import Path
import sys
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.theme import Theme

# =============================================================================
#  VISUALS
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
#  PATHS Y CONFIG
# =============================================================================
BASE_PATH = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_PATH / ".env"
load_dotenv(ENV_PATH)
DATASETS = {
    "edge_iiot": {
        "url": "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
        "raw_path": BASE_PATH / "data" / "raw" / "edge_iiot",
        "processed_path": BASE_PATH / "data" / "processed" / "edge_iiot",
        "binary_model_path": BASE_PATH / "models" / "binary" / "edge_iiot",
    },
}

# =============================================================================
#  FUNCTIONS
# =============================================================================

def setup_environment():
    """Makes sure the .env file exists and updates the PROJECT_BASE_PATH variable."""
    if not ENV_PATH.exists():
        ENV_PATH.touch()
        console.print("[warning].env file created[/warning]")

    set_key(str(ENV_PATH), "PROJECT_BASE_PATH", str(BASE_PATH))
    console.print(f"[info]Variable PROJECT_BASE_PATH synchronized in:[/info] [path]{ENV_PATH}[/path]")

def init_project_structure():
    """Creates the necessary folder structure for the project."""
    console.print(Panel(f"[bold info]Init structure[/bold info]\Path: [path]{BASE_PATH}[/path]", expand=False))
    
    # Carpetas generales y específicas
    folders = [
        BASE_PATH / "data" / "raw",
        BASE_PATH / "data" / "processed",
        BASE_PATH / "models",
        BASE_PATH / "models" / "binary",
    ]
    for ds in DATASETS.values():
        folders.extend([ds["raw_path"], ds["processed_path"], ds["binary_model_path"]])

    for folder in folders:
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
            console.print(f" [green]+[/green] Creada: [path]{folder.relative_to(BASE_PATH)}[/path]")
        else:
            console.print(f" [blue]>[/blue] Existe: [path]{folder.relative_to(BASE_PATH)}[/path]")

if __name__ == "__main__":
    setup_environment()
    init_project_structure()
    console.print("\n[bold success]Success[/bold success]\n")