from pathlib import Path
from dotenv import load_dotenv, set_key

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

SEED = 42

# =============================================================================
#  FUNCTIONS
# =============================================================================

def setup_environment():
    """Makes sure the .env file exists and updates the PROJECT_BASE_PATH variable."""
    if not ENV_PATH.exists():
        ENV_PATH.touch()
        print(".env file created")

    set_key(str(ENV_PATH), "PROJECT_BASE_PATH", str(BASE_PATH))
    print(f"Variable PROJECT_BASE_PATH synchronized in: {ENV_PATH}")

def init_project_structure():
    """Creates the necessary folder structure for the project."""
    print(f"Init structure \nPath: {BASE_PATH}")
    
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
            print(f" + Creada: {folder.relative_to(BASE_PATH)}")
        else:
            print(f" > Existe: {folder.relative_to(BASE_PATH)}")

if __name__ == "__main__":
    setup_environment()
    init_project_structure()
    print("\nSuccess\n")