from pathlib import Path
import os
from datetime import datetime

def get_project_root() -> Path:
    """Return absolute path to project root."""
    return Path(__file__).resolve().parents[3]  # adjust for depth relative to this file

def get_data_dir() -> Path:
    data_dir = get_project_root() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

# -------------------------------------------------------------------------
# Helper: create timestamped run folder under model_runs/base/
# -------------------------------------------------------------------------
def create_run_folder(base_dir="model_runs/base"):
    """
    Creates a unique folder with timestamp inside base_dir.

    Example:
        model_runs/base/2025-12-14_18-04-11/

    Returns:
        run_dir (str): The created directory path.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(base_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir