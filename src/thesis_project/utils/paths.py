from pathlib import Path

def get_project_root() -> Path:
    """Return absolute path to project root."""
    return Path(__file__).resolve().parents[3]  # adjust for depth relative to this file

def get_data_dir() -> Path:
    return get_project_root() / "data"