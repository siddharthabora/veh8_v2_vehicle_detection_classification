import yaml
from pathlib import Path


def load_yaml(path):
    """
    Load a YAML config file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)
