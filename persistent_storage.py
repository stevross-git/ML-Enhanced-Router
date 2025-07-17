import json
import os
from typing import Any, Dict

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from a file."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_json(file_path: str, data: Dict[str, Any]) -> None:
    """Save JSON data to a file, creating directories if needed."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
