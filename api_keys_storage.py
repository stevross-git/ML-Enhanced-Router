import os
from typing import Dict
from persistent_storage import load_json, save_json

FILE_PATH = "instance/api_keys.json"

def load_api_keys() -> Dict[str, str]:
    """Load API keys from disk and populate environment variables."""
    data = load_json(FILE_PATH)
    for env_key, value in data.items():
        if value:
            os.environ[env_key] = value
    return data


def save_api_keys(keys: Dict[str, str]) -> None:
    """Persist API keys to disk."""
    existing = load_json(FILE_PATH)
    existing.update(keys)
    save_json(FILE_PATH, existing)
