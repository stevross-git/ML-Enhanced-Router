from typing import Dict, Any
from persistent_storage import load_json, save_json

FILE_PATH = "instance/user_settings.json"

def load_settings() -> Dict[str, Any]:
    return load_json(FILE_PATH)


def save_settings(settings: Dict[str, Any]) -> None:
    save_json(FILE_PATH, settings)
