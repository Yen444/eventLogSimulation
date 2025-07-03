import os
import json
from typing import Dict, Literal

class ConfigManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, Dict[str, str]]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as file:
            return json.load(file)
