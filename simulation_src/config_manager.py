import os
import json
from .bpmn_extraction import BPMNExtractor

class ConfigManager:
    def __init__(self,  bpmn_path, extracted_config_path, user_config_path, simulation_config_path):
        self.bpmn_path = bpmn_path
        self.extracted_config_path = extracted_config_path
        self.user_config_path = user_config_path
        self.simulation_config_path = simulation_config_path
        self.configs = None

    def _load_config(self, path):
        """Load a JSON config file if it exists, otherwise return an empty dict."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_config(self, path, configs):
        """Save a config dictionary to a JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, ensure_ascii=False, indent=4)

    def _extract_process_data(self):
        """Extract process data from the BPMN file and save to extracted_config."""
        bpmn_extractor = BPMNExtractor(self.bpmn_path, self.extracted_config_path)
        extracted_data = bpmn_extractor.extract_all_process_data()
        self._save_config(self.extracted_config_path, extracted_data)
        return extracted_data

    def update_and_load_configs(self):
        """Load user and extracted configs, extract if needed, merge and save simulation config."""
        # If simulation config exists and is valid, load and return it
        if self.simulation_config_path:
            configs = self._load_config(self.simulation_config_path)
            if configs:
                self.configs = configs
                return configs

        # Load user config (required)
        user_config = self._load_config(self.user_config_path)
        if not user_config:
            raise ValueError("User config is required.")

        # Load or extract extracted config
        extracted_config = self._load_config(self.extracted_config_path)
        if not extracted_config:
            extracted_config = self._extract_process_data()
            self._save_config(self.extracted_config_path, extracted_config)

        # Validate configs
        if not isinstance(user_config, dict) or not isinstance(extracted_config, dict):
            raise ValueError("Both user and extracted configs must be dictionaries.")

        # Merge and save
        merged_config = {**user_config, **extracted_config}
        self._save_config(self.simulation_config_path, merged_config)
        self.configs = merged_config
        return merged_config