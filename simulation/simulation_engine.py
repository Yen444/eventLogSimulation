from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from simulation.bpmn_extraction import BPMNExtractor
from simulation.config_loader import ConfigLoader
from simulation.agent import Agent

import os
import json


class SimulationEngine:
    def __init__(self, bpmn_path, config_path):
        self.bpmn_path = bpmn_path
        self.config_path = config_path
        self.agents: List[Agent] = []
    
    def extract_process_data(self):
        """Extract process data from BPMN file."""
        bpmn_extractor = BPMNExtractor(self.bpmn_path, self.config_path)
        all_process_data_dict = bpmn_extractor.extract_all_process_data()
        return all_process_data_dict
    
    def update_configs(self):
        """Update the configs with the process data"""
        process_data = self.extract_process_data()
        # Load existing data if the file exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                simulation_configs = json.load(f)
        else:
            simulation_configs = {}

        # Merge the new process data with the existing data
        if isinstance(simulation_configs, dict) and isinstance(process_data, dict):
            simulation_configs.update(process_data)
        else:
            raise ValueError("Merging non-dict JSON data is not supported in this function.")

        # Write back to file
        with open(self.config_path, 'w', encoding='utf-8') as f:
            # Ensure the file is written in a readable format
            json.dump(simulation_configs, f, ensure_ascii=False, indent=4)
    
    def load_configs(self):
        """Load the configs from the config file"""
        config_loader = ConfigLoader(self.config_path)
        simulation_configs = config_loader.load_config()
        return simulation_configs
    