from typing import Optional, List, Dict, Any
import os
import glob
import json
import yaml
from .utils_string import ensure_string


class PromptManager:
    """Minimal PromptManager focused on discovery and cached loading of fewshots and prompts.

    - Scans `fewshots/` (JSON files) and `prompts/` (YAML files) under `base_dir` by default.
    - Prompts are stored in YAML format with structure: {prompt_name: {template, description, input_variables, tools}}
    - Fewshots remain in JSON format as lists of examples
    - Provides: `list_fewshots()`, `list_prompts()`, `get_fewshots(name_or_path) -> List`, 
               `get_prompt(file_name, prompt_name, data='template') -> Any`, `refresh()`.
    - get_prompt() supports multi-prompt YAML files and returns specific fields (template, tools, etc.)
    - get_fewshots() returns list with error dict for missing/wrong names.
    """

    def __init__(self, base_dir: str = './prompts', fewshot_dir: str = 'fewshots', prompt_dir: str = 'prompts'):
        self.base_dir = os.path.abspath(base_dir or os.getcwd())
        self.fewshot_dir = os.path.join(self.base_dir, fewshot_dir)
        self.prompt_dir = os.path.join(self.base_dir, prompt_dir)

        # indexes
        self._fewshot_index: Dict[str, str] = {}
        self._prompt_index: Dict[str, str] = {}

        # cache
        self._cache: Dict[str, Any] = {}

        self._discover()

    def _discover(self) -> None:
        """Discover JSON files for fewshots and YAML files for prompts, index them by filename (no extension)."""
        self._fewshot_index.clear()
        self._prompt_index.clear()

        # Discover JSON files for fewshots
        if os.path.isdir(self.fewshot_dir):
            for path in glob.glob(os.path.join(self.fewshot_dir, '*.json')):
                name = os.path.splitext(os.path.basename(path))[0]
                self._fewshot_index[name] = os.path.abspath(path)

        # Discover YAML files for prompts
        if os.path.isdir(self.prompt_dir):
            for path in glob.glob(os.path.join(self.prompt_dir, '*.yaml')):
                name = os.path.splitext(os.path.basename(path))[0]
                self._prompt_index[name] = os.path.abspath(path)
            # Also support .yml extension
            for path in glob.glob(os.path.join(self.prompt_dir, '*.yml')):
                name = os.path.splitext(os.path.basename(path))[0]
                self._prompt_index[name] = os.path.abspath(path)

    def list_fewshots(self) -> List[str]:
        return sorted(self._fewshot_index.keys())

    def list_prompts(self) -> List[str]:
        return sorted(self._prompt_index.keys())

    def _load_json_cached(self, path: str) -> Optional[Any]:
        if not path:
            return None
        abs_path = os.path.abspath(path)
        if abs_path in self._cache:
            return self._cache[abs_path]
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return None
        self._cache[abs_path] = data
        return data

    def _load_yaml_cached(self, path: str) -> Optional[Any]:
        if not path:
            return None
        abs_path = os.path.abspath(path)
        if abs_path in self._cache:
            return self._cache[abs_path]
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception:
            return None
        self._cache[abs_path] = data
        return data

    def get_fewshots(self, name_or_path: str = None) -> List[Dict[str, Any]]:
        """Return few-shot examples as a list or error message.

        Args:
            name_or_path: The fewshot name or file path (required)

        Returns:
            - If name_or_path is missing/wrong: Returns list with single error dict
            - If successful: Returns list of few-shot examples
            - If invalid format: Returns empty list

        Accepts either a registered name (filename without extension) or an explicit path.
        """
        if not name_or_path:
            return [{"error": "Fewshot name or path is required"}]
        
        path = name_or_path if os.path.exists(name_or_path) else self._fewshot_index.get(name_or_path)
        if not path:
            return [{"error": f"Fewshot '{name_or_path}' not found"}]
        
        data = self._load_json_cached(path)
        if data is None:
            return [{"error": f"Failed to load fewshot '{name_or_path}'"}]
        
        if isinstance(data, list):
            return [{k: ensure_string(v) for k, v in example.items()} for example in data]
        
        return []

    def get_prompt(self, file_name: str, prompt_name: str, data: str = 'template') -> Any:
        """Return specific field from a named prompt within a YAML file containing multiple prompts.

        Args:
            file_name: The YAML file name (without extension, e.g., 'agent_action')
            prompt_name: The specific prompt name within the file (e.g., 'agent_action_first')
            data: The field to extract ('template', 'description', 'input_variables', 'tools', etc.)

        Returns:
            - If file/prompt/field not found: "Error: ..." message (str)
            - Otherwise: the requested field value as-is (str, list, dict, etc.)

        Example usage:
            get_prompt('agent_action', 'agent_action_first', 'template')
            get_prompt('agent_action', 'agent_action_modify', 'description')
            get_prompt('agent_action', 'agent_action_first', 'tools')
        """
        if not file_name:
            return "Error: File name is required"
        
        if not prompt_name:
            return "Error: Prompt name is required"
        
        # Get the file path
        path = self._prompt_index.get(file_name)
        if not path:
            return f"Error: Prompt file '{file_name}' not found"
        
        # Load the YAML file
        file_data = self._load_yaml_cached(path)
        if file_data is None:
            return f"Error: Failed to load prompt file '{file_name}'"
        
        # Check if the file contains the requested prompt
        if not isinstance(file_data, dict):
            return f"Error: Invalid YAML structure in file '{file_name}'"
        
        if prompt_name not in file_data:
            available_prompts = list(file_data.keys())
            return f"Error: Prompt '{prompt_name}' not found in file '{file_name}'. Available prompts: {available_prompts}"
        
        prompt_data = file_data[prompt_name]
        if not isinstance(prompt_data, dict):
            return f"Error: Invalid prompt structure for '{prompt_name}' in file '{file_name}'"
        
        # Extract the requested field
        if data not in prompt_data:
            available_fields = list(prompt_data.keys())
            return f"Error: Field '{data}' not found in prompt '{prompt_name}'. Available fields: {available_fields}"
        
        field_value = prompt_data[data]
        if isinstance(field_value, str):
            return field_value
        elif isinstance(field_value, list):
            # For lists like input_variables, return as list of strings
            return [str(item) for item in field_value]
        else:
            # For other types, return as-is
            return field_value

    def refresh(self) -> None:
        """Clear cache and re-discover files."""
        self._cache.clear()
        self._discover()


# helper
def create_prompt_manager(base_dir: Optional[str] = None) -> PromptManager:
    return PromptManager(base_dir=base_dir)


