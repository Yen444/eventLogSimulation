from typing import Optional, List, Dict, Any
import os
import glob
import json
from .utils_string import ensure_string


class PromptManager:
    """Minimal PromptManager focused on discovery and cached loading of fewshots and prompts.

    - Scans `fewshots/` and `prompts/` under `base_dir` by default.
    - Provides: `list_fewshots()`, `list_prompts()`, `get_fewshow(name_or_path) -> List`, `get_prompt(name_or_path) -> str`, `refresh()`.
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
        """Discover JSON files and index them by filename (no extension)."""
        self._fewshot_index.clear()
        self._prompt_index.clear()

        if os.path.isdir(self.fewshot_dir):
            for path in glob.glob(os.path.join(self.fewshot_dir, '*.json')):
                name = os.path.splitext(os.path.basename(path))[0]
                self._fewshot_index[name] = os.path.abspath(path)

        if os.path.isdir(self.prompt_dir):
            for path in glob.glob(os.path.join(self.prompt_dir, '*.json')):
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

    def get_fewshots(self, name_or_path: str) -> List[Dict[str, Any]]:
        """Return few-shot examples as a list. If not found or invalid, return an empty list.

        Accepts either a registered name (filename without extension) or an explicit path.
        """
        path = name_or_path if os.path.exists(name_or_path) else self._fewshot_index.get(name_or_path)
        data = self._load_json_cached(path) if path else None
        if isinstance(data, list):
            return [{k: ensure_string(v) for k, v in example.items()} for example in data]
        return []

    def get_prompt(self, name_or_path: str) -> str:
        """Return prompt text or empty string if missing.

        For JSON prompts that are dicts, try keys: 'template','text','prompt','template_str'.
        """
        path = name_or_path if os.path.exists(name_or_path) else self._prompt_index.get(name_or_path)
        data = self._load_json_cached(path) if path else None
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for k in ('template', 'text', 'prompt', 'template_str'):
                if k in data and isinstance(data[k], str):
                    return data[k]
            if 'parts' in data and isinstance(data['parts'], list):
                return '\n'.join([p for p in data['parts'] if isinstance(p, str)])
            try:
                return json.dumps(data)
            except Exception:
                return ""
        return ""

    def refresh(self) -> None:
        """Clear cache and re-discover files."""
        self._cache.clear()
        self._discover()


# helper
def create_prompt_manager(base_dir: Optional[str] = None) -> PromptManager:
    return PromptManager(base_dir=base_dir)
