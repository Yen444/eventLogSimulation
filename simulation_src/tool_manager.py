from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from .prompt_manager import PromptManager
from .tinydb_store import TinyDBStore
from langchain_core.tools import tool

class ToolManager(BaseModel):
    llm: Any = Field(..., description="The language model instance to be used.")
    db_store: TinyDBStore = Field(..., description="The tiny database instance for managing tools.")
    prompt_manager: PromptManager = Field(..., description="The prompt manager instance for managing prompts.")
    use_tools: bool = Field(True, description="Flag to indicate whether to use tools or not.")

    def is_tool_enabled(self) -> bool:
        """Check if tool usage is enabled."""
        return self.use_tools
    
    def get_stats_tool(self, name) -> Dict[str, Any]:
        """Given a tool name, returns the corresponding callable that produces database statistics."""
        @tool
        def get_event_data_stats(action: Optional[str]=None, event_attributes: Optional[List[str]]=None) -> Dict[str, Any]:
            """Retrieve statistics about event data per one specific action or per all actions"""
            if self.db_store:
                return self.db_store.get_event_data_stats(action=action, event_attributes=event_attributes)
            else:
                return {}
        @tool
        def get_case_data_stats(case_attributes: Optional[List[str]]=None) -> Dict[str, Any]:
            """Retrieve statistics about case data of all processes"""
            if self.db_store:
                return self.db_store.get_case_data_stats(case_attributes=case_attributes)
            else:
                return {}
        @tool
        def get_event_data_stats_by_resource(event_attribute: Optional[str]=None, resource_attribute: Optional[str]=None, action: Optional[str]=None) -> Dict[str, Any]:
            """Retrieve statistics about an event data grouped by a resource attribute per given action or for all given actions
            """
            if self.db_store:
                return self.db_store.get_event_data_stats_by_resource(event_attribute=event_attribute, resource_attribute=resource_attribute, action=action)
            else:
                return {}
        @tool
        def get_resource_data_stats(resource_attributes: Optional[List[str]]=None) -> Dict[str, Any]:
            """Retrieve statistics about resource data of all resources"""
            if self.db_store:
                return self.db_store.get_resource_data_stats(resource_attributes=resource_attributes)
            else:
                return {}
        tool_mapping = {
            "get_event_data_stats": get_event_data_stats,
            "get_case_data_stats": get_case_data_stats,
            "get_event_data_stats_by_resource": get_event_data_stats_by_resource,
            "get_resource_data_stats": get_resource_data_stats
        }
        if not name:
            return list(tool_mapping.values())
        if isinstance(name, list):
            return [tool_mapping.get(n, None) for n in name]
        if isinstance(name, str):
            return tool_mapping.get(name, None)
        return None

    