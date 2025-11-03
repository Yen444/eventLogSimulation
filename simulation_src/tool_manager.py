from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from .prompt_manager import PromptManager
from .tinydb_store import TinyDBStore
from langchain_core.tools import tool
import numpy as np

class ToolManager(BaseModel):
    llm: Any = Field(..., description="The language model instance to be used.")
    db_store: TinyDBStore = Field(..., description="The tiny database instance for managing tools.")
    prompt_manager: PromptManager = Field(..., description="The prompt manager instance for managing prompts.")
    use_tools: bool = Field(True, description="Flag to indicate whether to use tools or not.")
    
    model_config = {
        "arbitrary_types_allowed": True,
    }

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
        def get_event_data_stats_by_resource_group(event_attribute: Optional[str]=None, resource_attribute: Optional[str]=None, action: Optional[str]=None) -> Dict[str, Any]:
            """Retrieve statistics about event data grouped by a resource group.
            The statistics are aggregated by the specified resource_attribute (e.g., gender: male vs. female) for a given `action`, or for all actions if no specific one is provided.
            """
            if self.db_store:
                return self.db_store.get_event_data_stats_by_resource_group(event_attribute=event_attribute, resource_attribute=resource_attribute, action=action)
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
            "get_event_data_stats_by_resource_group": get_event_data_stats_by_resource_group,
            "get_resource_data_stats": get_resource_data_stats
        }
        if not name:
            return list(tool_mapping.values())
        if isinstance(name, list):
            return [tool_mapping.get(n, None) for n in name]
        if isinstance(name, str):
            return [tool_mapping.get(name, None)]
        return None

    def get_generate_tool(self):
        @tool
        def generate_numerical_data_uniform(attribute,lower_bound, upper_bound) -> float:
            """
            Generate a  numerical value with a uniform distribution.

            Args:
                attribute (str): The name of the attribute for which to generate data.
                lower_bound (float): Lower bound of the distribution.
                upper_bound (float): Upper bound of the distribution.

            Returns:
                float: A single uniformly distributed random value.
            """
            try:
                value = float(np.random.uniform(lower_bound, upper_bound))
                return f'One possible value for {attribute}: {value}'
            except Exception as e:
                return ''

        @tool
        def generate_numerical_data_normal(attribute, mean, std) -> float:
            """
            Generate a single numerical value with a normal (Gaussian) distribution.

            Args:
                attribute (str): The name of the attribute for which to generate data.
                mean (float): Mean of the distribution.
                std (float): Standard deviation (must be > 0).

            Returns:
                float: A single normally distributed random value.
            """
            try:
                value = float(np.random.normal(mean, std))
                return f'One possible value for {attribute}: {value}'
            except Exception as e:
                return ''
        return [generate_numerical_data_uniform, generate_numerical_data_normal]