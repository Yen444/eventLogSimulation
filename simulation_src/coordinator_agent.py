from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.messages import HumanMessage

from .prompt_manager import PromptManager
from .tool_manager import ToolManager

import json


class CoordinatorOutput(BaseModel):
    """Structured Coordinator Output"""
    process_id: Optional[str] = Field(default=None, description="Unique identifier for the process")
    agent_id: Optional[str] = Field(default=None, description="Unique identifier for the agent")
    agent_role: str = Field(description="Role of the agent")
    message: str = Field(description="Message for the agent")

class ListCoordinatorOutput(BaseModel):
    """Structured Coordinator Output List"""
    coordinator_outputs: List[CoordinatorOutput] = Field(description="List of coordinator outputs")

class Coordinator(BaseModel):
    """Coordinator Agent for managing processes and agents"""
    id: str = Field(description="Unique identifier for the coordinator")
    role: str = "coordinator"
    process_summary: str = Field(description="Summary of the process")
    conventions: str = Field(default="", description="Conventions to follow when assigning agents to tasks")
    llm: Any = Field(default=None, description="LLM for generating responses")
    prompt_manager: Optional[PromptManager] = Field(default=None, description="PromptManager instance for accessing prompts and fewshots")
    tool_manager: Optional[ToolManager] = Field(default=None, description="ToolManager instance for managing tools")

    model_config = {
        "arbitrary_types_allowed": True,
    }

    def hand_off(self, process_states, org_agents, external_agents, current_timestamp):
        org_agents_str = '\n'.join([f"ID: {agent.id}; (role: {agent.role}; actions: {agent.actions}; data: {agent.data})" for agent in org_agents])
        external_agents_str = '\n'.join([f"ID: {agent.id}; (role: {agent.role}; actions: {agent.actions}; data: {agent.data})" for agent in external_agents])
        
        # Get prompt from PromptManager
        if not self.prompt_manager:
            raise ValueError("PromptManager is required for coordinator operation")
        
        prompt = self.prompt_manager.get_prompt('coordinator_handoff', 'handoff', 'template')
        
        tool_prompt = self.prompt_manager.get_prompt('coordinator_handoff', 'handoff_tool', 'template')
        
        # Load fewshots on-demand from PromptManager
        fewshot_examples = []
        if self.prompt_manager.use_fewshots:
            fewshots = self.prompt_manager.get_fewshots('coordinator_handoff')
            # Check if fewshots were loaded successfully (not an error list)
            if fewshots and not (len(fewshots) == 1 and 'error' in fewshots[0]):
                fewshot_examples = fewshots
        
        if fewshot_examples:
            # example prompt template
            example_prompt = PromptTemplate(
                input_variables=["process_summary", "process_states", "organization_agents", "external_agents", "current_timestamp", "conventions", "output"],
                template="Process_summary: {process_summary}\nProcess_states: {process_states}\nOrganization_agents: {organization_agents}\nExternal_agents: {external_agents}\nCurrent_timestamp: {current_timestamp}\nConventions: {conventions}\nOutput: {output}"
            )
            final_prompt = FewShotPromptTemplate(
                examples=fewshot_examples,
                example_prompt=example_prompt,
                input_variables=["process_summary", "process_states", "org_agents", "external_agents", "current_timestamp", "conventions"],
                prefix=prompt
            )

        else:
            final_prompt = PromptTemplate.from_template(prompt)

        tool_messages = []
        
        # Add tools if available
        if self.tool_manager and self.tool_manager.use_tools:
            tools = None
            if tools:
                llm_with_tools = self.llm.bind_tools(tools)
                tool_prompt = PromptTemplate.from_template(tool_prompt)
                tool_chain = tool_prompt | llm_with_tools
                tool_output = tool_chain.invoke({
                    "process_summary": self.process_summary,
                    "process_states": process_states,
                    "org_agents": org_agents_str,   
                    "external_agents": external_agents_str,
                    "current_timestamp": current_timestamp,
                    'conventions':  self.conventions
                })
                # get tool calls output messages
                tool_calls = getattr(tool_output, 'tool_calls', None)
                if tool_calls:
                    for tool_call in tool_calls:
                        selected_tool = next((tool for tool in tools if tool.name.lower() == tool_call["name"].lower()), None)
                        if selected_tool is not None:
                            print(f'Tool: {tool_call["name"]} is being called by llm with arguments {tool_call.get("args", tool_call.get("arguments", {}))}')
                            try:
                                tool_msg = selected_tool.invoke(tool_call)
                                tool_messages.append(tool_msg)
                            except Exception as e:
                                print(f'Error occurred while invoking tool {tool_call["name"]}: {e}')
                        else:
                            print(f'Tool: {tool_call["name"]} not found in available tools.')
            else:
                llm_with_tools = self.llm
                tool_output = None
        else:
            llm_with_tools = self.llm
            tool_output = None
        
        base_message = final_prompt.invoke({
            "process_summary": self.process_summary,
            "process_states": process_states,
            "org_agents": org_agents_str,   
            "external_agents": external_agents_str,
            "current_timestamp": current_timestamp,
            'conventions':  self.conventions
        })
        if tool_messages:
            full_messages = [HumanMessage(content=base_message.text), tool_output] + tool_messages
        else:
            full_messages = [HumanMessage(content=base_message.text)]

        structured_llm = llm_with_tools.with_structured_output(ListCoordinatorOutput) 
        coordinator_output = structured_llm.invoke(full_messages)

        return coordinator_output