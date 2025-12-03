from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate


from .utils import generate_random_agent_id
from .config_manager import ConfigManager
from .agent import Agent
from .coordinator_agent import Coordinator
from .prompt_manager import PromptManager, create_prompt_manager
from .tinydb_store import TinyDBStore
from .tool_manager import ToolManager
from tinydb import where

from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
api_key_openai = os.getenv('OPENAI_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

llm_low = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0) # gemini-2.5-pro-preview-03-25
llm_mid = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.5)
llm_high = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)
llm_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    openai_api_key=api_key_openai
)

from .utils import create_dynamic_model
from langchain_core.tools import tool


class AgentManager:
    def __init__(self, config_manager: ConfigManager, prompt_manager: PromptManager, llm, store: TinyDBStore = None, tool_manager: ToolManager = None, use_tools=False, agent_data_creation_reflect=False):
        self.config_manager = config_manager
        self.prompt_manager = prompt_manager
        self.llm = llm
        self.store = store
        self.tool_manager = tool_manager
        self.agent_data_creation_reflect = agent_data_creation_reflect
        self.coordinator = None
        self.org_agents: List[Agent] = []
        self.ext_agents: List[Agent] = []
        self._create_dynamic_models()

    def _create_dynamic_models(self):
        configs = self.config_manager.configs
        self.ResourceData  = create_dynamic_model("ResourceData", configs.get('agent_data', {}))
        self.EventData = create_dynamic_model("EventData", configs.get('event_attributes', {}))
        self.CaseData = create_dynamic_model("CaseData", configs.get('case_attributes', {}))
        self.AgentOutput, self.ListAgentOutput = self._create_output_model(self.EventData, self.CaseData)

    def _create_output_model(self, event_data_model, case_data_model):
        # Create the dynamic single agent output model
        AgentOutput = create_model(
            'AgentOutput',
            process_id=(str, ...),
            action=(str, ...),
            start_timestamp=(str, ...),
            end_timestamp=(str, ...),
            message=(str, ...),
            event_data=(Optional[event_data_model], None),
            case_data=(Optional[case_data_model], None),
            end_action=(Optional[bool], None),
            reasoning=(Optional[str], None),
            __base__=BaseModel,
        )
        
        # Create the list model wrapping multiple agent outputs
        ListAgentOutput = create_model(
            'ListAgentOutput',
            agent_outputs=(List[AgentOutput], ...),
            __base__=BaseModel,
        )
        
        return AgentOutput, ListAgentOutput
    
    
    #def get_tool(self, name: str = None):
    #    @tool
    #    def get_event_data_stats(action: Optional[str]=None, event_attributes: Optional[List[str]]=None) -> Dict[str, Any]:
    #        """Retrieve statistics about event data per one specific action or per all actions"""
    #        if self.store:
    #            return self.store.get_event_data_stats(action=action, event_attributes=event_attributes)
    #        else:
    #            return {}
    #    @tool
    #    def get_case_data_stats(case_attributes: Optional[List[str]]=None) -> Dict[str, Any]:
    #        """Retrieve statistics about case data of all processes"""
    #        if self.store:
    #            return self.store.get_case_data_stats(case_attributes=case_attributes)
    #        else:
    #            return {}
    #    @tool
    #    def get_event_data_stats_by_resource(event_attribute: Optional[str]=None, resource_attribute: Optional[str]=None, action: Optional[str]=None) -> Dict[str, Any]:
    #        """Retrieve statistics about an event data grouped by a resource attribute per given action or for all given actions
    #        """
    #        if self.store:
    #            return self.store.get_event_data_stats_by_resource(event_attribute=event_attribute, resource_attribute=resource_attribute, action=action)
    #        else:
    #            return {}
    #    @tool
    #    def get_resource_data_stats(resource_attributes: Optional[List[str]]=None) -> Dict[str, Any]:
    #        """Retrieve statistics about resource data of all resources"""
    #        if self.store:
    #            return self.store.get_resource_data_stats(resource_attributes=resource_attributes)
    #        else:
    #            return {}
    #    
    #    if name == "get_event_data_stats":
    #        return get_event_data_stats
    #   elif name == "get_case_data_stats":
    #        return get_case_data_stats
    #    elif name == "get_event_data_stats_by_resource":
    #        return get_event_data_stats_by_resource
    #    elif name == "get_resource_data_stats":
    #        return get_resource_data_stats
    #    else:
    #        return None

    def create_register_org_agents(self):
        """Create agents based on the configuration."""
        simulation_configs = self.config_manager.configs
        role_actions = simulation_configs.get('org_role_action', {})
        agent_numbers = simulation_configs.get('org_role_num', {})
        organization_context = simulation_configs.get('organization_summary', '')

        org_agents = ""

        for agent_role, actions in role_actions.items():
            num_agents = agent_numbers.get(agent_role, 1)
            existing_agents = []

            # --- check existing agents in store ---
            if self.store:
                existing_agents = self.store.org_agents.search(where('role') == agent_role)
                existing_count = len(existing_agents)

                # ensure existing agents are added to in-memory list
                for existing in existing_agents:
                    if not any(a.id == existing['id'] for a in self.org_agents):
                        # reconstruct Agent object from stored data if needed
                        agent = Agent(
                            id=existing['id'],
                            role=existing['role'],
                            actions=existing.get('actions', []),
                            llm=self.llm,
                            prompt_manager=self.prompt_manager,
                            tool_manager=self.tool_manager
                        )
                        agent.data = existing.get('data', '')
                        self.org_agents.append(agent)
                        org_agents += f'{agent.data}, role: {agent_role}\n'

                # reduce number of new agents to create
                if existing_count >= num_agents:
                    continue
                num_agents -= existing_count

            # --- create missing agents ---
            for _ in range(num_agents):
                used_ids = self.store.get_all_agent_ids() if self.store else []
                agent_id = generate_random_agent_id(used_ids=used_ids)
                agent = Agent(
                    id=agent_id,
                    role=agent_role,
                    actions=actions,
                    llm=self.llm,
                    prompt_manager=self.prompt_manager,
                    tool_manager=self.tool_manager
                )
                agent.generate_data(
                    data_model=self.ResourceData,
                    llm=self.llm,
                    organization_context=organization_context,
                    org_agents=org_agents,
                    reflect=self.agent_data_creation_reflect
                )
                org_agents += f'{agent.data}, role: {agent_role}\n'
                self.org_agents.append(agent)

                try:
                    if self.store:
                        self.store.register_org_agent(agent)
                except Exception:
                    pass


    def create_register_ext_agent_onfly(self, role:str):
        """Create agents on the fly based on the configurations or handoff"""
        simulation_configs = self.config_manager.configs
        role_actions = simulation_configs.get('ext_role_actions', {})
        organization_context = simulation_configs.get('organization_summary', '')
        
        used_ids = [agent.id for agent in self.org_agents] + [agent.id for agent in self.ext_agents]
        agent_id = generate_random_agent_id(used_ids=used_ids)
        actions = role_actions.get(role, [])
        agent = Agent(
            id=agent_id,
            role=role,
            actions=actions,
            llm=self.llm,
            prompt_manager=self.prompt_manager,
            tool_manager=self.tool_manager
        )
        # TODO: retrieve existing external agents from db and add tools

        agent.generate_data_onfly(data_model=self.ResourceData, llm=self.llm, organization_context=organization_context, reflect=self.agent_data_creation_reflect)
        self.ext_agents.append(agent)
        # register external agent in DB if process_manager provided
        try:
            if self.store:
                self.store.register_ext_agent(agent)
        except Exception:
            pass
        return agent
    
    def create_register_coordinator_agent(self):
        """Create a coordinator agent"""
        # check if coordinator already exists in db, if so, load it to memory
        # TinyDB Table.count() requires a condition argument; use .all() and len() here
        if self.store and len(self.store.coordinators.all()) > 0:
            existing = self.store.coordinators.all()[0]
            self.coordinator = Coordinator(
                id=existing.get('id'),
                process_summary=existing.get('process_summary', ''),
                llm=self.llm,
                conventions=existing.get('conventions', ''),
                prompt_manager=self.prompt_manager,
                tool_manager=self.tool_manager
            )
            return
        
        simulation_configs = self.config_manager.configs
        process_summary = simulation_configs.get('process_summary', '')
        conventions = simulation_configs.get('conventions', [])
        coordinator = Coordinator(
            id=generate_random_agent_id(used_ids=self.store.get_all_agent_ids() if self.store else []),
            process_summary=process_summary,
            llm=self.llm,
            conventions=', '.join(conventions),
            prompt_manager=self.prompt_manager,
            tool_manager=self.tool_manager
        )
        self.coordinator = coordinator
        # register coordinator in DB if process_manager provided
        if self.store:
            self.store.register_coordinator(coordinator)
    
    def get_agent_by_id(self, agent_id: str):
        """Get an agent by its ID."""
        # Prefer the persistent store if available
        try:
            if self.store:
                agent = self.store.get_agent_by_id(agent_id)
                if agent:
                    # Ensure the returned Agent has the manager's llm and tools wired
                    try:
                        agent.llm = self.llm
                        agent.tool_manager = self.tool_manager
                        agent.prompt_manager = self.prompt_manager
                    except Exception:
                        pass
                    return agent
        except Exception:
            pass

        # Fallback: search in-memory org and external agents
        for a in self.org_agents + self.ext_agents:
            if getattr(a, 'id', None) == agent_id:
                # also ensure llm/tools are present
                try:
                    a.llm = a.llm or self.llm
                    a.tool_manager = self.tool_manager
                    a.prompt_manager = self.prompt_manager
                except Exception:
                    pass
                return a
        return None
    
    def get_available_org_agents(self, current_time: str):
        """Get all available agents based on current time."""
        available_org_agents = []
        for agent in self.org_agents:
            if not agent.available_from or datetime.fromisoformat(current_time) >= datetime.fromisoformat(agent.available_from):
                available_org_agents.append(agent)

        return available_org_agents