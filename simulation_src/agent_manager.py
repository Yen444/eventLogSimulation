from pydantic import BaseModel, Field, create_model
from typing import List, Dict, Optional, Any
from langchain_core.prompts import PromptTemplate


from .llm_utils import generate_random_text_with_mem_chatbot
from .utils import generate_random_agent_id
from .config_manager import ConfigManager
from .agent import Agent
from .coordinator_agent import Coordinator
from .prompt_manager import PromptManager, create_prompt_manager
from .tinydb_store import TinyDBStore

from dotenv import load_dotenv
import os

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

class AgentManager:
    def __init__(self, config_manager: ConfigManager, prompt_manager: PromptManager, llm, process_manager=None):
        self.config_manager = config_manager
        self.prompt_manager = prompt_manager
        self.llm = llm
        self.process_manager = process_manager
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
            __base__=BaseModel,
        )
        
        # Create the list model wrapping multiple agent outputs
        ListAgentOutput = create_model(
            'ListAgentOutput',
            agent_outputs=(List[AgentOutput], ...),
            __base__=BaseModel,
        )
        
        return AgentOutput, ListAgentOutput


    def create_org_agents(self):
        """Create agents based on the configuration."""
        simulation_configs = self.config_manager.configs
        role_actions = simulation_configs.get('org_role_action', {})
        agent_numbers = simulation_configs.get('org_role_num', {})
        organization_context = simulation_configs.get('organization_summary', '')

        org_agents = ""

        for agent_role, actions in role_actions.items():
            num_agents = agent_numbers.get(agent_role, 1)
            for _ in range(num_agents):
                used_ids = self.process_manager.store.get_all_agent_ids() if self.process_manager else []
                agent_id = generate_random_agent_id(used_ids=used_ids)
                agent = Agent(
                    id=agent_id,
                    role=agent_role,
                    actions=actions,
                    action_fewshots=self.prompt_manager.get_fewshots('agent_action')
                )
                
                agent.generate_data(data_model=self.ResourceData, llm=self.llm, organization_context=organization_context, org_agents=org_agents)

                #print(agent.data)
                # append to organization agents
                org_agents += f'{agent.data}, role: {agent_role}\n'
                self.org_agents.append(agent)
                
                # register org agent in DB if process_manager provided
                try:
                    if self.process_manager:
                        self.process_manager.store.register_org_agent(agent)
                except Exception:
                    pass

    def create_ext_agent_onfly(self, role:str):
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
            action_fewshots=self.prompt_manager.get_fewshots('agent_action')
        )
        agent.generate_data_onfly(data_model=self.ResourceData, llm=self.llm, organization_context=organization_context)
        self.ext_agents.append(agent)
        # register external agent in DB if process_manager provided
        try:
            if self.process_manager:
                self.process_manager.store.register_ext_agent(agent)
        except Exception:
            pass
        return agent
    
    def create_coordinator_agent(self):
        """Create a coordinator agent"""
        simulation_configs = self.config_manager.configs
        process_summary = simulation_configs.get('process_summary', '')
        conventions = simulation_configs.get('conventions', [])
        coordinator = Coordinator(
            id=generate_random_agent_id(used_ids=self.process_manager.store.get_all_agent_ids() if self.process_manager else []),
            process_summary=process_summary,
            llm=self.llm,
            conventions=', '.join(conventions),
            handoff_fewshots=self.prompt_manager.get_fewshots('coordinator_handoff')
        )
        self.coordinator = coordinator
        # register coordinator in DB if process_manager provided
        self.process_manager.store.register_coordinator(coordinator)
    
    
    def get_agent_by_id(self, agent_id: str):
        """Get an agent by its ID."""
        return self.process_manager.store.get_agent_by_id(agent_id) if self.process_manager else None