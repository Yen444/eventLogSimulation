from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

from .bpmn_extraction import BPMNExtractor
from .agent import Agent
from .utils import generate_process_id, generate_agent_id
from .cordinator import Cordinator

from collections import OrderedDict
import datetime
import csv

import os
import json

from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0)

class logEvent(BaseModel):
    """Log event for the simulation"""
    process_id: str = Field(description="Unique identifier for the process instance")
    action: str = Field(description="Type of the event")
    timestamp: str = Field(description="Timestamp of the event")
    agent: Agent = Field(description="Agent who performed the action")
    # external factor / action related attributes
    external_values: Optional[str] = Field(default=None, description="External factors which can affect the agent's action and the whole process") 
    # internal factor/ trace related attributes
    internal_values: Optional[str] = Field(default=None, description="Internal factors which can affect the agent's action and the whole process")
    # experience-related data / action related attributes
    feedback: Optional[str] = Field(default=None, description="Feedback from the agent about the action they took (easy, hard, sastisfied, unsatisfied, etc.)")
    
class ProcessInstance(BaseModel):
    """Process instance for the simulation"""
    process_id: str = Field(description="Unique identifier for the process instance")
    events: List[logEvent] = Field(default_factory=list, description="List of events in the process instance")


import os
import json

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

    
class AgentManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.org_agents: List[Agent] = []
        self.ext_agents: List[Agent] = []
        self.cordinator: Cordinator = None
    
    
    def create_trigger_agent_onfly(self):
        """Create a trigger agent based on the configuration."""
        simulation_configs = self.config_manager.update_and_load_configs()
        role_actions = simulation_configs.get('role_actions', {})
        demographics = simulation_configs.get('demographics', [])
        trigger_agent = simulation_configs.get('trigger_agent', None)

        if trigger_agent:
            agent = Agent(
                role=trigger_agent,
                demographics_attributes=demographics,
                actions=role_actions.get(trigger_agent, [])
            )
            agent.create_trigger_prompt()
            used_ids = [agent.id for agent in self.org_agents] + [agent.id for agent in self.ext_agents]
            agent_id = generate_agent_id(used_ids=used_ids)
            agent.id = agent_id
            self.ext_agents.append(agent)
            return agent
        else:
            raise ValueError("Trigger agent role is not defined in the configuration.")


    def create_org_agents(self):
        """Create agents based on the configuration."""
        simulation_configs = self.config_manager.update_and_load_configs()
        role_actions = simulation_configs.get('role_actions', {})
        demographics = simulation_configs.get('demographics', [])
        agent_numbers = simulation_configs.get('num_org_agents', {})

        for agent_role, actions in role_actions.items():
            num_agents = agent_numbers.get(agent_role, 2)
            for _ in range(num_agents):
                agent = Agent(
                    role=agent_role,
                    demographics_attributes=demographics,
                    actions=actions,
                )
                agent.create_prompt()
                used_ids = [agent.id for agent in self.org_agents] + [agent.id for agent in self.ext_agents]
                agent_id = generate_agent_id(used_ids=used_ids)
                agent.id = agent_id
                self.org_agents.append(agent)

    def create_ext_agent_onfly(self, role:str):
        """Create agents on the fly based on the cofigurations or handoff"""
        simulation_configs = self.config_manager.update_and_load_configs()
        role_actions = simulation_configs.get('role_actions', {})
        demographics = simulation_configs.get('demographics', [])
        agent = Agent(
            role=role,
            demographics_attributes=demographics,
            actions=role_actions.get(role, []),
        )
        agent.create_prompt()
        used_ids = [agent.id for agent in self.org_agents] + [agent.id for agent in self.ext_agents]
        agent_id = generate_agent_id(used_ids=used_ids)
        agent.id = agent_id
        self.ext_agents.append(agent)
        return agent
    
    def create_cordinator(self):
        """Create a cordinator agent"""
        simulation_configs = self.config_manager.update_and_load_configs()
        process_summary = simulation_configs.get('process_summary', '')
        return Cordinator(
            role='cordinator',
            process_summary=process_summary
        ) 
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by its ID."""
        for agent in self.org_agents + self.ext_agents:
            if agent.id == agent_id:
                return agent

class ProcessManager:
    def __init__(self, log_path):
        self.incomplete_processes: List[ProcessInstance] = []
        self.log_events: List[logEvent] = []
        self.log_path = log_path 

        # open for writing and create the log file if it does not exist

        if not os.path.exists(log_path):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("")

    def _write_to_log_file(self, log_event: logEvent):
        """Write log event to the log file."""
        # open the log file in append mode
        # and write the log event to it
        with open(self.log_path, 'a', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["process_id", "action", "timestamp", 'role', "agent", "external_values"])
            #writer.writeheader()
            writer.writerow({
                "process_id": log_event.process_id,
                "action": log_event.action,
                "timestamp": log_event.timestamp,
                'role': log_event.agent.role,
                "external_values": json.dumps(log_event.external_values),
                "agent": json.dumps({attr: value for attr, value in zip(log_event.agent.demographics_attributes, log_event.agent.demographics_values)})
            })
    
    # generating process id if trigger event happens

    def generate_process_id(self):
        """Generate a unique process ID."""
        used_process_ids = list({log.process_id for log in self.log_events})
        process_id = generate_process_id(used_ids=used_process_ids)
        return process_id
    
    def get_related_process(self, process_id: str) -> str:
        """Get the related process instance."""
        for process_instance in self.incomplete_processes:
            if process_instance.process_id == process_id:
                process_state = f'Process ID: {process_instance.process_id}, Last Event: {process_instance.events[-1].action if process_instance.events else 'No action is taken so far'}, Last Timestamp: {process_instance.events[-1].timestamp if process_instance.events else 'None'}\n'
                return process_state
    
    def get_incomplete_process_state(self) ->str:
        """Get the current state of incomplete processes."""
        process_states = ""
        for process_instance in self.incomplete_processes:
            last_event = process_instance.events[-1] if process_instance.events else None
            process_state = f'Process ID: {process_instance.process_id}, Last Event: {last_event.action if last_event else 'No action is taken so far'}, Last Timestamp: {last_event.timestamp if last_event else 'None'}\n'
            process_states += process_state
        return process_states
    
    def update_process(self, log_event: logEvent, last_activities: List[str]):
        """Add a log event and update the process instance."""
        # add log event to all log events
        self.log_events.append(log_event)
        # write log event to the log file
        self._write_to_log_file(log_event)

        # update process instances
        if log_event.action in last_activities:
            # remove the process instance from the incomplete processes
            for process_instance in self.incomplete_processes:
                if process_instance.process_id == log_event.process_id:
                    # remove the process instance from the incomplete processes
                    self.incomplete_processes.remove(process_instance)
                    break
        else:
            # add the process instance to the incomplete processes
            process_id = log_event.process_id
            # check if the process instance is already in the incomplete processes
            if log_event.process_id not in [process_instance.process_id for process_instance in self.incomplete_processes]:
                process_instance = ProcessInstance(process_id=process_id)
                # add the log event to the process instance
                process_instance.events.append(log_event)
                self.incomplete_processes[process_id] = process_instance
            else:
                # if the process instance is already in the incomplete processes, just add the log event to it
                # and update the process instance
                for process_instance in self.incomplete_processes:
                    if process_instance.process_id == process_id:
                        # add the log event to the process instance
                        process_instance.events.append(log_event)
                        break

class ExternalFactorsManager:
    def __init__(self, config_manager: ConfigManager, llm=llm):
        self.config_manager = config_manager
        self.llm = llm
        self.recent_external_values = OrderedDict()

    def _generate_external_factors_model(self):
        """Generate external factors Pydantic model."""
        if self.config_manager.configs is None:
            simulation_configs = self.config_manager.update_and_load_configs()
        else:
            simulation_configs = self.config_manager.configs

        external_factors = simulation_configs.get('external_factors', [])
        if not external_factors:
            raise ValueError("External factors are required.")
        fields = {field: (str, Field(..., description=f"{field.capitalize()} condition")) for field in external_factors}
        return create_model("ExternalFactors", **fields)
    
    def _get_recent_external_values(self)->str:
        """"Get recent external values in string"""
        recent_external_values = ""
        for key, value in self.recent_external_values.items():
            value_str = ', '.join([f"{k}: {v}" for k, v in value.items()])
            recent_external_value = f'External conditions at {key} is: {value_str}.'
            recent_external_values += recent_external_value
        return recent_external_values
    
    def generate_external_values(self, current_time)->str:
        """Generate external values for the current step."""
        if self.config_manager.configs is None:
            simulation_configs = self.config_manager.update_and_load_configs()
        else:
            simulation_configs = self.config_manager.configs
        external_factors = simulation_configs.get('external_factors', [])

        external_factors_model = self._generate_external_factors_model()

        sys_prompt = """
        You are simulating a business process influenced by external contextual factors.
        Given the following external factors: {external_factors}, generate their current values based on the current time: {current_time} and past values: {past_external_values}.
        The output should be in a structured format, listing each external factor and its corresponding value:
        - <external_factor>: <value>
        - <external_factor>: <value>
        ...
        """
        prompt = PromptTemplate.from_template(sys_prompt)
        structured_llm = self.llm.with_structured_output(external_factors_model)
        chain = prompt | structured_llm
        external_values = chain.invoke({
            "external_factors": ', '.join(external_factors),
            "current_time": current_time,
            "past_external_values": self.recent_external_values
        })
        # Only keep at most 10 recent external values
        if len(self.recent_external_values) >= 10:
            self.recent_external_values.popitem(last=False)
        # add new external values to the recent external values    
        self.recent_external_values[current_time] = external_values.model_dump()

        return ', '.join([f"{k}: {v}" for k, v in external_values.model_dump().items()])




class SimulationEngine:
    def __init__(self, bpmn_path, config_path, log_path):
        self.config_manager = ConfigManager(config_path, bpmn_path)
        self.agent_manager = AgentManager(self.config_manager)
        self.process_manager = ProcessManager(log_path)
        self.external_factors_manager = ExternalFactorsManager(self.config_manager)

    def step(self, agent: Agent, current_time):
        """Run a single step of the simulation."""

        process_summary = self.config_manager.configs.get('process_summary', '')
        process_state = self.process_manager.get_incomplete_process_state()
        external_factors = self.config_manager.configs.get('external_factors', [])
        external_factors = ', '.join(external_factors)
        external_values = self.external_factors_manager.generate_external_values(current_time)

        out = agent.take_action(
            process_summary=process_summary,
            process_state=process_state,
            current_time=current_time,
            external_factors=external_factors,
            external_values=external_values
        )
        ####
        for agent_output in out.agent_outputs:
            if agent_output.action != None:
                if agent_output.process_id:
                    log_event = logEvent(
                        process_id= agent_output.process_id,
                        action=agent_output.action,
                        timestamp=agent_output.timestamp,
                        external_values=external_values,
                        agent=agent
                    )
                else:
                    log_event = logEvent(
                        process_id=self.process_manager.generate_process_id(),
                        action=agent_output.action,
                        timestamp=agent_output.timestamp,
                        external_values=external_values,
                        agent=agent
                    )
                self.process_manager.update_process(log_event=log_event, last_activities=self.config_manager.configs.get('start_end_events', {}).get('last_activities', []))

    def run(self, start_time, end_time, inter_arrival_time):
        """Run the simulation."""
        self.agent_manager.create_agents()
        start_time = datetime.datetime.fromisoformat(start_time)
        end_time = datetime.datetime.fromisoformat(end_time)
        inter_arrival_time = datetime.timedelta(seconds=inter_arrival_time)

        current_time = start_time
        while current_time < end_time:
            print(f"Current time: {current_time}")
            # We have to initialize the process by calling the trigger agent
            print(f'Calling trigger agent')
            trigger_agent = self.create_trigger_agent_onfly()
            # trigger agent takes action
            print(f'Trigger agent takes action')
            self.step(trigger_agent, current_time)
            # calling on organization agents to take action
            print(f'Calling cordinator agent')
            cordinator = self.agent_manager.create_cordinator()
            # cordinator hands off the process to the organization agents
            print(f'Cordinator hands off the process to the organization agents')
            process_states = self.process_manager.get_incomplete_process_state()
            org_agents = self.agent_manager.get_org_agents()
            cordinator_outputs = cordinator.hand_off(process_states=process_states, org_agents=org_agents)
            for cordinator_output in cordinator_outputs:
                # create new agent on the fly
                process_id = cordinator_output.process_id
                agent_id = cordinator_output.agent_id
                agent_role = cordinator_output.agent_role
                if agent_id:
                    agent = self.agent_manager.get_agent_by_id(agent_id)
                elif agent_role:
                    agent = self.agent_manager.create_ext_agent_onfly(agent_role)
                else:
                    raise ValueError("Either agent_id or agent_role must be provided.")
                # agent takes action
                print(f'Agent {agent.id} {agent.role} takes action')
                self.step(agent, current_time)

            