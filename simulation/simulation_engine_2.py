from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

from .bpmn_extraction import BPMNExtractor
from .config_loader import ConfigLoader
from .agent import Agent
from .utils import generate_process_id

import random
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
    external_values: str = Field(description="External factors which can affect the agent's action and the whole process")
    agent: Agent = Field(description="Agent who performed the action")
    
class ProcessInstance(BaseModel):
    """Process instance for the simulation"""
    process_id: str = Field(description="Unique identifier for the process instance")
    events: List[logEvent] = Field(default_factory=list, description="List of events in the process instance")


class ConfigManager:
    def __init__(self, config_path, bpmn_path):
        self.config_path = config_path
        self.bpmn_path = bpmn_path
        self.configs = None

    def _extract_process_data(self):
        """Extract process data from the BPMN file."""
        bpmn_extractor = BPMNExtractor(self.bpmn_path, self.config_path)
        return bpmn_extractor.extract_all_process_data()

    def _load_existing_configs(self):
        """Load existing simulation configs if the file exists, otherwise return an empty dict."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_configs(self, configs):
        """Save the merged configs to the config file."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(configs, f, ensure_ascii=False, indent=4)

    def update_and_load_configs(self):
        """Update config with extracted process data and return the full merged config."""
        existing_configs = self._load_existing_configs()
        process_data = self._extract_process_data()

        if not isinstance(existing_configs, dict) or not isinstance(process_data, dict):
            raise ValueError("Both existing configs and process data must be dictionaries.")

        merged_configs = {**existing_configs, **process_data}
        self._save_configs(merged_configs)

        self.configs = merged_configs
        return merged_configs
    
class AgentManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.agents: List[Agent] = []

    def create_agents(self):
        """Create agents based on the configuration."""
        simulation_configs = self.config_manager.update_and_load_configs()
        role_actions = simulation_configs.get('role_actions', {})
        demographics = simulation_configs.get('demographics', [])
        agent_numbers = simulation_configs.get('num_agents', {})

        for agent_role, actions in role_actions.items():
            if actions is not None:
                num_agents = agent_numbers.get(agent_role, 2)
                for _ in range(num_agents):
                    agent = Agent(
                        role=agent_role,
                        demographics_attributes=demographics,
                        actions=actions,
                    )
                    agent.create_prompt()
                    self.agents.append(agent)

class ProcessManager:
    def __init__(self, log_path):
        self.incomplete_processes: Dict[str, ProcessInstance] = {}
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
    
    def initialize_process(self):
        """Initialize a new process instance."""
        # generate a unique process id
        used_process_ids = [log.process_id for log in self.log_events]
        process_id = generate_process_id(used_ids=used_process_ids)
        # create a new process instance
        process_instance = ProcessInstance(process_id=process_id)
        # add the process instance to the incomplete processes
        self.incomplete_processes[process_id] = process_instance
    
    def get_incomplete_process_state(self) ->str:
        """Get the current state of incomplete processes."""
        process_states = ""
        for process_id, process_instance in self.incomplete_processes.items():
            last_event = process_instance.events[-1] if process_instance.events else None
            process_state = f'"Process ID: {process_id}, Last Event: {last_event.action if last_event else 'No action is taken so far'}, Last Timestamp: {last_event.timestamp if last_event else 'None'}'
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
            if log_event.process_id in self.incomplete_processes:
                del self.incomplete_processes[log_event.process_id]
        else:
            # add the process instance to the incomplete processes
            process_id = log_event.process_id
            # check if the process instance is already in the incomplete processes
            if log_event.process_id not in self.incomplete_processes:
                process_instance = ProcessInstance(process_id=process_id)
                # add the log event to the process instance
                process_instance.events.append(log_event)
                self.incomplete_processes[process_id] = process_instance
            else:
                # if the process instance is already in the incomplete processes, just add the log event to it
                # and update the process instance
                process_instance = self.incomplete_processes[process_id]
                process_instance.events.append(log_event)

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
            log_event = logEvent(
                process_id= agent_output.process_id,
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

        """current_time = start_time
        while current_time < end_time:
            print(f"Current time: {current_time}")
            # We have to initialize the process, it is better than letting llms generate process id because they will always generate the same id
            self.process_manager.initialize_process()
            # calling on agents to take action
            for agent in self.agent_manager.agents:
                self.step(agent, current_time)
            current_time += inter_arrival_time"""

