from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Optional

from simulation.bpmn_extraction import BPMNExtractor
from simulation.config_loader import ConfigLoader
from simulation.agent import Agent
from simulation.utils import generate_process_id

import random

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
    external_values: Dict[str, Any] = Field(description="External factors which can affect the agent's action and the whole process")
    agent: Agent = Field(description="Agent who performed the action")
    
class ProcessInstance(BaseModel):
    """Process instance for the simulation"""
    process_id: str = Field(description="Unique identifier for the process instance")
    events: List[logEvent] = Field(default_factory=list, description="List of events in the process instance")

class SimulationEngine:
    def __init__(self, bpmn_path, config_path):
        self.bpmn_path = bpmn_path
        self.config_path = config_path
        self.agents: List[Agent] = []
        self.log_events: List[logEvent] = []
        self.incomplete_processes: Dict[str:ProcessInstance] = {}
    
    def _extract_process_data(self):
        """Extract process data from BPMN file."""
        bpmn_extractor = BPMNExtractor(self.bpmn_path, self.config_path)
        all_process_data_dict = bpmn_extractor.extract_all_process_data()
        return all_process_data_dict
    
    def _update_configs(self):
        """Update the configs with the process data"""
        process_data = self._extract_process_data()
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
        self._update_configs()
        config_loader = ConfigLoader(self.config_path)
        simulation_configs = config_loader.get_config()
        return simulation_configs
    
    def create_agents(self):
        """Create agents based on the simulation configs"""
        simulation_configs = self.load_configs()
        process_sumary = simulation_configs['process_data'].get('process_summary', '')
        external_factors = simulation_configs.get('external_factors', [])
        demographics = simulation_configs.get('demographics', [])
        agent_numbers = simulation_configs.get('num_agents', {})
        for agent_role, actions in simulation_configs['process_data'].get('agent_actions', {}).items():
            if agent_role in agent_numbers:
                num_agents = agent_numbers[agent_role]
            else:
                num_agents = 1
            for _ in range(num_agents):
                agent = Agent(
                    process_summary=process_sumary,
                    role=agent_role,
                    demographics_attributes=demographics,
                    external_factors=external_factors,
                    actions=actions
                )
                agent.create_prompt()
                self.agents.append(agent)
    
    # create a pydantic model for the external factors
    def _generate_external_factors_model(self):
        """Generate external factors pydantic model"""
        simulation_configs = self.load_configs()
        external_factors = simulation_configs.get('external_factors', [])
        if not external_factors:
            raise ValueError("External factors are required to generate external factors.")
        # create string from list of external factors
        fields = {field: (str, Field(None, description=f"{field.capitalize()} condition, which can affect the agent's action and the whole process")) for field in external_factors}
        external_factors_model = create_model("ExternalFactors", **fields)
        return external_factors_model
    
    # generate external factors of the agent
    def _generate_external_values(self, current_time, external_factors, past_external_values, llm = llm):
        """Generate external factors of the agent"""
        # create string from list of external factors
        external_factors_list = ', '.join(external_factors)
        # pydantic model for external factors
        external_factors_model = self._generate_external_factors_model()

        sys_prompt = """
        You are simulating a business process influenced by external contextual factors.

        You are provided with:
        - A list of external attributes
        - The current timestamp
        - The most recent past values for each external attribute

        Your task is to generate realistic and contextually appropriate random values for each external attribute.

        Guidelines:
        - Ensure the values align with the semantics of each attribute and the current time.
        For example:
        - Temperature should fall within plausible ranges.
        - Weather should logically correspond to the temperature (e.g., no snow at 35°C).
        - Traffic levels should follow typical daily patterns (e.g., traffic is unlikely to be busy at 3 AM).

        Input:
        - External attributes: {external_factors_list}
        - Current time: {current_time}
        - Past external factor values: {past_external_values}

        Please output the results in the following structured format:
        - attribute: value
        - attribute: value
        ...
        """
        prompt = PromptTemplate.from_template(sys_prompt)
        structured_llm = llm.with_structured_output(external_factors_model)
        chain = prompt | structured_llm
        external_values = chain.invoke({"external_factors": external_factors_list, "current_time": current_time, "past_external_values": past_external_values})
        return external_values
    
    def _get_incomplete_process_state(self):
        """Get the current state of the incomplete process"""
        process_states = {}
        incomplete_process = self.incomplete_processes
        for _, incomplete_process in self.incomplete_processes.items():
            # get the process id of the process instance
            process_id = incomplete_process.process_id
            # get the events of the process instance
            events = incomplete_process.events
            # get the lastest event and timestamp of the process
            last_event = events[-1] if events else None
            if last_event:
                process_state = {
                    "process_id": process_id,
                    "last_event": last_event.action,
                    "last_timestamp": last_event.timestamp,
                }
            else:
                process_state = {
                    "process_id": process_id,
                    "last_event": None,
                    "last_timestamp": None,
                }
            process_states[process_id] = process_state
        return process_states
    
    def _get_current_external_values(self):
        """Get the current external values of the process"""
        current_external_values = []
        # take the lastest 10 events of the process
        for event in self.log_events[-10:]:
            external_value = {event.timestamp: event.external_values}
            current_external_values.append(external_value)
        return current_external_values
    
    def _act2role(self, activities:str)-> List[str]:
        """decide which roles can perform the activities"""
        simulation_configs = self.load_configs()
        agent_actions = simulation_configs.get('process_data').get('agent_actions', {})
        # get all possibles roles who can perform any or all of the given activities
        roles = []
        for activity in activities:
            # get the roles of the agents
            for role, actions in agent_actions.items():
                if activity in actions:
                    roles.append(role)
        # remove duplicates from the list of roles
        roles = list(set(roles))
        return roles
    
    def _get_next_activities(self, activity:str)-> List[str]:
        """Get the next activities of the process"""
        simulation_configs = self.load_configs()
        activities_flow = simulation_configs.get('process_data').get('activities_flow', {})
        # get the next activities of the process
        next_activities = activities_flow.get(activity, [])
        return next_activities

    def _write_to_log(self, agent:Agent, agent_outputs, external_values):
        """Write the outputs of the agent to the log file"""
        # write to logs
        for output in agent_outputs.agent_outputs:
            # get the process id of the process instance
            process_id = output.process_id
            # get the action of the agent
            action = output.action
            # get the timestamp of the action
            timestamp = output.timestamp
            # get the external values of the agent
            # create a log event for the simulation
            log_event = logEvent(
                process_id=process_id,
                action=action,
                timestamp=timestamp,
                external_values=external_values,
                agent=agent
            )
            self.log_events.append(log_event)
            # check if the action ends the process
            simulation_configs = self.load_configs()
            last_activities = simulation_configs.get('process_data').get('start_end_events', {}).get('last_activities', []) 
            if action in last_activities:
                # remove the process instance from the incomplete processes 
                self.incomplete_processes.pop(process_id, None)
            else:
                # add the event log to the process instance
                if process_id not in self.incomplete_processes:
                    process_instance = ProcessInstance(process_id=process_id)
                    process_instance.events.append(log_event)
                    self.incomplete_processes[process_id] = process_instance
                else:
                    process_instance = self.incomplete_processes[process_id]
                    process_instance.events.append(log_event)
                    self.incomplete_processes[process_id] = process_instance         

    def step(self, agent: Agent, process_state, current_time, past_external_values)-> logEvent:
        """Run a single step of the simulation"""
        prompt = agent.prompt
        simulation_configs = self.load_configs()
        # get the external factors of the simulation
        external_factors = simulation_configs.get('external_factors', [])
        # get start and end events of the process
        start_end_events = simulation_configs.get('process_data').get('start_end_events', {})
        # get agents actions of the process
        agent_actions = simulation_configs.get('process_data').get('agent_actions', {})
        # get activities flow of the process
        activities_flow = simulation_configs.get('process_data').get('activities_flow', {})
        # get the process summary of the process
        process_summary = simulation_configs.get('process_data').get('process_summary', '')
        # get latest process state of the process
        process_state = self._get_incomplete_process_state()
        # get external values
        external_values = self._generate_external_values(
            current_time=current_time,
            external_factors=external_factors,
            past_external_values=self._get_current_external_values()
        )
        # agent taking action
        agent_outputs = agent.take_action(
            process_summary=process_summary,
            process_state=process_state,
            current_time=current_time,
            external_factors=external_factors,
            external_values=external_values
        )
        # write the outputs of the agent to the log file
        self._write_to_log(agent=agent, agent_outputs=agent_outputs, external_values=external_values)

    def run(self, start_time, end_time, inter_arrival_time):
        """Run the simulation"""

        current_time = start_time   # Example current time 
        # get simulation configs first
        simulation_configs = self.load_configs()
        # get the external factors of the simulation
        external_factors = simulation_configs.get('external_factors', [])
        # get start and end events of the process
        start_end_events = simulation_configs.get('process_data').get('start_end_events', {})

        while current_time < end_time:
            # create a process instance
            process_id = generate_process_id([event.process_id for event in self.log_events])
            process_instance = ProcessInstance(process_id=process_id)
            # get first activity of the process
            first_activities = start_end_events.get('start_event', {}).get('first_activities', None)
            # get roles who can perform the first activities
            roles = self._act2role(first_activities)
            # pick one role randomly from the list of roles
            role = random.choice(roles)
            # get random agents from the list of agents who has the same role
            agents = [agent for agent in self.agents if agent.role == role]
            # pick one agent randomly from the list of agents
            agent = random.choice(agents)





    