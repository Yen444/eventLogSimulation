from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import PromptTemplate

from .agent import Agent
from .agent_manager import AgentManager
from .config_manager import ConfigManager
from .process_manager import ProcessManager, logEvent

from collections import OrderedDict
import datetime
import os

from dotenv import load_dotenv
import os
import random

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


class SimulationEngine:
    def __init__(self, llm, config_path, log_path, arrivals):
        self.config_manager = ConfigManager(config_path=config_path)
        self.agent_manager = AgentManager(self.config_manager, llm)
        self.process_manager = ProcessManager(log_path)
        self.llm = llm
        self.arrivals = arrivals # List of tuple of simulated arrival (timestamp, lamda)

    def step(self, agent: Agent, process_id, current_time, update_interval, message, trigger=False, reflect=True):
        """Run a single step of the simulation."""

        process_summary = self.config_manager.configs.get('process_summary', '')
        action_data_mapping = self.config_manager.configs.get('action_data', '')
        event_data_attributes = self.config_manager.configs.get('event_attributes')
        case_data_attributes = self.config_manager.configs.get('case_attributes')
        conventions = self.config_manager.configs.get('conventions', [])

        process_state = self.process_manager.get_related_process(process_id=process_id)
        # Make sure Agent is not None
        if agent is not None:
            # agent takes action
            out = agent.act(
                llm=self.llm,
                process_summary=process_summary,
                process_state=process_state,
                action_data_mapping=action_data_mapping,
                current_time=current_time,
                event_data_attributes=event_data_attributes,
                case_data_attributes=case_data_attributes,
                update_interval=update_interval,
                message=message,
                output_model=self.agent_manager.ListAgentOutput,
                reflect=reflect,
                conventions=conventions
            )
            print(f'Agent takes {len(out.agent_outputs)} actions')
            if out:
                for agent_output in out.agent_outputs:
                    if agent_output.action != None:
                        if trigger:
                            log_event = logEvent(
                                process_id=self.process_manager.generate_process_id(),
                                agent=agent,
                                event=agent_output
                            )
                        else:
                            log_event = logEvent(
                                process_id=agent_output.process_id,
                                agent=agent,
                                event=agent_output
                            )
                        self.process_manager.update_process(log_event=log_event, last_activities=self.config_manager.configs.get('last_activities', []))
            
    def run(self, start_time, end_time=None, inter_arrival_time=0, num_traces=None):
        """Run the simulation.
        Either `end_time` or `num_traces` must be provided.
        """
        if end_time is None and num_traces is None:
            raise ValueError("You must provide either end_time or num_traces.")
        
        print(f'creating coordinator agent')
        self.agent_manager.create_coordinator_agent()
        print(f'creating organization agents')
        self.agent_manager.create_org_agents()
        
        start_time = datetime.datetime.fromisoformat(start_time)

        if end_time is not None:
            end_time = datetime.datetime.fromisoformat(end_time)

        inter_arrival_time = datetime.timedelta(hours=inter_arrival_time)

        current_time = start_time

        while True:
            if end_time is not None and current_time >= end_time:
                break
            if num_traces is not None and self.process_manager.num_complete_processes > num_traces:
                break

            print(f"Current time: {current_time}")
            # coordinator calls on agent to initialize the process

            # Check if any arrivals within the time period current_time until the next timestamp
            next_current_time = current_time + inter_arrival_time
            sub_arrivals = [arrival for arrival in self.arrivals if current_time <= arrival[0] < next_current_time]

            print(f'Number of arrivals in this updating step: {len(sub_arrivals)}')

            if len(sub_arrivals) > 0:
                for i, (arrival_time,_) in enumerate(sub_arrivals):
            
                    coordinator_outputs = self.agent_manager.coordinator.hand_off(process_states='', org_agents=self.agent_manager.org_agents, external_agents=self.agent_manager.ext_agents, current_timestamp=current_time)

                    #print(f'Coordinator output: {coordinator_outputs}')
                    for coordinator_output in coordinator_outputs.coordinator_outputs:
                        if coordinator_output.process_id:
                            process_id = coordinator_output.process_id
                        else:
                            process_id = None
                        agent_id = coordinator_output.agent_id
                        agent_role = coordinator_output.agent_role
                        message = coordinator_output.message
                        print(f"Coordinator hands off to trigger agent: {agent_role} with agent_id: {agent_id}")
                        if agent_id:
                            agent = self.agent_manager.get_agent_by_id(agent_id)
                        elif agent_role:
                            agent = self.agent_manager.create_ext_agent_onfly(agent_role)
                        else:
                            raise ValueError("Either agent_id or agent_role must be provided.")
                        
                        print(f'Trigger agent takes action')

                        self.step(agent=agent, process_id=process_id, current_time=arrival_time, update_interval=inter_arrival_time, message=message, trigger=True, reflect=False)
            
            update_current_time = current_time
            
            for i in range(2):
                print(f'coordinator hands off the process to the organization or external agents')
                process_states = self.process_manager.get_incomplete_process_state(current_time=update_current_time)
                if process_states == "":
                    update_current_time += (inter_arrival_time/2)
                    print('There is no process to update')
                    continue
                org_agents = self.agent_manager.org_agents
                external_agents = self.agent_manager.ext_agents
                coordinator_outputs = self.agent_manager.coordinator.hand_off(process_states=process_states, org_agents=org_agents, external_agents=external_agents, current_timestamp=update_current_time)
                #print(f'Coordinator output: {coordinator_outputs}')
                if coordinator_outputs is None or len(coordinator_outputs.coordinator_outputs) == 0:
                    print('No coordinator output, no agent to call')
                    update_current_time += (inter_arrival_time/2)
                    continue
                for coordinator_output in coordinator_outputs.coordinator_outputs:
                    if coordinator_output.process_id:
                        process_id = coordinator_output.process_id
                    else:
                        process_id = None
                    agent_id = coordinator_output.agent_id
                    agent_role = coordinator_output.agent_role
                    message = coordinator_output.message
                    if agent_id:
                        agent = self.agent_manager.get_agent_by_id(agent_id)
                    elif agent_role:
                        # still have to check in agent role is an organization agent or not
                        matching_agents = [a for a in org_agents if a.role == agent_role]
                        if matching_agents:
                            agent = random.choice(matching_agents)
                        else:
                            print(f'Create Ext Agent on fly')
                            agent = self.agent_manager.create_ext_agent_onfly(agent_role)

                    else:
                        raise ValueError("Either agent_id or agent_role must be provided.")
                    # agent takes action
                    print(f'Agent {agent.id} {agent.role} takes action')
                    self.step(agent=agent, process_id=process_id, current_time=update_current_time, update_interval=inter_arrival_time/2, message=message, trigger=False, reflect=False)

                    update_current_time += (inter_arrival_time/2)
            
            current_time += inter_arrival_time

            