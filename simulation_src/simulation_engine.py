from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import PromptTemplate

from .agent import Agent
from .agent_manager import AgentManager
from .config_manager import ConfigManager
from .tinydb_store import TinyDBStore

from .prompt_manager import create_prompt_manager
from .process_manager_db import ProcessManagerDB, logEvent

import datetime
import os
import random
import csv
from collections import OrderedDict

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('GOOGLE_API_KEY')
api_key_openai = os.getenv('OPENAI_API_KEY')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

llm_low = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.0) # gemini-2.0-flash 0.0005
llm_mid = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.5) # gemini-2.0-flash-lite 0.0004 
llm_high = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=1.0) # gemini-2.5-flash-lite 0.0005

llm_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0.5,
    openai_api_key=api_key_openai
)


class SimulationEngine:
    def __init__(self, llm, config_path, log_path, protocol_path, db_path, arrivals):
        self.config_manager = ConfigManager(config_path=config_path)
        # create a PromptManager and pass it to AgentManager
        prompt_manager = create_prompt_manager()
        # create a single TinyDBStore and share it between AgentManager and ProcessManagerDB
        self.store = TinyDBStore(db_path)

        # ProcessManagerDB (use the shared store)
        self.process_manager_db = ProcessManagerDB(log_path, store=self.store)

        # create AgentManager and pass the shared store so it can register/get agents from DB
        self.agent_manager = AgentManager(self.config_manager, prompt_manager, llm, store=self.store)

        self.protocol_path = protocol_path
        self.llm = llm
        self.arrivals = arrivals # List of tuple of simulated arrival (timestamp, lamda)
        # iteration counter used to track steps/hand-offs in the simulation
        self.iteration_id = 0

        self._initialize_protocol_file()

    def _initialize_protocol_file(self):
        """Ensure the protocol file exists and has a header."""
        if not os.path.exists(self.protocol_path):
            os.makedirs(os.path.dirname(self.protocol_path), exist_ok=True)
            with open(self.protocol_path, 'w', encoding='utf-8') as f:
                f.write("")
        if os.path.getsize(self.protocol_path) == 0:
            # open with newline='' so csv.writer won't add extra blank lines on Windows
            with open(self.protocol_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # store combined iteration id as iteration_sub (e.g. 0_0, 0_1)
                writer.writerow(["iteration_id", "timestamp", "from_agent", "to_agent", "process_id", "message", "process_state"])
                # add a blank line after the header to separate header from data
                f.write('\n')

    def log_protocol(self, timestamp, from_agent, to_agent, process_id, message, process_state="", combined_iteration=None):
        """Log the protocol of the simulation."""
        def _write_to_csv():
            with open(self.protocol_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([combined, timestamp, from_agent, to_agent, process_id, message, process_state])
                f.write('\n')

        combined = combined_iteration if combined_iteration is not None else f'{self.iteration_id}'
        _write_to_csv()
        try:
            self.process_manager_db.record_protocol(
                iteration_id=combined,
                timestamp=timestamp,
                from_agent=from_agent,
                to_agent=to_agent,
                process_id=process_id,
                message=message,
                process_state=process_state
            )
        except Exception:
            pass

    def _process_arrivals(self, current_time, inter_arrival_time):
        """Helper function to process arrivals within the current time window."""
        next_current_time = current_time + inter_arrival_time
        sub_arrivals = [arrival for arrival in self.arrivals if current_time <= arrival[0] < next_current_time]
        print(f'Number of arrivals in this updating step: {len(sub_arrivals)}')
        return sub_arrivals

    def _process_agent_outputs(self, agent, agent_outputs, process_state, combined_iteration, trigger, last_activities):
        """Helper function to process agent outputs."""
        multi_output = len(agent_outputs) > 1
        for idx, agent_output in enumerate(agent_outputs):
            safe_combined = f"{combined_iteration}_{idx}" if multi_output else combined_iteration
            log_event = logEvent(
                process_id=agent_output.process_id if not trigger else self.process_manager_db.generate_process_id(),
                agent=agent,
                event=agent_output,
                process_state=process_state,
                iteration_id=safe_combined
            )
            if agent_output.action is not None:
                self.process_manager_db.update_process(log_event=log_event, last_activities=last_activities)
            else:
                self.process_manager_db.write_to_log_file(log_event)

    def _step(self, agent: Agent, process_id, current_time, update_interval, message, trigger, reflect, combined_iteration=None):
        """Run a single step of the simulation.

        This method is intended as an internal helper. Use `run` to drive the
        full simulation externally. Renamed to `_step` to mark it as internal.
        """
        process_summary = self.config_manager.configs.get('process_summary', '')
        action_data_mapping = self.config_manager.configs.get('action_data', '')
        event_data_attributes = self.config_manager.configs.get('event_attributes')
        case_data_attributes = self.config_manager.configs.get('case_attributes')
        conventions = self.config_manager.configs.get('conventions', [])
        last_activities = self.config_manager.configs.get('last_activities', []) or []

        process_state = self.process_manager_db.get_related_process_state(process_id=process_id)
        if agent is not None:
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
            if out is None or not getattr(out, 'agent_outputs'):
                print(f"Agent {getattr(agent, 'id', 'unknown')} produced no outputs")
                return None

            print(f"Agent takes {len(out.agent_outputs)} actions")
            self._process_agent_outputs(agent, out.agent_outputs, process_state, combined_iteration, trigger, last_activities)

    def _handle_coordinator_outputs(self, coordinator_outputs, current_time, combined_iteration, trigger):
        """Helper function to handle coordinator outputs."""
        for j, coordinator_output in enumerate(coordinator_outputs.coordinator_outputs):
            process_id = coordinator_output.process_id
            agent_id = coordinator_output.agent_id
            agent_role = coordinator_output.agent_role
            message = coordinator_output.message
            print(f"Coordinator hands off to trigger agent: {agent_role} with agent_id: {agent_id}")
            if agent_id:
                agent = self.agent_manager.get_agent_by_id(agent_id)
            elif agent_role:
                agent = self.agent_manager.create_register_ext_agent_onfly(agent_role)
            else:
                raise ValueError("Either agent_id or agent_role must be provided.")

            print(f'Trigger agent takes action')
            self._step(agent=agent, process_id=process_id, current_time=current_time, update_interval=datetime.timedelta(hours=1), message=message, trigger=trigger, combined_iteration=combined_iteration, reflect=False)

    def run(self, start_time, end_time=None, inter_arrival_time=0, num_traces=None):
        """Run the simulation.
        Either `end_time` or `num_traces` must be provided.
        """
        # create and register coordinator agent
        self.agent_manager.create_register_coordinator_agent()
        # create and register org agents
        self.agent_manager.create_register_org_agents()

        if end_time is None and num_traces is None:
            raise ValueError("You must provide either end_time or num_traces.")

        start_time = datetime.datetime.fromisoformat(start_time)
        if end_time is not None:
            end_time = datetime.datetime.fromisoformat(end_time)

        inter_arrival_time = datetime.timedelta(hours=inter_arrival_time)
        current_time = start_time
        
        while True:
            if len(self.arrivals) == 0 or (end_time and current_time >= end_time) or (num_traces and self.process_manager_db.store.get_num_completed_processes() >= num_traces):
                print('Ending simulation')
                break
            
            # Process arrivals within the current time window
            sub_arrivals = self._process_arrivals(current_time, inter_arrival_time)
            
            # Coordinator calls agent to trigger process for each arrival
            if sub_arrivals:
                for i, (arrival_time, _) in enumerate(sub_arrivals):
                    combined_iteration = f'{self.iteration_id}_0_{i}'
                    # provide coordinator with current incomplete process states so it can make informed handoffs
                    process_states = ""
                    raw, coordinator_outputs = self.agent_manager.coordinator.hand_off(process_states=process_states, org_agents=self.agent_manager.org_agents, external_agents=self.agent_manager.ext_agents, current_timestamp=current_time)
                    print(coordinator_outputs)
                    if not coordinator_outputs or not coordinator_outputs.coordinator_outputs:
                        # coordinator returned nothing; log an empty process_state
                        self.log_protocol(timestamp=arrival_time, from_agent='coordinator', to_agent='', process_id=None, message='', process_state='', combined_iteration=combined_iteration)
                        self.arrivals.pop(0)
                        continue
                    self._handle_coordinator_outputs(coordinator_outputs, current_time=arrival_time, combined_iteration=combined_iteration, trigger=True)
                    self.arrivals.pop(0)
            
            # Update process states for all incomplete processes
            for j in range(3): # update 3 times within the inter_arrival_time window
                sub_current_time = current_time + j * (inter_arrival_time / 3)
                combined_iteration = f'{self.iteration_id}_1_{j}'
                process_states = self.process_manager_db.get_all_incomplete_process_state(sub_current_time)
                raw, coordinator_outputs = self.agent_manager.coordinator.hand_off(process_states=process_states, org_agents=self.agent_manager.org_agents, external_agents=self.agent_manager.ext_agents, current_timestamp=sub_current_time)
                print(coordinator_outputs)
                if not coordinator_outputs or not coordinator_outputs.coordinator_outputs:
                    # coordinator returned nothing; log an empty process_state
                    self.log_protocol(timestamp=sub_current_time, from_agent='coordinator', to_agent='', process_id=None, message='', process_state='', combined_iteration=combined_iteration)
                    self.arrivals.pop(0)
                    continue
                self._handle_coordinator_outputs(coordinator_outputs, current_time=sub_current_time, combined_iteration=combined_iteration, trigger=False)
                self.arrivals.pop(0)

            current_time += inter_arrival_time
            self.iteration_id += 1

