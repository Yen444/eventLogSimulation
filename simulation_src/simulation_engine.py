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
    def __init__(self, llm, config_path, log_path, protocol_path, db_path, arrivals, use_fewshots: bool = True, use_tools: bool = True, agent_data_creation_reflect: bool = False, action_reflect: bool = True):
        self.config_manager = ConfigManager(config_path=config_path)
        # create a PromptManager and pass it to AgentManager
        prompt_manager = create_prompt_manager(use_fewshots=use_fewshots)
        # create a single TinyDBStore and share it between AgentManager and ProcessManagerDB
        self.store = TinyDBStore(db_path)

        # ProcessManagerDB (use the shared store)
        self.process_manager_db = ProcessManagerDB(log_path, store=self.store)

        self.action_reflect = action_reflect
        self.use_tools = use_tools
        self.agent_data_creation_reflect = agent_data_creation_reflect

        # create AgentManager and pass the shared store so it can register/get agents from DB
        self.llm = llm
        self.agent_manager = AgentManager(config_manager=self.config_manager, prompt_manager=prompt_manager, store=self.store, llm=self.llm, use_tools=self.use_tools, agent_data_creation_reflect=self.agent_data_creation_reflect)

        self.protocol_path = protocol_path
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
        #print(f'Number of arrivals in this updating step: {len(sub_arrivals)}')
        return sub_arrivals

    def _process_agent_outputs(self, agent, agent_outputs, process_state, combined_iteration, trigger, last_activities):
        """Helper function to process agent outputs."""
        multi_output = len(agent_outputs) > 1
        start_activities = self.config_manager.configs.get('start_activities', []) or []
        start_activities = [act.strip().lower() for act in start_activities if isinstance(act, str)]

        # Case: multiple outputs, all empty process_ids, one is start activity
        all_empty = all(not getattr(out, 'process_id', None) for out in agent_outputs)
        has_start_activity = any(
            getattr(out, 'action', '').strip().lower() in start_activities for out in agent_outputs
        )

        if multi_output and all_empty and has_start_activity:
            new_process_id = self.process_manager_db.generate_process_id()
            for idx, agent_output in enumerate(agent_outputs):
                safe_combined = f"{combined_iteration}_{idx}" if multi_output else combined_iteration
                log_event = logEvent(
                    process_id=new_process_id,
                    agent=agent,
                    event=agent_output,
                    process_state=process_state,
                    iteration_id=safe_combined
                )
                self.process_manager_db.update_process(log_event=log_event, last_activities=last_activities)
            return  # prevent double-logging

        # Regular processing
        for idx, agent_output in enumerate(agent_outputs):
            process_id = getattr(agent_output, 'process_id', None)
            action = getattr(agent_output, 'action', None)
            if not action:
                continue

            action = action.strip().lower()
            is_valid, reason = self._check_valid_action(process_id, action)
            if not is_valid:
                print(f"Skipping invalid action: {reason}")
                continue

            process_id = process_id or (self.process_manager_db.generate_process_id() if trigger else None)
            safe_combined = f"{combined_iteration}_{idx}" if multi_output else combined_iteration

            log_event = logEvent(
                process_id=process_id,
                agent=agent,
                event=agent_output,
                process_state=process_state,
                iteration_id=safe_combined
            )
            self.process_manager_db.update_process(log_event=log_event, last_activities=last_activities)

        # BUG: if an agent produces multiple outputs with empty process_id, only the first one will be recorded with a new process_id, the rest will be skipped as invalid

    def _check_valid_action(self, process_id, action: str):
        """Validate whether an action is valid given start_activities and existing processes.

        Rules:
        - If action is in start_activities: it's a trigger/start action, it is valid.
        - If action is not in start_activities: the process_id MUST already exist in the DB, otherwise the agent output is invalid.

        Returns: (is_valid: bool, reason: str)
        """

        start_activities = self.config_manager.configs.get('start_activities', []) or []
        start_activities = [act.strip().lower() for act in start_activities if isinstance(act, str)]
        # if action is a start activity, it's allowed regardless of process_id

        if action in start_activities:
            return True, 'start_activity'
        else:
            # the process_id muss be provided and it must exist in the DB
            if not process_id:
                return False, 'missing_process_id_for_non_start_action'
            else:
                # check DB for existence
                if process_id not in self.process_manager_db.store.get_all_process_ids():
                    return False, 'process_id_not_found'
                else:
                    return True, 'ok'
    
    def _step(self, agent: Agent, process_id, current_time, message, trigger, reflect, combined_iteration):
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
        if trigger:
            # if this is a trigger step, process_id may be None (new process)
            process_state = ""
        else:
            process_state = self.process_manager_db.get_related_process_state(process_id=process_id)
            if process_state == "" or process_state is None:
                print(f"Process state for process_id {process_id} is empty even no trigger")
        if agent is not None:
            out = agent.act(
                process_summary=process_summary,
                process_state=process_state,
                action_data_mapping=action_data_mapping,
                current_time=current_time,
                event_data_attributes=event_data_attributes,
                case_data_attributes=case_data_attributes,
                message=message,
                output_model=self.agent_manager.ListAgentOutput,
                reflect=reflect,
                conventions=conventions,
                llm=self.llm
            )
            if out is None or not getattr(out, 'agent_outputs'):
                #print(f"Agent {getattr(agent, 'id', 'unknown')} produced no outputs")
                return None

            #print(f"Agent takes {len(out.agent_outputs)} actions")
            self._process_agent_outputs(agent, out.agent_outputs, process_state, combined_iteration, trigger, last_activities)

    def _handle_coordinator_outputs(self, coordinator_outputs, current_time, combined_iteration, trigger, process_state):
        """Helper function to handle coordinator outputs."""
        for j, coordinator_output in enumerate(coordinator_outputs.coordinator_outputs):
            process_id = coordinator_output.process_id if trigger is False else None
            agent_id = coordinator_output.agent_id
            agent_role = coordinator_output.agent_role
            message = coordinator_output.message
            is_valid, reason = self._check_valid_handoff(process_id, trigger)
            if is_valid is False:
                print(f"Skipping invalid handoff: {reason}")
                continue  # skip invalid handoffs

            if agent_id:
                agent = self.agent_manager.get_agent_by_id(agent_id)
            elif agent_role:
                agent = self.agent_manager.create_register_ext_agent_onfly(agent_role)
            else:
                raise ValueError("Either agent_id or agent_role must be provided.")
            # write coordinator handoff to protocol
            self.log_protocol(timestamp=current_time, from_agent='coordinator', to_agent=agent.id if agent else '', process_id=process_id, message=message, process_state=process_state, combined_iteration=combined_iteration)
            #print(f'Trigger agent takes action')
            self._step(agent=agent, process_id=process_id, current_time=current_time, message=message, trigger=trigger, combined_iteration=combined_iteration, reflect=self.action_reflect)
    
    def _check_valid_handoff(self, process_id, trigger):
        """Validate whether a handoff is valid given existing processes and agents.

        Rules:
        - if trigger is True, process_id must be None or empty
        - if trigger is False: process_id must exist in the DB

        Returns: (is_valid: bool, reason: str)
        """
        if trigger:
            if process_id not in (None, ""):
                return False, 'process_id_must_be_none_or_empty_for_trigger'
            else:
                return True, 'ok'
        else:
            if not process_id:
                return False, 'missing_process_id_for_handoff'
            else:
                if process_id not in self.process_manager_db.store.get_all_process_ids():
                    return False, 'process_id_not_found'
                else:
                    return True, 'ok'

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

        # determine starting time: prefer persisted latest timestamp if present
        persisted_latest = None
        try:
            persisted_latest = self.process_manager_db.store.get_latest_timestamp()
        except Exception:
            persisted_latest = None

        if persisted_latest:
            try:
                start_time = datetime.datetime.fromisoformat(persisted_latest)
            except Exception:
                # fallback to provided start_time string
                start_time = datetime.datetime.fromisoformat(start_time)
        else:
            start_time = datetime.datetime.fromisoformat(start_time)
        if end_time is not None:
            end_time = datetime.datetime.fromisoformat(end_time)

        inter_arrival_time = datetime.timedelta(hours=inter_arrival_time)
        current_time = start_time

        # Filter arrivals so new arrivals start from the latest timestamp
        try:
            # ensure self.arrivals is list of tuples (datetime, lambda)
            filtered = []
            for a in self.arrivals:
                if isinstance(a[0], str):
                    try:
                        a_ts = datetime.datetime.fromisoformat(a[0])
                    except Exception:
                        # assume it's a float timestamp
                        a_ts = datetime.datetime.fromtimestamp(float(a[0]))
                else:
                    a_ts = a[0]
                if a_ts >= current_time:
                    filtered.append((a_ts, a[1]))
            self.arrivals = filtered
        except Exception:
            pass

        # No explicit startup incomplete-process scan — the regular sub-step loop will
        # pick up incomplete processes and hand them off to the coordinator.
        simulation_start_time = datetime.datetime.now()
        while True:
            complete_processes = self.process_manager_db.store.get_num_completed_processes()
            print(f'Number of completed processes so far: {complete_processes}')
            # determine why the simulation should end and print reasons for diagnostics
            end_reasons = []
            if not self.arrivals:
                end_reasons.append('no_more_arrivals')
            if end_time is not None and current_time >= end_time:
                end_reasons.append('end_time_reached')
            if num_traces is not None and complete_processes >= num_traces:
                end_reasons.append('num_traces_reached')

            if end_reasons:
                # join multiple reasons if more than one applies
                print(f"Ending simulation: {', '.join(end_reasons)}")
                print(f"Simulation took {datetime.datetime.now() - simulation_start_time} for {complete_processes}")
                break
            
            # Process arrivals within the current time window
            sub_arrivals = self._process_arrivals(current_time, inter_arrival_time)
            
            # Coordinator calls agent to trigger process for each arrival
            if sub_arrivals:
                for i, (arrival_time, _) in enumerate(sub_arrivals):
                    combined_iteration = f'{self.iteration_id}_0_{i}'
                    # provide coordinator with current incomplete process states so it can make informed handoffs
                    process_states = "Process state is empty, you must call an agent to start a new process."
                    coordinator_outputs = self.agent_manager.coordinator.hand_off(process_states=process_states, org_agents=self.agent_manager.org_agents, external_agents=self.agent_manager.ext_agents, current_timestamp=current_time)
                    if not coordinator_outputs or not coordinator_outputs.coordinator_outputs:
                        print("Coordinator returned no outputs during arrival step.")
                        self.log_protocol(timestamp=arrival_time, from_agent='coordinator', to_agent='', process_id=None, message='', process_state='', combined_iteration=combined_iteration)
                        self.arrivals.pop(0)
                        continue
                    self._handle_coordinator_outputs(coordinator_outputs, current_time=arrival_time, combined_iteration=combined_iteration, process_state=process_states, trigger=True)
                    self.arrivals.pop(0)
            
            # Update process states for all incomplete processes
            for j in range(3): # update 3 times within the inter_arrival_time window
                sub_current_time = current_time + j * (inter_arrival_time / 3)
                combined_iteration = f'{self.iteration_id}_1_{j}'
                process_states = self.process_manager_db.get_all_incomplete_process_state(sub_current_time)
                if process_states == "":
                    print("No incomplete processes to update.")
                    continue
                coordinator_outputs = self.agent_manager.coordinator.hand_off(process_states=process_states, org_agents=self.agent_manager.org_agents, external_agents=self.agent_manager.ext_agents, current_timestamp=sub_current_time)
                if not coordinator_outputs or not coordinator_outputs.coordinator_outputs:
                    print("Coordinator returned no outputs during update step.")
                    self.log_protocol(timestamp=sub_current_time, from_agent='coordinator', to_agent='', process_id=None, message='', process_state='', combined_iteration=combined_iteration)
                    self.arrivals.pop(0)
                    continue
                self._handle_coordinator_outputs(coordinator_outputs, current_time=sub_current_time, combined_iteration=combined_iteration, process_state=process_states, trigger=False)
                self.arrivals.pop(0)

                # persist latest timestamp at the end of sub-iteration
                try:
                    if hasattr(self.process_manager_db.store, 'set_latest_timestamp'):
                        self.process_manager_db.store.set_latest_timestamp(sub_current_time.isoformat())
                except Exception:
                    pass

            
            # persist latest timestamp at the end of main iteration
            current_time += inter_arrival_time

            try:
                if hasattr(self.process_manager_db.store, 'set_latest_timestamp'):
                    self.process_manager_db.store.set_latest_timestamp(current_time.isoformat())
            except Exception:
                pass

            self.iteration_id += 1

