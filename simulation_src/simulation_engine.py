from pydantic import BaseModel, Field, create_model
from langchain_core.prompts import PromptTemplate

from .agent import Agent
from .agent_manager import AgentManager
from .config_manager import ConfigManager
from .process_manager import ProcessManager, logEvent
from .prompt_manager import create_prompt_manager

from collections import OrderedDict
import datetime
import os

from dotenv import load_dotenv
import os
import random
import csv

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
    def __init__(self, llm, config_path, log_path, protocol_path, arrivals):
        self.config_manager = ConfigManager(config_path=config_path)
        # create a PromptManager and pass it to AgentManager
        prompt_manager = create_prompt_manager()
        self.agent_manager = AgentManager(self.config_manager, prompt_manager, llm)
        self.process_manager = ProcessManager(log_path)

        self.protocol_path = protocol_path
        self.llm = llm
        self.arrivals = arrivals # List of tuple of simulated arrival (timestamp, lamda)
        # iteration counter used to track steps/hand-offs in the simulation
        self.iteration_id = 0

        # ensure protocol file exists and has a header
        if not os.path.exists(self.protocol_path):
            os.makedirs(os.path.dirname(self.protocol_path), exist_ok=True)
            with open(self.protocol_path, 'w', encoding='utf-8') as f:
                f.write("")
        # if protocol file is empty, write header
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
        # Log to CSV
        # open with newline='' so csv.writer handles newlines correctly
        with open(self.protocol_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            combined = combined_iteration if combined_iteration is not None else f'{self.iteration_id}'
            writer.writerow([combined, timestamp, from_agent, to_agent, process_id, message, process_state])
            # add a blank line after each protocol entry to separate entries visually
            f.write('\n')

    def step(self, agent: Agent, process_id, current_time, update_interval, message, trigger=False, reflect=True, combined_iteration=None):
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
                multi_output = True if len(out.agent_outputs) > 1 else False
                if trigger:
                    process_id = self.process_manager.generate_process_id()
                    for idx, agent_output in enumerate(out.agent_outputs):
                        # Always create a log_event. If the agent actually performed an action, update process state;
                        # otherwise write an empty-output row to the log file so we can track "no action" decisions.
                        safe_combined = combined_iteration if combined_iteration is not None else f'{self.iteration_id}'
                        log_event = logEvent(
                            process_id=process_id,
                            agent=agent,
                            event=agent_output,
                            process_state=process_state,
                            iteration_id=f'{safe_combined}_{idx}' if multi_output else safe_combined
                        )
                        if agent_output.action is not None:
                            self.process_manager.update_process(log_event=log_event, last_activities=self.config_manager.configs.get('last_activities', []))
                        else:
                            # write an empty-output row without modifying process instances
                            self.process_manager._write_to_log_file(log_event)
                else:
                    for idx, agent_output in enumerate(out.agent_outputs):
                        # Always create a log_event. Write to process manager if action present; otherwise log empty output row.
                        safe_combined = combined_iteration if combined_iteration is not None else f'{self.iteration_id}'
                        log_event = logEvent(
                            process_id=agent_output.process_id,
                            agent=agent,
                            event=agent_output,
                            process_state=process_state,
                            iteration_id=f'{safe_combined}_{idx}' if multi_output else safe_combined
                        )
                        if agent_output.action is not None:
                            self.process_manager.update_process(log_event=log_event, last_activities=self.config_manager.configs.get('last_activities', []))
                        else:
                            # write an empty-output row without modifying process instances
                            self.process_manager._write_to_log_file(log_event)
            
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
            # End when there are no more scheduled arrivals
            if len(self.arrivals) == 0:
                print('No more arrivals, ending simulation')
                break
            # End when the current time exceeds the end time
            if end_time is not None and current_time >= end_time:
                break
            # End when num traces achieved
            if num_traces is not None and self.process_manager.num_complete_processes >= num_traces:
                break

            print(f"Current time: {current_time}")
            # coordinator calls on agent to initialize the process

            # Check if any arrivals within the time period current_time until the next timestamp
            next_current_time = current_time + inter_arrival_time
            sub_arrivals = [arrival for arrival in self.arrivals if current_time <= arrival[0] < next_current_time]

            print(f'Number of arrivals in this updating step: {len(sub_arrivals)}')


            if len(sub_arrivals) > 0:
                sub_iteration = 0
                for i, (arrival_time,_) in enumerate(sub_arrivals):
                    sub_sub_iteration = i
                    # prepare combined iteration id for this arrival
                    combined_iteration = f'{self.iteration_id}_{sub_iteration}_{sub_sub_iteration}'

                    coordinator_outputs = self.agent_manager.coordinator.hand_off(process_states='', org_agents=self.agent_manager.org_agents, external_agents=self.agent_manager.ext_agents, current_timestamp=current_time)

                    # If coordinator returned nothing for this arrival, still log an empty trigger handoff
                    if coordinator_outputs is None or len(coordinator_outputs.coordinator_outputs) == 0:
                        # write a protocol entry to indicate a trigger handoff was attempted but no coordinator outputs were produced
                        self.log_protocol(
                            timestamp=current_time,
                            from_agent='coordinator',
                            to_agent='',
                            process_id=None,
                            message='',
                            process_state=self.process_manager.get_related_process(process_id=None),
                            combined_iteration=combined_iteration
                        )
                        # consume the earliest arrival for this iteration
                        if self.arrivals:
                            self.arrivals.pop(0)
                        continue

                    for j, coordinator_output in enumerate(coordinator_outputs.coordinator_outputs):
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

                        # pass sub-iteration (index within this arrival batch)
                        self.step(agent=agent, process_id=process_id, current_time=arrival_time, update_interval=inter_arrival_time, message=message, trigger=True, reflect=False, combined_iteration=combined_iteration)

                        self.log_protocol(
                            timestamp=current_time,
                            from_agent='coordinator',
                            to_agent=agent_role if agent_id is None else agent.id,
                            process_id=process_id,
                            message=message,
                            process_state=self.process_manager.get_related_process(process_id=process_id),
                            combined_iteration=combined_iteration
                        )
                    # consume the earliest arrival once per arrival (not per coordinator output)
                    if self.arrivals:
                        self.arrivals.pop(0)
                    
            update_current_time = current_time
            
            for i in range(2):
                sub_iteration = i+1
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
                for j, coordinator_output in enumerate(coordinator_outputs.coordinator_outputs):
                    sub_sub_iteration = j
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
                    # pass current sub_iteration index so logs and events can be correlated
                    combined_iteration = f'{self.iteration_id}_{sub_iteration}_{sub_sub_iteration}'
                    self.step(agent=agent, process_id=process_id, current_time=update_current_time, update_interval=inter_arrival_time/2, message=message, trigger=False, reflect=False, combined_iteration=combined_iteration)
                    self.log_protocol(
                        timestamp=update_current_time,
                        from_agent='coordinator',
                        to_agent=agent_role if agent_id is None else agent.id,
                        process_id=process_id,
                        message=message,
                        process_state=process_states,
                        combined_iteration=combined_iteration
                    )

                    # increment iteration id after each coordinator handoff

                update_current_time += (inter_arrival_time/2)
            
            current_time += inter_arrival_time
            self.iteration_id += 1

            