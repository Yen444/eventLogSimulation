from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from .agent import Agent
from .utils import generate_process_id


import csv
import os
import json

class logEvent(BaseModel):
    """Log event for the simulation"""
    process_id: str = ""
    agent: Agent
    event: Any
    process_state: str = ""
    iteration_id: str = ""

class ProcessInstance(BaseModel):
    """Process instance for the simulation"""
    process_id: str = Field(description="Unique identifier for the process instance")
    events: List[logEvent] = Field(default_factory=list, description="List of events in the process instance")
    data: dict = Field(default_factory=dict, description="Data related to the process instance")

class ProcessManager:
    def __init__(self, log_path):
        self.incomplete_processes: List[ProcessInstance] = []
        self.num_complete_processes = 0
        self.log_events: List[logEvent] = []
        self.log_path = log_path

        # open for writing and create the log file if it does not exist

        if not os.path.exists(log_path):
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write("")
    

    def _write_to_log_file(self, log_event: logEvent):
        """Write log event to the log file."""
        # open the log file in append mode and write a header if empty
        def data2string(data):
            if data:
                return ', '.join([f'{k}: {v}' for k, v in data.model_dump().items()])
            else:
                return ""
        # ensure header includes iteration_id; write rows appended
        # use newline='' to avoid extra blank lines on Windows
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            # store combined iteration id as iteration_sub (e.g. 0_0, 0_1)
            fieldnames = ["iteration_id", "process_id", "action", "agent_id", "start_timestamp", "end_timestamp", "agent_role", "agent_data", "case_data", "event_data", "message", "end_action", "process_state"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # if the file is empty, write header first
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "iteration_id": log_event.iteration_id,
                "process_id": log_event.process_id,
                "action": log_event.event.action,
                "agent_id": log_event.agent.id,
                "start_timestamp": log_event.event.start_timestamp,
                "end_timestamp": log_event.event.end_timestamp,
                "agent_role": log_event.agent.role,
                "agent_data": log_event.agent.data,
                "event_data": data2string(log_event.event.event_data),
                "case_data": data2string(log_event.event.case_data),
                "message": log_event.event.message,
                "end_action": log_event.event.end_action,
                "process_state": log_event.process_state
            })
    def _update_process_data(self, log_event:logEvent, process_instance: ProcessInstance):
        """Update the process data for a given process ID."""
        case_data = log_event.event.case_data.model_dump() if log_event.event.case_data else {}
        for key, value in case_data.items():
            if key not in process_instance.data:
                process_instance.data[key] = value
        return process_instance

    # generating process id if trigger event happens

    def generate_process_id(self):
        """Generate a unique process ID."""
        used_process_ids = list({log.process_id for log in self.log_events})
        process_id = generate_process_id(used_ids=used_process_ids)
        return process_id
    
    def get_related_process(self, process_id) -> str:
        """Get the related process instance."""
        # return empty string if process_id is None
        if not process_id:
            return ""
        for process_instance in self.incomplete_processes:
            if process_instance.process_id == process_id:
                actions_with_timestamps = [
                    f'{event.event.action} at {event.event.end_timestamp}'
                    for event in process_instance.events
                ]
                actions_str = ', '.join(actions_with_timestamps) if actions_with_timestamps else 'No actions taken so far'
                process_state = f'Process ID: {process_instance.process_id}, Last Actions: {actions_str}, Process Data: {process_instance.data}'
                return process_state
        return f'No process instance found with ID: {process_id}'

    def get_incomplete_process_state(self, current_time) ->str:
        """Get the current state of incomplete processes.
        Get only the processes that their latest event > current_time
        """

        process_state_lines = []

        for process_instance in self.incomplete_processes:
            if not process_instance.events:
                continue

            last_event = process_instance.events[-1]
            last_timestamp = last_event.event.end_timestamp

            if datetime.fromisoformat(last_timestamp) <= current_time:
                actions_with_timestamps = [
                    f'{event.event.action} at {event.event.end_timestamp}'
                    for event in process_instance.events
                ]

                actions_str = ', '.join(actions_with_timestamps) if actions_with_timestamps else 'No actions taken so far'

                process_state = (
                    f'Process ID: {process_instance.process_id}, '
                    f'Last Actions: {actions_str}, '
                    f'Process Data: {process_instance.data}'
                )
                process_state_lines.append(process_state)

        return '\n'.join(process_state_lines)
    
    def update_process(self, log_event: logEvent, last_activities: List[str]):
        """Add a log event and update the process instance."""
        if log_event.process_id is None:
            return
        # add log event to all log events
        self.log_events.append(log_event)
        # write log event to the log file
        self._write_to_log_file(log_event)


        # update process instances
        if log_event.event.action in last_activities or log_event.event.end_action:
            for process_instance in list(self.incomplete_processes):
                if process_instance.process_id == log_event.process_id:
                    self.incomplete_processes.remove(process_instance)
                    self.num_complete_processes += 1
                    print(f'Num traces complete: {self.num_complete_processes}')
                    break

        else:
            # add the process instance to the incomplete processes
            process_id = log_event.process_id
            # check if the process instance is already in the incomplete processes
            if log_event.process_id not in [process_instance.process_id for process_instance in self.incomplete_processes]:
                process_instance = ProcessInstance(process_id=process_id)
                process_instance = self._update_process_data(log_event, process_instance)
                # add the log event to the process instance
                process_instance.events.append(log_event)
                self.incomplete_processes.append(process_instance)
            else:
                # if the process instance is already in the incomplete processes, just add the log event to it
                # and update the process instance
                for process_instance in self.incomplete_processes:
                    if process_instance.process_id == process_id:
                        process_instance = self._update_process_data(log_event, process_instance)
                        # add the log event to the process instance
                        process_instance.events.append(log_event)
                        break