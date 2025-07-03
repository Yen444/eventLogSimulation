from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Type

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

class ProcessInstance(BaseModel):
    """Process instance for the simulation"""
    process_id: str = Field(description="Unique identifier for the process instance")
    events: List[logEvent] = Field(default_factory=list, description="List of events in the process instance")
    data: dict = Field(default_factory=dict, description="Data related to the process instance")

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
        def data2string(data):
            if data:
                return ', '.join([f'{k}: {v}' for k, v in data.model_dump().items()])
            else:
                return ""
        with open(self.log_path, 'a', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["process_id", "action", "timestamp", "agent_role", "agent_data", "case_data", "event_data"])
            #writer.writeheader()
            writer.writerow({
                "process_id": log_event.process_id,
                "action": log_event.event.action,
                "timestamp": log_event.event.timestamp,
                "agent_role": log_event.agent.role,
                "agent_data": log_event.agent.data,
                "event_data": data2string(log_event.event.event_data),
                "case_data": data2string(log_event.event.case_data),
                "message": log_event.event.message
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
                first_event = process_instance.events[0] if process_instance.events else None
                last_event = process_instance.events[-1] if process_instance.events else None
                process_state = f'Process ID: {process_instance.process_id}, First Action:: {first_event.event.action if first_event else 'No action is taken so far'} finished at {first_event.event.timestamp if first_event else ''}, Last Action: {last_event.event.action if last_event else 'No action is taken so far'} finished at {last_event.event.timestamp if last_event else ''}, Process Data: {process_instance.data}\n'
                return process_state
        return f'No process instance found with ID: {process_id}'

    def get_incomplete_process_state(self) ->str:
        """Get the current state of incomplete processes."""
        process_states = ""
        for process_instance in self.incomplete_processes:
            first_event = process_instance.events[0] if process_instance.events else None
            last_event = process_instance.events[-1] if process_instance.events else None
            process_state = f'Process ID: {process_instance.process_id}, First Action:: {first_event.event.action if first_event else 'No action is taken so far'} finished at {first_event.event.timestamp if first_event else ''}, Last Action: {last_event.event.action if last_event else 'No action is taken so far'} finished at {last_event.event.timestamp if last_event else ''}, Process Data: {process_instance.data}\n'
            process_states += process_state
        return process_states
    
    def update_process(self, log_event: logEvent, last_activities: List[str]):
        """Add a log event and update the process instance."""
        # add log event to all log events
        self.log_events.append(log_event)
        # write log event to the log file
        self._write_to_log_file(log_event)


        # update process instances
        if log_event.event.action in last_activities:
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