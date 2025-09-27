from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from .agent import Agent
from .utils import generate_random_process_id


from .tinydb_store import TinyDBStore

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


class ProcessManagerDB:
    """Process manager with a simple human-readable JSON DB backing.

    This class mirrors the behaviour of the original ProcessManager but keeps a
    JSON file (alongside the CSV log) which stores agents, coordinators,
    processes and events in a human-readable format.
    """
    def __init__(self, log_path, db_path):
        # in-memory structures removed — use the persistent TinyDB store instead
        self.num_complete_processes = 0
        self.log_path = log_path
        # open for writing and create the log file if it does not exist
        if not os.path.exists(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write("")

        # TinyDBStore stored alongside the csv log by default (JSON file)
        self.store = TinyDBStore(db_path)

        # initialize completed traces counter from DB
        try:
            self.num_complete_processes = 0
        except Exception:
            self.num_complete_processes = 0

    def record_protocol(self, iteration_id: str, timestamp: str, from_agent: str, to_agent: str, process_id: Optional[str], message: str, process_state: str = ""):
        """Record a handoff/protocol entry into the underlying DB store.

        This mirrors the CSV protocol file but stores it in the TinyDB `handoff_protocols` table.
        """
        try:
            handoff = {
                'iteration_id': iteration_id,
                'timestamp': timestamp if isinstance(timestamp, str) else str(timestamp),
                'from_agent': from_agent,
                'to_agent': to_agent,
                'process_id': process_id,
                'message': message,
                'process_state': process_state,
            }
            self.store.insert_handoff(handoff)
        except Exception:
            # don't fail the simulation if DB write fails
            pass


    def write_to_log_file(self, log_event: logEvent):
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
                # add a blank line after the header to separate header from data
                f.write('\n')
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
                "end_action": log_event.event.end_action
            })
            # add a blank line after each log entry to separate entries visually
            f.write('\n')


    def record_event(self, log_event: logEvent):
        """Record an event into the in-memory DB and persist it.

        The stored event is a JSON-friendly dict with key metadata and stringified data fields.
        """
        def model_to_dict(m):
            try:
                return m.model_dump() if hasattr(m, "model_dump") else m
            except Exception:
                return str(m)

        evt = {
            "process_id": log_event.process_id,
            "action": getattr(log_event.event, "action", None),
            "agent_id": getattr(log_event.agent, "id", None),
            "start_timestamp": getattr(log_event.event, "start_timestamp", None),
            "end_timestamp": getattr(log_event.event, "end_timestamp", None),
            "event_data": model_to_dict(getattr(log_event.event, "event_data", None)),
            "case_data": model_to_dict(getattr(log_event.event, "case_data", None)),
            "message": getattr(log_event.event, "message", None),
            "end_action": getattr(log_event.event, "end_action", None)
        }

        # insert via TinyDBStore which also merges case data into processes
        try:
            self.store.insert_event(evt)
            self.store.insert_process_data(log_event.process_id, {"data": model_to_dict(log_event.event.case_data) if log_event.event.case_data else {}})
        except Exception:
            # fallback: ignore DB errors to avoid breaking simulation
            pass

    def update_process(self, log_event: logEvent, last_activities: List[str]):
        """Add a log event and update the process instance."""
        # If there's no process_id we don't manage a process instance here
        if log_event.process_id is None:
            return

        # write CSV log
        self.write_to_log_file(log_event)

        # persist the event to the DB/store (TinyDB) which also merges case data into processes
        try:
            self.record_event(log_event)
        except Exception:
            # best-effort: do not interrupt the simulation on DB write failure
            pass

        # if this event ends the process (either explicit end_action or an action in last_activities), mark the process completed in the store
        process_id = log_event.process_id
        action = getattr(log_event.event, 'action', None)
        end_action_flag = getattr(log_event.event, 'end_action', False)
        if (action in last_activities) or end_action_flag:
            try:
                # use TinyDBStore helper to mark completion
                self.store.mark_process_completed(process_id)
            except Exception:
                pass

        # recompute completed traces count from the DB
        try:
            self.num_complete_processes = self.store.get_num_completed_processes()
            if self.num_complete_processes > 0:
                print(f'Num traces complete: {self.num_complete_processes}')
        except Exception:
            pass

    # generating process id if trigger event happens

    def generate_process_id(self):
        """Generate a unique process ID."""
        # derive used process ids from the persistent store
        try:
            procs = self.store.processes.all()
            used_process_ids = [p.get('process_id') for p in procs if p.get('process_id')]
        except Exception:
            used_process_ids = []
        process_id = generate_random_process_id(used_ids=used_process_ids)
        return process_id
    
    def get_related_process_state(self, process_id) -> str:
        """Get the related process instance."""
        # return empty string if process_id is None
        if not process_id:
            return ""
        # query DB for process and its events
        proc = self.store.get_process(process_id)
        events = self.store.get_events_for_process(process_id)
        if not events:
            return f'No process events found with ID: {process_id}'

        actions_with_timestamps = []
        for e in events:
            action = e.get('action') or e.get('message') or 'unknown'
            ts = e.get('end_timestamp') or ''
            actions_with_timestamps.append(f"{action} at {ts}")

        actions_str = ', '.join(actions_with_timestamps) if actions_with_timestamps else 'No actions taken so far'
        proc_data = proc.get('data', {}) if proc else {}
        process_state = f'Process ID: {process_id}, Last Actions: {actions_str}, Process Data: {proc_data}'
        return process_state

    def get_all_incomplete_process_state(self, current_time) -> str:
        """Get the current state of incomplete processes.
        Get only the processes that their latest event > current_time
        """
        process_state_lines = []

        # iterate all processes stored in DB
        # TinyDB does not have a direct table iterator method that returns primary key,
        # so read all processes
        try:
            all_procs = self.store.get_incomplete_processes()
        except Exception:
            all_procs = []

        for proc in all_procs:
            pid = proc.get('process_id')
            events = self.store.get_events_for_process(pid)
            if not events:
                continue
            last_event = events[-1]
            last_timestamp = last_event.get('end_timestamp')
            try:
                if last_timestamp and datetime.fromisoformat(last_timestamp) <= current_time:
                    actions_with_timestamps = [f"{e.get('action') or e.get('message') or 'unknown'} at {e.get('end_timestamp', '')}" for e in events]
                    actions_str = ', '.join(actions_with_timestamps) if actions_with_timestamps else 'No actions taken so far'
                    process_state = f"Process ID: {pid}, Last Actions: {actions_str}, Process Data: {proc.get('data', {})}"
                    process_state_lines.append(process_state)
            except Exception:
                # skip processes with bad timestamps
                continue

        return '\n'.join(process_state_lines)
