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
    tool_messages: str = ""  # Tool messages used during action decision


class ProcessManagerDB:
    """Process manager with a simple human-readable JSON DB backing.

    This class mirrors the behaviour of the original ProcessManager but keeps a
    JSON file (alongside the CSV log) which stores agents, coordinators,
    processes and events in a human-readable format.
    """
    def __init__(self, log_path, store: TinyDBStore = None):
        """Initialize ProcessManagerDB.

        If a `store` (TinyDBStore) is provided, it will be used. Otherwise a new
        TinyDBStore will be created from `db_path`.
        """
        self.log_path = log_path
        self.store = store

        # Ensure log file exists
        if not os.path.exists(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write("")

    def record_protocol(self, iteration_id: str, timestamp: str, from_agent: str, to_agent: str, process_id: Optional[str], message: str, process_state: str = ""):
        """Record a handoff/protocol entry into the underlying DB store.

        This mirrors the CSV protocol file but stores it in the TinyDB `handoff_protocols` table.
        """
        handoff = {
            'iteration_id': iteration_id,
            'timestamp': timestamp,
            'from_agent': from_agent,
            'to_agent': to_agent,
            'process_id': process_id,
            'message': message,
            'process_state': process_state,
        }
        self.store.insert_handoff(handoff)

    def write_to_log_file(self, log_event: logEvent):
        """Write log event to the log file."""
        # open the log file in append mode and write a header if empty
        def data2string(data):
            return ', '.join([f'{k}: {v}' for k, v in data.model_dump().items()]) if data else ""

        # ensure header includes iteration_id; write rows appended
        # use newline='' to avoid extra blank lines on Windows
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            # store combined iteration id as iteration_sub (e.g. 0_0, 0_1)
            fieldnames = ["iteration_id", "process_id", "action", "agent_id", "start_timestamp", "end_timestamp", "agent_role", "agent_data", "case_data", "event_data", "message", "reasoning", "end_action", "process_state", "tool_messages"]
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
                "reasoning": getattr(log_event.event, 'reasoning', ''),
                "end_action": log_event.event.end_action,
                "process_state": log_event.process_state,
                "tool_messages": log_event.tool_messages
            })

    def record_event(self, log_event: logEvent):
        """Record an event into the in-memory DB and persist it.
        """
        def safe_str(value):
            """Convert any value to a safe string representation."""
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float, bool)):
                return str(value)
            if isinstance(value, datetime):
                return value.isoformat()  # standardized, reversible
            if isinstance(value, (list, dict)):
                try:
                    # Try JSON serialization for dicts, lists, etc.
                    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                except (TypeError, ValueError):
                    # Fallback for unknown objects
                    return str(value)
            return str(value)
        def safe_get(obj, attr):
            return getattr(obj, attr, None) if obj else None
        evt = {
            "process_id": safe_str(safe_get(log_event, "process_id")),
            "action": safe_str(safe_get(log_event.event, "action")),
            "agent_id": safe_str(safe_get(log_event.agent, "id")),
            "start_timestamp": safe_str(safe_get(log_event.event, "start_timestamp")),
            "end_timestamp": safe_str(safe_get(log_event.event, "end_timestamp")),
            "event_data": safe_str(safe_get(log_event.event, "event_data")),
            "case_data": safe_str(safe_get(log_event.event, "case_data")),
            "message": safe_str(safe_get(log_event.event, "message")),
            "end_action": safe_get(log_event.event, "end_action")
        }
        self.store.insert_event(evt)

    def update_process(self, log_event: logEvent, last_activities: List[str]):
        """Add a log event and update the process instance."""
        # If there's no process_id (None or empty) we don't manage a process instance here
        if not getattr(log_event, 'process_id', None):
            return

        # write CSV log
        self.write_to_log_file(log_event)

        # persist the event to the DB/store (TinyDB)
        self.record_event(log_event)

        # Ensure a process record exists (or is updated) in the store so that
        # incomplete processes are discoverable. Merge case data into process.data
        try:
            pid = log_event.process_id

            def _extract_case_data(case_data):
                if case_data is None:
                    return {}
                if hasattr(case_data, 'model_dump'):
                    try:
                        return case_data.model_dump()
                    except Exception:
                        pass
                if isinstance(case_data, dict):
                    return case_data
                if isinstance(case_data, str):
                    try:
                        return json.loads(case_data)
                    except Exception:
                        pass
                return {}

            case_data = _extract_case_data(getattr(log_event.event, 'case_data', None))

            existing_proc = None
            try:
                existing_proc = self.store.get_process(pid)
            except Exception:
                existing_proc = None

            existing_data = existing_proc.get('data', {}) if existing_proc else {}
            merged_data = {**existing_data, **case_data} if isinstance(existing_data, dict) else case_data

            last_ts = getattr(log_event.event, 'end_timestamp', None) or getattr(log_event.event, 'start_timestamp', None)

            proc_record = {
                'data': merged_data,
                'last_event_timestamp': last_ts,
                # completed flag will be set separately when end action occurs
                'completed': False
            }

            if hasattr(self.store, 'upsert_process'):
                self.store.upsert_process(pid, proc_record)
            else:
                # fallback: only insert if not present
                if not existing_proc:
                    self.store.insert_process_data(pid, proc_record)
        except Exception:
            # don't let DB issues interrupt the simulation
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
        all_procs = self.store.get_incomplete_processes()
        for proc in all_procs:
            pid = proc.get('process_id')
            events = self.store.get_events_for_process(pid)
            
            last_event = events[-1]
            last_timestamp = last_event.get('end_timestamp')
            
            # Clean timestamp string by removing markdown code block formatting
            # TODO: more robust cleaning if needed
            if last_timestamp:
                last_timestamp = last_timestamp.strip('`')
            
            #print(last_timestamp, current_time)
        
            if last_timestamp and datetime.fromisoformat(last_timestamp) <= current_time:
                actions_with_timestamps = [f"{e.get('action') or e.get('message') or 'unknown'} at {e.get('end_timestamp', '')}" for e in events]
                actions_str = ', '.join(actions_with_timestamps) if actions_with_timestamps else 'No actions taken so far'
                process_state = f"Process ID: {pid}, Last Actions: {actions_str}, Process Data: {proc.get('data', {})}"
                process_state_lines.append(process_state)

        return '\n'.join(process_state_lines)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processes and events in the store."""
        # Get statistics basing on all the case data and events data in terms of processes and events
        # What kind of statistics would be useful and general given arbitrary process data?
        # What we may care about is time, resource, event data (arbitrary), case data (arbitrary) in relation to process and activities.
        # These statistics will be given to llm agent so that it can adapt their output to meet up some biases or criteria, for example, female resources take more time to finish activity rather than male, which statistics can help to define it?

        pass
