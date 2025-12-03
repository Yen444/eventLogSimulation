import os
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from tinydb import TinyDB, where
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

import random
import datetime
import statistics
import json

class TinyDBStore:
    """Small wrapper around TinyDB for the simulation store.

    Responsibilities:
    - Open/close TinyDB
    - Provide simple helpers: register_agent, register_coordinator, insert_event
    - Keep tables: agents, processes, handoff_protocols, logging
    """

    def __init__(self, db_path):

        os.makedirs(db_path, exist_ok=True)
        db_file_path = os.path.join(db_path, "simulation_store.json")
        
        self.db = TinyDB(
            db_file_path,
            storage=CachingMiddleware(
                lambda p: JSONStorage(p, indent=2, ensure_ascii=False)
            )
        )
        self.org_agents = self.db.table('org_agents')
        self.ext_agents = self.db.table('ext_agents')
        self.coordinators = self.db.table('coordinators')
        self.processes = self.db.table('processes')
        self.events = self.db.table('events')
        self.handoffs = self.db.table('handoff_protocols')
        self.logging = self.db.table('llm_call_logs')
        # meta table for storing simulation metadata (latest timestamp, etc.)
        self.meta = self.db.table('meta')

    def register_org_agent(self, agent):
        agent = agent.model_dump()
        # remove non-serializable runtime objects
        if isinstance(agent, dict):
            agent.pop('llm', None)
            agent.pop('tool_manager', None)
            agent.pop('prompt_manager', None)
        agent = self._sanitize(agent)
        existing = self.org_agents.get(where('id') == agent.get('id'))
        if existing:
            self.org_agents.update(agent, where('id') == agent.get('id'))
        else:
            self.org_agents.insert(agent)
        self.flush()
    
    def register_ext_agent(self, agent):
        agent = agent.model_dump()
        # remove non-serializable runtime objects
        if isinstance(agent, dict):
            agent.pop('llm', None)
            agent.pop('tool_manager', None)
            agent.pop('prompt_manager', None)
        agent = self._sanitize(agent)
        existing = self.ext_agents.get(where('id') == agent.get('id'))
        if existing:
            self.ext_agents.update(agent, where('id') == agent.get('id'))
        else:
            self.ext_agents.insert(agent)
        self.flush()

    def register_coordinator(self, coordinator):
        # Use plain model_dump to get a dict and avoid Pydantic json-mode serialization
        # which can attempt to serialize class objects (ModelMetaclass) and fail.
        coordinator = coordinator.model_dump()
        # remove tool_manager and prompt_manager references
        if isinstance(coordinator, dict):
            coordinator.pop('tool_manager', None)
            coordinator.pop('prompt_manager', None)
            coordinator.pop('llm', None)
        coordinator = self._sanitize(coordinator)
        self.coordinators.insert(coordinator)
        self.flush()
        print(f"Registering coordinator with ID: {coordinator.get('id')}")
    
    def update_org_agent_availability(self, agent):
        """Update availability info (agent.available_from) for an existing agent
        """
        agent_id = agent.id
        available_from = agent.available_from

        existing = self.org_agents.get(where('id') == agent_id)
        if existing:
            self.org_agents.update({'available_from': available_from}, where('id') == agent_id)
            self.flush()

    def insert_event(self, evt: Dict[str, Any]) -> Optional[int]:
        """Insert event into logging table and merge case data into processes table.

        Returns the TinyDB doc_id for the inserted event.
        """
        evt_sanitized = self._sanitize(evt)
        doc_id = self.events.insert(evt_sanitized)
        self.flush()
        return doc_id
    
    def insert_process_data(self, process_id, process_data: Dict[str, Any]) -> Optional[int]:
        """Insert process data into the processes table."""
        try:
            doc_id = self.processes.insert(self._sanitize({'process_id': process_id, **process_data}))
            self.flush()
            return doc_id
        except Exception:
            return None

    def upsert_process(self, process_id: str, process_data: Dict[str, Any]) -> Optional[int]:
        """Insert or update a process record.

        If a process with `process_id` exists, update it with the provided
        `process_data`. Otherwise insert a new record. Returns the TinyDB
        doc_id (or None on failure).
        """
        try:
            existing = self.processes.get(where('process_id') == process_id)
            record = {'process_id': process_id, **process_data}
            if existing:
                # update only the provided keys
                self.processes.update(self._sanitize(process_data), where('process_id') == process_id)
                # return existing doc id if present
                return existing.doc_id if hasattr(existing, 'doc_id') else None
            else:
                doc_id = self.processes.insert(self._sanitize(record))
                self.flush()
                return doc_id
        except Exception:
            return None

    def count_agents(self) -> int:
        # return total number of registered agents (org + external)
        return len(self.org_agents) + len(self.ext_agents)
    
    def get_all_agent_ids(self) -> List[str]:
        agents = self.org_agents.all() + self.ext_agents.all()
        return [agent.get('id') for agent in agents if agent.get('id')]
    
    def get_all_process_ids(self) -> List[str]:
        processes = self.processes.all()
        return [proc.get('process_id') for proc in processes if proc.get('process_id')]
    
    def get_org_agents(self) -> List[Dict[str, Any]]:
        return self.org_agents.all()

    def get_ext_agents(self) -> List[Dict[str, Any]]:
        return self.ext_agents.all()

    def count_events(self) -> int:
        # count persisted events stored in the 'events' table
        return len(self.events)

    def fetch_one_event(self):
        # return the first persisted event if available
        all_events = self.events.all()
        return all_events[0] if all_events else None
    
    def get_num_completed_processes(self) -> int:
        """Return the number of completed processes."""
        return len(self.processes.search(where('completed') == True))

    def get_latest_timestamp(self) -> Optional[str]:
        """Return latest stored simulation timestamp as ISO string, or None."""
        try:
            rec = self.meta.get(where('key') == 'latest_timestamp')
            if not rec:
                return None
            return rec.get('value')
        except Exception:
            return None

    def set_latest_timestamp(self, ts_iso: str):
        """Set latest stored simulation timestamp (ISO string)."""
        try:
            existing = self.meta.get(where('key') == 'latest_timestamp')
            if existing:
                self.meta.update({'value': ts_iso}, where('key') == 'latest_timestamp')
            else:
                self.meta.insert({'key': 'latest_timestamp', 'value': ts_iso})
            self.flush()
        except Exception:
            pass
    
    def get_incomplete_processes(self) -> List[str]:
        """Return the list of process IDs that are not completed."""
        incomplete = self.processes.search(
            (where('completed') != True) | (~where('completed').exists())
        )
        return incomplete
    
    def get_process(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Return process record by process_id."""
        try:
            proc = self.processes.get(where('process_id') == process_id)
            return proc
        except Exception:
            return None
    
    def get_agent_by_id(self, agent_id: str):
        """Return agent record by agent_id."""
        try:
            agent = self.org_agents.get(where('id') == agent_id)
            if not agent:
                agent = self.ext_agents.get(where('id') == agent_id)
            if not agent:
                return None
            from simulation_src.agent import Agent
            return Agent(**agent)
        except Exception:
            return None
    
    def get_agent_by_role(self, role: str):
        """Return a single random Agent with the given role from org_agents + ext_agents."""

        # Fetch all agents with the given role from both tables
        try:
            docs = self.org_agents.search(where('role') == role) + \
                self.ext_agents.search(where('role') == role)
            
            if not docs:
                return None            
            # Pick one at random and convert to Agent
            from simulation_src.agent import Agent
            return Agent(**random.choice(docs))
        except Exception:
            return None
        
    def get_events_for_process(self, process_id: str):
        """Return list of events for a process ordered by end_timestamp."""
        try:
            events = self.events.search(where('process_id') == process_id)
        except Exception:
            events = []
        # sort by end_timestamp if present (fall back to insertion order)
        def _key(e):
            ts = e.get('end_timestamp')
            return ts or ''
        try:
            events_sorted = sorted(events, key=_key)
        except Exception:
            events_sorted = events
        return events_sorted

    def insert_handoff(self, handoff: Dict[str, Any]) -> Optional[int]:
        """Insert a handoff/protocol entry into the handoff_protocols table.

        The handoff dict is expected to contain keys like:
        - iteration_id, timestamp, from_agent, to_agent, process_id, message, process_state
        Returns the TinyDB doc_id for the inserted entry.
        """
        try:
            doc_id = self.handoffs.insert(self._sanitize(handoff))
            self.flush()
            return doc_id
        except Exception:
            return None

    def mark_process_completed(self, pid: str):
        """Mark a process record as completed.

        Creates the process record if it does not exist.
        """
        try:
            existing = self.processes.get(where('process_id') == pid)
            if existing:
                self.processes.update({'completed': True}, where('process_id') == pid)
            else:
                # create minimal process record
                self.processes.insert({'process_id': pid, 'completed': True})
            self.flush()
        except Exception:
            pass

    def flush(self):
        self.db.storage.flush()

    def _sanitize(self, obj):
        """Recursively convert non-JSON-serializable types to serializable ones.

        - datetime -> ISO string
        - date -> ISO string
        - timedelta -> total seconds (float)
        - recursively handle dicts and lists
        """
        # Handle Pydantic BaseModel instances by dumping them to dict first
        try:
            if isinstance(obj, BaseModel):
                return self._sanitize(obj.model_dump())
        except Exception:
            # If pydantic isn't available or checking fails, continue
            pass

        # Primitive and None types
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj

        # Datetime / date / timedelta
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, datetime.date):
            return obj.isoformat()
        if isinstance(obj, datetime.timedelta):
            return obj.total_seconds()

        # Dict -> sanitize values
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}

        # List / tuple -> sanitize elements
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]

        # Fallback: convert to string
        try:
            return str(obj)
        except Exception:
            return None
    def close(self):
        self.flush()
        self.db.close()
    
    def _parse_stringified_dict(self, raw_data: Any) -> Optional[Dict[str, Any]]:
        """Helper to parse a stringified dict safely, supporting JSON, key=value, and key: value formats."""
        if not raw_data:
            return None

        if isinstance(raw_data, dict):
            return raw_data

        if isinstance(raw_data, str):
            # --- Case 1: Try JSON ---
            try:
                if '{' in raw_data and '}' in raw_data:
                    return json.loads(raw_data)
            except json.JSONDecodeError:
                pass  # fallback to other formats

            # --- Case 2: Try key=value style ---
            try:
                pairs = [item.split('=', 1) for item in raw_data.split()]
                if all(len(p) == 2 for p in pairs):
                    return {k: v.strip("'\"") for k, v in pairs}
            except Exception:
                pass

            # --- Case 3: Try key: value, comma-separated style ---
            try:
                segments = [seg.strip() for seg in raw_data.split(',')]
                pairs = [seg.split(':', 1) for seg in segments if ':' in seg]
                return {k.strip(): v.strip() for k, v in pairs}
            except Exception:
                return None

        return None

    def _get_top_frequencies(self, value_freq: Dict[str, int], max_items: int = 20) -> Dict[str, int]:
        """Helper to get the top N most frequent values from a frequency dictionary."""
        if len(value_freq) <= max_items:
            return value_freq
        
        # Sort by frequency (descending) and take top N
        sorted_items = sorted(value_freq.items(), key=lambda x: x[1], reverse=True)
        if max_items:
            return dict(sorted_items[:max_items])
        return dict(sorted_items)
    
    def _compute_attribute_stats(self, values: List[Any]) -> Dict[str, Any]:
        """Helper to compute statistics for a list of values."""
        if not values:
            return {}

        # Frequency distribution
        value_freq = {v: values.count(v) for v in set(values)}

        # Attempt numeric conversion
        try:
            num_values = [float(v) for v in values]
        except (ValueError, TypeError):
            num_values = []

        # If all values can be converted to numeric → compute descriptive stats
        if len(num_values) == len(values) and num_values:
            return {
                'avg': statistics.mean(num_values),
                'min': min(num_values),
                'max': max(num_values),
                'std': statistics.pstdev(num_values) if len(num_values) > 1 else 0.0,
                'total_values': len(num_values),
                'unique_values': len(value_freq),
                'top_value_frequencies': self._get_top_frequencies(value_freq)
            }
        else:
            return {
                'total_values': len(values),
                'unique_values': len(value_freq),
                'top_value_frequencies': self._get_top_frequencies(value_freq)
            }
    
    def _group_by_resource_attribute(self, events: List[Dict[str, Any]], 
                                      data_attribute: str, 
                                      resource_attribute: str,
                                      data_key: str = 'event_data') -> Dict[str, List[Any]]:
        """Generic helper to group data attribute values by resource attribute values.
        
        Args:
            events: List of event records
            data_attribute: The attribute to extract from event/case data
            resource_attribute: The attribute to extract from resource data
            data_key: The key in event dict containing the data ('event_data' or 'case_data')
            
        Returns:
            Dictionary mapping resource attribute values to lists of data attribute values
        """
        grouped = {}
        
        for e in events:
            # Extract resource attribute value
            res_id = e.get('agent_id')
            resource_data_value = None
            if res_id:
                agent = self.get_agent_by_id(res_id)
                if agent and agent.data:
                    data = self._parse_stringified_dict(agent.data)
                    if data:
                        resource_data_value = data.get(resource_attribute)
            # Extract data attribute value
            data_value = None
            raw_data = e.get(data_key)
            if raw_data:
                data = self._parse_stringified_dict(raw_data)
                if data:
                    data_value = data.get(data_attribute)

            # Skip if either is missing
            if resource_data_value is None or data_value is None:
                continue

            grouped.setdefault(resource_data_value, []).append(data_value)
        
        return grouped

    def get_event_data_stats(self, action: str=None, event_attributes: List[str]=None) -> Dict[str, Any]:
        """Get statistics about event data per one action or per all actions.
        
        The event data is stored in the attribute 'event_data' as a stringified dict.
        Example: '{"temperature": 20, "duration": 3.5}'
        """
        #return f'This is the stats: 1000'
        if not event_attributes:
            return {}
        try:
            events = self.events.all()
            if action:
                
                events = [e for e in events if e.get('action') == action]
            if not events:
                return {}   

            # Parse event_data (string of dict)
            data_entries = []
            for e in events:
                raw_data = e.get('event_data')
                if not raw_data:
                    continue
                data = self._parse_stringified_dict(raw_data)
                if data:
                    data_entries.append(data)

            if not data_entries:
                return {}
            # Compute stats for each attribute     
            stats = {}

            for attr in event_attributes:
                # Extract all values for this attribute
                values = [d.get(attr) for d in data_entries if attr in d and d.get(attr) is not None]
                stats[attr] = self._compute_attribute_stats(values)

            return stats

        except Exception:
            return {}

    def get_case_data_stats(self, case_attributes: List[str]=None) -> Dict[str, Any]:
        """Get statistics about case data for all processes.

        The case data is stored in the attribute 'data' as a stringified dict.
        Example: '{"amount": 5, "year": 3}'
        """
        if not case_attributes:
            return {}

        try:
            processes = self.processes.all()
            if not processes:
                return {}

            # Parse the case data (stringified dict)
            data_entries = []
            for p in processes:
                raw_data = p.get('data')
                if not raw_data:
                    continue
                data = self._parse_stringified_dict(raw_data)
                if data:
                    data_entries.append(data)

            if not data_entries:
                return {}

            stats = {}

            for attr in case_attributes:
                values = [d.get(attr) for d in data_entries if attr in d and d.get(attr) is not None]
                stats[attr] = self._compute_attribute_stats(values)

            return stats

        except Exception:
            return {}

    def get_event_data_stats_by_resource_group(self, event_attribute: str, resource_attribute: str, action: str=None) -> Dict[str, Any]:
        """Get statistics about event data per given action (or all actions) grouped by a resource attribute.

        Example: compute average event duration grouped by resource's experience level.
        """
        if not event_attribute or not resource_attribute:
            return {}

        try:
            events = self.events.all()
            if action:
                events = [e for e in events if e.get('action') == action]
            
            # Group event attribute values by resource attribute value
            grouped = self._group_by_resource_attribute(events, event_attribute, resource_attribute, 'event_data')
            if not grouped:
                return {}

            # Compute stats for each group
            return {res_value: self._compute_attribute_stats(values) 
                    for res_value, values in grouped.items()}

        except Exception:
            return {}
    def get_case_data_stats_by_resource_group(self, case_attribute: str, resource_attribute: str) -> Dict[str, Any]:
        """
        Compute statistics for a case data attribute grouped by a resource attribute.

        Note:
            Case data are given during certain activities. Their relationship to 
            resources is determined by the resource performing the corresponding action.

        Example:
            To get statistics of insurance_amount by resource gender: the insurance 
            amount is recorded when a customer submits a claim, and the resource is 
            the agent handling that action.
        """
        if not case_attribute or not resource_attribute:
            return {}

        try:
            events = self.events.all()
            
            # Group case attribute values by resource attribute value
            grouped = self._group_by_resource_attribute(events, case_attribute, resource_attribute, 'case_data')
            
            if not grouped:
                return {}

            # Compute stats for each group
            return {res_value: self._compute_attribute_stats(values) 
                    for res_value, values in grouped.items()}

        except Exception:
            return {}

    def get_resource_data_stats(self, resource_attributes: List[str] = None) -> Dict[str, Any]:
        """Compute statistics for resource data across organizational and external agents.
        
        The resource data is stored in the attribute 'data' as a stringified dict.
        Example: '{"experience": 5, "age": 30}'
        """
        if not resource_attributes:
            return {}

        agents_data = {
            'org_agents': [self._parse_stringified_dict(a.get('data')) for a in self.org_agents.all() if a.get('data')],
            'ext_agents': [self._parse_stringified_dict(a.get('data')) for a in self.ext_agents.all() if a.get('data')]
        }

        # Remove any None or invalid parses
        for key in agents_data:
            agents_data[key] = [d for d in agents_data[key] if isinstance(d, dict)]

        if not agents_data['org_agents'] and not agents_data['ext_agents']:
            return {}

        stats = {'org_agents': {}, 'ext_agents': {}}

        # Compute stats for each agent group and attribute
        for group_name, data_list in agents_data.items():
            for attr in resource_attributes:
                values = [d[attr] for d in data_list if attr in d and d[attr] is not None]
                stats[group_name][attr] = self._compute_attribute_stats(values)

        return stats
    
    def get_trace_stats(self):
        """Get trace variants and their frequencies."""
        processes = self.processes.all()
        events = self.events.all()
        trace_variants = {}
        for p in processes:
            p_events = [e for e in events if e.get('process_id') == p.get('process_id')]
            # Sort events by end_timestamp
            p_events_sorted = sorted(p_events, key=lambda e: e.get('end_timestamp') or '')
            trace = tuple(e.get('action') for e in p_events_sorted if e.get('action'))
            if trace:
                trace_variants[trace] = trace_variants.get(trace, 0) + 1
        trace_variants_percent = {
            trace: (count, count/ sum(trace_variants.values()) * 100)
            for trace, count in trace_variants.items()
        }
        return trace_variants_percent

    def get_resource_schedule_stats(self):
        """Get statistics on resource schedules across processes and events.
        For each resource (agent), compute:
        - hours worked per day of the week
        - Return a dict mapping agent IDs to their schedule stats {'agent_id': {'Monday': 8, 'Tuesday': 6, ...}}
        """
        pass
    def get_resource_allocation_stats(self):
        """Get statistics on resource allocation across processes and events.
        For each resource (agent), compute:
        - number of unique processes handled
        - total number of events handled
        - total time spent (sum of event durations)"""
        pass


def main():
    # Initialize the database
    pass
    #store = TinyDBStore(db_path='./db_1')
    
    # Call the function with the parameters from the user's example
    #result = store.get_event_data_stats_by_resource_group(
    #    event_attribute='task_duration',
    #    resource_attribute='gender', 
    #    #action='SRM: Created'
    #)
    #result = store.get_case_data_stats_by_resource_group(
    #    case_attribute='spend_area',
    #    resource_attribute='gender'
    #)
    
    #print(f"Result: {result}")
    #a = store.get_agent_by_id('11557')
    #print(f"Agent: {a}")

if __name__ == "__main__":
    main()

