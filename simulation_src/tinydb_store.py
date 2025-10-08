import os
from typing import Dict, Any, List, Optional
from .agent import Agent
from .coordinator_agent import Coordinator

from tinydb import TinyDB, where
from tinydb.storages import JSONStorage
from tinydb.middlewares import CachingMiddleware

import random
import datetime


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

    def register_org_agent(self, agent: Agent):
        agent = agent.model_dump()
        agent = self._sanitize(agent)
        existing = self.org_agents.get(where('id') == agent.get('id'))
        if existing:
            self.org_agents.update(agent, where('id') == agent.get('id'))
        else:
            self.org_agents.insert(agent)
        self.flush()
    
    def register_ext_agent(self, agent: Agent):
        agent = agent.model_dump()
        agent = self._sanitize(agent)
        existing = self.ext_agents.get(where('id') == agent.get('id'))
        if existing:
            self.ext_agents.update(agent, where('id') == agent.get('id'))
        else:
            self.ext_agents.insert(agent)
        self.flush()

    def register_coordinator(self, coordinator: Coordinator):
        coordinator = coordinator.model_dump(mode='json')
        coordinator = self._sanitize(coordinator)
        self.coordinators.insert(coordinator)
        self.flush()
        print(f"Registering coordinator with ID: {coordinator.get('id')}")

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
    
    def get_agent_by_id(self, agent_id: str) -> Agent:
        """Return agent record by agent_id."""
        try:
            agent = self.org_agents.get(where('id') == agent_id)
            if not agent:
                agent = self.ext_agents.get(where('id') == agent_id)
            return Agent(**agent) if agent else None
        except Exception:
            return None
    def get_agent_by_role(self, role: str) -> Optional[Agent]:
        """Return a single random Agent with the given role from org_agents + ext_agents."""
        try:
            # Fetch all agents with the given role from both tables
            docs = self.org_agents.search(where('role') == role) + \
                self.ext_agents.search(where('role') == role)
            
            if not docs:
                return None
            
            # Pick one at random and convert to Agent
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
