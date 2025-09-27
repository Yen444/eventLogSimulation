import os
import sqlite3
import json
from typing import Dict, Any, Optional


class SQLiteStore:
    """Small wrapper around sqlite3 for the simulation store.

    Responsibilities:
    - Open/close connection
    - Ensure schema exists
    - Insert/update agents, coordinators, processes and events
    - Provide simple query helpers
    """
    def __init__(self, db_path: str, wal: bool = True, config_manager: Optional[object] = None):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        if wal:
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
            except Exception:
                pass
        # store configs if provided (expects object with .configs dict)
        self.configs = getattr(config_manager, 'configs', {}) if config_manager is not None else {}
        self._ensure_schema()

    def _sanitize_col(self, name: str) -> str:
        # allow only alnum and underscore
        import re
        s = re.sub(r'[^0-9a-zA-Z_]', '_', name)
        if s[0].isdigit():
            s = '_' + s
        return s

    def get_table_columns(self, table: str):
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        rows = cur.fetchall()
        return [r[1] for r in rows]

    def _ensure_extra_columns(self):
        """Add dynamic columns to tables based on config keys.

        - agent_data -> add columns to agents table: agent_<key>
        - event_attributes -> add columns to events table: event_<key>
        - case_attributes -> add columns to events table: case_<key>
        """
        cur = self.conn.cursor()
        # agents extra
        agent_data = self.configs.get('agent_data', {}) if isinstance(self.configs, dict) else {}
        for key in agent_data.keys():
            col = self._sanitize_col(f'agent_{key}')
            existing = self.get_table_columns('agents')
            if col not in existing:
                cur.execute(f"ALTER TABLE agents ADD COLUMN {col} TEXT")

        # event and case attributes
        event_attrs = self.configs.get('event_attributes', {}) if isinstance(self.configs, dict) else {}
        case_attrs = self.configs.get('case_attributes', {}) if isinstance(self.configs, dict) else {}
        existing_events = self.get_table_columns('events')
        for key in event_attrs.keys():
            col = self._sanitize_col(f'event_{key}')
            if col not in existing_events:
                cur.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
        for key in case_attrs.keys():
            col = self._sanitize_col(f'case_{key}')
            if col not in existing_events:
                cur.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
        self.conn.commit()

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            role TEXT,
            data TEXT,
            actions TEXT,
            action_fewshots TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS coordinators (
            id TEXT PRIMARY KEY,
            role TEXT,
            data TEXT,
            actions TEXT,
            action_fewshots TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS processes (
            process_id TEXT PRIMARY KEY,
            data TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration_id TEXT,
            process_id TEXT,
            action TEXT,
            agent_id TEXT,
            start_timestamp TEXT,
            end_timestamp TEXT,
            agent_role TEXT,
            agent_data TEXT,
            event_data TEXT,
            case_data TEXT,
            message TEXT,
            end_action INTEGER,
            process_state TEXT,
            FOREIGN KEY(process_id) REFERENCES processes(process_id) ON DELETE SET NULL
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_process_id ON events(process_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_agent_id ON events(agent_id)")
        self.conn.commit()
        # add dynamic columns if config_manager provided
        try:
            self._ensure_extra_columns()
        except Exception:
            # ignore if something goes wrong with dynamic columns
            pass

    def register_agent(self, payload: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO agents (id, role, data, actions, action_fewshots) VALUES (?, ?, ?, ?, ?)",
            (
                payload.get('id'),
                payload.get('role'),
                json.dumps(payload.get('data', None), default=str) if payload.get('data', None) is not None else None,
                json.dumps(payload.get('actions', [])),
                json.dumps(payload.get('action_fewshots', [])),
            ),
        )
        self.conn.commit()

    def register_coordinator(self, payload: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO coordinators (id, role, data, actions, action_fewshots) VALUES (?, ?, ?, ?, ?)",
            (
                payload.get('id'),
                payload.get('role'),
                json.dumps(payload.get('data', None), default=str) if payload.get('data', None) is not None else None,
                json.dumps(payload.get('actions', [])),
                json.dumps(payload.get('action_fewshots', [])),
            ),
        )
        self.conn.commit()

    def insert_event(self, evt: Dict[str, Any]) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO events (iteration_id, process_id, action, agent_id, start_timestamp, end_timestamp, agent_role, agent_data, event_data, case_data, message, end_action, process_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                evt.get('iteration_id'), evt.get('process_id'), evt.get('action'), evt.get('agent_id'), evt.get('start_timestamp'), evt.get('end_timestamp'), evt.get('agent_role'),
                json.dumps(evt.get('agent_data'), default=str) if evt.get('agent_data') is not None else None,
                json.dumps(evt.get('event_data'), default=str) if evt.get('event_data') is not None else None,
                json.dumps(evt.get('case_data'), default=str) if evt.get('case_data') is not None else None,
                evt.get('message'), 1 if evt.get('end_action') else 0, evt.get('process_state')
            )
        )
        last_id = cur.lastrowid

        # ensure process exists and merge case_data into process data
        pid = evt.get('process_id')
        if pid:
            cur.execute("SELECT data FROM processes WHERE process_id = ?", (pid,))
            row = cur.fetchone()
            existing = {}
            if row and row[0]:
                try:
                    existing = json.loads(row[0])
                except Exception:
                    existing = {}
            case_data = evt.get('case_data') or {}
            if isinstance(case_data, dict):
                updated = False
                for k, v in case_data.items():
                    if k not in existing:
                        existing[k] = v
                        updated = True
                if row is None:
                    cur.execute("INSERT INTO processes (process_id, data) VALUES (?, ?)", (pid, json.dumps(existing, default=str)))
                elif updated:
                    cur.execute("UPDATE processes SET data = ? WHERE process_id = ?", (json.dumps(existing, default=str), pid))

        self.conn.commit()
        return last_id

    def count_agents(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM agents")
        return cur.fetchone()[0]

    def count_events(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        return cur.fetchone()[0]

    def fetch_one_event(self):
        cur = self.conn.cursor()
        cur.execute("SELECT iteration_id, process_id, action, agent_id, message FROM events LIMIT 1")
        return cur.fetchone()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
