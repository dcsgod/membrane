"""
SQLite substrate — default storage backend for all memory kinds (§16, A6.2).

Uses stdlib sqlite3 only — no external dependencies.
Serves as: structured, KV, graph edges, temporal events, and FTS5 lexical.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterator, Optional

from membrane.core.models import (
    MemoryEvent,
    MemoryRecord,
    MemoryState,
    MemoryType,
    Namespace,
    RecallResult,
    Snapshot,
    Source,
    StructuredFields,
    TrustTier,
    Query,
)


def _dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _dts(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SQLiteSubstrate:
    """
    SQLite-backed memory substrate.

    Provides: structured storage, event log, FTS5 lexical search,
    graph edges (provenance), snapshots, and KV-style lookups.

    Thread-safe via a threading.Lock for write serialization.
    Read connections are per-thread with check_same_thread=False.
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Connection management
    # ------------------------------------------------------------------ #

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                detect_types=sqlite3.PARSE_DECLTYPES,
                isolation_level=None,  # autocommit; we manage transactions manually
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        conn = self._conn()
        with self._lock:
            conn.execute("BEGIN")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id              TEXT PRIMARY KEY,
                    content         TEXT NOT NULL,
                    type            TEXT NOT NULL DEFAULT 'semantic',
                    state           TEXT NOT NULL DEFAULT 'candidate',

                    confidence      REAL NOT NULL DEFAULT 1.0,
                    importance      REAL NOT NULL DEFAULT 0.5,
                    utility         REAL NOT NULL DEFAULT 0.5,
                    strength        REAL NOT NULL DEFAULT 1.0,

                    recorded_at     TEXT,
                    created_at      TEXT,
                    observed_at     TEXT,
                    valid_from      TEXT,
                    valid_until     TEXT,
                    updated_at      TEXT,
                    last_accessed_at TEXT,
                    expires_at      TEXT,

                    source_json     TEXT DEFAULT '{}',
                    trust_tier      TEXT NOT NULL DEFAULT 'user_asserted',

                    version         INTEGER NOT NULL DEFAULT 1,
                    supersedes      TEXT,
                    superseded_by   TEXT,

                    tenant_id       TEXT NOT NULL DEFAULT 'default',
                    user_id         TEXT,
                    agent_id        TEXT,
                    session_id      TEXT,

                    structured_json TEXT,
                    pii_json        TEXT DEFAULT '[]',
                    metadata_json   TEXT DEFAULT '{}',
                    embedding_ref   TEXT,
                    consolidated_from_json TEXT DEFAULT '[]',
                    idempotency_key TEXT UNIQUE
                );

                CREATE INDEX IF NOT EXISTS idx_mem_tenant
                    ON memories(tenant_id, state);
                CREATE INDEX IF NOT EXISTS idx_mem_user
                    ON memories(tenant_id, user_id, state);
                CREATE INDEX IF NOT EXISTS idx_mem_type
                    ON memories(type, state);
                CREATE INDEX IF NOT EXISTS idx_mem_entity
                    ON memories(tenant_id, json_extract(structured_json, '$.entity'), state);
                CREATE INDEX IF NOT EXISTS idx_mem_idem
                    ON memories(idempotency_key) WHERE idempotency_key IS NOT NULL;

                -- FTS5 for lexical search (A14)
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(
                        id UNINDEXED,
                        content,
                        tenant_id UNINDEXED,
                        user_id UNINDEXED,
                        content='memories',
                        content_rowid='rowid'
                    );

                -- FTS triggers
                CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, id, content, tenant_id, user_id)
                    VALUES (new.rowid, new.id, new.content, new.tenant_id, new.user_id);
                END;
                CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, id, content, tenant_id, user_id)
                    VALUES ('delete', old.rowid, old.id, old.content, old.tenant_id, old.user_id);
                END;
                CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, id, content, tenant_id, user_id)
                    VALUES ('delete', old.rowid, old.id, old.content, old.tenant_id, old.user_id);
                    INSERT INTO memories_fts(rowid, id, content, tenant_id, user_id)
                    VALUES (new.rowid, new.id, new.content, new.tenant_id, new.user_id);
                END;

                -- Event log (§56, A17.4)
                CREATE TABLE IF NOT EXISTS events (
                    id              TEXT PRIMARY KEY,
                    memory_id       TEXT NOT NULL,
                    event_type      TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    actor           TEXT NOT NULL DEFAULT 'system',
                    namespace_key   TEXT NOT NULL DEFAULT 'default',
                    policy_id       TEXT NOT NULL DEFAULT 'default',
                    policy_version  TEXT NOT NULL DEFAULT '1.0',
                    gate_scores_json TEXT DEFAULT '{}',
                    payload_json    TEXT DEFAULT '{}',
                    trace_id        TEXT,
                    prev_hash       TEXT,
                    hash            TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_evt_memory
                    ON events(memory_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_evt_type
                    ON events(event_type, timestamp);
                CREATE INDEX IF NOT EXISTS idx_evt_ns
                    ON events(namespace_key, timestamp);

                -- Provenance edges (A8)
                CREATE TABLE IF NOT EXISTS provenance_edges (
                    id          TEXT PRIMARY KEY,
                    from_id     TEXT NOT NULL,   -- memory_id or source_ref
                    to_id       TEXT NOT NULL,
                    relation    TEXT NOT NULL,   -- derived_from|supersedes|consolidates|cites|produced_by|caused|evaluated_by
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_prov_from ON provenance_edges(from_id);
                CREATE INDEX IF NOT EXISTS idx_prov_to   ON provenance_edges(to_id);

                -- Snapshots (A16.2)
                CREATE TABLE IF NOT EXISTS snapshots (
                    id              TEXT PRIMARY KEY,
                    label           TEXT,
                    created_at      TEXT NOT NULL,
                    event_cursor    TEXT,
                    namespace_key   TEXT NOT NULL DEFAULT 'default'
                );
            """)

    # ------------------------------------------------------------------ #
    # Memory CRUD
    # ------------------------------------------------------------------ #

    def put(self, memory: MemoryRecord, *, tx: sqlite3.Connection | None = None) -> None:
        """Insert or replace a memory record (idempotent upsert)."""
        row = self._to_row(memory)
        sql = """
            INSERT INTO memories (
                id, content, type, state,
                confidence, importance, utility, strength,
                recorded_at, created_at, observed_at,
                valid_from, valid_until, updated_at, last_accessed_at, expires_at,
                source_json, trust_tier, version, supersedes, superseded_by,
                tenant_id, user_id, agent_id, session_id,
                structured_json, pii_json, metadata_json, embedding_ref,
                consolidated_from_json, idempotency_key
            ) VALUES (
                :id, :content, :type, :state,
                :confidence, :importance, :utility, :strength,
                :recorded_at, :created_at, :observed_at,
                :valid_from, :valid_until, :updated_at, :last_accessed_at, :expires_at,
                :source_json, :trust_tier, :version, :supersedes, :superseded_by,
                :tenant_id, :user_id, :agent_id, :session_id,
                :structured_json, :pii_json, :metadata_json, :embedding_ref,
                :consolidated_from_json, :idempotency_key
            )
            ON CONFLICT(id) DO UPDATE SET
                content=excluded.content, type=excluded.type, state=excluded.state,
                confidence=excluded.confidence, importance=excluded.importance,
                utility=excluded.utility, strength=excluded.strength,
                recorded_at=excluded.recorded_at, updated_at=excluded.updated_at,
                last_accessed_at=excluded.last_accessed_at, expires_at=excluded.expires_at,
                source_json=excluded.source_json, trust_tier=excluded.trust_tier,
                version=excluded.version, superseded_by=excluded.superseded_by,
                structured_json=excluded.structured_json, pii_json=excluded.pii_json,
                metadata_json=excluded.metadata_json, embedding_ref=excluded.embedding_ref,
                consolidated_from_json=excluded.consolidated_from_json
        """
        if tx:
            tx.execute(sql, row)
        else:
            with self._transaction() as conn:
                conn.execute(sql, row)

    def get(self, memory_id: str, *, at: datetime | None = None) -> MemoryRecord | None:
        """Retrieve a memory by id. Supports as-of (bitemporal) lookup."""
        conn = self._conn()
        if at:
            row = conn.execute(
                """
                SELECT * FROM memories WHERE id=?
                AND (recorded_at <= ? OR recorded_at IS NULL)
                """,
                (memory_id, _dts(at)),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return self._from_row(row) if row else None

    def update_state(
        self,
        memory_id: str,
        state: MemoryState,
        *,
        superseded_by: str | None = None,
        tx: sqlite3.Connection | None = None,
        now: datetime | None = None,
    ) -> None:
        """Update only the lifecycle state of a memory."""
        ts = _dts(now or _now())
        sql = "UPDATE memories SET state=?, updated_at=?, superseded_by=? WHERE id=?"
        args = (state.value, ts, superseded_by, memory_id)
        if tx:
            tx.execute(sql, args)
        else:
            with self._transaction() as conn:
                conn.execute(sql, args)

    def touch(self, memory_id: str) -> None:
        """Update last_accessed_at."""
        with self._transaction() as conn:
            conn.execute(
                "UPDATE memories SET last_accessed_at=? WHERE id=?",
                (_dts(_now()), memory_id),
            )

    def list_by_namespace(
        self,
        namespace: Namespace,
        state_filter: list[MemoryState] | None = None,
        type_filter: list[MemoryType] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List memories filtered by namespace and optional state/type."""
        conn = self._conn()
        params: list[Any] = [namespace.tenant_id]
        where = ["tenant_id = ?"]

        if namespace.user_id:
            where.append("user_id = ?")
            params.append(namespace.user_id)
        if namespace.agent_id:
            where.append("agent_id = ?")
            params.append(namespace.agent_id)
        if namespace.session_id:
            where.append("session_id = ?")
            params.append(namespace.session_id)

        if state_filter:
            placeholders = ",".join("?" for _ in state_filter)
            where.append(f"state IN ({placeholders})")
            params.extend(s.value for s in state_filter)
        else:
            where.append("state = 'active'")

        if type_filter:
            placeholders = ",".join("?" for _ in type_filter)
            where.append(f"type IN ({placeholders})")
            params.extend(t.value for t in type_filter)

        params.extend([limit, offset])
        sql = (
            f"SELECT * FROM memories WHERE {' AND '.join(where)} "
            f"ORDER BY created_at DESC LIMIT ? OFFSET ?"
        )
        rows = conn.execute(sql, params).fetchall()
        return [self._from_row(r) for r in rows]

    def search_fts(
        self,
        query_text: str,
        namespace: Namespace,
        state_filter: list[MemoryState] | None = None,
        limit: int = 20,
    ) -> list[MemoryRecord]:
        """Full-text search using SQLite FTS5 (A14, A6.2)."""
        conn = self._conn()
        states = [s.value for s in (state_filter or [MemoryState.ACTIVE])]
        placeholders = ",".join("?" for _ in states)

        import re
        terms = [w for w in re.sub(r'[^\w\s]', '', query_text).split() if len(w) > 1]
        if not terms:
            return []
        safe_query = " OR ".join(terms)

        params: list[Any] = [safe_query, namespace.tenant_id]
        ns_filter = "AND m.tenant_id = ?"
        if namespace.user_id:
            ns_filter += " AND m.user_id = ?"
            params.append(namespace.user_id)

        params.extend(states)
        params.append(limit)

        sql = f"""
            SELECT m.* FROM memories m
            JOIN memories_fts f ON m.id = f.id
            WHERE memories_fts MATCH ?
              {ns_filter}
              AND m.state IN ({placeholders})
            ORDER BY rank
            LIMIT ?
        """
        try:
            rows = conn.execute(sql, params).fetchall()
            return [self._from_row(r) for r in rows]
        except sqlite3.OperationalError:
            # FTS syntax error — fall back to LIKE
            return self._search_like(query_text, namespace, state_filter, limit)

    def _search_like(
        self,
        query_text: str,
        namespace: Namespace,
        state_filter: list[MemoryState] | None,
        limit: int,
    ) -> list[MemoryRecord]:
        conn = self._conn()
        states = [s.value for s in (state_filter or [MemoryState.ACTIVE])]
        placeholders = ",".join("?" for _ in states)
        params: list[Any] = [f"%{query_text}%", namespace.tenant_id]
        ns_filter = "AND tenant_id = ?"
        if namespace.user_id:
            ns_filter += " AND user_id = ?"
            params.append(namespace.user_id)
        params.extend(states)
        params.append(limit)
        sql = (
            f"SELECT * FROM memories WHERE content LIKE ? {ns_filter} "
            f"AND state IN ({placeholders}) ORDER BY created_at DESC LIMIT ?"
        )
        rows = conn.execute(sql, params).fetchall()
        return [self._from_row(r) for r in rows]

    def get_by_idempotency_key(self, key: str) -> MemoryRecord | None:
        conn = self._conn()
        row = conn.execute("SELECT * FROM memories WHERE idempotency_key=?", (key,)).fetchone()
        return self._from_row(row) if row else None

    def get_by_entity_attribute(
        self,
        entity: str,
        attribute: str,
        namespace: Namespace,
        at: datetime | None = None,
    ) -> list[MemoryRecord]:
        """Exact structured state lookup (A4.2, A14) — no embeddings."""
        conn = self._conn()
        params: list[Any] = [entity, attribute, namespace.tenant_id, MemoryState.ACTIVE.value]
        ns_filter = "AND tenant_id = ?"
        if namespace.user_id:
            ns_filter += " AND user_id = ?"
            params.insert(-1, namespace.user_id)

        if at:
            time_filter = "AND (valid_from IS NULL OR valid_from <= ?) AND (valid_until IS NULL OR valid_until >= ?)"
            params_extra = [_dts(at), _dts(at)]
        else:
            time_filter = ""
            params_extra = []

        params.extend(params_extra)
        sql = f"""
            SELECT * FROM memories
            WHERE json_extract(structured_json, '$.entity') = ?
              AND json_extract(structured_json, '$.attribute') = ?
              {ns_filter}
              AND state = ?
              {time_filter}
            ORDER BY recorded_at DESC
            LIMIT 10
        """
        rows = conn.execute(sql, params).fetchall()
        return [self._from_row(r) for r in rows]

    def timeline(
        self,
        entity: str,
        namespace: Namespace,
        start: datetime | None = None,
        end: datetime | None = None,
        include_superseded: bool = True,
    ) -> list[MemoryRecord]:
        """Return entity state history ordered by valid_from (§57, A16.2)."""
        conn = self._conn()
        states = [MemoryState.ACTIVE.value, MemoryState.SUPERSEDED.value] if include_superseded else [MemoryState.ACTIVE.value]
        placeholders = ",".join("?" for _ in states)
        params: list[Any] = [entity, namespace.tenant_id]
        params.extend(states)
        time_clauses = ""
        if start:
            time_clauses += " AND (valid_from >= ? OR created_at >= ?)"
            params.extend([_dts(start), _dts(start)])
        if end:
            time_clauses += " AND (valid_from <= ? OR created_at <= ?)"
            params.extend([_dts(end), _dts(end)])
        sql = f"""
            SELECT * FROM memories
            WHERE json_extract(structured_json, '$.entity') = ?
              AND tenant_id = ?
              AND state IN ({placeholders})
              {time_clauses}
            ORDER BY COALESCE(valid_from, created_at) ASC
        """
        rows = conn.execute(sql, params).fetchall()
        return [self._from_row(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    def append_event(
        self, event: MemoryEvent, *, tx: sqlite3.Connection | None = None
    ) -> None:
        last = self._get_last_event_hash(event.namespace_key)
        event.prev_hash = last
        event.hash = event.compute_hash()

        sql = """
            INSERT INTO events (
                id, memory_id, event_type, timestamp, actor,
                namespace_key, policy_id, policy_version,
                gate_scores_json, payload_json, trace_id, prev_hash, hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        args = (
            event.id,
            event.memory_id,
            event.event_type,
            _dts(event.timestamp),
            event.actor,
            event.namespace_key,
            event.policy_id,
            event.policy_version,
            json.dumps(event.gate_scores),
            json.dumps(event.payload),
            event.trace_id,
            event.prev_hash,
            event.hash,
        )
        if tx:
            tx.execute(sql, args)
        else:
            with self._transaction() as conn:
                conn.execute(sql, args)

    def _get_last_event_hash(self, namespace_key: str) -> str | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT hash FROM events WHERE namespace_key=? ORDER BY timestamp DESC LIMIT 1",
            (namespace_key,),
        ).fetchone()
        return row["hash"] if row else None

    def list_events(
        self,
        memory_id: str | None = None,
        namespace_key: str | None = None,
        since: datetime | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        conn = self._conn()
        where = []
        params: list[Any] = []
        if memory_id:
            where.append("memory_id = ?")
            params.append(memory_id)
        if namespace_key:
            where.append("namespace_key = ?")
            params.append(namespace_key)
        if since:
            where.append("timestamp >= ?")
            params.append(_dts(since))
        if event_type:
            where.append("event_type = ?")
            params.append(event_type)
        params.append(limit)
        where_clause = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"SELECT * FROM events {where_clause} ORDER BY timestamp ASC LIMIT ?"
        rows = conn.execute(sql, params).fetchall()
        return [self._event_from_row(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #

    def add_provenance_edge(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        metadata: dict | None = None,
    ) -> None:
        with self._transaction() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO provenance_edges (id, from_id, to_id, relation, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (
                    f"pe_{uuid.uuid4().hex[:12]}",
                    from_id,
                    to_id,
                    relation,
                    json.dumps(metadata or {}),
                ),
            )

    def get_provenance(self, memory_id: str) -> list[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM provenance_edges WHERE from_id=? OR to_id=?",
            (memory_id, memory_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Snapshots (A16.2)
    # ------------------------------------------------------------------ #

    def create_snapshot(self, snapshot: Snapshot) -> None:
        # Capture current event cursor
        conn = self._conn()
        row = conn.execute(
            "SELECT id FROM events WHERE namespace_key=? ORDER BY timestamp DESC LIMIT 1",
            (snapshot.namespace_key,),
        ).fetchone()
        cursor = row["id"] if row else None

        with self._transaction() as c:
            c.execute(
                "INSERT INTO snapshots (id, label, created_at, event_cursor, namespace_key) VALUES (?, ?, ?, ?, ?)",
                (
                    snapshot.id,
                    snapshot.label,
                    _dts(snapshot.created_at),
                    cursor,
                    snapshot.namespace_key,
                ),
            )
        snapshot.event_cursor = cursor

    def get_snapshot(self, snapshot_id: str) -> Snapshot | None:
        conn = self._conn()
        row = conn.execute("SELECT * FROM snapshots WHERE id=?", (snapshot_id,)).fetchone()
        if not row:
            return None
        return Snapshot(
            id=row["id"],
            label=row["label"],
            created_at=_dt(row["created_at"]) or _now(),
            event_cursor=row["event_cursor"],
            namespace_key=row["namespace_key"],
        )

    def list_snapshots(self, namespace_key: str) -> list[Snapshot]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM snapshots WHERE namespace_key=? ORDER BY created_at DESC",
            (namespace_key,),
        ).fetchall()
        return [
            Snapshot(
                id=r["id"],
                label=r["label"],
                created_at=_dt(r["created_at"]) or _now(),
                event_cursor=r["event_cursor"],
                namespace_key=r["namespace_key"],
            )
            for r in rows
        ]

    def diff(
        self,
        from_cursor: str | None,
        to_cursor: str | None,
        namespace_key: str,
    ) -> list[MemoryEvent]:
        """Return events between two event cursors (A16.2)."""
        conn = self._conn()
        params: list[Any] = [namespace_key]
        where = "WHERE namespace_key = ?"

        if from_cursor:
            from_ts = conn.execute(
                "SELECT timestamp FROM events WHERE id=?", (from_cursor,)
            ).fetchone()
            if from_ts:
                where += " AND timestamp > ?"
                params.append(from_ts["timestamp"])

        if to_cursor:
            to_ts = conn.execute(
                "SELECT timestamp FROM events WHERE id=?", (to_cursor,)
            ).fetchone()
            if to_ts:
                where += " AND timestamp <= ?"
                params.append(to_ts["timestamp"])

        rows = conn.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp ASC", params
        ).fetchall()
        return [self._event_from_row(r) for r in rows]

    def count(self, namespace: Namespace, state: MemoryState | None = None) -> int:
        conn = self._conn()
        params: list[Any] = [namespace.tenant_id]
        where = "tenant_id = ?"
        if namespace.user_id:
            where += " AND user_id = ?"
            params.append(namespace.user_id)
        if state:
            where += " AND state = ?"
            params.append(state.value)
        row = conn.execute(f"SELECT COUNT(*) as cnt FROM memories WHERE {where}", params).fetchone()
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #

    def _to_row(self, m: MemoryRecord) -> dict:
        return {
            "id": m.id,
            "content": m.content,
            "type": m.type.value,
            "state": m.state.value,
            "confidence": m.confidence,
            "importance": m.importance,
            "utility": m.utility,
            "strength": m.strength,
            "recorded_at": _dts(m.recorded_at),
            "created_at": _dts(m.created_at),
            "observed_at": _dts(m.observed_at),
            "valid_from": _dts(m.valid_from),
            "valid_until": _dts(m.valid_until),
            "updated_at": _dts(m.updated_at),
            "last_accessed_at": _dts(m.last_accessed_at),
            "expires_at": _dts(m.expires_at),
            "source_json": m.source.model_dump_json(),
            "trust_tier": m.trust_tier.value,
            "version": m.version,
            "supersedes": m.supersedes,
            "superseded_by": m.superseded_by,
            "tenant_id": m.namespace.tenant_id,
            "user_id": m.namespace.user_id,
            "agent_id": m.namespace.agent_id,
            "session_id": m.namespace.session_id,
            "structured_json": m.structured.model_dump_json() if m.structured else None,
            "pii_json": json.dumps(m.pii),
            "metadata_json": json.dumps(m.metadata),
            "embedding_ref": m.embedding_reference,
            "consolidated_from_json": json.dumps(m.consolidated_from),
            "idempotency_key": m.idempotency_key,
        }

    def _from_row(self, row: sqlite3.Row) -> MemoryRecord:
        d = dict(row)
        source = Source.model_validate_json(d.get("source_json") or "{}")
        structured = None
        if d.get("structured_json"):
            try:
                structured = StructuredFields.model_validate_json(d["structured_json"])
            except Exception:
                pass

        ns = Namespace(
            tenant_id=d.get("tenant_id", "default"),
            user_id=d.get("user_id"),
            agent_id=d.get("agent_id"),
            session_id=d.get("session_id"),
        )

        return MemoryRecord(
            id=d["id"],
            content=d["content"],
            type=MemoryType(d.get("type", "semantic")),
            state=MemoryState(d.get("state", "candidate")),
            confidence=float(d.get("confidence", 1.0)),
            importance=float(d.get("importance", 0.5)),
            utility=float(d.get("utility", 0.5)),
            strength=float(d.get("strength", 1.0)),
            recorded_at=_dt(d.get("recorded_at")) or _now(),
            created_at=_dt(d.get("created_at")) or _now(),
            observed_at=_dt(d.get("observed_at")),
            valid_from=_dt(d.get("valid_from")),
            valid_until=_dt(d.get("valid_until")),
            updated_at=_dt(d.get("updated_at")),
            last_accessed_at=_dt(d.get("last_accessed_at")),
            expires_at=_dt(d.get("expires_at")),
            source=source,
            trust_tier=TrustTier(d.get("trust_tier", "user_asserted")),
            version=int(d.get("version", 1)),
            supersedes=d.get("supersedes"),
            superseded_by=d.get("superseded_by"),
            namespace=ns,
            structured=structured,
            pii=json.loads(d.get("pii_json") or "[]"),
            metadata=json.loads(d.get("metadata_json") or "{}"),
            embedding_reference=d.get("embedding_ref"),
            consolidated_from=json.loads(d.get("consolidated_from_json") or "[]"),
            idempotency_key=d.get("idempotency_key"),
        )

    def _event_from_row(self, row: sqlite3.Row) -> MemoryEvent:
        d = dict(row)
        return MemoryEvent(
            id=d["id"],
            memory_id=d["memory_id"],
            event_type=d["event_type"],
            timestamp=_dt(d["timestamp"]) or _now(),
            actor=d.get("actor", "system"),
            namespace_key=d.get("namespace_key", "default"),
            policy_id=d.get("policy_id", "default"),
            policy_version=d.get("policy_version", "1.0"),
            gate_scores=json.loads(d.get("gate_scores_json") or "{}"),
            payload=json.loads(d.get("payload_json") or "{}"),
            trace_id=d.get("trace_id"),
            prev_hash=d.get("prev_hash"),
            hash=d.get("hash"),
        )

    def health(self) -> dict:
        """Return basic health information."""
        conn = self._conn()
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE state='active'"
        ).fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {
            "substrate": "sqlite",
            "db_path": self.db_path,
            "total_memories": total,
            "active_memories": active,
            "total_events": events,
            "ok": True,
        }
