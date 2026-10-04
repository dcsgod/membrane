"""
REST API server (§26, A26).

Optional — requires: pip install membrane-memory[server]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query as FastAPIQuery
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install the [server] extra: pip install membrane-memory[server]")

from membrane.core.memory import Memory
from membrane.core.models import MemoryState, MemoryType, TrustTier


class RememberRequest(BaseModel):
    content: str
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    tenant_id: str = "default"
    type: str | None = None
    confidence: float = 0.9
    importance: float = 0.5
    trust_tier: str = "user_asserted"
    idempotency_key: str | None = None
    persist: bool = True
    metadata: dict[str, Any] | None = None


class RecallRequest(BaseModel):
    query: str
    user_id: str | None = None
    tenant_id: str = "default"
    limit: int = 10
    token_budget: int | None = None
    entity: str | None = None
    attribute: str | None = None
    include_superseded: bool = False


class FeedbackRequest(BaseModel):
    recall_id: str
    outcome: float  # 0.0 (bad) to 1.0 (good)
    user_id: str | None = None
    tenant_id: str = "default"


def create_app(db_path: str = ":memory:") -> FastAPI:
    memory = Memory(db_path=db_path)

    app = FastAPI(
        title="Membrane API",
        description="The programmable memory layer for AI.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/health")
    def health():
        return memory.health()

    @app.post("/v1/memories")
    def remember(req: RememberRequest):
        record = memory.remember(
            content=req.content,
            user_id=req.user_id,
            agent_id=req.agent_id,
            session_id=req.session_id,
            tenant_id=req.tenant_id,
            type=req.type,
            confidence=req.confidence,
            importance=req.importance,
            trust_tier=req.trust_tier,
            idempotency_key=req.idempotency_key,
            persist=req.persist,
            metadata=req.metadata,
        )
        if record is None:
            raise HTTPException(status_code=422, detail="Memory was rejected by write gate")
        return record.model_dump(mode="json")

    @app.get("/v1/memories/{memory_id}")
    def get_memory(memory_id: str):
        from membrane.substrates.sqlite.substrate import SQLiteSubstrate

        record = memory._substrate.get(memory_id)
        if not record:
            raise HTTPException(status_code=404, detail="Memory not found")
        return record.model_dump(mode="json")

    @app.post("/v1/memory/recall")
    def recall(req: RecallRequest):
        result = memory.recall(
            query=req.query,
            user_id=req.user_id,
            tenant_id=req.tenant_id,
            limit=req.limit,
            token_budget=req.token_budget,
            entity=req.entity,
            attribute=req.attribute,
            include_superseded=req.include_superseded,
        )
        return {
            "recall_id": result.recall_id,
            "query": result.query,
            "memories": [m.model_dump(mode="json") for m in result.memories],
            "scores": [s.model_dump() for s in result.scores],
            "tokens_used": result.tokens_used,
            "dropped": result.dropped,
            "degraded": result.degraded,
        }

    @app.delete("/v1/memories/{memory_id}")
    def forget_memory(memory_id: str, mode: str = "archive"):
        ok = memory.forget(memory_id, mode=mode)
        if not ok:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"ok": True, "mode": mode}

    @app.get("/v1/memory/explain/{ref_id}")
    def explain(ref_id: str):
        result = memory.explain(ref_id)
        return result.model_dump(mode="json")

    @app.post("/v1/memory/plan")
    def plan(body: dict):
        query = body.get("query", "")
        return memory.explain_plan(query)

    @app.get("/v1/timeline/{entity}")
    def timeline(entity: str, user_id: str | None = None, tenant_id: str = "default"):
        records = memory.timeline(entity, user_id=user_id, tenant_id=tenant_id)
        return [r.model_dump(mode="json") for r in records]

    @app.post("/v1/snapshots")
    def snapshot(body: dict):
        snap = memory.snapshot(label=body.get("label"), tenant_id=body.get("tenant_id", "default"))
        return snap.model_dump(mode="json")

    @app.get("/v1/diff")
    def diff(from_snap: str, to_snap: str | None = None, tenant_id: str = "default"):
        events = memory.diff(from_snap, to_snap, tenant_id=tenant_id)
        return [e.model_dump(mode="json") for e in events]

    @app.post("/v1/rollback")
    def rollback(body: dict):
        count = memory.rollback(body.get("snapshot_id", ""), tenant_id=body.get("tenant_id", "default"))
        return {"rolled_back": count}

    @app.get("/v1/review")
    def review(tenant_id: str = "default"):
        queue = memory.review_queue(tenant_id=tenant_id)
        return [m.model_dump(mode="json") for m in queue]

    @app.post("/v1/review/{memory_id}/approve")
    def approve(memory_id: str):
        ok = memory.approve(memory_id)
        return {"ok": ok}

    @app.post("/v1/review/{memory_id}/reject")
    def reject(memory_id: str, body: dict = {}):
        ok = memory.reject(memory_id, reason=body.get("reason", ""))
        return {"ok": ok}

    @app.post("/v1/memory/consolidate")
    def consolidate(body: dict = {}):
        count = memory.consolidate(
            user_id=body.get("user_id"),
            tenant_id=body.get("tenant_id", "default"),
        )
        return {"consolidated": count}

    @app.get("/v1/provenance/{memory_id}")
    def provenance(memory_id: str):
        return memory.provenance(memory_id)

    @app.get("/v1/doctor")
    def doctor(tenant_id: str = "default"):
        return memory.reflect(tenant_id=tenant_id)

    return app
