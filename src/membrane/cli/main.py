"""
Membrane CLI (§43, A43).

Usage:
    membrane init [db_path]
    membrane status
    membrane remember "content" --user USER
    membrane recall "query" --user USER
    membrane forget MEMORY_ID
    membrane explain MEMORY_ID
    membrane plan "query"
    membrane memories [--user USER] [--limit N]
    membrane timeline ENTITY [--user USER]
    membrane snapshot [--label LABEL]
    membrane diff SNAP_ID
    membrane rollback SNAP_ID
    membrane review
    membrane approve MEMORY_ID
    membrane reject MEMORY_ID
    membrane consolidate [--user USER]
    membrane health
    membrane serve [--port PORT]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional


def _get_memory(db_path: str) -> "Memory":  # type: ignore[name-defined]
    from membrane.core.memory import Memory

    return Memory(db_path=db_path)


def _default_db() -> str:
    return str(Path.cwd() / "membrane.db")


def app() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="membrane",
        description="Membrane — the programmable memory layer for AI.",
    )
    parser.add_argument("--db", default=None, help="Database path (default: ./membrane.db)")
    parser.add_argument("--version", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command")

    # init
    init_p = subparsers.add_parser("init", help="Initialize a Membrane database")
    init_p.add_argument("db_path", nargs="?", default=None)

    # status / health
    subparsers.add_parser("status", help="Show database status")
    subparsers.add_parser("health", help="Show memory health")

    # remember
    rem_p = subparsers.add_parser("remember", help="Remember a piece of information")
    rem_p.add_argument("content")
    rem_p.add_argument("--user", default=None)
    rem_p.add_argument("--tenant", default="default")
    rem_p.add_argument("--importance", type=float, default=0.5)
    rem_p.add_argument("--confidence", type=float, default=0.9)

    # recall
    rec_p = subparsers.add_parser("recall", help="Recall memories matching a query")
    rec_p.add_argument("query")
    rec_p.add_argument("--user", default=None)
    rec_p.add_argument("--tenant", default="default")
    rec_p.add_argument("--limit", type=int, default=5)
    rec_p.add_argument("--json", action="store_true", dest="output_json")

    # forget
    fgt_p = subparsers.add_parser("forget", help="Forget a memory")
    fgt_p.add_argument("memory_id")
    fgt_p.add_argument("--mode", choices=["archive", "delete", "expire"], default="archive")
    fgt_p.add_argument("--reason", default="manual")

    # explain
    exp_p = subparsers.add_parser("explain", help="Explain a memory or recall decision")
    exp_p.add_argument("ref_id")

    # plan
    plan_p = subparsers.add_parser("plan", help="Show query plan (dry run)")
    plan_p.add_argument("query")

    # memories
    mem_p = subparsers.add_parser("memories", help="List memories")
    mem_p.add_argument("--user", default=None)
    mem_p.add_argument("--tenant", default="default")
    mem_p.add_argument("--limit", type=int, default=20)
    mem_p.add_argument("--state", default="active")

    # timeline
    tl_p = subparsers.add_parser("timeline", help="Show entity timeline")
    tl_p.add_argument("entity")
    tl_p.add_argument("--user", default=None)
    tl_p.add_argument("--tenant", default="default")

    # snapshot
    snap_p = subparsers.add_parser("snapshot", help="Create a snapshot")
    snap_p.add_argument("--label", default=None)
    snap_p.add_argument("--user", default=None)
    snap_p.add_argument("--tenant", default="default")

    # diff
    diff_p = subparsers.add_parser("diff", help="Show events between two snapshots")
    diff_p.add_argument("from_snap")
    diff_p.add_argument("to_snap", nargs="?", default=None)
    diff_p.add_argument("--tenant", default="default")

    # rollback
    rb_p = subparsers.add_parser("rollback", help="Rollback to a snapshot")
    rb_p.add_argument("snapshot_id")
    rb_p.add_argument("--tenant", default="default")

    # review
    rev_p = subparsers.add_parser("review", help="Show quarantine queue")
    rev_p.add_argument("--tenant", default="default")

    # approve
    app_p = subparsers.add_parser("approve", help="Approve a quarantined memory")
    app_p.add_argument("memory_id")

    # reject
    rej_p = subparsers.add_parser("reject", help="Reject a quarantined memory")
    rej_p.add_argument("memory_id")
    rej_p.add_argument("--reason", default="")

    # consolidate
    con_p = subparsers.add_parser("consolidate", help="Consolidate episodic memories")
    con_p.add_argument("--user", default=None)
    con_p.add_argument("--tenant", default="default")

    # serve
    srv_p = subparsers.add_parser("serve", help="Start the REST API server")
    srv_p.add_argument("--port", type=int, default=8000)
    srv_p.add_argument("--host", default="127.0.0.1")

    args = parser.parse_args()

    if args.version:
        from membrane import __version__

        print(f"membrane {__version__}")
        return

    if not args.command:
        parser.print_help()
        return

    db_path = args.db or _default_db()

    # Commands that don't need the DB
    if args.command == "init":
        target = args.db_path or db_path
        mem = _get_memory(target)
        h = mem.health()
        print(f"✓ Membrane initialized at: {target}")
        print(f"  Substrate: {h.get('substrate', 'sqlite')}")
        return

    if args.command == "serve":
        _cmd_serve(db_path, args)
        return

    # All other commands
    memory = _get_memory(db_path)

    if args.command == "status":
        h = memory.health()
        print(f"Membrane status ({db_path})")
        for k, v in h.items():
            print(f"  {k}: {v}")

    elif args.command == "health":
        h = memory.health()
        score = 100  # simple placeholder
        print(f"Memory Health Score: {score}/100")
        for k, v in h.items():
            print(f"  {k}: {v}")

    elif args.command == "remember":
        record = memory.remember(
            args.content,
            user_id=args.user,
            tenant_id=args.tenant,
            importance=args.importance,
            confidence=args.confidence,
        )
        if record:
            print(f"✓ Remembered: [{record.id}] {record.content[:80]}")
            print(f"  type={record.type.value} state={record.state.value}")
        else:
            print("✗ Memory was rejected (below write threshold or duplicate).")

    elif args.command == "recall":
        result = memory.recall(
            args.query,
            user_id=args.user,
            tenant_id=args.tenant,
            limit=args.limit,
        )
        if args.output_json:
            print(json.dumps([m.model_dump(mode="json") for m in result.memories], indent=2, default=str))
        else:
            if not result.memories:
                print("No memories found.")
            else:
                print(f"Found {len(result.memories)} memories (tokens≈{result.tokens_used}):")
                for i, (mem, score) in enumerate(zip(result.memories, result.scores), 1):
                    print(f"\n  [{i}] {mem.id} (score={score.final_score:.3f})")
                    print(f"      {mem.content[:120]}")
                    print(f"      type={mem.type.value} trust={mem.trust_tier.value}")

    elif args.command == "forget":
        ok = memory.forget(args.memory_id, mode=args.mode, reason=args.reason)
        if ok:
            print(f"✓ Memory {args.memory_id} {args.mode}d.")
        else:
            print(f"✗ Memory {args.memory_id} not found.")

    elif args.command == "explain":
        result = memory.explain(args.ref_id)
        print(f"Explanation for {args.ref_id}:")
        print(f"  {result.summary}")
        if result.gate_scores:
            print(f"  Gate scores: {result.gate_scores}")
        if result.provenance:
            print(f"  Provenance: {result.provenance}")
        if result.events:
            print(f"\n  Events ({len(result.events)}):")
            for e in result.events[:5]:
                print(f"    {e.timestamp.strftime('%Y-%m-%d %H:%M')} {e.event_type}")

    elif args.command == "plan":
        plan = memory.explain_plan(args.query)
        print(f"Query plan for: {args.query!r}")
        print(f"  intent:       {plan['intent']}")
        print(f"  needs_memory: {plan['needs_memory']}")
        print(f"  rationale:    {plan['rationale']}")
        print(f"  steps:")
        for step in plan["steps"]:
            print(f"    {step['substrate']:10} {step['operation']:12} {step.get('params', {})}")

    elif args.command == "memories":
        from membrane.core.models import MemoryState

        try:
            state = MemoryState(args.state)
        except ValueError:
            state = MemoryState.ACTIVE
        memories = memory.list_memories(
            user_id=args.user, tenant_id=args.tenant, state=state, limit=args.limit
        )
        if not memories:
            print("No memories found.")
        else:
            print(f"{'ID':<22} {'Type':<14} {'State':<14} {'Conf':>6} {'Content'}")
            print("-" * 80)
            for m in memories:
                print(
                    f"{m.id:<22} {m.type.value:<14} {m.state.value:<14} "
                    f"{m.confidence:>5.2f}  {m.content[:45]}"
                )

    elif args.command == "timeline":
        records = memory.timeline(args.entity, user_id=args.user, tenant_id=args.tenant)
        if not records:
            print(f"No timeline entries for entity: {args.entity}")
        else:
            print(f"Timeline for '{args.entity}':")
            for r in records:
                ts = (r.valid_from or r.created_at).strftime("%Y-%m-%d")
                val = ""
                if r.structured:
                    val = f"  {r.structured.attribute}={r.structured.value}"
                print(f"  {ts}  [{r.state.value}]  {r.content[:60]}{val}")

    elif args.command == "snapshot":
        snap = memory.snapshot(args.label, user_id=args.user, tenant_id=args.tenant)
        print(f"✓ Snapshot created: {snap.id}")
        print(f"  label: {snap.label or '(none)'}")
        print(f"  cursor: {snap.event_cursor or '(empty)'}")

    elif args.command == "diff":
        events = memory.diff(args.from_snap, args.to_snap, tenant_id=args.tenant)
        if not events:
            print("No changes between snapshots.")
        else:
            print(f"{len(events)} event(s) since snapshot {args.from_snap}:")
            for e in events[:20]:
                print(f"  {e.timestamp.strftime('%Y-%m-%d %H:%M')}  {e.event_type}  memory={e.memory_id}")

    elif args.command == "rollback":
        count = memory.rollback(args.snapshot_id, tenant_id=args.tenant)
        print(f"✓ Rolled back {count} memories to snapshot {args.snapshot_id}")

    elif args.command == "review":
        queue = memory.review_queue(tenant_id=args.tenant)
        if not queue:
            print("No quarantined memories.")
        else:
            print(f"{len(queue)} quarantined memories:")
            for m in queue:
                print(f"  {m.id}  {m.content[:80]}")
                print(f"    trust={m.trust_tier.value} source={m.source.type}")

    elif args.command == "approve":
        ok = memory.approve(args.memory_id)
        print(f"{'✓ Approved' if ok else '✗ Not found or not quarantined'}: {args.memory_id}")

    elif args.command == "reject":
        ok = memory.reject(args.memory_id, reason=args.reason)
        print(f"{'✓ Rejected' if ok else '✗ Not found or not quarantined'}: {args.memory_id}")

    elif args.command == "consolidate":
        count = memory.consolidate(user_id=args.user, tenant_id=args.tenant)
        print(f"✓ Consolidated {count} memory cluster(s).")


def _cmd_serve(db_path: str, args) -> None:
    try:
        from membrane.server.api import create_app
        import uvicorn

        app_instance = create_app(db_path=db_path)
        print(f"Starting Membrane REST API on http://{args.host}:{args.port}")
        uvicorn.run(app_instance, host=args.host, port=args.port)
    except ImportError:
        print(
            "The [server] extra is required. Install it with:\n"
            "  pip install membrane-memory[server]"
        )
        sys.exit(1)
