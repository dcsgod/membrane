from __future__ import annotations

import sys

from membrane.cli.main import app
from membrane.core.memory import Memory


def test_memory_persists_across_instances(tmp_path):
    db = tmp_path / "membrane.db"

    writer = Memory(db_path=db)
    record = writer.remember(
        "Membrane is a programmable memory layer for AI agents",
        user_id="ravi",
    )
    assert record is not None

    reader = Memory(db_path=db)
    result = reader.recall("What is Membrane?", user_id="ravi")

    assert any(
        memory.id == record.id
        and memory.content == "Membrane is a programmable memory layer for AI agents"
        for memory in result.memories
    )


def test_cli_remember_and_recall_persist(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "membrane",
            "remember",
            "Membrane is a programmable memory layer",
            "--user",
            "ravi",
        ],
    )
    app()
    remember_output = capsys.readouterr().out

    assert "Remembered:" in remember_output
    assert "state=active" in remember_output

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "membrane",
            "recall",
            "What is Membrane?",
            "--user",
            "ravi",
        ],
    )
    app()
    recall_output = capsys.readouterr().out

    assert "Found 1 memories" in recall_output
    assert "Membrane is a programmable memory layer" in recall_output
