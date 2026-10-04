<div align="center">
  <picture>
    <!-- Replace with actual logo URL -->
    <img src="https://raw.githubusercontent.com/membrane-memory/membrane/main/docs/assets/logo.svg" alt="Membrane Logo" width="200" height="200">
  </picture>
  
  <br/>
  
  # Membrane
  
  **The programmable memory layer for AI agents.**

  [![PyPI Version](https://img.shields.io/pypi/v/membrane-memory.svg?style=for-the-badge&color=blue)](https://pypi.org/project/membrane-memory/)
  [![Python Versions](https://img.shields.io/pypi/pyversions/membrane-memory.svg?style=for-the-badge)](https://pypi.org/project/membrane-memory/)
  [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)](https://opensource.org/licenses/Apache-2.0)
  [![Tests](https://img.shields.io/github/actions/workflow/status/membrane-memory/membrane/test.yml?branch=main&label=tests&style=for-the-badge)](https://github.com/membrane-memory/membrane/actions)
  
  <p align="center">
    <a href="#why-membrane">Why Membrane?</a> •
    <a href="#quickstart">Quickstart</a> •
    <a href="#how-it-works">How it Works</a> •
    <a href="#features">Features</a> •
    <a href="https://membrane-memory.github.io/membrane">Documentation</a>
  </p>
</div>

---

**Membrane** is a production-grade, programmable memory runtime for LLMs and AI agents. 

Unlike standard RAG pipelines that treat memory as a static text dump, Membrane treats memory as a **first-class managed resource**. It actively manages the entire memory lifecycle—from deciding *what* to write, to automatically superseding stale facts, tracking causal provenance, and retrieving data via intelligent intent planning.

No API keys. No Docker required. No mandatory LLM dependencies. Local-first by default.

## ⚡ Quickstart

Install Membrane via pip:

```bash
pip install membrane-memory
```

Get started in under 5 minutes:

```python
from membrane import Memory

# 1. Initialize an embedded, local-first memory store (SQLite default)
memory = Memory()

# 2. Tell the system something to remember
memory.remember(
    "The user prefers concise technical explanations.",
    user_id="alice_123"
)

# 3. Recall the most relevant facts dynamically
result = memory.recall(
    "How should I communicate with this user?",
    user_id="alice_123"
)

print(result.memories[0].content)
# > "The user prefers concise technical explanations."
```

Want to know *why* the system recalled that fact? Ask Membrane to explain itself:

```python
explanation = memory.explain(result.recall_id)
print(explanation.summary)
```

---

## 🧠 Why Membrane?

Current agent architectures treat memory as an afterthought—usually just an embedding database (RAG). Membrane is different. Inspired by human cognitive architecture (LSTM/Working Memory models), Membrane introduces **programmable gates**:

- 🛡️ **Write Gate**: Filters out noise, redundant data, and prompt-injection attempts before they hit the database.
- 🔄 **Update Gate**: Detects when new information contradicts old facts and automatically *supersedes* them instead of creating duplicates.
- ⏱️ **Forget Gate**: Applies exponential age decay curves so your agent's memory doesn't become bloated with useless session state.
- 🎯 **Read Gate**: Scores memories across 10+ dimensions (lexical, temporal freshness, trust, utility, confidence) to curate the perfect working context.

### The Agent Universal Pattern

Membrane sits alongside your agent framework (LangChain, LangGraph, AutoGen) and takes total ownership of state.

```text
USER ──► AGENT ──► Membrane.recall()  (curates token-budgeted context)
           │
           ▼
         LLM ──► tools ──► RESULT
           │
           └───► Membrane.remember()  (supersedes old facts, applies safety gates)
```

---

## ✨ Features

- **Multi-tenant by Design**: Every operation is scoped by `tenant_id`, `user_id`, or `session_id`. Perfect for multi-user SaaS.
- **Bitemporal History**: Membrane never silently deletes. It keeps a tamper-evident event log. Ask *"What did the agent know as of last Tuesday?"* using native time-travel snapshots.
- **Intelligent Query Planner**: A built-in router detects if a query is semantic, temporal (*"timeline"*), or structured (*"current price"*), routing it to the most optimal backend.
- **Security & Quarantine**: In-built trust tiers. Memories from untrusted external sources containing instruction-like text (*"ignore previous prompts"*) are automatically placed in a `QUARANTINED` state pending review.
- **Local-First & Portable**: Powered by standard SQLite. It fits in a single file and scales up to Postgres or Vector DBs when you need them.

## 🔌 Ecosystem & Integrations

Membrane is designed to plug right into your stack. Use the core locally, or expand it with optional extras:

```bash
# Add an MCP server for immediate agent integration
pip install "membrane-memory[mcp]"

# Add FastAPI REST Server
pip install "membrane-memory[server]"

# Add local HuggingFace embeddings
pip install "membrane-memory[embeddings-local]"
```

Start the built-in REST API instantly:
```bash
membrane serve --port 8000
```

---

## 🛠️ Advanced Usage

### Explicit Temporal State Tracking
Don't rely on embeddings for exact facts. Use structured fields:

```python
from membrane.core.models import StructuredFields

# Yesterday: price is $10
memory.remember(
    "SKU 123 price is $10",
    structured=StructuredFields(entity="sku_123", attribute="price", value="10")
)

# Today: price is $12 (this automatically supersedes the $10 memory!)
memory.remember(
    "SKU 123 price is $12",
    structured=StructuredFields(entity="sku_123", attribute="price", value="12")
)

# The agent only sees the freshest data
print(memory.recall("price", entity="sku_123").memories[0].content) 
# > "SKU 123 price is $12"

# But the timeline preserves history!
history = memory.timeline("sku_123")
```

### Context Budgeting
Never blow your LLM context window again:

```python
# Membrane guarantees the returned memories fit inside 1,000 tokens
result = memory.recall("query", token_budget=1000)
```

---

## 🏗️ Architecture

Read the full [Project Specification (A3 / Architecture)](./PROJECT.md) for deep-dive technical details on how Membrane manages the memory lifecycle through Substrates, Policies, and Outcome-based reinforcement.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to set up the development environment, run tests, and submit Pull Requests.

```bash
git clone https://github.com/membrane-memory/membrane.git
cd membrane
pip install -e ".[dev]"
pytest tests/
```

## 📄 License

Membrane is open-sourced under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

---

<div align="center">
  <i>"Those who cannot remember the past are condemned to hallucinate it."</i>
</div>
