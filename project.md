# Membrane Project Specification

## Goal

Build Membrane as a framework-agnostic, LLM-optional programmable memory runtime.

> Memory should be managed like a computational resource, not retrieved like a document collection.

## Non-goals

- Do not become a generic RAG framework.
- Do not require a vector database.
- Do not require an LLM.
- Do not couple the core to LangChain, LangGraph, OpenAI, Anthropic, or MCP.
- Do not treat embeddings as the canonical representation of every memory.

## Architecture

Application / Agent
→ Memory API
→ Memory Controller
→ Query Planner
→ Memory Router
→ heterogeneous substrates
→ rank / merge / budget
→ working context

## Controller

z_t = [q_t ; x_t ; c_t ; h_{t-1}]

f_t = σ(W_f z_t + b_f)
i_t = σ(W_i z_t + b_i)
M̃_t = φ(W_m z_t + b_m)
u_t = σ(W_u z_t + b_u)

M̄_t = f_t ⊙ M_{t-1} + i_t ⊙ M̃_t
M_t = (1-u_t) ⊙ M_{t-1} + u_t ⊙ M̄_t

o_t = σ(W_o [h_t ; q_t] + b_o)
R_t = o_t ⊙ Retrieve(Plan(q_t, c_t), M)

These equations define controller policy signals. External databases are not treated as literal LSTM tensors.

## Core API

- memory.remember(...)
- memory.recall(...)
- memory.update(...)
- memory.forget(...)
- memory.timeline(...)
- memory.as_of(...)
- memory.consolidate(...)
- memory.explain(...)

## Lifecycle

CANDIDATE → ACTIVE → UPDATED / SUPERSEDED → ARCHIVED → FORGOTTEN

Unsafe observations may enter QUARANTINED.

Every lifecycle transition must be event logged.

## Planner

Minimum intents:

- semantic
- structured
- temporal
- graph
- current-state / KV
- episodic
- hybrid

Planner output must be inspectable and serializable.

## Substrate Interface

Every substrate should implement:

- write(memory)
- read(query)
- update(memory_id, patch)
- delete(memory_id)
- timeline(scope)
- health()

Optional capabilities:

- vector_search
- graph_traversal
- point_in_time
- batch_write
- batch_read

SQLite is mandatory for v0.1.

## Memory Model

Minimum fields:

- id
- tenant_id
- user_id
- session_id
- content
- memory_type
- entity
- attribute
- value
- source
- source_id
- confidence
- trust
- utility
- created_at
- valid_from
- valid_to
- state
- supersedes_id
- metadata

## Provenance

Every persistent memory should retain source, lifecycle events, revision lineage, supersession relationships, and retrieval explanation.

## Security

Implement trust tiers, quarantine, tenant boundaries, source metadata, policy-controlled writes, and audit events.

Untrusted instruction-like content must not automatically become trusted executable instructions.

## Outcome Memory

Represent:

decision → action → outcome → utility

Utility may later influence ranking, retention, or learned gate policies.

## Consolidation

Consolidation should merge duplicates, summarize repeated episodes, preserve provenance, create durable semantic memories, and archive source episodes when policy permits.

## Benchmark

MemoryBench should measure:

- factual recall
- temporal recall
- contradiction resolution
- write precision
- forget precision
- consolidation quality
- provenance accuracy
- context efficiency
- latency
- storage growth
- token consumption

Baselines:

1. vector RAG
2. vector memory
3. deterministic memory
4. Membrane without planner
5. Membrane without gates
6. Membrane without temporal state
7. full Membrane

## Implementation Order

### Phase 1
typed models → SQLite → lifecycle → remember / recall → provenance

### Phase 2
deterministic gates → planner → router → structured + temporal retrieval

### Phase 3
semantic substrate → embeddings adapter → hybrid retrieval → context budgeting

### Phase 4
consolidation → outcome memory → replay → explainability

### Phase 5
MCP → REST → LangGraph / LangChain / AutoGen adapters

### Phase 6
learned gate policies → MemoryBench → research experiments

## Quality Bar

Before v0.1:

- no mandatory API key
- no mandatory LLM
- SQLite works offline
- lifecycle transitions tested
- temporal queries tested
- contradiction cases tested
- tenant isolation tested
- planner decisions explainable
- router decisions explainable
- deterministic test suite
- reproducible benchmark
