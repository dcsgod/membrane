# Membrane — Project Specification

> **Membrane — The programmable memory layer for AI.**
>
> A programmable memory runtime that dynamically manages information across heterogeneous storage systems for AI applications.

| Field | Value |
|---|---|
| Project | **Membrane** |
| Repository (proposed) | `dcsgod/membrane` (verify the GitHub org is available, see A1.5) |
| Python distribution | `membrane-memory` (import name: `membrane`) |
| Status | Specification — implementation not started |
| Audience | Codex / Claude Code / human contributors implementing the project |
| Ambition | Serious open-source infrastructure **and** a publishable research program. Target: state of the art (see A28 for what that means and how it is proven) |

---

## How to read this document

This specification has two parts.

- **PART I — Membrane Architecture Charter (normative).** Defines the architecture, the research thesis, the SOTA bar, and the README/documentation mandate. Part I is authoritative.
- **PART II — Core Specification (retained in full).** The detailed engineering specification: data model, gates, storage, integrations, security, deployment, testing, phases, and rules for the implementing agent. It is amended in place where marked `> **Membrane amendment**`.

**Precedence:** if Part II conflicts with Part I, Part I wins. If something is unspecified, prefer the choice that is more inspectable, more local-first, and less coupled.

**Keywords:** MUST, MUST NOT, SHOULD, MAY are used as in RFC 2119.

**Architecture figure:** the canonical architecture figure is `docs/assets/membrane-architecture.png` (title: *"Membrane: Memory-Gated Architecture (LSTM-Inspired)"*). The repository MUST contain this figure, and the implementation MUST match it. Sections A2 and A3 are the textual form of that figure. If code and figure disagree, fix one of them in the same pull request.

### Table of contents

```text
PART I — ARCHITECTURE CHARTER
  A1  Thesis, positioning, naming
  A2  The Membrane Memory Cell (LSTM-inspired gated architecture)
  A3  Full system architecture
  A4  Memory Query Planner
  A5  Memory Router
  A6  Memory Substrates
  A7  Memory State and State Machine
  A8  Provenance Graph
  A9  Outcome Memory and outcome-aware utility
  A10 Strength, decay, reinforcement
  A11 Memory Replay
  A12 Compression and consolidation with provenance
  A13 Policies and the Policy Engine
  A14 LLM-free-first principle
  A15 MemoryQL and the policy DSL
  A16 Bitemporal memory, diff, snapshot, branch, rollback
  A17 Memory security: poisoning defense, trust tiers, quarantine
  A18 Explainability
  A19 Context assembly and token-budgeted recall
  A20 Multi-agent shared memory and access control
  A21 Entity registry and resolution
  A22 Memory health, doctor, calibration
  A23 Portability: Membrane Memory Format, import/export
  A24 Universal adapter protocol
  A25 Tiering and cost control
  A26 Deterministic testing and simulation
  A27 Research program
  A28 The SOTA charter
  A29 World-class README.md and documentation mandate
  A30 Scope control and release gating

PART II — CORE SPECIFICATION (sections 0–71, retained and amended)
```

---

# PART I — MEMBRANE ARCHITECTURE CHARTER

---

# A1. Thesis, Positioning, Naming

## A1.1 Thesis

> **Membrane is a programmable memory runtime that dynamically manages information across heterogeneous storage systems for AI applications.**

Memory is treated as a **computational resource that is managed**, not a document collection that is retrieved.

```text
Memory = state + policy + lifecycle + substrate
```

| Term | Meaning |
|---|---|
| **state** | A memory is a versioned, temporal, provenance-carrying state object, not a text chunk. |
| **policy** | What to write, retrieve, update, forget, consolidate is governed by inspectable, programmable policies (rules → ML → learned). |
| **lifecycle** | Every memory moves through an explicit state machine, and every transition emits an event. |
| **substrate** | Each kind of memory lives in the storage system best suited to it (SQL, KV, vector, graph, temporal, object), chosen by the planner and executed by the router. |

## A1.2 Why the name

A biological membrane decides **what enters, what leaves, what stays, and what is transported**. That maps directly onto the system:

| Membrane function | Membrane (software) |
|---|---|
| what enters | **write** gate |
| what leaves | **forget** gate |
| what stays / changes | **update** gate and the memory state machine |
| what is exposed | **read** gate |
| what is transported, and where | **query planner + router** over substrates |

The name must also work as a product verb: `memory.remember()`, `memory.recall()`.

## A1.3 Research thesis

> **AI memory should be managed like a computational resource, not retrieved like a document collection.**

Research question:

> **Can a database-independent memory runtime dynamically learn what information to retain, retrieve, consolidate, and forget while minimizing memory and context cost?**

## A1.4 Killer technical idea

> **Memory Query Planning + Memory Gating + Heterogeneous Memory Substrates.**

Traditional RAG assumes every question goes to vector search. Membrane decides **where the answer lives** (exact state, semantic memory, timeline, causal graph, or a hybrid), then decides **whether it should be exposed** and **how much of it**.

## A1.5 Positioning, naming, and distribution

Positioning narrative for the README (see A29):

```text
RAG retrieves documents.
Membrane manages knowledge over time.
```

Capability dimensions Membrane is designed to own (validated by MemoryBench, never merely asserted):

| Dimension | Chat history | Vector DB / RAG | Typical agent-memory library | **Membrane (design target)** |
|---|---|---|---|---|
| Works with no LLM and no API key | n/a | partly | rarely | **yes, by default** |
| Facts that change over time (supersession) | no | no | partly | **first-class, bitemporal** |
| Explains why a memory was written/read/forgotten | no | partial | rarely | **every decision** |
| Routes a query to the right store type | no | no | rarely | **query planner + router** |
| Learns utility from outcomes | no | no | rarely | **outcome-aware** |
| Forgetting is policy-driven and auditable | no | no | partly | **archive/supersede/expire/delete distinguished** |
| Storage-agnostic with a conformance suite | n/a | n/a | rarely | **yes** |

> The table states **design targets**. The README MUST NOT publish a competitor comparison it has not verified against the current versions of those projects, and MUST link to the benchmark methodology that backs every claim.

### Naming and distribution facts (verified 2026-10-04; re-verify before first release)

- PyPI `membrane` exists: an **unrelated, unmaintained** Flask argument parser (last upload 2019). Importing `membrane` could collide if both are installed. Consider a PEP 541 request for the name; until then document the collision.
- PyPI `membrane-ai` exists: an **unrelated, active** project (PII anonymization middleware). Do **not** use `membrane-ai` as the distribution name and avoid brand confusion with it.
- PyPI `membrane-memory`, `membrane-os`, `membranedb`, `membrane-mem` were **unclaimed** at verification time.
- npm `membrane` exists (unrelated).
- Decision: Python **distribution `membrane-memory`**, **import `membrane`**, **CLI `membrane`**. Reserve `membrane-memory` on PyPI (and a TypeScript package scope) **before** any public announcement.
- The GitHub org/repo name MUST be checked for availability by a human before the first public push.

`pip install membrane-memory`, then `from membrane import Memory`. The CLI and the import name stay `membrane`; only the PyPI distribution is `membrane-memory`.

---

# A2. The Membrane Memory Cell (LSTM-Inspired Gated Architecture)

This section is **normative** and corresponds to the top half of `docs/assets/membrane-architecture.png`.

## A2.1 Why an LSTM analogy

An LSTM carries a **cell state** through time and uses **gates** to control what is forgotten, what is written, and what is exposed. Membrane applies the same structure to *external, persistent* memory:

- a **memory state** `M_t` that persists across time and sessions,
- **gates** that decide, per step, what to forget, write, update, and expose,
- an additive/identity-preserving update path, so **information is not destroyed by default**,
- a read-out `R_t` (retrieved memories) analogous to the LSTM hidden state `h_t`.

The analogy is **architectural and mathematical**. It does **not** require neural networks in v0.x: Phase 1 gates are deterministic rule functions with the same interface that later learned gates will use.

## A2.2 The cell

```text
                    ┌───────────────────────── memory state rail ──────────────────────────┐
 M(t-1) ───────────▶(⊙)───────────────────────────────▶(⊕)─────────────────────────────────▶ M(t)
 (memory state)      ▲  f(t) ⊙ M(t-1)                    ▲ ▲   (updated memory state)
                     │                                    │ │
                     │                    i(t) ⊙ M~(t) ───┘ └─── u(t) ⊙ Δ(M(t-1), M~(t))
                     │                         ▲                     ▲
                     │                         │ (⊗)                 │
                  ┌──┴───┐   ┌───────┐    ┌────┴───┐   ┌────────┐  ┌─┴──────┐
                  │  σ   │   │   σ   │    │  tanh  │   │   σ    │  │   σ    │
                  │FORGET│   │ WRITE │    │CANDID. │   │ UPDATE │  │  READ  │
                  │ f(t) │   │ i(t)  │    │  M~(t) │   │  u(t)  │  │  o(t)  │
                  │expire│   │ store │    │  new   │   │ merge/ │  │retrieve│
                  │or del│   │  new  │    │  info  │   │ revise │  │/expose │
                  └──▲───┘   └───▲───┘    └────▲───┘   └───▲────┘  └───▲────┘
                     └───────────┴─────────────┴───────────┴───────────┘
                          inputs  z(t) = [ M(t-1), q(t), x(t), c(t) ]
                          query q(t) · new information x(t) · context c(t): user, task, time, environment
                          external signals s(t): feedback, outcome

 R(t) = o(t) ⊙ ψ(M(t), q(t))        ψ = query planner + router + substrate search   ──▶  R(t) retrieved memories
```

Reading rule (matches the figure): the **top rail** carries the memory state; the **forget gate** multiplies the old state; the **write gate** multiplies new candidate memory and adds it in; the **update gate** adds a merge/revise term computed from old state and candidate; the **read gate** multiplies the retrieval function output to produce `R_t`.

## A2.3 Equations

**Vanilla LSTM (reference):**

```text
f_t  = σ(W_f [h_{t-1}, x_t] + b_f)      forget gate
i_t  = σ(W_i [h_{t-1}, x_t] + b_i)      input gate
c̃_t  = tanh(W_c [h_{t-1}, x_t] + b_c)   candidate
c_t  = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t        cell state update
o_t  = σ(W_o [h_{t-1}, x_t] + b_o)      output gate
h_t  = o_t ⊙ tanh(c_t)                  hidden state
```

**Membrane (normative):**

```text
f_t  = σ(W_f [M_{t-1}, q_t, x_t, c_t] + b_f)        forget gate
i_t  = σ(W_i [M_{t-1}, q_t, x_t, c_t] + b_i)        write gate
M̃_t  = φ(W_c [M_{t-1}, q_t, x_t, c_t])              candidate memory
u_t  = σ(W_u [M_{t-1}, q_t, x_t, c_t] + b_u)        update gate
M_t  = f_t ⊙ M_{t-1} + i_t ⊙ M̃_t + u_t ⊙ Δ(M_{t-1}, M̃_t)
o_t  = σ(W_o [M_t, q_t, c_t] + b_o)                 read gate
R_t  = o_t ⊙ ψ(M_t, q_t)                            retrieved memories
```

## A2.4 Gate mapping: LSTM → Membrane

| LSTM | Symbol | Membrane | Role |
|---|---|---|---|
| Forget gate | `f_t` | **Forget gate** | expire / archive / delete irrelevant memory |
| Input gate | `i_t` | **Write gate** | store new memory |
| Candidate | `c̃_t` | **Candidate memory** `M̃_t` | new information proposed for storage |
| *(new)* | `u_t` | **Update gate** | merge / revise existing memory (supersession) |
| Output gate | `o_t` | **Read gate** | retrieve / expose relevant memory |

Symbols and operators (same as the figure): `σ` sigmoid gate with outputs in `[0,1]`; `tanh`/`φ` candidate activation with outputs in `[-1,1]` (in Membrane, a candidate-generation function that may be deterministic); `⊙` element-wise gating (the gate controls information flow); `⊕`/`+` combine memory content; arrows denote state updates over time.

## A2.5 Variables

| Symbol | Definition | Concrete type |
|---|---|---|
| `M_{t-1}` | memory state before the step (the set of active memory records and their strengths, within the namespace in scope) | `MemoryState` |
| `q_t` | query (optional; absent on pure-write steps) | `Query | None` |
| `x_t` | new information (optional; absent on pure-read steps) | `Observation | None` |
| `c_t` | context: user, task, time, environment, session, entities | `Context` |
| `s_t` | external signals: feedback, outcome (consumed by Utility, Outcome Feedback and learned gates) | `Signals` |
| `f_t` | forget scores in `[0,1]`, one per existing memory (1 = keep fully, 0 = forget fully) | `GateOutput` |
| `i_t` | write scores in `[0,1]`, one per candidate | `GateOutput` |
| `M̃_t` | candidate memories derived from `x_t` (via extractor `φ`) | `list[Candidate]` |
| `u_t` | update scores in `[0,1]`, one per (existing, candidate) pair that overlaps | `GateOutput` |
| `Δ(M_{t-1}, M̃_t)` | merge/revise operator: produces supersession/revision of overlapping memories | `Delta` |
| `o_t` | read scores in `[0,1]`, one per retrieved candidate | `GateOutput` |
| `ψ(M_t, q_t)` | retrieval function = **query planner + router + substrate search + scoring** (A4, A5, A6) | `Retriever` |
| `R_t` | the retrieved memories exposed to the agent | `RecallResult` |

## A2.6 Operational semantics: continuous gates → discrete lifecycle actions

Gate outputs are continuous scores in `[0,1]`. Memory operations are discrete. The **Policy Engine** (A13) converts scores into actions via thresholds and rules:

- `f_t[j]` below the forget threshold ⟹ candidate for **expire / archive / supersede / delete**. The policy chooses **which** of the four (they are not equivalent, see §11). Default is archive/supersede. **Nothing disappears silently**: a lifecycle event is always emitted.
- `i_t[k]` above the write threshold ⟹ the candidate is persisted (state `candidate → written → active`).
- `u_t[j,k]` above the update threshold ⟹ `Δ` is applied: the old memory becomes `superseded` (history preserved), or fields are merged into a new version.
- `o_t[r]` above the read threshold (and within the token/budget limits) ⟹ the memory enters `R_t`.

The M_t equation is semantic, over sets of records:

```text
M_t = Retain(M_{t-1}, f_t)            # ⊙ with forget gate (soft strength + hard actions)
    ∪ Persist(M̃_t, i_t)              # ⊙ with write gate
    ∪ Revise(Δ(M_{t-1}, M̃_t), u_t)   # ⊙ with update gate
```

## A2.7 The step function (reference pseudocode)

```python
class MembraneCell:
    def step(
        self,
        state: MemoryState,          # M_{t-1}
        query: Query | None,         # q_t
        observation: Observation | None,   # x_t
        context: Context,            # c_t
        signals: Signals | None = None,    # s_t
    ) -> StepResult:                 # (M_t, R_t, events, trace)
        z = GateInput(state, query, observation, context, signals)   # [M_{t-1}, q_t, x_t, c_t]

        f = self.forget_gate(z)                                      # f_t
        candidates = self.candidate_fn(z) if observation else []     # M̃_t = φ(...)
        i = self.write_gate(z, candidates)                           # i_t
        delta = self.delta_fn(state, candidates)                     # Δ(M_{t-1}, M̃_t)
        u = self.update_gate(z, candidates, delta)                   # u_t

        plan = self.policy.decide(f, i, u, state, candidates, delta) # scores -> actions
        new_state, events = self.apply(plan)                         # transactional commit (§55)

        recalled = []
        if query:
            o = None
            hits = self.retrieve(new_state, query, context)          # ψ: planner -> router -> substrates
            o = self.read_gate(GateInput(new_state, query, None, context, signals), hits)   # o_t
            recalled = self.policy.select_reads(hits, o, context)    # budget-aware (A19)

        trace = CellTrace(z, f, i, u, o, plan, recalled)             # feeds explain() (A18)
        return StepResult(new_state, RecallResult(recalled), events, trace)
```

`remember()`, `recall()`, `update()`, `forget()`, `consolidate()` are thin, typed wrappers around `step()` with masks (for example, `recall()` sets `observation=None` and disables write/update/forget in the plan).

## A2.8 Mandatory implementation requirements for the cell

1. There MUST be a single cell abstraction (`MembraneCell` or equivalent) whose structure mirrors the figure: five gate-shaped components (forget, write, candidate, update, read), a state, and a retrieval function `ψ`.
2. Every gate MUST implement one protocol (`Gate`) taking `GateInput` and returning `GateOutput` containing: scores in `[0,1]`, the feature vector used, human-readable reasons, `policy_id`, and `gate_version`.
3. Gate implementations MUST be swappable without changing callers: `RuleGate` (Phase 1), `LinearGate` / `GradientBoostedGate` (Phase 2), `NeuralGate` (Phase 3), `RLGate` (Phase 4). All learned gates keep the `[W, b]` conceptual interface but may be any function.
4. A gate MUST NOT perform storage I/O. Gates are pure functions of `GateInput` (plus injected clock and RNG seed). Storage happens in `apply()` and in `ψ`.
5. Hard safety rules (namespace isolation, never silently delete, provenance preserved, PII rules, trust tiers) are enforced **after** the gates by the Policy Engine and MUST override any gate score, learned or not.
6. Every `step()` MUST produce a `CellTrace` with all gate inputs and outputs, retained according to the retention policy, enabling `explain()` (A18).
7. `step()` MUST be deterministic given inputs, injected `Clock`, and seed.
8. Learned gates MUST be trainable offline from event logs and outcome signals (A27) without changing the runtime.

## A2.9 Gate evolution roadmap

```text
Phase 1   Rules                 deterministic scoring, configurable weights
Phase 2   Classical ML          logistic regression / gradient-boosted trees on gate features
Phase 3   Neural controller     small network, W and b learned (PyTorch, research extra)
Phase 4   Reinforcement /       outcome-optimized policies with offline policy evaluation
          outcome optimization
```

Objective for learned gates (A27):

```text
maximize   downstream_task_utility
         − λ · tokens_in_context
         − μ · storage_cost
         − ν · stale_or_wrong_exposure
subject to hard safety constraints (A2.8 item 5)
```

---

# A3. Full System Architecture

This section corresponds to the bottom half of `docs/assets/membrane-architecture.png` ("Full Membrane Architecture with LSTM-Style Memory Cell").

## A3.1 Layered architecture

```text
 INPUTS                      MEMBRANE MEMORY CELL                MEMORY ROUTER               MEMORY SUBSTRATES
                             (LSTM-style)                        (Query Planner)             (pluggable)

 ┌────────────────┐     ┌─────────────────────────────┐      ┌───────────────────┐      ┌───────────────────────────┐
 │ User query q(t)│     │ M(t-1) ─▶(⊙)─▶(⊕)─▶ M(t)     │      │ Structured Memory │─────▶│ PostgreSQL / SQLite       │
 │ New info x(t)  │ ──▶ │        ▲      ▲ ▲            │      │ (SQL / Relational)│      │ (MySQL, ...)              │
 │   (optional)   │     │  [σ f][σ i][tanh][σ u][σ o]  │ ───▶ ├───────────────────┤      ├───────────────────────────┤
 │ Context c(t)   │     │  Forget Write Cand. Update   │      │ Semantic Memory   │─────▶│ Qdrant / Milvus           │
 │ (task, time,   │     │  Read   ──▶ R(t)             │      │ (Vector DB)       │      │ (Vector DB)               │
 │  environment)  │     └─────────────────────────────┘      ├───────────────────┤      ├───────────────────────────┤
 │ External       │                                           │ Graph Memory      │─────▶│ Neo4j / TigerGraph        │
 │ signals s(t)   │                                           │ (Graph DB)        │      │ (Graph DB)                │
 │ (feedback,     │                                           ├───────────────────┤      ├───────────────────────────┤
 │  outcome)      │                                           │ Temporal Memory   │─────▶│ ClickHouse / Timescale    │
 └────────────────┘                                           │ (Time-series /    │      │ (Temporal DB)             │
                                                              │  Events)          │      ├───────────────────────────┤
                                                              ├───────────────────┤      │ Redis / DynamoDB          │
                                                              │ KV / State Memory │─────▶│ (KV Store)                │
                                                              │ (Redis, KV stores)│      ├───────────────────────────┤
                                                              └───────────────────┘      │ Custom Backend (your own) │
                                                                                         └───────────────────────────┘

 ┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ AUXILIARY MODULES  (operate with the same gates / signals)                                                         │
 │ Memory Consolidation (summarize/compress) · Memory Replay (periodic training) · Utility & Decay (importance        │
 │ scoring) · Provenance Graph (source & lineage) · Outcome Feedback (reinforce/penalize) · Policy Engine (rules/learned)│
 └───────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## A3.2 Broader runtime view

```text
                         AI APPLICATION
                               │
                               ▼
                    ┌─────────────────────┐
                    │      MEMBRANE       │
                    │   Memory Runtime    │
                    └──────────┬──────────┘
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          Memory Gates    Query Planner   Policy Engine
                └──────────────┼──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │    Memory Router    │
                    └──────────┬──────────┘
       ┌───────────────────────┼────────────────────────┐
       ▼                       ▼                        ▼
   Structured               Semantic                 Relational
    Memory                   Memory                    State
       │                       │                        │
      SQL                    Vector                    KV / Graph
       └───────────────────────┼────────────────────────┘
                               ▼
                     Temporal / Event Store
                               ▼
                     Memory State Graph (provenance + supersession + consolidation links)
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
          Consolidation      Replay        Reflection
                └──────────────┼──────────────┘
                               ▼
                        Updated Memory
```

## A3.3 Component responsibilities

| Component | Responsibility | Package path (see §39) |
|---|---|---|
| **Memory API** | `remember/recall/update/forget/consolidate/reflect/timeline/explain` public façade | `membrane/core/memory.py` |
| **Memory Controller** | wraps the cell; applies policies; owns transactions | `membrane/core/controller.py` |
| **Memory Cell** | the gated step function (A2) | `membrane/cell/` |
| **Gates** | forget, write, update, read, candidate | `membrane/gates/` |
| **Query Planner** | decides which substrate(s) and strategy answer a query | `membrane/planner/` |
| **Memory Router** | executes the plan, fans out, merges results | `membrane/router/` |
| **Substrates** | backends specialised per memory kind | `membrane/substrates/` |
| **Policy Engine** | thresholds, budgets, hard safety rules, presets | `membrane/policy/` |
| **Event Log** | append-only record of all transitions and decisions | `membrane/events/` |
| **Provenance Graph** | lineage of every memory | `membrane/provenance/` |
| **Consolidation / Replay / Reflection** | offline and online memory improvement | `membrane/consolidation/`, `membrane/replay/`, `membrane/reflection/` |
| **Utility & Decay** | strength, importance, reinforcement | `membrane/utility/` |
| **Outcome Feedback** | outcome capture and credit assignment | `membrane/outcomes/` |
| **Clock** | injectable time source for determinism | `membrane/core/clock.py` |

## A3.4 Canonical flows

**remember(x):** `Observation → φ (candidates) → write gate (i_t) → update gate (u_t)+Δ → forget gate (f_t) on affected memories → Policy Engine (hard rules, trust tier, PII) → transactional apply → events → substrates (via router) `.

**recall(q):** `Query → Query Planner (plan) → Router (fan-out) → substrates → merge → scoring (§8, §31) → read gate (o_t) → budget-aware selection (A19) → R_t + recall_id`; emits `memory.read`/`memory.recalled`.

**outcome(recall_id, outcome):** `Outcome → credit assignment over memories in recall → utility/strength update (A9, A10) → events`.

**replay():** `Event log → pattern detection → consolidation candidates → write/update gates → consolidated memory with provenance links` (A11, A12).

---

# A4. Memory Query Planner

The Query Planner is the most important systems addition. It answers: **where does the answer to this query live, and what is the cheapest correct way to get it?** It is analogous to a database query optimizer.

## A4.1 Principle

```text
Traditional RAG:   every question → embed → vector search
Membrane:          every question → plan → the right substrate(s) → merge → gate
```

## A4.2 Worked examples (normative behavior for the rule-based planner)

| Query | Detected intent | Plan | Notes |
|---|---|---|---|
| "What's yesterday's sales?" | current/at-time **structured state** | `StructuredLookup` (SQL/KV), exact entity + attribute + time | **No embedding, no vector DB, no LLM** |
| "What does this customer usually prefer?" | **semantic / preference** memory | `SemanticSearch` (vector or lexical fallback) filtered by `type=preference` | consolidated semantic memories ranked above raw episodes |
| "What changed about SKU 123?" | **temporal / event** memory | `TimelineScan(entity=SKU123, range)` over the event/temporal substrate | returns ordered state changes |
| "What caused sales to decline?" | **causal** memory | `CausalTraversal` over graph + analytical store, then `StructuredLookup` for evidence | uses stored cause→effect edges with confidence |
| "What was the price last month?" | **bitemporal state** | `StructuredLookup(at=…)` | see A16 |
| "Why do you believe X?" | **provenance** | `ProvenanceTrace(memory_id)` | answers from the provenance graph |

## A4.3 Planner interface

```python
class QueryPlanner(Protocol):
    def plan(self, query: Query, context: Context, capabilities: SubstrateCapabilities) -> QueryPlan: ...

@dataclass
class QueryPlan:
    plan_id: str
    intent: Intent                    # STATE | SEMANTIC | TEMPORAL | CAUSAL | PROVENANCE | HYBRID | NONE
    steps: list[PlanStep]             # ordered or parallel steps
    fallbacks: list[PlanStep]         # used under failure_mode=best_effort (§61)
    estimated_cost: Cost              # latency, tokens, $ if any
    rationale: dict[str, float|str]   # features that drove the decision (explainable)
    needs_memory: bool                # the read gate's "whether memory is needed" decision (§8 item 1)

@dataclass
class PlanStep:
    substrate: SubstrateKind          # SQL | KV | VECTOR | GRAPH | TEMPORAL | OBJECT | LEXICAL
    operation: str                    # e.g. "lookup", "knn", "traverse", "range_scan"
    params: dict
    merge: MergeStrategy              # union | intersect | rank_fusion (RRF) | first_hit
```

## A4.4 Planner levels (all behind the same interface)

1. **Rule planner (MUST, Phase 1–2):** deterministic intent detection from query shape, explicit parameters (`entity=`, `at=`, `type=`), entity registry hits (A21), and temporal expressions; no LLM.
2. **Cost-based planner (SHOULD, Phase 3):** chooses among equivalent plans using substrate statistics (cardinality estimates, latency histograms, token cost), like a DB optimizer.
3. **Learned planner (research, Phase 4+):** routing policy trained on query→substrate success logs (research direction R7).
4. **LLM-assisted planner (optional extra):** may propose a plan, but the plan MUST be validated against capabilities and passes through the same Policy Engine. An LLM never executes a plan directly.

## A4.5 Planner requirements

- Query intent classification MUST work without an LLM and without embeddings.
- If the planner has low confidence, it MUST produce a **hybrid plan** (fan-out + rank fusion) rather than guessing one substrate.
- Every plan MUST be inspectable: `memory.explain_plan(query)` returns the `QueryPlan` and rationale **without executing it** (dry run), and `recall(..., return_trace=True)` returns the executed plan.
- The planner MUST degrade gracefully when a substrate is unavailable (§61): use `fallbacks`.
- Planner behavior MUST be covered by a **planner conformance suite**: a table of (query, context, capabilities) → expected intent/plan.

---

# A5. Memory Router

The planner decides the strategy. The **router executes it**.

```text
Query
  │
  ▼
Query Planner ──▶ QueryPlan
  │
  ▼
Memory Router
  ├── SQL        (structured state)
  ├── KV         (current state, caches)
  ├── Vector     (semantic)
  ├── Graph      (relationships, causal)
  ├── Temporal   (events, time series)
  └── Hybrid     (fan-out + merge)
  │
  ▼
Backend(s) ──▶ candidates ──▶ scoring (§8/§31) ──▶ read gate (o_t) ──▶ R_t
```

## A5.1 Router requirements

- Parallel fan-out with per-substrate timeouts and cancellation.
- Result **merge strategies**: union, intersect, reciprocal rank fusion (RRF), score-normalized fusion, first-hit short-circuit.
- **Namespace enforcement at the router level** in addition to each store (defense in depth). A router that returns cross-tenant results is a critical bug (§27).
- Partial-failure semantics follow `failure_mode` (§61): `strict` raises, `best_effort` returns what succeeded plus a `degraded=True` flag and the reason, `disabled` returns empty.
- Writes: the router performs **multi-substrate writes** (for example, SQL row + vector index + graph edge + event) using an **outbox pattern** so that the primary store and secondary indexes converge even if a secondary substrate is down. Secondary indexes are *derived* data that can be rebuilt from the primary store and event log (`membrane reindex`).
- Idempotent (§54) and transactional where the substrate supports it (§55). Where cross-substrate atomicity is impossible, the contract is: **primary store + event are atomic; secondaries are eventually consistent and repairable**.
- The router collects **per-substrate statistics** (latency, hit rate, error rate) used by the cost-based planner.

---

# A6. Memory Substrates

## A6.1 Definition

A **Memory Substrate** is a backend optimized for a particular *kind* of memory. Do not call everything a "memory store". The core never depends on a specific database: it depends on **substrate capabilities**.

```text
                 Memory Substrates
                        │
       ┌────────────────┼─────────────────┐
       ▼                ▼                 ▼
   Key-Value          Vector             Graph
   current state      semantics        relationships
       │                │                 │
       └────────────────┼─────────────────┘
                        │
       ┌────────────────┼─────────────────┐
       ▼                ▼                 ▼
    Temporal           SQL              Object
    events/series    structured          blobs
```

## A6.2 Substrate kinds and example backends (pluggable)

| Substrate kind | Best for | Example backends |
|---|---|---|
| **Structured (SQL / relational)** | typed facts, entity state, metadata, event log | PostgreSQL, SQLite, MySQL, DuckDB |
| **Semantic (vector)** | similarity over text/embeddings | Qdrant, Milvus, LanceDB, Chroma, pgvector, OpenSearch/Elasticsearch |
| **Graph** | relationships, provenance, causal edges | Neo4j, TigerGraph, or SQLite/Postgres edge tables (default) |
| **Temporal (time-series / events)** | change history, time-bucketed series | ClickHouse, TimescaleDB, or the built-in event table (default) |
| **KV / State** | hot current state, caches, working memory | Redis, DynamoDB, or SQLite KV table (default) |
| **Object** | large payloads, documents, artifacts | S3 / Blob / GCS, local filesystem |
| **Custom** | anything else | user-defined backend via the substrate protocol |

The default install uses **SQLite for every kind** (structured, KV, graph edge tables, temporal events, lexical FTS5). No external service is needed.

## A6.3 Capability model

Substrates declare capabilities; the planner uses them; the conformance suite verifies them.

```python
class SubstrateCapabilities(Flag):
    EXACT_LOOKUP   = auto()
    RANGE_SCAN     = auto()
    FULL_TEXT      = auto()
    VECTOR_KNN     = auto()
    GRAPH_TRAVERSE = auto()
    TIME_TRAVEL    = auto()      # as-of queries
    TRANSACTIONS   = auto()
    TTL            = auto()
    ATOMIC_CAS     = auto()
    STREAMING      = auto()

class Substrate(Protocol):
    kind: SubstrateKind
    capabilities: SubstrateCapabilities
    def health(self) -> Health: ...
    def put(self, record, *, tx=None): ...
    def get(self, id, *, at=None): ...
    def query(self, step: PlanStep, scope: Scope) -> list[Hit]: ...
    def delete(self, id, *, mode: DeleteMode): ...       # delete | archive | expire
    def stats(self) -> SubstrateStats: ...
```

Rules:

- **Do not invent unsupported capabilities** (see §69 rule 18). If a backend lacks a capability, declare it absent; the planner routes around it or the composite substrate emulates it explicitly and documents the cost.
- Existing §15 protocols (`MemoryStore`, `VectorStore`, `GraphStore`, `KeyValueStore`, `EventStore`) are the *storage-facing* interfaces; `Substrate` is the *capability-facing* wrapper the router uses. A `CompositeSubstrate` composes several.
- The **same conformance suite** (§52) is parameterized by capabilities.

## A6.4 Substrate selection by memory type (default policy)

| Memory type | Primary substrate | Secondary index |
|---|---|---|
| Semantic | SQL (record) | vector + lexical |
| Episodic | temporal/event table (record) | vector + lexical, graph edges to entities |
| Working | KV (TTL) | none |
| Procedural | SQL | vector |
| Preference | SQL | vector (optional) |
| State | SQL/KV (current) + temporal (history) | none |
| Outcome | SQL + temporal | graph edges (decision → outcome) |
| Causal | graph | SQL for evidence |

---

# A7. Memory State and the Memory State Machine

Memory is **state**, not a document. Extend §4.1 and §6 as follows.

## A7.1 Memory state record (required fields)

```json
{
  "id": "m_123",
  "content": "User likes Python",
  "type": "preference",

  "state": "active",

  "confidence": 0.94,
  "importance": 0.71,
  "utility": 0.83,
  "strength": 0.66,

  "recorded_at": "...",
  "created_at": "...",
  "observed_at": "...",
  "valid_from": "...",
  "valid_until": null,
  "updated_at": "...",
  "last_accessed_at": "...",
  "expires_at": null,

  "source": { "type": "conversation", "session_id": "..." },
  "trust_tier": "user_asserted",
  "version": 3,
  "supersedes": null,
  "superseded_by": null,

  "namespace": { "tenant_id": "...", "user_id": "...", "agent_id": "...", "session_id": "..." },
  "structured": { "entity": "user_1", "attribute": "preferred_language", "value": "python" },
  "pii": [],
  "metadata": {},
  "embedding_reference": null
}
```

`strength`, `trust_tier`, `recorded_at`, `superseded_by`, `structured`, and `pii` are additions to §4.1 and are required.

## A7.2 State machine (normative)

```text
                         candidate
                             │
                          WRITE gate
              ┌──────────────┼───────────────┐
              │ rejected     │ accepted      │ low trust / suspicious
              ▼              ▼               ▼
         (discarded,      written       quarantined ──(review/approve)──▶ written
          event emitted)     │               │
                             ▼               └──(reject)──▶ rejected (event emitted)
                           active
                 ┌─────────┬─┴────────┬───────────────┐
              UPDATE     EXPIRE     CONSOLIDATE     FORGET (policy)
                 │         │            │               │
                 ▼         ▼            ▼               ▼
            superseded  expired   consolidated_into   archived ──(policy/erasure)──▶ forgotten
                 │         │            │
                 └─────────┴─────┬──────┘
                                 ▼
                              archived
```

Transition rules:

- Every transition MUST emit an event (`memory.created`, `memory.read`, `memory.updated`, `memory.superseded`, `memory.expired`, `memory.archived`, `memory.quarantined`, `memory.forgotten`, `memory.consolidated`, `memory.conflict`, `memory.recalled`).
- Illegal transitions MUST raise `InvalidTransitionError`; the state machine MUST be testable as a pure table.
- `forgotten` = content removed or irrecoverably masked **by policy or erasure request**, with a tombstone and audit event retained (§36). Default forget is **archive**, not destructive delete.
- `superseded` memories remain queryable via as-of queries (A16).

---

# A8. Provenance Graph

Provenance is a **graph**, not a single field. Extends §28.

## A8.1 Model

```text
                  Memory
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Dataset    Query     Analysis
          │         │         │
          ▼         ▼         ▼
      sales.csv    SQL     causal model
```

Nodes: `Memory`, `Source` (conversation, document, database, API, tool, agent, human, system), `Derivation` (query, analysis, model, consolidation, inference), `Decision`, `Outcome`.
Edges: `derived_from`, `supersedes`, `consolidates`, `cites`, `produced_by`, `caused`, `evaluated_by`.

## A8.2 API

```python
memory.provenance(memory_id)            # full lineage tree
memory.why(memory_id)                   # human-readable chain: "Why does the system believe this?"
memory.dependents(source_ref)           # everything derived from a source (used by cascading delete)
```

## A8.3 Requirements

- A developer MUST be able to ask **"Why does the system believe this?"** and receive the chain down to raw sources.
- Provenance MUST survive consolidation and compression (A12): a summary links to every source memory.
- Provenance MUST drive **cascading deletion** (§36): "delete source-linked memories" removes or marks derived memories according to an explicit, previewable cascade plan (`dry_run=True`).
- Provenance is stored in the graph substrate (default: edge tables in SQLite/Postgres).
- Export to standard lineage formats SHOULD be supported (PROV-JSON, OpenLineage events) as a plugin.

---

# A9. Outcome Memory and Outcome-Aware Utility

Do not only remember "Agent recommended X". Remember the **decision, expectation, action, and outcome**, then learn which memories actually mattered.

## A9.1 Outcome record

```text
Decision:   Run promotion X.
Expected:   +8% sales, +3% margin
Actual:     +11% sales, +1% margin

Decision → Action → Outcome → Memory update
```

```python
decision = memory.record_decision(
    "Run promotion X on SKU 123",
    expected={"sales_lift": 0.08, "margin_delta": 0.03},
    used_memories=recall_result.ids,        # which memories informed the decision
    namespace=ns,
)
memory.record_outcome(
    decision.id,
    actual={"sales_lift": 0.11, "margin_delta": 0.01},
    quality=memory.outcome_quality(expected, actual),   # pluggable scorer in [0,1] or [-1,1]
)
```

## A9.2 Feedback API

```python
memory.feedback(recall_id, outcome=+1)      # simple signal: success/failure/score
memory.feedback(recall_id, outcome=0.7, per_memory={"m_1": +1, "m_2": -1})   # optional credit hints
```

## A9.3 Outcome-aware utility

```text
Utility(m) = f( retrieval_frequency, task_success, decision_quality, outcome, recency_of_evidence )
```

Phase 1–2 estimator (no neural network required):

- Maintain per-memory **Beta(α, β)** posterior over "was useful when retrieved". Success adds to α, failure to β. `utility = α / (α + β)` with a prior; expose credible interval for explain().
- **Credit assignment:** distribute outcome credit across the memories in the recall by their read score `o_t` (and by position-aware attribution if available), with a configurable discount so that low-scored memories are not over-credited.
- Optionally bandit-style exploration (Thompson sampling) in the read gate to avoid never retrieving low-evidence memories (gated by a policy flag; deterministic mode uses the posterior mean).

## A9.4 Outcome memory type

Add memory type `outcome` (and `decision`) to §5 (see amendment there). Outcomes are linked to the memories used (provenance edge `evaluated_by`) and are first-class in MemoryBench (A27/§34).

---

# A10. Memory Strength, Decay, and Reinforcement

Biologically inspired dynamic strength so that **useful memories get stronger and useless memories decay**, instead of "newest wins".

## A10.1 Strength

```text
S_i(t) = I_i · C_i · U_i · exp(−λ_i · Δt_i)

I_i = importance      C_i = confidence      U_i = utility
λ_i = decay rate (per memory type / per memory, configurable)
Δt_i = time since last reinforcement (not merely creation)
```

## A10.2 Reinforcement

```text
S_i(t+1) = S_i(t) + α · outcome_i · (1 − S_i(t))        # bounded, saturating
```

Retrieval followed by success reinforces; retrieval followed by failure weakens (negative `outcome`). `α` is configurable and may depend on memory type.

## A10.3 Rules

- `S_i` MUST be normalized to `[0, 1]` (consistent with §31).
- Decay rates are **per type defaults**, overridable: working memory decays fastest; state memory is *not* decayed by age but invalidated by supersession; procedural memory decays slowly and is reinforced by successful use.
- `S_i` feeds the **forget gate** (low strength ⟹ candidate for archive/expire) and the **read score** (§8).
- Decay is computed **lazily at read time** and **materialized by replay/maintenance jobs** (A11), never by a hot-path scan of all memories.
- Spaced-repetition-style reinforcement MAY be added as a research option.
- Ablate it: MemoryBench MUST include a configuration with decay/reinforcement disabled (A27).

---

# A11. Memory Replay

Inspired by biological memory consolidation during replay, but implemented as an **offline, scheduled, budgeted** process, so the hot path stays cheap.

```text
experience → replay → consolidation

Memory Log (events)
     │
     ▼
   Replay
     │
     ▼
Pattern detection  (repeated entities, co-occurrences, recurring sequences, contradictions)
     │
     ▼
Consolidation  (A12)
     │
     ▼
Long-term memory
```

## A11.1 Requirements

- `memory.replay(scope=None, budget=None, since=None)` and `membrane replay`.
- Replay reads from the **event log and memory store**, never from live traffic paths.
- Replay jobs MUST be **resumable, idempotent, rate-limited, and budget-bounded** (time, memory count, optional LLM tokens).
- Replay also performs: materializing decay/strength (A10), recomputing utility from outcome logs (A9), detecting stale and contradicted memories, proposing consolidations, rebuilding secondary indexes if drifted.
- Replay MUST NOT modify memory except through the normal gates + Policy Engine + events.
- Replay can run: on a schedule (`consolidation_interval="1h"`), on a trigger (N new memories), or manually.
- Replay is also the training-data generator for learned gates (A27): it can emit `(state, action, outcome)` trajectories.

---

# A12. Memory Compression and Consolidation With Provenance

First-class, as extension of §12.

```text
10,000 episodic memories
          ↓  clustering
          ↓  summarization
 1,000 semantic memories
          ↓
        index
```

## A12.1 Rules

- Compression MUST preserve traceability:

```text
summary
  ├── source 1
  ├── source 2
  ├── source 3
  └── source 4
```

- Required fields on a consolidated memory (extends §12): `consolidated_memory_id`, `source_memory_ids`, `consolidation_method` (`rule_cluster`, `frequency`, `llm_summary`, `learned`), `confidence`, `created_at`, `compression_ratio`, `information_loss_estimate` (optional), `reversible` (can the sources be recovered? default yes since sources are archived, not deleted).
- Source memories become `archived` (state `consolidated_into`) with a link, never silently removed.
- Deterministic consolidation methods (no LLM): frequency/recurrence detection (for example "N of the last M purchases are organic"), entity-attribute aggregation, time-window roll-ups (daily → weekly → monthly), numeric summary statistics for state/time-series memories.
- Optional LLM summarization is allowed behind the write gate; the produced summary is a *candidate* and is validated (provenance present, no new named entities not in sources, confidence calibrated).
- Report **Memory Utility per Token** and **compression ratio** (A27) for each consolidation method.

---

# A13. Memory Policies and the Policy Engine

Memory behavior is **programmable** via policies. The Policy Engine converts gate scores into actions and enforces hard rules.

## A13.1 Using policies

```python
memory = Memory(policy="balanced")          # presets: "conservative", "balanced", "aggressive", "strict_audit"

memory = Memory(
    policy=MemoryPolicy(
        max_tokens=5000,
        retention_days=90,
        min_confidence=0.7,
        consolidation_interval="1h",
    )
)

class MyPolicy(MemoryPolicy):
    def should_write(self, candidate, gate_output, context) -> Decision: ...
    def should_forget(self, memory, gate_output, context) -> Decision: ...
    def should_update(self, existing, candidate, gate_output, context) -> Decision: ...
    def select_reads(self, hits, gate_output, context) -> list[Hit]: ...
    def route(self, query, context) -> QueryPlan | None: ...    # optional planner override
```

## A13.2 Presets

| Preset | Behavior |
|---|---|
| `conservative` (default, §37) | writes only high-confidence, user-asserted or explicit facts; slow decay; archive over everything |
| `balanced` | thresholded write on novelty+importance+confidence; moderate decay; periodic consolidation |
| `aggressive` | broader writes, fast decay, strong compression, tight token budgets |
| `strict_audit` | everything logged, no destructive operations, mandatory provenance, full trace retention |

## A13.3 Engine duties

1. Convert continuous gate scores to actions (thresholds, top-k, budgets).
2. Enforce **hard safety rules after the gates** (A2.8 item 5): namespace isolation, never-silent-delete, provenance required, trust tiers (A17), PII handling, retention/legal hold.
3. Resolve gate conflicts (for example write says store, contradiction detector says conflict: apply update path or quarantine, per policy).
4. **Shadow mode / counterfactual evaluation:** run policy B alongside the active policy A on the same inputs and log what B *would* have done without applying it (`policy_shadow="B"`). This yields offline A/B data and is the basis for safe policy rollout and learned-gate evaluation.
5. Versioned and hot-reloadable: every decision records `policy_id` and `policy_version`.
6. Policies are declared in code, in YAML (A15), or loaded from a plugin; all validated at load time.

---

# A14. LLM-Free-First Principle

This is a prominent differentiator and MUST be advertised in the README.

```bash
pip install membrane-memory
```

```python
from membrane import Memory

memory = Memory()
memory.remember("The user prefers Python over JavaScript.", user_id="u1")
memory.recall("What does this user prefer?", user_id="u1")
```

No OpenAI, no Anthropic, no Gemini, no Ollama, no API key, no Docker, no external database.

```text
                Membrane
                   │
          ┌────────┴─────────┐
          ▼                  ▼
      LLM-free           LLM-assisted
        mode                 mode
          │                    │
   rules / classical ML    extraction (φ)
   SQL / KV / lexical      reflection
   optional embeddings     consolidation / summarization
                           contradiction interpretation
```

Opt in explicitly: `Memory(extractor="openai")`, `Memory(llm=my_llm)`.

Rules:

- Every major capability has three implementations: **deterministic**, **embedding-assisted**, **LLM-assisted**. Tests run all three tiers where possible; CI MUST run the full suite with **no network access** and with the `llm=none` configuration.
- An LLM MUST NEVER write directly to storage. It proposes candidates; the write gate and Policy Engine decide (§45).
- No data leaves the process unless an external provider is explicitly configured (§69 rule 16). Record each external call as an audit event (provider, purpose, bytes, redaction status).
- A built-in **lexical engine** (SQLite FTS5 / BM25) and a built-in **tiny local embedding option** make "no LLM" still useful. Embeddings are optional (§14).
- MemoryBench reports an **LLM-free vs LLM-assisted** comparison (A27, research direction R8).

---

# A15. MemoryQL and the Policy DSL

A small, typed, declarative language for AI memory. It is a staged deliverable (Phase 3+), not a v0.1 blocker, but the internal query model (`Query`, `Filter`, `Order`, `Limit`, `At`) MUST be designed from day one so that MemoryQL is a thin parser over it.

## A15.1 MemoryQL (query)

```sql
SELECT memories
FROM user_memory
WHERE entity = 'customer_123'
  AND type = 'preference'
  AND confidence > 0.8
  AND valid_at = NOW()
ORDER BY utility DESC
LIMIT 10
WITH EXPLAIN;
```

Additional forms:

```sql
RECALL 'what does this customer prefer?' IN user_memory WITH BUDGET 800 TOKENS;
TIMELINE entity = 'product_123' FROM '2026-01-01' TO '2026-12-31';
AS OF '2026-09-01' SELECT memories WHERE entity = 'product_123' AND attribute = 'price';
DIFF user_memory BETWEEN '2026-09-01' AND '2026-10-01';
EXPLAIN WHY memory 'm_123';
```

Requirements: parser produces an AST → `Query` object → planner; no string-interpolated SQL ever reaches a database (injection-safe by construction); errors are precise (line/column).

## A15.2 Policy DSL (YAML)

```yaml
policy:
  name: support_agent_v1
  extends: balanced
  rules:
    - match: { type: preference, scope: user }
      retention: long
      storage: semantic
      write_if: { confidence: ">0.8" }
    - match: { type: working }
      ttl: 2h
    - match: { source.trust_tier: untrusted }
      action: quarantine
  budgets:
    max_context_tokens: 1200
    max_memories_per_recall: 8
  consolidation:
    interval: 1h
    min_cluster_size: 4
```

Python equivalent:

```python
memory.policy(type="preference", retention="long", storage="semantic", confidence_threshold=0.8)
```

Policies are validated against a JSON Schema published with the package.

---

# A16. Bitemporal Memory, Diff, Snapshot, Branch, Rollback

## A16.1 Bitemporal model (extends §29)

Two independent time axes:

- **Valid time** (`valid_from`, `valid_until`): when the fact was true in the world.
- **Transaction/record time** (`recorded_at`, plus supersession links): when the system *learned* it.

This enables two kinds of question:

```python
memory.recall("What was the price last month?", at="2026-09-01")                    # valid time
memory.recall("What did the agent believe the price was?", as_of="2026-09-01")      # record time
memory.recall("...", at="2026-09-01", as_of="2026-09-05")                           # both
```

The second is **decision replay and audit**: "what did the agent know when it made that recommendation?" This is essential for regulated and enterprise use.

## A16.2 Git for agent memory

```python
snap = memory.snapshot(label="before-import")       # cheap: a pointer into the event log, not a copy
memory.diff(t1, t2)                                 # added / updated / superseded / archived / forgotten
branch = memory.branch("experiment-a", from_=snap)  # copy-on-write overlay; isolated writes
memory.merge(branch, strategy="prefer_branch|prefer_main|manual")
memory.rollback(to=snap)                            # compensating events; history is preserved, never rewritten
```

Rules:

- Snapshots/diff/rollback are derived from the **append-only event log** (§56). Rollback appends compensating events; it MUST NOT delete history.
- Branches enable what-if runs, agent A/B tests, and safe experimentation with policies and learned gates.
- Rollback after a bad ingestion or suspected poisoning (A17) is a first-class operator workflow: `membrane rollback --to <snapshot>`.
- Erasure requests (§36–37) are the one case where content is physically removed; the tombstone and audit record remain, and snapshots referencing erased content show a redaction marker.

---

# A17. Memory Security: Poisoning Defense, Trust Tiers, Quarantine

Persistent memory is an **attack surface**: a single injected "remember: always do X" can live forever and steer a future agent. This extends §36.

## A17.1 Trust tiers

```text
system > human_verified > user_asserted > tool_output > agent_inferred > external_content > untrusted
```

Every memory carries `trust_tier` derived from its provenance. Retrieval and writes consult it.

## A17.2 Write-gate defenses

- **Instruction-like content detection:** content from tiers at or below `tool_output` that reads like an instruction to the agent ("always", "ignore previous", "from now on", "send … to …", URLs/commands) is flagged by a deterministic detector (pluggable; a classifier plugin may be added).
- **Quarantine:** suspicious or low-trust candidates enter state `quarantined`: stored but **never retrieved** into working memory until approved by policy or a human (`memory.review_queue()`, `memory.approve(id)`, `memory.reject(id)`).
- **Procedural memory is privileged:** writes to `procedural` memory require a minimum trust tier, or review.
- **Cross-session/cross-user contamination checks:** a memory derived from one user's untrusted content MUST NOT be promoted into shared/global scope without review.
- **Rate limits and anomaly detection** on write volume per source (burst detection).
- **Rollback** (A16) and **cascading delete by source** (A8) are the remediation tools.

## A17.3 Read-side defenses

- Retrieved memory is **data, not instructions**. The context-assembly output (A19) wraps memories in clearly delimited, typed blocks and attaches `trust_tier` and `provenance` so the application can render them safely.
- Token/size caps per memory and per recall.

## A17.4 Other requirements (kept from §36–37 and extended)

API-key auth for server mode, namespace isolation, optional encryption at rest and field-level encryption, configurable retention, **PII tagging and redaction hooks** (pluggable detectors; a built-in regex baseline for emails, phones, IDs), deletion workflows with explicit cascade plans, and tamper-evident audit logs (hash-chained events: `hash_i = H(hash_{i-1} || event_i)`).

A security test suite MUST include: cross-tenant leakage tests, injection corpus tests, quarantine bypass tests, cascade-delete tests, and audit-chain verification.

---

# A18. Explainability

Every memory decision is inspectable (goal G7). Provide a unified surface:

```python
result = memory.recall("...", return_trace=True)
memory.explain(result.recall_id)           # why these memories, why not others, which features mattered
memory.explain(memory_id)                  # lifecycle: why written / updated / superseded / forgotten
memory.explain_plan("...")                 # planner decision without executing
memory.why(memory_id)                      # provenance chain (A8)
memory.explain_gate("forget", memory_id)   # last forget-gate evaluation with feature contributions
```

`explain()` returns structured data **and** a human-readable rendering, including:

- per-memory score breakdown (`semantic`, `lexical`, `entity`, `temporal`, `freshness`, `confidence`, `utility`, `access_history`, `strength`),
- gate scores `f_t, i_t, u_t, o_t` and the thresholds/policy that converted them to actions,
- the query plan and which substrates were used,
- near-misses (top excluded candidates and the reason each lost),
- counterfactual: "what would have changed under policy B" when shadow mode is on (A13).

CLI and the web demo render the same data (score bars, plan graph, timeline).

---

# A19. Context Assembly and Token-Budgeted Recall

Agent developers care about tokens at least as much as recall@k.

```python
ctx = memory.recall(query, user_id="u1", token_budget=800, format="prompt")
ctx.text         # ready-to-inject block
ctx.memories     # structured list
ctx.tokens_used
ctx.dropped      # what did not fit, and why
```

Requirements:

- **Budget-aware selection:** choose the set maximizing expected utility subject to the token budget (greedy by `score / tokens` with diversity penalty, MMR-style, to avoid redundant memories).
- **Auto-compression:** lower-ranked items may be replaced by their consolidated summary (A12) when it fits and utility per token is higher.
- **Dedup and conflict-collapse:** never inject two contradictory *current* facts; inject the current one and, if useful, a marked superseded note.
- **Ordering:** most important items placed at the start and end of the block (mitigates long-context "lost in the middle" effects); configurable.
- **Token counting is pluggable** (`TokenCounter` protocol; a dependency-free approximate default; exact tokenizers via extras).
- **Formats:** `prompt` (text with typed delimiters), `messages` (chat role messages), `json`, `xml`.
- Output labels memory type, time validity, trust tier, and confidence so downstream prompts can reason about staleness.
- This subsumes "working memory" assembly: the working-memory builder (§5.3) is this component.

---

# A20. Multi-Agent Shared Memory and Access Control

Many real systems have several agents and users sharing memory.

- **Scopes:** `private` (single agent/user), `shared` (explicit group), `tenant`, `global`. Scope is enforced at store level and router level (§27).
- **Access-control hook:** `AccessPolicy.check(principal, action, memory|namespace) -> Allow|Deny|Redact`, invoked on every read/write/forget/export. Default: namespace-based RBAC. Pluggable for ABAC/OPA.
- **Sharing contracts:** an agent can *publish* a memory (or consolidated summary) into a shared namespace; the publish operation preserves provenance and trust tier, and lowers trust unless re-verified.
- **Concurrent writers:** optimistic concurrency via `version` (compare-and-swap), idempotency keys (§54), and deterministic conflict resolution policy (`last_writer_wins`, `highest_trust`, `highest_confidence`, `manual`) recorded as `memory.conflict` events.
- **Multi-replica/CRDT-friendly design (research/optional):** state memories SHOULD be modeled so that merges are well defined (per-field LWW registers with hybrid logical clocks; set-like semantics for episodic memory), enabling future offline-first and edge deployments.
- **Agent identity:** `agent_id` is first-class; every event records the acting principal.

---

# A21. Entity Registry and Resolution

Structured and temporal memory depend on stable entities.

- A lightweight **entity registry** (`entity_id`, type, canonical name, aliases, attributes) lives in the structured substrate.
- **Deterministic entity resolution:** exact/alias match, normalized-string match, fuzzy match with threshold, user-provided resolvers. Embedding-assisted and LLM-assisted resolvers are optional plugins.
- Used by: the planner (intent and entity detection), the read score (`entity_match`), contradiction detection (§30 needs `entity` + `attribute`), the timeline API, the graph substrate, and the retail example (SKU, store, region hierarchies).
- Supports **hierarchies and relationships** (SKU → category L5, store → region West) as first-class edges.
- Merge/split of entities is an event with provenance and is reversible.

---

# A22. Memory Health, `doctor`, and Calibration

Memory needs maintenance and observability beyond logs.

- `membrane doctor` / `memory.health()` reports: stale memories, duplicates/near-duplicates, unresolved contradictions, orphaned provenance, derived-index drift (vector/graph vs primary), quarantine backlog, namespace leaks (sampled), growth rate, retrieval hit-rate, dead memories (never retrieved), memories with high retrieval but poor outcomes.
- A **Memory Health Score** (0–100, documented formula) is exposed as a metric and in the web demo.
- **Confidence calibration:** track whether `confidence` predicts correctness, using outcome and human-verification feedback. Report **expected calibration error (ECE)** per source/trust tier; allow a calibration map (isotonic/Platt) as a plugin so confidence values become meaningful probabilities rather than guesses.
- **Budgets:** per-namespace limits (count, bytes, tokens); behavior on exceeding (compress, archive lowest strength, reject) is policy-defined.
- `membrane doctor --fix` performs only safe, evented, reversible repairs (rebuild indexes, mark stale), never destructive actions.

---

# A23. Portability: Membrane Memory Format, Import and Export

Low switching cost is an adoption lever, and the data model is meant to be portable (G6).

- **Membrane Memory Format (MMF):** an open, versioned, documented export/import format: JSONL records + manifest + JSON Schema, covering memories, events, provenance edges, entities, policies, and (optionally) embeddings with the embedder id and dimension.
- `memory.export(path, namespace=..., include_history=True, include_embeddings=False)` and `memory.import_(path, mode="merge|replace|dry_run")`.
- **Importers (plugins):** conversation exports and other agent-memory systems; start with a generic JSONL/CSV importer and one or two well-documented sources; importers go through the write gate and mark `source.type=import` with provenance.
- Round-trip test: export → import into a different backend → identical recall results (a conformance requirement for substrates).
- Versioned schema with migrations (G5, §50).

---

# A24. Universal Adapter Protocol

Framework integrations (§20–25) MUST be thin and generated from one contract, so "switch agent framework without changing memory semantics".

```python
class AgentMemoryAdapter(Protocol):
    def before_model_call(self, request, ctx) -> request: ...      # recall + inject (A19)
    def after_model_call(self, request, response, ctx) -> None: ... # optional remember/extract
    def on_tool_result(self, tool, result, ctx) -> None: ...
    def on_outcome(self, run_id, outcome, ctx) -> None: ...         # feedback (A9)
```

- Adapters MUST not import each other, MUST live under `integrations/`, MUST be optional extras, and MUST pass a shared **adapter contract test** (same scenario, same expected memory semantics across LangGraph, LangChain, OpenAI, Anthropic, MCP, generic).
- Adapters MUST preserve the principle: the framework controls the loop; Membrane controls memory (§46, §60).
- Provide a **decorator/middleware** form for incremental adoption (G8): `@membrane.remembers(...)`, plus context-manager and callback forms.
- TypeScript client: generated from the OpenAPI schema (REST, §26); a TypeScript client MAY ship after v0.5 and MUST NOT delay Python releases.

---

# A25. Tiering and Cost Control

Memory has storage, compute, and context costs.

- **Hot / warm / cold tiering:** hot = KV/cache + recent active; warm = primary store; cold = archived, optionally in object storage in MMF. Tier moves are evented and policy-driven (by strength, access recency, type).
- **Cost model:** the router/planner track estimated and actual cost per operation (latency, tokens, optional $ per external call). Expose `memory.cost_report()`.
- **Lazy embeddings:** embed on demand or in background; never block `remember()` on embedding unless configured; embeddings are derived data (rebuildable).
- **Compaction** of event log (snapshots + truncation) with audit-preserving checkpoints, never losing the hash chain.
- Targets are tracked in benchmarks (§53, A27), including storage growth over interactions.

---

# A26. Deterministic Testing and Simulation

Temporal and lifecycle systems are untestable without control of time and randomness.

- **`Clock` protocol** (system clock default; `FrozenClock`, `SteppedClock`, `SimulatedClock` for tests); no direct `datetime.now()` in core code (enforced by a lint rule).
- **Seeded RNG** injected everywhere randomness is used.
- **Memory trajectory simulator** (`membrane.sim`): generates synthetic multi-session interaction streams with controllable properties (fact drift rate, contradiction rate, noise, repetition, adversarial injections, outcome feedback) and ground-truth answers. This is the engine behind MemoryBench's synthetic tasks and for training learned gates.
- **Property-based tests (Hypothesis):** e.g. "superseded memories never appear in current-state recall", "namespace A never returns namespace B", "every state transition has an event", "export→import is lossless", "rollback restores recall behavior".
- **Golden/snapshot tests** for planner decisions and `explain()` output.
- **Mutation testing** on the state machine and policy engine (critical lifecycle paths: 100%).
- **Fault injection:** kill a secondary substrate mid-write; verify outbox repair and `failure_mode` behavior.
- **Fuzzing** of MemoryQL and the policy DSL parsers.

---

# A27. Research Program

Membrane is designed to produce a credible research artifact alongside the library.

## A27.1 Research questions

1. Can memory gates reduce irrelevant retrieval?
2. Can temporal state reduce stale answers?
3. Can memory routing select the optimal storage substrate?
4. Can consolidation reduce memory footprint without hurting accuracy?
5. Can outcome feedback improve memory utility?
6. Can the system operate without an LLM, and how much does an LLM add?
7. Can learned policies outperform heuristic policies?

Plus the original directions R1–R8 (§59), which remain in force. Additional directions:

- **R9 — Calibrated memory confidence:** do calibrated confidences improve gating?
- **R10 — Memory security:** poisoning attack/defense benchmark for persistent memory.
- **R11 — Counterfactual / offline policy evaluation** for memory policies using shadow-mode logs (importance sampling, doubly robust estimators).
- **R12 — Bitemporal reasoning** for agents: decision replay accuracy.

## A27.2 Metrics (extends §34)

All of §34's metrics, and:

```text
Memory Utility per Token (MUT) = downstream_task_utility / memory_tokens_retrieved
Stale Retrieval Rate · Contradiction Rate · Temporal Accuracy (valid-time and as-of)
Routing Accuracy (planner chose a substrate that contains the answer)
Plan Efficiency (cost of chosen plan vs oracle plan)
Decision Quality Uplift (outcome-aware vs not)
Poisoning Success Rate (attack success against defenses)
Calibration (ECE) · Consolidation Ratio · Information Retention after consolidation
Storage Growth · p50/p95/p99 latency per operation at 1K–10M memories
```

MemoryBench dimensions (map to the figure in the project plan):

```text
MemoryBench
   ├── Retrieval      accuracy
   ├── Evolution      temporal consistency, supersession, contradiction
   ├── Decisions      outcome quality, decision replay
   ├── Forgetting     stale/irrelevant handling
   ├── Consolidation  pattern formation, compression fidelity
   ├── Security       poisoning, leakage
   └── Efficiency     tokens, latency, storage
```

## A27.3 Baselines and ablations

Compare at least: vanilla RAG; RAG + reranker; long-context stuffing (full history); naive summary memory; popular open agent-memory libraries (pin exact versions and configs in the benchmark card); Membrane rule-based; Membrane with classical-ML gates; Membrane with learned gates; Membrane with LLM-assisted extraction.

Ablate each component independently: planner off (always vector), router fan-out off, each gate off, decay off, reinforcement off, outcome utility off, consolidation off, replay off, provenance off, trust tiers off, bitemporal off.

## A27.4 Public benchmarks plus own benchmark

- Run on established long-term-memory/conversational-memory benchmarks (for example LoCoMo, LongMemEval, and their successors available at implementation time; verify current versions and licenses) so results are comparable.
- Add **MemoryBench** for what they miss: state evolution, bitemporal queries, outcome-aware decisions, routing, forgetting, poisoning, token efficiency.
- Datasets and runners are versioned and reproducible: fixed seeds, pinned dependencies, one command (`membrane benchmark --suite <name>`), results as machine-readable JSON plus a generated **benchmark card** (hardware, versions, configs, confidence intervals).

## A27.5 Paper plan

Working title: *"Membrane: Memory as a Managed Resource — Gated, Planned, and Routed Memory for AI Systems."*
Contributions: (1) the gated memory cell formulation with a deterministic-to-learned continuum; (2) memory query planning and routing across heterogeneous substrates; (3) bitemporal, provenance-preserving state memory with outcome-aware utility; (4) MemoryBench; (5) open-source reference implementation and conformance suite. Report negative results honestly. Pre-register ablations in `research/experiments/`.

## A27.6 Repository layout for research (extends §33)

```text
research/
  memory_controller/     learned gates, training code (PyTorch extra)
  benchmarks/            MemoryBench definitions
  datasets/              generators and loaders (no private data)
  ablations/             configs for each ablation
  experiments/           pre-registered experiment specs + results
  paper/                 LaTeX sources, figures generated from results
```

---

# A28. The SOTA Charter

"SOTA" is a claim that must be **earned and demonstrated**, not asserted. This project defines state of the art along measurable axes and gates releases on them.

## A28.1 What SOTA means here

1. **Accuracy:** matches or beats the strongest reproducible baselines on public long-term-memory benchmarks and on MemoryBench at equal or lower token cost.
2. **Efficiency:** best-in-class **Memory Utility per Token**, lowest stale-retrieval rate, and competitive latency (A27, §53).
3. **Correctness over time:** near-perfect temporal accuracy and zero silent history loss (verified by property tests).
4. **Trust:** every decision explainable; safety properties verified (A17).
5. **Engineering:** storage conformance suite adopted by third-party adapters; stable typed API; production deployment references.
6. **Usability:** working in under five minutes with no keys or services (G4).

## A28.2 Claims discipline

- No README claim without a linked, reproducible benchmark card.
- Benchmarks run in CI on a schedule; regression > configured tolerance fails the build.
- Publish methodology, seeds, hardware, and failure cases. Include where Membrane loses.
- Never tune on test splits; keep a held-out MemoryBench split private until release.

## A28.3 Engineering quality bar

Type-checked (strict mypy/pyright), Ruff, ≥ 90% coverage with 100% on critical lifecycle paths, mutation testing on state machine/policy engine, property-based tests, fuzzed parsers, SBOM and dependency scanning, signed releases, reproducible builds, semantic versioning, deprecation policy, documented performance targets (§53) benchmarked at 1K–10M memories, Python 3.10–3.13 matrix, Linux/macOS/Windows CI, offline CI job (no network).

## A28.4 Release gates

A version may be tagged only if: the full test suite passes offline; the adapter and substrate conformance suites pass; the README code blocks execute in CI (A29); benchmark cards are regenerated; docs build with no warnings; changelog and migration notes exist.

---

# A29. World-Class README.md and Documentation Mandate

The implementing agent MUST write a world-class `README.md` as a first-class deliverable, not an afterthought. The README is the project's front page, pitch, and onboarding.

## A29.1 README structure (in this order)

1. **Hero:** project name + logo, tagline (*"The programmable memory layer for AI."*), a one-sentence description, and badges (CI, coverage, PyPI version, Python versions, license, docs, benchmark status, discord/discussions). No more than ~8 badges.
2. **Demo GIF/video** (≤ 15 s) directly under the hero: the web demo showing remember → ask → contradiction → supersession → explain → forget (§65). Provide a static fallback image.
3. **The 60-second explanation:**
   ```text
   RAG retrieves documents.
   Membrane manages knowledge over time.

   Remember · Retrieve · Update · Forget · Consolidate · Reflect
   ```
   plus the one-paragraph thesis and the architecture figure `docs/assets/membrane-architecture.png`.
4. **Why Membrane:** 5–6 sharp bullets (LLM-free-first, local-first, facts that change over time, explainable, routes to the right store, learns from outcomes, framework- and storage-agnostic).
5. **Quickstart (under 5 minutes):** `pip install membrane-memory`, a 10-line script that works with no API key, expected output shown.
6. **See it think:** a short `explain()` / `explain_plan()` output sample showing score breakdowns.
7. **Core concepts in one screen:** the memory cell diagram (gates), memory states, substrates, planner/router — each a two-line description with link to docs.
8. **Integrations:** minimal copy-paste snippets for LangGraph, LangChain, OpenAI SDK, Anthropic SDK, MCP (one-line `uvx` or equivalent start), and custom agent. Each ≤ 15 lines.
9. **Storage:** matrix of substrates and backends (SQLite → Postgres → pgvector/Qdrant → hybrid) and how to switch with `Memory.from_config("postgres.yaml")` without code changes.
10. **Beyond chatbots:** the retail decision-intelligence example (§58) with outcome memory.
11. **Benchmarks:** headline results table with confidence intervals, link to benchmark cards and methodology, and an honest "where we lose" note. Only include claims backed by committed results.
12. **Comparison:** an honest, versioned comparison table (A1.5 rules).
13. **Production:** Docker/Kubernetes/cloud links, observability, security & privacy model summary.
14. **Roadmap:** the v0.1 / v0.5 / v1.0 checklists (§68) with current status.
15. **Docs, community, contributing, security policy, license, citation** (`CITATION.cff` + BibTeX block), acknowledgments.

## A29.2 README quality rules

- A new reader understands what it is, why it matters, and how to try it **within 60 seconds** of landing on the page.
- Every code block in the README MUST be **executed in CI** (for example via `pytest --markdown-docs`/`mktestdocs` or an equivalent runner). A README that does not run fails the build.
- Copy-paste ready; no placeholders that fail; show real output.
- Images/diagrams are committed assets (SVG/PNG), render in light and dark themes, and have alt text.
- Concise: the README is a front door; deep content lives in `docs/`.
- Tone: clear, confident, precise, no hype; the claims discipline in A28.2 applies.
- Accessibility: descriptive link text, alt text, no information conveyed by color alone.
- Keep a top-level table of contents only if the README exceeds roughly 3 screens.

## A29.3 Documentation set (extends §64)

`docs/` built with a modern static site generator (for example MkDocs Material), versioned per release, with search, copy buttons, and tested snippets. Contents:

```text
docs/
  index.md                     landing
  quickstart.md                five minutes
  concepts/                    memory model, cell & gates, planner/router, substrates,
                               lifecycle, temporal & bitemporal, provenance, outcomes,
                               policies, security & privacy
  guides/                      integrations, storage, MCP, REST, deployment, MemoryQL,
                               migrating/importing, writing a policy, writing a substrate
  reference/                   API (autogenerated from type hints and docstrings),
                               CLI, config schema, event schema, MMF schema
  architecture.md              architecture (mirrors A2–A3 and the figure)
  research.md                  gates, planner, MemoryBench, ablations, paper
  benchmarks/                  benchmark cards and methodology
  production.md                Docker, Kubernetes, AWS/Azure/GCP/Databricks references
  security.md                  threat model (A17), data handling, audit
  contributing/                dev setup, conformance suites, adding an adapter/substrate
  assets/                      membrane-architecture.png, logo, demo GIF
```

Also required at repo root: `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `CITATION.cff`, `LICENSE` (permissive OSS license such as Apache-2.0 recommended; owner to confirm), `PROJECT.md` (this file), issue/PR templates, and a `GOVERNANCE.md` once there are outside contributors.

## A29.4 Examples

`examples/` MUST contain runnable, tested examples: `basic`, `bitemporal_state`, `explain`, `outcome_feedback`, `query_planner`, `langgraph`, `langchain`, `openai`, `anthropic`, `mcp`, `custom_agent`, `retail_decision_intelligence`, `poisoning_defense`, `policy_dsl`. Each has its own README, runs offline unless it demonstrates a provider integration, and is executed in CI (provider examples via recorded fixtures).

## A29.5 Web demo (extends §65)

Single-page demo (shipped as `membrane demo`, no cloud required) with: **Timeline**, **Memory Graph (provenance + supersession)**, **Gate Events (READ / WRITE / UPDATE / FORGET)**, **Plan view** (which substrate answered), **Score breakdown bars**, **Branch/diff/rollback**, **Quarantine queue**, **Token-budget slider**. This is the source of the README GIF.

---

# A30. Scope Control and Release Gating

The project is large. Ambition is protected by sequencing.

## A30.1 Membrane v0.1 scope lock

Build **only** this first, and make it excellent:

```text
              MEMBRANE CORE
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
    Memory       Memory     Memory
    State        Gates     Router/Planner (rule-based, single SQLite substrate behind the interface)
        │          │          │
        └──────────┼──────────┘
                   ▼
             SQLite backend
```

v0.1 MUST prove three things (each with a test and a mini-MemoryBench task):

1. **It remembers intelligently:** not everything is stored (write gate), duplicates and noise are rejected, with reasons.
2. **It retrieves intelligently:** the planner picks the right retrieval path (exact state vs lexical/semantic vs timeline) and `explain()` shows why.
3. **It evolves intelligently:** old knowledge is superseded, expired, or consolidated, with history, bitemporal as-of queries, and events.

Also in v0.1 (cheap, high leverage): event log, `explain()`, snapshot/diff, the cell abstraction (A2) with rule gates, namespaces, provenance, idempotency, CLI basics, mini-MemoryBench in CI, **and an MCP server preview** (fastest route to real users).

## A30.2 After v0.1

Only after the three proofs hold: embeddings and hybrid retrieval, Postgres/pgvector, LangGraph/LangChain/OpenAI/Anthropic adapters, REST, observability, Docker/K8s/cloud references, learned gates, full MemoryBench, MemoryQL, replay at scale, multi-agent features, TypeScript client.

## A30.3 Non-negotiables (carried forward)

Never couple core to an LLM, a database vendor, a framework, or a cloud. Never silently delete. Preserve provenance and temporal history. Make every decision inspectable. Keep optional things optional. No premature microservices.

---

---

# PART II — CORE SPECIFICATION (RETAINED IN FULL)

The sections below are the complete core specification (sections 0–71). They are retained in full and amended in place. Where a section carries a `> **Membrane amendment**` note, the note adds to or refines the original text; nothing is removed. Section numbers are stable and referenced throughout Part I as `§N`.

---

# 0. Project Identity

**Project name:** Membrane  
**Working title:** Programmable Memory Runtime for AI Systems (memory-gated, query-planned, substrate-routed)  
**Repository:** `membrane-memory/membrane` (Python distribution `membrane-memory`, import `membrane`; see A1.5)  
**Primary goal:** Build an open-source, provider-neutral memory layer that can be embedded into any AI application or agent framework.

### One-line pitch

> **Membrane is the memory runtime for AI agents: remember, retrieve, update, forget, and consolidate knowledge across sessions without coupling the application to a particular LLM, vector database, or cloud.**

> **Membrane is a programmable memory runtime that dynamically manages information across heterogeneous storage systems for AI applications.**

> **Tagline: The programmable memory layer for AI.**
> **Core abstraction: Memory = state + policy + lifecycle + substrate.**

### Core principle

Membrane is **not** an LLM framework, RAG framework, vector database, or agent framework.

It is a **memory abstraction + memory controller + storage adapter system** that sits underneath them.

In Membrane's architecture (Part I, A2–A6) this is realized as a **gated memory cell** (forget / write / candidate / update / read), a **memory query planner**, a **memory router**, and pluggable **memory substrates**, governed by a **policy engine** and observable through an **append-only event log** and a **provenance graph**.

The system must work with:

- LangGraph
- LangChain
- OpenAI SDK
- Anthropic SDK
- LlamaIndex
- custom Python/TypeScript agents
- MCP-based agents
- any future agent framework

It must support:

- SQLite
- PostgreSQL
- Redis
- DuckDB
- Qdrant
- LanceDB
- Chroma
- Milvus
- pgvector
- Elasticsearch/OpenSearch
- user-defined databases
- object storage
- hybrid stores

It must run:

- entirely locally
- on a VM
- in Docker
- Kubernetes
- serverless/container platforms
- Databricks
- AWS
- Azure
- GCP
- on-premise

The core package must not depend on any cloud provider.


> **Membrane amendment (§0)** — The identity above is extended, not replaced. Part I (A1–A30) is the authoritative architecture charter. The canonical architecture figure is `docs/assets/membrane-architecture.png` ("Membrane: Memory-Gated Architecture (LSTM-Inspired)" plus "Full Membrane Architecture with LSTM-Style Memory Cell"); the implementation MUST match it. The research thesis is: *AI memory should be managed like a computational resource, not retrieved like a document collection.*

---

# 1. Product Vision

Most AI applications currently treat memory as one of:

1. chat history
2. a vector database
3. a conversation summary
4. a key-value store
5. a framework-specific memory abstraction

These approaches do not provide a general memory lifecycle.

Membrane introduces a unified lifecycle:

```text
                    EXPERIENCE
                        |
                        v
                 +-------------+
                 |  INGESTION  |
                 +------+------+
                        |
                        v
                 +-------------+
                 | WRITE GATE  |
                 +------+------+
                        |
                        v
                  MEMORY STORE
                        |
             +----------+----------+
             |                     |
             v                     v
        RETRIEVAL              TIMELINE
             |                     |
             v                     |
        READ GATE                  |
             |                     |
             +----------+----------+
                        |
                        v
                  WORKING MEMORY
                        |
                        v
                     AGENT
                        |
                        v
                    OUTCOME
                        |
                        v
              REFLECTION / UPDATE
                        |
             +----------+----------+
             |                     |
             v                     v
        CONSOLIDATION          FORGETTING
```

The core research/product idea is:

> **Memory should be stateful and governed by policies rather than being an unbounded collection of retrieved chunks.**


> **Membrane amendment (§1)** — The lifecycle is formalized by the memory cell (A2) and extended with planning, routing, outcome feedback, and replay. Updated flow:

```text
 EXPERIENCE (x_t, c_t, s_t)                       QUERY (q_t, c_t)
        │                                                │
        ▼                                                ▼
 INGESTION → CANDIDATE φ(…)                      QUERY PLANNER (A4) ── plan
        │                                                │
        ▼                                                ▼
 FORGET f_t · WRITE i_t · UPDATE u_t (+Δ)        MEMORY ROUTER (A5) ── fan-out
        │            (POLICY ENGINE A13)                 │
        ▼                                                ▼
 MEMORY STATE M_t ──▶ ROUTER ──▶ SUBSTRATES ◀── SUBSTRATE SEARCH (SQL/KV/Vector/Graph/Temporal)
        │                                                │
        │                                           READ GATE o_t
        │                                                │
        │                                  CONTEXT ASSEMBLY (A19) = WORKING MEMORY
        │                                                │
        │                                              AGENT ──▶ ACTION ──▶ OUTCOME
        │                                                                     │
        ◀─────────── OUTCOME FEEDBACK (A9) · UTILITY & DECAY (A10) ◀──────────┘
        │
        ▼
 REPLAY (A11) → CONSOLIDATION (A12) · REFLECTION · FORGETTING (provenance-preserving)
```

The core research/product idea is extended: memory is **state + policy + lifecycle + substrate**, managed as a computational resource.

---

# 2. Design Goals

## 2.1 Mandatory goals

### G1 — Framework agnostic

The core package must not import LangChain, LangGraph, OpenAI, Anthropic, LlamaIndex, or any agent framework.

Integrations live in separate packages/modules.

### G2 — Model agnostic

Membrane must not require an LLM.

Every major capability must have:

- deterministic implementation
- optional embedding provider
- optional LLM-assisted implementation

The system must remain usable without an LLM.

### G3 — Storage agnostic

Storage must be implemented through interfaces.

No business logic may depend directly on SQLite, Postgres, Qdrant, etc.

### G4 — Local first

A new developer should be able to:

```bash
pip install membrane-memory
```

and start with:

```python
from membrane import Memory

memory = Memory()
```

with no API keys and no external service.

The default backend should be SQLite.

### G5 — Production ready

Support:

- concurrency
- transactions
- retries
- idempotency
- observability
- structured logging
- health checks
- migrations
- configuration
- secrets through environment variables
- tenant isolation
- namespaces
- access control hooks

### G6 — Portable

The core data model must be portable across storage backends.

### G7 — Explainable memory decisions

Every memory decision should be inspectable:

- why was this memory written?
- why was it retrieved?
- why was it updated?
- why was it forgotten?
- why was it consolidated?

### G8 — Incremental adoption

A developer should be able to add Membrane to an existing application without changing the agent architecture.


> **Membrane amendment (§2)** — Additional mandatory goals:

### G9 — Planned and routed retrieval

Retrieval MUST go through a query planner and router (A4, A5). No code path may assume "every question is a vector search". A deterministic rule planner MUST exist.

### G10 — Outcome-aware

The system MUST be able to learn which memories were actually useful from downstream outcomes (A9, A10), without requiring a neural network.

### G11 — Safe by construction

Persistent memory is an attack surface. Trust tiers, quarantine, and write-gate defenses (A17) are core, not optional.

### G12 — Reproducible and benchmarked

Every performance or quality claim is backed by a reproducible benchmark card (A28). Time and randomness are injectable (A26).

### G13 — Auditable history

History is append-only and branchable: snapshot, diff, rollback, and bitemporal as-of queries (A16). Decisions are replayable.

### G14 — Programmable

Memory behavior is programmable via policies, the policy DSL, and (later) MemoryQL (A13, A15).

---

# 3. Non-Goals

Do NOT turn Membrane into:

- another LangChain clone
- another agent orchestration framework
- another vector database
- a proprietary LLM service
- a mandatory cloud service
- a mandatory RAG pipeline
- an autonomous agent that controls the user's application
- an opaque "magic memory" system

Membrane manages memory. The application remains responsible for agent execution.


> **Membrane amendment (§3)** — Also do NOT turn Membrane into:

- a general-purpose database or query engine (the planner **routes among** existing databases; it does not replace them)
- a hosted memory-as-a-service that is required to use the library
- a system that lets an LLM write to storage directly (A14, A17)

---

# 4. Core Concepts

## 4.1 Memory

A memory is a durable representation of information that may be useful later.

A memory has:

```text
id
content
memory_type
namespace
subject
scope
source
confidence
importance
utility
created_at
updated_at
last_accessed_at
expires_at
version
status
supersedes
metadata
embedding_reference
```


> **Membrane amendment (§4)** — The memory record carries additional required fields (full schema in A7.1): `state`, `strength`, `trust_tier`, `recorded_at` (transaction time), `observed_at`, `valid_from`, `valid_until`, `superseded_by`, `structured` (`entity`, `attribute`, `value`, `unit`), `pii` tags, `tenant_id`/`user_id`/`agent_id`/`session_id`, and `idempotency_key`. Field semantics: `confidence` = belief the content is true; `importance` = intrinsic significance; `utility` = learned usefulness from outcomes (A9); `strength` = dynamic combination with decay/reinforcement (A10).

---

# 5. Memory Types

The initial implementation must support these types.

## 5.1 Semantic memory

Stable facts.

Examples:

```text
"The user's preferred language is Python."
"SKU 123 belongs to category L5."
"Store 104 belongs to region West."
```

## 5.2 Episodic memory

Events or experiences.

Examples:

```text
"User purchased product X on Monday."
"Agent recommended promotion Y."
"Promotion Y generated +8.4% lift."
```

## 5.3 Working memory

Short-lived task state.

Examples:

```text
Current task:
Investigate sales decline in East Michigan.

Current entities:
East Michigan, Own Brand, Week 39.

Current hypotheses:
OSA decline may explain sales decline.
```

## 5.4 Procedural memory

Reusable strategies or workflows.

Examples:

```text
"For this class of question, retrieve sales and margin before promotion."
```

## 5.5 Preference memory

User/application preferences.

Examples:

```text
"User prefers concise answers."
```

## 5.6 State memory

Current state of an entity.

Examples:

```text
Product X:
price = 12.99
inventory = 431
promotion = active
```

State memory is explicitly temporal and should support historical versions.


> **Membrane amendment (§5)** — Additional memory types (all first-class and supported by the state machine, provenance, and planner):

## 5.7 Outcome memory

The result of a decision or action, linked to the decision and to the memories used.

```text
Decision: Run promotion X.
Expected: +8% sales, +3% margin
Actual:   +11% sales, +1% margin
```

## 5.8 Decision memory

A decision taken by an agent or human, with rationale, expectation, and the `used_memories` that informed it. Enables decision replay (A16) and credit assignment (A9).

## 5.9 Causal memory

```text
observation · cause · effect · confidence · intervention · outcome
```

Example: "Promotion X on SKU 123 → +8.4% lift (confidence 0.8, intervention=true, evidence=[analysis_17])". Stored in the graph substrate with evidence in structured storage (research direction R6).

## 5.10 Consolidated (summary) memory

A derived semantic memory produced by consolidation, with links to all source memories (A12).

## 5.11 Typing rule

`memory_type` determines the default primary substrate (A6.4), default decay rate (A10), default trust requirement (A17), and default consolidation behavior.

---

# 6. Memory Lifecycle

Every memory passes through a lifecycle.

```text
candidate
   |
   v
written
   |
   v
active
   |
   +----> updated
   |
   +----> superseded
   |
   +----> archived
   |
   +----> forgotten
```

A memory must never silently disappear.

Forget/archive operations should generate an event.


> **Membrane amendment (§6)** — The full normative state machine is in A7.2. Added states: `rejected`, `quarantined`, `expired`, `consolidated_into`. The lifecycle is:

```text
candidate → (rejected | quarantined → written | written) → active
active → updated → superseded → archived
active → expired → archived
active → consolidated_into → archived
archived → forgotten   (policy or erasure; tombstone + audit event retained)
```

Illegal transitions raise `InvalidTransitionError`. Every transition emits an event.

---

# 7. Memory Gates

The central Membrane abstraction is the **Memory Controller**.

The controller exposes:

```python
class MemoryController:
    def read(self, query, context=None): ...
    def write(self, candidate, context=None): ...
    def update(self, memory_id, update, context=None): ...
    def forget(self, memory_id, context=None): ...
    def consolidate(self, scope=None, context=None): ...
```

The controller uses pluggable policies.


> **Membrane amendment (§7)** — The `MemoryController` is a thin façade over the **MembraneCell** (A2.7). Additional controller surface:

```python
class MemoryController:
    # original
    def read(self, query, context=None): ...
    def write(self, candidate, context=None): ...
    def update(self, memory_id, update, context=None): ...
    def forget(self, memory_id, context=None): ...
    def consolidate(self, scope=None, context=None): ...
    # added
    def plan(self, query, context=None): ...                 # explain_plan, dry run (A4)
    def step(self, state, query=None, observation=None, context=None, signals=None): ...  # the cell (A2.7)
    def feedback(self, recall_id, outcome, **kw): ...        # A9
    def replay(self, scope=None, budget=None): ...           # A11
    def explain(self, ref): ...                              # A18
    def snapshot(self, label=None): ...                      # A16
    def diff(self, a, b): ...
    def branch(self, name, from_=None): ...
    def rollback(self, to): ...
    def review_queue(self): ...                              # A17 quarantine
```

The five gates are: **forget, write, update, read** (all with `[0,1]` scores) plus the **candidate** function `φ`. Policies (A13) turn scores into actions.

---

# 8. Read Gate

The read gate determines:

1. whether memory is needed
2. which memory stores to query
3. which memories are relevant
4. how many memories should enter working memory

Inputs may include:

```text
query
user_id
session_id
namespace
current task
entities
time range
memory type
recency
importance
confidence
historical utility
```

A baseline scoring function:

```text
read_score =
    w1 * semantic_relevance
  + w2 * lexical_relevance
  + w3 * entity_match
  + w4 * temporal_relevance
  + w5 * freshness
  + w6 * confidence
  + w7 * utility
  + w8 * access_history
```

All weights must be configurable.

The initial implementation must use deterministic scoring.

A learned scorer may be added later.


> **Membrane amendment (§8)** — The read gate is the `o_t` gate (A2.3). Retrieval `ψ(M_t, q_t)` is performed by **query planner → router → substrates** (A4–A6), not by a hard-coded vector search. Additions:

- Step 1 of this gate ("whether memory is needed") is decided by the planner (`QueryPlan.needs_memory`); trivial queries MAY skip memory entirely.
- Step 2 ("which memory stores to query") is the plan's substrate selection.
- The baseline `read_score` gains terms for `strength` (A10), `trust_tier` (A17), and `outcome_utility` (A9), and a diversity/redundancy penalty (A19):

```text
read_score =  w1·semantic + w2·lexical + w3·entity_match + w4·temporal_relevance
            + w5·freshness + w6·confidence + w7·utility + w8·access_history
            + w9·strength + w10·trust − w11·redundancy
```

- All weights remain configurable; scores remain normalized to `[0,1]`; the deterministic scorer remains the default.
- Token-budgeted selection and context assembly: A19.

---

# 9. Write Gate

The write gate determines whether new information deserves durable memory.

Baseline features:

```text
novelty
confidence
importance
future_utility
source_reliability
persistence
contradiction
scope
```

Example:

```text
write_score =
    novelty
  + importance
  + confidence
  + future_utility
  - redundancy
```

The write policy must support:

- always
- never
- threshold
- rule-based
- custom callback
- learned policy


> **Membrane amendment (§9)** — The write gate is the `i_t` gate (A2.3). Additional features: `trust_tier`, `instruction_likeness` (A17), `pii_risk`, `source_rate_anomaly`, `scope_risk`. Outcomes of the write decision: `reject`, `quarantine`, `write`, or `route_to_update` (when it overlaps an existing memory, defer to the update gate and `Δ`). Candidate extraction `φ` MAY be deterministic (rules, structured ingestion), embedding-assisted, or LLM-assisted; **an LLM never writes directly**. Persistence flags (`persist=True/False`, §37) are honored before scoring. Every decision, including rejections, produces an inspectable record (rejections are logged as events, bounded by retention policy).

---

# 10. Update Gate

The update gate handles evolving knowledge.

Example:

```text
Existing:
price = 10.00

New:
price = 12.00
```

The controller should identify that the new state supersedes the old state rather than storing two equally current facts.

Update decisions must preserve history.


> **Membrane amendment (§10)** — The update gate is the `u_t` gate with merge/revise operator `Δ(M_{t-1}, M̃_t)` (A2.3). `Δ` strategies (pluggable): `supersede` (new current version, old becomes `superseded` with `valid_until` set), `merge_fields` (field-wise merge into a new version), `append_evidence` (raise confidence, add provenance), `conflict` (emit `memory.conflict`, apply the configured conflict-resolution policy from A20). For structured memories (`entity`, `attribute`, `value`) the detection is exact and deterministic (§30). All updates are transactional with their events (§55) and preserve history (§29, A16).

---

# 11. Forget Gate

Forgetting is policy-driven.

Candidate features:

```text
age
last_accessed_at
access_count
utility
importance
confidence
contradiction
superseded
expiration
historical_value
```

Example policy:

```text
forget_score =
    age_decay
  + low_utility
  + low_access_frequency
  + superseded
  - historical_importance
```

The system must distinguish:

- delete
- archive
- supersede
- expire

These are not equivalent.

Default behavior should be **archive/supersede**, not destructive deletion.


> **Membrane amendment (§11)** — The forget gate is the `f_t` gate (A2.3). Its score is driven by `strength` (A10), age decay, access frequency, utility, superseded/contradicted status, expiry, historical value, and retention/legal-hold rules. The four operations remain distinct: **delete**, **archive**, **supersede**, **expire** (plus **erasure** for privacy requests, which leaves a tombstone and audit record). Default remains archive/supersede. Learned forget gates may *propose*; the Policy Engine's hard rules (never silently delete; legal holds; provenance) *decide* (A2.8).

---

# 12. Consolidation

Consolidation transforms repeated episodic memories into higher-level memories.

Example:

```text
Monday:
User bought organic milk.

Tuesday:
User bought organic yogurt.

Wednesday:
User bought organic vegetables.

Friday:
User bought organic eggs.
```

Consolidation may produce:

```text
User frequently purchases organic products.
```

Consolidation must be explicit and inspectable.

The system should retain links to source memories.

Required fields:

```text
consolidated_memory_id
source_memory_ids
consolidation_method
confidence
created_at
```


> **Membrane amendment (§12)** — Consolidation is triggered by **replay** (A11) or manually, and follows A12: deterministic methods first (frequency/recurrence, aggregation, time-window roll-ups), optional LLM summarization behind the write gate, full provenance links, reversible by default (sources archived, not deleted), with `compression_ratio` and Memory-Utility-per-Token reported.

---

# 13. Reflection

Reflection is optional and must not require an LLM.

A reflection engine can detect:

- repeated patterns
- contradictions
- recurring entities
- stale memories
- frequently accessed memories
- candidate summaries
- memory clusters

Optional LLM-based reflection may produce semantic summaries.


> **Membrane amendment (§13)** — Reflection additionally includes: replay-driven pattern detection (A11), memory health analysis and `doctor` checks (A22), confidence calibration (A22), outcome-based utility recomputation (A9), and detection of poisoning indicators (A17). Reflection remains LLM-optional and never mutates memory except through gates and the Policy Engine.

---

# 14. Memory Representation

Use a hybrid representation.

```text
Memory
 |
 +-- content
 +-- structured fields
 +-- metadata
 +-- temporal information
 +-- relationships
 +-- optional embedding
 +-- provenance
```

Do not make embeddings mandatory.

A memory can exist without an embedding.


> **Membrane amendment (§14)** — The hybrid representation includes a typed `structured` block (`entity`, `attribute`, `value`, `unit`, `qualifiers`) enabling exact lookups, contradiction detection (§30), and as-of queries (A16) without embeddings. Embeddings, graph edges, and lexical indexes are **derived data** that can be rebuilt from the primary record and event log (`membrane reindex`).

---

# 15. Storage Abstraction

Create these interfaces:

```python
class MemoryStore(Protocol):
    def put(self, memory): ...
    def get(self, memory_id): ...
    def update(self, memory): ...
    def delete(self, memory_id): ...
    def list(self, query): ...
    def search(self, query): ...
```

Separate interfaces for capabilities where appropriate:

```python
class VectorStore(Protocol): ...
class GraphStore(Protocol): ...
class KeyValueStore(Protocol): ...
class EventStore(Protocol): ...
```

The Membrane controller should compose these capabilities.


> **Membrane amendment (§15)** — The protocols below are the storage-facing interfaces. The router interacts with them through the capability-declaring `Substrate` wrapper (A6.3), and the `CompositeSubstrate` composes several. Add `ObjectStore` for large payloads. Every protocol is covered by the conformance suite (§52) parameterized by declared capabilities.

---

# 16. Storage Backends

## Tier 1 — Must implement

### SQLite

Default local backend.

Use SQLite for:

- metadata
- memories
- events
- relationships
- state
- configuration

Optional local vector extension should be supported through an adapter rather than hard-coding a dependency.

### PostgreSQL

Production relational backend.

Support:

- standard PostgreSQL
- pgvector when installed

## Tier 2 — Implement adapters

- Qdrant
- LanceDB
- Chroma
- DuckDB
- Redis

## Tier 3 — Extension interface

Allow external contributors to implement:

- Milvus
- Elasticsearch
- OpenSearch
- Pinecone
- Weaviate
- MongoDB
- Cosmos DB
- DynamoDB
- Snowflake
- Databricks/Lakebase

Do not require all adapters in the initial release.


> **Membrane amendment (§16)** — Backends are organized by **substrate kind** (A6.2). Additional example backends per kind (all Tier 3 extension unless listed above): **MySQL** (structured); **Neo4j, TigerGraph** (graph); **ClickHouse, TimescaleDB** (temporal); **DynamoDB** (KV); **S3/Blob/GCS** (object); **Custom backend (yours)** via the substrate protocol. Tier 1 remains SQLite and PostgreSQL (+pgvector). Default install uses SQLite for every substrate kind (SQL, KV, graph edge tables, event/temporal tables, FTS5 lexical).

---

# 17. Storage Architecture

The application should be able to configure:

```yaml
storage:
  metadata: sqlite
  vector: local
  graph: sqlite
  events: sqlite
```

or:

```yaml
storage:
  metadata: postgres
  vector: qdrant
  graph: postgres
  events: postgres
```

or:

```yaml
storage:
  metadata: postgres
  vector: pgvector
  graph: postgres
  events: postgres
```

or a custom combination.


> **Membrane amendment (§17)** — Storage configuration is expressed **per substrate kind** (A6). The legacy keys remain valid aliases (`metadata → structured`, `vector → semantic`, `events → temporal`).

```yaml
storage:
  structured: postgres      # alias: metadata
  semantic: qdrant          # alias: vector
  graph: neo4j
  temporal: timescale       # alias: events
  kv: redis
  object: s3
```

```yaml
storage:                    # the zero-config default
  structured: sqlite
  semantic: local           # optional; lexical FTS5 works without embeddings
  graph: sqlite
  temporal: sqlite
  kv: sqlite
  object: filesystem
```

---

# 18. Provider Abstractions

Embeddings must be abstracted.

```python
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

Support adapters for:

- OpenAI
- Anthropic-compatible external embedding services if applicable
- Hugging Face
- sentence-transformers
- local embedding models
- custom providers

Do not make an embedding provider mandatory.

---

# 19. LLM Abstraction

Membrane may optionally use an LLM for:

- memory extraction
- memory consolidation
- reflection
- semantic compression
- contradiction interpretation

But the core API must work without one.

```python
class LanguageModel(Protocol):
    def generate(self, prompt, **kwargs): ...
```

Adapters should be separate.


> **Membrane amendment (§19)** — Allowed LLM roles are: candidate extraction (`φ`), consolidation/summarization, reflection, contradiction interpretation, optional planner assistance (A4.4), and entity resolution. In all cases the LLM produces *proposals*; gates and the Policy Engine decide. Every external call is audited (A14).

---

# 20. Framework Integrations

## 20.1 LangGraph

Provide:

```python
from membrane.integrations.langgraph import MemoryStore
```

The integration should support:

- node-level memory access
- state integration
- checkpoint-aware memory
- thread/session namespaces
- long-term memory independent of graph checkpoints

Example:

```python
memory = Memory()

def agent_node(state):
    memories = memory.recall(
        state["messages"][-1].content,
        namespace=state["user_id"]
    )

    ...
```

Do not take over LangGraph orchestration.

---

# 21. LangChain

Provide:

```python
from membrane.integrations.langchain import MembraneRetriever
```

Support:

- Retriever interface
- Runnable-compatible components
- optional chat history adapter

Membrane must remain usable without LangChain.

---

# 22. OpenAI SDK

Provide a thin integration layer.

Example:

```python
from membrane.integrations.openai import MemoryMiddleware

memory = Memory()

response = client.responses.create(
    model="...",
    input=memory.wrap_input(
        user_input,
        user_id="123"
    )
)
```

Do not assume or hard-code a particular OpenAI API shape.

Keep the adapter versioned.

---

# 23. Anthropic SDK

Provide:

```python
from membrane.integrations.anthropic import MemoryMiddleware
```

The adapter should:

1. retrieve relevant memories
2. construct optional context
3. call the Anthropic client
4. optionally extract/write memories from the interaction

The adapter must not require Anthropic.

---

# 24. Generic Agent API

Every framework should be able to use the generic API:

```python
memory = Memory()

memory.remember(
    content="...",
    user_id="user-123"
)

results = memory.recall(
    query="...",
    user_id="user-123"
)

memory.update(...)

memory.forget(...)

memory.consolidate(...)
```

This is the canonical API.

---

# 25. MCP Integration

Provide an optional MCP server exposing tools:

```text
memory.remember
memory.recall
memory.update
memory.forget
memory.list
memory.timeline
memory.reflect
memory.consolidate
```

This allows any MCP-capable agent to use Membrane.


> **Membrane amendment (§25)** — Additional MCP tools: `memory.explain`, `memory.plan`, `memory.feedback`, `memory.record_decision`, `memory.record_outcome`, `memory.snapshot`, `memory.diff`, `memory.rollback`, `memory.review_queue`, `memory.approve`, `memory.reject`, `memory.provenance`, `memory.health`. Zero-config start (for example `uvx membrane-memory mcp`) MUST give an MCP client persistent, inspectable memory backed by local SQLite. An MCP server preview ships in v0.1 (A30). Memory returned over MCP is typed, delimited **data**, not instructions (A17.3).

---

# 26. REST API

Provide an optional API server.

Endpoints:

```text
POST   /v1/memories
GET    /v1/memories/{id}
POST   /v1/memories/search
PATCH  /v1/memories/{id}
DELETE /v1/memories/{id}

POST   /v1/memory/recall
POST   /v1/memory/consolidate
POST   /v1/memory/reflect

GET    /v1/timeline/{namespace}
GET    /v1/health
GET    /v1/metrics
```

Use FastAPI for the reference Python server.

The server must be optional.


> **Membrane amendment (§26)** — Additional endpoints:

```text
POST   /v1/memory/plan                 # explain_plan (dry run)
GET    /v1/memory/explain/{id}         # recall or memory explanation
POST   /v1/memory/feedback             # outcome feedback (A9)
POST   /v1/decisions                   # record decision
POST   /v1/decisions/{id}/outcome      # record outcome
POST   /v1/memory/replay
POST   /v1/snapshots                   # snapshot / list
GET    /v1/diff?from=&to=
POST   /v1/branches                    # create / merge
POST   /v1/rollback
GET    /v1/provenance/{id}
GET    /v1/review                      # quarantine queue
POST   /v1/review/{id}/approve | reject
POST   /v1/export  |  POST /v1/import  # Membrane Memory Format (A23)
GET    /v1/health  |  GET /v1/doctor
```

An OpenAPI 3.1 schema is generated, versioned, and used to generate clients (A24).

---

# 27. Namespaces and Multi-Tenancy

Every memory must support namespaces.

Examples:

```text
tenant
user
agent
application
session
project
organization
```

A memory can have:

```text
tenant_id
user_id
agent_id
session_id
namespace
```

The storage layer must enforce namespace filters.

Never return cross-tenant memories accidentally.


> **Membrane amendment (§27)** — Namespace filters are enforced at three layers: the store, the router (A5.1), and the access-control hook (A20). Cross-tenant leakage tests are mandatory (A17.4). Scopes: `private`, `shared`, `tenant`, `global`.

---

# 28. Provenance

Every memory must retain provenance.

Example:

```json
{
  "source": {
    "type": "conversation",
    "provider": "openai",
    "session_id": "abc",
    "message_id": "msg-123"
  }
}
```

Other source types:

```text
document
database
API
tool
agent
human
system
inference
consolidation
```

The API should expose provenance.


> **Membrane amendment (§28)** — Provenance is a **graph** (A8) supporting `memory.provenance()`, `memory.why()`, and `memory.dependents()`, and drives cascading deletion with a previewable plan. Source types additionally include `import`, `replay`, `policy`, and `outcome`. Provenance carries `trust_tier` (A17).

---

# 29. Temporal Model

Every memory must support:

```text
created_at
observed_at
valid_from
valid_until
updated_at
last_accessed_at
expires_at
```

This allows:

- current state
- historical state
- temporal queries
- stale-memory detection
- event timelines

Example:

```python
memory.recall(
    "What was the price last month?",
    at="2026-09-01"
)
```


> **Membrane amendment (§29)** — The temporal model is **bitemporal** (A16): valid time (`valid_from`, `valid_until`) and record time (`recorded_at`, supersession chain). Additional queries: `as_of=` (what did the system believe then), `at=` (what was true then), and both. Snapshot, diff, branch, rollback (A16.2). All time comes from an injectable `Clock` (A26).

---

# 30. Contradiction Detection

The initial implementation should support deterministic contradiction detection for structured memories.

Example:

```text
entity = product_123
attribute = price

old:
value = 10
valid_until = T2

new:
value = 12
valid_from = T2
```

For unstructured memories, provide a pluggable contradiction detector.

No LLM should be required.


> **Membrane amendment (§30)** — Contradiction detection is a pluggable `ConflictDetector`. Deterministic implementations: exact structured conflict (same `entity`+`attribute`, different `value`, overlapping validity), numeric-range conflict, mutually-exclusive-value sets, source-reliability/trust-tier precedence. Resolution policies: `supersede_newer`, `highest_trust`, `highest_confidence`, `keep_both_flagged`, `quarantine`, `manual` (A20). Embedding/LLM-based detectors are optional plugins. Every conflict emits `memory.conflict`.

---

# 31. Memory Scoring

Implement a baseline scoring engine.

```python
@dataclass
class MemoryScore:
    relevance: float
    freshness: float
    confidence: float
    utility: float
    importance: float
    temporal_match: float
    final_score: float
```

All values must be normalized to `[0, 1]`.

Allow custom scoring functions.


> **Membrane amendment (§31)** — `MemoryScore` is extended; all fields stay in `[0,1]`:

```python
@dataclass
class MemoryScore:
    relevance: float
    freshness: float
    confidence: float
    utility: float
    importance: float
    temporal_match: float
    final_score: float
    # added
    strength: float          # A10
    trust: float             # A17
    entity_match: float
    lexical: float
    semantic: float
    redundancy: float        # penalty
    breakdown: dict[str, float]   # exact per-feature contributions for explain() (A18)
```

Custom scoring functions MUST receive and return this structure. The default scorer is deterministic and dependency-free.

---

# 32. Learned Memory Controller

This is an optional research module.

Do not put it in the mandatory core.

Possible implementation:

```text
query features
memory features
history
metadata
        |
        v
small neural controller
        |
        +--> read probability
        +--> write probability
        +--> update probability
        +--> forget probability
```

Framework:

- PyTorch

Training data can come from:

- user feedback
- retrieval success
- downstream task success
- human labels
- synthetic memory trajectories


> **Membrane amendment (§32)** — The learned controller is the learned instantiation of the gate equations in A2.3 (`W`, `b` learned). Training data: replay trajectories `(state, action, outcome)` (A11), user feedback, retrieval success, downstream task success, human labels, synthetic trajectories from the simulator (A26). Evaluation uses shadow mode and offline policy evaluation (A13.3, R11). **Learned gates can propose; hard safety rules always decide** (A2.8). Gate progression: rules → classical ML → neural → RL/outcome optimization (A2.9). Remains optional and outside the mandatory core.

---

# 33. Research Mode

The repository should contain a research package:

```text
research/
  memory_controller/
  benchmarks/
  datasets/
  ablations/
  experiments/
```

Research mode should allow comparing:

```text
Vanilla RAG
RAG + reranker
Naive memory
Rule-based Membrane
Learned Membrane
```


> **Membrane amendment (§33)** — Compare at least the configurations listed in A27.3 (vanilla RAG; RAG + reranker; long-context stuffing; naive summary memory; popular open agent-memory libraries with pinned versions; Membrane rule-based; Membrane classical-ML gates; Membrane learned gates; Membrane LLM-assisted), and run the full ablation grid. Add `research/paper/` for sources and generated figures.

---

# 34. MemoryBench

Create an open benchmark.

Benchmark categories:

### Temporal

- changing facts
- historical facts
- latest-state questions

### Contradiction

- conflicting facts
- superseded facts
- source reliability

### Long-term

- 10
- 100
- 1,000
- 10,000+
  interactions

### Personalization

- stable preferences
- changing preferences

### Consolidation

- repeated events
- pattern formation

### Forgetting

- stale information
- irrelevant information

### Efficiency

- token usage
- retrieval latency
- storage size
- number of retrieved memories

Metrics:

```text
Recall@K
MRR
NDCG
Temporal Accuracy
Contradiction Rate
Stale Retrieval Rate
Memory Precision
Memory Recall
Write Precision
Forget Precision
Consolidation Ratio
Latency
Token Reduction
Storage Growth
```


> **Membrane amendment (§34)** — MemoryBench is built from day one as a mini-benchmark in CI (A30), then expanded. Added categories: **Routing** (planner substrate selection), **Decisions** (outcome-aware quality, decision replay), **Security** (poisoning, leakage, quarantine), **Bitemporal** (`at` vs `as_of`). Added metrics: Memory Utility per Token (MUT), Routing Accuracy, Plan Efficiency, Decision Quality Uplift, Poisoning Success Rate, Calibration (ECE), Information Retention after consolidation, latency percentiles at 1K–10M memories. Also run established public conversational/long-term memory benchmarks for comparability (A27.4). Results are published as benchmark cards (A28.2).

---

# 35. Observability

Memory decisions must be observable.

Every operation should optionally produce an event:

```json
{
  "event": "memory.read",
  "memory_id": "...",
  "score": 0.87,
  "reason": {
    "semantic": 0.91,
    "freshness": 0.72,
    "utility": 0.83
  }
}
```

Events:

```text
memory.created
memory.read
memory.updated
memory.superseded
memory.archived
memory.forgotten
memory.consolidated
memory.conflict
memory.recalled
```

Provide OpenTelemetry integration.

Do not require OpenTelemetry.


> **Membrane amendment (§35)** — Additional events: `memory.expired`, `memory.quarantined`, `memory.rejected`, `memory.approved`, `plan.created`, `plan.executed`, `gate.evaluated`, `policy.shadow`, `outcome.recorded`, `feedback.applied`, `replay.started`, `replay.completed`, `snapshot.created`, `branch.created`, `rollback.applied`, `substrate.degraded`, `export.completed`, `import.completed`, `llm.call`. Events carry `trace_id`, `actor`, `policy_id`, `policy_version`, and a hash-chain link (A17.4). OpenTelemetry spans cover plan → route → substrate → gate (optional; never required).

---

# 36. Security

The project must support:

- API key authentication for server mode
- namespace isolation
- optional encryption at rest
- optional field-level encryption
- configurable retention
- PII tagging
- deletion workflows
- audit logs

Memory deletion must support:

```text
delete exact memory
delete user memories
delete namespace
delete source-linked memories
```

Be explicit about cascading deletion.

---

# 37. Privacy

Membrane must follow a privacy-by-design model.

Never automatically assume that every conversation should become permanent memory.

Default behavior should be conservative.

Provide:

```python
memory.remember(
    "...",
    persist=True
)
```

and:

```python
memory.remember(
    "...",
    persist=False
)
```

Also support:

```python
memory.forget_user(user_id)
```


> **Membrane amendment (§36–37)** — Security and privacy are extended by A17 (trust tiers, quarantine, poisoning defenses, tamper-evident audit chain), A20 (access control, scopes), and A22 (health, calibration). Erasure is the only case where content is physically removed; tombstones and audit records remain. PII tagging/redaction detectors are pluggable. Retention and legal-hold are policy-defined. An external-call audit trail is mandatory for any LLM/embedding provider (A14).

---

# 38. Configuration

Support:

```yaml
memory:
  default_namespace: default

  gates:
    read: rule
    write: rule
    update: rule
    forget: rule
    consolidate: rule

storage:
  metadata: sqlite
  vector: local
  graph: sqlite

embedding:
  provider: local

llm:
  provider: none

observability:
  enabled: false
```

Environment variables should override configuration.


> **Membrane amendment (§38)** — Expanded reference configuration (environment variables still override; secrets only via env/secret stores):

```yaml
memory:
  default_namespace: default
  failure_mode: best_effort          # strict | best_effort | disabled
  policy: balanced                   # conservative | balanced | aggressive | strict_audit | path/to/policy.yaml
  clock: system

  gates:
    read: rule
    write: rule
    update: rule
    forget: rule
    consolidate: rule
    candidate: rule                  # φ: rule | embedding | llm

planner:
  type: rule                         # rule | cost_based | learned | llm_assisted
  fallback: hybrid

router:
  parallel: true
  timeout_ms: 800
  merge: rrf

storage:
  structured: sqlite
  semantic: local
  graph: sqlite
  temporal: sqlite
  kv: sqlite
  object: filesystem

embedding:
  provider: local

llm:
  provider: none

trust:
  default_tier: user_asserted
  quarantine_below: external_content
  detectors: [instruction_like, pii_regex]

utility:
  decay: { semantic: slow, episodic: medium, working: fast, state: none }
  reinforcement_alpha: 0.1
  outcome_credit: proportional

replay:
  interval: 1h
  budget: { seconds: 30, llm_tokens: 0 }

observability:
  enabled: false
  otel: false
  trace_retention_days: 30
```

---

# 39. Package Architecture

Use a clean modular structure.

```text
membrane/
│
├── pyproject.toml
├── README.md                      # world-class README (A29)
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── CHANGELOG.md
├── CITATION.cff
├── GOVERNANCE.md
├── PROJECT.md                     # this file
│
├── src/
│   └── membrane/
│       ├── __init__.py
│       │
│       ├── core/
│       │   ├── memory.py          # public Memory façade
│       │   ├── controller.py
│       │   ├── models.py
│       │   ├── policies.py
│       │   ├── scoring.py
│       │   ├── lifecycle.py
│       │   ├── clock.py           # injectable Clock (A26)
│       │   └── exceptions.py
│       │
│       ├── cell/                  # the Membrane Memory Cell (A2)
│       │   ├── cell.py            # MembraneCell.step()
│       │   ├── state.py           # MemoryState, GateInput, GateOutput, CellTrace
│       │   ├── candidate.py       # φ: candidate generation
│       │   └── delta.py           # Δ: merge/revise operators
│       │
│       ├── gates/
│       │   ├── base.py            # Gate protocol
│       │   ├── read.py            # o_t
│       │   ├── write.py           # i_t
│       │   ├── update.py          # u_t
│       │   ├── forget.py          # f_t
│       │   ├── consolidate.py
│       │   └── learned/           # linear / GBDT / neural / RL gates (optional extras)
│       │
│       ├── planner/               # Memory Query Planner (A4)
│       │   ├── base.py
│       │   ├── rule.py
│       │   ├── cost_based.py
│       │   ├── intents.py
│       │   └── explain.py
│       │
│       ├── router/                # Memory Router (A5)
│       │   ├── router.py
│       │   ├── merge.py           # RRF, union, intersect, ...
│       │   ├── outbox.py          # multi-substrate write convergence
│       │   └── stats.py
│       │
│       ├── substrates/            # Memory Substrates (A6)
│       │   ├── base.py            # Substrate protocol + capabilities
│       │   ├── sqlite/
│       │   ├── postgres/
│       │   ├── vector/            # local, qdrant, lancedb, chroma, pgvector, ...
│       │   ├── graph/
│       │   ├── temporal/
│       │   ├── kv/
│       │   ├── object/
│       │   └── composite.py
│       │
│       ├── stores/                # storage-facing protocols (§15)
│       │   ├── base.py
│       │   ├── sqlite.py
│       │   ├── postgres.py
│       │   ├── vector.py
│       │   └── composite.py
│       │
│       ├── state/                 # memory state machine (A7)
│       │   ├── machine.py
│       │   └── transitions.py
│       │
│       ├── policy/                # Policy Engine (A13) + DSL (A15)
│       │   ├── engine.py
│       │   ├── presets.py
│       │   ├── dsl.py
│       │   └── shadow.py
│       │
│       ├── provenance/            # Provenance Graph (A8)
│       ├── outcomes/              # decisions, outcomes, feedback, credit assignment (A9)
│       ├── utility/               # strength, decay, reinforcement (A10)
│       ├── replay/                # Memory Replay (A11)
│       ├── consolidation/         # compression & consolidation (A12)
│       ├── reflection/
│       ├── entities/              # entity registry & resolution (A21)
│       ├── security/              # trust tiers, quarantine, injection/PII detectors, audit chain (A17)
│       ├── access/                # access control hook, scopes (A20)
│       ├── context/               # context assembly, token budgeting (A19)
│       ├── temporal/              # bitemporal queries, snapshot, diff, branch, rollback (A16)
│       ├── health/                # doctor, health score, calibration (A22)
│       ├── tiering/               # hot/warm/cold, cost report (A25)
│       ├── memoryql/              # parser -> AST -> Query (A15)
│       ├── portability/           # Membrane Memory Format, importers, exporters (A23)
│       ├── sim/                   # trajectory simulator (A26)
│       │
│       ├── embeddings/
│       │   ├── base.py
│       │   ├── local.py
│       │   └── providers/
│       │
│       ├── llm/
│       │   ├── base.py
│       │   └── providers/
│       │
│       ├── integrations/          # all optional extras (A24)
│       │   ├── protocol.py        # AgentMemoryAdapter
│       │   ├── langchain/
│       │   ├── langgraph/
│       │   ├── openai/
│       │   ├── anthropic/
│       │   ├── mcp/
│       │   └── generic/
│       │
│       ├── server/
│       │   ├── api.py
│       │   ├── schemas.py
│       │   └── auth.py
│       │
│       ├── observability/
│       │   ├── events.py
│       │   └── telemetry.py
│       │
│       ├── demo/                  # local web demo (A29.5)
│       │
│       └── cli/
│           └── main.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── adapters/
│   ├── temporal/
│   ├── security/
│   ├── benchmarks/
│   ├── property/                  # Hypothesis
│   ├── planner/                   # planner conformance
│   ├── conformance/               # substrate + adapter contract suites
│   └── e2e/
│
├── examples/
│   ├── basic/
│   ├── bitemporal_state/
│   ├── explain/
│   ├── outcome_feedback/
│   ├── query_planner/
│   ├── langgraph/
│   ├── langchain/
│   ├── openai/
│   ├── anthropic/
│   ├── mcp/
│   ├── custom_agent/
│   ├── retail_decision_intelligence/
│   ├── poisoning_defense/
│   └── policy_dsl/
│
├── benchmarks/
│   ├── memorybench/
│   ├── datasets/
│   └── runners/
│
├── research/                      # A27.6 (optional research extra)
│   ├── memory_controller/
│   ├── benchmarks/
│   ├── datasets/
│   ├── ablations/
│   ├── experiments/
│   └── paper/
│
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── architecture.md
│   ├── memory-model.md
│   ├── storage.md
│   ├── integrations.md
│   ├── security.md
│   ├── research.md
│   ├── production.md
│   ├── concepts/
│   ├── guides/
│   ├── reference/
│   ├── benchmarks/
│   ├── contributing/
│   └── assets/
│       ├── membrane-architecture.png
│       ├── logo.svg
│       └── demo.gif
│
├── deploy/
│   ├── docker/
│   ├── kubernetes/
│   ├── aws/
│   ├── azure/
│   ├── gcp/
│   └── databricks/
│
└── scripts/
```

---

# 40. Dependency Philosophy

Core dependencies must be minimal.

Core should ideally require only:

```text
Python >= 3.10
Pydantic
SQLAlchemy or equivalent minimal persistence abstraction
```

Do not install:

- LangChain
- LangGraph
- OpenAI
- Anthropic
- Qdrant
- PyTorch

as mandatory dependencies.

Use optional extras:

```bash
pip install membrane-memory[langchain]
pip install membrane-memory[langgraph]
pip install membrane-memory[openai]
pip install membrane-memory[anthropic]
pip install membrane-memory[qdrant]
pip install membrane-memory[postgres]
pip install membrane-memory[research]
pip install membrane-memory[all]
```


> **Membrane amendment (§40)** — Core still requires only Python ≥ 3.10, Pydantic, and a minimal persistence layer (stdlib `sqlite3` preferred). Additional optional extras (distribution `membrane-memory`):

```bash
pip install "membrane-memory[mcp]"
pip install "membrane-memory[server]"       # FastAPI reference server
pip install "membrane-memory[otel]"
pip install "membrane-memory[embeddings-local]"
pip install "membrane-memory[tokenizers]"
pip install "membrane-memory[neo4j]"
pip install "membrane-memory[bench]"
pip install "membrane-memory[demo]"
pip install "membrane-memory[research]"     # PyTorch, learned gates
pip install "membrane-memory[all]"
```

Rule: importing `membrane` MUST NOT import any optional dependency; optional features fail with a clear, actionable `MissingExtraError`. Measure and track import time and install size in CI.

---

# 41. Type Safety

Use:

- Python type hints
- Pydantic models where appropriate
- Protocol interfaces
- mypy or pyright
- Ruff
- pytest

Public APIs must be typed.

---

# 42. Sync and Async APIs

Support both:

```python
memory.recall(...)
```

and:

```python
await memory.arecall(...)
```

Async should be implemented correctly rather than wrapping synchronous calls in fake async methods.


> **Membrane amendment (§42)** — Async-first internals: the router's fan-out, the planner's execution, and replay are natively async; the sync API is a thin wrapper that runs the same core (no duplicated logic). Public methods exist in both forms (`recall`/`arecall`, `remember`/`aremember`, `feedback`/`afeedback`, `explain`/`aexplain`).

---

# 43. CLI

Provide:

```bash
membrane init
membrane status
membrane inspect
membrane memories
membrane timeline
membrane search
membrane consolidate
membrane forget
membrane benchmark
membrane serve
```

Example:

```bash
membrane init ./memory.db
membrane serve --port 8000
```


> **Membrane amendment (§43)** — Additional CLI commands:

```bash
membrane explain <recall_id|memory_id>
membrane plan "<query>"              # explain_plan, dry run
membrane replay
membrane doctor [--fix]
membrane snapshot | diff | branch | merge | rollback
membrane review | approve | reject   # quarantine queue
membrane export | import             # Membrane Memory Format
membrane reindex
membrane migrate
membrane mcp                         # MCP server
membrane demo                        # local web demo
membrane benchmark --suite <name>
```

---

# 44. Developer Experience

The first five minutes must be excellent.

Quickstart:

```bash
pip install membrane-memory
```

Then:

```python
from membrane import Memory

memory = Memory()

memory.remember(
    "The user prefers concise technical explanations.",
    user_id="u1"
)

result = memory.recall(
    "How should I communicate with this user?",
    user_id="u1"
)

print(result)
```

No API key.

No Docker.

No external database.

No LLM.


> **Membrane amendment (§44)** — The first five minutes MUST also include one `explain()` call so the developer immediately sees *why* a memory was returned, and a one-liner for the MCP server. The quickstart is executed in CI (A29.2).

---

# 45. Optional LLM Memory Extraction

Provide:

```python
memory.remember_conversation(
    messages,
    user_id="u1"
)
```

This can optionally use an LLM to extract candidate memories.

Pipeline:

```text
conversation
     |
     v
candidate extraction
     |
     v
deduplication
     |
     v
write gate
     |
     v
memory store
```

The LLM must not directly write to the database without the write gate.

---

# 46. Agent Integration Pattern

Membrane should support this universal pattern:

```text
USER
 |
 v
AGENT
 |
 +----> Membrane.recall()
 |
 v
LLM
 |
 +----> tools
 |
 v
RESULT
 |
 +----> Membrane.remember()
 |
 v
MEMORY
```

The agent framework controls the loop.

Membrane controls memory.


> **Membrane amendment (§46)** — Extend the universal pattern with the outcome loop:

```text
USER → AGENT → Membrane.recall()  (plan → route → gate → budgeted context)
          │
          ▼
        LLM → tools → RESULT
          │
          ├──▶ Membrane.remember()            (write / update gates)
          └──▶ Membrane.feedback(recall_id)   (outcome → utility, strength)
```

The framework controls the loop; Membrane controls memory (A24).

---

# 47. Database Independence

Never expose database-specific concepts through the core public API.

Bad:

```python
memory.qdrant_collection.search(...)
```

Good:

```python
memory.recall(...)
```

The adapter translates the operation.

---

# 48. Cloud Independence

The core must not contain:

```text
AWS-specific logic
Azure-specific logic
GCP-specific logic
Databricks-specific logic
```

Cloud deployments belong under:

```text
deploy/
integrations/
```

---

# 49. Deployment Targets

Provide reference deployment configurations for:

## Local

```text
Python + SQLite
```

## Docker

```text
Membrane API + PostgreSQL
```

## Kubernetes

```text
Membrane API
PostgreSQL
optional vector DB
```

## AWS

Reference:

```text
ECS/Fargate
RDS PostgreSQL
S3
```

## Azure

Reference:

```text
Container Apps
Azure Database for PostgreSQL
Blob Storage
```

## GCP

Reference:

```text
Cloud Run
Cloud SQL
GCS
```

## Databricks

Reference:

```text
Databricks App
Lakebase/PostgreSQL-compatible store
Unity Catalog where appropriate
```

These are deployment examples, not hard dependencies.

---

# 50. API Stability

Use semantic versioning.

Public API:

```text
0.x = experimental
1.x = stable
```

Document compatibility guarantees.


> **Membrane amendment (§50)** — In addition to the public Python API, the following are versioned, documented, and have migration paths: the REST/OpenAPI schema, the event schema, the Membrane Memory Format (A23), the config schema, the policy DSL/MemoryQL grammar, and the plugin API. Document a deprecation policy (minimum one minor release of warnings before removal in 1.x).

---

# 51. Testing Strategy

Minimum target:

```text
Unit tests: 90%+
Critical lifecycle paths: 100%
```

Test:

- write
- read
- update
- forget
- supersede
- consolidate
- namespaces
- temporal queries
- concurrency
- idempotency
- adapter behavior
- failure recovery

Every storage adapter must pass the same abstract conformance test suite.


> **Membrane amendment (§51)** — Add the testing practices in A26: injectable `Clock`, seeded RNG, the trajectory simulator, property-based tests (Hypothesis), golden tests for planner decisions and `explain()` output, mutation testing on the state machine and policy engine, fault injection (substrate failure mid-write), parser fuzzing, an **offline CI job** (no network), and an adapter contract test shared across all framework integrations. Add test packages: `tests/property/`, `tests/planner/`, `tests/conformance/`, `tests/e2e/`, `tests/security/` (injection corpus, leakage, quarantine, audit chain).

---

# 52. Storage Conformance Tests

Define:

```python
class MemoryStoreConformanceSuite:
    def test_put()
    def test_get()
    def test_update()
    def test_search()
    def test_delete()
    def test_namespace_isolation()
    def test_concurrency()
```

Any third-party adapter can run this suite.

This is important for ecosystem growth.


> **Membrane amendment (§52)** — The suite is parameterized by declared `SubstrateCapabilities` (A6.3). Additional required tests: `test_time_travel_as_of`, `test_tombstone_and_erasure`, `test_idempotency_key`, `test_cas_version_conflict`, `test_export_import_roundtrip`, `test_reindex_from_primary`, `test_ttl_expiry`, `test_degraded_mode`, and, for substrates declaring them, `test_vector_knn`, `test_graph_traverse`, `test_range_scan`, `test_fulltext`. A **planner conformance suite** (table of query/context/capabilities → expected plan) and an **adapter contract suite** (A24) are also required. Third parties can run all suites to certify a backend; publish a "certified substrates" matrix.

---

# 53. Performance Targets

Initial target for local SQLite:

```text
simple exact lookup: <5 ms
metadata lookup: <10 ms
basic recall: <50 ms
memory write: <20 ms
```

These are engineering targets, not claims.

Benchmark realistically under:

```text
1K memories
10K memories
100K memories
1M memories
10M memories
```


> **Membrane amendment (§53)** — Add targets (engineering targets, not claims) for: planner overhead (rule planner < 2 ms), router overhead, `explain()` cost, token-budgeted assembly cost, p50/p95/p99 at 1K–10M memories, write throughput with and without secondary indexes, replay throughput, storage growth per 1K interactions with and without consolidation. Benchmarks run on a schedule in CI and publish hardware, versions, and confidence intervals (A28.2).

---

# 54. Idempotency

Memory writes should support an idempotency key:

```python
memory.remember(
    content="...",
    idempotency_key="event-123"
)
```

Repeated writes with the same key should not create duplicates.


> **Membrane amendment (§54–55)** — Multi-substrate writes use an **outbox**: the primary record + event commit atomically; secondary indexes (vector, graph, lexical) are updated asynchronously and are repairable (`membrane reindex`). Cross-substrate atomicity is not claimed where the backends cannot provide it (§69 rule 18).

---

# 55. Transactions

Memory updates should be atomic where the backend supports transactions.

Example:

```text
new memory
+
supersede old memory
+
create event
```

must either all succeed or all fail.

---

# 56. Event Log

Every important memory state transition should be represented as an event.

```text
MemoryEvent
------------
id
memory_id
event_type
timestamp
actor
source
payload
```

This creates an auditable memory timeline.


> **Membrane amendment (§56)** — The `MemoryEvent` additionally carries `trace_id`, `policy_id`, `policy_version`, `gate_scores`, `prev_hash`, and `hash` (tamper-evident chain). The event log is the source for snapshot/diff/rollback (A16), replay (A11), audit, and learned-gate training data.

---

# 57. Memory Timeline API

Provide:

```python
memory.timeline(
    entity="product_123",
    start="2026-01-01",
    end="2026-12-31"
)
```

Output:

```text
Jan 10  price = 10
Feb 03  price = 12
Mar 11  promotion = active
Mar 18  promotion = ended
```


> **Membrane amendment (§57)** — The timeline API accepts `as_of=` (record time) in addition to `start`/`end` (valid time), supports `include_superseded=True`, and returns causally-linked entries (decision → outcome). Also `memory.diff(t1, t2)` (A16.2).

---

# 58. Example: Retail Decision Intelligence

Membrane should demonstrate at least one non-chatbot example.

Example:

```text
Retail agent
     |
     v
Membrane
     |
     +-- historical sales
     +-- promotions
     +-- prices
     +-- inventory
     +-- previous decisions
     +-- outcomes
     |
     v
Decision engine
```

Question:

> "Why did sales decline in East Michigan?"

Membrane retrieves:

```text
sales state
inventory state
promotion history
OSA history
previous causal findings
```

The decision engine can operate without an LLM.

This demonstrates that Membrane is useful beyond conversational assistants.


> **Membrane amendment (§58)** — The retail example MUST demonstrate the full Membrane story, not just storage:

- **Planner/router in action:** "What's yesterday's sales?" → structured lookup (no embedding, no LLM); "What changed about SKU 123?" → timeline; "Why did sales decline in East Michigan?" → state + temporal + causal-graph plan with evidence.
- **Outcome memory:** record the decision ("run promotion X"), expected vs actual lift/margin, and let utility update (A9).
- **Causal memory:** cause→effect edges with confidence and provenance back to datasets/queries/analyses.
- **Bitemporal replay:** "what did the agent believe when it recommended promotion X?" (A16).
- **Runs without an LLM.** Ships in `examples/retail_decision_intelligence/` and is executed in CI.

---

# 59. Research Direction

The project should explicitly support future research into:

### R1 — Learned memory gates

Learn:

```text
read
write
update
forget
```

policies.

### R2 — Memory consolidation

Learn when multiple episodic memories should become semantic memory.

### R3 — Temporal memory

Improve reasoning over evolving facts.

### R4 — Memory compression

Maximize utility per byte/token.

### R5 — Outcome-aware memory

Increase/decrease memory utility based on downstream task success.

### R6 — Causal memory

Store:

```text
observation
cause
effect
confidence
intervention
outcome
```

### R7 — Memory routing

Learn which backend should answer a query:

```text
KV
SQL
vector
graph
event log
```

### R8 — LLM-free memory control

Compare:

```text
rules
small ML model
neural controller
LLM controller
```


> **Membrane amendment (§59)** — The research program is expanded in A27 (research questions, additional directions R9–R12, baselines, ablations, public benchmarks, paper plan). R1–R8 remain in force.

---

# 60. Critical Architectural Principle

The system should separate:

```text
MEMORY
from
REASONING
from
GENERATION
from
ACTION
```

Architecture:

```text
             +------------------+
             |     MEMBRANE     |
             |                  |
             | memory lifecycle |
             +--------+---------+
                      |
                      v
             +------------------+
             |    AGENT / LLM   |
             +--------+---------+
                      |
                      v
             +------------------+
             |      TOOLS       |
             +--------+---------+
                      |
                      v
             +------------------+
             |    ENVIRONMENT   |
             +------------------+
```

Membrane should never assume that it is responsible for the other layers.

---

# 61. Failure Handling

Memory failures should not necessarily break the agent.

Provide configurable modes:

```text
strict
best_effort
disabled
```

Example:

```python
Memory(failure_mode="best_effort")
```

If vector retrieval fails but structured storage works, the system should degrade gracefully.


> **Membrane amendment (§61)** — Router fallbacks (A5.1) implement graceful degradation: with `best_effort`, a failed vector substrate falls back to structured + lexical retrieval and the result carries `degraded=True` plus reasons; secondary-index failures never lose primary writes (outbox).

---

# 62. Caching

Support optional caches:

```text
query cache
embedding cache
memory score cache
working-memory cache
```

Caching must never violate namespace or freshness guarantees.


> **Membrane amendment (§62)** — Add a **plan cache** (keyed by normalized query shape + capabilities + policy version). All caches are keyed by namespace and invalidated by memory version/`recorded_at`; they MUST NOT cross tenants or serve stale superseded state.

---

# 63. Plugin Architecture

Third-party plugins should be able to implement:

```text
Store
Embedder
LanguageModel
Gate
Scorer
Consolidator
ConflictDetector
Retriever
Observer
```

Example:

```python
memory.register_gate(
    "my_gate",
    MyCustomGate()
)
```


> **Membrane amendment (§63)** — Additional plugin extension points: `Substrate`, `QueryPlanner`, `Router`/`MergeStrategy`, `Delta` strategy, `PolicyProvider`, `AccessPolicy`, `Importer`/`Exporter`, `PIIDetector`, `InjectionDetector`, `TokenCounter`, `Clock`, `Calibrator`, `OutcomeScorer`, `EntityResolver`. Plugins are discovered through Python entry points, declare an API version, are validated at load time, and are covered by the conformance suites. `memory.register_gate()` remains supported.

---

# 64. Documentation Requirements

The repository must include:

### README

Explain the concept in under 60 seconds.

### Architecture

Detailed system architecture.

### Quickstart

Working in under five minutes.

### Integrations

One example each for:

- LangGraph
- LangChain
- OpenAI
- Anthropic
- custom agent
- MCP

### Storage

SQLite → Postgres → vector → hybrid.

### Research

Memory gates and benchmark.

### Production

Docker/Kubernetes/cloud deployment.


> **Membrane amendment (§64)** — The README and documentation requirements are expanded in **A29**: a world-class `README.md` is a mandatory deliverable, every README code block is executed in CI, and `docs/` follows the layout in A29.3.

---

# 65. Demo Requirements

Create a web demo showing:

```text
              Membrane
                  |
       +----------+----------+
       |                     |
    Timeline              Memory Graph
       |                     |
       +----------+----------+
                  |
             Gate Events
                  |
       +----------+----------+
       |          |          |
      READ       WRITE      FORGET
```

The demo should allow a user to:

1. add memories
2. ask questions
3. inspect retrieved memories
4. see memory scores
5. update facts
6. create contradictions
7. watch supersession
8. trigger consolidation
9. inspect timeline
10. forget a memory

This should become the main README demo/GIF.


> **Membrane amendment (§65)** — The demo additionally shows the plan view, score-breakdown bars, branch/diff/rollback, quarantine queue, and the token-budget slider (A29.5). It runs locally with `membrane demo` and produces the README GIF.

---

# 66. Example Repository Narrative

The README should communicate:

```text
RAG retrieves documents.

Membrane manages knowledge over time.
```

Then:

```text
Remember
Retrieve
Update
Forget
Consolidate
Reflect
```


> **Membrane amendment (§66)** — Keep the narrative, and add the architectural line: *a gated memory cell, a query planner, a router, and pluggable substrates — with every decision explainable.*

---

# 67. Implementation Phases

## Phase 1 — Core

Implement:

- Memory model
- MemoryStore protocol
- SQLite store
- deterministic read/write/update/forget gates
- Memory API
- namespaces
- provenance
- events
- temporal fields
- tests

No LLM.

No vector DB required.

Goal:

```python
memory = Memory()
memory.remember(...)
memory.recall(...)
```

---

## Phase 2 — Retrieval

Add:

- embedding abstraction
- local embeddings
- vector adapter
- hybrid retrieval
- scoring
- reranking interface

---

## Phase 3 — Memory Intelligence

Add:

- consolidation
- reflection
- contradiction handling
- temporal state
- memory compression

---

## Phase 4 — Integrations

Implement:

1. LangGraph
2. LangChain
3. OpenAI
4. Anthropic
5. MCP
6. generic agent API

---

## Phase 5 — Production

Implement:

- PostgreSQL
- pgvector
- REST API
- authentication hooks
- OpenTelemetry
- Docker
- Kubernetes
- cloud deployment examples

---

## Phase 6 — Research

Implement:

- learned memory controller
- MemoryBench
- ablations
- long-term evaluation
- performance benchmarking


> **Membrane amendment (§67)** — Insert **Phase 0 — Scope Lock (v0.1 spike)** per A30 before Phase 1, and carry these additions into the phases:

- **Phase 0:** SQLite backend; the cell abstraction with rule gates; rule query planner + single-substrate router; state machine + events; `explain()`; snapshot/diff; trust tiers + quarantine; mini-MemoryBench in CI; MCP server preview; release-quality README.
- **Phase 1 adds:** bitemporal queries, hash-chained events, injectable `Clock`, simulator, property tests.
- **Phase 2 adds:** token-budgeted context assembly (A19), router fan-out + RRF, lexical engine.
- **Phase 3 adds:** outcome memory + utility, strength/decay/reinforcement, replay, cost-based planner, MemoryQL, policy DSL, calibration, `doctor`.
- **Phase 4 adds:** the universal adapter protocol + contract tests.
- **Phase 5 adds:** Membrane Memory Format import/export, multi-agent scopes + access control, tiering, TypeScript client (after v0.5).
- **Phase 6 adds:** learned planner, shadow-mode offline policy evaluation, full MemoryBench and public benchmarks, paper.

---

# 68. Definition of Done

Version 0.1 is complete when:

```text
[ ] pip install membrane-memory works
[ ] SQLite works without external services
[ ] remember() works
[ ] recall() works
[ ] update() works
[ ] forget() works
[ ] timeline() works
[ ] namespaces work
[ ] provenance works
[ ] temporal state works
[ ] memory events work
[ ] deterministic gates work
[ ] test suite passes
[ ] documentation works
```

Version 0.5:

```text
[ ] vector retrieval
[ ] hybrid retrieval
[ ] PostgreSQL
[ ] LangGraph
[ ] LangChain
[ ] OpenAI
[ ] Anthropic
[ ] MCP
[ ] REST API
[ ] Docker
```

Version 1.0:

```text
[ ] stable public API
[ ] storage conformance suite
[ ] MemoryBench
[ ] observability
[ ] production deployment references
[ ] security documentation
[ ] research results
```


> **Membrane amendment (§68)** — Additional Definition-of-Done items:

Version 0.1:

```text
[ ] MembraneCell abstraction mirrors the architecture figure (A2)
[ ] forget/write/update/read gates + candidate function behind one Gate protocol
[ ] rule query planner + router + explain()/explain_plan()
[ ] bitemporal as-of queries (at= and as_of=)
[ ] snapshot / diff
[ ] trust tiers + quarantine
[ ] hash-chained event log
[ ] mini-MemoryBench runs in CI; README code blocks run in CI; offline CI job passes
[ ] MCP server preview
[ ] world-class README.md (A29) with demo GIF
```

Version 0.5:

```text
[ ] token-budgeted context assembly
[ ] outcome feedback + utility + strength/decay
[ ] replay + consolidation with provenance
[ ] cost-based planner
[ ] adapter contract suite passes for all integrations
```

Version 1.0:

```text
[ ] Membrane Memory Format + importers
[ ] branch / rollback
[ ] multi-agent scopes + access control
[ ] MemoryQL + policy DSL
[ ] public-benchmark results + MemoryBench benchmark cards
[ ] paper submitted
[ ] certified-substrate matrix
```

---

# 69. Instructions for Codex / Claude Code

You are implementing an open-source project called **Membrane**.

Do not treat this document as a suggestion. Treat it as the product specification.

## Mandatory rules

1. Do not introduce unnecessary framework dependencies.
2. Do not couple core logic to any LLM provider.
3. Do not couple core logic to a database vendor.
4. Do not couple core logic to LangChain/LangGraph.
5. Prefer protocols/interfaces and dependency injection.
6. Keep optional integrations isolated.
7. Implement tests before or alongside each major subsystem.
8. Preserve backward compatibility of public APIs.
9. Use type hints everywhere.
10. Document public APIs.
11. Never silently swallow errors.
12. Never silently delete memory.
13. Preserve provenance.
14. Preserve temporal history.
15. Make memory decisions inspectable.
16. Never send data to an external LLM/provider unless explicitly configured.
17. Local SQLite mode must work without internet access.
18. Do not invent unsupported database capabilities.
19. Add conformance tests for every storage adapter.
20. Avoid premature microservices.
21. Implement the memory cell (A2) so that the code structure mirrors the architecture figure: five gate-shaped components (forget, write, candidate, update, read), a memory state, and a retrieval function `ψ` realized by planner + router + substrates.
22. Gates are pure functions of `GateInput` (plus injected clock/seed). No storage I/O inside gates.
23. Hard safety rules are enforced by the Policy Engine after the gates and override any gate score.
24. Never let an LLM write directly to storage. Quarantine low-trust and instruction-like candidates.
25. Use the injectable `Clock`; `datetime.now()` is forbidden in core code (lint-enforced).
26. Namespace filters are enforced in the store and again in the router (defense in depth).
27. Every README/doc claim about quality or speed MUST link to a reproducible benchmark card (A28.2). Do not claim SOTA without evidence.
28. **Write a world-class `README.md` per A29, as a deliverable from the first release.** Every code block in the README MUST be executed in CI. Keep `docs/assets/membrane-architecture.png` and the docs in sync with the code.
29. When in doubt about scope or ordering, follow A30 (v0.1 scope lock) before adding features.
30. Update `docs/architecture.md` and the architecture figure in the same pull request as any change to the cell, planner, router, or substrate model.

## Development order

Implement in this exact order unless a technical dependency requires otherwise:

```text
1. project skeleton (+ CI: lint, strict type-check, offline test job, docs skeleton, README skeleton)
2. core models (incl. bitemporal + trust + strength fields) and injectable Clock
3. storage protocols and the Substrate capability model
4. SQLite backend
5. event log (hash-chained)
6. deterministic scoring
7. memory cell abstraction + Gate protocol (A2)              [new]
8. read gate
9. write gate
10. update/supersession (update gate + Δ)
11. forget/archive (forget gate + state machine)
12. Memory API
13. temporal queries (bitemporal)
14. namespaces
15. provenance (graph)
16. unit/conformance tests
17. query planner (rule) + router (single substrate) + explain()   [new]
18. trajectory simulator + mini-MemoryBench in CI            [new]
19. snapshot / diff / rollback                               [new]
20. trust tiers + quarantine + injection/PII detectors       [new]
21. MCP server preview                                       [new]
22. release-quality README.md + demo GIF (A29)               [new; v0.1 gate]
23. embedding abstraction
24. vector adapter
25. hybrid retrieval (router fan-out + RRF) and token-budgeted context assembly   [extended]
26. consolidation (with provenance)
27. reflection + replay
28. outcome feedback, utility, strength/decay/reinforcement  [new]
29. PostgreSQL
30. LangGraph integration
31. LangChain integration
32. OpenAI integration
33. Anthropic integration
34. MCP server (full)
35. REST API
36. observability
37. Docker
38. MemoryBench (full) + public benchmarks + benchmark cards
39. research/learned gates, cost-based and learned planner
40. MemoryQL, policy DSL, multi-agent scopes, Membrane Memory Format, TypeScript client   [new]
```

## README and documentation mandate

You MUST produce a **world-class `README.md`** as specified in **A29**. It is not optional and not an afterthought:

- Follow the A29.1 structure exactly (hero → demo GIF → 60-second explanation with the architecture figure → why → quickstart → `explain()` sample → concepts → integrations → storage → beyond chatbots → benchmarks → comparison → production → roadmap → docs/community/citation).
- A new reader must understand what Membrane is, why it matters, and how to try it within 60 seconds.
- The quickstart must work with `pip install membrane-memory`, no API key, no Docker, and no external service.
- Every code block in the README is executed in CI; a README that does not run fails the build.
- Every quality/performance claim links to a committed benchmark card; show where Membrane loses, too.
- Build the docs site, examples, and `docs/assets/membrane-architecture.png` alongside the code.

## Coding philosophy

Prefer:

```python
memory.recall(query)
```

over:

```python
memory.vector_db.search(...)
```

Prefer:

```python
store.search(...)
```

over vendor-specific APIs in core code.

Prefer:

```python
controller.read(...)
```

over hard-coded retrieval logic.

Every major subsystem must be replaceable.

---

# 70. Final Product Test

A developer should eventually be able to run:

```bash
pip install membrane-memory
```

and:

```python
from membrane import Memory

memory = Memory()

memory.remember(
    "The user prefers Python over JavaScript.",
    user_id="user-1"
)

memory.remember(
    "The user is building an AI agent.",
    user_id="user-1"
)

memories = memory.recall(
    "What technologies does this user prefer?",
    user_id="user-1"
)

print(memories)
```

Then they should be able to switch storage:

```python
Memory.from_config("postgres.yaml")
```

or:

```python
Memory.from_config("qdrant.yaml")
```

without changing application code.

They should be able to switch agent framework without changing memory semantics.

They should be able to use Membrane with:

```text
LangGraph
LangChain
OpenAI SDK
Anthropic SDK
MCP
custom agents
```

without Membrane becoming an agent framework itself.


> **Membrane amendment (§70)** — The final product test also includes:

```python
result = memory.recall("What technologies does this user prefer?", user_id="user-1", return_trace=True)
print(memory.explain(result.recall_id))     # why these memories, scores, plan
print(memory.recall("...", user_id="user-1", at="2026-09-01"))   # bitemporal
```

...still with no API key, no Docker, and no external service.

---

# 71. North Star

The ultimate goal is not:

> "Build another RAG library."

The goal is:

> **Establish memory as a first-class infrastructure primitive for AI applications.**

The long-term abstraction should be as simple as:

```text
                 AI APPLICATION
                       |
                       v
              +----------------+
              |   Membrane      |
              +----------------+
              | Remember        |
              | Recall          |
              | Update          |
              | Forget          |
              | Consolidate     |
              | Reflect         |
              +----------------+
                       |
                       v
               Any Storage Layer
                       |
          +------------+------------+
          |            |            |
        SQL          Vector        Graph
```

If successful, developers should think:

> "I need memory for my agent."

and immediately think:

> **Membrane.**

That is the product objective.