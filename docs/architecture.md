# Membrane Architecture

Membrane separates the **memory control plane** from the **memory data plane**.

## Control plane

- LSTM-inspired memory controller
- Forget gate
- Write gate
- Update gate
- Read gate
- Policy engine
- Query planner
- Memory router
- Utility and decay
- Consolidation
- Provenance
- Explainability
- Outcome learning

## Data plane

Pluggable substrates:

- Structured
- Semantic
- Temporal
- Graph
- KV / Working
- Episodic / Object

## LSTM-inspired controller

Let:

```text
z_t = [q_t ; x_t ; c_t ; h_{t-1}]
```

Gates:

```text
f_t = σ(W_f z_t + b_f)
i_t = σ(W_i z_t + b_i)
M̃_t = φ(W_m z_t + b_m)
u_t = σ(W_u z_t + b_u)
```

State update:

```text
M̄_t = f_t ⊙ M_{t-1} + i_t ⊙ M̃_t
M_t = (1-u_t) ⊙ M_{t-1} + u_t ⊙ M̄_t
```

Read:

```text
o_t = σ(W_o [h_t ; q_t] + b_o)
R_t = o_t ⊙ Retrieve(Plan(q_t, c_t), M)
```

These equations define controller policy signals. External databases are not treated as literal LSTM tensors.

## Planner

The planner selects the minimum substrate set for the query:

```text
semantic
structured
temporal
graph
state / KV
episodic
hybrid
```

Exact facts stay structured. Similarity search stays semantic. History stays temporal. Relationships stay graph-based.

## Lifecycle

```text
CANDIDATE → ACTIVE → UPDATED / SUPERSEDED → ARCHIVED → FORGOTTEN
                                               → QUARANTINED for unsafe observations
```

Every transition should be event logged and explainable.

## Research hypothesis

A system that dynamically manages **read + write + update + forget + consolidate + route** across heterogeneous memory should improve temporal correctness, contradiction handling, context efficiency, latency, and memory utility over retrieval-only approaches.
