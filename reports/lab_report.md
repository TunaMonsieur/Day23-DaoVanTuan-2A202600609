# Day 08 Lab Report — LangGraph Agentic Orchestration

## 1. Team / student

- Name: Dao Van Tuan
- Repo/commit: Day23-DaoVanTuan-2A202600609
- Date: 2026-06-29

## 2. Architecture

The graph is a `StateGraph(AgentState)` with 11 nodes:

```
START → intake → classify → [route_after_classify]
  simple       → answer → finalize → END
  tool         → tool → evaluate → [route_after_evaluate]
                                     success     → answer → finalize → END
                                     needs_retry → retry → [route_after_retry]
                                                            attempt<max → tool (loop)
                                                            else        → dead_letter → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → [route_after_approval]
                                            approved → tool → evaluate → answer → finalize → END
                                            rejected → clarify → finalize → END
  error        → retry → [route_after_retry] → tool → evaluate → ... (bounded loop)
```

`classify_node` and `answer_node` make real LLM calls (local Ollama
`kamekichi128/qwen3-4b-instruct-2507`, swappable for Gemini/OpenAI/Anthropic via `get_llm`).
Classification uses
`.with_structured_output(Classification)` so the route is always a valid enum. Routing is done
by four pure functions used in `add_conditional_edges`. Every path terminates at
`finalize → END`.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| query / route / risk_level | overwrite | current request + classification |
| attempt / max_attempts | overwrite | bounded retry counter + limit |
| evaluation_result | overwrite | retry-loop gate read by route_after_evaluate |
| pending_question | overwrite | clarification text for missing_info |
| proposed_action | overwrite | risky action awaiting approval |
| approval | overwrite | HITL decision read by route_after_approval |
| final_answer | overwrite | terminal output |
| messages / tool_results / errors / events | append (add) | audit trail across nodes |

## 4. Scenario results

- Total scenarios: **7**
- Success rate: **100%**
- Avg nodes visited: **6.4**
- Total retries: **3**
- Total interrupts (approvals): **2**
- Resume success: **False**

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | ✅ | 0 | 0 |
| S02_tool | tool | tool | ✅ | 0 | 0 |
| S03_missing | missing_info | missing_info | ✅ | 0 | 0 |
| S04_risky | risky | risky | ✅ | 0 | 1 |
| S05_error | error | error | ✅ | 2 | 0 |
| S06_delete | risky | risky | ✅ | 0 | 1 |
| S07_dead_letter | error | error | ✅ | 1 | 0 |

## 5. Failure analysis

1. **Tool failure / retry exhaustion** — `tool_node` simulates transient errors for the `error`
   route. `evaluate_node` flags `needs_retry`; `retry_or_fallback_node` increments `attempt` and
   `route_after_retry` is bounded (`attempt < max_attempts`). When the limit is hit (e.g. S07 with
   `max_attempts=1`) the flow falls through to `dead_letter_node`, which escalates rather than
   looping forever.
2. **Risky action without approval** — risky requests cannot reach `tool` directly; they must pass
   `risky_action → approval`. `route_after_approval` only proceeds when `approval.approved` is true,
   otherwise it diverts to `clarify`. This guarantees no destructive action runs unapproved.

## 6. Persistence / recovery evidence

A SQLite checkpointer (`SqliteSaver` with WAL mode) is wired in `persistence.py`. Each scenario runs
under its own `thread_id` (`thread-<scenario_id>`), so state is checkpointed per super-step and can be
inspected with `graph.get_state_history(config)` or resumed after a crash.

Evidence (`scripts/demo_extensions.py` → `outputs/state_history.txt`): a single error-route run
produced **12 saved checkpoints**, capturing the full bounded retry loop (`attempt` 0 → 1 → 2):

```
[04] next=('tool',)     attempt=2   [05] next=('retry',)    attempt=1
[06] next=('evaluate',) attempt=1   [07] next=('tool',)     attempt=1
[08] next=('retry',)    attempt=0   ...
```

Because every super-step is persisted to `outputs/demo_checkpoints.sqlite`, the run is resumable
after a process kill and replayable via time-travel from any checkpoint.

## 7. Extension work

- **SQLite persistence backend** (`SqliteSaver`, WAL) — crash-resumable checkpoints, evidence above.
- **Time travel** — `get_state_history()` replays all 12 checkpoints of a run.
- **Graph diagram** — Mermaid exported to `outputs/graph.mmd` via `draw_mermaid()`.
- **Real HITL** — `LANGGRAPH_INTERRUPT=true` switches `approval_node` to `interrupt()`/resume.
- **Streamlit demo UI** (`app/streamlit_app.py`, `make ui`) — live per-super-step trace timeline,
  route/retry/approval metrics, in-UI Approve/Reject on the interrupt, and a time-travel view of
  the checkpoint history.

## 8. Improvement plan

With one more day: replace the heuristic `evaluate_node` with an LLM-as-judge, add structured
tool schemas with real retries/backoff, expose the approval interrupt through a small Streamlit UI,
and add tracing/observability for latency per node.
