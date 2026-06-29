# Day 08 Lab — LangGraph Agentic Orchestration

Build a production-style LangGraph workflow for a support-ticket agent with state management, conditional routing, retry loops, human-in-the-loop approval, persistence, and metrics.

This is a **starter skeleton**. All node implementations, routing logic, and graph wiring are left as `TODO(student)` — you must build them from scratch.

---

## Submission

| | |
|---|---|
| **Student** | Đào Văn Tuân |
| **Student ID** | 2A202600609 |
| **Lab** | Day 08 — LangGraph Agentic Orchestration |
| **Date** | 2026-06-29 |
| **LLM** | Local Ollama · `kamekichi128/qwen3-4b-instruct-2507` (no cloud key required) |

### Status — all TODOs implemented ✅

| Component | File | Status |
|---|---|---|
| State schema (+4 fields: `evaluation_result`, `pending_question`, `proposed_action`, `approval`) | `src/.../state.py` | ✅ |
| 10 nodes — LLM `classify` (structured output) + LLM `answer` (grounded) | `src/.../nodes.py` | ✅ |
| 4 routing functions (bounded retry, approval gate) | `src/.../routing.py` | ✅ |
| Graph wiring — 11 nodes, all paths terminate at `finalize → END` | `src/.../graph.py` | ✅ |
| SQLite checkpointer (WAL) | `src/.../persistence.py` | ✅ |
| Report renderer | `src/.../report.py` | ✅ |

### Results

- **Scenario routing: 7/7 correct — success_rate = 100%** (`outputs/metrics.json`)
- `total_retries = 3`, `total_interrupts = 2` (S04, S06 approvals)
- **Tests: 25/25 pass** (19 unit + 6 end-to-end smoke running a real LLM)
- `make grade-local` → metrics valid

### Extensions (bonus)

- **SQLite persistence + time travel** — 12 checkpoints captured per error run (`outputs/state_history.txt`)
- **Real HITL** — `LANGGRAPH_INTERRUPT=true` pauses risky actions at `interrupt()` and resumes via `Command`
- **Streamlit demo UI with live tracing** — `make ui` (see [Demo UI](#demo-ui-streamlit--tracing))
- **Mermaid graph diagram** — `outputs/graph.mmd` via `draw_mermaid()`

### Reproduce

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e '.[dev,ollama,sqlite,ui]'
ollama pull kamekichi128/qwen3-4b-instruct-2507
cp .env.example .env   # set OLLAMA_MODEL=kamekichi128/qwen3-4b-instruct-2507:latest
make test && make run-scenarios && make grade-local
make ui                # interactive demo
```

---

## How you will be graded

| Category | Points | What we look for |
|---|---:|---|
| Architecture & state schema | 15 | Typed state with correct reducers, student-added fields, lean serializable state |
| Graph construction & wiring | 15 | All nodes registered, edges correct, conditional edges work, graph compiles |
| LLM integration | 15 | classify_node + answer_node use real LLM calls (structured output, grounded generation) |
| Graph behavior | 20 | All scenario routes correct, bounded retry loop, HITL approval path, all routes terminate |
| Persistence & recovery | 10 | Checkpointer wired, thread_id per run, state history or crash-resume evidence |
| Metrics & tests | 15 | `metrics.json` valid, scenario coverage, tests pass, meaningful counts |
| Report & demo | 10 | Architecture explanation, metrics table, failure analysis, improvement ideas |

**Grade bands:**
- **90–100**: Production-quality graph + LLM integration + metrics + report + at least one bonus extension
- **75–89**: Core graph works with LLM, metrics valid, report explains trade-offs
- **60–74**: Graph mostly works but LLM integration, persistence, or report incomplete
- **< 60**: Does not run, hard-codes scenarios, or lacks LLM integration/metrics/report

> **Critical rule**: Do NOT hard-code answers to specific scenario queries. Your graph must route based on **LLM classification and state logic**, not by matching exact scenario IDs. We grade with additional hidden scenarios.

---

## LLM Integration Requirements

This lab requires real LLM API calls in specific nodes:

| Node | Requirement | Pattern |
|---|---|---|
| `classify_node` | **MUST use LLM** | Structured output (`.with_structured_output()`) for intent classification |
| `answer_node` | **MUST use LLM** | Grounded response generation using tool_results/context |
| `evaluate_node` | **SHOULD use LLM** (bonus) | LLM-as-judge to evaluate tool results quality |

A helper is provided in `src/langgraph_agent_lab/llm.py` — it reads your config from `.env` and returns a LangChain chat model. It supports **local Ollama** (no API key) as well as Gemini / OpenAI / Anthropic, selected automatically by which variable is set.

```bash
# Option 1 — Local Ollama (no API key, used for this submission)
pip install langchain-ollama
ollama pull kamekichi128/qwen3-4b-instruct-2507   # or any chat model
# .env:  OLLAMA_MODEL=kamekichi128/qwen3-4b-instruct-2507:latest

# Option 2 — Cloud provider
pip install langchain-openai      # for OpenAI
# OR
pip install langchain-anthropic   # for Anthropic
# OR
pip install langchain-google-genai  # for Gemini

# Configure .env
cp .env.example .env
# Edit .env and set OLLAMA_MODEL  (or GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY)
```

> **This submission runs on local Ollama** (`kamekichi128/qwen3-4b-instruct-2507`). `get_llm()`
> prefers Ollama whenever `OLLAMA_MODEL` is set, so no cloud key is required to reproduce results.

---

## Understanding `scenarios.jsonl`

The file `data/sample/scenarios.jsonl` contains **7 sample scenarios** your graph must handle:

```jsonl
{"id":"S01_simple",      "query":"How do I reset my password?",                          "expected_route":"simple"}
{"id":"S02_tool",        "query":"Please lookup order status for order 12345",            "expected_route":"tool"}
{"id":"S03_missing",     "query":"Can you fix it?",                                      "expected_route":"missing_info"}
{"id":"S04_risky",       "query":"Refund this customer and send confirmation email",      "expected_route":"risky"}
{"id":"S05_error",       "query":"Timeout failure while processing request",              "expected_route":"error"}
{"id":"S06_delete",      "query":"Delete customer account after support verification",    "expected_route":"risky"}
{"id":"S07_dead_letter", "query":"System failure cannot recover after multiple attempts", "expected_route":"error", "max_attempts":1}
```

### What each field means

| Field | Purpose |
|---|---|
| `id` | Unique scenario identifier — used in metrics output |
| `query` | The user's support-ticket text — input to your graph |
| `expected_route` | Which route your `classify_node` should pick: `simple`, `tool`, `missing_info`, `risky`, or `error` |
| `requires_approval` | If `true`, your graph must hit the approval/HITL node before answering |
| `should_retry` | If `true`, scenario simulates transient tool failure requiring retry |
| `max_attempts` | Override retry limit (default 3). S07 sets this to 1, so it exhausts retries immediately → dead letter |
| `tags` | Descriptive labels for your reference |

### How scenarios flow through your code

```
scenarios.jsonl  →  scenarios.py loads them  →  cli.py runs each through your graph
                                              →  metrics.py collects results
                                              →  outputs/metrics.json
```

1. `make run-scenarios` reads `data/sample/scenarios.jsonl`
2. For each scenario, it calls `initial_state(scenario)` → `graph.invoke(state)`
3. After execution, it checks: did `actual_route` match `expected_route`? Did HITL fire when required?
4. Results go to `outputs/metrics.json`

### How to design your classification

Your `classify_node` should use an LLM to classify intent. Design a prompt that routes queries:

| Route | Intent |
|---|---|
| `risky` | Actions with side effects: refunds, deletions, sending emails, cancellations |
| `tool` | Information lookups: order status, tracking, search queries |
| `missing_info` | Vague/incomplete queries lacking actionable context |
| `error` | System failures: timeouts, crashes, service unavailable |
| `simple` | General questions answerable without tools or actions |

**Priority matters**: risky > tool > missing_info > error > simple. Design your LLM prompt to respect this priority.

### Adding your own test scenarios

You can add extra lines to `scenarios.jsonl` to test edge cases:

```jsonl
{"id":"S08_custom","query":"Cancel my subscription immediately","expected_route":"risky","requires_approval":true,"tags":["custom"]}
```

The grading script will also test with scenarios you haven't seen.

---

## Quick start

```bash
# venv (Windows shown; use source .venv/bin/activate on macOS/Linux)
python -m venv .venv
.venv/Scripts/activate
pip install -e '.[dev,ollama,sqlite,ui]'    # local Ollama + persistence + demo UI
# (for cloud instead: pip install -e '.[dev]' && pip install langchain-openai)

# Configure LLM
cp .env.example .env
# Edit .env — set OLLAMA_MODEL=...  (or a cloud *_API_KEY)

# Verify setup
make test          # all tests pass once nodes/routing/graph are implemented
make run-scenarios # → outputs/metrics.json + reports/lab_report.md
make ui            # interactive demo with live tracing
```

---

## Step-by-step workflow

### Phase 1: State + nodes (0–90 min) — worth 30 points

1. **`state.py`** — Review existing fields. Add missing fields as you discover them:
   - `evaluation_result` for retry loop gate
   - `pending_question` for clarification flow
   - `proposed_action` for risky action flow
   - `approval` for HITL decisions

2. **`llm.py`** — Review the helper. Configure `.env` with your API key.

3. **`nodes.py`** — Implement all 10 node functions:
   - `classify_node`: **LLM + structured output** for intent classification
   - `tool_node`: mock tool with error simulation
   - `evaluate_node`: tool result quality check (LLM-as-judge for bonus)
   - `answer_node`: **LLM-generated** grounded response
   - `ask_clarification_node`: generate clarification question
   - `risky_action_node`: prepare action for approval
   - `approval_node`: mock approval with optional interrupt()
   - `retry_or_fallback_node`: increment attempt counter
   - `dead_letter_node`: handle max retry exhaustion
   - `finalize_node`: emit final audit event

### Phase 2: Routing + graph (90–150 min) — worth 35 points

4. **`routing.py`** — Implement all 4 routing functions from scratch
5. **`graph.py`** — Build the complete StateGraph:
   - Import and register all 11 nodes
   - Wire fixed + conditional edges
   - All paths must terminate at finalize → END
6. **Verify**: `make test` and `make run-scenarios`

### Phase 3: Persistence (150–180 min) — worth 10 points

7. **`persistence.py`** — Implement SQLite checkpointer
   - Show evidence: thread_id per run, state history, or crash-resume

### Phase 4: Metrics & report (180–240 min) — worth 25 points

8. **`report.py`** — Implement `render_report()` from metrics data
9. **Run**: `make run-scenarios` → `outputs/metrics.json`
10. **Validate**: `make grade-local`
11. **Report**: Fill `reports/lab_report.md`

### Phase 5: Extensions (240+ min) — push toward 90+

Pick one or more:
- **Parallel fan-out**: Use `Send()` for concurrent tool calls
- **Real HITL**: `LANGGRAPH_INTERRUPT=true` with `interrupt()`
- **Streamlit UI**: Build approval/reject interface
- **Time travel**: `get_state_history()` replay
- **Crash recovery**: SQLite checkpoint survives process kill
- **Graph diagram**: `graph.get_graph().draw_mermaid()`

---

## Make commands

| Command | What it does |
|---|---|
| `make install` | Install project + dev dependencies |
| `make test` | Run pytest |
| `make lint` | Run ruff linter |
| `make typecheck` | Run mypy type checker |
| `make run-scenarios` | Execute all scenarios → `outputs/metrics.json` (+ `reports/lab_report.md`) |
| `make grade-local` | Validate metrics.json schema |
| `make ui` | Launch the Streamlit demo UI with live tracing |
| `make demo` | Generate extension evidence (Mermaid diagram + SQLite state history) |
| `make clean` | Remove caches and generated files |

---

## Demo UI (Streamlit + tracing)

An interactive demo is provided in [`app/streamlit_app.py`](app/streamlit_app.py):

```bash
pip install -e '.[ui]'      # streamlit + python-dotenv
make ui                     # → http://localhost:8501
```

What it shows:

- **Live trace** — every graph super-step is streamed and rendered as a timeline (node icon,
  state delta, per-step latency), so you can watch routing, the retry loop, and finalize fire.
- **Result panel** — route badge, nodes-visited / retries / approvals metrics, grounded answer.
- **Human-in-the-loop** — toggle *Manual approval* to pause risky actions at `interrupt()` and
  **Approve / Reject** from the UI (resumes via `Command(resume=…)`).
- **Time travel** — inspect the full checkpoint history of the last run (use the `sqlite`
  checkpointer in the sidebar for persistent, crash-resumable runs).
- Pick a sample from `data/sample/scenarios.jsonl` or type any query.

---

## Submission checklist

- [ ] All `TODO(student)` sections implemented
- [ ] `.env` configured with LLM API key
- [ ] `classify_node` uses real LLM call with structured output
- [ ] `answer_node` uses real LLM call for grounded responses
- [ ] `make test` passes
- [ ] `make run-scenarios` generates valid `outputs/metrics.json`
- [ ] `make grade-local` passes validation
- [ ] `reports/lab_report.md` completed with architecture, metrics, and analysis
- [ ] Can explain at least one route and one failure mode during demo

**For 90+ points, also include:**
- [ ] At least one bonus extension (persistence, parallel fan-out, HITL, time travel, diagram)
- [ ] Evidence of extension in report (screenshot, log output, or diagram)

---

## Common pitfalls

1. **Missing state fields**: The starter intentionally omits some fields from `AgentState`. You must add `evaluation_result`, `pending_question`, `proposed_action`, and `approval` as you implement nodes that need them.

2. **LLM structured output**: Use `.with_structured_output(YourModel)` to get reliable classification. Raw text parsing is fragile and will fail on hidden test scenarios.

3. **Unbounded retry**: Always check `attempt < max_attempts` in `route_after_retry`. Without this bound, error scenarios loop forever.

4. **Graph wiring**: Every path must end at `finalize → END`. Missing this means the graph hangs for some scenarios.

5. **SqliteSaver API**: In `langgraph-checkpoint-sqlite` 3.x, use `SqliteSaver(conn=sqlite3.connect(...))` not `SqliteSaver.from_conn_string()`.

6. **API key not set**: If you get "No LLM API key found", check your `.env` file and make sure it's loaded (use `python-dotenv` or export manually).
