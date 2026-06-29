"""Streamlit demo UI for the Day 08 LangGraph support agent — with live tracing.

Features
--------
- Run any query (or pick a sample scenario) through the compiled graph.
- LIVE TRACE: each super-step is streamed and rendered as a timeline (node, state delta, timing).
- Routing / retry / approval are visualised; metrics (route, retries, interrupts, nodes) summarised.
- Human-in-the-loop: enable "Manual approval" to pause risky actions at an interrupt() and
  Approve / Reject from the UI (resume via Command).
- Time travel: inspect the full checkpoint history of the last run.

Run:  make ui      (or)   streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import time
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import streamlit as st
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.scenarios import load_scenarios
from langgraph_agent_lab.state import Route, Scenario, initial_state

st.set_page_config(page_title="LangGraph Support Agent — Demo + Tracing", layout="wide")

ROUTE_COLOR = {
    "simple": "🟢",
    "tool": "🔵",
    "missing_info": "🟡",
    "risky": "🔴",
    "error": "🟠",
    "dead_letter": "⚫",
}
NODE_ICON = {
    "intake": "📥",
    "classify": "🧭",
    "tool": "🛠️",
    "evaluate": "🔎",
    "answer": "💬",
    "clarify": "❓",
    "risky_action": "⚠️",
    "approval": "🧑‍⚖️",
    "retry": "🔁",
    "dead_letter": "📨",
    "finalize": "🏁",
}


# ───────────────────────── helpers ─────────────────────────
def provider_label() -> str:
    if os.getenv("OLLAMA_MODEL"):
        return f"Ollama · {os.getenv('OLLAMA_MODEL')}"
    if os.getenv("GEMINI_API_KEY"):
        return f"Gemini · {os.getenv('LLM_MODEL', 'gemini-2.5-flash-lite')}"
    if os.getenv("OPENAI_API_KEY"):
        return f"OpenAI · {os.getenv('LLM_MODEL', 'gpt-4o-mini')}"
    if os.getenv("ANTHROPIC_API_KEY"):
        return f"Anthropic · {os.getenv('LLM_MODEL', 'claude')}"
    return "⚠️ no LLM configured"


def render_step(container: Any, idx: int, node: str, update: dict, dt_ms: int) -> None:
    icon = NODE_ICON.get(node, "•")
    with container:
        with st.expander(f"{icon}  **{idx:02d} · {node}**  ·  {dt_ms} ms", expanded=False):
            for key, val in (update or {}).items():
                if key == "events":
                    for ev in val:
                        st.caption(f"event · {ev.get('event_type')} · {ev.get('message')}")
                else:
                    st.write(f"`{key}` →", val)


def summarise(state: dict) -> None:
    route = state.get("route", "")
    events = state.get("events", []) or []
    nodes = [e.get("node") for e in events]
    retries = sum(1 for n in nodes if n == "retry")
    approvals = sum(1 for n in nodes if n == "approval")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Route", f"{ROUTE_COLOR.get(route, '')} {route}")
    c2.metric("Nodes visited", len(nodes))
    c3.metric("Retries", retries)
    c4.metric("Approvals", approvals)
    if state.get("approval"):
        ap = state["approval"]
        st.info(f"🧑‍⚖️ Approval: **{'APPROVED' if ap.get('approved') else 'REJECTED'}** "
                f"by {ap.get('reviewer')} — {ap.get('comment')}")
    if state.get("final_answer"):
        st.success(state["final_answer"])
    elif state.get("pending_question"):
        st.warning(state["pending_question"])


def show_history(graph: Any, config: dict) -> None:
    try:
        history = list(graph.get_state_history(config))
    except Exception as exc:  # noqa: BLE001
        st.caption(f"(state history unavailable: {exc})")
        return
    st.caption(f"🕰️ {len(history)} checkpoints saved (time travel)")
    rows = []
    for i, snap in enumerate(reversed(history)):
        nxt = snap.next or ("END",)
        rows.append({"#": i, "next": " → ".join(nxt), "attempt": snap.values.get("attempt"),
                     "route": snap.values.get("route")})
    st.dataframe(rows, use_container_width=True, hide_index=True)


# ───────────────────────── sidebar ─────────────────────────
st.sidebar.title("⚙️ Configuration")
st.sidebar.write(f"**LLM:** {provider_label()}")
checkpointer_kind = st.sidebar.selectbox("Checkpointer", ["memory", "sqlite"], index=0)
manual_approval = st.sidebar.toggle("Manual approval (HITL interrupt)", value=False,
                                    help="Pause risky actions at interrupt() and approve/reject here.")
max_attempts = st.sidebar.slider("max_attempts (retry bound)", 1, 5, 3)

st.sidebar.divider()
st.sidebar.caption("Sample scenarios")
samples = {}
try:
    for sc in load_scenarios("data/sample/scenarios.jsonl"):
        samples[f"{sc.id} — {sc.query[:30]}"] = sc.query
except Exception:  # noqa: BLE001
    pass
picked = st.sidebar.selectbox("Load a sample", ["—"] + list(samples.keys()))


# ───────────────────────── main ─────────────────────────
st.title("🤖 LangGraph Support Agent")
st.caption("Day 08 lab · conditional routing · bounded retry · HITL approval · live tracing")

default_query = samples.get(picked, "") if picked != "—" else ""
query = st.text_area("Support ticket / query", value=default_query, height=80,
                     placeholder="e.g. Please lookup order status for order 12345")

run = st.button("▶️ Run through graph", type="primary", use_container_width=True)


def build() -> Any:
    # Manual approval requires the interrupt() branch in approval_node.
    os.environ["LANGGRAPH_INTERRUPT"] = "true" if manual_approval else ""
    return build_graph(checkpointer=build_checkpointer(checkpointer_kind))


def stream_run(graph: Any, payload: Any, config: dict, trace_box: Any, start_idx: int) -> int:
    """Stream updates into the trace timeline; returns the next step index."""
    idx = start_idx
    t = time.perf_counter()
    for chunk in graph.stream(payload, config=config, stream_mode="updates"):
        now = time.perf_counter()
        dt = int((now - t) * 1000)
        t = now
        for node, update in chunk.items():
            if node == "__interrupt__":
                continue
            render_step(trace_box, idx, node, update, dt)
            idx += 1
    return idx


if run and query.strip():
    graph = build()
    scenario = Scenario(id="ui", query=query, expected_route=Route.SIMPLE, max_attempts=max_attempts)
    state = initial_state(scenario)
    thread_id = f"ui-{int(time.time())}"
    config = {"configurable": {"thread_id": thread_id}}
    st.session_state["config"] = config
    st.session_state["graph"] = graph

    left, right = st.columns([3, 2])
    with left:
        st.subheader("🧵 Live trace")
        trace_box = st.container()
    with right:
        st.subheader("📊 Result")
        result_box = st.container()

    with st.spinner("Running graph…"):
        idx = stream_run(graph, state, config, trace_box, 0)

    snapshot = graph.get_state(config)
    if snapshot.next:  # paused at an interrupt (manual approval)
        st.session_state["paused"] = True
        st.session_state["next_idx"] = idx
        pending = state["query"]
        try:
            pending = snapshot.tasks[0].interrupts[0].value.get("proposed_action", pending)
        except Exception:  # noqa: BLE001
            pass
        st.warning(f"⏸️ Paused for approval:\n\n**{pending}**")
    else:
        st.session_state["paused"] = False
        with result_box:
            summarise(snapshot.values)
        with st.expander("🕰️ State history (time travel)"):
            show_history(graph, config)

# ── resume after interrupt ──
if st.session_state.get("paused"):
    st.subheader("🧑‍⚖️ Human approval required")
    col_a, col_r = st.columns(2)
    decision = None
    if col_a.button("✅ Approve", use_container_width=True):
        decision = True
    if col_r.button("❌ Reject", use_container_width=True):
        decision = False
    if decision is not None:
        graph = st.session_state["graph"]
        config = st.session_state["config"]
        trace_box = st.container()
        with st.spinner("Resuming…"):
            stream_run(graph, Command(resume={"approved": decision}),
                       config, trace_box, st.session_state.get("next_idx", 0))
        st.session_state["paused"] = False
        snapshot = graph.get_state(config)
        st.subheader("📊 Result")
        summarise(snapshot.values)
        with st.expander("🕰️ State history (time travel)"):
            show_history(graph, config)
