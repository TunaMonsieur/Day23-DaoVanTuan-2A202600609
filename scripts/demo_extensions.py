"""Bonus-extension evidence generator.

Produces:
1. outputs/graph.mmd        — Mermaid diagram of the compiled graph.
2. outputs/state_history.txt — SQLite-checkpointed run + full state history (time travel).

Run:  python scripts/demo_extensions.py
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)


def dump_mermaid() -> None:
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    mermaid = graph.get_graph().draw_mermaid()
    (OUT / "graph.mmd").write_text(mermaid, encoding="utf-8")
    print(f"[1] Wrote Mermaid diagram -> {OUT / 'graph.mmd'}")


def demo_sqlite_persistence() -> None:
    """Run a scenario under a SQLite checkpointer and replay its state history."""
    db = OUT / "demo_checkpoints.sqlite"
    if db.exists():
        db.unlink()
    checkpointer = build_checkpointer("sqlite", str(db))
    graph = build_graph(checkpointer=checkpointer)

    scenario = Scenario(id="demo_error", query="Timeout failure while processing", expected_route=Route.ERROR)
    state = initial_state(scenario)
    config = {"configurable": {"thread_id": state["thread_id"]}}
    final = graph.invoke(state, config=config)

    lines: list[str] = []
    lines.append(f"thread_id      : {state['thread_id']}")
    lines.append(f"sqlite db      : {db} (exists={db.exists()})")
    lines.append(f"final route    : {final.get('route')}")
    lines.append(f"final answer   : {final.get('final_answer')}")
    lines.append(f"attempts       : {final.get('attempt')}")
    lines.append("")
    lines.append("=== STATE HISTORY (time travel — newest first) ===")
    history = list(graph.get_state_history(config))
    lines.append(f"checkpoints saved: {len(history)}")
    for i, snap in enumerate(history):
        nxt = snap.next or ("END",)
        lines.append(f"  [{i:02d}] next={nxt} attempt={snap.values.get('attempt')} route={snap.values.get('route')!r}")

    text = "\n".join(lines)
    (OUT / "state_history.txt").write_text(text, encoding="utf-8")
    print(f"[2] Wrote SQLite persistence + state-history evidence -> {OUT / 'state_history.txt'}")
    print(f"    checkpoints saved = {len(history)} (proves checkpointer is recording every super-step)")


if __name__ == "__main__":
    dump_mermaid()
    demo_sqlite_persistence()
    print("Done.")
