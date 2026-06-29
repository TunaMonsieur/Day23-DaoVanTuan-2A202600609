"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, make_event

VALID_ROUTES = {"simple", "tool", "missing_info", "risky", "error"}


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Structured-output schema for LLM classification ─────────────────
class Classification(BaseModel):
    """Schema the LLM must fill — guarantees a valid enum route."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="The single best route for the support ticket."
    )
    reason: str = Field(default="", description="One short sentence justifying the route.")


_CLASSIFY_SYSTEM = """You are a support-ticket router. Classify the user's request into EXACTLY one route.

Routes (apply this STRICT priority order — pick the highest one that matches):
1. risky        — actions with side effects: refunds, deletions, cancellations, sending emails,
                  charging cards, account changes, anything destructive or irreversible.
2. tool         — read-only information lookups: order status, tracking number, account info,
                  searching records. No side effects, but needs a tool/data source.
3. missing_info — vague or incomplete requests lacking actionable context ("fix it", "help",
                  "it broke") where you cannot tell what the user wants.
4. error        — system/infrastructure failures: timeouts, crashes, "service unavailable",
                  "cannot recover", processing failures.
5. simple       — general questions answerable from knowledge alone, no tool or action
                  (e.g. "how do I reset my password?").

Priority means: if a request both looks up data AND performs a risky action, choose risky.
Return only the structured route."""


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "")
    llm = get_llm()
    classifier = llm.with_structured_output(Classification)
    result: Classification = classifier.invoke(
        [
            ("system", _CLASSIFY_SYSTEM),
            ("human", f"Support ticket: {query!r}\nClassify it."),
        ]
    )
    route = result.route if result.route in VALID_ROUTES else "simple"
    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event("classify", "completed", f"route={route}", reason=result.reason)
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with transient-failure simulation for retry testing."""
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    query = state.get("query", "")
    if route == "error" and attempt < 2:
        result = f"ERROR: transient tool failure (attempt={attempt}) for query: {query[:40]}"
        event = make_event("tool", "failed", "tool returned transient error", attempt=attempt)
    else:
        result = f"TOOL_OK: result for '{query[:40]}' (attempt={attempt})"
        event = make_event("tool", "completed", "tool returned success", attempt=attempt)
    return {"tool_results": [result], "events": [event]}


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Heuristic: a tool result containing 'ERROR' needs a retry, otherwise it is a success.
    (Acceptable for base score; an LLM-as-judge could replace this for bonus.)
    """
    tool_results = state.get("tool_results", []) or []
    latest = tool_results[-1] if tool_results else ""
    if "ERROR" in latest:
        evaluation = "needs_retry"
        event = make_event("evaluate", "completed", "tool result unsatisfactory → retry")
    else:
        evaluation = "success"
        event = make_event("evaluate", "completed", "tool result satisfactory")
    return {"evaluation_result": evaluation, "events": [event]}


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM, grounded in available context."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval")
    context_lines = []
    if tool_results:
        context_lines.append("Tool results:\n- " + "\n- ".join(tool_results))
    if approval:
        context_lines.append(
            f"Approval: approved={approval.get('approved')} by {approval.get('reviewer')}"
        )
    context = "\n\n".join(context_lines) if context_lines else "(no tool results)"

    llm = get_llm()
    response = llm.invoke(
        [
            (
                "system",
                "You are a helpful support agent. Write a concise, accurate reply to the "
                "customer GROUNDED ONLY in the provided context. Do not invent order numbers "
                "or facts not present in the context. 2-4 sentences.",
            ),
            ("human", f"Customer request: {query!r}\n\nContext:\n{context}\n\nWrite the reply."),
        ]
    )
    answer = response.content if hasattr(response, "content") else str(response)
    return {
        "final_answer": answer,
        "messages": ["answer:generated"],
        "events": [make_event("answer", "completed", "LLM grounded answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    question = (
        "I'd like to help, but I need more detail. Could you tell me which order, account, "
        f"or issue this refers to, and what outcome you want? (Your message was: {query!r})"
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "messages": ["clarify:asked"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed = (
        f"Proposed action requiring approval: '{query}'. This has side effects "
        "(refund/delete/email/cancel) and must be confirmed by a human reviewer before execution."
    )
    return {
        "proposed_action": proposed,
        "messages": ["risky:prepared"],
        "events": [make_event("risky_action", "completed", "risky action prepared for approval")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock approval (approved=True) so tests/CI run offline.
    Extension: if LANGGRAPH_INTERRUPT=true, pause via langgraph.types.interrupt() for real HITL.
    """
    proposed = state.get("proposed_action", "")
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt({"proposed_action": proposed, "question": "Approve this action?"})
        approval = {
            "approved": bool(decision.get("approved", True))
            if isinstance(decision, dict)
            else bool(decision),
            "reviewer": "human",
            "comment": "via interrupt",
        }
    else:
        approval = {"approved": True, "reviewer": "mock-reviewer", "comment": "auto-approved"}
    return {
        "approval": approval,
        "messages": [f"approval:{approval['approved']}"],
        "events": [
            make_event("approval", "completed", f"approved={approval['approved']}",
                       reviewer=approval["reviewer"])
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt (increments the bounded retry counter)."""
    attempt = state.get("attempt", 0) + 1
    error_msg = f"transient failure — retry attempt {attempt}"
    return {
        "attempt": attempt,
        "errors": [error_msg],
        "messages": [f"retry:{attempt}"],
        "events": [make_event("retry", "completed", error_msg, attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded (retry → fallback → dead letter)."""
    attempt = state.get("attempt", 0)
    answer = (
        "We were unable to complete your request automatically after "
        f"{attempt} attempt(s). It has been escalated to a human support engineer "
        "and you will be contacted shortly."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:escalated"],
        "events": [make_event("dead_letter", "completed", "max retries exceeded → escalated")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "messages": ["finalize:done"],
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route", ""),
                attempts=state.get("attempt", 0),
            )
        ],
    }
