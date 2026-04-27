"""
Architecture 3 — Multi-step graph with self-correction.

Graph shape:
    triage  →  route  →  (if reply_now)  →  draft  →  critique
                                                         │
                                                  pass ──┴── revise (max 1) → END
                                       (else: archive / unsubscribe / reply_later) → END

Nodes (each is a pure function over the working state):
    triage_node      — classify; LLM call
    draft_node       — generate reply (only on reply_now)
    critique_node    — score draft against rubric; LLM call
    revise_node      — regenerate draft with critique guidance; LLM call

If `langgraph` is installed, run() executes the graph via langgraph.StateGraph.
Otherwise it falls back to a hand-rolled state machine with the same control
flow, so the harness can be exercised in mock mode without the dep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from llm_client import LLMClient, CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[1]
TRIAGE_PROMPT = (REPO_ROOT / "prompts" / "arch3_triage.txt").read_text()
DRAFT_PROMPT = (REPO_ROOT / "prompts" / "arch3_draft.txt").read_text()
CRITIQUE_PROMPT = (REPO_ROOT / "prompts" / "arch3_critique.txt").read_text()


def _format_email(email: dict) -> str:
    return (
        f"EMAIL_ID: {email['id']}\n"
        f"From: {email['from']}\n"
        f"To: {email.get('to', '')}\n"
        f"Subject: {email['subject']}\n"
        f"Date: {email.get('date', '')}\n\n"
        f"{email['body']}"
    )


def _parse_json(text: str, fallback: dict) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if first.strip().lower() in {"json", ""}:
                text = rest
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return dict(fallback)


class GraphState(TypedDict, total=False):
    email: dict
    client: LLMClient  # not serialisable, but langgraph allows arbitrary fields
    category: str
    confidence: float
    triage_reasoning: str
    draft: str | None
    critique: dict
    revisions_made: int
    stats: dict


def _stats_init() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_ms": 0.0,
        "tool_calls": 0,
        "node_calls": 0,
        "schema_ok": True,
    }


def _accumulate(state: GraphState, resp) -> None:
    s = state["stats"]
    s["input_tokens"] += resp.input_tokens
    s["output_tokens"] += resp.output_tokens
    s["cost_usd"] += resp.cost_usd
    s["latency_ms"] += resp.latency_ms
    s["node_calls"] += 1


# ---------------- Node implementations ----------------------------------

def triage_node(state: GraphState) -> GraphState:
    client: LLMClient = state["client"]
    email = state["email"]
    resp = client.complete(
        system=TRIAGE_PROMPT,
        messages=[{"role": "user", "content": _format_email(email)}],
    )
    _accumulate(state, resp)
    parsed = _parse_json(resp.text, {"email_id": email["id"], "category": "archive", "confidence": 0.5, "reasoning": ""})
    cat = parsed.get("category")
    if cat not in CATEGORIES:
        cat = "archive"
        state["stats"]["schema_ok"] = False
    state["category"] = cat
    state["confidence"] = float(parsed.get("confidence", 0.5) or 0.5)
    state["triage_reasoning"] = parsed.get("reasoning", "")
    return state


def draft_node(state: GraphState) -> GraphState:
    client: LLMClient = state["client"]
    email = state["email"]
    user_msg = (
        f"EMAIL_ID: {email['id']}\n"
        f"INCOMING EMAIL:\n{_format_email(email)}\n\n"
        f"Write the draft reply now."
    )
    resp = client.complete(system=DRAFT_PROMPT, messages=[{"role": "user", "content": user_msg}])
    _accumulate(state, resp)
    parsed = _parse_json(resp.text, {"email_id": email["id"], "draft_reply": ""})
    state["draft"] = parsed.get("draft_reply") or ""
    return state


def critique_node(state: GraphState) -> GraphState:
    client: LLMClient = state["client"]
    email = state["email"]
    user_msg = (
        f"EMAIL_ID: {email['id']}\n"
        f"INCOMING EMAIL:\n{_format_email(email)}\n\n"
        f"DRAFT TO REVIEW:\n{state.get('draft', '')}"
    )
    resp = client.complete(system=CRITIQUE_PROMPT, messages=[{"role": "user", "content": user_msg}])
    _accumulate(state, resp)
    parsed = _parse_json(resp.text, {"decision": "pass", "issues": [], "revision_guidance": ""})
    decision = parsed.get("decision", "pass")
    if decision not in {"pass", "revise"}:
        decision = "pass"
        state["stats"]["schema_ok"] = False
    state["critique"] = {
        "decision": decision,
        "issues": parsed.get("issues", []) or [],
        "revision_guidance": parsed.get("revision_guidance", "") or "",
    }
    return state


def revise_node(state: GraphState) -> GraphState:
    client: LLMClient = state["client"]
    email = state["email"]
    crit = state.get("critique", {})
    user_msg = (
        f"EMAIL_ID: {email['id']}\n"
        f"REVISION REQUESTED.\n"
        f"INCOMING EMAIL:\n{_format_email(email)}\n\n"
        f"PREVIOUS DRAFT:\n{state.get('draft', '')}\n\n"
        f"CRITIQUE ISSUES: {crit.get('issues', [])}\n"
        f"REVISION GUIDANCE: {crit.get('revision_guidance', '')}\n\n"
        f"Rewrite the draft addressing the issues above."
    )
    resp = client.complete(system=DRAFT_PROMPT, messages=[{"role": "user", "content": user_msg}])
    _accumulate(state, resp)
    parsed = _parse_json(resp.text, {"draft_reply": state.get("draft", "")})
    state["draft"] = parsed.get("draft_reply") or state.get("draft", "")
    state["revisions_made"] = state.get("revisions_made", 0) + 1
    return state


# ---------------- Routing ----------------------------------------------

def route_after_triage(state: GraphState) -> str:
    return "draft" if state.get("category") == "reply_now" else "end"


def route_after_critique(state: GraphState) -> str:
    crit = state.get("critique", {}) or {}
    if crit.get("decision") == "revise" and state.get("revisions_made", 0) < 1:
        return "revise"
    return "end"


# ---------------- LangGraph builder (optional) -------------------------

def _try_build_langgraph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError:
        return None
    g = StateGraph(GraphState)
    g.add_node("triage", triage_node)
    g.add_node("draft", draft_node)
    g.add_node("critique", critique_node)
    g.add_node("revise", revise_node)
    g.set_entry_point("triage")
    g.add_conditional_edges("triage", route_after_triage, {"draft": "draft", "end": END})
    g.add_edge("draft", "critique")
    g.add_conditional_edges("critique", route_after_critique, {"revise": "revise", "end": END})
    g.add_edge("revise", END)
    return g.compile()


# ---------------- Public entrypoint ------------------------------------

def run(email: dict, client: LLMClient) -> dict:
    initial: GraphState = {
        "email": email,
        "client": client,
        "draft": None,
        "revisions_made": 0,
        "stats": _stats_init(),
    }
    graph = _try_build_langgraph()
    if graph is not None:
        # Use LangGraph
        final = graph.invoke(initial)
    else:
        # Hand-rolled fallback with identical control flow
        s = triage_node(initial)
        if route_after_triage(s) == "draft":
            s = draft_node(s)
            s = critique_node(s)
            if route_after_critique(s) == "revise":
                s = revise_node(s)
        final = s

    cat = final.get("category", "archive")
    return {
        "email_id": email["id"],
        "category": cat,
        "confidence": float(final.get("confidence", 0.5) or 0.5),
        "draft_reply": final.get("draft") if cat == "reply_now" else None,
        "reasoning": final.get("triage_reasoning", ""),
        "_meta": {
            "input_tokens": final["stats"]["input_tokens"],
            "output_tokens": final["stats"]["output_tokens"],
            "cost_usd": final["stats"]["cost_usd"],
            "latency_ms": final["stats"]["latency_ms"],
            "tool_calls": 0,
            "schema_ok": final["stats"]["schema_ok"],
            "stop_reason": "end_turn",
            "node_calls": final["stats"]["node_calls"],
            "critique_fired": final.get("critique", {}).get("decision") == "revise",
            "revisions_made": final.get("revisions_made", 0),
        },
    }
