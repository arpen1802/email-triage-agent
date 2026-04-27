"""
Architecture 1 — Single structured call.

One LLM call per email. JSON output. No tools, no loop.
The boring baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import LLMClient, CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (REPO_ROOT / "prompts" / "arch1_system.txt").read_text()


def _format_email(email: dict) -> str:
    return (
        f"EMAIL_ID: {email['id']}\n"
        f"From: {email['from']}\n"
        f"To: {email.get('to', '')}\n"
        f"Subject: {email['subject']}\n"
        f"Date: {email.get('date', '')}\n\n"
        f"{email['body']}"
    )


def _parse_or_repair(text: str, email_id: str) -> dict:
    """Best-effort parse of the model's JSON output, with a sane fallback."""
    text = text.strip()
    # The model sometimes wraps JSON in fences despite instructions; strip them.
    if text.startswith("```"):
        text = text.strip("`")
        # remove an optional language tag on the first line
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if first.strip().lower() in {"json", ""}:
                text = rest
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                obj = {}
        else:
            obj = {}
    cat = obj.get("category")
    if cat not in CATEGORIES:
        cat = "archive"
    return {
        "email_id": obj.get("email_id", email_id),
        "category": cat,
        "confidence": float(obj.get("confidence", 0.5) or 0.5),
        "draft_reply": obj.get("draft_reply"),
        "reasoning": obj.get("reasoning", ""),
    }


def run(email: dict, client: LLMClient) -> dict:
    """Run Arch 1 on a single email. Returns a triage result with metadata."""
    messages = [{"role": "user", "content": _format_email(email)}]
    resp = client.complete(system=SYSTEM_PROMPT, messages=messages)
    parsed = _parse_or_repair(resp.text, email["id"])
    schema_ok = parsed["category"] in CATEGORIES
    return {
        **parsed,
        "_meta": {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "latency_ms": resp.latency_ms,
            "tool_calls": 0,
            "schema_ok": schema_ok,
            "stop_reason": resp.stop_reason,
        },
    }
