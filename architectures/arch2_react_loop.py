"""
Architecture 2 — ReAct loop with mock tools.

Hand-rolled tool-use loop. The model can call up to MAX_TOOL_CALLS tools per
email; after that the harness forces a final answer by removing tools from the
request and asking once more.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import LLMClient, CATEGORIES
from tools.mock_tools import TOOL_SCHEMAS, call_tool

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (REPO_ROOT / "prompts" / "arch2_system.txt").read_text()

MAX_TOOL_CALLS = 6


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
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first, rest = text.split("\n", 1)
            if first.strip().lower() in {"json", ""}:
                text = rest
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        try:
            obj = json.loads(text[start : end + 1]) if start != -1 and end != -1 else {}
        except json.JSONDecodeError:
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
    messages: list[dict] = [{"role": "user", "content": _format_email(email)}]
    total_in = 0
    total_out = 0
    total_cost = 0.0
    total_latency = 0.0
    tool_calls = 0
    schema_ok = True
    stop_reason = "end_turn"
    final_text = ""

    for _step in range(MAX_TOOL_CALLS + 1):
        tools = TOOL_SCHEMAS if tool_calls < MAX_TOOL_CALLS else None
        resp = client.complete(system=SYSTEM_PROMPT, messages=messages, tools=tools)
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        total_cost += resp.cost_usd
        total_latency += resp.latency_ms
        stop_reason = resp.stop_reason

        if resp.tool_uses:
            # Add the assistant turn and the tool_result turn
            assistant_blocks: list[dict] = []
            if resp.text:
                assistant_blocks.append({"type": "text", "text": resp.text})
            for tu in resp.tool_uses:
                assistant_blocks.append({"type": "tool_use", "id": tu["id"], "name": tu["name"], "input": tu["input"]})
            messages.append({"role": "assistant", "content": assistant_blocks})

            tool_result_blocks: list[dict] = []
            for tu in resp.tool_uses:
                result = call_tool(tu["name"], tu["input"] or {})
                tool_calls += 1
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": json.dumps(result),
                })
            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        # No tool use → this is a final answer
        final_text = resp.text
        break
    else:
        # ran out of steps without a final text answer; force one more turn with no tools
        resp = client.complete(system=SYSTEM_PROMPT, messages=messages, tools=None)
        total_in += resp.input_tokens
        total_out += resp.output_tokens
        total_cost += resp.cost_usd
        total_latency += resp.latency_ms
        final_text = resp.text
        stop_reason = resp.stop_reason

    parsed = _parse_or_repair(final_text, email["id"])
    if parsed["category"] not in CATEGORIES:
        schema_ok = False
    return {
        **parsed,
        "_meta": {
            "input_tokens": total_in,
            "output_tokens": total_out,
            "cost_usd": total_cost,
            "latency_ms": total_latency,
            "tool_calls": tool_calls,
            "schema_ok": schema_ok,
            "stop_reason": stop_reason,
        },
    }
