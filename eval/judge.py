"""
LLM-as-judge for draft quality.

The judge is run on a different model from the agents under test, per
PLAN.md. It scores each candidate draft on 5 dimensions against the human
reference reply.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_client import LLMClient

REPO_ROOT = Path(__file__).resolve().parents[1]
JUDGE_PROMPT = (REPO_ROOT / "prompts" / "judge_rubric.txt").read_text()


def _format_judge_input(email: dict, reference: str, candidate: str) -> str:
    return (
        f"EMAIL_ID: {email['id']}\n\n"
        f"INCOMING EMAIL:\n"
        f"From: {email['from']}\n"
        f"Subject: {email['subject']}\n\n{email['body']}\n\n"
        f"REFERENCE DRAFT:\n{reference}\n\n"
        f"CANDIDATE DRAFT:\n{candidate}"
    )


def _parse_or_default(text: str, email_id: str) -> dict:
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
        if start != -1 and end != -1:
            try:
                obj = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                obj = {}
        else:
            obj = {}
    scores = obj.get("scores") or {}
    out = {
        "email_id": email_id,
        "scores": {
            "addresses_ask": int(scores.get("addresses_ask", 3)),
            "tone": int(scores.get("tone", 3)),
            "length_fit": int(scores.get("length_fit", 3)),
            "no_hallucinations": int(scores.get("no_hallucinations", 3)),
            "usefulness_vs_reference": int(scores.get("usefulness_vs_reference", 3)),
        },
        "comments": obj.get("comments", ""),
    }
    out["overall"] = sum(out["scores"].values()) / len(out["scores"])
    return out


def judge_one(email: dict, reference: str, candidate: str, client: LLMClient) -> dict:
    user_msg = _format_judge_input(email, reference, candidate)
    resp = client.complete(system=JUDGE_PROMPT, messages=[{"role": "user", "content": user_msg}])
    parsed = _parse_or_default(resp.text, email["id"])
    parsed["_meta"] = {
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "cost_usd": resp.cost_usd,
        "latency_ms": resp.latency_ms,
    }
    return parsed
