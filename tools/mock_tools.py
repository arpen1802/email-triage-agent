"""
Mock tools for Architecture 2 (ReAct loop).

These return canned data — explicitly disclosed in the writeup. The point is to
measure whether *access to tools* changes outcomes, not to build a real backend.

Determinism: outputs are seeded by sender / email_id so repeated runs give the
same answers. This isolates architecture variance from tool-output variance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


# Anthropic-format tool schemas. Reused verbatim by Arch 2 when building the
# request to the model.
TOOL_SCHEMAS = [
    {
        "name": "search_past_emails",
        "description": "Look up whether the user has corresponded with this sender before, and what tone they typically use.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sender": {"type": "string", "description": "Email address or name of the sender."}
            },
            "required": ["sender"],
        },
    },
    {
        "name": "check_calendar",
        "description": "Check the user's calendar for free/busy in a date range. Use only when the email proposes meeting times.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_iso": {"type": "string", "description": "ISO-8601 start datetime."},
                "end_iso": {"type": "string", "description": "ISO-8601 end datetime."},
            },
            "required": ["start_iso", "end_iso"],
        },
    },
    {
        "name": "get_user_context",
        "description": "Get a brief profile of the user: role, current projects, communication style, working hours.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "lookup_thread",
        "description": "Fetch prior messages in the thread for a given email_id. Use only when context from the thread is needed.",
        "input_schema": {
            "type": "object",
            "properties": {"email_id": {"type": "string"}},
            "required": ["email_id"],
        },
    },
]


def _hash_pick(seed: str, options: list):
    h = hashlib.sha256(seed.encode()).hexdigest()
    return options[int(h[:4], 16) % len(options)]


def search_past_emails(sender: str) -> dict:
    sender = sender or ""
    # Friends/family-looking domains get a casual tone.
    casual_domains = ("gmail.com", "yahoo.com", "icloud.com", "hotmail.com", "@dad", "@mom", "Sam Patel", "Theo Brandt")
    is_casual = any(c in sender for c in casual_domains)
    has_replied = _hash_pick(sender + "::has_replied", [True, True, True, False])
    if not has_replied:
        return {
            "has_replied_before": False,
            "typical_tone": "n/a",
            "last_interaction": "n/a",
            "count": 0,
        }
    if is_casual:
        tone = _hash_pick(sender + "::tone", ["casual lowercase, brief", "warm and chatty", "very short, emoji-friendly"])
    elif "linkedin" in sender.lower() or "noreply" in sender.lower() or "no-reply" in sender.lower():
        tone = "n/a (automated)"
    else:
        tone = _hash_pick(sender + "::tone", ["professional, concise", "friendly professional", "formal"])
    last = _hash_pick(sender + "::last", ["3 days ago", "2 weeks ago", "last month", "yesterday"])
    count = (int(hashlib.sha256(sender.encode()).hexdigest()[:2], 16) % 25) + 1
    return {
        "has_replied_before": True,
        "typical_tone": tone,
        "last_interaction": last,
        "count": count,
    }


def check_calendar(start_iso: str, end_iso: str) -> dict:
    seed = f"{start_iso}|{end_iso}"
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:4], 16)
    free_pool = [
        "2026-04-24T14:00:00-07:00 (1h)",
        "2026-04-24T16:00:00-07:00 (30m)",
        "2026-04-25T10:00:00-07:00 (1h)",
        "2026-04-25T15:30:00-07:00 (1h)",
        "2026-05-07T09:00:00-07:00 (1h)",
        "2026-05-08T14:00:00-07:00 (1h)",
    ]
    conflict_pool = [
        "2026-04-24T13:00:00-07:00 — 'Standup'",
        "2026-04-25T11:00:00-07:00 — 'Q3 planning'",
    ]
    free = [free_pool[(h + i) % len(free_pool)] for i in range(2)]
    conflicts = [conflict_pool[h % len(conflict_pool)]] if h % 2 == 0 else []
    return {"free_slots": free, "conflicts": conflicts}


def get_user_context() -> dict:
    return {
        "role": "Engineering Lead, platform team",
        "current_projects": ["Q3 launch", "storage migration", "billing refactor v2"],
        "communication_style": "concise; lowercase fine for casual; uses [fill in] for unknowns; avoids fluff",
        "working_hours": "08:30–18:00 Pacific, Mon–Fri",
    }


def lookup_thread(email_id: str) -> dict:
    # Canned thread data for a few emails that look like Re: threads.
    canned = {
        "e019": {
            "thread_messages": [
                {"from": "Hana", "date": "2026-04-19", "snippet": "Initial proposal: option A (atomic cutover) vs option B (gradual)..."},
                {"from": "Jin", "date": "2026-04-20", "snippet": "Leaning B. Risk profile better for our SLA."},
                {"from": "Karim", "date": "2026-04-21", "snippet": "+1 to B; instrumentation on dual-writes already done."},
            ]
        },
        "e035": {
            "thread_messages": [
                {"from": "Priya", "date": "2026-04-22", "snippet": "Sending over the launch plan v3 — eng please confirm dates."},
                {"from": "me", "date": "2026-04-22", "snippet": "Got it, will review tomorrow."},
            ]
        },
        "e050": {
            "thread_messages": [
                {"from": "Sam", "date": "2026-04-22", "snippet": "are you free saturday?"},
                {"from": "me", "date": "2026-04-22", "snippet": "yeah i'm in — what time does it start?"},
                {"from": "Sam", "date": "2026-04-23", "snippet": "lol"},
            ]
        },
    }
    return canned.get(email_id, {"thread_messages": []})


# Dispatcher used by the harness.
TOOL_FUNCS = {
    "search_past_emails": search_past_emails,
    "check_calendar": check_calendar,
    "get_user_context": get_user_context,
    "lookup_thread": lookup_thread,
}


def call_tool(name: str, kwargs: dict) -> dict:
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**(kwargs or {}))
    except TypeError as e:
        return {"error": f"bad arguments to {name}: {e}"}
