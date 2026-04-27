"""
Shared LLM client used by all three architectures and the judge.

Two backends:
- "anthropic"  — real Anthropic API. Requires ANTHROPIC_API_KEY.
- "mock"       — deterministic canned-response backend. Loads data/labels.jsonl
                 and data/reference_replies.jsonl and produces plausible outputs
                 with controlled noise per architecture, so the eval harness can
                 be validated end-to-end without an API key.

The mock injects different noise rates per architecture so the metrics are not
identical across the three. This is for harness validation only — real numbers
come from running with the anthropic backend.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Load .env at the repo root, if python-dotenv is installed and a .env exists.
# This lets users paste ANTHROPIC_API_KEY into .env instead of exporting it.
try:
    from dotenv import load_dotenv  # type: ignore

    _env_path = REPO_ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

# Per-million-token pricing (USD). Conservative defaults; override per-model if needed.
PRICING = {
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-6":   (15.0, 75.0),
    "claude-haiku-4-5":  (1.0, 5.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # Older versions kept in case someone passes them explicitly.
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-opus-4-5":   (15.0, 75.0),
    # Pricing for mock is zero so the eval can produce a $-table without
    # accidentally claiming real costs.
    "mock":              (0.0, 0.0),
}

CATEGORIES = ["reply_now", "reply_later", "archive", "unsubscribe"]


@dataclass
class LLMResponse:
    text: str = ""
    tool_uses: list = field(default_factory=list)
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


def _approx_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _stable_hash(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def _detect_role(system: str) -> str:
    """Identify which prompt is being executed from the system message."""
    s = system or ""
    if "evaluation judge" in s or "scoring an email reply draft" in s:
        return "judge"
    # Check critique BEFORE draft — the critique prompt references "DRAFT node"
    # in its body, which would otherwise make this misroute.
    if "CRITIQUE node" in s:
        return "arch3_critique"
    if "TRIAGE node" in s:
        return "arch3_triage"
    if "DRAFT node" in s:
        return "arch3_draft"
    if "tools" in s.lower() and "WORKING LOOP" in s:
        return "arch2"
    return "arch1"


def _email_id_from_messages(messages: list) -> str:
    """Find an EMAIL_ID:... or email_id field in the user message text."""
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            text = content
        else:
            # list of blocks
            text = ""
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    text += b.get("text", "")
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("EMAIL_ID:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


class _MockData:
    _instance = None

    @classmethod
    def get(cls) -> "_MockData":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.labels: dict = {}
        self.refs: dict = {}
        self.emails: dict = {}
        labels_path = REPO_ROOT / "data" / "labels.jsonl"
        refs_path = REPO_ROOT / "data" / "reference_replies.jsonl"
        emails_path = REPO_ROOT / "data" / "emails.jsonl"
        if labels_path.exists():
            for line in labels_path.read_text().splitlines():
                if line.strip():
                    o = json.loads(line)
                    self.labels[o["id"]] = o
        if refs_path.exists():
            for line in refs_path.read_text().splitlines():
                if line.strip():
                    o = json.loads(line)
                    self.refs[o["id"]] = o["draft"]
        if emails_path.exists():
            for line in emails_path.read_text().splitlines():
                if line.strip():
                    o = json.loads(line)
                    self.emails[o["id"]] = o


def _mock_arch1(email_id: str) -> dict:
    """Single-call baseline: ~10% misclassification on ambiguous emails, draft = reference truncated."""
    data = _MockData.get()
    label = data.labels.get(email_id, {})
    truth = label.get("category", "archive")
    is_amb = label.get("ambiguous", False)
    h = _stable_hash(email_id, "arch1") % 100
    # Misclassify on a deterministic subset of ambiguous emails
    if is_amb and h < 40:
        # flip to an adjacent plausible category
        flip_map = {
            "reply_now": "reply_later",
            "reply_later": "archive",
            "archive": "reply_later",
            "unsubscribe": "archive",
        }
        cat = flip_map[truth]
    else:
        cat = truth
    draft = None
    if cat == "reply_now":
        ref = data.refs.get(email_id, "[fill in]")
        # Arch 1 produces a slightly worse draft: truncate at first paragraph
        draft = ref.split("\n\n")[0] if ref else "[fill in]"
    return {
        "email_id": email_id,
        "category": cat,
        "confidence": 0.78 if is_amb else 0.92,
        "draft_reply": draft,
        "reasoning": f"Arch1 mock: matched on email content; truth={truth}.",
    }


def _mock_arch2_should_use_tool(email_id: str, prior_tool_calls: int) -> dict | None:
    """Arch 2 mock: on first turn, call search_past_emails for reply candidates."""
    data = _MockData.get()
    label = data.labels.get(email_id, {})
    truth = label.get("category", "archive")
    if prior_tool_calls >= 1:
        return None
    if truth in {"reply_now", "reply_later"}:
        # exercise the tool once to demonstrate the loop
        return {
            "id": f"tool_{email_id}_1",
            "name": "search_past_emails",
            "input": {"sender": data.emails.get(email_id, {}).get("from", "")},
        }
    return None


def _mock_arch2_final(email_id: str) -> dict:
    """Arch 2 mock final answer: ~6% noise (slightly better than arch1)."""
    data = _MockData.get()
    label = data.labels.get(email_id, {})
    truth = label.get("category", "archive")
    is_amb = label.get("ambiguous", False)
    h = _stable_hash(email_id, "arch2") % 100
    if is_amb and h < 25:
        flip_map = {
            "reply_now": "reply_later",
            "reply_later": "reply_now",
            "archive": "reply_later",
            "unsubscribe": "archive",
        }
        cat = flip_map[truth]
    else:
        cat = truth
    draft = None
    if cat == "reply_now":
        # Arch 2 uses the full reference reply (tone-matching from search_past_emails)
        draft = data.refs.get(email_id, "[fill in]")
    return {
        "email_id": email_id,
        "category": cat,
        "confidence": 0.86 if is_amb else 0.94,
        "draft_reply": draft,
        "reasoning": "Arch2 mock: called search_past_emails; matched tone from history.",
    }


def _mock_arch3_triage(email_id: str) -> dict:
    """Arch 3 triage node — ~5% noise."""
    data = _MockData.get()
    label = data.labels.get(email_id, {})
    truth = label.get("category", "archive")
    is_amb = label.get("ambiguous", False)
    h = _stable_hash(email_id, "arch3_triage") % 100
    if is_amb and h < 20:
        flip_map = {
            "reply_now": "reply_later",
            "reply_later": "archive",
            "archive": "reply_later",
            "unsubscribe": "archive",
        }
        cat = flip_map[truth]
    else:
        cat = truth
    return {
        "email_id": email_id,
        "category": cat,
        "confidence": 0.88 if is_amb else 0.95,
        "reasoning": "Arch3 mock triage.",
    }


def _mock_arch3_draft(email_id: str, revising: bool) -> dict:
    data = _MockData.get()
    ref = data.refs.get(email_id, "[fill in]")
    if revising:
        # revised draft = full reference (assume revise improves)
        return {"email_id": email_id, "draft_reply": ref}
    # initial draft: ~80% identical to reference, 20% slightly weaker
    h = _stable_hash(email_id, "arch3_draft") % 100
    if h < 20:
        # weaker first draft
        return {"email_id": email_id, "draft_reply": ref.split("\n")[0] + "\n\n[fill in]"}
    return {"email_id": email_id, "draft_reply": ref}


def _mock_arch3_critique(email_id: str, draft: str) -> dict:
    """Critique fires on ~20% of drafts; flags the weaker ones from _mock_arch3_draft."""
    h = _stable_hash(email_id, "arch3_draft") % 100
    if h < 20:
        return {
            "email_id": email_id,
            "decision": "revise",
            "issues": ["draft is too short and contains an unfilled placeholder"],
            "revision_guidance": "expand the draft to address all asks; avoid leaving [fill in] unless truly unknown.",
        }
    # Occasional false positive on non-ambiguous emails (the "self-correction tax")
    h2 = _stable_hash(email_id, "arch3_critique_fp") % 100
    if h2 < 5:
        return {
            "email_id": email_id,
            "decision": "revise",
            "issues": ["tone could be warmer"],
            "revision_guidance": "soften the opening line.",
        }
    return {
        "email_id": email_id,
        "decision": "pass",
        "issues": [],
        "revision_guidance": "",
    }


def _mock_judge(email_id: str, candidate: str) -> dict:
    """Judge mock: score against reference. Lower scores when candidate is shorter or has placeholders."""
    data = _MockData.get()
    ref = data.refs.get(email_id, "")
    if not candidate:
        scores = {"addresses_ask": 1, "tone": 2, "length_fit": 1, "no_hallucinations": 5, "usefulness_vs_reference": 1}
    else:
        # crude similarity: ratio of length and "fill in" usage
        len_ratio = min(len(candidate), len(ref)) / max(len(candidate), len(ref), 1)
        has_placeholder_unfilled = "[fill in]" in candidate
        addresses = 5 if len_ratio > 0.7 else (4 if len_ratio > 0.4 else 3)
        tone = 5 if len_ratio > 0.5 else 4
        length_fit = 5 if 0.6 <= len_ratio <= 1.6 else 3
        no_hallu = 5  # mock candidates don't hallucinate by construction
        useful = 5 if len_ratio > 0.85 else (4 if len_ratio > 0.5 else 3)
        if has_placeholder_unfilled and len_ratio < 0.6:
            addresses -= 1
            useful -= 1
        scores = {
            "addresses_ask": max(1, addresses),
            "tone": max(1, tone),
            "length_fit": max(1, length_fit),
            "no_hallucinations": no_hallu,
            "usefulness_vs_reference": max(1, useful),
        }
    overall = sum(scores.values()) / len(scores)
    return {
        "email_id": email_id,
        "scores": scores,
        "overall": overall,
        "comments": "Mock judge scored on length-ratio and placeholder heuristics.",
    }


class LLMClient:
    """Uniform interface used by all architectures and the judge.

    Usage:
        client = LLMClient(model="claude-sonnet-4-5")               # real
        client = LLMClient(model="claude-sonnet-4-5", mock=True)    # mock
        resp = client.complete(system="...", messages=[...], tools=None)
    """

    def __init__(self, model: str = "claude-sonnet-4-6", mock: bool = False, max_tokens: int = 2048, temperature: float = 0.0):
        self.model = model
        self.mock = mock
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._anthropic = None
        if not mock:
            try:
                from anthropic import Anthropic
                self._anthropic = Anthropic()
            except ImportError as e:
                raise ImportError("anthropic not installed; pip install anthropic, or pass mock=True") from e

    def _price(self, in_tok: int, out_tok: int) -> float:
        in_p, out_p = PRICING.get(self.model, (3.0, 15.0))
        if self.mock:
            in_p, out_p = PRICING["mock"]
        return (in_tok * in_p + out_tok * out_p) / 1_000_000

    def complete(self, system: str, messages: list, tools: list | None = None) -> LLMResponse:
        if self.mock:
            return self._mock(system, messages, tools)
        return self._real(system, messages, tools)

    # ---- real backend --------------------------------------------------
    def _real(self, system: str, messages: list, tools: list | None) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs = dict(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        if tools:
            kwargs["tools"] = tools
        resp = self._anthropic.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        text_parts: list[str] = []
        tool_uses: list[dict] = []
        for block in resp.content:
            t = getattr(block, "type", None)
            if t == "text":
                text_parts.append(block.text)
            elif t == "tool_use":
                tool_uses.append({"id": block.id, "name": block.name, "input": dict(block.input or {})})
        in_tok = getattr(resp.usage, "input_tokens", 0)
        out_tok = getattr(resp.usage, "output_tokens", 0)
        return LLMResponse(
            text="".join(text_parts),
            tool_uses=tool_uses,
            stop_reason=getattr(resp, "stop_reason", "end_turn"),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self._price(in_tok, out_tok),
            latency_ms=latency_ms,
        )

    # ---- mock backend --------------------------------------------------
    def _mock(self, system: str, messages: list, tools: list | None) -> LLMResponse:
        t0 = time.perf_counter()
        # tiny sleep to make latency numbers > 0 in summary
        time.sleep(0.001)
        role = _detect_role(system)
        email_id = _email_id_from_messages(messages)
        data = _MockData.get()

        if role == "arch1":
            answer = _mock_arch1(email_id)
            text = json.dumps(answer)
            return self._wrap_text(system, messages, text, t0)

        if role == "arch2":
            # Count prior tool_use blocks in the conversation
            prior_tool_calls = 0
            for m in messages:
                if m.get("role") == "assistant":
                    c = m.get("content")
                    if isinstance(c, list):
                        for b in c:
                            if isinstance(b, dict) and b.get("type") == "tool_use":
                                prior_tool_calls += 1
            tu = _mock_arch2_should_use_tool(email_id, prior_tool_calls)
            if tu is not None:
                # Return a tool_use response. The harness will execute the tool
                # and feed back tool_result; the mock then returns the final JSON.
                return self._wrap_tool_use(system, messages, tu, t0)
            answer = _mock_arch2_final(email_id)
            text = json.dumps(answer)
            return self._wrap_text(system, messages, text, t0)

        if role == "arch3_triage":
            answer = _mock_arch3_triage(email_id)
            return self._wrap_text(system, messages, json.dumps(answer), t0)

        if role == "arch3_draft":
            # Detect if this is a revision turn: user message will mention "REVISION"
            revising = any("REVISION" in (m.get("content") if isinstance(m.get("content"), str) else "") for m in messages if m.get("role") == "user")
            answer = _mock_arch3_draft(email_id, revising)
            return self._wrap_text(system, messages, json.dumps(answer), t0)

        if role == "arch3_critique":
            # Pull the draft out of the user message
            draft = ""
            for m in messages:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    draft = m["content"]
            answer = _mock_arch3_critique(email_id, draft)
            return self._wrap_text(system, messages, json.dumps(answer), t0)

        if role == "judge":
            # Pull the candidate draft from the user message
            candidate = ""
            for m in messages:
                if m.get("role") == "user" and isinstance(m.get("content"), str):
                    text = m["content"]
                    # candidate is between "CANDIDATE DRAFT:\n" and end
                    if "CANDIDATE DRAFT:" in text:
                        candidate = text.split("CANDIDATE DRAFT:", 1)[1].strip()
            answer = _mock_judge(email_id, candidate)
            return self._wrap_text(system, messages, json.dumps(answer), t0)

        # fallback
        return self._wrap_text(system, messages, json.dumps({"error": "unknown role"}), t0)

    def _wrap_text(self, system: str, messages: list, text: str, t0: float) -> LLMResponse:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        sys_str = system or ""
        msg_str = json.dumps(messages)
        in_tok = _approx_tokens(sys_str + msg_str)
        out_tok = _approx_tokens(text)
        return LLMResponse(
            text=text,
            tool_uses=[],
            stop_reason="end_turn",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self._price(in_tok, out_tok),
            latency_ms=latency_ms,
        )

    def _wrap_tool_use(self, system: str, messages: list, tu: dict, t0: float) -> LLMResponse:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        sys_str = system or ""
        msg_str = json.dumps(messages)
        in_tok = _approx_tokens(sys_str + msg_str)
        out_tok = _approx_tokens(json.dumps(tu))
        return LLMResponse(
            text="",
            tool_uses=[tu],
            stop_reason="tool_use",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=self._price(in_tok, out_tok),
            latency_ms=latency_ms,
        )
