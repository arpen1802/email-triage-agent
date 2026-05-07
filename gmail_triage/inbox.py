"""
Read messages from a user's Gmail inbox and reshape them into the same
email dict format the architectures consume.

Output dict shape (matches data/emails.jsonl):
    {
      "id": "<gmail message id>",
      "from": "Sender Name <sender@example.com>",
      "to": "you@example.com",
      "subject": "...",
      "date": "<RFC-3339 ISO datetime>",
      "body": "<plain-text body, HTML stripped>",
      # Gmail-specific extras (used by actions.py, ignored by architectures):
      "_thread_id": "...",
      "_label_ids": [...],
      "_message_id_header": "<rfc-822 Message-ID>",
    }

The body extraction prefers text/plain, falls back to a stripped text/html.
"""

from __future__ import annotations

import base64
import email.utils
import re
from html.parser import HTMLParser
from typing import Any


class _HTMLStripper(HTMLParser):
    """Minimal HTML-to-text — collapses tags, preserves block boundaries."""

    BLOCK_TAGS = {"p", "br", "div", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def text(self) -> str:
        s = "".join(self._parts)
        # collapse runs of whitespace, preserve paragraph breaks
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()


def _strip_html(s: str) -> str:
    parser = _HTMLStripper()
    parser.feed(s)
    return parser.text()


def _decode_b64url(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _walk_parts(payload: dict) -> list[dict]:
    """Yield every leaf MIME part in a Gmail message payload."""
    out: list[dict] = []
    if "parts" in payload:
        for p in payload["parts"]:
            out.extend(_walk_parts(p))
    else:
        out.append(payload)
    return out


def _extract_body(payload: dict) -> str:
    """Pick the best body text from a Gmail message payload."""
    plain_parts: list[str] = []
    html_parts: list[str] = []
    for part in _walk_parts(payload):
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        data = body.get("data")
        if not data:
            continue
        try:
            raw = _decode_b64url(data).decode("utf-8", errors="replace")
        except Exception:
            continue
        if mime == "text/plain":
            plain_parts.append(raw)
        elif mime == "text/html":
            html_parts.append(raw)
    if plain_parts:
        return "\n".join(plain_parts).strip()
    if html_parts:
        return _strip_html("\n".join(html_parts))
    return ""


def _header(headers: list[dict], name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "") or ""
    return ""


def _parse_message(msg: dict) -> dict[str, Any]:
    payload = msg.get("payload", {}) or {}
    headers = payload.get("headers", []) or []
    from_h = _header(headers, "From")
    to_h = _header(headers, "To")
    subject = _header(headers, "Subject")
    msg_id_header = _header(headers, "Message-ID")

    # Date: prefer the RFC-822 Date header; fall back to internalDate (ms epoch).
    date_iso = ""
    date_h = _header(headers, "Date")
    if date_h:
        parsed = email.utils.parsedate_to_datetime(date_h)
        if parsed is not None:
            date_iso = parsed.isoformat()
    if not date_iso and msg.get("internalDate"):
        from datetime import datetime, timezone

        ts = int(msg["internalDate"]) / 1000
        date_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

    body = _extract_body(payload)
    if not body and msg.get("snippet"):
        body = msg["snippet"]

    return {
        "id": msg["id"],
        "from": from_h,
        "to": to_h,
        "subject": subject,
        "date": date_iso,
        "body": body,
        "_thread_id": msg.get("threadId", ""),
        "_label_ids": msg.get("labelIds", []) or [],
        "_message_id_header": msg_id_header,
    }


def fetch_recent(service, last_n: int = 50, query: str = "in:inbox") -> list[dict]:
    """Return up to last_n messages from the user's inbox, newest first.

    Args:
        service: authorised Gmail API client (from gmail_triage.auth.gmail_service()).
        last_n:  how many messages to return.
        query:   Gmail search query. Default = inbox only. Pass e.g.
                 "in:inbox -label:AI/Triaged" to skip already-processed mail.
    """
    out: list[dict] = []
    page_token: str | None = None
    while len(out) < last_n:
        kwargs: dict = {
            "userId": "me",
            "q": query,
            "maxResults": min(100, last_n - len(out)),
        }
        if page_token:
            kwargs["pageToken"] = page_token
        resp = service.users().messages().list(**kwargs).execute()
        ids = [m["id"] for m in resp.get("messages", []) or []]
        if not ids:
            break
        for mid in ids:
            full = (
                service.users()
                .messages()
                .get(userId="me", id=mid, format="full")
                .execute()
            )
            out.append(_parse_message(full))
            if len(out) >= last_n:
                break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return out
