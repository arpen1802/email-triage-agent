"""
Apply Gmail labels and create draft replies based on triage results.

Labels we manage (the tool creates them on first use if they don't exist):
- AI/ReplyNow      — needs your attention soon
- AI/ReplyLater    — non-urgent, respond later
- AI/Archive       — informational, no response needed
- AI/Unsubscribe   — promotional, candidate for unsubscribe
- AI/Triaged       — applied to every message we've processed, so we can skip on re-run

Drafts are created in the user's Drafts folder, threaded to the original message.
We never send mail. The user reviews and presses send (or doesn't).
"""

from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Any

LABEL_PREFIX = "AI/"
CATEGORY_LABELS = {
    "reply_now": "AI/ReplyNow",
    "reply_later": "AI/ReplyLater",
    "archive": "AI/Archive",
    "unsubscribe": "AI/Unsubscribe",
}
TRIAGED_LABEL = "AI/Triaged"


def ensure_labels(service) -> dict[str, str]:
    """Create any missing AI/* labels and return {label_name: label_id}."""
    existing = service.users().labels().list(userId="me").execute().get("labels", [])
    name_to_id = {l["name"]: l["id"] for l in existing}
    needed = list(CATEGORY_LABELS.values()) + [TRIAGED_LABEL]
    for name in needed:
        if name in name_to_id:
            continue
        body = {
            "name": name,
            "labelListVisibility": "labelShow",
            "messageListVisibility": "show",
        }
        created = service.users().labels().create(userId="me", body=body).execute()
        name_to_id[name] = created["id"]
    return name_to_id


def apply_category(service, message_id: str, category: str, label_ids: dict[str, str]) -> None:
    """Add the AI/<Category> label and AI/Triaged to a message."""
    label_name = CATEGORY_LABELS.get(category)
    add: list[str] = [label_ids[TRIAGED_LABEL]]
    if label_name:
        add.append(label_ids[label_name])
    service.users().messages().modify(
        userId="me", id=message_id, body={"addLabelIds": add, "removeLabelIds": []}
    ).execute()


def _b64url_encode_bytes(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("ascii")


def create_draft_reply(service, original: dict[str, Any], draft_text: str) -> dict:
    """Create a draft reply to the given message and return the draft resource.

    The draft is threaded to the original (so it appears in the same conversation),
    addressed to the original sender, and has Re: prefixed if not already.
    """
    sender = original.get("from", "")
    subject = original.get("subject", "") or ""
    if not subject.lower().startswith("re:"):
        subject = "Re: " + subject

    msg = EmailMessage()
    msg["To"] = sender
    msg["Subject"] = subject
    if original.get("_message_id_header"):
        msg["In-Reply-To"] = original["_message_id_header"]
        msg["References"] = original["_message_id_header"]
    msg.set_content(draft_text)

    raw = _b64url_encode_bytes(msg.as_bytes())
    draft_body = {
        "message": {
            "raw": raw,
            "threadId": original.get("_thread_id", ""),
        }
    }
    return service.users().drafts().create(userId="me", body=draft_body).execute()
