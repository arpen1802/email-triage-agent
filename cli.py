"""
CLI for the email triage tool.

Usage:
    python -m cli gmail setup
        Run the OAuth consent flow and cache a token.

    python -m cli gmail triage --last 50
        DRY RUN. Read your last 50 inbox messages, classify each, print a summary.
        Doesn't change anything in your inbox.

    python -m cli gmail triage --last 50 --apply-labels
        Same, but also adds AI/<Category> labels in Gmail so you can filter on them.

    python -m cli gmail triage --last 50 --apply-labels --create-drafts
        Same, plus creates draft replies for `reply_now` messages in your Drafts
        folder, threaded to the original. Never sends mail — you review and send.

    Flags:
        --last N             Number of messages from inbox (default 50).
        --skip-triaged       Skip messages already labelled AI/Triaged (idempotent runs).
        --apply-labels       Actually apply Gmail labels.
        --create-drafts      Actually create drafts in Drafts folder.
        --model MODEL        LLM model to use (default claude-sonnet-4-6).
        --confirm            Pause before each apply step. Useful on first real run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import shorten

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# Imported here so `python -m cli gmail setup` doesn't require an LLM key
# unless the user is actually classifying.
from llm_client import LLMClient  # noqa: E402
from architectures import arch1_single_call  # noqa: E402
from gmail_triage import auth as gmail_auth  # noqa: E402
from gmail_triage import inbox as gmail_inbox  # noqa: E402
from gmail_triage import actions as gmail_actions  # noqa: E402

CATEGORY_EMOJI = {  # only used in stdout output, not in any committed file
    "reply_now": "[REPLY NOW]",
    "reply_later": "[reply later]",
    "archive": "[archive]   ",
    "unsubscribe": "[unsubscribe]",
}


def cmd_gmail_setup(args: argparse.Namespace) -> int:
    """Run the OAuth flow once and exit."""
    try:
        gmail_auth.get_credentials(interactive=True)
    except gmail_auth.CredentialsMissingError as e:
        print(str(e), file=sys.stderr)
        return 2
    print("OK. Gmail access authorised. Token cached at token.json.")
    print("You can now run: python -m cli gmail triage --last 50")
    return 0


def _short(s: str, n: int = 60) -> str:
    return shorten(s.replace("\n", " ").strip(), width=n, placeholder="…") if s else ""


def cmd_gmail_triage(args: argparse.Namespace) -> int:
    try:
        service = gmail_auth.gmail_service()
    except gmail_auth.CredentialsMissingError as e:
        print(str(e), file=sys.stderr)
        return 2

    query = "in:inbox"
    if args.skip_triaged:
        query += " -label:AI/Triaged"

    print(f"Fetching up to {args.last} messages with query: '{query}' ...")
    messages = gmail_inbox.fetch_recent(service, last_n=args.last, query=query)
    if not messages:
        print("No messages found.")
        return 0
    print(f"Got {len(messages)} message(s). Classifying with model {args.model} ...")

    client = LLMClient(model=args.model)

    label_ids: dict[str, str] | None = None
    if args.apply_labels or args.create_drafts:
        label_ids = gmail_actions.ensure_labels(service)

    bucket_counts: dict[str, int] = {"reply_now": 0, "reply_later": 0, "archive": 0, "unsubscribe": 0}
    drafts_created = 0
    labels_applied = 0
    total_cost = 0.0

    for i, email in enumerate(messages, 1):
        try:
            pred = arch1_single_call.run(email, client)
        except Exception as e:
            print(f"  [{i}/{len(messages)}] ERROR on {email.get('id')}: {e}", file=sys.stderr)
            continue

        cat = pred["category"]
        bucket_counts[cat] = bucket_counts.get(cat, 0) + 1
        total_cost += pred["_meta"]["cost_usd"]

        marker = CATEGORY_EMOJI.get(cat, "[?]")
        from_short = _short(email.get("from", ""), 30)
        subj_short = _short(email.get("subject", ""), 60)
        print(f"  {i:>2}. {marker}  {from_short:<30}  {subj_short}")

        if args.confirm and (args.apply_labels or args.create_drafts):
            ans = input("       apply changes for this message? [y/N/q] ").strip().lower()
            if ans == "q":
                print("Stopped by user.")
                break
            if ans != "y":
                continue

        if args.apply_labels and label_ids is not None:
            try:
                gmail_actions.apply_category(service, email["id"], cat, label_ids)
                labels_applied += 1
            except Exception as e:
                print(f"       label error: {e}", file=sys.stderr)

        if args.create_drafts and cat == "reply_now" and pred.get("draft_reply"):
            try:
                gmail_actions.create_draft_reply(service, email, pred["draft_reply"])
                drafts_created += 1
            except Exception as e:
                print(f"       draft error: {e}", file=sys.stderr)

    print()
    print("Summary:")
    for cat in ["reply_now", "reply_later", "archive", "unsubscribe"]:
        print(f"  {cat:<13} {bucket_counts.get(cat, 0)}")
    print(f"  total cost   ${total_cost:.4f}")
    if args.apply_labels:
        print(f"  labels applied: {labels_applied}")
    if args.create_drafts:
        print(f"  drafts created: {drafts_created}")
    if not (args.apply_labels or args.create_drafts):
        print()
        print("DRY RUN — no changes made to your inbox.")
        print("Re-run with --apply-labels and/or --create-drafts to act on these.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cli", description="Email triage tool.")
    sub = parser.add_subparsers(dest="surface", required=True)

    gmail_p = sub.add_parser("gmail", help="Gmail integration.")
    gmail_sub = gmail_p.add_subparsers(dest="action", required=True)

    setup_p = gmail_sub.add_parser("setup", help="Run OAuth consent flow once.")
    setup_p.set_defaults(func=cmd_gmail_setup)

    triage_p = gmail_sub.add_parser("triage", help="Classify recent inbox messages.")
    triage_p.add_argument("--last", type=int, default=50, help="Number of messages to fetch (default 50).")
    triage_p.add_argument("--skip-triaged", action="store_true", help="Skip messages already labelled AI/Triaged.")
    triage_p.add_argument("--apply-labels", action="store_true", help="Apply AI/<Category> labels in Gmail.")
    triage_p.add_argument("--create-drafts", action="store_true", help="Create drafts for reply_now messages.")
    triage_p.add_argument("--model", default="claude-sonnet-4-6", help="LLM model to use.")
    triage_p.add_argument("--confirm", action="store_true", help="Pause before applying changes per message.")
    triage_p.set_defaults(func=cmd_gmail_triage)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
