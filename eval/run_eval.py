"""
Run the eval harness: 3 architectures x N trials over the 50-email dataset.

Outputs one JSON per (arch, trial) into results/raw_outputs/. Then writes
results/summary.md via eval.metrics.

Usage:
    python -m eval.run_eval                      # real run, 3 trials
    python -m eval.run_eval --trials 1           # one trial
    python -m eval.run_eval --mock --trials 1    # mock backend, no API needed
    python -m eval.run_eval --arch arch1         # only one architecture
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from llm_client import LLMClient  # noqa: E402
from architectures import arch1_single_call, arch2_react_loop, arch3_graph  # noqa: E402
from eval.judge import judge_one  # noqa: E402

RAW_DIR = REPO_ROOT / "results" / "raw_outputs"
DATA_DIR = REPO_ROOT / "data"

ARCHES = {
    "arch1": ("Arch 1 — single call", arch1_single_call.run),
    "arch2": ("Arch 2 — ReAct loop", arch2_react_loop.run),
    "arch3": ("Arch 3 — graph + critique", arch3_graph.run),
}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def run_one_trial(arch_key: str, run_fn, emails: list[dict], client: LLMClient, judge_client: LLMClient | None, references: dict, labels: dict) -> dict:
    predictions: list[dict] = []
    judge_records: list[dict] = []
    t0 = time.perf_counter()
    for i, email in enumerate(emails, 1):
        try:
            pred = run_fn(email, client)
        except Exception as e:
            # Re-raise immediately on authentication/configuration errors so the
            # user sees a loud failure instead of a "successful" run full of
            # fallback predictions.
            msg = str(e)
            if "authentication_error" in msg or "401" in msg or "invalid x-api-key" in msg or "permission_error" in msg or "404" in msg and "model" in msg.lower():
                raise SystemExit(
                    f"\n\nFATAL: API call failed with what looks like an auth or config error:\n  {msg}\n\n"
                    f"Check your ANTHROPIC_API_KEY (in .env or your shell) and that the model "
                    f"({client.model}) is accessible to your account.\n"
                ) from e
            pred = {
                "email_id": email["id"],
                "category": "archive",
                "confidence": 0.0,
                "draft_reply": None,
                "reasoning": f"ERROR: {e}",
                "_meta": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0, "tool_calls": 0, "schema_ok": False, "stop_reason": "error"},
            }
        predictions.append(pred)

        # Judge only when truth label is reply_now AND the architecture actually drafted
        truth_cat = labels[email["id"]]["category"]
        if judge_client is not None and truth_cat == "reply_now" and pred.get("draft_reply"):
            ref = references.get(email["id"], "")
            j = judge_one(email, ref, pred["draft_reply"], judge_client)
            judge_records.append(j)

        if i % 10 == 0:
            print(f"    {arch_key}: {i}/{len(emails)} emails processed", flush=True)
    elapsed = time.perf_counter() - t0
    return {
        "arch": arch_key,
        "n": len(emails),
        "wallclock_s": elapsed,
        "predictions": predictions,
        "judge": judge_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--judge-model", default="claude-opus-4-6")
    parser.add_argument("--mock", action="store_true", help="Use the deterministic mock backend (no API key needed).")
    parser.add_argument("--arch", choices=list(ARCHES.keys()) + ["all"], default="all")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-as-judge pass.")
    parser.add_argument("--limit", type=int, default=None, help="Limit to first N emails (for debugging).")
    args = parser.parse_args()

    emails = load_jsonl(DATA_DIR / "emails.jsonl")
    labels_list = load_jsonl(DATA_DIR / "labels.jsonl")
    labels = {l["id"]: l for l in labels_list}
    refs_list = load_jsonl(DATA_DIR / "reference_replies.jsonl")
    references = {r["id"]: r["draft"] for r in refs_list}
    if args.limit:
        emails = emails[: args.limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    client = LLMClient(model=args.model, mock=args.mock)
    judge_client = None if args.no_judge else LLMClient(model=args.judge_model, mock=args.mock)

    arch_keys = list(ARCHES.keys()) if args.arch == "all" else [args.arch]

    print(f"Running {len(arch_keys)} architecture(s) x {args.trials} trial(s) on {len(emails)} emails (mock={args.mock}).")
    for arch_key in arch_keys:
        label, run_fn = ARCHES[arch_key]
        for trial in range(1, args.trials + 1):
            print(f"\n[{label}] trial {trial}/{args.trials}")
            payload = run_one_trial(arch_key, run_fn, emails, client, judge_client, references, labels)
            payload["model"] = args.model
            payload["judge_model"] = args.judge_model
            payload["trial"] = trial
            payload["mock"] = args.mock
            out_path = RAW_DIR / f"{arch_key}_trial{trial}.json"
            out_path.write_text(json.dumps(payload, indent=2, default=str))
            n_correct = sum(1 for p in payload["predictions"] if p["category"] == labels[p["email_id"]]["category"])
            tot_cost = sum(p["_meta"]["cost_usd"] for p in payload["predictions"])
            print(f"    -> wrote {out_path.relative_to(REPO_ROOT)} | accuracy {n_correct}/{len(emails)} | cost ${tot_cost:.4f} | wall {payload['wallclock_s']:.1f}s")

    # Build summary
    print("\nWriting results/summary.md ...")
    from eval import metrics as metrics_module
    metrics_module.main([])


if __name__ == "__main__":
    main()
