"""
Compute summary metrics from raw_outputs/*.json and write results/summary.md.

Usage:
    python -m eval.metrics                  # writes results/summary.md
    python -m eval.metrics --print-only     # prints to stdout

Metrics:
- Classification accuracy (overall + per-category F1)
- Cost per email (mean USD, summed across all calls in a run)
- Latency (p50, p95) per email
- Reliability (% schema-valid)
- Draft quality (mean of judge overall scores, on reply_now-labeled items)
- Variance across trials
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = REPO_ROOT / "results" / "raw_outputs"
SUMMARY = REPO_ROOT / "results" / "summary.md"
LABELS_PATH = REPO_ROOT / "data" / "labels.jsonl"

CATS = ["reply_now", "reply_later", "archive", "unsubscribe"]


def load_labels() -> dict:
    out = {}
    for line in LABELS_PATH.read_text().splitlines():
        if line.strip():
            o = json.loads(line)
            out[o["id"]] = o
    return out


def per_category_f1(preds: list[dict], labels: dict) -> dict:
    """Compute precision/recall/F1 per category."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for p in preds:
        truth = labels[p["email_id"]]["category"]
        pred = p["category"]
        if pred == truth:
            tp[truth] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1
    out = {}
    for c in CATS:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[c] = {"precision": precision, "recall": recall, "f1": f1, "support": tp[c] + fn[c]}
    return out


def accuracy(preds: list[dict], labels: dict) -> float:
    correct = sum(1 for p in preds if p["category"] == labels[p["email_id"]]["category"])
    return correct / len(preds) if preds else 0.0


def over_reply_rate(preds: list[dict], labels: dict) -> float:
    """Fraction of non-reply_now emails that got drafted anyway."""
    n_non = 0
    n_drafted = 0
    for p in preds:
        if labels[p["email_id"]]["category"] != "reply_now":
            n_non += 1
            if p.get("draft_reply"):
                n_drafted += 1
    return n_drafted / n_non if n_non else 0.0


def reliability(preds: list[dict]) -> float:
    return sum(1 for p in preds if p.get("_meta", {}).get("schema_ok", False)) / len(preds) if preds else 0.0


def cost_per_email(preds: list[dict]) -> float:
    return statistics.fmean(p.get("_meta", {}).get("cost_usd", 0.0) for p in preds) if preds else 0.0


def total_cost(preds: list[dict]) -> float:
    return sum(p.get("_meta", {}).get("cost_usd", 0.0) for p in preds)


def latency_pcts(preds: list[dict]) -> tuple[float, float]:
    lats = sorted(p.get("_meta", {}).get("latency_ms", 0.0) for p in preds)
    if not lats:
        return 0.0, 0.0
    p50 = lats[len(lats) // 2]
    p95 = lats[max(0, int(len(lats) * 0.95) - 1)]
    return p50, p95


def mean_judge_overall(judge_records: list[dict]) -> float:
    if not judge_records:
        return 0.0
    return statistics.fmean(j.get("overall", 0.0) for j in judge_records)


def summarise_run(arch_label: str, run_path: Path, labels: dict) -> dict:
    payload = json.loads(run_path.read_text())
    preds = payload["predictions"]
    judges = payload.get("judge", [])
    p50, p95 = latency_pcts(preds)
    return {
        "arch": arch_label,
        "trial": payload.get("trial"),
        "model": payload.get("model"),
        "judge_model": payload.get("judge_model"),
        "mock": payload.get("mock", False),
        "n": len(preds),
        "accuracy": accuracy(preds, labels),
        "f1": per_category_f1(preds, labels),
        "over_reply_rate": over_reply_rate(preds, labels),
        "reliability": reliability(preds),
        "cost_per_email": cost_per_email(preds),
        "total_cost": total_cost(preds),
        "latency_p50_ms": p50,
        "latency_p95_ms": p95,
        "judge_overall_mean": mean_judge_overall(judges),
    }


def aggregate_across_trials(runs: list[dict]) -> dict:
    """Mean and stdev across trials for the headline numbers."""
    keys = ["accuracy", "over_reply_rate", "reliability", "cost_per_email", "total_cost", "latency_p50_ms", "latency_p95_ms", "judge_overall_mean"]
    out = {"trials": len(runs)}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k + "_mean"] = statistics.fmean(vals) if vals else 0.0
        out[k + "_stdev"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return out


def render_markdown(per_arch_runs: dict, labels: dict) -> str:
    lines: list[str] = []
    lines.append("# Email Triage Agents — Results Summary")
    lines.append("")
    lines.append(f"Dataset: {len(labels)} emails. Categories: " + ", ".join(f"{c} ({sum(1 for v in labels.values() if v['category']==c)})" for c in CATS) + ".")
    lines.append("")
    sample_run = next((r for arch_runs in per_arch_runs.values() for r in arch_runs), None)
    if sample_run:
        lines.append(f"Model under test: `{sample_run['model']}`. Judge model: `{sample_run['judge_model']}`.")
        if sample_run.get("mock"):
            lines.append("")
            lines.append("> **MOCK RUN.** These numbers come from the deterministic mock backend used to validate the harness. "
                         "Cost and latency are not meaningful. The accuracy/F1/judge numbers reflect the controlled noise built "
                         "into `llm_client._mock_*`, not real model behavior. Re-run with `python -m eval.run_eval --trials 3` "
                         "(no `--mock`) and a valid `ANTHROPIC_API_KEY` for real measurements.")
        lines.append("")

    # Headline table
    lines.append("## Headline metrics (mean ± stdev across trials)")
    lines.append("")
    lines.append("| Architecture | Accuracy | Over-reply rate | Reliability | Cost / email | Total cost | Latency p50 | Latency p95 | Judge score |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for arch, runs in per_arch_runs.items():
        a = aggregate_across_trials(runs)
        lines.append(
            f"| {arch} "
            f"| {a['accuracy_mean']:.3f} ± {a['accuracy_stdev']:.3f} "
            f"| {a['over_reply_rate_mean']:.3f} "
            f"| {a['reliability_mean']:.3f} "
            f"| ${a['cost_per_email_mean']:.5f} "
            f"| ${a['total_cost_mean']:.4f} "
            f"| {a['latency_p50_ms_mean']:.0f} ms "
            f"| {a['latency_p95_ms_mean']:.0f} ms "
            f"| {a['judge_overall_mean_mean']:.2f}/5 |"
        )

    lines.append("")
    lines.append("## Per-category F1 (last trial)")
    lines.append("")
    lines.append("| Architecture | reply_now | reply_later | archive | unsubscribe |")
    lines.append("|---|---|---|---|---|")
    for arch, runs in per_arch_runs.items():
        f1 = runs[-1]["f1"]
        lines.append(f"| {arch} | {f1['reply_now']['f1']:.2f} | {f1['reply_later']['f1']:.2f} | {f1['archive']['f1']:.2f} | {f1['unsubscribe']['f1']:.2f} |")

    lines.append("")
    lines.append("## Notes on reading these numbers")
    lines.append("")
    lines.append("- **Accuracy** is exact-match against `data/labels.jsonl`. ~9 emails are deliberately ambiguous; perfect accuracy is not the target.")
    lines.append("- **Over-reply rate** is the fraction of non-`reply_now` emails for which the architecture nonetheless drafted a reply. Lower is better.")
    lines.append("- **Reliability** is the share of predictions that parsed cleanly into the required JSON schema.")
    lines.append("- **Judge score** is the mean of the LLM-as-judge overall (1-5) across all `reply_now` emails.")
    lines.append("- All figures aggregate across the trials specified at `--trials`.")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args(argv)

    labels = load_labels()
    per_arch_runs: dict[str, list[dict]] = defaultdict(list)
    for run_file in sorted(RAW.glob("*.json")):
        # filename: arch1_trial1.json etc.
        stem = run_file.stem
        arch_key = stem.split("_trial")[0] if "_trial" in stem else stem
        summary = summarise_run(arch_key, run_file, labels)
        per_arch_runs[arch_key].append(summary)

    if not per_arch_runs:
        print("No runs in results/raw_outputs/. Run `python -m eval.run_eval` first.")
        return

    md = render_markdown(per_arch_runs, labels)
    if args.print_only:
        print(md)
    else:
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(md)
        print(f"Wrote {SUMMARY.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
