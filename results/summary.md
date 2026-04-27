# Email Triage Agents — Results Summary

Dataset: 50 emails. Categories: reply_now (16), reply_later (11), archive (13), unsubscribe (10).

Model under test: `claude-sonnet-4-6`. Judge model: `claude-opus-4-6`.

## Headline metrics (mean ± stdev across trials)

| Architecture | Accuracy | Over-reply rate | Reliability | Cost / email | Total cost | Latency p50 | Latency p95 | Judge score |
|---|---|---|---|---|---|---|---|---|
| arch1 | 0.627 ± 0.543 | 0.000 | 0.667 | $0.00000 | $0.0000 | 1 ms | 1 ms | 2.69/5 |
| arch2 | 0.640 ± 0.554 | 0.020 | 0.667 | $0.00000 | $0.0000 | 2 ms | 2 ms | 3.33/5 |
| arch3 | 0.627 ± 0.543 | 0.000 | 0.667 | $0.00000 | $0.0000 | 1 ms | 3 ms | 3.33/5 |

## Per-category F1 (last trial)

| Architecture | reply_now | reply_later | archive | unsubscribe |
|---|---|---|---|---|
| arch1 | 1.00 | 0.87 | 0.88 | 1.00 |
| arch2 | 0.97 | 0.95 | 0.96 | 0.95 |
| arch3 | 1.00 | 0.90 | 0.90 | 0.95 |

## Notes on reading these numbers

- **Accuracy** is exact-match against `data/labels.jsonl`. ~9 emails are deliberately ambiguous; perfect accuracy is not the target.
- **Over-reply rate** is the fraction of non-`reply_now` emails for which the architecture nonetheless drafted a reply. Lower is better.
- **Reliability** is the share of predictions that parsed cleanly into the required JSON schema.
- **Judge score** is the mean of the LLM-as-judge overall (1-5) across all `reply_now` emails.
- All figures aggregate across the trials specified at `--trials`.
