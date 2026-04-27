# Email Triage Agents — Results Summary

Dataset: 50 emails. Categories: reply_now (16), reply_later (11), archive (13), unsubscribe (10).

Model under test: `claude-sonnet-4-6`. Judge model: `claude-opus-4-6`.

## Headline metrics (mean ± stdev across trials)

| Architecture | Accuracy | Over-reply rate | Reliability | Cost / email | Total cost | Latency p50 | Latency p95 | Judge score |
|---|---|---|---|---|---|---|---|---|
| arch1 | 0.800 ± 0.000 | 0.029 | 1.000 | $0.00370 | $0.1848 | 2897 ms | 6388 ms | 4.36/5 |
| arch2 | 0.740 ± 0.000 | 0.294 | 1.000 | $0.01299 | $0.6493 | 8384 ms | 16689 ms | 4.00/5 |
| arch3 | 0.780 ± 0.000 | 0.059 | 1.000 | $0.00447 | $0.2237 | 3265 ms | 11691 ms | 4.27/5 |

## Per-category F1 (last trial)

| Architecture | reply_now | reply_later | archive | unsubscribe |
|---|---|---|---|---|
| arch1 | 0.90 | 0.63 | 0.73 | 0.90 |
| arch2 | 0.75 | 0.27 | 0.85 | 0.95 |
| arch3 | 0.88 | 0.56 | 0.73 | 0.90 |

## Notes on reading these numbers

- **Accuracy** is exact-match against `data/labels.jsonl`. ~9 emails are deliberately ambiguous; perfect accuracy is not the target.
- **Over-reply rate** is the fraction of non-`reply_now` emails for which the architecture nonetheless drafted a reply. Lower is better.
- **Reliability** is the share of predictions that parsed cleanly into the required JSON schema.
- **Judge score** is the mean of the LLM-as-judge overall (1-5) across all `reply_now` emails.
- All figures aggregate across the trials specified at `--trials`.
