# Email Triage Agents — Experiment Plan

A comparative study of three agent architectures on the same realistic task. The goal is to answer a question every applied AI team is asking: **do we actually need an agent here, or is one good prompt enough?**

The output of this project is a public writeup (LinkedIn + GitHub repo) with honest, reproducible measurements of accuracy, cost, latency, and reliability across architectures.

---

## The task

Given an inbox of 50 emails, the system must:

1. **Classify** each email into one of four categories:
   - `reply_now` — needs a response soon, draft one
   - `reply_later` — needs a response but not urgent
   - `archive` — informational, no response needed
   - `unsubscribe` — promotional/newsletter, should be removed
2. **Draft a reply** for every email classified as `reply_now`
3. **Return structured output** per email:
   ```json
   {
     "email_id": "string",
     "category": "reply_now | reply_later | archive | unsubscribe",
     "confidence": 0.0,
     "draft_reply": "string or null",
     "reasoning": "string"
   }
   ```

---

## The dataset

**Target: 50 emails with hand-labeled ground truth.**

Composition:
- ~35 synthetic emails generated with a strong model, deliberately varied across:
  - Newsletters and promotional content
  - Calendar/meeting requests
  - Client or customer questions
  - Sales pitches and cold outreach
  - Internal team threads
  - Personal notes from friends/family
  - Phishing-adjacent suspicious mail
  - Receipts and notifications
- ~15 real emails from personal archive, redacted (names, companies, specifics replaced)
- **8–10 deliberately ambiguous emails** — short, context-free, or borderline between two categories. These are where architectures will diverge, and where the post gets interesting.

For each `reply_now` email, also write a **reference reply** for qualitative comparison (not exact-match scoring).

Files:
- `data/emails.jsonl` — the 50 emails
- `data/labels.jsonl` — ground-truth categories
- `data/reference_replies.jsonl` — human-written reference drafts

---

## The three architectures

All three are evaluated on the **same model** for the primary comparison (suggested: Claude Sonnet or GPT-4o-class). A secondary run with a smaller model on Arch 1 is a stretch goal.

### Architecture 1 — Single structured call

One LLM call per email (or small batch), schema-constrained JSON output. No tools, no loop.

```
[email] → [LLM with system prompt + JSON schema] → [classification + draft]
```

- ~50 lines of code
- The "boring baseline" — often the one that wins on cost and latency
- File: `architectures/arch1_single_call.py`

### Architecture 2 — ReAct loop with tools

The agent has access to mock tools and decides when to use them.

Tools (mock implementations returning canned data — explicitly documented as such):
- `search_past_emails(sender)` — has the user replied to this person before? what tone?
- `check_calendar(date_range)` — is the user free when a meeting is proposed?
- `get_user_context()` — small profile: role, current projects, communication style
- `lookup_thread(email_id)` — fetch prior messages in the thread

- ~200 lines of code
- Hand-rolled loop, no framework
- File: `architectures/arch2_react_loop.py`

### Architecture 3 — Multi-step graph (LangGraph)

Explicit graph with self-correction:

```
email → triage → (if reply_now) → draft → critique → (if fails) → revise → output
                ↓
              (else) → output
```

Nodes:
- `triage` — classify only
- `route` — branch on category
- `draft` — generate reply for `reply_now`
- `critique` — second LLM call reviews tone, length, factual grounding
- `revise` — regenerate once if critique fails

- ~350 lines of code
- The "fancy" version with self-correction
- File: `architectures/arch3_graph.py`

---

## Evaluation

### Quantitative metrics

| Metric | How to measure |
|---|---|
| **Classification accuracy** | Exact match against labels, plus per-category F1 |
| **Draft quality** | LLM-as-judge with rubric (1–5 on tone, addresses question, length, no hallucinated facts). Judge model must differ from the model being tested. |
| **Cost per email** | Sum of input + output tokens × pricing |
| **Latency** | p50 and p95 per email |
| **Reliability** | % of runs completing without schema violation, timeout, or tool error |
| **Variance** | Each architecture run 3× over the full set |

### Qualitative analysis (the applied-product gold)

- Which architecture handles ambiguous emails better?
- When Arch 3's critique fires, is it catching real problems or hallucinating new ones?
- Does Arch 2 actually use its tools, or ignore them and behave like Arch 1?
- Does any architecture **over-reply** — drafting for things that should've been archived? (An agent that drafts too aggressively is worse than one that drafts too little.)

---

## Design decisions (lock these before coding)

- **Model:** Same strong model for all three architectures in primary run. Stretch: re-run Arch 1 on a cheaper model.
- **Temperature:** 0 (or lowest available) for agents and judge. Multi-trial anyway to capture residual variance.
- **Batching for Arch 1:** Decide one-by-one vs. batched (e.g., 5 per call). Document the choice — it materially affects cost and latency.
- **Prompt fairness:** Spend roughly equal effort on the system prompt for each architecture. Document all prompts in the repo. A common failure mode of these comparisons is that the baseline gets a tuned prompt and the "fancy" version gets a hasty one (or vice versa).
- **Tool implementations:** Mock data is fine and should be explicitly disclosed. The point is to measure whether *access to tools* changes outcomes, not to build a real email backend.

---

## Predicted findings (to validate or contradict)

These are hypotheses, not conclusions. The writeup gets interesting whether they hold or not.

- Arch 1 hits ~85–90% classification accuracy at ~$0.003/email and ~1s latency
- Arch 2 doesn't help much on classification but produces noticeably better drafts (tone-matching from past emails). ~3× the cost of Arch 1.
- Arch 3's critique catches ~20% of bad drafts but introduces its own errors ~5% of the time — the "self-correction tax." 5–8× cost, 4× latency.
- The most interesting finding will likely be in the ambiguous bucket: one architecture will have a specific failure mode the others don't.

If actual results contradict these, the post writes itself: *"I expected X, got Y, here's why."*

---

## Repo structure

```
email-triage-agents/
├── README.md
├── PLAN.md                       # this file
├── data/
│   ├── emails.jsonl
│   ├── labels.jsonl
│   └── reference_replies.jsonl
├── architectures/
│   ├── arch1_single_call.py
│   ├── arch2_react_loop.py
│   └── arch3_graph.py
├── tools/
│   └── mock_tools.py
├── eval/
│   ├── run_eval.py               # runs all 3 archs × 3 trials
│   ├── judge.py                  # LLM-as-judge for draft quality
│   └── metrics.py
├── prompts/
│   ├── arch1_system.txt
│   ├── arch2_system.txt
│   ├── arch3_triage.txt
│   ├── arch3_draft.txt
│   ├── arch3_critique.txt
│   └── judge_rubric.txt
├── results/
│   ├── raw_outputs/              # full JSON per run, gitignored if large
│   └── summary.md                # the table that goes in the LinkedIn post
└── requirements.txt
```

---

## Build sequence

Suggested order. Each step should be a separate commit (or PR) so the build-in-public timeline is visible.

1. **Repo scaffold** — directory structure, `requirements.txt`, README stub
2. **Dataset generation** — generate 35 synthetic emails, redact 15 real ones, hand-label all 50, write reference replies for `reply_now` emails
3. **Architecture 1** — single structured call, end to end
4. **Eval harness** — `run_eval.py` and `metrics.py` working against Arch 1 only
5. **LLM-as-judge** — `judge.py` with rubric, validated on a few examples by hand
6. **Architecture 2** — ReAct loop with mock tools
7. **Architecture 3** — LangGraph version with critique/revise
8. **Full eval run** — all 3 architectures × 3 trials, write `results/summary.md`
9. **Writeup** — README with results table, architecture diagram, key findings
10. **LinkedIn sequence** — teaser, mid-experiment finding, results post, followup

---

## Publication plan

Four posts over four weeks:

1. **Teaser (week 1).** "Running an experiment: same email triage task, three architectures. Predictions?" Link to repo with PLAN.md committed.
2. **Mid-experiment (week 2).** Share one surprising finding or failure mode, with a code snippet or screenshot.
3. **Results (week 3).** Full writeup, results table, repo, opinionated takeaway.
4. **Followup (week 4).** "Three things I'd do differently" or "applying this approach to a different task."

---

## Open questions to resolve during build

- Final model choice for primary run
- Whether to batch in Arch 1, and at what batch size
- Which LangGraph features to use in Arch 3 (just nodes/edges, or also checkpointing/streaming?)
- How to handle multi-turn email threads in the dataset (include any? how to format?)
- Whether to publish the synthetic dataset as a standalone artifact (it might be useful to others)
