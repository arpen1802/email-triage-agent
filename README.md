# Email Triage

Open-source CLI that triages your Gmail inbox using a single LLM call per message. Categorises every email as `reply_now` / `reply_later` / `archive` / `unsubscribe`, applies Gmail labels you can filter on, and (optionally) creates draft replies in your Drafts folder for the urgent ones.

Costs roughly **$0.004 per email** at default settings. For a typical 50-email-per-day inbox, that's about **$60 a year in API charges** — vs. $84-360/year for SaaS triage tools, or $3,000/year for Google's AI Ultra. Your data stays on your machine and in Google's API, never on a third-party server.

Built on top of an [architecture comparison](#why-just-one-llm-call) showing that the simple single-call approach beats ReAct loops and LangGraph multi-step pipelines on this task — at a fraction of the cost. The eval harness, dataset, and prompts are all in this repo.

## Quickstart

```bash
git clone https://github.com/arpen1802/email-triage-agent.git
cd email-triage-agent
python3 -m venv triage-venv && source triage-venv/bin/activate
pip install -r requirements.txt
```

Two keys to configure once:

```bash
# 1. Anthropic API key — put it in .env (gitignored)
cp .env.example .env
$EDITOR .env   # paste your sk-ant-... key

# 2. Gmail OAuth — see "Gmail setup" below for the one-time Google Cloud setup.
#    Once credentials.json is at the repo root, run:
python -m cli gmail setup
```

Then run a dry-run triage on your last 20 emails — nothing changes in your inbox, just a printed summary:

```bash
python -m cli gmail triage --last 20
```

When you're happy, let it actually act:

```bash
# Adds AI/ReplyNow, AI/ReplyLater, AI/Archive, AI/Unsubscribe labels in Gmail
python -m cli gmail triage --last 50 --apply-labels

# Plus creates drafts in your Drafts folder for reply_now emails (never sends)
python -m cli gmail triage --last 50 --apply-labels --create-drafts

# Idempotent: skip emails already labelled by a previous run
python -m cli gmail triage --last 100 --apply-labels --create-drafts --skip-triaged
```

The tool **never sends email**. Drafts go to your Drafts folder; you review and press send (or delete).

## Gmail setup (one time)

You need a Google Cloud OAuth client ID. This is annoying but free, and your data stays in Google's API — nothing flows through a third-party server.

1. Open [console.cloud.google.com](https://console.cloud.google.com/) and create a new project (free).
2. APIs & Services → Library → search **Gmail API** → Enable.
3. APIs & Services → OAuth consent screen → User type **External**. Fill the required app fields. Add your own email address as a **test user**.
4. APIs & Services → Credentials → Create credentials → **OAuth client ID**. Application type: **Desktop app**.
5. Download the JSON, rename to `credentials.json`, place at the repo root.

Then `python -m cli gmail setup` opens a browser, you grant access, and the resulting refresh token is cached in `token.json`. Both files are gitignored.

The tool requests **only** the `gmail.modify` scope: read messages, apply/remove labels, create drafts. It cannot send mail.

## Why just one LLM call?

The first thing you'd reach for in 2026 to do email triage is an "agent" — a multi-step pipeline with tools, self-correction, and a graph. I built three versions to see which actually works:

1. **Single LLM call** with a JSON schema. ~50 lines of code.
2. **ReAct loop** with four mock tools (search past emails, check calendar, get user context, lookup thread). ~200 lines.
3. **LangGraph multi-step pipeline** (triage → draft → critique → revise). ~350 lines.

Same model on all three (Claude Sonnet 4.6), 50 hand-labeled emails, equal effort spent on each prompt:

| Architecture | Accuracy | Over-reply rate | Cost / email | Total cost | Latency p50 | Draft quality |
|---|---|---|---|---|---|---|
| **Single call** | **80%** | 2.9% | **$0.0037** | **$0.18** | **2.9 s** | **4.36/5** |
| ReAct loop | 74% | 29.4% | $0.0130 | $0.65 | 8.4 s | 4.00/5 |
| Multi-step graph | 78% | 5.9% | $0.0045 | $0.22 | 3.3 s | 4.27/5 |

The single call won every metric. The most striking failure was the ReAct loop, which drafted replies to 29% of emails it shouldn't have — including newsletters, calendar reminders, and a phishing email. Adding tools didn't make the model smarter; it nudged it toward "this is a real human, draft a reply" on emails that should have been archived.

That's why this CLI uses just one prompt per email. The full eval and dataset are in this repo if you want to reproduce the comparison or run it on a different model — see [Reproducing the comparison](#reproducing-the-comparison) below.

## Layout

```
cli.py                       CLI entry: python -m cli gmail {setup,triage}
gmail_triage/                Gmail integration
  auth.py                    OAuth flow
  inbox.py                   Reads messages, normalises to email dict
  actions.py                 Applies labels, creates drafts (never sends)
architectures/               The three architectures benchmarked above
  arch1_single_call.py       The one the CLI actually uses
  arch2_react_loop.py        ReAct loop with mock tools
  arch3_graph.py             LangGraph multi-step pipeline
tools/mock_tools.py          Canned tools for Arch 2 — research only
prompts/                     System prompts for each architecture and the judge
data/                        50-email evaluation dataset (JSONL)
eval/                        run_eval.py, metrics.py, judge.py
results/                     raw_outputs/ and summary.md
```

## Reproducing the comparison

```bash
# Mock backend — validates the harness without spending API tokens
python -m eval.run_eval --mock --trials 1

# Real run with the live API
python -m eval.run_eval --trials 1 --fresh
python -m eval.metrics
```

The summary lands at `results/summary.md`. Run with `--trials 3` for variance bars.

## Cost and privacy

**Per-message cost** at default settings (`claude-sonnet-4-6`): ~$0.004. Cheaper with `--model claude-haiku-4-5`. The Anthropic SDK is the only place your message bodies are sent outside of Google.

**No third-party server**: the tool runs on your machine. Gmail API calls go directly Google ↔ your machine. LLM calls go directly Anthropic ↔ your machine. There is no middleware operator that sees your mail.

**Scope-limited**: the OAuth scope is `gmail.modify`. The tool can read, label, and draft. It cannot send mail and cannot access other Google services.

**Reversible**: labels can be removed manually or by deleting `AI/*` labels in Gmail's labels page. Drafts can be deleted from your Drafts folder. Nothing the tool does is irreversible.

## License

MIT.
