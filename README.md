# Email Triage Agents

Comparative study of three agent architectures on the same email-triage task.

See [PLAN.md](PLAN.md) for the full experimental design, hypotheses, and publication plan.

## Quickstart

```bash
pip install -r requirements.txt

# Put your Anthropic key in .env (already gitignored)
cp .env.example .env
$EDITOR .env   # paste your sk-ant-... key

# Smoke test the harness without spending API tokens (deterministic mock backend)
python -m eval.run_eval --mock --trials 1

# Real run: 3 architectures x 3 trials over the 50-email dataset
python -m eval.run_eval --trials 3
```

The `.env` file is loaded automatically by `llm_client.py` when `python-dotenv` is installed (it's in `requirements.txt`). If you'd rather not use `.env`, `export ANTHROPIC_API_KEY=...` in your shell still works.

Results land in `results/raw_outputs/` and are summarised in `results/summary.md`.

## Layout

```
data/                     50-email dataset, labels, reference replies (JSONL)
architectures/            arch1_single_call.py, arch2_react_loop.py, arch3_graph.py
tools/mock_tools.py       Canned tools for Arch 2 (search past emails, calendar, etc.)
prompts/                  System prompts for each architecture and the judge rubric
eval/                     run_eval.py, metrics.py, judge.py
results/                  raw_outputs/ (one JSON per arch x trial) and summary.md
```

## Architectures under test

1. **Single structured call** — one LLM call per email with a JSON schema. The boring baseline.
2. **ReAct loop** — hand-rolled tool-use loop with 4 mock tools (search past emails, check calendar, get user context, lookup thread).
3. **LangGraph multi-step** — explicit graph: `triage -> route -> draft -> critique -> (revise) -> output`.

All three run on the same model (default `claude-sonnet-4-6`). The judge runs on a different model (default `claude-opus-4-6`).

## Reproducing the writeup numbers

The writeup numbers in `results/summary.md` are produced by:

```bash
python -m eval.run_eval --trials 3 --model claude-sonnet-4-6 --judge-model claude-opus-4-6
python -m eval.metrics > results/summary.md
```

## Notes on honest comparison

- The ReAct tools return canned data — explicitly disclosed. The point is to measure whether *access to tools* changes outcomes, not to build a real email backend.
- Prompt-engineering effort was spent equally on each architecture's prompts. Read them in `prompts/`.
- Each architecture is run 3x to capture residual variance at temperature 0. The judge is also run at temperature 0.
- See [PLAN.md](PLAN.md) for the full set of design decisions and predicted findings.
