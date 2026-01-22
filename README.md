# TriModel-Bench Demo (MVP)

This repo contains a runnable demo prototype for the PhD project **TriModel-Bench**.
The demo implements a 3-tier evaluation pipeline aligned with the cognitive path:

1) Situation Model (Tier-1): closed QA for problem understanding
2) Real Model (Tier-2): abstraction to structured elements (variables/assumptions/constraints/concepts)
3) Mathematical Model (Tier-3): generate solver-ready code/spec and verify by execution

All runnable code lives under `demo/`.

## Quick Start

```bash
python -m demo.demo
```

By default, `demo/demo.py` selects the second entry (`problems[1]`) in the dataset; edit it if you want `problems[0]`.

To run without calling any remote model APIs (offline / mock mode):

```bash
export TRIMODEL_OFFLINE=1
python -m demo.demo
```

## API Keys (.env)

You can put keys in a repo-root `.env` (or `demo/.env`):

```bash
API_KEY_DOUBAO=YOUR_KEY_FOR_DOUBAO
API_KEY_QWEN=YOUR_KEY_FOR_QWEN
```

Optional runtime knobs:
- `LLM_TIMEOUT_SEC` (default: `30`)
- `LLM_MAX_RETRIES` (default: `1`)
- `TRIMODEL_OFFLINE=1` (force-disable remote calls)

## Data Format

`demo/benchmark/data/example_problem.jsonl` is a JSONL file (one JSON object per line).
For convenience, the loader also supports a single JSON array file (`[...]`).

Each entry contains:
- Tier-1: `context`, `questions`, `ground_truth_answers`
- Tier-2: `real_model_ground_truth`
- Tier-3: `math_model_type`, `solver_code_prompt`, `standard_answer`

Supported `math_model_type`:
- `LP`
- `ODE`
- `Polynomial Regression (Degree 2)`

## Outputs

When you run `python -m demo.demo`:
- Console prints Tier-1/Tier-2/Tier-3 results (including solver trace and readable math formulation).
- A report is written to `doc/modeling_report.md` (expert model if available; otherwise an offline template).
