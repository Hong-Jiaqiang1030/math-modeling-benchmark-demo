# TriModel-Bench Demo (MVP)

This repository is a minimal runnable prototype for **TriModel-Bench** that demonstrates:

- **Tier-1: Situational model evaluation** (problem understanding via closed QA, scored by F1 + brief "expert LLM" comment)
- **Tier-3: Mathematical model evaluation for closed-ended tasks** (LLM generates solver-ready code; external solver verifies with **Mamo-style relative error threshold**)

## Project Structure

```
demo/
  benchmark/
    data/
      example_problem.jsonl
    evaluators/
      situational_model_evaluator.py
      mathematical_model_evaluator.py
    tools/
      solver_wrapper.py
  demo.py
  requirements.txt
```

## Setup

1) Create a virtual environment (optional) and install dependencies:

```bash
pip install -r requirements.txt
```

2) Set your OpenAI API key (recommended). If not set, the demo falls back to deterministic mock outputs:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

You can also put it in a `.env` file (repo root or `demo/`), for example:

```bash
OPENAI_API_KEY=YOUR_KEY
```

## Run

```bash
python3 demo.py
```

## What You Should See

- Tier-1: Each question has
  - LLM answer
  - token-overlap **F1 score**
  - one-sentence "expert" comment (another LLM, or heuristic if no API key)
- Tier-3 (LP example): LLM-produced LP spec is solved by PuLP; the objective value is compared with `standard_answer`
  - pass/fail using relative error threshold (default `1e-4`)
