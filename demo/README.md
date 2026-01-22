# TriModel-Bench Demo (MVP)

This repository is a minimal runnable prototype for **TriModel-Bench** that demonstrates:

- **Tier-1: Situational model evaluation** (problem understanding via closed QA, scored by expert semantic score + brief comment; offline fallback exists)
- **Tier-2: Real model evaluation** (structured abstraction; similarity scored by lexical Jaccard or expert semantic matching when available)
- **Tier-3: Mathematical model evaluation** (LLM generates solver-ready Python; solver verifies with **Mamo-style relative error threshold**)

## Project Structure

```
demo/
  benchmark/
    data/
      example_problem.jsonl
    evaluators/
      situational_model_evaluator.py
      real_model_evaluator.py
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

2) Set your API keys (recommended). If not set, the demo falls back to deterministic mock outputs:

```bash
export API_KEY_DOUBAO="YOUR_KEY_FOR_DOUBAO"
export API_KEY_QWEN="YOUR_KEY_FOR_QWEN"
```

You can also put it in a `.env` file (repo root or `demo/`), for example:

```bash
API_KEY_DOUBAO=YOUR_KEY_FOR_DOUBAO
API_KEY_QWEN=YOUR_KEY_FOR_QWEN
```

Optional runtime knobs:

```bash
export TRIMODEL_OFFLINE=1        # force-disable all remote LLM calls
export LLM_TIMEOUT_SEC=30        # request timeout seconds
export LLM_MAX_RETRIES=1         # retry count
```

## Run

```bash
python3 -m demo.demo
```

By default, `demo/demo.py` selects `problems[1]` (the second entry) from the dataset. You can change it to `problems[0]`.

## What You Should See

- Tier-1: Each question has
  - LLM answer
  - semantic score (expert LLM; falls back to lexical score offline)
  - one-sentence expert comment (or heuristic offline)
- Tier-2: Structured real-model output is evaluated by
  - Jaccard similarity vs `real_model_ground_truth`
  - expert LLM multi-dimension scores (or heuristic offline)
- Tier-3: solver verification output (LP/ODE/Polynomial Regression Degree-2)
  - pass/fail using relative error threshold (default `1e-4`)
  - prints parsed model spec + solve trace (fallback solvers exist if third-party deps are missing)
- A report is written to `doc/modeling_report.md`
