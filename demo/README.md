# TriModel-Bench Demo (MVP)

This repository is a minimal runnable prototype for **TriModel-Bench** that demonstrates:

- **Tier-1: Situational model evaluation** (problem understanding via closed QA, scored by F1 + brief "expert LLM" comment)
- **Tier-2: Real model evaluation** (structured abstraction, scored by Jaccard similarity + expert LLM multi-dim scores)
- **Tier-3: Mathematical model evaluation for closed-ended tasks** (LLM generates solver-ready code; external solver verifies with **Mamo-style relative error threshold**)

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

## Run

```bash
python3 -m demo.demo
```

## What You Should See

- Tier-1: Each question has
  - LLM answer
  - token-overlap **F1 score**
  - one-sentence "expert" comment (another LLM, or heuristic if no API keys)
- Tier-2: Structured real-model output is evaluated by
  - Jaccard similarity vs `real_model_ground_truth`
  - expert LLM multi-dimension scores (or heuristic offline)
- Tier-3 (LP example): LLM-produced LP spec is solved by PuLP; the objective value is compared with `standard_answer`
  - pass/fail using relative error threshold (default `1e-4`)
  - prints parsed model spec + solve trace (bruteforce fallback if PuLP is not installed)
