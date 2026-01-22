import json
from pathlib import Path

from benchmark.evaluators.situational_model_evaluator import evaluate_situational_model
from benchmark.evaluators.mathematical_model_evaluator import evaluate_mathematical_model
from benchmark.utils.env import load_env


def load_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    load_env()
    data_path = Path(__file__).parent / "benchmark" / "data" / "example_problem.jsonl"
    problems = load_jsonl(data_path)
    if not problems:
        raise RuntimeError(f"No problems found in {data_path}")

    p = problems[0]
    print("=== TriModel-Bench Demo (MVP) ===")
    print(f"problem_id: {p['problem_id']}")
    print()

    # Tier-1: situation model evaluation
    print("---- Tier-1: Situational Model Evaluation ----")
    tier1 = evaluate_situational_model(
        context=p["context"],
        questions=p["questions"],
        ground_truth_answers=p["ground_truth_answers"],
        llm_model="gpt-4o",
    )
    for i, row in enumerate(tier1["per_question"], 1):
        print(f"[Q{i}] {row['question']}")
        print(f"  LLM answer: {row['predicted_answer']}")
        print(f"  Ground truth: {row['ground_truth_answer']}")
        print(f"  F1: {row['f1']:.3f}")
        print(f"  Expert comment: {row['expert_comment']}")
        print()
    print(f"Tier-1 mean F1: {tier1['mean_f1']:.3f}")
    print()

    # Tier-3: mathematical modeling evaluation (closed-ended)
    print("---- Tier-3: Mathematical Model Solve Verification ----")
    tier3 = evaluate_mathematical_model(
        solver_code_prompt=p["solver_code_prompt"],
        standard_answer=p["standard_answer"],
        math_type=p["math_model_type"],
        llm_model="gpt-4o",
    )
    print(f"math_type: {p['math_model_type']}")
    print(f"pass: {tier3['passed']}")
    print(f"solver_output: {tier3['solver_output']}")
    if tier3.get("relative_error") is not None:
        print(f"relative_error: {tier3['relative_error']:.6g}")
    if tier3.get("llm_code") is not None:
        print("\n(LLM code used)\n")
        print(tier3["llm_code"])


if __name__ == "__main__":
    main()
