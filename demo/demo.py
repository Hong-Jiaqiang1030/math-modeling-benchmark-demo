import json
import sys
from datetime import datetime
from pathlib import Path

# When executed as a script (e.g. `python3 demo/demo.py` or `python3 demo.py` inside `demo/`),
# add the repo root to sys.path so `import demo...` works.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demo.benchmark.evaluators.situational_model_evaluator import evaluate_situational_model
from demo.benchmark.evaluators.real_model_evaluator import evaluate_real_model
from demo.benchmark.evaluators.mathematical_model_evaluator import evaluate_mathematical_model
from demo.benchmark.tools.llm_chat import DEFAULT_LLM_MODEL, DEFAULT_EXPERT_MODEL, call_chat, has_credentials
from demo.benchmark.utils.env import load_env


def load_jsonl(path: Path):
    """
    Load either JSONL (one JSON object per line) or a single JSON array file.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]

    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def _format_linear_expr(coeffs: dict) -> str:
    if not isinstance(coeffs, dict) or not coeffs:
        return "0"
    parts = []
    for name, coef in coeffs.items():
        try:
            c = float(coef)
        except Exception:
            c = coef
        parts.append(f"{c}*{name}")
    return " + ".join(parts)


def main():
    load_env()
    data_path = Path(__file__).parent / "benchmark" / "data" / "example_problem.jsonl"
    problems = load_jsonl(data_path)
    if not problems:
        raise RuntimeError(f"No problems found in {data_path}")

    p = problems[1]
    print("=== TriModel-Bench Demo (MVP) ===")
    print(f"problem_id: {p['problem_id']}")
    print()

    # Tier-1: situation model evaluation
    print("---- Tier-1: Situational Model Evaluation ----")
    tier1 = evaluate_situational_model(
        context=p["context"],
        questions=p["questions"],
        ground_truth_answers=p["ground_truth_answers"],
        llm_model=DEFAULT_LLM_MODEL,
        expert_model=DEFAULT_EXPERT_MODEL,
    )
    for i, row in enumerate(tier1["per_question"], 1):
        print(f"[Q{i}] {row['question']}")
        print(f"  LLM answer: {row['predicted_answer']}")
        print(f"  Ground truth: {row['ground_truth_answer']}")
        print(f"  Semantic score: {row['semantic_score']:.3f}")
        print(f"  Expert comment: {row['expert_comment']}")
        print()
    print(f"Tier-1 mean semantic score: {tier1['mean_semantic_score']:.3f}")
    print()

    # Tier-2: real model evaluation (combined automated + expert LLM)
    print("---- Tier-2: Real Model Evaluation (Combined) ----")
    tier2 = evaluate_real_model(
        context=p["context"],
        ground_truth=p.get("real_model_ground_truth", {}),
        llm_model=DEFAULT_LLM_MODEL,
        expert_model=DEFAULT_EXPERT_MODEL,
    )
    print(f"Similarity mode: {tier2.get('similarity_mode')}")
    print("Real-model predicted:")
    print(json.dumps(tier2.get("predicted", {}), ensure_ascii=False, indent=2))
    print("Real-model ground truth:")
    print(json.dumps(p.get("real_model_ground_truth", {}), ensure_ascii=False, indent=2))
    sim = tier2["similarity"]
    print(f"Jaccard (variables): {sim['variables_jaccard']:.3f}")
    print(f"Jaccard (assumptions): {sim['assumptions_jaccard']:.3f}")
    print(f"Jaccard (constraints): {sim['constraints_jaccard']:.3f}")
    print(f"Jaccard (concept_triples): {sim['concept_triples_jaccard']:.3f}")
    print(f"Tier-2 mean Jaccard: {sim['mean_jaccard']:.3f}")

    expert = tier2["expert"]
    if isinstance(expert, dict) and "raw" not in expert:
        print(
            "Expert scores "
            f"(reasonableness/completeness/simplicity/innovativeness): "
            f"{expert.get('reasonableness')}/"
            f"{expert.get('completeness')}/"
            f"{expert.get('simplicity')}/"
            f"{expert.get('innovativeness')}"
        )
        if expert.get("comment"):
            print(f"Expert comment: {expert.get('comment')}")
    else:
        print("Expert output (raw):")
        print(expert.get("raw", expert))
    print()

    # Tier-3: mathematical modeling evaluation (closed-ended)
    print("---- Tier-3: Mathematical Model Solve Verification ----")
    tier3 = evaluate_mathematical_model(
        solver_code_prompt=p["solver_code_prompt"],
        standard_answer=p["standard_answer"],
        math_type=p["math_model_type"],
        llm_model=DEFAULT_LLM_MODEL,
        context=p.get("context"),
        situational_results=tier1,
        real_model_results=tier2,
    )
    math_type = str(p["math_model_type"]).upper()
    solver_out = tier3.get("solver_output") or {}

    print(f"Math type: {math_type}")
    print(f"Passed: {tier3['passed']}")
    print(f"Standard answer: {p['standard_answer']}")
    if tier3.get("relative_error") is not None:
        print(f"Relative error: {tier3['relative_error']:.6g} (threshold={tier3.get('threshold')})")
    else:
        print(f"Relative error: N/A (threshold={tier3.get('threshold')})")

    if isinstance(solver_out, dict):
        trace = solver_out.get("trace") or {}
        method = trace.get("method") if isinstance(trace, dict) else None
        status = solver_out.get("status")
        print("\nSolve summary:")
        print(f"- Method: {method}")
        print(f"- Solver status: {status}")
        if math_type == "LP":
            obj_val = solver_out.get("objective_value")
            sol = solver_out.get("variables") or {}
            print(f"- Objective value: {obj_val}")
            print(f"- Solution: {json.dumps(sol, ensure_ascii=False)}")
            if not tier3["passed"] and obj_val is not None:
                print("\nNote:")
                print(
                    f"- Solver objective != standard answer ({obj_val} vs {p['standard_answer']}); please verify the dataset constraints/standard_answer."
                )
        elif "final_efficiency" in solver_out:
            yhat = solver_out.get("final_efficiency")
            xq = solver_out.get("x_query")
            print(f"- Predicted value: final_efficiency={yhat} at x={xq}")

    if tier3.get("llm_code") is not None:
        print("\nLLM-generated code:\n")
        print(tier3["llm_code"])

    # Print a transparent "math model" view + solve trace (LP/ODE).
    if isinstance(solver_out, dict):
        trace = solver_out.get("trace")
        if trace is not None:
            print("\nSolve trace:")
            print(json.dumps(trace, ensure_ascii=False, indent=2))

        if math_type == "LP" and "lp_spec" in solver_out:
            lp_spec = solver_out.get("lp_spec") or {}
            print("\nParsed model (lp_spec):")
            print(json.dumps(lp_spec, ensure_ascii=False, indent=2))

            vars_ = (lp_spec.get("variables") or {}) if isinstance(lp_spec, dict) else {}
            obj = (lp_spec.get("objective") or {}) if isinstance(lp_spec, dict) else {}
            cons = (lp_spec.get("constraints") or []) if isinstance(lp_spec, dict) else []

            print("\nMath formulation:")
            print(f"- Decision variables (with bounds/types): {json.dumps(vars_, ensure_ascii=False)}")
            print(f"- Objective: maximize {_format_linear_expr(obj)}")
            print(f"- Constraints: {len(cons)}")
            for i, c in enumerate(cons):
                lhs = c.get("lhs", {}) or {}
                sense = c.get("sense", "<=")
                rhs = c.get("rhs", 0)
                print(f"  - c{i}: {_format_linear_expr(lhs)} {sense} {rhs}")

            cons_eval = solver_out.get("constraints_eval") or []
            if cons_eval:
                print("\nConstraint check at solution:")
                for row in cons_eval:
                    ok = "OK" if row.get("satisfied") else "FAIL"
                    print(
                        f"- {row.get('name')}: lhs={row.get('lhs_value')} {row.get('sense')} rhs={row.get('rhs')} => {ok}"
                    )

            # Explain the model in plain language.
            if isinstance(vars_, dict):
                print("\nModel explanation:")
                print("- 变量表示要购买的数量（例如 A=买A型车数量，B=买B型车数量），并受上下限约束。")
                print("- 目标函数最大化总数量（A+B）。")
                print("- 预算约束把单价与购买数量相乘并求和，要求不超过总预算。")
        elif math_type.startswith("POLYNOMIAL") or "POLYNOMIAL" in math_type:
            if "final_efficiency" in solver_out:
                print("\nModel explanation:")
                print("- 多项式回归（2阶）假设 y = a*x^2 + b*x + c，通过最小二乘拟合 (x, y) 数据点。")
                print("- 先拟合得到系数 (a,b,c)，再在指定 x 上代入得到预测值 final_efficiency。")

    # Save an expert analysis report of the end-to-end modeling process.
    repo_root = Path(__file__).resolve().parents[1]
    doc_dir = repo_root / "doc"
    doc_dir.mkdir(parents=True, exist_ok=True)
    report_path = doc_dir / "modeling_report.md"
    # Clean up any legacy timestamped reports (keep only the latest single report file).
    for old in doc_dir.glob("*_modeling_report.md"):
        try:
            old.unlink()
        except Exception:
            pass

    report_md: str
    can_call_expert = has_credentials(DEFAULT_EXPERT_MODEL)
    if can_call_expert:
        try:
            report_md = call_chat(
                DEFAULT_EXPERT_MODEL,
                [
                    {
                        "role": "system",
                        "content": (
                            "你是数学建模评审专家。请对“从文本到真实模型到数学模型再到求解验证”的全过程"
                            "输出一份结构化评审报告（Markdown）。要求：\n"
                            "1) 简述问题重述与建模目标\n"
                            "2) 解释决策变量/目标函数/约束的建立是否合理，是否与题意一致\n"
                            "3) 检查标准答案与约束是否一致（如不一致请指出）\n"
                            "4) 对LLM生成的模型代码/规格的质量给出建议（健壮性、格式、可验证性）\n"
                            "5) 给出总体结论与改进建议\n"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"problem_id：{p.get('problem_id')}\n\n"
                            f"原始题目背景（context）：{p.get('context')}\n\n"
                            f"Tier-1 results（situational_model）：\n{json.dumps(tier1, ensure_ascii=False, indent=2)}\n\n"
                            f"Tier-2 results（real_model）：\n{json.dumps(tier2, ensure_ascii=False, indent=2)}\n\n"
                            f"Tier-3 solver_code_prompt：\n{p.get('solver_code_prompt')}\n\n"
                            f"LLM generated code：\n{tier3.get('llm_code')}\n\n"
                            f"Solver output：\n{json.dumps(solver_out, ensure_ascii=False, indent=2)}\n\n"
                            f"Standard answer：{p.get('standard_answer')}\n"
                            f"Relative error：{tier3.get('relative_error')}\n"
                            f"Passed：{tier3.get('passed')}\n"
                        ),
                    },
                ],
                temperature=0.0,
            )
        except Exception as e:
            report_md = f"# Modeling Process Expert Report\n\nExpert model call failed: `{e!r}`\n"
    else:
        if math_type == "LP":
            model_section = (
                "## Model Formulation (LP)\n"
                "- Variables: A, B (integer)\n"
                "- Objective: maximize A + B\n"
                "- Constraints:\n"
                "  - 50*A + 30*B <= 1000\n"
                "  - 0<=A<=15, 0<=B<=25\n\n"
            )
        elif "POLYNOMIAL" in math_type:
            model_section = (
                "## Model Formulation (Polynomial Regression, Degree 2)\n"
                "- Assume: y = a*x^2 + b*x + c\n"
                "- Fit (a,b,c) by least squares on provided (x,y) data\n"
                "- Predict: final_efficiency = y(x_query)\n\n"
            )
        elif math_type == "ODE":
            model_section = (
                "## Model Formulation (ODE)\n"
                "- Define: dy/dt = f(t, y)\n"
                "- Provide: initial condition y0 and time span t_span\n"
                "- Solve numerically to get y_final\n\n"
            )
        else:
            model_section = "## Model Formulation\n(unsupported math_type in offline template)\n\n"

        report_md = (
            "# Modeling Process Expert Report (Offline)\n\n"
            "当前环境无法调用专家模型（缺少 openai 依赖/密钥/网络），因此生成离线说明。\n\n"
            f"- problem_id: {p.get('problem_id')}\n"
            f"- math_type: {math_type}\n\n"
            "## Notes\n"
            f"- Solver output: {json.dumps(solver_out, ensure_ascii=False)}\n"
            f"- Standard answer: {p.get('standard_answer')}\n"
            f"- Relative error: {tier3.get('relative_error')}\n\n"
            + model_section
            + "如果标准答案与约束/数据不一致，应修正数据或约束。\n"
        )

    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nSaved expert report to: {report_path}")


if __name__ == "__main__":
    main()
