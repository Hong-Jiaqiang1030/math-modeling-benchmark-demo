import re
from typing import Any, Dict

from demo.benchmark.tools.solver_wrapper import solve_lp, solve_ode, solve_poly_reg_degree2
from demo.benchmark.tools.llm_chat import call_chat, has_credentials, DEFAULT_LLM_MODEL


RELATIVE_ERROR_THRESHOLD = 1e-4  # Mamo-style threshold


def _extract_python_code(text: str) -> str:
    # Prefer fenced code blocks; otherwise return raw text.
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def _relative_error(x: float, y: float) -> float:
    denom = max(1e-12, abs(y))
    return abs((x - y) / denom)


def _mock_lp_code() -> str:
    # A correct deterministic lp_spec for the example problem.
    return (
        "lp_spec = {\n"
        "  'sense': 'max',\n"
        "  'variables': {\n"
        "    'A': {'low': 0, 'up': 15, 'cat': 'Integer'},\n"
        "    'B': {'low': 0, 'up': 25, 'cat': 'Integer'},\n"
        "  },\n"
        "  'objective': {'A': 1, 'B': 1},\n"
        "  'constraints': [\n"
        "    {'lhs': {'A': 50, 'B': 30}, 'sense': '<=', 'rhs': 1000}\n"
        "  ]\n"
        "}\n"
    )


def _mock_poly_reg_deg2_code(x_query: float = 35.0) -> str:
    return (
        "x_vessels = [5, 12, 20, 30, 45]\n"
        "y_eff = [42, 58, 65, 54, 31]\n"
        "coeffs = polyfit2(x_vessels, y_eff)\n"
        f"final_efficiency = polyval2(coeffs, {float(x_query)})\n"
    )


def _extract_x_query_from_prompt(prompt: str, default: float = 35.0) -> float:
    # Prefer the "当...为 XX 艘" pattern; otherwise fallback.
    m = re.search(r"为\\s*(\\d+(?:\\.\\d+)?)\\s*艘", prompt)
    if m:
        return float(m.group(1))
    m = re.search(r"船舶\\s*为\\s*(\\d+(?:\\.\\d+)?)", prompt)
    if m:
        return float(m.group(1))
    return float(default)


def _build_comment_prelude(
    context: str | None,
    situational_results: dict | None,
    real_model_results: dict | None,
) -> str:
    lines = []
    if context:
        lines.append("# [Context]")
        lines.append(f"# {context}")
    if situational_results and situational_results.get("per_question"):
        lines.append("# [Tier-1 Q&A (model answers)]")
        for row in situational_results.get("per_question", []):
            q = (row or {}).get("question")
            a = (row or {}).get("predicted_answer")
            if q and a:
                lines.append(f"# Q: {q}")
                lines.append(f"# A: {a}")
    if real_model_results and real_model_results.get("predicted"):
        lines.append("# [Tier-2 Real Model (predicted)]")
        lines.append(f"# {real_model_results.get('predicted')}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n\n"


def evaluate_mathematical_model(
    solver_code_prompt: str,
    standard_answer: float,
    math_type: str,
    llm_model: str = DEFAULT_LLM_MODEL,
    context: str | None = None,
    situational_results: dict | None = None,
    real_model_results: dict | None = None,
) -> Dict[str, Any]:
    """
    Tier-3 (Math Model) evaluation for closed-ended tasks:
    - Ask LLM to generate solver-ready code/spec
    - Run external solver (SciPy for ODE; PuLP for LP) via wrapper
    - Verify by relative error threshold
    """
    can_call = has_credentials(llm_model)

    math_type_u = str(math_type).upper()
    is_poly2 = ("POLYNOMIAL" in math_type_u) and ("DEGREE 2" in math_type_u or "DEGREE2" in math_type_u or "2" in math_type_u)
    x_query = _extract_x_query_from_prompt(solver_code_prompt, default=35.0)

    if can_call:
        extra = ""
        if context:
            extra += f"\n\n【情景文本】\n{context}\n"
        if situational_results:
            extra += "\n【情景模型(Tier-1)结果：问答】\n"
            try:
                for row in situational_results.get("per_question", []):
                    extra += f"- Q: {row.get('question')}\n  A: {row.get('predicted_answer')}\n"
            except Exception:
                pass
        if real_model_results:
            extra += "\n【真实模型(Tier-2)结果：结构化要素】\n"
            try:
                pred = real_model_results.get("predicted", {})
                extra += f"{pred}\n"
            except Exception:
                pass

        system_msg = (
            "你是数学建模助手。你将基于情景理解与真实模型抽象结果来生成可验证的求解代码。\n"
            "注意：输出必须是可运行的Python代码；可以包含少量注释作为思考过程，但不要输出解释性段落。\n"
        )
        if is_poly2:
            # Our sandboxed executor disallows imports; provide helper functions.
            system_msg += (
                "\n【重要约束】禁止使用 import。\n"
                "你可以直接使用已提供的函数：\n"
                "- coeffs = polyfit2(x_list, y_list)  # 返回 (a,b,c)\n"
                "- yhat = polyval2(coeffs, x)\n"
                f"最终结果必须写入变量 final_efficiency（float），并计算 x={x_query} 时的预测值。\n"
            )

        raw = call_chat(
            llm_model,
            [
                {
                    "role": "system",
                    "content": system_msg,
                },
                {"role": "user", "content": solver_code_prompt + extra},
            ],
        )
        code = _extract_python_code(raw)
    else:
        prelude = _build_comment_prelude(context, situational_results, real_model_results)
        if math_type_u == "LP":
            code = prelude + _mock_lp_code()
        elif is_poly2:
            code = prelude + _mock_poly_reg_deg2_code(x_query=x_query)
        else:
            code = ""

    if math_type_u == "LP":
        solver_out = solve_lp(code)
        value = float(solver_out["objective_value"])
    elif math_type_u == "ODE":
        solver_out = solve_ode(code)
        value = float(solver_out["y_final"])
    elif is_poly2:
        solver_out = solve_poly_reg_degree2(code, x_query=x_query)
        value = float(solver_out["final_efficiency"])
    else:
        # Don't crash the whole demo; return a structured "skipped".
        return {
            "passed": False,
            "solver_output": {"status": "skipped", "reason": f"Unsupported math_type: {math_type}"},
            "relative_error": None,
            "llm_code": code,
            "threshold": RELATIVE_ERROR_THRESHOLD,
        }

    rel_err = _relative_error(value, float(standard_answer))
    passed = rel_err < RELATIVE_ERROR_THRESHOLD

    return {
        "passed": passed,
        "solver_output": solver_out,
        "relative_error": rel_err,
        "llm_code": code,
        "threshold": RELATIVE_ERROR_THRESHOLD,
    }
