import os
import re
from typing import Any, Dict, Optional, Tuple

from benchmark.tools.solver_wrapper import solve_lp, solve_ode
from benchmark.utils.env import load_env

# Load .env early so OPENAI_API_KEY is available even when running from `demo/`.
load_env()


RELATIVE_ERROR_THRESHOLD = 1e-4  # Mamo-style threshold


def _call_openai_chat(model: str, messages):
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )
    return (resp.choices[0].message.content or "").strip()


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


def evaluate_mathematical_model(
    solver_code_prompt: str,
    standard_answer: float,
    math_type: str,
    llm_model: str = "gpt-4o",
) -> Dict[str, Any]:
    """
    Tier-3 (Math Model) evaluation for closed-ended tasks:
    - Ask LLM to generate solver-ready code/spec
    - Run external solver (SciPy for ODE; PuLP for LP) via wrapper
    - Verify by relative error threshold
    """
    has_key = bool(os.getenv("OPENAI_API_KEY"))

    if has_key:
        raw = _call_openai_chat(
            llm_model,
            [
                {"role": "system", "content": "你是数学建模助手，只输出可运行的Python代码，不要解释。"},
                {"role": "user", "content": solver_code_prompt},
            ],
        )
        code = _extract_python_code(raw)
    else:
        code = _mock_lp_code() if math_type.upper() == "LP" else ""

    math_type_u = math_type.upper()
    if math_type_u == "LP":
        solver_out = solve_lp(code)
        value = float(solver_out["objective_value"])
    elif math_type_u == "ODE":
        solver_out = solve_ode(code)
        value = float(solver_out["y_final"])
    else:
        raise ValueError(f"Unsupported math_type: {math_type}")

    rel_err = _relative_error(value, float(standard_answer))
    passed = rel_err < RELATIVE_ERROR_THRESHOLD

    return {
        "passed": passed,
        "solver_output": solver_out,
        "relative_error": rel_err,
        "llm_code": code,
        "threshold": RELATIVE_ERROR_THRESHOLD,
    }
