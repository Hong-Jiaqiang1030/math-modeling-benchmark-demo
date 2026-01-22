import math
import multiprocessing as mp
from typing import Any, Dict, Tuple


class SolverExecutionError(RuntimeError):
    pass


def _safe_exec_in_subprocess(code: str, required_vars: Tuple[str, ...], queue: "mp.Queue"):
    # Restrict builtins; disallow imports/open/network by omission.
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "float": float,
        "int": int,
        "dict": dict,
        "list": list,
        "set": set,
        "tuple": tuple,
    }
    g = {"__builtins__": safe_builtins, "math": math}
    l: Dict[str, Any] = {}
    try:
        exec(code, g, l)
        out = {k: l.get(k, g.get(k)) for k in required_vars}
        queue.put(("ok", out))
    except Exception as e:
        queue.put(("err", repr(e)))


def _safe_exec(code: str, required_vars: Tuple[str, ...], timeout_sec: int = 5) -> Dict[str, Any]:
    q: "mp.Queue" = mp.Queue()
    p = mp.Process(target=_safe_exec_in_subprocess, args=(code, required_vars, q))
    p.start()
    p.join(timeout=timeout_sec)
    if p.is_alive():
        p.terminate()
        p.join(1)
        raise SolverExecutionError("Code execution timed out")
    status, payload = q.get() if not q.empty() else ("err", "no result")
    if status != "ok":
        raise SolverExecutionError(f"Code execution failed: {payload}")
    return payload


def solve_lp(lp_code_str: str) -> Dict[str, Any]:
    """
    Execute LLM-produced code (restricted) to obtain lp_spec dict, then solve with PuLP.

    Expected: code defines a variable named `lp_spec` with schema:
    {
      'sense': 'max'|'min',
      'variables': {name: {'low': num, 'up': num|None, 'cat': 'Continuous'|'Integer'|'Binary'}},
      'objective': {name: coef, ...},
      'constraints': [{'lhs': {name: coef, ...}, 'sense': '<='|'>='|'==', 'rhs': num}, ...]
    }
    """
    from pulp import (
        LpProblem,
        LpVariable,
        LpMaximize,
        LpMinimize,
        lpSum,
        PULP_CBC_CMD,
        LpStatus,
        value,
        LpBinary,
        LpInteger,
        LpContinuous,
    )

    out = _safe_exec(lp_code_str, required_vars=("lp_spec",), timeout_sec=5)
    lp_spec = out.get("lp_spec")
    if not isinstance(lp_spec, dict):
        raise SolverExecutionError("lp_spec not found or not a dict")

    sense = str(lp_spec.get("sense", "max")).lower()
    prob = LpProblem("llm_lp", LpMaximize if sense == "max" else LpMinimize)

    var_specs = lp_spec.get("variables", {})
    if not isinstance(var_specs, dict) or not var_specs:
        raise SolverExecutionError("lp_spec.variables missing/invalid")

    def _cat(cat: str):
        c = str(cat or "Continuous").lower()
        if c == "binary":
            return LpBinary
        if c == "integer":
            return LpInteger
        return LpContinuous

    vars_map = {}
    for name, spec in var_specs.items():
        low = spec.get("low", 0)
        up = spec.get("up", None)
        cat = _cat(spec.get("cat", "Continuous"))
        vars_map[name] = LpVariable(name, lowBound=low, upBound=up, cat=cat)

    obj = lp_spec.get("objective", {})
    if not isinstance(obj, dict) or not obj:
        raise SolverExecutionError("lp_spec.objective missing/invalid")
    prob += lpSum(float(c) * vars_map[n] for n, c in obj.items()), "objective"

    constraints = lp_spec.get("constraints", [])
    if not isinstance(constraints, list):
        raise SolverExecutionError("lp_spec.constraints missing/invalid")
    for i, cons in enumerate(constraints):
        lhs = cons.get("lhs", {})
        s = cons.get("sense", "<=")
        rhs = float(cons.get("rhs", 0))
        expr = lpSum(float(c) * vars_map[n] for n, c in lhs.items())
        if s == "<=":
            prob += expr <= rhs, f"c{i}"
        elif s == ">=":
            prob += expr >= rhs, f"c{i}"
        elif s == "==" or s == "=":
            prob += expr == rhs, f"c{i}"
        else:
            raise SolverExecutionError(f"Unsupported constraint sense: {s}")

    prob.solve(PULP_CBC_CMD(msg=False))
    status = LpStatus.get(prob.status, str(prob.status))

    sol = {n: float(value(v)) for n, v in vars_map.items()}
    obj_val = float(value(prob.objective))
    return {"status": status, "objective_value": obj_val, "variables": sol}


def solve_ode(code_str: str) -> Dict[str, Any]:
    """
    Execute restricted code that defines `deriv`, `t_span`, `y0`, then solve with SciPy.

    Expected variables:
    - deriv: callable (t, y) -> dy/dt
    - t_span: (t0, t1)
    - y0: list/tuple of initial values
    """
    from scipy.integrate import solve_ivp

    out = _safe_exec(code_str, required_vars=("deriv", "t_span", "y0"), timeout_sec=5)
    deriv = out.get("deriv")
    t_span = out.get("t_span")
    y0 = out.get("y0")
    if not callable(deriv):
        raise SolverExecutionError("deriv not found/callable")
    if not (isinstance(t_span, (list, tuple)) and len(t_span) == 2):
        raise SolverExecutionError("t_span missing/invalid")
    if not isinstance(y0, (list, tuple)):
        raise SolverExecutionError("y0 missing/invalid")

    sol = solve_ivp(deriv, (float(t_span[0]), float(t_span[1])), list(map(float, y0)), rtol=1e-8, atol=1e-10)
    y_final = float(sol.y[:, -1][0]) if sol.y.size else float("nan")
    return {"status": "ok" if sol.success else "failed", "t_final": float(sol.t[-1]), "y_final": y_final}
