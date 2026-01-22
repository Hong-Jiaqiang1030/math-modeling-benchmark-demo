import math
import multiprocessing as mp
from typing import Any, Dict, Tuple


class SolverExecutionError(RuntimeError):
    pass


def _solve_linear_system_3x3(a11, a12, a13, a21, a22, a23, a31, a32, a33, b1, b2, b3):
    # Simple Gaussian elimination for 3x3.
    A = [
        [float(a11), float(a12), float(a13), float(b1)],
        [float(a21), float(a22), float(a23), float(b2)],
        [float(a31), float(a32), float(a33), float(b3)],
    ]

    for col in range(3):
        pivot = col
        for r in range(col, 3):
            if abs(A[r][col]) > abs(A[pivot][col]):
                pivot = r
        if abs(A[pivot][col]) < 1e-18:
            raise SolverExecutionError("Singular system in polyfit2")
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]

        div = A[col][col]
        for j in range(col, 4):
            A[col][j] /= div

        for r in range(3):
            if r == col:
                continue
            factor = A[r][col]
            for j in range(col, 4):
                A[r][j] -= factor * A[col][j]

    return A[0][3], A[1][3], A[2][3]


def polyfit2(x_list, y_list):
    """
    Fit y ~= a*x^2 + b*x + c by least squares (degree 2).
    Returns (a, b, c).
    """
    if len(x_list) != len(y_list) or len(x_list) < 3:
        raise SolverExecutionError("polyfit2 requires at least 3 points")

    n = float(len(x_list))
    s_x1 = 0.0
    s_x2 = 0.0
    s_x3 = 0.0
    s_x4 = 0.0
    s_y = 0.0
    s_xy = 0.0
    s_x2y = 0.0

    for x, y in zip(x_list, y_list):
        x = float(x)
        y = float(y)
        x2 = x * x
        s_x1 += x
        s_x2 += x2
        s_x3 += x2 * x
        s_x4 += x2 * x2
        s_y += y
        s_xy += x * y
        s_x2y += x2 * y

    # [sum x^4, sum x^3, sum x^2] [a]   [sum x^2 y]
    # [sum x^3, sum x^2, sum x  ] [b] = [sum x y]
    # [sum x^2, sum x  , n      ] [c]   [sum y]
    a, b, c = _solve_linear_system_3x3(
        s_x4,
        s_x3,
        s_x2,
        s_x3,
        s_x2,
        s_x1,
        s_x2,
        s_x1,
        n,
        s_x2y,
        s_xy,
        s_y,
    )
    return a, b, c


def polyval2(coeffs, x):
    a, b, c = coeffs
    x = float(x)
    return float(a) * x * x + float(b) * x + float(c)


def _solve_lp_bruteforce(lp_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal pure-Python LP/MILP solver for demo purposes.

    Supports only small problems with finite bounds and Integer/Binary variables by enumeration.
    """
    sense = str(lp_spec.get("sense", "max")).lower()
    maximize = sense == "max"

    var_specs = lp_spec.get("variables", {})
    if not isinstance(var_specs, dict) or not var_specs:
        raise SolverExecutionError("lp_spec.variables missing/invalid")

    var_names = list(var_specs.keys())
    bounds = []
    for name in var_names:
        spec = var_specs[name] or {}
        low = spec.get("low", 0)
        up = spec.get("up", None)
        cat = str(spec.get("cat", "Continuous")).lower()
        if cat == "binary":
            low, up = 0, 1
        elif cat == "integer":
            pass
        else:
            raise SolverExecutionError("Bruteforce solver only supports Integer/Binary variables (install pulp for others)")
        if up is None:
            raise SolverExecutionError("Bruteforce solver requires finite upper bounds (install pulp)")
        bounds.append((int(low), int(up)))

    obj = lp_spec.get("objective", {})
    if not isinstance(obj, dict) or not obj:
        raise SolverExecutionError("lp_spec.objective missing/invalid")

    constraints = lp_spec.get("constraints", [])
    if not isinstance(constraints, list):
        raise SolverExecutionError("lp_spec.constraints missing/invalid")

    best_val = -float("inf") if maximize else float("inf")
    best_sol: Dict[str, float] = {}
    found = False
    total_assignments = 1
    for lo, hi in bounds:
        total_assignments *= max(0, (hi - lo + 1))
    feasible_count = 0

    def feasible(assign: Dict[str, int]) -> bool:
        for cons in constraints:
            lhs = cons.get("lhs", {}) or {}
            s = cons.get("sense", "<=")
            rhs = float(cons.get("rhs", 0))
            total = 0.0
            for n, c in lhs.items():
                total += float(c) * float(assign.get(n, 0))
            if s == "<=":
                if total > rhs + 1e-12:
                    return False
            elif s == ">=":
                if total < rhs - 1e-12:
                    return False
            elif s == "==" or s == "=":
                if abs(total - rhs) > 1e-12:
                    return False
            else:
                raise SolverExecutionError(f"Unsupported constraint sense: {s}")
        return True

    def obj_value(assign: Dict[str, int]) -> float:
        return sum(float(c) * float(assign.get(n, 0)) for n, c in obj.items())

    # Enumerate all combinations (cartesian product of integer ranges).
    def rec(i: int, cur: Dict[str, int]):
        nonlocal best_val, best_sol, found, feasible_count
        if i == len(var_names):
            if not feasible(cur):
                return
            feasible_count += 1
            val = obj_value(cur)
            if (maximize and val > best_val) or ((not maximize) and val < best_val):
                best_val = val
                best_sol = {k: float(v) for k, v in cur.items()}
                found = True
            return
        name = var_names[i]
        lo, hi = bounds[i]
        for v in range(lo, hi + 1):
            cur[name] = v
            rec(i + 1, cur)
        cur.pop(name, None)

    rec(0, {})
    if not found:
        return {
            "status": "infeasible",
            "objective_value": float("nan"),
            "variables": {},
            "lp_spec": lp_spec,
            "trace": {
                "method": "bruteforce",
                "total_assignments": int(total_assignments),
                "feasible_count": int(feasible_count),
            },
        }
    cons_eval = []
    for i, cons in enumerate(constraints):
        lhs = cons.get("lhs", {}) or {}
        s = cons.get("sense", "<=")
        rhs = float(cons.get("rhs", 0))
        lhs_val = 0.0
        for n, c in lhs.items():
            lhs_val += float(c) * float(best_sol.get(n, 0.0))
        cons_eval.append(
            {
                "name": f"c{i}",
                "lhs_value": lhs_val,
                "sense": s,
                "rhs": rhs,
                "satisfied": (
                    (s == "<=" and lhs_val <= rhs + 1e-8)
                    or (s == ">=" and lhs_val >= rhs - 1e-8)
                    or (s in ("==", "=") and abs(lhs_val - rhs) <= 1e-8)
                ),
            }
        )
    return {
        "status": "optimal",
        "objective_value": float(best_val),
        "variables": best_sol,
        "lp_spec": lp_spec,
        "trace": {
            "method": "bruteforce",
            "total_assignments": int(total_assignments),
            "feasible_count": int(feasible_count),
        },
        "constraints_eval": cons_eval,
    }


def _safe_exec_in_subprocess(
    code: str, required_vars: Tuple[str, ...], queue: "mp.Queue", extra_globals: Dict[str, Any] | None = None
):
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
    if extra_globals:
        g.update(extra_globals)
    l: Dict[str, Any] = {}
    try:
        exec(code, g, l)
        out = {k: l.get(k, g.get(k)) for k in required_vars}
        queue.put(("ok", out))
    except Exception as e:
        queue.put(("err", repr(e)))


def _safe_exec(
    code: str,
    required_vars: Tuple[str, ...],
    timeout_sec: int = 5,
    extra_globals: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    q: "mp.Queue" = mp.Queue()
    p = mp.Process(target=_safe_exec_in_subprocess, args=(code, required_vars, q, extra_globals))
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
    out = _safe_exec(lp_code_str, required_vars=("lp_spec",), timeout_sec=5)
    lp_spec = out.get("lp_spec")
    if not isinstance(lp_spec, dict):
        raise SolverExecutionError("lp_spec not found or not a dict")

    try:
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
    except Exception:
        # Keep the demo runnable even without third-party deps.
        return _solve_lp_bruteforce(lp_spec)

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

    # Also return a small solve trace and constraint evaluations for transparency.
    cons_eval = []
    for i, cons in enumerate(constraints):
        lhs = cons.get("lhs", {}) or {}
        s = cons.get("sense", "<=")
        rhs = float(cons.get("rhs", 0))
        lhs_val = 0.0
        for n, c in lhs.items():
            lhs_val += float(c) * float(sol.get(n, 0.0))
        cons_eval.append(
            {
                "name": f"c{i}",
                "lhs_value": lhs_val,
                "sense": s,
                "rhs": rhs,
                "satisfied": (
                    (s == "<=" and lhs_val <= rhs + 1e-8)
                    or (s == ">=" and lhs_val >= rhs - 1e-8)
                    or (s in ("==", "=") and abs(lhs_val - rhs) <= 1e-8)
                ),
            }
        )

    return {
        "status": status,
        "objective_value": obj_val,
        "variables": sol,
        "lp_spec": lp_spec,
        "trace": {"method": "pulp", "status": status},
        "constraints_eval": cons_eval,
    }


def solve_ode(code_str: str) -> Dict[str, Any]:
    """
    Execute restricted code that defines `deriv`, `t_span`, `y0`, then solve with SciPy.

    Expected variables:
    - deriv: callable (t, y) -> dy/dt
    - t_span: (t0, t1)
    - y0: list/tuple of initial values
    """
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

    t0, t1 = float(t_span[0]), float(t_span[1])
    y = list(map(float, y0))

    try:
        from scipy.integrate import solve_ivp

        sol = solve_ivp(deriv, (t0, t1), y, rtol=1e-8, atol=1e-10)
        y_final = float(sol.y[:, -1][0]) if sol.y.size else float("nan")
        return {"status": "ok" if sol.success else "failed", "t_final": float(sol.t[-1]), "y_final": y_final}
    except Exception:
        # Pure-Python fallback: simple explicit Euler for 1D/low-dim ODEs.
        steps = 2000
        dt = (t1 - t0) / float(steps)
        t = t0
        for _ in range(steps):
            dy = deriv(t, y)
            if isinstance(dy, (list, tuple)):
                dy_list = list(dy)
            else:
                dy_list = [float(dy)]
            for i in range(min(len(y), len(dy_list))):
                y[i] = float(y[i]) + dt * float(dy_list[i])
            t += dt
        return {"status": "ok", "t_final": t1, "y_final": float(y[0]) if y else float("nan")}


def solve_poly_reg_degree2(code_str: str, x_query: float) -> Dict[str, Any]:
    """
    Execute restricted code that defines `final_efficiency` (float).
    We provide helpers polyfit2(x_list, y_list) and polyval2((a,b,c), x).
    """
    out = _safe_exec(
        code_str,
        required_vars=("final_efficiency",),
        timeout_sec=5,
        extra_globals={"polyfit2": polyfit2, "polyval2": polyval2},
    )
    val = out.get("final_efficiency")
    try:
        yhat = float(val)
    except Exception:
        raise SolverExecutionError("final_efficiency missing/invalid")
    return {
        "status": "ok",
        "final_efficiency": yhat,
        "x_query": float(x_query),
        "trace": {"method": "poly_reg_deg2"},
    }
