# File: methods/three_point.py

from typing import Callable, List, Dict, Any, Optional
import numpy as np
import sympy as sp

def three_point_midpoint(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float], float],
    x0: float,
    h: float
) -> Dict[str, Any]:
    """
    Three‐point midpoint formula for f'(x0):
      f'(x0) ≈ (f(x0 + h) - f(x0 - h)) / (2h)
    Error term: (h^2 / 6) f'''(ξ) for some ξ ∈ [x0 - h, x0 + h]
    OUTPUT dict: same structure as finite_differences
    """
    log: List[Dict[str, Any]] = []
    
    # Evaluate f at x0 ± h
    f_ph = f_numeric(x0 + h)
    f_mh = f_numeric(x0 - h)
    log.append({"step": "eval_f", "point": x0 + h, "value": f_ph})
    log.append({"step": "eval_f", "point": x0 - h, "value": f_mh})
    
    approx = (f_ph - f_mh) / (2*h)
    log.append({
        "step": "three_point_mid",
        "formula": f"(f({x0+h}) - f({x0-h}))/(2*{h})",
        "value": approx
    })
    
    if f_sym is not None:
        x = sp.symbols('x')
        fprime = sp.diff(f_sym, x)
        actual = float(fprime.subs(x, x0))
        error = actual - approx
        log.append({"step": "exact_derivative", "f'(x0)": actual})
    else:
        actual = None
        error = None
    
    if f_sym is not None:
        x = sp.symbols('x')
        f3 = sp.diff(f_sym, x, 3)
        # Estimate max |f'''| over [x0-h, x0+h] by sampling
        xs = np.linspace(x0 - h, x0 + h, 21)
        f3_func = sp.lambdify(x, f3, 'numpy')
        f3_vals = np.abs(f3_func(xs))
        M = float(np.max(f3_vals))
        error_bound = (M * h**2) / 6.0
        log.append({"step": "error_bound", "M": M, "bound": error_bound})
    else:
        error_bound = None
    
    return {
        "approx": approx,
        "actual": actual,
        "error": error,
        "error_bound": error_bound,
        "log": log
    }


def three_point_endpoint(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float], float],
    x0: float,
    h: float,
    endpoint: str = "left"
) -> Dict[str, Any]:
    """
    Three‐point endpoint formula for f'(x0) at a boundary:
      - left endpoint (x0): f'(x0) ≈ (-3f(x0) + 4f(x0 + h) - f(x0 + 2h)) / (2h)
      - right endpoint (x0): f'(x0) ≈ (3f(x0) - 4f(x0 - h) + f(x0 - 2h)) / (2h)
    Error term: (h^2/3) f'''(ξ).
    """
    log: List[Dict[str, Any]] = []
    if endpoint == "left":
        f0 = f_numeric(x0)
        f1 = f_numeric(x0 + h)
        f2 = f_numeric(x0 + 2*h)
        log.append({"step": "eval_f", "point": x0, "value": f0})
        log.append({"step": "eval_f", "point": x0 + h, "value": f1})
        log.append({"step": "eval_f", "point": x0 + 2*h, "value": f2})
        approx = (-3*f0 + 4*f1 - f2) / (2*h)
        formula = f"(-3f({x0}) + 4f({x0+h}) - f({x0+2*h}))/(2*{h})"
    else:  # right endpoint
        f0 = f_numeric(x0)
        f1 = f_numeric(x0 - h)
        f2 = f_numeric(x0 - 2*h)
        log.append({"step": "eval_f", "point": x0, "value": f0})
        log.append({"step": "eval_f", "point": x0 - h, "value": f1})
        log.append({"step": "eval_f", "point": x0 - 2*h, "value": f2})
        approx = (3*f0 - 4*f1 + f2) / (2*h)
        formula = f"(3f({x0}) - 4f({x0-h}) + f({x0-2*h}))/(2*{h})"
    log.append({"step": "three_point_end", "formula": formula, "value": approx})
    
    if f_sym is not None:
        x = sp.symbols('x')
        fprime = sp.diff(f_sym, x)
        actual = float(fprime.subs(x, x0))
        error = actual - approx
        log.append({"step": "exact_derivative", "f'(x0)": actual})
    else:
        actual = None
        error = None
    
    if f_sym is not None:
        x = sp.symbols('x')
        f3 = sp.diff(f_sym, x, 3)
        if endpoint == "left":
            xs = np.linspace(x0, x0 + 2*h, 21)
        else:
            xs = np.linspace(x0 - 2*h, x0, 21)
        f3_func = sp.lambdify(x, f3, 'numpy')
        f3_vals = np.abs(f3_func(xs))
        M = float(np.max(f3_vals))
        error_bound = (M * h**2) / 3.0
        log.append({"step": "error_bound", "M": M, "bound": error_bound})
    else:
        error_bound = None
    
    return {
        "approx": approx,
        "actual": actual,
        "error": error,
        "error_bound": error_bound,
        "log": log
    }
