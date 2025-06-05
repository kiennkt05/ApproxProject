# File: methods/five_point.py

from typing import Callable, List, Dict, Any, Optional
import numpy as np
import sympy as sp

def five_point_central(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float], float],
    x0: float,
    h: float
) -> Dict[str, Any]:
    """
    Five‐point central difference for f'(x0):
      f'(x0) ≈ (f(x0 - 2h) - 8f(x0 - h) + 8f(x0 + h) - f(x0 + 2h)) / (12h)
    Error: (h^4 / 30) f^{(5)}(ξ) for some ξ.
    """
    log: List[Dict[str, Any]] = []
    pts = [x0 - 2*h, x0 - h, x0 + h, x0 + 2*h]
    f_vals = [f_numeric(pt) for pt in [x0 - 2*h, x0 - h, x0 + h, x0 + 2*h]]
    for pt, fv in zip(pts, f_vals):
        log.append({"step": "eval_f", "point": pt, "value": fv})
    approx = (f_vals[0] - 8*f_vals[1] + 8*f_vals[2] - f_vals[3]) / (12*h)
    formula = f"(f({x0-2*h}) - 8f({x0-h}) + 8f({x0+h}) - f({x0+2*h}))/(12*{h})"
    log.append({"step": "five_point", "formula": formula, "value": approx})
    
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
        f5 = sp.diff(f_sym, x, 5)
        xs = np.linspace(x0 - 2*h, x0 + 2*h, 41)
        f5_func = sp.lambdify(x, f5, 'numpy')
        f5_vals = np.abs(f5_func(xs))
        M = float(np.max(f5_vals))
        error_bound = (h**4 * M) / 30.0
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

def five_point_endpoint(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float], float],
    x0: float,
    h: float,
    endpoint: str = "left"
) -> Dict[str, Any]:
    """
    Five-point endpoint formula for f'(x0) at a boundary:
      - left endpoint (x0): 
        f'(x0) ≈ [-25f(x0) + 48f(x0 + h) - 36f(x0 + 2h) + 16f(x0 + 3h) - 3f(x0 + 4h)] / (12h)
        Error: (h^4 / 5) f^{(5)}(ξ) for some ξ in [x0, x0+4h]
      - right endpoint (x0): 
        f'(x0) ≈ [25f(x0) - 48f(x0 - h) + 36f(x0 - 2h) - 16f(x0 - 3h) + 3f(x0 - 4h)] / (12h)
        Error: (h^4 / 5) f^{(5)}(ξ) for some ξ in [x0-4h, x0]
    """
    log: List[Dict[str, Any]] = []
    if endpoint == "left":
        pts = [x0, x0 + h, x0 + 2*h, x0 + 3*h, x0 + 4*h]
        f_vals = [f_numeric(pt) for pt in pts]
        for pt, fv in zip(pts, f_vals):
            log.append({"step": "eval_f", "point": pt, "value": fv})
        approx = (-25*f_vals[0] + 48*f_vals[1] - 36*f_vals[2] + 16*f_vals[3] - 3*f_vals[4]) / (12*h)
        formula = f"(-25f({x0}) + 48f({x0+h}) - 36f({x0+2*h}) + 16f({x0+3*h}) - 3f({x0+4*h}))/(12*{h})"
    else:  # right endpoint
        pts = [x0, x0 - h, x0 - 2*h, x0 - 3*h, x0 - 4*h]
        f_vals = [f_numeric(pt) for pt in pts]
        for pt, fv in zip(pts, f_vals):
            log.append({"step": "eval_f", "point": pt, "value": fv})
        approx = (25*f_vals[0] - 48*f_vals[1] + 36*f_vals[2] - 16*f_vals[3] + 3*f_vals[4]) / (12*h)
        formula = f"(25f({x0}) - 48f({x0-h}) + 36f({x0-2*h}) - 16f({x0-3*h}) + 3f({x0-4*h}))/(12*{h})"
    log.append({"step": "five_point_endpoint", "formula": formula, "value": approx})

    if f_sym is not None:
        x = sp.symbols('x')
        fprime = sp.diff(f_sym, x)
        actual = float(fprime.subs(x, x0))
        error = actual - approx
        log.append({"step": "exact_derivative", "f'(x0)": actual})
        f5 = sp.diff(f_sym, x, 5)
        if endpoint == "left":
            xs = np.linspace(x0, x0 + 4*h, 41)
        else:
            xs = np.linspace(x0 - 4*h, x0, 41)
        f5_func = sp.lambdify(x, f5, 'numpy')
        f5_vals = np.abs(f5_func(xs))
        M = float(np.max(f5_vals))
        error_bound = (h**4 * M) / 5.0
        log.append({"step": "error_bound", "M": M, "bound": error_bound})
    else:
        actual = None
        error = None
        error_bound = None

    return {
        "approx": approx,
        "actual": actual,
        "error": error,
        "error_bound": error_bound,
        "log": log
    }
