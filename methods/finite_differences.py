# File: methods/finite_differences.py

from typing import Callable, List, Dict, Any, Optional
import numpy as np
import sympy as sp

def forward_difference(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float], float],
    x0: float,
    h: float
) -> Dict[str, Any]:
    """
    Approximates f'(x0) using the forward-difference formula:
      f'(x0) ≈ (f(x0 + h) - f(x0)) / h
    Error term: (h/2) f''(ξ) for some ξ ∈ (x0, x0 + h)
    
    INPUT:
      - f_sym     : sympy expression for f(x), or None if not available
      - f_numeric : Python callable f(x) (must exist)
      - x0        : point at which to approximate derivative
      - h         : step size (h > 0 for forward)
    
    OUTPUT dictionary:
      {
        "approx": float,     # numerical forward difference
        "actual": float,     # if f_sym given, the exact f'(x0)
        "error": float,      # actual - approx
        "error_bound": float,# |f''(ξ)| * h/2, bounded by M*h/2 where M = max |f''(x)| on [x0, x0+h]
        "log": [ ... ]       # step-by-step details
      }
    """
    log: List[Dict[str, Any]] = []
    
    # Step 1: compute f(x0 + h) and f(x0)
    f_x0 = f_numeric(x0)
    f_x0h = f_numeric(x0 + h)
    log.append({
        "step": "eval_f",
        "point": x0,
        "value": f_x0
    })
    log.append({
        "step": "eval_f",
        "point": x0 + h,
        "value": f_x0h
    })
    
    # Step 2: forward difference
    approx = (f_x0h - f_x0) / h
    log.append({
        "step": "forward_diff",
        "formula": f"(f({x0+h}) - f({x0}))/h",
        "value": approx
    })
    
    # Actual derivative f'(x0), if sympy expression provided
    if f_sym is not None:
        x = sp.symbols('x')
        fprime = sp.diff(f_sym, x)
        actual = float(fprime.subs(x, x0))
        error = actual - approx
        log.append({
            "step": "exact_derivative",
            "f'(x0)": actual
        })
    else:
        actual = None
        error = None
    
    # Error bound: need max |f''(ξ)| on [x0, x0+h]
    if f_sym is not None:
        x = sp.symbols('x')
        f2 = sp.diff(f_sym, x, 2)
        # We approximate max|f''(x)| on [x0, x0+h] by sampling or analytic:
        # Here simply sample at 10 equally spaced points (could be refined)
        xs = np.linspace(x0, x0 + h, 11)
        f2_func = sp.lambdify(x, f2, 'numpy')
        f2_vals = np.abs(f2_func(xs))
        M = float(np.max(f2_vals))
        error_bound = (M * h) / 2.0
        log.append({
            "step": "error_bound",
            "M (max |f''|)": M,
            "bound": error_bound
        })
    else:
        error_bound = None
    
    return {
        "approx": approx,
        "actual": actual,
        "error": error,
        "error_bound": error_bound,
        "log": log
    }


def backward_difference(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float], float],
    x0: float,
    h: float
) -> Dict[str, Any]:
    """
    Approximates f'(x0) using the backward‐difference formula:
      f'(x0) ≈ (f(x0) - f(x0 - |h|)) / |h|
    Error term: (|h|/2) f''(ξ) for some ξ ∈ (x0 - |h|, x0)
    
    INPUT:
      - f_sym     : sympy expression for f(x), or None
      - f_numeric : Python callable f(x)
      - x0        : point at which to approximate derivative
      - h         : step size (note: here we expect h < 0 for backward)
    
    OUTPUT dictionary like forward_difference
    """
    log: List[Dict[str, Any]] = []
    h_abs = abs(h)
    
    # Step 1: f(x0) and f(x0 - h_abs)
    f_x0 = f_numeric(x0)
    f_x0mh = f_numeric(x0 - h_abs)
    log.append({"step": "eval_f", "point": x0, "value": f_x0})
    log.append({"step": "eval_f", "point": x0 - h_abs, "value": f_x0mh})
    
    approx = (f_x0 - f_x0mh) / h_abs
    log.append({
        "step": "backward_diff",
        "formula": f"(f({x0}) - f({x0-h_abs}))/|h|",
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
        f2 = sp.diff(f_sym, x, 2)
        xs = np.linspace(x0 - h_abs, x0, 11)
        f2_func = sp.lambdify(x, f2, 'numpy')
        f2_vals = np.abs(f2_func(xs))
        M = float(np.max(f2_vals))
        error_bound = (M * h_abs) / 2.0
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
