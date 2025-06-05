# File: methods/trapezoidal.py

from typing import Callable, List, Dict, Any, Optional
import numpy as np
import sympy as sp

def composite_trapezoidal(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float], float],
    a: float,
    b: float,
    n: int
) -> Dict[str, Any]:
    """
    Composite Trapezoidal Rule on [a,b] with n subintervals (n+1 equally spaced points):
      h = (b - a)/n
      T_n = (h/2)[ f(x0) + 2 Σ_{i=1..n-1} f(x_i) + f(x_n) ]
    Error bound: (b-a)/12 * h^2 * max |f''(ξ)| on [a,b]

    OUTPUT:
      {
        "approx": T_n,
        "actual": ∫_a^b f(x) dx (if f_sym provided),
        "error": actual - approx,
        "error_bound": ((b-a)/12) h^2 * M,  where M = max |f''(x)| on [a,b],
        "log": [
          {"step": "eval_f", "i": i, "x_i": xi, "f(x_i)": fxi},
          {"step": "sum_terms", "sum": ..., "i": i},
          {"step": "error_bound", "M": M, "bound": bound}
        ]
      }
    """
    log: List[Dict[str, Any]] = []
    h = (b - a)/n
    xs = [a + i*h for i in range(n+1)]
    f_vals = [f_numeric(xi) for xi in xs]
    for i, xi in enumerate(xs):
        log.append({"step": "eval_f", "i": i, "x_i": xi, "f(x_i)": f_vals[i]})
    # Composite trapezoid
    total = f_vals[0] + f_vals[-1] + 2.0 * sum(f_vals[1:-1])
    approx = (h/2.0) * total
    log.append({"step": "trapezoidal_sum", "h": h, "sum": total, "approx": approx})
    
    if f_sym is not None:
        x = sp.symbols('x')
        exact = float(sp.integrate(f_sym, (x, a, b)))
        error = exact - approx
        log.append({"step": "exact_integral", "value": exact})
    else:
        exact = None
        error = None
    
    if f_sym is not None:
        x = sp.symbols('x')
        f2 = sp.diff(f_sym, x, 2)
        # sample to find M = max|f''(x)| on [a,b]
        xsamp = np.linspace(a, b, 101)
        f2_func = sp.lambdify(x, f2, 'numpy')
        f2_vals = np.abs(f2_func(xsamp))
        M = float(np.max(f2_vals))
        bound = ((b - a) * h**2 * M) / 12.0
        log.append({"step": "error_bound", "M": M, "bound": bound})
    else:
        bound = None
    
    return {
        "approx": approx,
        "actual": exact,
        "error": error,
        "error_bound": bound,
        "log": log
    }
