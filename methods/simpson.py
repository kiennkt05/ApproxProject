# File: methods/simpson.py

from typing import Callable, List, Dict, Any, Optional
import numpy as np
import sympy as sp

def composite_simpson(
    f_sym: Optional[sp.Expr],
    f_numeric: Callable[[float], float],
    a: float,
    b: float,
    n: int
) -> Dict[str, Any]:
    """
    Composite Simpson's Rule on [a, b] with n subintervals (n must be even):
      h = (b - a)/n
      S_n = (h/3)[ f(x0) + 4 Σ f(x_{odd}) + 2 Σ f(x_{even, interior}) + f(x_n) ]
    Error bound: ((b-a)/180) h^4 * max |f^{(4)}(ξ)| on [a,b]

    OUTPUT structure:
      {
        "approx": S_n,
        "actual": ∫_a^b f(x) dx (if f_sym given),
        "error": actual - approx,
        "error_bound": ((b-a)/180) h^4 * M,
        "log": [ ... ]
      }
    """
    if n % 2 == 1:
        raise ValueError("Simpson's rule requires n to be even.")

    log: List[Dict[str, Any]] = []
    h = (b - a)/n
    xs = [a + i*h for i in range(n+1)]
    f_vals = [f_numeric(xi) for xi in xs]
    for i, xi in enumerate(xs):
        log.append({"step": "eval_f", "i": i, "x_i": xi, "f(x_i)": f_vals[i]})

    sum_odd = sum(f_vals[i] for i in range(1, n, 2))
    sum_even = sum(f_vals[i] for i in range(2, n-1, 2))
    approx = (h/3) * (f_vals[0] + 4*sum_odd + 2*sum_even + f_vals[-1])
    log.append({
        "step": "simpson_sum",
        "h": h,
        "sum_odd": sum_odd,
        "sum_even": sum_even,
        "approx": approx
    })

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
        f4 = sp.diff(f_sym, x, 4)
        xsamp = np.linspace(a, b, 101)
        f4_func = sp.lambdify(x, f4, 'numpy')
        f4_vals = np.abs(f4_func(xsamp))
        M = float(np.max(f4_vals))
        bound = ((b - a) * h**4 * M) / 180.0
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
