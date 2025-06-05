# File: methods/chebyshev_approx.py

from typing import List, Dict, Any
import numpy as np
import sympy as sp

def chebyshev_approximation(
    f_sym: sp.Expr,
    a: float,
    b: float,
    degree: int,
    x_eval: float,
    use_nodes: bool = True
) -> Dict[str, Any]:
    """
    Approximates f(x) on [a,b] by a Chebyshev polynomial of given degree.
    Two modes:
      1. use_nodes=True: sample f at Chebyshev nodes {x_k = (a+b)/2 + (b-a)/2 * cos((2k+1)/(2n+2)*π)},
         then do discrete least‐squares (or interpolation) on those n+1 points.
      2. use_nodes=False: compute Chebyshev series coefficients by projecting f onto T_k(x')
         where x' = (2x - (b+a))/(b-a) maps [a,b]→[-1,1].
    
    We illustrate the **interpolation at Chebyshev nodes** approach:
      1. Build Chebyshev nodes in [a,b].
      2. Evaluate f at these nodes.
      3. Fit an interpolating polynomial (degree n) through those points (which is known to minimize Runge phenomenon).
      4. Evaluate at x_eval.
    
    OUTPUT:
      {
        "nodes": [x_0..x_n],
        "f_nodes": [f(x_0)..f(x_n)],
        "coeffs": [c0..c_n],     # of the resulting interpolating polynomial
        "polynomial": string P(x),
        "value": P(x_eval),
        "log": [
          {"step": "cheb_node", "k": k, "x_k": xk},
          {"step": "eval_f", "point": xk, "value": f_k},
          {"step": "interpolate", "coeffs": [...]}
        ]
      }
    """
    x = sp.symbols('x')
    n = degree

    # Step 1: Chebyshev nodes in [-1,1]
    nodes_ref = [np.cos((2*k + 1) * np.pi / (2*(n+1))) for k in range(n+1)]
    # Map to [a,b]: x_k = (a+b)/2 + (b-a)/2 * t_k
    nodes = [ (a+b)/2 + (b-a)/2 * t for t in nodes_ref ]
    log: List[Dict[str, Any]] = []
    for k, xk in enumerate(nodes):
        log.append({"step": "cheb_node", "k": k, "x_k": xk})

    # Step 2: evaluate f at nodes
    f_nodes = [float(f_sym.subs(x, xk)) for xk in nodes]
    for k, fk in enumerate(f_nodes):
        log.append({"step": "eval_f", "point": nodes[k], "value": fk})

    # Step 3: build Vandermonde on these nodes and solve for interpolating polynomial
    # (In fact, interpolation on Chebyshev nodes is well‐conditioned)
    A = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(n+1):
            A[i, j] = nodes[i]**j
    y_vec = np.array(f_nodes)
    coeffs = np.linalg.solve(A, y_vec)
    log.append({"step": "interpolate", "coeffs": coeffs.tolist()})

    # Build symbolic P(x)
    P_sym = sum(coeffs[j] * x**j for j in range(n+1))
    P_simp = sp.simplify(P_sym)
    P_str = sp.python(P_simp)

    # Evaluate at x_eval
    P_val = float(P_simp.subs(x, x_eval))

    return {
        "nodes": nodes,
        "f_nodes": f_nodes,
        "coeffs": coeffs.tolist(),
        "polynomial": P_str,
        "value": P_val,
        "log": log
    }
