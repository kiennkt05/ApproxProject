# File: methods/least_squares_continuous.py

from typing import List, Dict, Any
import numpy as np
import sympy as sp

def continuous_least_squares(
    f_sym: sp.Expr,
    a: float,
    b: float,
    degree: int
) -> Dict[str, Any]:
    """
    Continuous Least‐Squares Polynomial on [a, b]:
      Find P_n(x) = Σ_{j=0..n} c_j φ_j(x) that minimizes ∫_a^b [f(x) - P(x)]^2 dx,
      where {φ_j(x)} are chosen basis functions (here monomials 1, x, x^2, …).
      
    Steps:
      1. Build Gram matrix G where G[i,j] = ∫_a^b x^{i+j} dx
      2. Build right‐hand side vector b_i = ∫_a^b f(x) x^i dx
      3. Solve G c = b for coefficients c
      4. Symbolic P(x) = Σ c_j x^j
      5. Compute L2 error = ∫_a^b [f(x) - P(x)]^2 dx (if desired)

    OUTPUT:
      {
        "coeffs": [c0..c_degree],
        "polynomial": string of P(x),
        "L2_error": value of ∫(f - P)^2,
        "log": [
          {"step": "G_entry", "i": i, "j": j, "value": G[i][j]},
          {"step": "b_entry", "i": i, "value": b_i},
          {"step": "solve", "coeffs": [...]} 
        ]
      }
    """

    x = sp.symbols('x')
    n = degree

    # Step 1: build Gram matrix
    G = np.zeros((n+1, n+1), dtype=float)
    log: List[Dict[str, Any]] = []
    for i in range(n+1):
        for j in range(n+1):
            integrand = x**(i+j)
            val = sp.integrate(integrand, (x, a, b))
            G[i, j] = float(val)
            log.append({
                "step": "G_entry",
                "i": i,
                "j": j,
                "value": float(val)
            })

    # Step 2: build right‐hand side vector
    b_vec = np.zeros(n+1, dtype=float)
    for i in range(n+1):
        integrand = f_sym * x**i
        val = sp.integrate(integrand, (x, a, b))
        b_vec[i] = float(val)
        log.append({
            "step": "b_entry",
            "i": i,
            "value": float(val)
        })

    # Step 3: solve for coefficients
    c = np.linalg.solve(G, b_vec)
    log.append({"step": "solve", "coeffs": c.tolist()})

    # Step 4: symbolic polynomial
    P_sym = sum(c[j] * x**j for j in range(n+1))
    P_simplified = sp.simplify(P_sym)

    # Step 5: compute L2 error ∫_a^b (f(x) - P(x))^2 dx
    integrand = (f_sym - P_simplified)**2
    L2_err = float(sp.integrate(integrand, (x, a, b)))

    return {
        "coeffs": c.tolist(),
        "polynomial": sp.python(P_simplified),
        "L2_error": L2_err,
        "log": log
    }
