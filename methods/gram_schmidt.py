# File: methods/gram_schmidt.py

from typing import List, Dict, Any
import sympy as sp
import numpy as np

def gram_schmidt_polynomials(
    a: float,
    b: float,
    degree: int
) -> Dict[str, Any]:
    """
    Constructs orthonormal polynomials φ_0(x), φ_1(x), ..., φ_degree(x) on [a, b]
    using the Gram–Schmidt process with respect to the inner product 
      ⟨p, q⟩ = ∫_a^b p(x) q(x) dx.

    We start with the monomial basis {1, x, x^2, ..., x^degree}, then orthonormalize.

    OUTPUT:
      {
        "phis": [φ0, φ1, ..., φ_degree]    # sympy expressions, each normalized
        "log": [ ... ]                     # step-by-step projection/re-normalization
      }
    """
    x = sp.symbols('x')
    n = degree
    # Initial monomial basis
    basis = [x**i for i in range(n+1)]
    phis = []      # will hold orthonormal polynomials
    log: List[Dict[str, Any]] = []

    # Inner product function
    def ip(p, q):
        val = sp.integrate(p * q, (x, a, b))
        return float(val)

    for i in range(n+1):
        # Start with p_i = x^i
        p_i = basis[i]
        # Subtract projections onto earlier φ_j
        for j in range(i):
            proj_coeff = ip(p_i, phis[j])  # since φ_j is already normalized
            p_i = sp.simplify(p_i - proj_coeff * phis[j])
            log.append({
                "step": "subtract_projection",
                "i": i,
                "j": j,
                "proj_coeff": proj_coeff
            })
        # Normalize p_i to get φ_i
        norm_sq = ip(p_i, p_i)
        if abs(norm_sq) < 1e-14:
            raise ValueError(f"Zero norm encountered at i={i}")
        norm = np.sqrt(norm_sq)
        phi_i = sp.simplify(p_i / norm)
        phis.append(phi_i)
        log.append({
            "step": "normalize",
            "i": i,
            "norm": norm,
            "phi_i": sp.srepr(phi_i)
        })

    return {
        "phis": [sp.python(phi) for phi in phis],  # Python‐parsable strings
        "log": log
    }
