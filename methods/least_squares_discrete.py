# File: methods/least_squares_discrete.py

from typing import List, Dict, Any
import numpy as np
import sympy as sp

def discrete_least_squares(
    x_data: List[float],
    y_data: List[float],
    degree: int
) -> Dict[str, Any]:
    """
    Discrete (data‐based) Least‐Squares: fit a polynomial P_n(x) of given degree
    to points (x_i, y_i) by minimizing Σ [y_i - P(x_i)]^2.

    Steps:
      1. Assemble Vandermonde‐like matrix A where A[i][j] = (x_i)^j, j=0..degree.
      2. Solve normal equations (A^T A) c = A^T y for coefficients c[0..degree].
      3. Build symbolic polynomial P(x) = Σ c_j x^j.
      4. Compute residuals, SSE, etc.

    OUTPUT:
      {
        "coeffs": [c0..c_degree],
        "polynomial": string of P(x),
        "residuals": [...],            # y_i - P(x_i)
        "SSE": float,                  # sum of squared errors
        "log": [                        # matrices and steps
          {"step": "Vandermonde", "A_row": ..., "i": i},
          {"step": "normal_matrix", "ATA": ..., "ATy": ...},
          {"step": "solve", "c": [...]} 
        ]
      }
    """

    n = len(x_data)
    m = degree + 1
    log: List[Dict[str, Any]] = []

    # Step 1: build design matrix A (n×(degree+1))
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            A[i, j] = x_data[i]**j
        log.append({
            "step": "vandermonde_row",
            "i": i,
            "A_row": A[i, :].tolist()
        })

    y_vec = np.array(y_data)

    # Step 2: form normal equations
    ATA = A.T @ A
    ATy = A.T @ y_vec
    log.append({"step": "normal_matrix", "ATA": ATA.tolist(), "ATy": ATy.tolist()})

    # Solve for coefficients c
    c = np.linalg.solve(ATA, ATy)
    log.append({"step": "solve", "coeffs": c.tolist()})

    # Build symbolic polynomial
    x = sp.symbols('x')
    P_sym = sum(c[j] * x**j for j in range(m))
    P_sym_simplified = sp.simplify(P_sym)

    # Compute residuals and SSE
    residuals = []
    for i in range(n):
        Pi = float(P_sym_simplified.subs(x, x_data[i]))
        residuals.append(y_data[i] - Pi)
    SSE = float(np.sum(np.array(residuals)**2))

    return {
        "coeffs": c.tolist(),
        "polynomial": sp.python(P_sym_simplified),
        "residuals": residuals,
        "SSE": SSE,
        "log": log
    }
