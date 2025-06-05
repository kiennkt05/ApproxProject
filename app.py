# File: app.py

import streamlit as st
import pandas as pd
import numpy as np
import sympy as sp

from methods.finite_differences       import forward_difference, backward_difference
from methods.three_point              import three_point_midpoint, three_point_endpoint
from methods.five_point               import five_point_central, five_point_endpoint
from methods.least_squares_discrete   import discrete_least_squares
from methods.least_squares_continuous  import continuous_least_squares
from methods.gram_schmidt             import gram_schmidt_polynomials
from methods.chebyshev_approx         import chebyshev_approximation
from methods.trapezoidal              import composite_trapezoidal
from methods.simpson                  import composite_simpson

st.set_page_config(page_title="Numerical Approximation Package", layout="wide")
st.title("🔍 Numerical Approximation Toolbox")

st.markdown(
    r"""
    This app implements **nine classical numerical‐approximation routines**.  Choose one from the sidebar, provide
    the necessary inputs (data points or symbolic \(f(x)\), interval \([a,b]\), step size \(h\), polynomial degree, etc.),
    and hit **Run**.  You will see a detailed step‐by‐step log, the numerical approximation, the actual error
    (when an "exact" formula is available), and the theoretical error bound from the remainder‐term formula.
    """
)

# --- Sidebar: select method ---
method = st.sidebar.selectbox(
    "Choose a method:",
    [
        "Forward‐Difference (h > 0)",
        "Backward‐Difference (h < 0)",
        "Three‐Point Formula (Midpoint/Endpoint)",
        "Five‐Point Central Formula",
        "Discrete Least‐Squares Fit",
        "Continuous Least‐Squares Fit",
        "Gram–Schmidt Orthonormal Polynomials",
        "Chebyshev Polynomial Approximation",
        "Composite Trapezoidal Rule",
        "Composite Simpson's Rule"
    ]
)

st.header(f"🔹 {method}")

# --- Helper to parse a symbolic function safely ---
@st.cache_data
def parse_symbolic(expr: str):
    """
    Given a string like 'sin(x) + x**2', return a sympy expression and error.
    """
    x = sp.symbols('x')
    try:
        fsym = sp.sympify(expr)
        return fsym, None
    except Exception as e:
        return None, str(e)

def parse_function(expr: str):
    """
    Given a string like 'sin(x) + x**2', return a sympy expression and a numeric lambda.
    """
    fsym, err = parse_symbolic(expr)
    if err:
        return None, None, err
    try:
        x = sp.symbols('x')
        fnum = sp.lambdify(x, fsym, 'numpy')
        return fsym, fnum, None
    except Exception as e:
        return None, None, str(e)

# Add this helper function after the parse_function definition
def format_log_entry(entry: dict) -> str:
    """
    Format a log entry with LaTeX representation.
    """
    step = entry.get("step", "")

    if step == "eval_f":
        if "point" in entry and "value" in entry:
            return f"Evaluate $f({entry['point']:.6f}) = {entry['value']:.6f}$"
        elif "x_i" in entry and "f(x_i)" in entry:
            return f"Evaluate $f({entry['x_i']:.6f}) = {entry['f(x_i)']:.6f}$"
        else:
            return "Function evaluation step (missing data)"
    elif step == "forward_diff":
        return f"Forward difference: ${entry.get('formula', '')} = {entry.get('value', ''):.6f}$"
    elif step == "backward_diff":
        return f"Backward difference: ${entry.get('formula', '')} = {entry.get('value', ''):.6f}$"
    elif step == "three_point_mid":
        return f"Three-point midpoint: ${entry.get('formula', '')} = {entry.get('value', ''):.6f}$"
    elif step == "three_point_end":
        return f"Three-point endpoint: ${entry.get('formula', '')} = {entry.get('value', ''):.6f}$"
    elif step == "five_point":
        return f"Five-point central: ${entry.get('formula', '')} = {entry.get('value', ''):.6f}$"
    elif step == "five_point_endpoint":
        return f"Five-point endpoint: ${entry.get('formula', '')} = {entry.get('value', ''):.6f}$"
    elif step == "exact_derivative":
        fprime_val = entry.get("f'(x0)", '')
        return f"Exact derivative $f'({entry.get('x0', 'x_0')}) = {fprime_val:.6f}$"
    elif step == "error_bound":
        if "M (max |f''|)" in entry:
            return f"Error bound: $M = {entry['M (max |f\'\'|)']:.6f}$, bound $= {entry['bound']:.6f}$"
        elif "M" in entry:
            return f"Error bound: $M = {entry['M']:.6f}$, bound $= {entry['bound']:.6f}$"
        else:
            return f"Error bound: bound $= {entry.get('bound', ''):.6f}$"
    elif step == "vandermonde_row":
        return f"Vandermonde row {entry.get('i', '')}: $A_{{{entry.get('i', '')}}} = {entry.get('A_row', '')}$"
    elif step == "normal_matrix":
        return f"Normal equations: $A^T A = {entry.get('ATA', '')}$, $A^T y = {entry.get('ATy', '')}$"
    elif step == "solve":
        return f"Solve for coefficients: $c = {entry.get('coeffs', '')}$"
    elif step == "cheb_node":
        return f"Chebyshev node {entry.get('k', '')}: $x_{{{entry.get('k', '')}}} = {entry.get('x_k', ''):.6f}$"
    elif step == "interpolate":
        return f"Interpolating polynomial coefficients: $c = {entry.get('coeffs', '')}$"
    elif step == "trapezoidal_sum":
        return f"Trapezoidal sum: $h = {entry.get('h', ''):.6f}$, $\\sum = {entry.get('sum', ''):.6f}$, approx $= {entry.get('approx', ''):.6f}$"
    elif step == "simpson_sum":
        return f"Simpson's sum: $h = {entry.get('h', ''):.6f}$, $\\sum_{{\\text{{odd}}}} = {entry.get('sum_odd', ''):.6f}$, $\\sum_{{\\text{{even}}}} = {entry.get('sum_even', ''):.6f}$, approx $= {entry.get('approx', ''):.6f}$"
    elif step == "exact_integral":
        return f"Exact integral $= {entry.get('value', ''):.6f}$"
    elif step == "G_entry":
        return f"Gram matrix entry $G_{{{entry.get('i', '')},{entry.get('j', '')}}} = {entry.get('value', ''):.6f}$"
    elif step == "b_entry":
        return f"Right-hand side entry $b_{{{entry.get('i', '')}}} = {entry.get('value', ''):.6f}$"
    elif step == "normalize":
        i = entry.get("i", "")
        norm = entry.get("norm", "")
        phi_i = entry.get("phi_i", "")
        try:
            norm_fmt = f"{float(norm):.6f}"
        except Exception:
            norm_fmt = str(norm)
        return f"Normalize $\\varphi_{{{i}}}(x)$: $\\|p_{i}\\| = {norm_fmt}$, $\\varphi_{{{i}}}(x) = $ `{phi_i}`"
    elif step == "subtract_projection":
        i = entry.get("i", "")
        j = entry.get("j", "")
        proj_coeff = entry.get("proj_coeff", "")
        try:
            proj_coeff_fmt = f"{float(proj_coeff):.6f}"
        except Exception:
            proj_coeff_fmt = str(proj_coeff)
        return f"Subtract projection: $p_{{{i}}} \\gets p_{{{i}}} - ({proj_coeff_fmt})\\varphi_{{{j}}}(x)$"
    else:
        # Fallback for unknown or incomplete log entries
        return str(entry)

# --- 1) Finite‐Difference Methods require data points (xi, yi) AND h, x0 ---
if method in ["Forward‐Difference (h > 0)", "Backward‐Difference (h < 0)",
              "Three‐Point Formula (Midpoint/Endpoint)", "Five‐Point Central Formula"]:
    st.subheader("Enter data points (xi, yi)")
    st.info("Upload a CSV with two columns (x,y), or paste lines `x, y`.")
    uploaded = st.file_uploader("Upload CSV of points", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, header=None)
        df.columns = ["x", "y"]
        x_data = df["x"].astype(float).tolist()
        y_data = df["y"].astype(float).tolist()
    else:
        text = st.text_area(
            "Or paste data lines (each `x, y`):",
            "0.0, 0.0\n0.1, 0.09983\n0.2, 0.19867\n0.3, 0.29552"
        )
        try:
            pairs = [line.split(",") for line in text.strip().split("\n")]
            x_data = [float(p[0]) for p in pairs]
            y_data = [float(p[1]) for p in pairs]
        except:
            st.error("Could not parse data points.")
            st.stop()
    # Sort by x
    sorted_indices = sorted(range(len(x_data)), key=lambda i: x_data[i])
    x_data = [x_data[i] for i in sorted_indices]
    y_data = [y_data[i] for i in sorted_indices]
    st.write("Data preview (sorted):")
    st.dataframe(pd.DataFrame({"x": x_data, "y": y_data}))

    # Ask user if they have an analytic f(x) (for "actual" derivative)
    st.write("If you know the analytical f(x), you may enter it below (for actual error).  Otherwise leave blank.")
    f_expr = st.text_input("Enter f(x) in Python syntax (e.g. 'sin(x)' or 'x**2 + 3'):", "")
    if f_expr.strip():
        fsym, fnum, err = parse_function(f_expr)
        if err:
            st.error(f"Could not parse function: {err}")
            st.stop()
    else:
        fsym = None
        fnum = lambda xx: np.interp(xx, x_data, y_data)

    # Common inputs: x0, h
    x0 = st.number_input("Enter x0 (point of derivative):", value=float(x_data[0]))
    h = st.number_input("Enter h (step size):", value=0.1)

    if method == "Forward‐Difference (h > 0)":
        run_label = "Run Forward‐Difference"
    elif method == "Backward‐Difference (h < 0)":
        run_label = "Run Backward‐Difference"
    elif method == "Three‐Point Formula (Midpoint/Endpoint)":
        run_label = "Run All Three‐Point Approaches"
    else:
        run_label = "Run Five‐Point"

    if st.button(run_label):
        if method == "Forward‐Difference (h > 0)":
            res = forward_difference(fsym, fnum, x0, h)
            st.subheader("Approximation & Errors")
            st.write(f"**Forward difference** ≈ {res['approx']:.8f}")
            if res["actual"] is not None:
                st.write(f"Actual f'({x0}) = {res['actual']:.8f}")
                st.write(f"Error = {res['error']:.8f}")
                st.write(f"Error bound ≤ {res['error_bound']:.8f}")
            st.subheader("Step‐by‐Step Log")
            for entry in res["log"]:
                st.markdown(f"$\\bullet$ {format_log_entry(entry)}")

        elif method == "Backward‐Difference (h < 0)":
            res = backward_difference(fsym, fnum, x0, h)
            st.subheader("Approximation & Errors")
            st.write(f"**Backward difference** ≈ {res['approx']:.8f}")
            if res["actual"] is not None:
                st.write(f"Actual f'({x0}) = {res['actual']:.8f}")
                st.write(f"Error = {res['error']:.8f}")
                st.write(f"Error bound ≤ {res['error_bound']:.8f}")
            st.subheader("Step‐by‐Step Log")
            for entry in res["log"]:
                st.markdown(f"$\\bullet$ {format_log_entry(entry)}")

        elif method == "Three‐Point Formula (Midpoint/Endpoint)":
            # Run all three approaches
            st.subheader("Three‐Point Midpoint Approximation")
            res_mid = three_point_midpoint(fsym, fnum, x0, h)
            st.write(f"Approx $f'({x0}) \\approx$ {res_mid['approx']:.8f}")
            if res_mid["actual"] is not None:
                st.write(f"Actual $f'({x0}) = {res_mid['actual']:.8f}$")
                st.write(f"Error = {res_mid['error']:.8f}")
                st.write(f"Error bound $\\leq$ {res_mid['error_bound']:.8f}")
            st.subheader("Step‐by‐Step Log (Midpoint)")
            for entry in res_mid["log"]:
                st.markdown(f"$\\bullet$ {format_log_entry(entry)}")

            st.subheader("Three‐Point Endpoint (Left) Approximation")
            res_left = three_point_endpoint(fsym, fnum, x0, h, "left")
            st.write(f"Approx $f'({x0}) \\approx$ {res_left['approx']:.8f}")
            if res_left["actual"] is not None:
                st.write(f"Actual $f'({x0}) = {res_left['actual']:.8f}$")
                st.write(f"Error = {res_left['error']:.8f}")
                st.write(f"Error bound $\\leq$ {res_left['error_bound']:.8f}")
            st.subheader("Step‐by‐Step Log (Endpoint Left)")
            for entry in res_left["log"]:
                st.markdown(f"$\\bullet$ {format_log_entry(entry)}")

            st.subheader("Three‐Point Endpoint (Right) Approximation")
            res_right = three_point_endpoint(fsym, fnum, x0, h, "right")
            st.write(f"Approx $f'({x0}) \\approx$ {res_right['approx']:.8f}")
            if res_right["actual"] is not None:
                st.write(f"Actual $f'({x0}) = {res_right['actual']:.8f}$")
                st.write(f"Error = {res_right['error']:.8f}")
                st.write(f"Error bound $\\leq$ {res_right['error_bound']:.8f}")
            st.subheader("Step‐by‐Step Log (Endpoint Right)")
            for entry in res_right["log"]:
                st.markdown(f"$\\bullet$ {format_log_entry(entry)}")

        else:  # Five‐point
            st.subheader("Five‐Point Central (Midpoint) Approximation")
            res_central = five_point_central(fsym, fnum, x0, h)
            st.write(f"Approx $f'({x0}) \\approx$ {res_central['approx']:.8f}")
            if res_central["actual"] is not None:
                st.write(f"Actual $f'({x0}) = {res_central['actual']:.8f}$")
                st.write(f"Error = {res_central['error']:.8f}")
                st.write(f"Error bound $\\leq$ {res_central['error_bound']:.8f}")
            st.subheader("Step‐by‐Step Log (Central)")
            for entry in res_central["log"]:
                st.markdown(f"$\\bullet$ {format_log_entry(entry)}")

            st.subheader("Five‐Point Endpoint (Left) Approximation")
            res_left = five_point_endpoint(fsym, fnum, x0, h, "left")
            st.write(f"Approx $f'({x0}) \\approx$ {res_left['approx']:.8f}")
            if res_left["actual"] is not None:
                st.write(f"Actual $f'({x0}) = {res_left['actual']:.8f}$")
                st.write(f"Error = {res_left['error']:.8f}")
                st.write(f"Error bound $\\leq$ {res_left['error_bound']:.8f}")
            st.subheader("Step‐by‐Step Log (Endpoint Left)")
            for entry in res_left["log"]:
                st.markdown(f"$\\bullet$ {format_log_entry(entry)}")

            st.subheader("Five‐Point Endpoint (Right) Approximation")
            res_right = five_point_endpoint(fsym, fnum, x0, h, "right")
            st.write(f"Approx $f'({x0}) \\approx$ {res_right['approx']:.8f}")
            if res_right["actual"] is not None:
                st.write(f"Actual $f'({x0}) = {res_right['actual']:.8f}$")
                st.write(f"Error = {res_right['error']:.8f}")
                st.write(f"Error bound $\\leq$ {res_right['error_bound']:.8f}")
            st.subheader("Step‐by‐Step Log (Endpoint Right)")
            for entry in res_right["log"]:
                st.markdown(f"$\\bullet$ {format_log_entry(entry)}")


# --- 2) Discrete Least‐Squares Fit (data → polynomial) ---

elif method == "Discrete Least‐Squares Fit":
    st.subheader("Enter data points (xi, yi)")
    st.info("Upload a CSV (x,y) or paste lines `x, y`.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded, header=None)
        df.columns = ["x", "y"]
        x_data = df["x"].astype(float).tolist()
        y_data = df["y"].astype(float).tolist()
    else:
        text = st.text_area(
            "Or paste data lines:",
            "0, 1\n1, 2\n2, 1.3\n3, 3.75\n4, 2.5"
        )
        try:
            pairs = [line.split(",") for line in text.strip().split("\n")]
            x_data = [float(p[0]) for p in pairs]
            y_data = [float(p[1]) for p in pairs]
        except:
            st.error("Could not parse data.")
            st.stop()
    st.write("Data preview:")
    st.dataframe(pd.DataFrame({"x": x_data, "y": y_data}))

    degree = int(st.number_input("Enter polynomial degree (n):", min_value=0, value=1))
    if st.button("Run Discrete Least‐Squares"):
        res = discrete_least_squares(x_data, y_data, degree)
        st.subheader("Fitted Polynomial P(x)")
        st.code(res["polynomial"], language="python")
        st.subheader("Coefficients [c0..c_n]")
        st.write(res["coeffs"])
        st.subheader("Residuals y_i - P(x_i)")
        st.write(res["residuals"])
        st.subheader(f"Sum of Squared Errors (SSE) = {res['SSE']:.6e}")
        st.subheader("Step‐by‐Step Log")
        for entry in res["log"]:
            st.markdown(f"$\\bullet$ {format_log_entry(entry)}")


# --- 3) Continuous Least‐Squares Fit (symbolic f) ---

elif method == "Continuous Least‐Squares Fit":
    st.subheader("Enter f(x) symbolically (for ∫_a^b )")
    f_expr = st.text_input("f(x) =", "sin(x)")
    fsym, fnum, err = parse_function(f_expr)
    if err:
        st.error(f"Could not parse f(x): {err}")
        st.stop()

    a = st.number_input("Left endpoint a:", value=0.0, format="%f")
    b = st.number_input("Right endpoint b:", value=np.pi, format="%f")
    degree = int(st.number_input("Enter polynomial degree (n):", min_value=0, value=1))
    if st.button("Run Continuous Least‐Squares"):
        res = continuous_least_squares(fsym, a, b, degree)
        st.subheader("Least‐Squares Polynomial $P(x)$")
        st.code(res["polynomial"], language="python")
        
        # --- LaTeX for coefficients ---
        st.subheader("Coefficients $[c_0, \\ldots, c_n]$")
        coeffs = res["coeffs"]
        coeffs_latex = ",\\;".join([f"{c:.6g}" for c in coeffs])
        st.latex(f"\\vec{{c}} = \\begin{{bmatrix}} {coeffs_latex} \\end{{bmatrix}}")
        
        # --- LaTeX for L2 error ---
        st.subheader("L2 Error")
        st.latex(r"L_2\ \text{error} = \int_a^b [f(x) - P(x)]^2\,dx\\")
        st.write(f"$= {res['L2_error']:.6e}$")
        
        st.subheader("Step‐by‐Step Log")
        for entry in res["log"]:
            st.markdown(f"$\\bullet$ {format_log_entry(entry)}")


# --- 4) Gram–Schmidt Orthonormal Polynomials ---

elif method == "Gram–Schmidt Orthonormal Polynomials":
    st.subheader("Construct φ0, φ1, φ2, φ3 on [a,b]")
    a = st.number_input("Left endpoint a:", value=-1.0, format="%f")
    b = st.number_input("Right endpoint b:", value=1.0, format="%f")
    degree = int(st.number_input("Enter max degree (≤3 for φ0..φ3):", min_value=0, max_value=10, value=3))
    if st.button("Run Gram–Schmidt"):
        res = gram_schmidt_polynomials(a, b, degree)
        st.subheader("Orthonormal Polynomials φ_i(x)")
        for i, phi_str in enumerate(res["phis"]):
            st.markdown(f"**φ_{i}(x)** = `{phi_str}`")
        st.subheader("Step‐by‐Step Log")
        for entry in res["log"]:
            st.markdown(f"$\\bullet$ {format_log_entry(entry)}")


# --- 5) Chebyshev Polynomial Approximation ---

elif method == "Chebyshev Polynomial Approximation":
    st.subheader("Enter f(x) symbolically")
    f_expr = st.text_input("f(x) =", "1/(1 + x**2)")
    fsym, fnum, err = parse_function(f_expr)
    if err:
        st.error(f"Could not parse f(x): {err}")
        st.stop()

    a = st.number_input("Left endpoint a:", value=-1.0, format="%f")
    b = st.number_input("Right endpoint b:", value=1.0, format="%f")
    degree = int(st.number_input("Enter polynomial degree (n):", min_value=0, value=2))
    st.write("Chebyshev Approximation will be constructed by interpolating at Chebyshev nodes.")
    x_eval = st.number_input("Enter x_eval:", value=float((a+b)/2), format="%f")
    if st.button("Run Chebyshev Approx"):
        res = chebyshev_approximation(fsym, a, b, degree, x_eval, use_nodes=True)
        st.subheader("Chebyshev Nodes and f(nodes)")
        df_nodes = pd.DataFrame({"node": res["nodes"], "f(node)": res["f_nodes"]})
        st.dataframe(df_nodes)

        st.subheader("Approximation Polynomial P(x)")
        st.code(res["polynomial"], language="python")
        st.subheader(f"P({x_eval}) ≈ {res['value']:.8f}")
        st.subheader("Step‐by‐Step Log")
        for entry in res["log"]:
            st.markdown(f"$\\bullet$ {format_log_entry(entry)}")


# --- 6) Composite Trapezoidal Rule ---

elif method == "Composite Trapezoidal Rule":
    st.subheader("Enter f(x) symbolically for ∫_a^b f(x) dx")
    f_expr = st.text_input("f(x) =", "exp(-x**2)")
    fsym, fnum, err = parse_function(f_expr)
    if err:
        st.error(f"Could not parse f(x): {err}")
        st.stop()

    a = st.number_input("Left endpoint a:", value=0.0, format="%f")
    b = st.number_input("Right endpoint b:", value=1.0, format="%f")
    n = int(st.number_input("Number of subintervals n:", min_value=1, value=10))

    if st.button("Run Trapezoidal"):
        res = composite_trapezoidal(fsym, fnum, a, b, n)
        st.subheader("Trapezoidal Approximation")
        st.write(f"Approx ∫_a^b f = {res['approx']:.8f}")
        if res["actual"] is not None:
            st.write(f"Exact ∫_a^b f = {res['actual']:.8f}")
            st.write(f"Error = {res['error']:.8f}")
            st.write(f"Error bound ≤ {res['error_bound']:.8f}")
        st.subheader("Step‐by‐Step Log")
        for entry in res["log"]:
            st.markdown(f"$\\bullet$ {format_log_entry(entry)}")


# --- 7) Composite Simpson's Rule ---

elif method == "Composite Simpson's Rule":
    st.subheader("Enter f(x) symbolically for ∫_a^b f(x) dx")
    f_expr = st.text_input("f(x) =", "sin(x)")
    fsym, fnum, err = parse_function(f_expr)
    if err:
        st.error(f"Could not parse f(x): {err}")
        st.stop()

    a = st.number_input("Left endpoint a:", value=0.0, format="%f")
    b = st.number_input("Right endpoint b:", value=np.pi, format="%f")
    n = int(st.number_input("Number of subintervals n (even):", min_value=2, value=10, step=2))
    if n % 2 == 1:
        st.warning("n must be even for Simpson's rule. Automatically incremented by 1.")
        n += 1

    if st.button("Run Simpson"):
        res = composite_simpson(fsym, fnum, a, b, n)
        st.subheader("Simpson's Approximation")
        st.write(f"Approx ∫_a^b f = {res['approx']:.8f}")
        if res["actual"] is not None:
            st.write(f"Exact ∫_a^b f = {res['actual']:.8f}")
            st.write(f"Error = {res['error']:.8f}")
            st.write(f"Error bound ≤ {res['error_bound']:.8f}")
        st.subheader("Step‐by‐Step Log")
        for entry in res["log"]:
            st.markdown(f"$\\bullet$ {format_log_entry(entry)}")
