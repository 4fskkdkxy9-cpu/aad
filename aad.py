import streamlit as st
import pandas as pd
from scipy.stats import norm   # pip install scipy
import numpy as np

st.title("ATE Calculator")

def load_data():
    return pd.read_csv("ATE.csv")

df = load_data()
df = df.dropna(subset=["Biomarker"])

# Infer within / between column names from header
within_col = [c for c in df.columns if "Within Subject" in c][0]
between_col = [c for c in df.columns if "Between Subject" in c][0]

st.sidebar.header("Controls")

# -----------------------------
# Biomarker selection
# -----------------------------
biomarker = st.sidebar.selectbox(
    "Select biomarker",
    df["Biomarker"].unique()
)

# Used for Home + Calculator tabs
row = df[df["Biomarker"] == biomarker].iloc[0]

# -----------------------------
# One sided CI options -> z score
# -----------------------------
st.sidebar.subheader("Confidence Interval")

ci_mode = st.sidebar.radio(
    "Choose CI preset or custom",
    ["Custom", "95%", "99%"],
    index=0
)

if ci_mode == "Custom":
    one_sided_ci = st.sidebar.slider(
        "One sided confidence interval (probability)",
        min_value=0.90,
        max_value=0.999,
        value=0.95,
        step=0.001
    )
    z_value = float(norm.ppf(one_sided_ci))

elif ci_mode == "95%":
    one_sided_ci = 0.95
    z_value = float(norm.ppf(one_sided_ci))

elif ci_mode == "99%":
    one_sided_ci = 0.99
    z_value = float(norm.ppf(one_sided_ci))

ci_percent = one_sided_ci * 100

st.sidebar.markdown(f"**Confidence interval:** {ci_percent:.1f}%")
st.sidebar.markdown(f"**Resulting z score:** {z_value:.3f}")

# -----------------------------
# Bias multiplier (with presets)
# -----------------------------
st.sidebar.subheader("Bias multiplier")

bias_mode = st.sidebar.radio(
    "Bias multiplier preset",
    ["Custom", "Optimal", "Desirable", "Minimum"],
    index=0
)

if bias_mode == "Custom":
    bias_mult = st.sidebar.slider(
        "Bias multiplier value",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.05
    )
else:
    preset_map = {
        "Optimal": 0.125,
        "Desirable": 0.25,
        "Minimum": 0.375,
    }
    bias_mult = preset_map[bias_mode]
    st.sidebar.markdown(f"**Bias multiplier value:** {bias_mult:.3f}")

# -----------------------------
# Imprecision multiplier (with presets)
# -----------------------------
st.sidebar.subheader("Imprecision multiplier")

imp_mode = st.sidebar.radio(
    "Imprecision multiplier preset",
    ["Custom", "Optimal", "Desirable", "Minimum"],
    index=0
)

if imp_mode == "Custom":
    imp_mult = st.sidebar.slider(
        "Imprecision multiplier value",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.05
    )
else:
    preset_map_imp = {
        "Optimal": 0.25,
        "Desirable": 0.5,
        "Minimum": 0.75,
    }
    imp_mult = preset_map_imp[imp_mode]
    st.sidebar.markdown(f"**Imprecision multiplier value:** {imp_mult:.3f}")

levels = ["Optimal", "Desirable", "Minimum"]

# -----------------------------
# Core ATE formula for main tab
# ATE = z * (imp_mult * within) + bias_mult * sqrt(within^2 + between^2)
# where within = imprecision column, between = bias column
# -----------------------------
def get_ate(r, level: str) -> float | None:
    bias_col = f"Bias - {level}"
    imp_col = f"impreiscion - {level}"  # uses your CSV spelling

    between = r.get(bias_col)
    within = r.get(imp_col)

    if pd.isna(between) or pd.isna(within):
        return None

    return (z_value * (imp_mult * within)) + (
        bias_mult * np.sqrt(within**2 + between**2)
    )

# Compute ATEs for each quality level
ate_opt = get_ate(row, "Optimal")
ate_des = get_ate(row, "Desirable")
ate_min = get_ate(row, "Minimum")

# -----------------------------
# TABS: Home, Calculator, Matrix
# -----------------------------
home_tab, calc_tab, matrix_tab = st.tabs(["Home", "ATE Calculator", "Matrix"])

with home_tab:
    st.header("Welcome to the ATE Calculator")
    st.markdown(
        """
        This tool helps you explore Allowable Total Error (ATE) for different biomarkers
        across optimal, desirable, and minimum performance specifications.
        """
    )

with calc_tab:
    st.subheader(f"Biomarker of Interest: {biomarker}")

    if ate_des is not None:
        st.markdown(f"### Desirable ATE: `{ate_des:.3f}`")
    else:
        st.info("No complete within / between variability values for Desirable level for this biomarker.")

    valid_values = [v for v in [ate_opt, ate_des, ate_min] if v is not None]

    if valid_values:
        low = min(valid_values)
        high = max(valid_values)
        st.markdown(f"### ATE range: `{low:.3f}` to `{high:.3f}`")

        summary_df = pd.DataFrame(
            {
                "Level": ["Optimal", "Desirable", "Minimum"],
                "Allowable Total Error": [ate_opt, ate_des, ate_min],
            }
        )
        st.dataframe(summary_df, hide_index=True)
    else:
        st.info("No valid ATE values across Optimal, Desirable, Minimum for this biomarker.")

with matrix_tab:
    st.subheader(f"Bias-Imprecision ATE Matrix vs Custom Threshold for {biomarker}")
    st.markdown(
        "This tab builds a 3 by 3 matrix of ATE values (Bias level x Imprecision level) "
        "for the selected biomarker and compares them to a custom ATE computed from user "
        "entered within and between subject variability."
    )

    # Row for current biomarker (for this tab)
    row_matrix = df[df["Biomarker"] == biomarker].iloc[0]

    # Biomarker-specific within / between variability from columns B / C
    within_biomarker = row_matrix[within_col]
    between_biomarker = row_matrix[between_col]

    st.markdown("##### Biomarker-specific variability from ATE.csv")
    st.write(
        {
            "Within subject variability": within_biomarker,
            "Between subject variability": between_biomarker,
        }
    )

    # Custom inputs for threshold ATE
    st.markdown("#### Custom variability inputs (for threshold ATE)")

    within_input = st.number_input(
        "Within subject variability",
        min_value=0.0,
        value=float(within_biomarker) if not pd.isna(within_biomarker) else 0.0,
        step=0.01,
        format="%.3f",
        key="within_input",
    )

    between_input = st.number_input(
        "Between subject variability",
        min_value=0.0,
        value=float(between_biomarker) if not pd.isna(between_biomarker) else 0.0,
        step=0.01,
        format="%.3f",
        key="between_input",
    )

    # Threshold ATE uses sidebar multipliers + user-entered within/between
    threshold_ate = (z_value * (imp_mult * within_input)) + (
        bias_mult * np.sqrt(within_input**2 + between_input**2)
    )

    st.markdown(f"**Calculated ATE from custom inputs:** `{threshold_ate:.3f}`")

    # Build 3x3 matrix using biomarker-specific variability
    st.markdown("#### ATE matrix for this biomarker (current z, per-level multipliers)")

    matrix_rows = []
    for bias_level in levels:
        for imp_level in levels:
            # per-level multipliers from the sheet
            bias_mult_cell = row_matrix[f"Bias - {bias_level}"]
            imp_mult_cell = row_matrix[f"impreiscion - {imp_level}"]

            if pd.isna(within_biomarker) or pd.isna(between_biomarker):
                ate_val = np.nan
            else:
                ate_val = (
                    z_value * (imp_mult_cell * within_biomarker)
                    + bias_mult_cell
                    * np.sqrt(within_biomarker**2 + between_biomarker**2)
                )

            matrix_rows.append(
                {
                    "Bias level": bias_level,
                    "Imprecision level": imp_level,
                    "ATE": ate_val,
                }
            )

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_pivot = matrix_df.pivot(
        index="Bias level",
        columns="Imprecision level",
        values="ATE",
    )

    # Force row/column order: Optimal, Desirable, Minimum
    order = ["Optimal", "Desirable", "Minimum"]
    matrix_pivot = matrix_pivot.reindex(index=order).reindex(columns=order)

    # High contrast coloring based on threshold_ate
    def color_cell(val):
        if pd.isna(val):
            return ""
        base_style = "color: white; font-weight: 600; border: 1px solid #222;"
        if val >= threshold_ate:
            return base_style + "background-color: #66bb6a;"  # green
        else:
            return base_style + "background-color: #ef5350;"  # red

    styled_matrix = matrix_pivot.style.format("{:.3f}").applymap(color_cell)

    # Render as HTML so it always refreshes with new biomarker
    st.markdown(styled_matrix.to_html(), unsafe_allow_html=True)
