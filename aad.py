import streamlit as st
import pandas as pd
from scipy.stats import norm
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title=(
        "Ad Astra Diagnostics Fall 2025 Practicum Project: "
        "Allowable Total Error Calculator App"
    ),
    layout="wide",
)

# ---------- GLOBAL APP TITLE ----------
st.markdown(
    """
    <div style="text-align:center; margin-bottom:6px;">
        <span style="font-size:38px; font-weight:700;">
            Allowable Total Error Calculator
        </span>
    </div>

    <div style="text-align:center; font-size:13px; color:#bbbbbb; margin-bottom:35px;">
        Created by Robby Jones, Tucker Parks, Gouri Kallambella, Rachel Morris,
        Amy Nguyen, and Francesca Balestrieri
    </div>
    """,
    unsafe_allow_html=True
)

# ===============================
# DATA LOADING
# ===============================
@st.cache_data
def load_data():
    return pd.read_csv("ATE.csv")

base_df = load_data()
base_df = base_df.dropna(subset=["Biomarker"])

# Editable copy in session_state so changes on the Raw Data tab
# propagate to all tabs.
if "df" not in st.session_state:
    st.session_state["df"] = base_df.copy()

df = st.session_state["df"]

# Infer within / between column names from header
within_col = [c for c in df.columns if "Within Subject" in c][0]
between_col = [c for c in df.columns if "Between Subject" in c][0]

levels = ["Optimal", "Desirable", "Minimum"]

# ===============================
# BIOMARKER LABELS & SUMMARY TEXT
# ===============================
BIOMARKER_LONG_NAMES = {
    "MCH": "Mean Corpuscular Hemoglobin (MCH)",
    "MCHC": "Mean Corpuscular Hemoglobin Concentration (MCHC)",
    "RDW": "Red Cell Distribution Width (RDW)",
    "RDW-SD": "Red Cell Distribution Width – Standard Deviation (RDW-SD)",
    "NLR": "Neutrophil-to-Lymphocyte Ratio (NLR)",
    "IG#": "Immature Granulocyte Count (IG#)",
    "IG%": "Immature Granulocyte Percentage (IG%)",
    "IG#/IG%": "Immature Granulocytes (IG#/IG%)",
    "IG%/IG#": "Immature Granulocytes (IG%/IG#)",
}

# Pulled / paraphrased from the project presentation
BIOMARKER_INFO = {
    "MCH": {
        "what": (
            "Average amount of hemoglobin in a **single red blood cell** "
            "(Mean Corpuscular Hemoglobin)."
        ),
        "purpose": (
            "Evaluates the **oxygen-carrying capacity** of red blood cells, "
            "supporting investigation of anemia and overall RBC health."
        ),
        "ate_range": (0.625, 4.80),  # Goal 1 slide (MCH 0.625–4.80%)
        "notes": [
            "Biological variation-based ATE range derived from within-subject (CVi) "
            "and between-subject (CVg) variance.",
            "Used alongside MCHC and RDW-SD in the initial Goal 1 ATE proposal."
        ],
    },
    "MCHC": {
        "what": (
            "Average hemoglobin **concentration within red blood cells** "
            "(Mean Corpuscular Hemoglobin Concentration)."
        ),
        "purpose": (
            "Assesses the **color and hemoglobin concentration** of RBCs, "
            "supporting classification of hypochromic versus normochromic anemias."
        ),
        "ate_range": (0.335, 4.36),  # Goal 1 slide
        "notes": [
            "Closely linked to RBC hemoglobinization and optical or colorimetric signals.",
            "Paired with MCH and RDW-SD for the first ATE demonstration."
        ],
    },
    "RDW": {
        "what": (
            "Red Cell Distribution Width: a measure of **size variation in red blood "
            "cells** across the circulating RBC population."
        ),
        "purpose": (
            "Supports differentiation of various anemia etiologies and can flag mixed "
            "cell populations (for example, microcytic plus macrocytic cells)."
        ),
        "ate_range": None,  # ATE range presented specifically for RDW-SD
        "notes": [
            "In practice, ATE work in this project focuses on **RDW-SD** as the "
            "quantitative metric derived from the RBC volume distribution.",
        ],
    },
    "RDW-SD": {
        "what": (
            "Standard deviation of red blood cell volume distribution "
            "(RDW-SD), capturing **absolute dispersion** in RBC size."
        ),
        "purpose": (
            "Provides a **numerical measure of anisocytosis**, supporting anemia "
            "workups and tracking of RBC morphology changes over time."
        ),
        "ate_range": (1.06, 9.37),  # Goal 1 slide
        "notes": [
            "ATE range based on published biological variation data for RDW-SD.",
            "Chosen as the analyte-level metric that maps most directly to the CBC "
            "device output used in this project."
        ],
    },
    "NLR": {
        "what": (
            "Ratio of **neutrophils to lymphocytes** "
            "(Neutrophil-to-Lymphocyte Ratio)."
        ),
        "purpose": (
            "Serves as a **marker of systemic inflammation**, with elevated values "
            "associated with infection, physiologic stress, and adverse outcomes "
            "in multiple clinical contexts."
        ),
        "ate_range": (0.395, 3.98),  # Goal 1: NLR slide
        "notes": [
            "Project slides emphasize limited direct NLR biological variation data.",
            "ATE derived by combining CVi and CVg for neutrophils and lymphocytes "
            "separately and propagating through the ratio."
        ],
    },
    "IG#": {
        "what": (
            "Absolute **Immature Granulocyte Count** (IG#): early granulocytes "
            "released into circulation."
        ),
        "purpose": (
            "Reflects **early granulocytes in circulation** and can indicate "
            "infection or inflammation, including **early signs of sepsis**."
        ),
        "ate_range": None,
        "notes": [
            "IG-focused slides highlight that IG# significantly discriminates "
            "infected versus non-infected patients.",
            "Reported sensitivities around the early SIRS window, with IG# "
            "thresholds around 0.03 × 10³/cu·mm associated with high sensitivity.",
        ],
    },
    "IG%": {
        "what": (
            "Relative **Immature Granulocyte Percentage** (IG%): the proportion "
            "of white blood cells that are immature granulocytes."
        ),
        "purpose": (
            "Acts as a **relative marker of bone marrow activation and infection**, "
            "complementing IG#."
        ),
        "ate_range": None,
        "notes": [
            "Project slides show IG% thresholds (for example, ≥0.5 percent) "
            "associated with high sensitivity for bacteremia.",
            "Higher IG% values (for example, >3 percent) combined with IG# "
            "confer very high specificity in some cohorts."
        ],
    },
    "IG#/IG%": {
        "what": (
            "Combined view of **absolute (IG#)** and **relative (IG%)** immature "
            "granulocytes."
        ),
        "purpose": (
            "Enables joint interpretation of count and percentage for **sepsis "
            "and bacteremia risk**, using both sensitivity and specificity."
        ),
        "ate_range": None,
        "notes": [
            "Goal 1 IG% and IG# slide highlights statistically significant AUCs "
            "for bacteremia discrimination (IG# ≈ 0.69, IG% ≈ 0.66).",
            "Additional individual-patient variance data would support tighter ATE "
            "justification, motivating flexible tools such as this calculator.",
        ],
    },
    "IG%/IG#": {
        "what": (
            "Alternative representation of the joint **IG% and IG#** signal, "
            "emphasizing how the two metrics move together."
        ),
        "purpose": (
            "Shares the same clinical goal as IG#/IG%: early identification of "
            "**infection and sepsis risk** based on immature granulocyte dynamics."
        ),
        "ate_range": None,
        "notes": [
            "Supports scenario-based interpretation (for example, elevated IG% with "
            "modest IG#, or the reverse pattern) for device result review.",
        ],
    },
}

# ===============================
# SIDEBAR CONTROLS
# ===============================
st.sidebar.markdown("## ⚙️ ATE Controls")
st.sidebar.markdown(
    "Configure the assumptions used to calculate "
    "**Allowable Total Error (ATE)** thresholds."
)

# ---------- Biomarker selection (long names with abbreviations)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Biomarker")

biomarker_codes = list(df["Biomarker"].unique())
biomarker_label_options = [
    BIOMARKER_LONG_NAMES.get(code, code) for code in biomarker_codes
]
label_to_code = dict(zip(biomarker_label_options, biomarker_codes))

selected_biomarker_label = st.sidebar.selectbox(
    "Select biomarker",
    biomarker_label_options,
    help="Biomarker for which biological variation and ATE matrix are displayed.",
)

# Short code used in calculations
biomarker = label_to_code[selected_biomarker_label]

# ---------- Confidence Interval
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Confidence Interval")

ci_mode = st.sidebar.radio(
    "Select one sided confidence level",
    ["95%", "99%", "Custom"],  # Custom last
    index=0,                   # default 95%
    help=(
        "This setting controls the z score used in the ATE formula. "
        "A one sided 95 percent confidence interval corresponds to "
        "z approximately 1.645."
    ),
)

if ci_mode == "95%":
    z_value = float(norm.ppf(0.95))
elif ci_mode == "99%":
    z_value = float(norm.ppf(0.99))
else:
    with st.sidebar.expander("Advanced: custom confidence level", expanded=True):
        z_value = st.number_input(
            "Custom z value",
            min_value=0.0,
            value=1.645,
            step=0.01,
            format="%.3f",
            help="Custom z value for a one sided confidence level.",
        )

st.sidebar.caption(f"Current z value: **{z_value:.3f}**")

# ---------- Bias factor presets
st.sidebar.markdown("---")
st.sidebar.markdown("### 📏 Bias factor")

bias_mode = st.sidebar.radio(
    "Bias factor preset",
    ["Optimal", "Desirable", "Minimum", "Custom"],  # Custom last
    index=1,  # default Desirable
    help=(
        "Bias factor scales the allowable systematic error component based on "
        "biological variation based quality goals."
    ),
)

bias_preset_map = {
    "Optimal": 0.25,
    "Desirable": 0.50,
    "Minimum": 0.75,
}

if bias_mode == "Custom":
    with st.sidebar.expander("Advanced: custom bias factor", expanded=True):
        bias_mult = st.number_input(
            "Custom bias factor",
            min_value=0.0,
            value=0.50,
            step=0.01,
            format="%.3f",
            help="Custom bias factor replacing the preset value.",
        )
else:
    bias_mult = bias_preset_map[bias_mode]

st.sidebar.caption(f"Current bias factor: **{bias_mult:.3f}**")

# ---------- Imprecision factor presets
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Imprecision factor")

imp_mode = st.sidebar.radio(
    "Imprecision factor preset",
    ["Optimal", "Desirable", "Minimum", "Custom"],  # Custom last
    index=1,  # default Desirable
    help=(
        "Imprecision factor scales the allowable random error component based on "
        "within-subject biological variation (CVi)."
    ),
)

imp_preset_map = {
    "Optimal": 0.125,
    "Desirable": 0.250,
    "Minimum": 0.375,
}

if imp_mode == "Custom":
    with st.sidebar.expander("Advanced: custom imprecision factor", expanded=True):
        imp_mult = st.number_input(
            "Custom imprecision factor",
            min_value=0.0,
            value=0.25,
            step=0.01,
            format="%.3f",
            help="Custom imprecision factor replacing the preset value.",
        )
else:
    imp_mult = imp_preset_map[imp_mode]

st.sidebar.caption(f"Current imprecision factor: **{imp_mult:.3f}**")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Presets reflect Optimal, Desirable, and Minimum quality goals derived from "
    "biological variation based Total Error models."
)

# ===============================
# CORE CALCULATIONS
# ===============================
# Main title already rendered via HTML; st.title call is optional and removed
# to avoid duplication.

# Current biomarker row
row = df[df["Biomarker"] == biomarker].iloc[0]


def get_ate(r: pd.Series, level: str) -> float | None:
    """
    Core ATE formula used in the ATE Calculator tab.

    ATE = z * (imp_mult * CVi_level) + bias_mult * sqrt(CVi_level^2 + CVg^2)
    """
    bias_col = f"Bias - {level}"
    imp_col = f"impreiscion - {level}"  # spelling matches CSV

    between = r.get(bias_col)
    within = r.get(imp_col)

    if pd.isna(between) or pd.isna(within):
        return None

    return (z_value * (imp_mult * within)) + (
        bias_mult * np.sqrt(within**2 + between**2)
    )


# Compute ATE per quality level for the selected biomarker
ate_opt = get_ate(row, "Optimal")
ate_des = get_ate(row, "Desirable")
ate_min = get_ate(row, "Minimum")

# Biomarker-level CVi and CVg for metrics
cvi = row[within_col]
cvg = row[between_col]

# ===============================
# TABS
# ===============================
overview_tab, biomarker_tab, calc_tab, matrix_tab, raw_tab = st.tabs(
    ["Overview", "Biomarker Summary", "ATE Calculator", "Matrix", "Raw Data"]
)

# ---------- OVERVIEW TAB ----------
with overview_tab:

    st.markdown(
        """
        This page provides a high-level overview of how the interface defines and uses
        **Allowable Total Error (ATE)** across biomarkers.
        """
    )

    # ---- Main Explanation Section ----
    st.markdown(
        """
        ### Definition of Allowable Total Error

        For this project, Total Error limits are grounded in **biological variation**:

        - **CVi (within-subject variation)** captures how much the biomarker
          naturally fluctuates within a single patient.
        - **CVg (between-subject variation)** captures how much the biomarker
          differs across patients in a reference population.
        - **Bias** and **Imprecision** factors (Optimal, Desirable, Minimum)
          translate these CV values into practical Total Error limits.
        """
    )

    # ---- ATE Formula ----
    st.markdown("#### ATE formula used in this tool")
    st.latex(r"""
        \text{ATE} =
        (Z\text{-score} \times \text{Imprecision Factor} \times CV_i)
        + (\text{Bias Factor} \times \sqrt{CV_i^2 + CV_g^2})
    """)

    # ---- Instructions Section ----
    st.markdown(
        """
        In this framework, the **z value**, **bias factor**, and **imprecision factor**
        are controlled through the sidebar settings.

        ### Structure of the interface

        1. The **sidebar** controls the confidence level and quality goal presets.  
        2. The **Biomarker Summary** tab describes each biomarker in the context of the
           Ad Astra Diagnostics project.  
        3. The **ATE Calculator** tab displays the resulting Total Error for Optimal,  
           Desirable, and Minimum quality goals for the selected biomarker.  
        4. The **Matrix** tab shows how combinations of bias and imprecision factors
           perform relative to a chosen Total Error threshold.  
        5. The **Raw Data** tab exposes the underlying biological variation and
           factor table used for all calculations.
        """
    )

# ---------- BIOMARKER SUMMARY TAB ----------
with biomarker_tab:
    info = BIOMARKER_INFO.get(biomarker, {})
    full_name = BIOMARKER_LONG_NAMES.get(biomarker, biomarker)

    st.subheader(full_name)

    col_left, col_right = st.columns([1.1, 1.6])

    with col_left:
        st.markdown("#### Snapshot")
        st.metric("Within-subject variation (CVi)", f"{cvi:.3f}")
        st.metric("Between-subject variation (CVg)", f"{cvg:.3f}")
        if info.get("ate_range"):
            lo, hi = info["ate_range"]
            st.metric("Proposed ATE range (%)", f"{lo:.3f} – {hi:.3f}")
        else:
            st.metric("Proposed ATE range (%)", "Defined in Matrix and Calculator tabs")

    with col_right:
        st.markdown("#### Biomarker interpretation")
        if info.get("what"):
            st.write(info["what"])
        if info.get("purpose"):
            st.write(info["purpose"])

        notes = info.get("notes", [])
        if notes:
            st.markdown("#### Project context")
            for n in notes:
                st.markdown(f"- {n}")

# ---------- ATE CALCULATOR TAB ----------
with calc_tab:
    st.subheader(f"ATE Calculator for {selected_biomarker_label}")

    # Horizontal key parameters row
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric("Within-subject variation (CVi)", f"{cvi:.3f}")
    with k2:
        st.metric("Between-subject variation (CVg)", f"{cvg:.3f}")
    with k3:
        st.metric("One sided z value", f"{z_value:.3f}")
    with k4:
        st.metric("Bias factor", f"{bias_mult:.3f}")
    with k5:
        st.metric("Imprecision factor", f"{imp_mult:.3f}")

    # Indicator for how the Total Error threshold is defined on the Matrix tab
    threshold_mode_current = st.session_state.get(
        "threshold_mode_matrix",
        "Enter within-subject and between-subject variation",
    )
    if threshold_mode_current.startswith("Enter within-subject"):
        st.info(
            "Matrix tab: Total Error threshold is currently defined from biological "
            "variation inputs (CVi and CVg)."
        )
    else:
        st.info(
            "Matrix tab: Total Error threshold is currently defined by a direct "
            "Total Error value."
        )

    st.markdown("---")

    # ATE summary
    if ate_des is not None:
        st.markdown(f"### Desirable ATE: `{ate_des:.3f}`")
    else:
        st.info(
            "No complete within- and between-subject variability values are available "
            "for the Desirable level for this biomarker."
        )

    valid_values = [v for v in [ate_opt, ate_des, ate_min] if v is not None]

    if valid_values:
        low = min(valid_values)
        high = max(valid_values)
        st.markdown(
            f"### ATE range across quality levels: `{low:.3f}` to `{high:.3f}`"
        )

        summary_df = pd.DataFrame(
            {
                "Level": ["Optimal", "Desirable", "Minimum"],
                "Allowable Total Error": [ate_opt, ate_des, ate_min],
            }
        )
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
    else:
        st.info(
            "No valid ATE values are available across Optimal, Desirable, and "
            "Minimum levels for this biomarker."
        )

# ---------- MATRIX TAB ----------
with matrix_tab:
    st.subheader(
        f"Bias and Imprecision ATE Matrix vs Custom Total Error Threshold "
        f"for {biomarker}"
    )

    st.markdown(
        """
        This tab summarizes **Allowable Total Error (ATE)** for the selected biomarker
        as a 3 × 3 matrix:

        - **Rows** represent imprecision factor levels (Optimal, Desirable, Minimum).  
        - **Columns** represent bias factor levels (Optimal, Desirable, Minimum).  
        - **Cells** are shaded to indicate whether the ATE at each combination 
          meets or exceeds the chosen Total Error threshold.  
        """
    )

    row_matrix = row
    within_biomarker = row_matrix[within_col]   # CVi
    between_biomarker = row_matrix[between_col]  # CVg

    # ---- How threshold is defined ----
    st.markdown("### Total Error Threshold")

    threshold_mode = st.radio(
        "Select how the Total Error threshold (ATE) is defined:",
        [
            "Enter Total Error directly",
            "Enter within-subject and between-subject variation",
        ],
        index=1,
        key="threshold_mode_matrix",
    )

    if threshold_mode == "Enter Total Error directly":
        threshold_ate = st.number_input(
            "Threshold Allowable Total Error (ATE)",
            min_value=0.0,
            value=1.0,
            step=0.01,
            format="%.3f",
            key="threshold_ate_direct",
        )
        st.markdown(
            f"**Using user-entered Total Error threshold:** `{threshold_ate:.3f}`"
        )
    else:
        st.markdown("### Biological variation inputs for Total Error threshold")

        within_input = st.number_input(
            "Within-subject biological variation (CVi)",
            min_value=0.0,
            value=float(within_biomarker),
            step=0.01,
            format="%.3f",
            key="within_input_matrix",
        )

        between_input = st.number_input(
            "Between-subject biological variation (CVg)",
            min_value=0.0,
            value=float(between_biomarker),
            step=0.01,
            format="%.3f",
            key="between_input_matrix",
        )

        threshold_ate = (z_value * (imp_mult * within_input)) + (
            bias_mult * np.sqrt(within_input**2 + between_input**2)
        )

        st.markdown(
            f"**Calculated Total Error threshold (ATE):** `{threshold_ate:.3f}`"
        )

    # ---- Build 3×3 numeric grid for the matrix ----
    numeric_grid: dict[str, dict[str, float | None]] = {}

    for imp_level in levels:
        numeric_grid[imp_level] = {}
        for bias_level in levels:
            bias_factor = row_matrix[f"Bias - {bias_level}"]
            imp_factor = row_matrix[f"impreiscion - {imp_level}"]

            if (
                pd.isna(within_biomarker)
                or pd.isna(between_biomarker)
                or pd.isna(bias_factor)
                or pd.isna(imp_factor)
            ):
                numeric_grid[imp_level][bias_level] = None
            else:
                ate_val = (
                    z_value * (imp_factor * within_biomarker)
                    + bias_factor
                    * np.sqrt(within_biomarker**2 + between_biomarker**2)
                )
                numeric_grid[imp_level][bias_level] = ate_val

    # ---- Styled HTML matrix ----
    st.markdown("## 🧬 ATE Performance Matrix")

    table_style = (
        "border-collapse:separate;"
        "border-spacing:0;"
        "margin-top:20px;"
        "margin-left:auto;margin-right:auto;"
        "font-size:20px;"
        "border-radius:14px;"
        "overflow:hidden;"
        "box-shadow:0 6px 22px rgba(0,0,0,0.35);"
    )

    th_style = (
        "padding:16px 24px;"
        "text-align:center;"
        "font-weight:650;"
        "color:white;"
        "border:1px solid #333;"
        "background:linear-gradient(to bottom, #2f3336, #1f2123);"
    )

    td_style_base = (
        "padding:16px 24px;"
        "text-align:center;"
        "border:1px solid #333;"
        "background-color:#26292b;"
        "color:white;"
        "font-weight:500;"
    )

    def cell_html(val):
        if val is None or pd.isna(val):
            return f"<td style='{td_style_base}'></td>"

        if val >= threshold_ate:
            bg = "#6fd38e"   # mint green
        else:
            bg = "#cc6f6f"   # coral red

        style = td_style_base + f"background-color:{bg};color:white;font-weight:600;"
        return f"<td style='{style}'>{val:.3f}</td>"

    html = f"<table style='{table_style}'>"

    # Row 1: Bias Factor banner
    html += "<tr>"
    html += f"<th style='{th_style}' colspan='2'></th>"
    html += f"<th style='{th_style}' colspan='3'>Bias Factor</th>"
    html += "</tr>"

    # Row 2: Bias headers
    html += "<tr>"
    html += f"<th style='{th_style}' colspan='2'></th>"
    for bias_level in levels:
        html += f"<th style='{th_style}'>{bias_level}</th>"
    html += "</tr>"

    # Row 3: Optimal row with merged Impresion Factor cell (label retained to match prior layout)
    html += "<tr>"
    html += f"<th style='{th_style}' rowspan='3'>Impresion Factor</th>"
    html += f"<th style='{th_style}'>Optimal</th>"
    for bias_level in levels:
        html += cell_html(numeric_grid["Optimal"][bias_level])
    html += "</tr>"

    # Row 4: Desirable
    html += "<tr>"
    html += f"<th style='{th_style}'>Desirable</th>"
    for bias_level in levels:
        html += cell_html(numeric_grid["Desirable"][bias_level])
    html += "</tr>"

    # Row 5: Minimum
    html += "<tr>"
    html += f"<th style='{th_style}'>Minimum</th>"
    for bias_level in levels:
        html += cell_html(numeric_grid["Minimum"][bias_level])
    html += "</tr>"

    html += "</table>"

    st.markdown(html, unsafe_allow_html=True)

# ---------- RAW DATA TAB ----------
with raw_tab:
    st.subheader("Raw ATE Input Data")

    st.markdown(
        """
        This tab displays the underlying table used for ATE calculations.  
        Cells are editable, and changes made in this view are stored in the current
        session and propagate automatically to all other tabs.
        """
    )

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="raw_editor",
    )

    # If the table has changed, update session_state so the rest
    # of the interface uses the revised data on the next rerun.
    if not edited_df.equals(df):
        st.session_state["df"] = edited_df
        st.success("Edits applied. Other tabs are now using the updated data.")
