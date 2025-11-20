import streamlit as st
import pandas as pd

st.title("Biomarker App")

# One connection for all tabs
conn = st.connection("postgresql", type="sql")

# Create tabs
dashboard_tab, sql_tab = st.tabs(["Dashboard", "SQL Workbench"])

# ===============================
# TAB 1 - DASHBOARD
# ===============================
with dashboard_tab:
    st.subheader("Dashboard")

    # ---------- NEW: Count rows in BioMarker ----------
    try:
        count_df = conn.query('SELECT COUNT(*) AS total_rows FROM public."BioMarker";')
        total = int(count_df["total_rows"][0])
        st.metric("Number of Extracted Test Results", total)
    except Exception as e:
        st.error(f"Error counting rows: {e}")

    # ---------- NEW: Count rows in BioMarker ----------
    try:
        count_df = conn.query('SELECT COUNT(*) AS total_rows FROM public."BioMarker";')
        total = int(count_df["total_rows"][0])
        st.metric("Number of Extracted Test Results", total)
    except Exception as e:
        st.error(f"Error counting rows: {e}")

    # Example biomarker preview
    try:
        biomarker_df = conn.query('SELECT * FROM public."BioMarker" LIMIT 20;')
        st.write("Biomarker preview")
        st.dataframe(biomarker_df)
    except Exception as e:
        st.error(f"Error loading Biomarker data: {e}")

# ===============================
# TAB 2 - SQL WORKBENCH
# ===============================
with sql_tab:
    st.subheader("SQL Workbench")

    # Get list of tables
    try:
        tables_df = conn.query("""
            SELECT table_name 
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)

        table_list = sorted([row[0] for row in tables_df.values])

        selected_table = st.selectbox("Select a table:", table_list)

        default_sql = f'SELECT * FROM public."{selected_table}" LIMIT 10;'

        sql_query = st.text_area("SQL query:", default_sql, height=150)

        if st.button("Run SQL"):
            try:
                result_df = conn.query(sql_query)
                st.success("Query executed successfully")
                st.dataframe(result_df)
            except Exception as e:
                st.error(f"SQL error: {e}")
    except Exception as e:
        st.error(f"Error fetching table list: {e}")
