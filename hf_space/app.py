import os
import re
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="US H-1B & F-1 STEM OPT Market Intelligence (2024–2026)",
    layout="wide"
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

@st.cache_data
def load_cube():
    p = os.path.join(DATA_DIR, "h1b_cube.parquet")
    return pd.read_parquet(p)

@st.cache_data
def load_aux_csv(filename):
    p = os.path.join(DATA_DIR, filename)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

df = load_cube()
fdr_df = load_aux_csv("state_bootstrap_fdr_results.csv")
pca_df = load_aux_csv("employer_pca_clusters.csv")
anova_df = load_aux_csv("anova_naics_wage_results.csv")

st.title("US H-1B & F-1 STEM OPT Market Intelligence Platform (FY 2024–2026)")
st.caption(
    "Built by Shaury Pratap Singh | Data Warehouse: 2.1M+ Federal Records (DOL LCA, USCIS H-1B Data Hub, E-Verify, DOL PERM) "
    "| Stack: MS SQL Server, Python (SciPy, Statsmodels, XGBoost), R (ANOVA), DuckDB, Plotly"
)

# Top-level KPI metrics from the 2.1M-row warehouse
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Deduplicated LCA Filings", "1,372,949", "FY 2024–2026 Q3")
k2.metric("E-Verify Records Indexed", "461,297", "STEM OPT Eligible")
k3.metric("USCIS Employer Records", "148,754", "NAICS + Initial Cap")
k4.metric("XGBoost / Poly GLM R²", "0.686", "MAE: $18,044")
k5.metric("2024→2026 Role Drift (PSI)", "0.1389", "KL Div: 0.0717")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Role, Company & State Explorer",
    "2. AI Text-to-SQL Assistant",
    "3. Statistical & ML Inference Lab",
    "4. Student Salary & Sponsor Calculator"
])

# ==============================================================================
# TAB 1: Interactive Multi-Dimensional Explorer
# ==============================================================================
with tab1:
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 2.2, 2.0, 1.4])
    years = sorted(df["DECISION_YEAR"].dropna().unique(), reverse=True)
    sel_years = c1.multiselect("Fiscal Year", years, default=[2025, 2026])
    states = ["All States"] + sorted(df["WORKSITE_STATE"].dropna().unique().tolist())
    sel_state = c2.selectbox("Worksite State", states, index=states.index("NJ") if "NJ" in states else 0)
    sectors = ["All Sectors"] + sorted(df["NAICS_SECTOR"].dropna().unique().tolist())
    sel_sector = c3.selectbox("Industry (NAICS Sector)", sectors, index=0)
    role_kw = c4.text_input("Role Keyword (e.g. DATA, SOFTWARE, ANALYST, AI, MEDICAL)", value="DATA")
    everify_only = c5.checkbox("E-Verify Open Only (STEM OPT)", value=True)

    sub = df[df["DECISION_YEAR"].isin(sel_years)] if sel_years else df.copy()
    if sel_state != "All States":
        sub = sub[sub["WORKSITE_STATE"] == sel_state]
    if sel_sector != "All Sectors":
        sub = sub[sub["NAICS_SECTOR"] == sel_sector]
    if role_kw.strip():
        kw = role_kw.strip().upper()
        sub = sub[sub["SOC_TITLE"].str.contains(kw, na=False) | sub["JOB_TITLE"].str.contains(kw, na=False)]
    if everify_only:
        sub = sub[sub["IS_EVERIFY_OPEN"] == 1]

    col_left, col_right = st.columns([1.3, 1.0])

    with col_left:
        top_comp = sub.groupby(["COMPANY", "NAICS_SECTOR", "WORKSITE_STATE", "SOC_TITLE"], as_index=False).agg(
            CERTIFIED_ROLES=("CERTIFIED_FILINGS", "sum"),
            F1_ENTRY_LEVEL_HIRES=("F1_ENTRY_LEVEL_HIRES", "sum"),
            USCIS_INITIAL_APPROVALS=("USCIS_INITIAL_H1B_APPROVALS", "max"),
            AVG_SALARY=("AVG_OFFERED_SALARY", "mean"),
            WAGE_PREMIUM=("AVG_WAGE_PREMIUM", "mean")
        ).sort_values(["F1_ENTRY_LEVEL_HIRES", "CERTIFIED_ROLES"], ascending=False).head(20)

        if not top_comp.empty:
            fig_bar = px.bar(
                top_comp.sort_values("F1_ENTRY_LEVEL_HIRES", ascending=True),
                x="F1_ENTRY_LEVEL_HIRES",
                y="COMPANY",
                color="AVG_SALARY",
                orientation="h",
                hover_data=["SOC_TITLE", "NAICS_SECTOR", "CERTIFIED_ROLES", "WAGE_PREMIUM"],
                title=f"Top 20 Sponsors by Entry-Level (Wage Level I/II) New Hires ({sel_state})",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.warning("No matching filings found for the selected filters.")

    with col_right:
        state_map_df = sub.groupby("WORKSITE_STATE", as_index=False).agg(
            F1_ENTRY_HIRES=("F1_ENTRY_LEVEL_HIRES", "sum"),
            CERTIFIED_FILINGS=("CERTIFIED_FILINGS", "sum"),
            AVG_SALARY=("AVG_OFFERED_SALARY", "mean")
        )
        if not state_map_df.empty:
            fig_map = px.choropleth(
                state_map_df,
                locations="WORKSITE_STATE",
                locationmode="USA-states",
                color="F1_ENTRY_HIRES",
                hover_data=["CERTIFIED_FILINGS", "AVG_SALARY"],
                scope="usa",
                title="Geographic Concentration of Matching Roles",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_map, width="stretch")

    st.subheader("Detailed Company, Role & Salary Breakdown")
    if not top_comp.empty:
        top_comp["AVG_SALARY"] = top_comp["AVG_SALARY"].round(0)
        top_comp["WAGE_PREMIUM"] = top_comp["WAGE_PREMIUM"].round(0)
        st.dataframe(top_comp, width="stretch")

# ==============================================================================
# TAB 2: Natural-Language & Live DuckDB SQL Assistant
# ==============================================================================
with tab2:
    st.subheader("Query the H-1B & F-1 Cube in Plain English or SQL")
    st.markdown(
        "Type a question in plain English (e.g., *'Top 15 companies in NJ for DATA roles in 2025'* or "
        "*'Compare hiring by industry in NY in 2026'*) or customize the generated SQL query directly."
    )

    nl_q = st.text_input(
        "Ask a question in plain English:",
        value="Which E-Verify companies in NJ hired the most entry-level DATA roles in 2025 and 2026?"
    )

    def nl_to_sql(question: str) -> str:
        q_up = question.upper()
        where_clauses = ["IS_EVERIFY_OPEN = 1"]
        # Detect 2-letter state code or state names
        state_map = {"NEW JERSEY": "NJ", "NEW YORK": "NY", "CALIFORNIA": "CA", "TEXAS": "TX", "WASHINGTON": "WA", "MASSACHUSETTS": "MA", "ILLINOIS": "IL", "VIRGINIA": "VA", "GEORGIA": "GA"}
        detected_state = None
        for s_name, s_code in state_map.items():
            if s_name in q_up:
                detected_state = s_code
                break
        if not detected_state:
            tokens = re.findall(r"\b[A-Z]{2}\b", q_up)
            valid_states = set(df["WORKSITE_STATE"].dropna().unique())
            for t in tokens:
                if t in valid_states and t not in ("IN", "OR", "BY", "TO", "ON", "AT", "IS", "AI", "IT", "US"):
                    detected_state = t
                    break
        if detected_state:
            where_clauses.append(f"WORKSITE_STATE = '{detected_state}'")

        # Detect years
        years_found = [y for y in ["2024", "2025", "2026"] if y in q_up]
        if years_found:
            where_clauses.append(f"DECISION_YEAR IN ({', '.join(years_found)})")

        # Detect role keywords
        for kw in ["DATA", "SOFTWARE", "ANALYST", "MACHINE LEARNING", "AI", "MEDICAL", "FINANCE", "RESEARCH", "ENGINEER"]:
            if kw in q_up:
                where_clauses.append(f"(SOC_TITLE LIKE '%{kw}%' OR JOB_TITLE LIKE '%{kw}%')")
                break

        where_str = " AND ".join(where_clauses)
        return f"""SELECT 
    COMPANY,
    NAICS_SECTOR,
    WORKSITE_STATE AS STATE,
    SOC_TITLE,
    SUM(CERTIFIED_FILINGS) AS CERTIFIED_FILINGS,
    SUM(F1_ENTRY_LEVEL_HIRES) AS F1_ENTRY_HIRES,
    ROUND(AVG(AVG_OFFERED_SALARY), 0) AS AVG_SALARY
FROM df
WHERE {where_str}
GROUP BY COMPANY, NAICS_SECTOR, WORKSITE_STATE, SOC_TITLE
ORDER BY F1_ENTRY_HIRES DESC, CERTIFIED_FILINGS DESC
LIMIT 20;"""

    generated_sql = nl_to_sql(nl_q)
    sql_box = st.text_area("Generated SQL Query (Editable):", value=generated_sql, height=180)

    if st.button("Run Query"):
        try:
            res_df = duckdb.query(sql_box).to_df()
            st.success(f"Returned {len(res_df)} rows.")
            st.dataframe(res_df, width="stretch")
        except Exception as e:
            st.error(f"SQL Error: {e}")

# ==============================================================================
# TAB 3: Statistical & ML Inference Lab (Your Syllabus Showcase)
# ==============================================================================
with tab3:
    st.subheader("Statistical Inference, Econometrics & Machine Learning Results")
    s1, s2 = st.columns(2)

    with s1:
        st.markdown("#### 1. Bootstrapped 95% CIs & Benjamini-Hochberg FDR (Top 15 States)")
        if not fdr_df.empty:
            fig_fdr = px.ErrorBar = px.scatter(
                fdr_df,
                x="STATE",
                y="MEDIAN_SALARY",
                color="SIGNIFICANT_FDR_05",
                error_y=fdr_df["BOOT_95_CI_HIGH"] - fdr_df["MEDIAN_SALARY"],
                error_y_minus=fdr_df["MEDIAN_SALARY"] - fdr_df["BOOT_95_CI_LOW"],
                title="State Median H-1B Salaries with Percentile Bootstrap 95% CIs (FDR alpha=0.05)"
            )
            st.plotly_chart(fig_fdr, width="stretch")
            st.dataframe(fdr_df, width="stretch")

    with s2:
        st.markdown("#### 2. PCA Dimensionality Reduction (81.8% Variance Explained)")
        if not pca_df.empty:
            fig_pca = px.scatter(
                pca_df,
                x="PCA_1",
                y="PCA_2",
                color="NAICS_SECTOR",
                size="TOTAL",
                hover_name="EMPLOYER_CLEAN_KEY",
                hover_data=["AVG_SALARY", "F1_ENTRY_HIRES", "BAYESIAN_SHRUNK_APPROVAL_RATE"],
                title="PCA Clustering of Top 300 Employers by Sponsorship Profile"
            )
            st.plotly_chart(fig_pca, width="stretch")

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.info(
        "**MLE Log-Normal & Law of Total Variance**\n\n"
        "- **MLE Parameters:** $\\hat{\\mu} = 11.6827$, $\\hat{\\sigma} = 0.3899$\n"
        "- **Within-State Mean Variance:** $4.06 \\times 10^9$\n"
        "- **Between-State Variance:** $3.81 \\times 10^8$\n"
        "- Proves role & company tier drive 10.6x more salary variance than state geography alone."
    )
    m2.info(
        "**Regression, XGBoost & Time Series Forecasting**\n\n"
        "- **Polynomial GLM Adjusted $R^2$:** `0.6566`\n"
        "- **XGBoost Salary Predictor $R^2$:** `0.6859` (`MAE: $18,044`)\n"
        "- **ETS Time Series MAE:** `47,834` filings (outperforming ARIMA `83,635` MAE by **42.8%**)."
    )
    m3.info(
        "**Two-Way ANOVA & Role Drift (FY24 vs FY26)**\n\n"
        "- **NAICS Sector Effect:** $F$-test $p < 10^{-16}$\n"
        "- **Prevailing Wage Level Effect:** $F$-test $p < 10^{-16}$\n"
        "- **Population Stability Index (PSI):** `0.1389` (`KL Div = 0.0717`), confirming significant structural shift toward AI & Data roles."
    )

# ==============================================================================
# TAB 4: Student Salary & Sponsor Calculator
# ==============================================================================
with tab4:
    st.subheader("Expected Offer & Wage-Premium Estimator")
    calc_s1, calc_s2 = st.columns(2)
    c_state = calc_s1.selectbox("Select Target State", sorted(df["WORKSITE_STATE"].dropna().unique()), index=0)
    c_role = calc_s2.selectbox("Select Standardized SOC Role", df["SOC_TITLE"].value_counts().head(40).index.tolist(), index=0)

    match = df[(df["WORKSITE_STATE"] == c_state) & (df["SOC_TITLE"] == c_role) & (df["IS_EVERIFY_OPEN"] == 1)]
    if not match.empty:
        mean_sal = match["AVG_OFFERED_SALARY"].mean()
        mean_pw = match["AVG_PREVAILING_WAGE"].mean()
        prem = mean_sal - mean_pw
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Expected Offered Salary", f"${mean_sal:,.0f}")
        sc2.metric("Average Prevailing Wage Baseline", f"${mean_pw:,.0f}")
        sc3.metric("Employer Wage Premium", f"+${prem:,.0f}")
    else:
        st.warning("Not enough E-Verify filings for this exact State + SOC pair.")