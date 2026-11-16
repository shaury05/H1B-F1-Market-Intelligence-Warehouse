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
    df_raw = pd.read_parquet(p)
    # Pre-compute weighted numerator columns for exact weighted averages
    df_raw["WEIGHTED_SAL_NUM"] = df_raw["TOTAL_FILINGS"] * df_raw["AVG_OFFERED_SALARY"]
    df_raw["WEIGHTED_PW_NUM"] = df_raw["TOTAL_FILINGS"] * df_raw["AVG_PREVAILING_WAGE"]
    df_raw["WEIGHTED_PREM_NUM"] = df_raw["TOTAL_FILINGS"] * df_raw["AVG_WAGE_PREMIUM"]
    return df_raw

@st.cache_data
def load_aux_csv(filename):
    p = os.path.join(DATA_DIR, filename)
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

df = load_cube()
fdr_df = load_aux_csv("state_bootstrap_fdr_results.csv")
pca_df = load_aux_csv("employer_pca_clusters.csv")
bayes_df = load_aux_csv("top_500_employers_bayesian_ranked.csv")

VALID_COLUMNS = set(df.columns)

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
    "2. Deterministic Text-to-SQL Assistant",
    "3. Statistical & ML Inference Lab",
    "4. Student Salary & Sponsor Calculator"
])

# ==============================================================================
# TAB 1: Multi-Select Role, Company, Industry & State Explorer
# ==============================================================================
with tab1:
    st.markdown("##### Filter Across Multiple States, Industries, Standardized Roles & Keywords")
    r1c1, r1c2, r1c3 = st.columns([1.1, 1.5, 2.4])

    all_years = sorted(df["DECISION_YEAR"].dropna().unique(), reverse=True)
    sel_years = r1c1.multiselect("Fiscal Year(s)", all_years, default=[2025, 2026])

    all_states = sorted(df["WORKSITE_STATE"].dropna().unique().tolist())
    sel_states = r1c2.multiselect(
        "Worksite State(s) (Leave empty for All States)",
        options=all_states,
        default=["NJ", "NY"] if "NJ" in all_states and "NY" in all_states else []
    )

    all_sectors = sorted([s for s in df["NAICS_SECTOR"].dropna().unique().tolist() if s != "Unmapped / Other"])
    sel_sectors = r1c3.multiselect(
        "Industry NAICS Sector(s) (Leave empty for All Sectors)",
        options=all_sectors,
        default=[]
    )

    r2c1, r2c2, r2c3 = st.columns([2.2, 1.8, 1.0])
    top_soc_options = df["SOC_TITLE"].value_counts().head(75).index.tolist()
    sel_socs = r2c1.multiselect(
        "Standardized Occupation (SOC Title) — Multi-Select",
        options=top_soc_options,
        default=[]
    )
    role_keywords_raw = r2c2.text_input(
        "Role Keywords (Comma-separated, e.g. DATA, AI, ANALYST, SOFTWARE)",
        value="DATA, AI, MACHINE LEARNING"
    )
    everify_only = r2c3.checkbox("E-Verify Open Only (STEM OPT)", value=True)

    # Apply multi-value filters
    sub = df[df["DECISION_YEAR"].isin(sel_years)] if sel_years else df.copy()
    if sel_states:
        sub = sub[sub["WORKSITE_STATE"].isin(sel_states)]
    if sel_sectors:
        sub = sub[sub["NAICS_SECTOR"].isin(sel_sectors)]
    if sel_socs:
        sub = sub[sub["SOC_TITLE"].isin(sel_socs)]
    if role_keywords_raw.strip():
        kw_list = [k.strip().upper() for k in role_keywords_raw.split(",") if k.strip()]
        if kw_list:
            pattern = "|".join([re.escape(k) for k in kw_list])
            sub = sub[
                sub["SOC_TITLE"].str.contains(pattern, na=False, regex=True) |
                sub["JOB_TITLE"].str.contains(pattern, na=False, regex=True)
            ]
    if everify_only:
        sub = sub[sub["IS_EVERIFY_OPEN"] == 1]

    col_left, col_right = st.columns([1.3, 1.0])

    with col_left:
        if not sub.empty:
            top_comp = sub.groupby(["COMPANY", "NAICS_SECTOR", "WORKSITE_STATE", "SOC_TITLE"], as_index=False).agg(
                TOTAL_FILINGS=("TOTAL_FILINGS", "sum"),
                CERTIFIED_ROLES=("CERTIFIED_FILINGS", "sum"),
                F1_ENTRY_LEVEL_HIRES=("F1_ENTRY_LEVEL_HIRES", "sum"),
                USCIS_INITIAL_APPROVALS=("USCIS_INITIAL_H1B_APPROVALS", "max"),
                WEIGHTED_SAL_SUM=("WEIGHTED_SAL_NUM", "sum"),
                WEIGHTED_PREM_SUM=("WEIGHTED_PREM_NUM", "sum")
            )
            top_comp["AVG_SALARY"] = (top_comp["WEIGHTED_SAL_SUM"] / top_comp["TOTAL_FILINGS"]).round(0)
            top_comp["WAGE_PREMIUM"] = (top_comp["WEIGHTED_PREM_SUM"] / top_comp["TOTAL_FILINGS"]).round(0)
            top_comp = top_comp.drop(columns=["WEIGHTED_SAL_SUM", "WEIGHTED_PREM_SUM", "TOTAL_FILINGS"])
            top_comp = top_comp.sort_values(["F1_ENTRY_LEVEL_HIRES", "CERTIFIED_ROLES"], ascending=False).head(20)

            state_label = ", ".join(sel_states) if sel_states else "All US States"
            fig_bar = px.bar(
                top_comp.sort_values("F1_ENTRY_LEVEL_HIRES", ascending=True),
                x="F1_ENTRY_LEVEL_HIRES",
                y="COMPANY",
                color="AVG_SALARY",
                orientation="h",
                hover_data=["SOC_TITLE", "NAICS_SECTOR", "WORKSITE_STATE", "CERTIFIED_ROLES", "WAGE_PREMIUM"],
                title=f"Top 20 Sponsors by Entry-Level (Wage Level I/II) New Hires ({state_label})",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_bar, width="stretch")
        else:
            top_comp = pd.DataFrame()
            st.warning("No matching filings found for the selected filter combination.")

    with col_right:
        if not sub.empty:
            state_map_df = sub.groupby("WORKSITE_STATE", as_index=False).agg(
                TOTAL_FILINGS=("TOTAL_FILINGS", "sum"),
                F1_ENTRY_HIRES=("F1_ENTRY_LEVEL_HIRES", "sum"),
                CERTIFIED_FILINGS=("CERTIFIED_FILINGS", "sum"),
                WEIGHTED_SAL_SUM=("WEIGHTED_SAL_NUM", "sum")
            )
            state_map_df["AVG_SALARY"] = (state_map_df["WEIGHTED_SAL_SUM"] / state_map_df["TOTAL_FILINGS"]).round(0)
            fig_map = px.choropleth(
                state_map_df,
                locations="WORKSITE_STATE",
                locationmode="USA-states",
                color="F1_ENTRY_HIRES",
                hover_data=["CERTIFIED_FILINGS", "AVG_SALARY"],
                scope="usa",
                title="Geographic Concentration of Selected Roles",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_map, width="stretch")

    st.subheader("Detailed Company, Role & Filing-Weighted Salary Breakdown")
    if not top_comp.empty:
        st.dataframe(top_comp, width="stretch")

# ==============================================================================
# TAB 2: Multi-Entity Text-to-SQL Assistant with Deterministic Guardrails
# ==============================================================================
with tab2:
    st.subheader("Query the Warehouse in Plain English or DuckDB SQL")
    st.markdown(
        "Supports **multi-state, multi-role, multi-year, and salary floor** parsing with deterministic read-only SQL validation. "
        "Try: *'Compare DATA, AI, and SOFTWARE roles in NJ, NY, and CA in 2025 and 2026 paying over 120000'*."
    )

    nl_q = st.text_input(
        "Ask a question in plain English:",
        value="Which E-Verify companies in NJ and NY hired the most entry-level DATA and AI roles in 2025 and 2026?"
    )

    def nl_to_sql(question: str) -> str:
        q_up = question.upper()
        where_clauses = ["IS_EVERIFY_OPEN = 1"]

        # 1. Detect ALL mentioned states (both full names and 2-letter codes)
        state_map = {
            "NEW JERSEY": "NJ", "NEW YORK": "NY", "CALIFORNIA": "CA", "TEXAS": "TX",
            "WASHINGTON": "WA", "MASSACHUSETTS": "MA", "ILLINOIS": "IL", "VIRGINIA": "VA",
            "GEORGIA": "GA", "PENNSYLVANIA": "PA", "FLORIDA": "FL", "NORTH CAROLINA": "NC",
            "OHIO": "OH", "MICHIGAN": "MI", "ARIZONA": "AZ", "MARYLAND": "MD", "COLORADO": "CO"
        }
        detected_states = set()
        for s_name, s_code in state_map.items():
            if s_name in q_up:
                detected_states.add(s_code)

        valid_states = set(df["WORKSITE_STATE"].dropna().unique())
        stop_words = {"IN", "OR", "BY", "TO", "ON", "AT", "IS", "AI", "IT", "US", "AS", "AN", "OF", "DO", "NO", "HI", "ME", "OK", "DE", "LA"}
        for token in re.findall(r"\b[A-Z]{2}\b", q_up):
            if token in valid_states and token not in stop_words:
                detected_states.add(token)

        if detected_states:
            st_list = ", ".join([f"'{s}'" for s in sorted(detected_states)])
            where_clauses.append(f"WORKSITE_STATE IN ({st_list})")

        # 2. Detect ALL mentioned years
        years_found = [y for y in ["2024", "2025", "2026"] if y in q_up]
        if years_found:
            where_clauses.append(f"DECISION_YEAR IN ({', '.join(years_found)})")

        # 3. Detect ALL mentioned role keywords (no early break!)
        role_Cues = ["DATA", "SOFTWARE", "ANALYST", "MACHINE LEARNING", "AI", "MEDICAL", "FINANCE", "RESEARCH", "CLOUD", "SECURITY", "STATISTIC", "BI"]
        matched_kws = [kw for kw in role_Cues if re.search(rf"\b{kw}\b", q_up)]
        if matched_kws:
            or_parts = [f"(SOC_TITLE LIKE '%{kw}%' OR JOB_TITLE LIKE '%{kw}%')" for kw in matched_kws]
            where_clauses.append("(" + " OR ".join(or_parts) + ")")

        # 4. Detect minimum salary threshold if mentioned (e.g., "over 120000")
        sal_match = re.search(r"(?:OVER|ABOVE|MORE THAN|>)\s*\$?(\d{5,6})", q_up)
        if sal_match:
            where_clauses.append(f"AVG_OFFERED_SALARY >= {int(sal_match.group(1))}")

        where_str = "\n  AND ".join(where_clauses)
        return f"""SELECT 
    COMPANY,
    NAICS_SECTOR,
    WORKSITE_STATE AS STATE,
    SOC_TITLE,
    SUM(CERTIFIED_FILINGS) AS CERTIFIED_FILINGS,
    SUM(F1_ENTRY_LEVEL_HIRES) AS F1_ENTRY_HIRES,
    ROUND(SUM(TOTAL_FILINGS * AVG_OFFERED_SALARY) / NULLIF(SUM(TOTAL_FILINGS), 0), 0) AS WEIGHTED_AVG_SALARY,
    ROUND(SUM(TOTAL_FILINGS * AVG_WAGE_PREMIUM) / NULLIF(SUM(TOTAL_FILINGS), 0), 0) AS WEIGHTED_WAGE_PREMIUM
FROM df
WHERE {where_str}
GROUP BY COMPANY, NAICS_SECTOR, WORKSITE_STATE, SOC_TITLE
ORDER BY F1_ENTRY_HIRES DESC, CERTIFIED_FILINGS DESC
LIMIT 25;"""

    def validate_sql_query(sql_str: str):
        clean = sql_str.strip()
        if not clean.upper().startswith("SELECT"):
            return False, "Guardrail Violation: Only read-only SELECT queries are permitted."
        forbidden = ["DROP ", "DELETE ", "UPDATE ", "INSERT ", "ALTER ", "TRUNCATE ", "CREATE ", "COPY ", "ATTACH "]
        for word in forbidden:
            if word in clean.upper():
                return False, f"Guardrail Violation: Keyword '{word.strip()}' is blocked."
        return True, "Valid read-only query."

    generated_sql = nl_to_sql(nl_q)
    sql_box = st.text_area("Generated SQL Query (Auditable & Editable):", value=generated_sql, height=210)

    if st.button("Validate & Run SQL Query"):
        is_valid, msg = validate_sql_query(sql_box)
        if not is_valid:
            st.error(msg)
        else:
            try:
                res_df = duckdb.query(sql_box).to_df()
                st.success(f"Guardrail check passed. Returned {len(res_df)} rows (Filing-Weighted Salary Aggregation).")
                st.dataframe(res_df, width="stretch")
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")

# ==============================================================================
# TAB 3: Statistical & ML Inference Lab
# ==============================================================================
with tab3:
    st.subheader("Statistical Inference, Econometrics & Machine Learning Results")
    s1, s2 = st.columns(2)

    with s1:
        st.markdown("#### 1. Bootstrapped 95% CIs & Benjamini-Hochberg FDR (Top 15 States)")
        if not fdr_df.empty:
            fdr_display = fdr_df.copy()
            fdr_display["MEDIAN_SALARY"] = fdr_display["MEDIAN_SALARY"].round(0)
            fdr_display["BOOT_95_CI_LOW"] = fdr_display["BOOT_95_CI_LOW"].round(0)
            fdr_display["BOOT_95_CI_HIGH"] = fdr_display["BOOT_95_CI_HIGH"].round(0)
            fdr_display["RAW_P_VAL"] = fdr_display["RAW_P_VAL"].apply(lambda p: "< 1e-16" if p < 1e-16 else f"{p:.2e}")
            fdr_display["BH_FDR_ADJ_P"] = fdr_display["BH_FDR_ADJ_P"].apply(lambda p: "< 1e-16" if p < 1e-16 else f"{p:.2e}")

            fig_fdr = px.scatter(
                fdr_df,
                x="STATE",
                y="MEDIAN_SALARY",
                color="SIGNIFICANT_FDR_05",
                error_y=fdr_df["BOOT_95_CI_HIGH"] - fdr_df["MEDIAN_SALARY"],
                error_y_minus=fdr_df["MEDIAN_SALARY"] - fdr_df["BOOT_95_CI_LOW"],
                title="State Median H-1B Salaries with Percentile Bootstrap 95% CIs (FDR alpha=0.05)"
            )
            st.plotly_chart(fig_fdr, width="stretch")
            st.dataframe(fdr_display, width="stretch")

    with s2:
        st.markdown("#### 2. PCA Dimensionality Reduction (81.9% Variance Explained)")
        if not pca_df.empty:
            pca_clean = pca_df[pca_df["NAICS_SECTOR"] != "Unmapped / Other"].copy()
            fig_pca = px.scatter(
                pca_clean,
                x="PCA_1",
                y="PCA_2",
                color="NAICS_SECTOR",
                size="TOTAL",
                hover_name="EMPLOYER_CLEAN_KEY",
                hover_data=["AVG_SALARY", "F1_ENTRY_HIRES", "BAYESIAN_SHRUNK_APPROVAL_RATE"],
                title="PCA Clustering of Top Employers by Sponsorship Profile"
            )
            st.plotly_chart(fig_pca, width="stretch")

        if not bayes_df.empty:
            st.markdown("#### 3. Empirical Bayes Beta-Binomial Sponsor Reliability Rankings")
            b_clean = bayes_df[bayes_df["NAICS_SECTOR"] != "Unmapped / Other"].head(15).copy()
            b_clean["AVG_SALARY"] = b_clean["AVG_SALARY"].round(0)
            b_clean["BAYESIAN_SHRUNK_APPROVAL_RATE"] = (b_clean["BAYESIAN_SHRUNK_APPROVAL_RATE"] * 100).round(2).astype(str) + "%"
            st.dataframe(
                b_clean[["EMPLOYER_CLEAN_KEY", "NAICS_SECTOR", "TOTAL", "F1_ENTRY_HIRES", "AVG_SALARY", "BAYESIAN_SHRUNK_APPROVAL_RATE"]],
                width="stretch"
            )

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
        "- **XGBoost Salary Predictor $R^2$:** `0.6859` (`MAE: $18,044`)\n"
        "- **Polynomial GLM Adjusted $R^2$:** `0.6505`\n"
        "- **ETS Time Series MAE:** `47,834` filings (outperforming ARIMA `83,635` MAE by **42.8%**)."
    )
    m3.info(
        "**Two-Way ANOVA & Role Drift (FY24 vs FY26)**\n\n"
        "- **NAICS Sector Effect:** $F$-test $p < 10^{-16}$\n"
        "- **Prevailing Wage Level Effect:** $F$-test $p < 10^{-16}$\n"
        "- **Population Stability Index (PSI):** `0.1389` (`KL Div = 0.0717`), confirming structural shift toward AI & Data roles."
    )

# ==============================================================================
# TAB 4: Student Salary & Target Sponsor Calculator (Multi-State & Multi-Role)
# ==============================================================================
with tab4:
    st.subheader("Expected Offer, Prevailing Wage Benchmark & Target Sponsor Finder")
    calc_s1, calc_s2 = st.columns(2)
    c_states = calc_s1.multiselect(
        "Select Target State(s)",
        sorted(df["WORKSITE_STATE"].dropna().unique()),
        default=["NJ", "NY"] if "NJ" in df["WORKSITE_STATE"].values else []
    )
    c_roles = calc_s2.multiselect(
        "Select Standardized SOC Role(s)",
        df["SOC_TITLE"].value_counts().head(50).index.tolist(),
        default=["DATA SCIENTISTS"] if "DATA SCIENTISTS" in df["SOC_TITLE"].values else []
    )

    match = df[df["IS_EVERIFY_OPEN"] == 1].copy()
    if c_states:
        match = match[match["WORKSITE_STATE"].isin(c_states)]
    if c_roles:
        match = match[match["SOC_TITLE"].isin(c_roles)]

    if not match.empty and match["TOTAL_FILINGS"].sum() > 0:
        tot_f = match["TOTAL_FILINGS"].sum()
        w_sal = match["WEIGHTED_SAL_NUM"].sum() / tot_f
        w_pw = match["WEIGHTED_PW_NUM"].sum() / tot_f
        w_prem = w_sal - w_pw
        p25, p50, p75 = np.percentile(match["AVG_OFFERED_SALARY"], [25, 50, 75])

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Filing-Weighted Expected Salary", f"${w_sal:,.0f}", f"Median: ${p50:,.0f}")
        sc2.metric("Prevailing Wage Baseline", f"${w_pw:,.0f}", f"25th–75th: ${p25:,.0f}–${p75:,.0f}")
        sc3.metric("Employer Wage Premium", f"+${w_prem:,.0f}", f"+{(w_prem/w_pw)*100:.1f}% above PW")
        sc4.metric("F-1 Entry-Level Hires (2024–2026)", f"{int(match['F1_ENTRY_LEVEL_HIRES'].sum()):,}", f"{int(match['CERTIFIED_FILINGS'].sum()):,} total certified")

        st.markdown("#### Top 15 Active E-Verify Sponsors for Your Selected State(s) & Role(s)")
        rec_sponsors = match.groupby(["COMPANY", "NAICS_SECTOR", "WORKSITE_STATE", "SOC_TITLE"], as_index=False).agg(
            TOTAL_FILINGS=("TOTAL_FILINGS", "sum"),
            CERTIFIED_FILINGS=("CERTIFIED_FILINGS", "sum"),
            F1_ENTRY_LEVEL_HIRES=("F1_ENTRY_LEVEL_HIRES", "sum"),
            WEIGHTED_SAL_SUM=("WEIGHTED_SAL_NUM", "sum"),
            WEIGHTED_PREM_SUM=("WEIGHTED_PREM_NUM", "sum")
        )
        rec_sponsors["EXPECTED_OFFER_SALARY"] = (rec_sponsors["WEIGHTED_SAL_SUM"] / rec_sponsors["TOTAL_FILINGS"]).round(0)
        rec_sponsors["WAGE_PREMIUM_OVER_PW"] = (rec_sponsors["WEIGHTED_PREM_SUM"] / rec_sponsors["TOTAL_FILINGS"]).round(0)
        rec_sponsors = rec_sponsors.drop(columns=["WEIGHTED_SAL_SUM", "WEIGHTED_PREM_SUM", "TOTAL_FILINGS"])
        rec_sponsors = rec_sponsors.sort_values(["F1_ENTRY_LEVEL_HIRES", "CERTIFIED_FILINGS"], ascending=False).head(15)
        st.dataframe(rec_sponsors, width="stretch")
    else:
        st.warning("Select at least one State or Role with active E-Verify filings to view salary benchmarks.")