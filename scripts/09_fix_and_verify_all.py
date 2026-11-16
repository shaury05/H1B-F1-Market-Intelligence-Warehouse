import os
import re
import subprocess
import sys
import pandas as pd

BASE_DIR = os.path.expanduser("~/Desktop/H1B_Market_Intelligence")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
HF_DIR = os.path.join(BASE_DIR, "hf_space")
HF_DATA_DIR = os.path.join(HF_DIR, "data")
GRADIO_DATA_DIR = os.path.join(BASE_DIR, "hf_gradio/data")

VALID_ORATION_WORDS = {
    "EXPLORATION", "RESTORATION", "COLLABORATION", "INCORPORATION",
    "DECORATION", "ELABORATION", "AMELIORATION", "CORPORATION", "ORATIONS"
}

def clean_corrupted_oration(name: str) -> str:
    if not isinstance(name, str):
        return name
    tokens = name.split()
    cleaned_tokens = []
    for tok in tokens:
        if "ORATION" in tok and tok not in VALID_ORATION_WORDS:
            # Fix strings like MICROSOFTORATION -> MICROSOFT or XYZORATION -> XYZ
            tok = re.sub(r"ORATIONS?$", "", tok)
        cleaned_tokens.append(tok)
    res = " ".join([t for t in cleaned_tokens if t])
    return res.strip()

print("1. Cleaning company suffix edge cases across Parquet and CSV files...")
for folder in [OUT_DIR, HF_DATA_DIR, GRADIO_DATA_DIR]:
    if not os.path.exists(folder):
        continue
    pq_path = os.path.join(folder, "h1b_cube.parquet")
    if os.path.exists(pq_path):
        df_pq = pd.read_parquet(pq_path)
        df_pq["COMPANY"] = df_pq["COMPANY"].apply(clean_corrupted_oration)
        df_pq.to_parquet(pq_path, index=False)

    for csv_name in ["top_500_employers_bayesian_ranked.csv", "employer_pca_clusters.csv"]:
        c_path = os.path.join(folder, csv_name)
        if os.path.exists(c_path):
            df_c = pd.read_csv(c_path)
            col = "EMPLOYER_CLEAN_KEY" if "EMPLOYER_CLEAN_KEY" in df_c.columns else "COMPANY"
            if col in df_c.columns:
                df_c[col] = df_c[col].apply(clean_corrupted_oration)
                df_c.to_csv(c_path, index=False)

# Remove duplicate large CSV inside hf_space/data so GitHub repo stays fast and light
dup_csv = os.path.join(HF_DATA_DIR, "master_h1b_cube_2024_2026.csv")
if os.path.exists(dup_csv):
    os.remove(dup_csv)
    print("   -> Removed redundant hf_space/data/master_h1b_cube_2024_2026.csv")

print("2. Updating scripts/08_verify_pipeline_spec.py with exact word-boundary check...")
spec_code = '''"""
Deterministic Specification & Ground-Truth Audit Suite
Validates data warehouse schema invariants, statistical bounds, and weighted-metric consistency.
"""
import os
import re
import sys
import pandas as pd

BASE_DIR = os.path.expanduser("~/Desktop/H1B_Market_Intelligence")
CUBE_PATH = os.path.join(BASE_DIR, "hf_space/data/h1b_cube.parquet")
FDR_PATH = os.path.join(BASE_DIR, "hf_space/data/state_bootstrap_fdr_results.csv")
PCA_PATH = os.path.join(BASE_DIR, "hf_space/data/employer_pca_clusters.csv")

VALID_ORATION_WORDS = {
    "EXPLORATION", "RESTORATION", "COLLABORATION", "INCORPORATION",
    "DECORATION", "ELABORATION", "AMELIORATION", "CORPORATION", "ORATIONS"
}

def has_corrupted_oration(name: str) -> bool:
    if not isinstance(name, str):
        return False
    for tok in name.split():
        if "ORATION" in tok and tok not in VALID_ORATION_WORDS:
            return True
    return False

def run_spec_audit():
    print("=== Running Deterministic Data Warehouse & Statistical Spec Audit ===")
    assert os.path.exists(CUBE_PATH), f"CRITICAL: Missing {CUBE_PATH}"
    df = pd.read_parquet(CUBE_PATH)

    # 1. Schema & Domain Invariants
    required_cols = {
        "DECISION_YEAR", "WORKSITE_STATE", "NAICS_SECTOR", "SOC_TITLE",
        "JOB_TITLE", "COMPANY", "IS_EVERIFY_OPEN", "TOTAL_FILINGS",
        "CERTIFIED_FILINGS", "F1_ENTRY_LEVEL_HIRES", "AVG_OFFERED_SALARY",
        "AVG_PREVAILING_WAGE", "AVG_WAGE_PREMIUM"
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Spec Failure: Missing columns {missing}"
    print(f"[PASS] Schema Contract Verified ({len(df):,} aggregated cube records).")

    # 2. Temporal & Salary Range Invariants
    assert set(df["DECISION_YEAR"].unique()).issubset({2024, 2025, 2026}), "Spec Failure: Out-of-bounds fiscal years."
    assert df["AVG_OFFERED_SALARY"].min() >= 40000 and df["AVG_OFFERED_SALARY"].max() <= 600000, "Spec Failure: Salary bounds violated."
    assert (df["WORKSITE_STATE"].str.len() == 2).all(), "Spec Failure: Non-standardized state codes detected."
    print("[PASS] Temporal (FY 2024-2026), Geographic (2-char State), and Salary ($40K-$600K) Invariants Verified.")

    # 3. Entity Resolution Invariant (Confirming no corrupted 'ORATION' suffixes)
    bad_corps = df[df["COMPANY"].apply(has_corrupted_oration)]
    assert len(bad_corps) == 0, f"Spec Failure: Found {len(bad_corps)} corrupted company suffixes."
    print("[PASS] Entity Resolution Suffix Invariant Verified (0 corrupted 'ORATION' strings).")

    # 4. Statistical Output Verification
    fdr = pd.read_csv(FDR_PATH)
    assert (fdr["BOOT_95_CI_LOW"] <= fdr["MEDIAN_SALARY"]).all(), "Spec Failure: Bootstrap CI Low > Median."
    assert (fdr["MEDIAN_SALARY"] <= fdr["BOOT_95_CI_HIGH"]).all(), "Spec Failure: Bootstrap CI High < Median."
    assert (fdr["BH_FDR_ADJ_P"] >= fdr["RAW_P_VAL"] - 1e-12).all(), "Spec Failure: BH-FDR p-value < raw p-value."
    print("[PASS] Percentile Bootstrap 95% CIs & Benjamini-Hochberg FDR Monotonicity Verified.")

    pca = pd.read_csv(PCA_PATH)
    assert "PCA_1" in pca.columns and "PCA_2" in pca.columns, "Spec Failure: Missing PCA coordinates."
    print("[PASS] PCA Dimensionality Reduction Output Verified.")
    print("\\nALL SPECIFICATION CHECKS PASSED (100% COMPLIANT)!")

if __name__ == "__main__":
    try:
        run_spec_audit()
    except AssertionError as err:
        print(f"[AUDIT FAILED]: {err}")
        sys.exit(1)
'''
with open(os.path.join(BASE_DIR, "scripts/08_verify_pipeline_spec.py"), "w", encoding="utf-8") as f:
    f.write(spec_code)

print("3. Writing complete portfolio-grade README.md...")
readme_md = """# US H-1B & F-1 STEM OPT Market Intelligence Platform (FY 2024–2026)

[![Live Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://h1b-f1-market-intelligence.streamlit.app)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![MS SQL Server](https://img.shields.io/badge/MS_SQL_Server-T--SQL_Warehouse-CC2927?logo=microsoftsqlserver&logoColor=white)
![R Statistical Computing](https://img.shields.io/badge/R-ANOVA_%26_GLM-276DC3?logo=r&logoColor=white)
![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory_OLAP-FFF000?logo=duckdb&logoColor=black)

**Author:** Shaury Pratap Singh (M.S. in Data Science, New Jersey Institute of Technology)  
**GitHub Repository:** [github.com/shaury05/H1B-F1-Market-Intelligence-Warehouse](https://github.com/shaury05/H1B-F1-Market-Intelligence-Warehouse)  
**Executive Audit Workbook:** [`outputs/H1B_Executive_Audit_Workbook.xlsx`](outputs/H1B_Executive_Audit_Workbook.xlsx)

---

## 1. Project Inspiration & Problem Statement
Every year, hundreds of thousands of F-1 international students and skilled professionals navigate the U.S. job market using fragmented public lookup sites (such as H1BGrader) and raw government portals. While exploring these platforms during my own Master's job search, I identified five structural blind spots:

1. **Broken Geographic Aggregation:** Public portals index locations strictly by raw city text rather than state-level rollups (e.g., querying "New Jersey" on H1BGrader returns only ~20 filings where an employer literally typed "New Jersey" in the city field, missing 3,600+ filings in Jersey City, Newark, and Princeton).
2. **Zero Cross-Portal Entity Resolution:** No existing platform links **DOL LCA role & wage filings** with **USCIS `NAICS` industry classifications** and **active `E-Verify` enrollment status** (which is legally mandatory for 24-month F-1 STEM OPT extensions).
3. **No Entry-Level International Student Signal:** Public portals lump senior executive transfers (`Wage Level IV`) together with new college hires. Isolating true entry-level international student hiring requires cross-filtering **DOL `NEW_EMPLOYMENT > 0` + `Prevailing Wage Level I & II`** alongside **USCIS Initial Cap Approvals**.
4. **Single-Value Filter Limitations:** Job seekers rarely target a single job title or state; they need multi-select filtering across adjacent states (e.g., `NJ`, `NY`, `PA`, `MA`), multiple industries (`Tech`, `Finance`, `Healthcare`, `Manufacturing`), and overlapping role families (`Data Scientist`, `AI/ML Engineer`, `Business Intelligence Analyst`).
5. **Unverified Averages & Lack of Statistical Rigor:** Simple averages distort salary expectations due to extreme right-skewness and unweighted aggregation across unequal filing volumes.

---

## 2. System Architecture & Spec-Driven Data Engineering (2.1M+ Federal Records)
To solve this at enterprise scale, I architected a multi-tier data warehouse in **MS SQL Server (Docker)** unifying **2,095,550 federal records** across four U.S. government disclosures (`FY 2024 – FY 2026 Q3`), backed by a deterministic specification verification suite (`scripts/08_verify_pipeline_spec.py`).

    +---------------------------------------------------------------------------------------------------+
    |                        TIER 1: FEDERAL DISCLOSURE SOURCES (FY 2024 - FY 2026 Q3)                  |
    |  1. DOL OFLC LCA Filings (9 Quarterly XLSX)      3. USCIS E-Verify Participating Employers (CSV)  |
    |  2. USCIS H-1B Employer Data Hub (CSV)           4. DOL OFLC PERM Disclosure Data (XLSX)          |
    +-------------------------------------------------+-------------------------------------------------+
                                                      |
                                                      v  [Rust Calamine Engine + UTF-16 Auto-Parser]
    +---------------------------------------------------------------------------------------------------+
    |                   TIER 2: DOCKERIZED MS SQL SERVER DATA WAREHOUSE (2,095,550 ROWS)                |
    |  * dbo.Fact_LCA_Filings (1,372,949 unique deduplicated cases via ROW_NUMBER() PARTITION BY CASE)  |
    |  * dbo.Stg_EVerify_Employers (461,297 records) -> dbo.Dim_Company_EVerify (Active 'Open' Status)  |
    |  * dbo.Stg_USCIS_H1B (148,754 records)         -> dbo.Dim_Company_NAICS_USCIS (Primary NAICS)     |
    |  * dbo.Fact_PERM_Filings (112,550 records)     -> Degree & Special Skills Extraction              |
    |  * Master Star View: dbo.vw_Master_H1B_Intelligence (Entity-Resolved & Annualized Wage Normalized)|
    +-------------------------------------------------+-------------------------------------------------+
                                                      |
                      +-------------------------------+-------------------------------+
                      v                                                               v
    +---------------------------------------------------+   +-------------------------------------------+
    |   TIER 3: STATISTICAL & ML ENGINE (PYTHON & R)    |   | TIER 4: BI & INTERACTIVE CLOUD DELIVERY   |
    |  * Log-Normal MLE & Law of Total Variance         |   | * Multi-Select Geospatial Explorer        |
    |  * Two-Way ANOVA (Python Statsmodels & R aov)     |   | * Deterministic Text-to-SQL DuckDB Agent  |
    |  * Percentile Bootstrap + Benjamini-Hochberg FDR  |   | * Student Salary & Sponsor Calculator     |
    |  * Population Stability Index (PSI) & KL Drift    |   | * Multi-Sheet Excel Audit Workbook        |
    |  * Empirical Bayes Beta-Binomial Shrinkage        |   | * Automated Spec Verification Suite       |
    |  * XGBoost Regressor, Poly GLM, ETS vs. ARIMA     |   |   (scripts/08_verify_pipeline_spec.py)    |
    +---------------------------------------------------+   +-------------------------------------------+

---

## 3. Core Statistical, Econometric & Machine Learning Findings

### A. Parametric Modeling & Variance Decomposition
* **Maximum Likelihood Estimation (MLE):** Fitted a Log-Normal distribution to right-skewed annualized H-1B salaries ($N = 1,238,350$), obtaining $\\hat{\\mu}_{\\text{MLE}} = 11.6827$ and $\\hat{\\sigma}_{\\text{MLE}} = 0.3899$.
* **Law of Total Variance ($\\text{Var}(Y) = \\mathbb{E}[\\text{Var}(Y \\mid X)] + \\text{Var}(\\mathbb{E}[Y \\mid X])$):** Decomposed total salary variance ($3.02 \\times 10^9$) into **Within-State Mean Variance ($4.06 \\times 10^9$)** versus **Between-State Variance ($3.81 \\times 10^8$)**. Within-state variance is **10.6x larger** than between-state variance, proving that **company tier, `NAICS` industry, and `SOC` role** drive compensation far more than state geography alone.

### B. Two-Way ANOVA, Non-Parametric Bootstrapping & FDR Control (Python & R)
* **Two-Way ANOVA (`statsmodels` & R `aov()`):** Tested main effects of `NAICS_SECTOR` ($SS = 2.13 \\times 10^{13}, p < 10^{-16}$) and `PW_WAGE_LEVEL` ($SS = 5.55 \\times 10^{13}, p < 10^{-16}$) on offered compensation, alongside Benjamini-Hochberg adjusted pairwise $t$-tests comparing **Healthcare (`NAICS 62`)** against **Information & Tech Services (`NAICS 51/54`)**.
* **Percentile Bootstrapping & Benjamini-Hochberg (FDR):** Executed Mann-Whitney $U$ tests and 500-iteration Percentile Bootstrapping across the top 15 worksite states with **Benjamini-Hochberg False Discovery Rate ($\\alpha = 0.05$)** control:
  * **Washington (`WA`):** Median `$155,000` (95% Bootstrap CI: `[$151,671, $158,947]`, $p_{\\text{BH}} < 10^{-16}$)
  * **New York (`NY`):** Median `$128,211` (95% Bootstrap CI: `[$123,656, $133,068]`, $p_{\\text{BH}} < 10^{-16}$)
  * **New Jersey (`NJ`):** Median `$115,918` (95% Bootstrap CI: `[$111,823, $117,575]`, $p_{\\text{BH}} < 10^{-16}$)

### C. Distribution Drift, Empirical Bayes Shrinkage & PCA
* **Labor Market Drift (`FY 2024` to `FY 2026`):** Computed **Population Stability Index ($\\text{PSI} = 0.1389$)** and **Kullback-Leibler Divergence ($D_{\\text{KL}} = 0.0717$)** across the top 30 occupation titles, quantifying the structural hiring shift toward AI, Data Science, and Cloud Infrastructure roles.
* **Empirical Bayes Beta-Binomial Shrinkage:** Applied a conjugate $\\text{Beta}(\\alpha_0 = 95, \\beta_0 = 5)$ prior to shrink noisy small-sample employer approval rates toward the industry baseline:
  $$\\hat{\\theta}_i = \\frac{c_i + \\alpha_0}{n_i + \\alpha_0 + \\beta_0}$$
* **Dimensionality Reduction (PCA):** Compressed 4 standardized employer metrics (`Total Filings`, `Mean Salary`, `F-1 Entry Hires`, `Bayesian Approval Rate`) across the top 300 sponsors into 2 principal components capturing **81.9% of total variance**.

### D. Predictive Modeling & Time Series Forecasting
* **Salary Benchmarking (`XGBoost` vs. `Polynomial GLM`):** Trained a degree-2 Polynomial OLS model ($\\text{Adjusted } R^2 = 0.6505$) and a gradient-boosted **XGBoost Regressor ($R^2 = 0.6859, \\text{MAE} = \\$18,044$)** to predict expected compensation and prevailing wage premiums.
* **Time Series Forecasting (`ETS` vs. `ARIMA`):** Modeled quarterly filing volume using **Holt-Winters Exponential Smoothing ($\\text{ETS MAE} = 47,834$)** vs. **ARIMA(1,1,0) ($\\text{MAE} = 83,635$)**, reducing out-of-sample forecast error by **42.8%**.

---

## 4. Interactive Application Modules & Engineering Guardrails
1. **Tab 1 — Multi-Select Role, Company & State Explorer:** Supports simultaneous filtering across multiple states, multiple `NAICS` sectors, multiple standardized `SOC` occupations, and comma-separated role keywords using exact **filing-weighted salary aggregation**.
2. **Tab 2 — Deterministic Text-to-SQL Assistant:** Translates natural-language questions (including multi-state, multi-role, and salary-threshold queries) into executable **DuckDB SQL** with read-only syntax guardrails that block destructive operations.
3. **Tab 3 — Statistical & ML Inference Lab:** Interactive Plotly visualizations for state Bootstrap 95% CIs, PCA employer clusters, and Empirical Bayes sponsor reliability rankings.
4. **Tab 4 — Student Salary & Target Sponsor Calculator:** Computes filing-weighted expected salary, 25th–75th percentile prevailing wage baselines, and the top 15 active E-Verify sponsors for any combination of target states and roles.

---

## 5. Repository Structure
    ├── hf_space/
    │   ├── app.py                                # 4-Tab Multi-Select Streamlit + DuckDB Web Application
    │   ├── requirements.txt                      # Cloud deployment dependencies
    │   └── data/                                 # Compressed Parquet OLAP cube & statistical lookup tables
    ├── outputs/
    │   ├── H1B_Executive_Audit_Workbook.xlsx     # 4-Sheet Executive Excel Workbook (Sponsors, States, FDR, ANOVA)
    │   ├── anova_naics_wage_results.csv          # Two-Way ANOVA output table
    │   ├── state_bootstrap_fdr_results.csv       # Bootstrap 95% CIs & Benjamini-Hochberg FDR table
    │   ├── top_500_employers_bayesian_ranked.csv # Empirical Bayes Beta-Binomial sponsor rankings
    │   └── employer_pca_clusters.csv             # PCA dimensionality reduction coordinates
    └── scripts/
        ├── 01_prepare_staging_csvs.py            # Rust calamine Excel-to-CSV ingestion & CASE_NUMBER deduplication
        ├── 02_load_to_mssql.py                   # Automated chunked loader into Dockerized MS SQL Server
        ├── 03_build_sql_warehouse.py             # T-SQL Star Schema views & cross-portal entity resolution
        ├── 04_stats_ml_pipeline.py               # MLE, ANOVA, Bootstrap/FDR, PSI/KL, XGBoost, ETS/ARIMA, PCA
        ├── 04_export_summary_cubes.py            # Multi-dimensional OLAP cube exporter
        ├── 05_prepare_hf_and_excel.py            # Parquet compressor & openpyxl Excel workbook generator
        ├── 06_r_anova_healthcare_tech.R          # R statistical modeling (Two-Way ANOVA, BH t-test, GLM)
        ├── 07_finalize_repo.py                   # Automated repository packaging utility
        └── 08_verify_pipeline_spec.py            # Deterministic data contract & ground-truth audit suite

---

## 6. Quick Start & Verification
    # 1. Run the deterministic data contract & statistical specification audit
    python scripts/08_verify_pipeline_spec.py

    # 2. Launch the 4-Tab Interactive Web Platform locally
    streamlit run hf_space/app.py
"""
with open(os.path.join(BASE_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_md)

print("4. Running scripts/08_verify_pipeline_spec.py...\n")
res = subprocess.run([sys.executable, os.path.join(BASE_DIR, "scripts/08_verify_pipeline_spec.py")])
sys.exit(res.returncode)