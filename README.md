# US H-1B & F-1 STEM OPT Market Intelligence Platform (FY 2024–2026)

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

## 3. Core Statistical, Econometric & Machine Learning Findings

### A. Parametric Modeling & Variance Decomposition
- **Maximum Likelihood Estimation (MLE):** Fitted a Log-Normal distribution to right-skewed annualized H-1B salaries ($N = 1,238,350$), obtaining $\hat{\mu}\_{\text{MLE}} = 11.6827$ and $\hat{\sigma}\_{\text{MLE}} = 0.3899$.
- **Law of Total Variance ($`\text{Var}(Y) = \mathbb{E}[\text{Var}(Y \mid X)] + \text{Var}(\mathbb{E}[Y \mid X])`$):** Decomposed total salary variance ($3.02 \times 10^9$) into **Within-State Mean Variance ($4.06 \times 10^9$)** versus **Between-State Variance ($3.81 \times 10^8$)**. Within-state variance is **10.6x larger** than between-state variance, proving that **company tier, `NAICS` industry, and `SOC` role** drive compensation far more than state geography alone.

### B. Two-Way ANOVA, Non-Parametric Bootstrapping & FDR Control (Python & R)
- **Two-Way ANOVA (`statsmodels` & R `aov()`):** Tested main effects of `NAICS_SECTOR` ($SS = 2.13 \times 10^{13}, p \lt 10^{-16}$) and `PW_WAGE_LEVEL` ($SS = 5.55 \times 10^{13}, p \lt 10^{-16}$) on offered compensation, alongside Benjamini-Hochberg adjusted pairwise $t$-tests comparing **Healthcare (`NAICS 62`)** against **Information & Tech Services (`NAICS 51/54`)**.
- **Percentile Bootstrapping & Benjamini-Hochberg (FDR):** Executed Mann-Whitney $U$ tests and 500-iteration Percentile Bootstrapping across the top 15 worksite states with **Benjamini-Hochberg False Discovery Rate ($\alpha = 0.05$)** control:
  - **Washington (`WA`):** Median `$155,000` (95% Bootstrap CI: `[$151,671, $158,947]`, $p\_{\text{BH}} \lt 10^{-16}$)
  - **New York (`NY`):** Median `$128,211` (95% Bootstrap CI: `[$123,656, $133,068]`, $p\_{\text{BH}} \lt 10^{-16}$)
  - **New Jersey (`NJ`):** Median `$115,918` (95% Bootstrap CI: `[$111,823, $117,575]`, $p\_{\text{BH}} \lt 10^{-16}$)

### C. Distribution Drift, Empirical Bayes Shrinkage & PCA
- **Labor Market Drift (`FY 2024` to `FY 2026`):** Computed **Population Stability Index ($\text{PSI} = 0.1389$)** and **Kullback-Leibler Divergence ($D\_{\text{KL}} = 0.0717$)** across the top 30 occupation titles, quantifying the structural hiring shift toward AI, Data Science, and Cloud Infrastructure roles.
- **Empirical Bayes Beta-Binomial Shrinkage:** Applied a conjugate $\text{Beta}(\alpha\_0 = 95, \beta\_0 = 5)$ prior to shrink noisy small-sample employer approval rates toward the industry baseline: $\displaystyle \hat{\theta}\_i = \frac{c\_i + \alpha\_0}{n\_i + \alpha\_0 + \beta\_0}$
- **Dimensionality Reduction (PCA):** Compressed 4 standardized employer metrics (`Total Filings`, `Mean Salary`, `F-1 Entry Hires`, `Bayesian Approval Rate`) across the top 300 sponsors into 2 principal components capturing **81.9% of total variance**.

### D. Predictive Modeling & Time Series Forecasting
- **Salary Benchmarking (`XGBoost` vs. `Polynomial GLM`):** Trained a degree-2 Polynomial OLS model ($\text{Adjusted } R^2 = 0.6505$) and a gradient-boosted **XGBoost Regressor ($R^2 = 0.6859$, $\text{MAE} = \text{USD } 18,044$)** to predict expected compensation and prevailing wage premiums.
- **Time Series Forecasting (`ETS` vs. `ARIMA`):** Modeled quarterly filing volume using **Holt-Winters Exponential Smoothing ($\text{ETS MAE} = 47,834$)** vs. **ARIMA(1,1,0) ($\text{MAE} = 83,635$)**, reducing out-of-sample forecast error by **42.8%**.

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
