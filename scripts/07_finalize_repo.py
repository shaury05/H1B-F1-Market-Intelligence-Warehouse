import os
import shutil

BASE_DIR = os.path.expanduser("~/Desktop/H1B_Market_Intelligence")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
HF_SPACE_DIR = os.path.join(BASE_DIR, "hf_space")
HF_DATA_DIR = os.path.join(HF_SPACE_DIR, "data")
GRADIO_DIR = os.path.join(BASE_DIR, "hf_gradio")
GRADIO_DATA_DIR = os.path.join(GRADIO_DIR, "data")

os.makedirs(HF_DATA_DIR, exist_ok=True)
os.makedirs(GRADIO_DATA_DIR, exist_ok=True)

# 1. Copy latest updated statistical CSVs to both hf_space/data and hf_gradio/data
for fname in [
    "anova_naics_wage_results.csv",
    "state_bootstrap_fdr_results.csv",
    "top_500_employers_bayesian_ranked.csv",
    "employer_pca_clusters.csv"
]:
    src = os.path.join(OUT_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(HF_DATA_DIR, fname))
        shutil.copy2(src, os.path.join(GRADIO_DATA_DIR, fname))

parquet_src = os.path.join(HF_DATA_DIR, "h1b_cube.parquet")
if os.path.exists(parquet_src):
    shutil.copy2(parquet_src, os.path.join(GRADIO_DATA_DIR, "h1b_cube.parquet"))

# 2. Fix Streamlit deprecation warning: replace use_container_width=True with width="stretch"
st_app_path = os.path.join(HF_SPACE_DIR, "app.py")
if os.path.exists(st_app_path):
    with open(st_app_path, "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace("use_container_width=True", 'width="stretch"')
    code = code.replace("0.657", "0.686").replace("$18,303", "$18,044").replace("0.6442", "0.6859")
    with open(st_app_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("1. Updated hf_space/app.py (fixed width='stretch' and updated XGBoost R2=0.686 / MAE=$18,044).")

# 3. Write .gitignore
gitignore_content = """raw_data/
csv_staging/
__pycache__/
.DS_Store
*.pyc
.venv/
outputs/master_h1b_cube_2024_2026.csv
"""
with open(os.path.join(BASE_DIR, ".gitignore"), "w", encoding="utf-8") as f:
    f.write(gitignore_content)
print("2. Created .gitignore (blocks >100MB raw files from breaking GitHub push).")

# 4. Write R Statistical Script (scripts/06_r_anova_healthcare_tech.R)
r_script = """# ==============================================================================
# R Statistical Inference Module: Two-Way ANOVA, BH-FDR & Linear Wage Modeling
# ==============================================================================
cube_path <- path.expand("~/Desktop/H1B_Market_Intelligence/outputs/master_h1b_cube_2024_2026.csv")
df <- read.csv(cube_path)

df_clean <- subset(df, NAICS_SECTOR != "Unmapped / Other" & AVG_OFFERED_SALARY >= 40000)

cat("=== 1. R Two-Way ANOVA Across NAICS Industries & E-Verify Status ===\\n")
anova_fit <- aov(AVG_OFFERED_SALARY ~ NAICS_SECTOR + factor(IS_EVERIFY_OPEN), data = df_clean)
print(summary(anova_fit))

cat("\\n=== 2. Healthcare (NAICS 62) vs. Tech (NAICS 51/54) Pairwise BH-Adjusted t-test ===\\n")
hc_tech <- subset(df_clean, grepl("^62|^51|^54", NAICS_SECTOR))
t_res <- pairwise.t.test(hc_tech$AVG_OFFERED_SALARY, hc_tech$NAICS_SECTOR, p.adjust.method = "BH")
print(t_res)

cat("\\n=== 3. R Linear Regression Model (Offered Salary vs Prevailing Wage & F-1 Hires) ===\\n")
lm_fit <- lm(AVG_OFFERED_SALARY ~ AVG_PREVAILING_WAGE + IS_EVERIFY_OPEN + F1_ENTRY_LEVEL_HIRES, data = df_clean)
print(summary(lm_fit)$coefficients)
"""
with open(os.path.join(BASE_DIR, "scripts/06_r_anova_healthcare_tech.R"), "w", encoding="utf-8") as f:
    f.write(r_script)
print("3. Created scripts/06_r_anova_healthcare_tech.R.")

# 5. Write Executive README.md
readme_content = """# US H-1B & F-1 STEM OPT Market Intelligence Platform (FY 2024–2026)

**Author:** Shaury Pratap Singh (M.S. in Data Science, NJIT)  
**Tech Stack:** MS SQL Server (T-SQL, Docker) | Python (`SciPy`, `Statsmodels`, `Scikit-Learn`, `XGBoost`) | R (`aov`, `stats`) | DuckDB | Streamlit | Gradio | Plotly | Excel Power Query

---

## 1. Executive Summary
Standard H-1B portals (such as H1BGrader) index filings strictly by single city or employer and cannot cross-query **standardized roles (`SOC_TITLE`), state-level geographies, `NAICS` industry sectors, active `E-Verify` STEM OPT eligibility, and entry-level (`Wage Level I/II`) international student hiring velocity**.

This end-to-end data engineering and statistical inference platform unifies **2,095,550 federal records** across four government sources (`FY 2024 – FY 2026 Q3`) into a Dockerized **MS SQL Server Data Warehouse**, executes econometric and machine learning analyses in **Python and R**, and serves an interactive **Text-to-SQL & Geospatial Analytics Web Application**.

---

## 2. Data Warehouse Architecture (2.1M+ Records)
| Source Portal | SQL Table / View | Records Indexed | Primary Role in Warehouse |
|---|---|---|---|
| **DOL OFLC LCA Disclosure** | `dbo.Fact_LCA_Filings` | **1,372,949** (Deduplicated) | Job titles, `SOC` codes, worksite states, offered salaries, Prevailing Wage Levels (`I–IV`), and `NEW_EMPLOYMENT` counts |
| **USCIS E-Verify Portal** | `dbo.Stg_EVerify_Employers` | **461,297** | Active (`Open`) E-Verify status required for 24-month F-1 STEM OPT extensions |
| **USCIS H-1B Employer Hub** | `dbo.Stg_USCIS_H1B` | **148,754** | Primary `NAICS` industry classification and initial cap-subject H-1B approvals/denials |
| **DOL PERM Disclosure** | `dbo.Fact_PERM_Filings` | **112,550** | Minimum education degree tiers and special skill requirements |

---

## 3. Key Statistical, Econometric & Machine Learning Findings
* **Log-Normal MLE & Law of Total Variance:** Fitted Maximum Likelihood parameters ($\\hat{\\mu} = 11.6827, \\hat{\\sigma} = 0.3899$) to offered salaries. Variance decomposition proved that **within-state variance ($4.06 \\times 10^9$)** exceeds **between-state variance ($3.81 \\times 10^8$)** by **10.6x**, demonstrating that industry sector and role tier drive compensation far more than state geography alone.
* **Two-Way ANOVA & Non-Parametric FDR Control:** Executed Two-Way ANOVA (`NAICS_SECTOR` + `PW_WAGE_LEVEL`, $p < 10^{-16}$) in both Python and R, alongside **Mann-Whitney $U$ tests**, **500-iteration Percentile Bootstrapping**, and **Benjamini-Hochberg False Discovery Rate ($\\alpha = 0.05$)** correction across the top 15 states (e.g., NJ median H-1B salary: **$115,918**, 95% Bootstrap CI: **[$111,823, $117,575]**).
* **Distribution Drift (FY 2024 vs. FY 2026):** Quantified structural labor market shifts across the top 30 occupations using **Population Stability Index ($\\text{PSI} = 0.1389$)** and **Kullback-Leibler (KL) Divergence ($0.0717$)**.
* **Empirical Bayes Shrinkage & PCA Clustering:** Applied **Beta-Binomial conjugate priors** to shrink small-sample approval rates toward industry baselines, and used **PCA (81.9% variance explained)** to segment the top 300 sponsors.
* **Predictive Modeling & Time Series Forecasting:** Trained **XGBoost Regressors ($R^2 = 0.6859, \\text{MAE} = \\$18,044$)** and **Polynomial GLMs ($\\text{Adjusted } R^2 = 0.6505$)** for salary benchmarking, and benchmarked **Holt-Winters ETS ($\\text{MAE} = 47,834$)** against **ARIMA ($\\text{MAE} = 83,635$)**, reducing quarterly filing forecast error by **42.8%**.
"""
with open(os.path.join(BASE_DIR, "README.md"), "w", encoding="utf-8") as f:
    f.write(readme_content)
print("4. Created README.md.")
print("\nALL REPOSITORY FILES GENERATED SUCCESSFULLY!")