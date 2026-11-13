# US H-1B & F-1 STEM OPT Market Intelligence Platform (FY 2024–2026)

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
* **Log-Normal MLE & Law of Total Variance:** Fitted Maximum Likelihood parameters ($\hat{\mu} = 11.6827, \hat{\sigma} = 0.3899$) to offered salaries. Variance decomposition proved that **within-state variance ($4.06 \times 10^9$)** exceeds **between-state variance ($3.81 \times 10^8$)** by **10.6x**, demonstrating that industry sector and role tier drive compensation far more than state geography alone.
* **Two-Way ANOVA & Non-Parametric FDR Control:** Executed Two-Way ANOVA (`NAICS_SECTOR` + `PW_WAGE_LEVEL`, $p < 10^{-16}$) in both Python and R, alongside **Mann-Whitney $U$ tests**, **500-iteration Percentile Bootstrapping**, and **Benjamini-Hochberg False Discovery Rate ($\alpha = 0.05$)** correction across the top 15 states (e.g., NJ median H-1B salary: **$115,918**, 95% Bootstrap CI: **[$111,823, $117,575]**).
* **Distribution Drift (FY 2024 vs. FY 2026):** Quantified structural labor market shifts across the top 30 occupations using **Population Stability Index ($\text{PSI} = 0.1389$)** and **Kullback-Leibler (KL) Divergence ($0.0717$)**.
* **Empirical Bayes Shrinkage & PCA Clustering:** Applied **Beta-Binomial conjugate priors** to shrink small-sample approval rates toward industry baselines, and used **PCA (81.9% variance explained)** to segment the top 300 sponsors.
* **Predictive Modeling & Time Series Forecasting:** Trained **XGBoost Regressors ($R^2 = 0.6859, \text{MAE} = \$18,044$)** and **Polynomial GLMs ($\text{Adjusted } R^2 = 0.6505$)** for salary benchmarking, and benchmarked **Holt-Winters ETS ($\text{MAE} = 47,834$)** against **ARIMA ($\text{MAE} = 83,635$)**, reducing quarterly filing forecast error by **42.8%**.
