import os
import pandas as pd
import pymssql
from sqlalchemy import create_engine

BASE_DIR = os.path.expanduser("~/Desktop/H1B_Market_Intelligence")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
HF_DIR = os.path.join(BASE_DIR, "hf_space/data")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(HF_DIR, exist_ok=True)

conn = pymssql.connect(
    server="localhost", port=1433, user="sa",
    password="StrongPassword123!", database="H1B_Analytics_DW", autocommit=True
)
cursor = conn.cursor()

print("1. Fixing Employer Name Normalization ('CORPORATION' before 'CORP') & Primary NAICS Ranking...")

# Helper SQL expression: strips CORPORATION before CORP so MICROSOFT CORPORATION -> MICROSOFT
CLEAN_EMP_USC = """
LTRIM(RTRIM(
    REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
        UPPER(EMPLOYER_PETITIONER_NAME),
        ',', ''), '.', ''), ' CORPORATION', ''), ' CORP', ''), ' LLC', ''), ' INC', ''), ' LIMITED', '')
))
"""

CLEAN_EMP_EV = """
LTRIM(RTRIM(
    REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
        UPPER(EMPLOYER),
        ',', ''), '.', ''), ' CORPORATION', ''), ' CORP', ''), ' LLC', ''), ' INC', ''), ' LIMITED', '')
))
"""

CLEAN_EMP_LCA = """
LTRIM(RTRIM(
    REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(
        UPPER(l.EMPLOYER_NAME),
        ',', ''), '.', ''), ' CORPORATION', ''), ' CORP', ''), ' LLC', ''), ' INC', ''), ' LIMITED', '')
))
"""

cursor.execute(f"""
CREATE OR ALTER VIEW dbo.Dim_Company_NAICS_USCIS AS
WITH Cleaned_USCIS AS (
    SELECT 
        {CLEAN_EMP_USC} AS EMPLOYER_CLEAN_KEY,
        INDUSTRY_NAICS_CODE AS NAICS_SECTOR,
        ISNULL(TRY_CAST(REPLACE(CAST(NEW_EMPLOYMENT_APPROVAL AS VARCHAR), ',', '') AS INT), 0) AS NEW_H1B_APPROVALS,
        ISNULL(TRY_CAST(REPLACE(CAST(NEW_EMPLOYMENT_DENIAL AS VARCHAR), ',', '') AS INT), 0) AS NEW_H1B_DENIALS,
        ISNULL(TRY_CAST(REPLACE(CAST(CONTINUATION_APPROVAL AS VARCHAR), ',', '') AS INT), 0) AS CONT_H1B_APPROVALS
    FROM dbo.Stg_USCIS_H1B
    WHERE EMPLOYER_PETITIONER_NAME IS NOT NULL
      AND INDUSTRY_NAICS_CODE IS NOT NULL
),
Sector_Totals AS (
    SELECT 
        EMPLOYER_CLEAN_KEY,
        NAICS_SECTOR,
        SUM(NEW_H1B_APPROVALS) AS SECTOR_NEW_APP,
        SUM(NEW_H1B_DENIALS) AS SECTOR_NEW_DEN,
        SUM(CONT_H1B_APPROVALS) AS SECTOR_CONT_APP,
        ROW_NUMBER() OVER (
            PARTITION BY EMPLOYER_CLEAN_KEY 
            ORDER BY SUM(NEW_H1B_APPROVALS + CONT_H1B_APPROVALS) DESC
        ) AS rn
    FROM Cleaned_USCIS
    GROUP BY EMPLOYER_CLEAN_KEY, NAICS_SECTOR
),
Company_Totals AS (
    SELECT 
        EMPLOYER_CLEAN_KEY,
        SUM(SECTOR_NEW_APP) AS USCIS_INITIAL_APPROVALS_24_26,
        SUM(SECTOR_NEW_DEN) AS USCIS_INITIAL_DENIALS_24_26,
        SUM(SECTOR_CONT_APP) AS USCIS_CONTINUING_APPROVALS_24_26
    FROM Sector_Totals
    GROUP BY EMPLOYER_CLEAN_KEY
)
SELECT 
    c.EMPLOYER_CLEAN_KEY,
    s.NAICS_SECTOR,
    c.USCIS_INITIAL_APPROVALS_24_26,
    c.USCIS_INITIAL_DENIALS_24_26,
    c.USCIS_CONTINUING_APPROVALS_24_26
FROM Company_Totals c
JOIN Sector_Totals s
  ON c.EMPLOYER_CLEAN_KEY = s.EMPLOYER_CLEAN_KEY AND s.rn = 1;
""")

cursor.execute(f"""
CREATE OR ALTER VIEW dbo.Dim_Company_EVerify AS
WITH Ranked_EV AS (
    SELECT 
        {CLEAN_EMP_EV} AS EMPLOYER_CLEAN_KEY,
        ACCOUNT_STATUS AS EVERIFY_STATUS,
        WORKFORCE_SIZE,
        ROW_NUMBER() OVER (
            PARTITION BY {CLEAN_EMP_EV}
            ORDER BY CASE WHEN ACCOUNT_STATUS = 'Open' THEN 1 ELSE 2 END
        ) AS rn
    FROM dbo.Stg_EVerify_Employers
    WHERE EMPLOYER IS NOT NULL
)
SELECT EMPLOYER_CLEAN_KEY, EVERIFY_STATUS, WORKFORCE_SIZE
FROM Ranked_EV
WHERE rn = 1;
""")

cursor.execute(f"""
CREATE OR ALTER VIEW dbo.vw_Master_H1B_Intelligence AS
SELECT 
    l.CASE_NUMBER,
    l.CASE_STATUS,
    YEAR(TRY_CAST(l.DECISION_DATE AS DATE)) AS DECISION_YEAR,
    DATEPART(QUARTER, TRY_CAST(l.DECISION_DATE AS DATE)) AS DECISION_QUARTER,
    UPPER(LTRIM(RTRIM(l.JOB_TITLE))) AS JOB_TITLE,
    l.SOC_CODE,
    UPPER(LTRIM(RTRIM(l.SOC_TITLE))) AS SOC_TITLE,
    UPPER(LTRIM(RTRIM(l.EMPLOYER_NAME))) AS EMPLOYER_NAME_RAW,
    {CLEAN_EMP_LCA} AS EMPLOYER_CLEAN_KEY,
    ISNULL(u.NAICS_SECTOR, 'Unmapped / Other') AS NAICS_SECTOR,
    ISNULL(u.USCIS_INITIAL_APPROVALS_24_26, 0) AS USCIS_INITIAL_APPROVALS_24_26,
    CASE WHEN ev.EVERIFY_STATUS = 'Open' THEN 1 ELSE 0 END AS IS_EVERIFY_OPEN,
    ISNULL(ev.WORKFORCE_SIZE, 'Unknown') AS WORKFORCE_SIZE,
    l.WORKSITE_CITY,
    l.WORKSITE_STATE,
    l.PW_WAGE_LEVEL,
    ISNULL(TRY_CAST(l.NEW_EMPLOYMENT AS INT), 0) AS NEW_EMPLOYMENT_COUNT,
    ISNULL(TRY_CAST(l.CONTINUED_EMPLOYMENT AS INT), 0) AS CONTINUED_EMPLOYMENT_COUNT,
    CASE 
        WHEN l.WAGE_UNIT_OF_PAY = 'Year' THEN TRY_CAST(l.WAGE_RATE_OF_PAY_FROM AS FLOAT)
        WHEN l.WAGE_UNIT_OF_PAY = 'Hour' THEN TRY_CAST(l.WAGE_RATE_OF_PAY_FROM AS FLOAT) * 2080.0
        WHEN l.WAGE_UNIT_OF_PAY = 'Month' THEN TRY_CAST(l.WAGE_RATE_OF_PAY_FROM AS FLOAT) * 12.0
        WHEN l.WAGE_UNIT_OF_PAY = 'Bi-Weekly' THEN TRY_CAST(l.WAGE_RATE_OF_PAY_FROM AS FLOAT) * 26.0
        WHEN l.WAGE_UNIT_OF_PAY = 'Week' THEN TRY_CAST(l.WAGE_RATE_OF_PAY_FROM AS FLOAT) * 52.0
        ELSE NULL
    END AS ANNUAL_SALARY,
    CASE 
        WHEN l.PW_UNIT_OF_PAY = 'Year' THEN TRY_CAST(l.PREVAILING_WAGE AS FLOAT)
        WHEN l.PW_UNIT_OF_PAY = 'Hour' THEN TRY_CAST(l.PREVAILING_WAGE AS FLOAT) * 2080.0
        WHEN l.PW_UNIT_OF_PAY = 'Month' THEN TRY_CAST(l.PREVAILING_WAGE AS FLOAT) * 12.0
        WHEN l.PW_UNIT_OF_PAY = 'Bi-Weekly' THEN TRY_CAST(l.PREVAILING_WAGE AS FLOAT) * 26.0
        WHEN l.PW_UNIT_OF_PAY = 'Week' THEN TRY_CAST(l.PREVAILING_WAGE AS FLOAT) * 52.0
        ELSE NULL
    END AS ANNUAL_PREVAILING_WAGE
FROM dbo.Fact_LCA_Filings l
LEFT JOIN dbo.Dim_Company_NAICS_USCIS u
    ON {CLEAN_EMP_LCA} = u.EMPLOYER_CLEAN_KEY
LEFT JOIN dbo.Dim_Company_EVerify ev
    ON {CLEAN_EMP_LCA} = ev.EMPLOYER_CLEAN_KEY
WHERE l.VISA_CLASS = 'H-1B';
""")
conn.close()

engine = create_engine("mssql+pymssql://sa:StrongPassword123!@localhost:1433/H1B_Analytics_DW")

print("2. Exporting Clean Parquet Cube for Hugging Face Spaces & Tableau...")
cube_query = """
SELECT 
    DECISION_YEAR,
    WORKSITE_STATE,
    NAICS_SECTOR,
    SOC_TITLE,
    JOB_TITLE,
    EMPLOYER_CLEAN_KEY AS COMPANY,
    IS_EVERIFY_OPEN,
    COUNT(*) AS TOTAL_FILINGS,
    SUM(CASE WHEN CASE_STATUS = 'Certified' THEN 1 ELSE 0 END) AS CERTIFIED_FILINGS,
    SUM(CASE WHEN CASE_STATUS = 'Certified' AND PW_WAGE_LEVEL IN ('I', 'II', 'Level I', 'Level II') AND NEW_EMPLOYMENT_COUNT > 0 THEN 1 ELSE 0 END) AS F1_ENTRY_LEVEL_HIRES,
    MAX(USCIS_INITIAL_APPROVALS_24_26) AS USCIS_INITIAL_H1B_APPROVALS,
    ROUND(AVG(ANNUAL_SALARY), 0) AS AVG_OFFERED_SALARY,
    ROUND(AVG(ANNUAL_PREVAILING_WAGE), 0) AS AVG_PREVAILING_WAGE,
    ROUND(AVG(ANNUAL_SALARY - ANNUAL_PREVAILING_WAGE), 0) AS AVG_WAGE_PREMIUM
FROM dbo.vw_Master_H1B_Intelligence
WHERE DECISION_YEAR IN (2024, 2025, 2026)
  AND WORKSITE_STATE IS NOT NULL
  AND LEN(WORKSITE_STATE) = 2
  AND ANNUAL_SALARY BETWEEN 40000 AND 600000
GROUP BY 
    DECISION_YEAR, WORKSITE_STATE, NAICS_SECTOR, SOC_TITLE, JOB_TITLE, EMPLOYER_CLEAN_KEY, IS_EVERIFY_OPEN
HAVING COUNT(*) >= 2;
"""
df_cube = pd.read_sql(cube_query, engine)
df_cube.to_parquet(os.path.join(HF_DIR, "h1b_cube.parquet"), index=False)
df_cube.to_csv(os.path.join(OUT_DIR, "master_h1b_cube_2024_2026.csv"), index=False)
print(f"   -> Saved {len(df_cube):,} rows to hf_space/data/h1b_cube.parquet")

# Copy statistical outputs into hf_space/data/
for fname in ["anova_naics_wage_results.csv", "state_bootstrap_fdr_results.csv", "top_500_employers_bayesian_ranked.csv", "employer_pca_clusters.csv"]:
    src = os.path.join(OUT_DIR, fname)
    if os.path.exists(src):
        pd.read_csv(src).to_csv(os.path.join(HF_DIR, fname), index=False)

print("3. Generating Multi-Sheet Excel Executive Workbook (outputs/H1B_Executive_Audit_Workbook.xlsx)...")
excel_path = os.path.join(OUT_DIR, "H1B_Executive_Audit_Workbook.xlsx")
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    # Sheet 1: Top 200 E-Verify F-1 Sponsors (2025-2026)
    top_f1 = df_cube[(df_cube["IS_EVERIFY_OPEN"] == 1) & (df_cube["DECISION_YEAR"].isin([2025, 2026]))].groupby(
        ["COMPANY", "NAICS_SECTOR", "WORKSITE_STATE", "SOC_TITLE"], as_index=False
    ).agg(
        CERTIFIED_FILINGS=("CERTIFIED_FILINGS", "sum"),
        F1_ENTRY_LEVEL_HIRES=("F1_ENTRY_LEVEL_HIRES", "sum"),
        USCIS_INITIAL_APPROVALS=("USCIS_INITIAL_H1B_APPROVALS", "max"),
        AVG_SALARY=("AVG_OFFERED_SALARY", "mean"),
        AVG_WAGE_PREMIUM=("AVG_WAGE_PREMIUM", "mean")
    ).sort_values("F1_ENTRY_LEVEL_HIRES", ascending=False).head(250)
    top_f1.to_excel(writer, sheet_name="Top_F1_EVerify_Sponsors", index=False)

    # Sheet 2: State-by-State Hiring & Salary Summary
    state_sum = df_cube.groupby(["WORKSITE_STATE", "DECISION_YEAR"], as_index=False).agg(
        TOTAL_CERTIFIED=("CERTIFIED_FILINGS", "sum"),
        F1_ENTRY_HIRES=("F1_ENTRY_LEVEL_HIRES", "sum"),
        MEAN_SALARY=("AVG_OFFERED_SALARY", "mean")
    )
    state_sum.to_excel(writer, sheet_name="State_Year_Trends", index=False)

    # Sheet 3: Statistical Bootstrap & FDR Results
    if os.path.exists(os.path.join(OUT_DIR, "state_bootstrap_fdr_results.csv")):
        pd.read_csv(os.path.join(OUT_DIR, "state_bootstrap_fdr_results.csv")).to_excel(writer, sheet_name="Stats_Bootstrap_FDR", index=False)

    # Sheet 4: Two-Way ANOVA Table
    if os.path.exists(os.path.join(OUT_DIR, "anova_naics_wage_results.csv")):
        pd.read_csv(os.path.join(OUT_DIR, "anova_naics_wage_results.csv")).to_excel(writer, sheet_name="Two_Way_ANOVA", index=False)

print(f"   -> Saved Excel Executive Audit Workbook: {excel_path}\nDONE!")