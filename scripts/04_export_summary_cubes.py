import os
import pandas as pd
from sqlalchemy import create_engine

BASE_DIR = os.path.expanduser("~/Desktop/H1B_Market_Intelligence")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
engine = create_engine("mssql+pymssql://sa:StrongPassword123!@localhost:1433/H1B_Analytics_DW")

print("1. Exporting Multi-Dimensional State x Year x Industry x Role x Company Cube...")
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
  AND ANNUAL_SALARY BETWEEN 40000 AND 600000
GROUP BY 
    DECISION_YEAR, WORKSITE_STATE, NAICS_SECTOR, SOC_TITLE, JOB_TITLE, EMPLOYER_CLEAN_KEY, IS_EVERIFY_OPEN
HAVING COUNT(*) >= 2;
"""
df_cube = pd.read_sql(cube_query, engine)
cube_path = os.path.join(OUT_DIR, "master_h1b_cube_2024_2026.csv")
df_cube.to_csv(cube_path, index=False)
print(f"   -> Saved {len(df_cube):,} aggregated rows to {cube_path}")

print("2. Extracting Company Education & Skill Requirements from dbo.Fact_PERM_Filings...")
perm_df = pd.read_sql("SELECT TOP 50000 * FROM dbo.Fact_PERM_Filings", engine)
print("   -> PERM Columns available:", list(perm_df.columns))
perm_summary_path = os.path.join(OUT_DIR, "perm_sample_requirements.csv")
perm_df.head(5000).to_csv(perm_summary_path, index=False)
print(f"   -> Saved PERM requirements sample to {perm_summary_path}")