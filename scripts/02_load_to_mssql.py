import os
import pandas as pd
import pymssql
from sqlalchemy import create_engine

BASE_DIR = os.path.expanduser("~/Desktop/H1B_Market_Intelligence")

# 1. Ensure H1B_Analytics_DW database exists
conn = pymssql.connect(server="localhost", port=1433, user="sa", password="StrongPassword123!", autocommit=True)
cursor = conn.cursor()
cursor.execute("""
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = N'H1B_Analytics_DW')
    CREATE DATABASE H1B_Analytics_DW;
""")
conn.close()

engine = create_engine("mssql+pymssql://sa:StrongPassword123!@localhost:1433/H1B_Analytics_DW")

def read_tableau_csv(filepath):
    """Automatically detects UTF-8 vs UTF-16 and comma vs tab delimiters from Tableau exports."""
    for enc in ["utf-8", "utf-16", "utf-8-sig", "latin1"]:
        for sep in ["\t", ","]:
            try:
                df = pd.read_csv(filepath, encoding=enc, sep=sep, low_memory=False)
                if len(df.columns) > 3:
                    return df
            except Exception:
                continue
    raise ValueError(f"Could not parse {filepath}")

def clean_col_names(df):
    df.columns = [
        str(c).strip().upper()
        .replace(" ", "_").replace("(", "").replace(")", "")
        .replace("/", "_").replace("-", "_").replace("+", "_PLUS").replace(".", "")
        for c in df.columns
    ]
    return df

if __name__ == "__main__":
    # 1. Load USCIS H-1B CSV
    uscis_path = os.path.join(BASE_DIR, "raw_data/uscis/uscis_h1b_2024_2026.csv")
    print("Loading USCIS H-1B Employer Information...")
    df_uscis = clean_col_names(read_tableau_csv(uscis_path))
    print("  -> USCIS Columns detected:", list(df_uscis.columns))
    df_uscis.to_sql("Stg_USCIS_H1B", engine, schema="dbo", if_exists="replace", index=False, chunksize=10000)
    print(f"  -> Loaded {len(df_uscis):,} rows into dbo.Stg_USCIS_H1B\n")

    # 2. Load E-Verify CSV
    ev_path = os.path.join(BASE_DIR, "raw_data/everify/everify_employers.csv")
    print("Loading E-Verify Employer List...")
    df_ev = clean_col_names(read_tableau_csv(ev_path))
    print("  -> E-Verify Columns detected:", list(df_ev.columns))
    df_ev.to_sql("Stg_EVerify_Employers", engine, schema="dbo", if_exists="replace", index=False, chunksize=10000)
    print(f"  -> Loaded {len(df_ev):,} rows into dbo.Stg_EVerify_Employers\n")

    # 3. Load PERM CSV
    perm_path = os.path.join(BASE_DIR, "csv_staging/perm_clean_2026.csv")
    if os.path.exists(perm_path):
        print("Loading PERM Disclosure Data...")
        df_perm = clean_col_names(pd.read_csv(perm_path, low_memory=False))
        df_perm.to_sql("Fact_PERM_Filings", engine, schema="dbo", if_exists="replace", index=False, chunksize=10000)
        print(f"  -> Loaded {len(df_perm):,} rows into dbo.Fact_PERM_Filings\n")

    # 4. Stream LCA Clean CSV
    lca_path = os.path.join(BASE_DIR, "csv_staging/lca_clean_2024_2026.csv")
    print("Streaming LCA Clean Data into dbo.Fact_LCA_Filings...")
    for i, chunk in enumerate(pd.read_csv(lca_path, chunksize=50000, low_memory=False)):
        chunk = clean_col_names(chunk)
        mode = "replace" if i == 0 else "append"
        chunk.to_sql("Fact_LCA_Filings", engine, schema="dbo", if_exists=mode, index=False, chunksize=5000)
        print(f"  -> Chunk {i+1} loaded ({(i+1)*50000:,} rows processed)")
    print("\nALL 4 TABLES SUCCESSFULLY LOADED INTO MS SQL SERVER!")