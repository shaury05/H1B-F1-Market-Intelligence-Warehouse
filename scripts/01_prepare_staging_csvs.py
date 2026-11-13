import glob
import os
import pandas as pd

BASE_DIR = os.path.expanduser("~/Desktop/H1B_Market_Intelligence")
LCA_DIR = os.path.join(BASE_DIR, "raw_data/lca")
PERM_DIR = os.path.join(BASE_DIR, "raw_data/perm")
OUT_DIR = os.path.join(BASE_DIR, "csv_staging")

LCA_COLS = [
    "CASE_NUMBER", "CASE_STATUS", "RECEIVED_DATE", "DECISION_DATE",
    "VISA_CLASS", "JOB_TITLE", "SOC_CODE", "SOC_TITLE",
    "FULL_TIME_POSITION", "BEGIN_DATE", "END_DATE",
    "TOTAL_WORKER_POSITIONS", "NEW_EMPLOYMENT", "CONTINUED_EMPLOYMENT",
    "CHANGE_PREVIOUS_EMPLOYMENT", "AMENDED_PETITION",
    "EMPLOYER_NAME", "TRADE_NAME_DBA", "EMPLOYER_CITY", "EMPLOYER_STATE",
    "WORKSITE_CITY", "WORKSITE_COUNTY", "WORKSITE_STATE", "WORKSITE_POSTAL_CODE",
    "WAGE_RATE_OF_PAY_FROM", "WAGE_RATE_OF_PAY_TO", "WAGE_UNIT_OF_PAY",
    "PREVAILING_WAGE", "PW_UNIT_OF_PAY", "PW_WAGE_LEVEL", "H_1B_DEPENDENT"
]

def process_lca_files():
    files = sorted(glob.glob(os.path.join(LCA_DIR, "*.xlsx")))
    print(f"[1/2] Found {len(files)} LCA files. Reading with Rust calamine engine...")
    chunks = []
    for f in files:
        fname = os.path.basename(f)
        print(f"  -> Reading {fname}...")
        df = pd.read_excel(f, engine="calamine")
        df.columns = [str(c).strip().upper() for c in df.columns]
        keep = [c for c in LCA_COLS if c in df.columns]
        df = df[keep].copy()
        df["SOURCE_FILE"] = fname
        chunks.append(df)

    full_lca = pd.concat(chunks, ignore_index=True)
    print(f"  -> Raw rows before deduplication: {len(full_lca):,}")

    full_lca["DECISION_DATE"] = pd.to_datetime(full_lca["DECISION_DATE"], errors="coerce")
    full_lca = full_lca.sort_values("DECISION_DATE", ascending=False)
    full_lca = full_lca.drop_duplicates(subset=["CASE_NUMBER"], keep="first")
    print(f"  -> Unique CASE_NUMBER rows after deduplication: {len(full_lca):,}")

    out_path = os.path.join(OUT_DIR, "lca_clean_2024_2026.csv")
    full_lca.to_csv(out_path, index=False)
    print(f"  -> Saved LCA staging CSV: {out_path}\n")

def process_perm_file():
    perm_files = glob.glob(os.path.join(PERM_DIR, "*.xlsx"))
    if not perm_files:
        print("No PERM file found, skipping.")
        return
    f = perm_files[0]
    print(f"[2/2] Reading {os.path.basename(f)} for Skills & Education requirements...")
    df = pd.read_excel(f, engine="calamine")
    df.columns = [str(c).strip().upper() for c in df.columns]
    # Keep columns related to Employer, Role, Wage, Degree, Experience, and Special Skills
    keywords = ["CASE_", "EMPLOYER", "JOB_TITLE", "SOC_", "PW_", "WAGE", "EDUCATION", "EXPERIENCE", "SKILL", "SPEC_REQ", "WORKSITE"]
    keep_cols = [c for c in df.columns if any(k in c for k in keywords)][:35]
    df = df[keep_cols].copy()
    out_path = os.path.join(OUT_DIR, "perm_clean_2026.csv")
    df.to_csv(out_path, index=False)
    print(f"  -> Saved PERM staging CSV ({len(df):,} rows, {len(keep_cols)} columns): {out_path}")

if __name__ == "__main__":
    process_lca_files()
    process_perm_file()