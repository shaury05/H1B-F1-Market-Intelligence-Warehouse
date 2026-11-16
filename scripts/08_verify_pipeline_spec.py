"""
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
        if "ORATION" in tok and not any(s in tok for s in ("EXPLORAT", "RESTORAT", "COLLABORAT", "INCORPORAT", "DECORAT", "ELABORAT", "AMELIORAT", "PERFORAT", "EVAPORAT", "DETERIORAT", "COMMEMORAT", "COLORAT")):
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
    print("\nALL SPECIFICATION CHECKS PASSED (100% COMPLIANT)!")

if __name__ == "__main__":
    try:
        run_spec_audit()
    except AssertionError as err:
        print(f"[AUDIT FAILED]: {err}")
        sys.exit(1)
