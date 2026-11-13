import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import rel_entr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, roc_auc_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
BASE_DIR = os.path.expanduser("~/Desktop/H1B_Market_Intelligence")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
engine = create_engine("mssql+pymssql://sa:StrongPassword123!@localhost:1433/H1B_Analytics_DW")

print("1. Fetching Master H-1B Dataset from MS SQL Server...")
query = """
SELECT 
    CASE_STATUS, DECISION_YEAR, DECISION_QUARTER, SOC_CODE, SOC_TITLE,
    EMPLOYER_CLEAN_KEY, NAICS_SECTOR, IS_EVERIFY_OPEN, WORKSITE_STATE,
    PW_WAGE_LEVEL, NEW_EMPLOYMENT_COUNT, ANNUAL_SALARY, ANNUAL_PREVAILING_WAGE
FROM dbo.vw_Master_H1B_Intelligence
WHERE DECISION_YEAR IN (2024, 2025, 2026)
  AND ANNUAL_SALARY BETWEEN 40000 AND 600000
  AND ANNUAL_PREVAILING_WAGE BETWEEN 35000 AND 500000
"""
df = pd.read_sql(query, engine)
print(f"   -> Loaded {len(df):,} clean records from MS SQL.\n")

# ==============================================================================
# MODULE A: MLE Log-Normal Fit & Law of Total Variance (Lectures 14-15)
# ==============================================================================
salaries = df["ANNUAL_SALARY"].dropna()
shape, loc, scale = stats.lognorm.fit(salaries, floc=0)
mu_mle, sigma_mle = np.log(scale), shape
total_var = salaries.var()
within_state_var = df.groupby("WORKSITE_STATE")["ANNUAL_SALARY"].var().mean()
between_state_var = df.groupby("WORKSITE_STATE")["ANNUAL_SALARY"].mean().var()
print(f"[Module A - MLE & Law of Total Variance]")
print(f"   -> Log-Normal MLE: mu = {mu_mle:.4f}, sigma = {sigma_mle:.4f}")
print(f"   -> Total Salary Variance: {total_var:,.0f} | Within-State Mean Var: {within_state_var:,.0f} | Between-State Var: {between_state_var:,.0f}\n")

# ==============================================================================
# MODULE B: One-Way & Two-Way ANOVA Across NAICS Industries & Wage Levels
# ==============================================================================
df_mapped = df[df["NAICS_SECTOR"] != "Unmapped / Other"].copy()
df_mapped["WAGE_PREMIUM"] = df_mapped["ANNUAL_SALARY"] - df_mapped["ANNUAL_PREVAILING_WAGE"]
anova_sample = df_mapped.dropna(subset=["WAGE_PREMIUM", "NAICS_SECTOR", "PW_WAGE_LEVEL"]).sample(
    n=min(100000, len(df_mapped)), random_state=42
)
anova_model = ols("ANNUAL_SALARY ~ C(NAICS_SECTOR) + C(PW_WAGE_LEVEL)", data=anova_sample).fit()
anova_table = sm.stats.anova_lm(anova_model, typ=2)
print("[Module B - Two-Way ANOVA (NAICS Sector + Wage Level)]")
print(anova_table, "\n")
anova_table.to_csv(os.path.join(OUT_DIR, "anova_naics_wage_results.csv"))

# ==============================================================================
# MODULE C: Kruskal-Wallis, Bootstrapping & Benjamini-Hochberg FDR (Lecture 16)
# ==============================================================================
top_states = df["WORKSITE_STATE"].value_counts().head(15).index
overall_median = df["ANNUAL_SALARY"].median()
p_vals, state_list, medians, boot_ci_low, boot_ci_high = [], [], [], [], []

rng = np.random.default_rng(42)
for st in top_states:
    sub = df.loc[df["WORKSITE_STATE"] == st, "ANNUAL_SALARY"].dropna().values
    _, p = stats.mannwhitneyu(sub, salaries.sample(10000, random_state=42))
    # Percentile Bootstrap (500 resamples on subsample)
    sub_boot = rng.choice(sub, size=(500, min(1000, len(sub))), replace=True)
    ci_l, ci_h = np.percentile(np.median(sub_boot, axis=1), [2.5, 97.5])
    state_list.append(st)
    medians.append(np.median(sub))
    p_vals.append(p)
    boot_ci_low.append(ci_l)
    boot_ci_high.append(ci_h)

reject_fdr, p_adj_bh, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
fdr_df = pd.DataFrame({
    "STATE": state_list, "MEDIAN_SALARY": medians,
    "BOOT_95_CI_LOW": boot_ci_low, "BOOT_95_CI_HIGH": boot_ci_high,
    "RAW_P_VAL": p_vals, "BH_FDR_ADJ_P": p_adj_bh, "SIGNIFICANT_FDR_05": reject_fdr
})
fdr_df.to_csv(os.path.join(OUT_DIR, "state_bootstrap_fdr_results.csv"), index=False)
print("[Module C - Bootstrap CIs & Benjamini-Hochberg FDR Completed -> Saved to CSV]\n")

# ==============================================================================
# MODULE D: KL Divergence & Population Stability Index (PSI) Role Drift (2024 vs 2026)
# ==============================================================================
top_socs = df["SOC_TITLE"].value_counts().head(30).index
d24 = df[df["DECISION_YEAR"] == 2024]["SOC_TITLE"].value_counts()
d26 = df[df["DECISION_YEAR"] == 2026]["SOC_TITLE"].value_counts()
p24 = (d24.reindex(top_socs, fill_value=1) / d24.reindex(top_socs, fill_value=1).sum()).values
p26 = (d26.reindex(top_socs, fill_value=1) / d26.reindex(top_socs, fill_value=1).sum()).values

psi_components = (p26 - p24) * np.log(p26 / p24)
total_psi = np.sum(psi_components)
kl_div = np.sum(rel_entr(p26, p24))
print(f"[Module D - Distribution Drift (FY2024 vs FY2026 Top 30 Roles)]")
print(f"   -> Total Population Stability Index (PSI): {total_psi:.4f} | KL Divergence: {kl_div:.4f}\n")

# ==============================================================================
# MODULE E: Empirical Bayes Beta-Binomial Sponsor Reliability Shrinkage
# ==============================================================================
df["IS_CERTIFIED"] = (df["CASE_STATUS"] == "Certified").astype(int)
emp_stats = df.groupby(["EMPLOYER_CLEAN_KEY", "NAICS_SECTOR"]).agg(
    TOTAL=("IS_CERTIFIED", "count"),
    CERTIFIED=("IS_CERTIFIED", "sum"),
    AVG_SALARY=("ANNUAL_SALARY", "mean"),
    F1_ENTRY_HIRES=("NEW_EMPLOYMENT_COUNT", lambda x: (x > 0).sum())
).reset_index()

# Prior parameters alpha_0 = 95, beta_0 = 5 (95% baseline approval rate prior)
alpha_0, beta_0 = 95.0, 5.0
emp_stats["RAW_APPROVAL_RATE"] = emp_stats["CERTIFIED"] / emp_stats["TOTAL"]
emp_stats["BAYESIAN_SHRUNK_APPROVAL_RATE"] = (emp_stats["CERTIFIED"] + alpha_0) / (emp_stats["TOTAL"] + alpha_0 + beta_0)
emp_stats.sort_values("TOTAL", ascending=False).head(500).to_csv(
    os.path.join(OUT_DIR, "top_500_employers_bayesian_ranked.csv"), index=False
)
print("[Module E - Empirical Bayes Beta-Binomial Sponsor Ranking Completed]\n")

# ==============================================================================
# MODULE F: Polynomial Regression, Logistic GLM (Odds Ratios) & XGBoost
# ==============================================================================
ml_df = df_mapped.dropna(subset=["ANNUAL_SALARY", "ANNUAL_PREVAILING_WAGE", "PW_WAGE_LEVEL"]).sample(
    n=min(80000, len(df_mapped)), random_state=42
).copy()
wage_map = {"I": 1, "Level I": 1, "II": 2, "Level II": 2, "III": 3, "Level III": 3, "IV": 4, "Level IV": 4}
ml_df["WAGE_LEVEL_NUM"] = ml_df["PW_WAGE_LEVEL"].map(wage_map).fillna(2)

# 1. Polynomial Regression (Degree 2) on Prevailing Wage & Wage Level -> Offered Salary
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(ml_df[["ANNUAL_PREVAILING_WAGE", "WAGE_LEVEL_NUM"]])
y_sal = ml_df["ANNUAL_SALARY"].values
poly_ols = sm.OLS(y_sal, sm.add_constant(X_poly)).fit()

# 2. Gradient Boosting Regressor (XGBoost)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_sal, test_size=0.2, random_state=42)
xgb_reg = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42)
xgb_reg.fit(X_train, y_train)
preds = xgb_reg.predict(X_test)
print(f"[Module F - Regression & Gradient Boosting]")
print(f"   -> Polynomial Regression Adjusted R2: {poly_ols.rsquared_adj:.4f}")
print(f"   -> XGBoost Salary Predictor R2: {r2_score(y_test, preds):.4f} | MAE: ${mean_absolute_error(y_test, preds):,.0f}\n")

# ==============================================================================
# MODULE G: Time Series Forecasting (ETS & ARIMA with MAE Metric)
# ==============================================================================
ts_df = df.groupby(["DECISION_YEAR", "DECISION_QUARTER"]).size().reset_index(name="FILINGS")
ts_df = ts_df.sort_values(["DECISION_YEAR", "DECISION_QUARTER"])
y_ts = ts_df["FILINGS"].values
if len(y_ts) >= 6:
    train_ts, test_ts = y_ts[:-2], y_ts[-2:]
    ets_fit = ExponentialSmoothing(train_ts, trend="add", seasonal=None).fit()
    ets_pred = ets_fit.forecast(len(test_ts))
    arima_fit = ARIMA(train_ts, order=(1, 1, 0)).fit()
    arima_pred = arima_fit.forecast(len(test_ts))
    print(f"[Module G - Time Series Forecasting (ETS & ARIMA)]")
    print(f"   -> ETS Forecast MAE: {mean_absolute_error(test_ts, ets_pred):,.1f} filings | ARIMA Forecast MAE: {mean_absolute_error(test_ts, arima_pred):,.1f} filings\n")

# ==============================================================================
# MODULE H: Dimensionality Reduction (PCA) on Top 300 Employers
# ==============================================================================
top300 = emp_stats.sort_values("TOTAL", ascending=False).head(300).copy()
features = StandardScaler().fit_transform(top300[["TOTAL", "AVG_SALARY", "F1_ENTRY_HIRES", "BAYESIAN_SHRUNK_APPROVAL_RATE"]])
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(features)
top300["PCA_1"], top300["PCA_2"] = coords[:, 0], coords[:, 1]
top300.to_csv(os.path.join(OUT_DIR, "employer_pca_clusters.csv"), index=False)
print(f"[Module H - PCA Dimensionality Reduction] Explained Variance Ratio: {pca.explained_variance_ratio_.sum()*100:.1f}%")
print("\nALL STATISTICAL & ML MODULES COMPLETED SUCCESSFULLY!")