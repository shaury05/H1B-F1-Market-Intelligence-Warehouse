# ==============================================================================
# R Statistical Inference Module: Two-Way ANOVA, BH-FDR & Linear Wage Modeling
# ==============================================================================
cube_path <- path.expand("~/Desktop/H1B_Market_Intelligence/outputs/master_h1b_cube_2024_2026.csv")
df <- read.csv(cube_path)

df_clean <- subset(df, NAICS_SECTOR != "Unmapped / Other" & AVG_OFFERED_SALARY >= 40000)

cat("=== 1. R Two-Way ANOVA Across NAICS Industries & E-Verify Status ===\n")
anova_fit <- aov(AVG_OFFERED_SALARY ~ NAICS_SECTOR + factor(IS_EVERIFY_OPEN), data = df_clean)
print(summary(anova_fit))

cat("\n=== 2. Healthcare (NAICS 62) vs. Tech (NAICS 51/54) Pairwise BH-Adjusted t-test ===\n")
hc_tech <- subset(df_clean, grepl("^62|^51|^54", NAICS_SECTOR))
t_res <- pairwise.t.test(hc_tech$AVG_OFFERED_SALARY, hc_tech$NAICS_SECTOR, p.adjust.method = "BH")
print(t_res)

cat("\n=== 3. R Linear Regression Model (Offered Salary vs Prevailing Wage & F-1 Hires) ===\n")
lm_fit <- lm(AVG_OFFERED_SALARY ~ AVG_PREVAILING_WAGE + IS_EVERIFY_OPEN + F1_ENTRY_LEVEL_HIRES, data = df_clean)
print(summary(lm_fit)$coefficients)
