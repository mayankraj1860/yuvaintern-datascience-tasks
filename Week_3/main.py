"""
Week 3 Task: Statistical Analysis and Hypothesis Testing in Python
Internship Domain: Data Science with Python
Platform: Yuva Intern
Author: Mayank Raj
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

# Visual formatting
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Dataset Generation
np.random.seed(42)
n_samples = 600

promotions = np.random.choice(
    ["Control", "Discount", "Cashback"], n_samples, p=[0.33, 0.34, 0.33]
)
segments = np.random.choice(["Retail", "Enterprise"], n_samples, p=[0.7, 0.3])
devices = np.random.choice(["Mobile", "Desktop"], n_samples, p=[0.6, 0.4])

base_spend = np.random.normal(500, 100, n_samples)
spend = (
    base_spend
    + (promotions == "Discount") * 65.0
    + (promotions == "Cashback") * 110.0
    + (segments == "Enterprise") * 200.0
    + np.random.normal(0, 40, n_samples)
).clip(50, 1500)

conversion_prob = (
    0.25
    + (promotions == "Discount") * 0.15
    + (promotions == "Cashback") * 0.22
    + (segments == "Enterprise") * 0.10
)
conversion = (np.random.rand(n_samples) < conversion_prob).astype(int)

df = pd.DataFrame({
    "Customer_ID": [f"CUST_{2000+i}" for i in range(n_samples)],
    "Promotion": promotions,
    "Segment": segments,
    "Device": devices,
    "Purchase_Amount": spend,
    "Converted": conversion,
})

df.to_csv("hypothesis_testing_dataset.csv", index=False)
print("=" * 60)
print("WEEK 3: STATISTICAL HYPOTHESIS TESTING PIPELINE")
print("=" * 60)

# ----------------------------------------------------
# 2. Hypothesis 1: Two-Sample Independent T-Test
# ----------------------------------------------------
retail_spend = df[df["Segment"] == "Retail"]["Purchase_Amount"]
ent_spend = df[df["Segment"] == "Enterprise"]["Purchase_Amount"]
t_stat, t_pval = stats.ttest_ind(ent_spend, retail_spend, equal_var=False)

print("\n--- Test 1: Two-Sample Welch's T-Test ---")
print(f"Retail Mean: ${retail_spend.mean():.2f}")
print(f"Enterprise Mean: ${ent_spend.mean():.2f}")
print(f"t-statistic: {t_stat:.4f}, p-value: {t_pval:.4e}")
print("Result:", "Reject H0" if t_pval < 0.05 else "Fail to reject H0")

plt.figure(figsize=(7, 4), dpi=300)
sns.kdeplot(
    data=df,
    x="Purchase_Amount",
    hue="Segment",
    fill=True,
    common_norm=False,
    palette="Set1",
    alpha=0.35,
)
plt.axvline(
    retail_spend.mean(),
    color="#e41a1c",
    linestyle="--",
    label=f"Retail Mean: ${retail_spend.mean():.1f}",
)
plt.axvline(
    ent_spend.mean(),
    color="#377eb8",
    linestyle="--",
    label=f"Enterprise Mean: ${ent_spend.mean():.1f}",
)
plt.title(
    f"Hypothesis 1 (Two-Sample T-Test): Spend Distribution by Segment\n(t ="
    f" {t_stat:.2f}, p-value = {t_pval:.2e})",
    fontsize=10.5,
    fontweight="bold",
)
plt.xlabel("Purchase Amount ($)")
plt.ylabel("Density")
plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig("w3_plot1_ttest.png")
plt.close()

# ----------------------------------------------------
# 3. Hypothesis 2: One-Way ANOVA
# ----------------------------------------------------
ctrl_spend = df[df["Promotion"] == "Control"]["Purchase_Amount"]
disc_spend = df[df["Promotion"] == "Discount"]["Purchase_Amount"]
cash_spend = df[df["Promotion"] == "Cashback"]["Purchase_Amount"]
f_stat, anova_pval = stats.f_oneway(ctrl_spend, disc_spend, cash_spend)

print("\n--- Test 2: One-Way ANOVA ---")
print(f"F-statistic: {f_stat:.4f}, p-value: {anova_pval:.4e}")
print("Result:", "Reject H0" if anova_pval < 0.05 else "Fail to reject H0")

plt.figure(figsize=(7, 4), dpi=300)
sns.boxplot(
    data=df,
    x="Promotion",
    y="Purchase_Amount",
    palette="Blues",
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "red",
        "markeredgecolor": "red",
        "markersize": "8",
    },
)
plt.title(
    f"Hypothesis 2 (One-Way ANOVA): Spend across Promotion Strategies\n(F ="
    f" {f_stat:.2f}, p-value = {anova_pval:.2e})",
    fontsize=10.5,
    fontweight="bold",
)
plt.xlabel("Promotion Type")
plt.ylabel("Purchase Amount ($)")
plt.tight_layout()
plt.savefig("w3_plot2_anova.png")
plt.close()

# ----------------------------------------------------
# 4. Hypothesis 3: Chi-Square Test of Independence
# ----------------------------------------------------
contingency_table = pd.crosstab(df["Promotion"], df["Converted"])
chi2_stat, chi2_pval, dof, _ = stats.chi2_contingency(contingency_table)

print("\n--- Test 3: Chi-Square Test of Independence ---")
print(f"Chi2-statistic: {chi2_stat:.4f}, p-value: {chi2_pval:.4e}, DoF: {dof}")
print("Result:", "Reject H0" if chi2_pval < 0.05 else "Fail to reject H0")

plt.figure(figsize=(6.5, 3.8), dpi=300)
conversion_rates = df.groupby("Promotion")["Converted"].mean() * 100
ax = sns.barplot(
    x=conversion_rates.index,
    y=conversion_rates.values,
    hue=conversion_rates.index,
    palette="viridis",
    legend=False,
)
for p in ax.patches:
  ax.annotate(
      f"{p.get_height():.1f}%",
      (p.get_x() + p.get_width() / 2.0, p.get_height() / 2),
      ha="center",
      va="center",
      fontsize=10,
      color="white",
      fontweight="bold",
  )
plt.title(
    f"Hypothesis 3 (Chi-Square Test): Conversion Rate by Promotion\n(χ² ="
    f" {chi2_stat:.2f}, p-value = {chi2_pval:.2e})",
    fontsize=10.5,
    fontweight="bold",
)
plt.xlabel("Promotion Type")
plt.ylabel("Conversion Rate (%)")
plt.tight_layout()
plt.savefig("w3_plot3_chisquare.png")
plt.close()

print("\nAll statistical tests executed and plots saved successfully.")