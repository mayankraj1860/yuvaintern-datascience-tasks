"""
Week 1 Task: Data Acquisition, Cleaning, and Exploratory Analysis
Internship Domain: Data Science with Python
Platform: Yuva Intern
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set visualization aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# ==========================================
# 1. DATA ACQUISITION & SYNTHESIS
# ==========================================
np.random.seed(42)
n_samples = 500

ages = np.random.normal(38, 12, n_samples).clip(18, 75).astype(int)
incomes = (ages * 1200 + np.random.normal(15000, 10000, n_samples)).clip(
    20000, 150000
)
spending_score = np.random.randint(1, 100, n_samples)
purchase_amounts = (
    spending_score * 45 + incomes * 0.02 + np.random.normal(200, 150, n_samples)
).clip(50, 5000)
churn = np.where((spending_score < 40) & (purchase_amounts < 1500), 1, 0)

# Introduce realistic nulls and duplicate entries
missing_mask = np.random.rand(n_samples) < 0.08
incomes_with_na = incomes.copy().astype(float)
incomes_with_na[missing_mask] = np.nan

df_raw = pd.DataFrame(
    {
        "Age": ages,
        "Annual_Income": incomes_with_na,
        "Spending_Score": spending_score,
        "Purchase_Amount": purchase_amounts,
        "Churn": churn,
    }
)

# Append duplicates to simulate real-world data issues
duplicates = df_raw.sample(15, random_state=42)
df_raw = pd.concat([df_raw, duplicates], ignore_index=True)

print("--- 1. Initial Data Inspection ---")
print(f"Raw Dataset Shape: {df_raw.shape}")
print("\nSchema / Data Types:\n", df_raw.dtypes)
print("\nMissing Values Count:\n", df_raw.isnull().sum())
print("\nSummary Statistics:\n", df_raw.describe().T)

# ==========================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================
# Step 1: Remove Duplicate Rows
initial_count = len(df_raw)
df_cleaned = df_raw.drop_duplicates().copy()
print(f"\nDuplicates removed: {initial_count - len(df_cleaned)}")

# Step 2: Handle Missing Values (Median Imputation)
income_median = df_cleaned["Annual_Income"].median()
df_cleaned["Annual_Income"] = df_cleaned["Annual_Income"].fillna(income_median)

# Step 3: Outlier Handling via IQR Capping
Q1 = df_cleaned["Purchase_Amount"].quantile(0.25)
Q3 = df_cleaned["Purchase_Amount"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned["Purchase_Amount"] = np.clip(
    df_cleaned["Purchase_Amount"], lower_bound, upper_bound
)

# Verify Cleaned Dataset
print("\n--- 2. Post-Cleaning Verification ---")
print(f"Cleaned Dataset Shape: {df_cleaned.shape}")
print(f"Remaining Missing Values: {df_cleaned.isnull().sum().sum()}")

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================

# Visualization 1: Missing Value Distribution
plt.figure(figsize=(7, 4), dpi=300)
missing_counts = df_raw.isnull().sum()
missing_counts["Total Records"] = len(df_raw)
colors = [
    "#e74c3c" if val > 0 and idx != "Total Records" else "#3498db"
    for idx, val in missing_counts.items()
]
sns.barplot(x=missing_counts.index, y=missing_counts.values, palette=colors)
plt.title(
    "Missing Value Assessment vs Total Records", fontsize=12, fontweight="bold"
)
plt.ylabel("Count")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("plot1_missing_values.png")
plt.show()

# Visualization 2: Distribution & Outliers (Boxplot + Histogram)
fig, (ax_box, ax_hist) = plt.subplots(
    2,
    1,
    figsize=(7, 4.5),
    sharex=True,
    gridspec_kw={"height_ratios": (0.25, 0.75)},
    dpi=300,
)
sns.boxplot(x=df_cleaned["Purchase_Amount"], ax=ax_box, color="#3498db")
ax_box.set(xlabel="")
ax_box.set_title(
    "Distribution & Outlier Analysis: Purchase Amount ($)",
    fontsize=11,
    fontweight="bold",
)
sns.histplot(
    df_cleaned["Purchase_Amount"],
    ax=ax_hist,
    kde=True,
    color="#2980b9",
    bins=25,
)
ax_hist.set_xlabel("Purchase Amount ($)")
ax_hist.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("plot2_distribution.png")
plt.show()

# Visualization 3: Pearson Correlation Heatmap
plt.figure(figsize=(6, 4.5), dpi=300)
sns.heatmap(
    df_cleaned.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title(
    "Pearson Correlation Matrix of Attributes", fontsize=11, fontweight="bold"
)
plt.tight_layout()
plt.savefig("plot3_correlation.png")
plt.show()

print("\nEDA Execution complete. Plots saved locally.")