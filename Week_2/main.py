"""
Week 2 Task: Advanced Data Visualization and Storytelling with Python
Internship Domain: Data Science with Python
Platform: Yuva Intern
Author: Mayank Raj
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visual configuration
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# 1. Dataset Generation / Acquisition
np.random.seed(42)
n_records = 600
categories = ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports"]
devices = ["Mobile App", "Desktop", "Tablet"]
payment_methods = ["Credit Card", "UPI", "Debit Card", "Cash on Delivery"]

df = pd.DataFrame(
    {
        "Customer_ID": [f"CUST_{1000+i}" for i in range(n_records)],
        "Age": np.random.normal(35, 10, n_records).clip(18, 70).astype(int),
        "Gender": np.random.choice(["Male", "Female"], n_records, p=[0.48, 0.52]),
        "Category": np.random.choice(
            categories, n_records, p=[0.30, 0.25, 0.20, 0.15, 0.10]
        ),
        "Device": np.random.choice(devices, n_records, p=[0.60, 0.30, 0.10]),
        "Payment_Method": np.random.choice(
            payment_methods, n_records, p=[0.40, 0.35, 0.15, 0.10]
        ),
        "Monthly_Visits": np.random.poisson(8, n_records).clip(1, 30),
        "Discount_Applied": np.random.choice([0, 1], n_records, p=[0.35, 0.65]),
    }
)

df["Purchase_Amount"] = (
    df["Monthly_Visits"] * 40
    + (df["Category"] == "Electronics") * 250
    + (df["Category"] == "Fashion") * 120
    + np.random.normal(150, 60, n_records)
).clip(30, 1800)

df["Satisfaction_Score"] = np.where(
    df["Purchase_Amount"] > 800,
    np.random.choice([4, 5], n_records, p=[0.3, 0.7]),
    np.random.choice([1, 2, 3, 4], n_records, p=[0.15, 0.25, 0.4, 0.2]),
)

# Export processed CSV
df.to_csv("ecommerce_customer_behavior.csv", index=False)
print("Data ready. Shape:", df.shape)

# ----------------------------------------------------
# Visualization 1: Ranked Revenue Bar Chart
# ----------------------------------------------------
plt.figure(figsize=(7, 4), dpi=300)
cat_revenue = (
    df.groupby("Category")["Purchase_Amount"].sum().sort_values(ascending=False)
)
ax = sns.barplot(
    x=cat_revenue.index,
    y=cat_revenue.values,
    hue=cat_revenue.index,
    palette="Blues_r",
    legend=False,
)
for p in ax.patches:
  ax.annotate(
      f"${p.get_height():,.0f}",
      (p.get_x() + p.get_width() / 2.0, p.get_height() / 2),
      ha="center",
      va="center",
      fontsize=9,
      color="white",
      fontweight="bold",
  )
plt.title(
    "Total Revenue Contribution by Product Category",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Product Category")
plt.ylabel("Total Revenue ($)")
plt.tight_layout()
plt.savefig("w2_plot1_revenue_category.png")
plt.close()

# ----------------------------------------------------
# Visualization 2: Multi-Variable Scatter Plot
# ----------------------------------------------------
plt.figure(figsize=(7.5, 4.5), dpi=300)
sns.scatterplot(
    data=df,
    x="Monthly_Visits",
    y="Purchase_Amount",
    hue="Category",
    size="Satisfaction_Score",
    sizes=(30, 150),
    palette="Set2",
    alpha=0.85,
)
plt.title(
    "Engagement vs Monetary Value: Monthly Visits vs Purchase Amount",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Monthly App / Website Visits")
plt.ylabel("Total Purchase Amount ($)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("w2_plot2_engagement_value.png")
plt.close()

# ----------------------------------------------------
# Visualization 3: Grouped Boxplot for Payment & Discounts
# ----------------------------------------------------
plt.figure(figsize=(7, 4), dpi=300)
sns.boxplot(
    data=df,
    x="Payment_Method",
    y="Purchase_Amount",
    hue="Discount_Applied",
    palette="coolwarm",
)
plt.title(
    "Spending Behavior Across Payment Methods & Discount Application",
    fontsize=11,
    fontweight="bold",
)
plt.xlabel("Payment Method")
plt.ylabel("Purchase Amount ($)")
plt.legend(title="Discount Applied", labels=["No Discount", "Discount Applied"])
plt.tight_layout()
plt.savefig("w2_plot3_payment_discount.png")
plt.close()

# ----------------------------------------------------
# Visualization 4: Demographic Density (KDE)
# ----------------------------------------------------
plt.figure(figsize=(7, 4), dpi=300)
sns.kdeplot(
    data=df,
    x="Age",
    hue="Device",
    fill=True,
    common_norm=False,
    palette="mako",
    alpha=0.3,
)
plt.title(
    "Demographic Age Distribution Across Device Platforms",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("Customer Age")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("w2_plot4_demographic_device.png")
plt.close()

# ----------------------------------------------------
# Visualization 5: Correlation Matrix
# ----------------------------------------------------
plt.figure(figsize=(6, 4.5), dpi=300)
corr_w2 = df[[
    "Age",
    "Monthly_Visits",
    "Discount_Applied",
    "Purchase_Amount",
    "Satisfaction_Score",
]].corr()
sns.heatmap(
    corr_w2,
    annot=True,
    cmap="vlag",
    fmt=".2f",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title(
    "Correlation Matrix of Behavioral & Financial Indicators",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("w2_plot5_correlation_heatmap.png")
plt.close()

print("Week 2 Visualizations successfully generated and exported.")