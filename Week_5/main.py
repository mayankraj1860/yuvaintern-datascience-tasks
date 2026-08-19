"""
Week 5 Task: Comprehensive Data Science Project Reporting and Strategic Recommendations
Internship Domain: Data Science with Python
Platform: Yuva Intern
Author: Mayank Raj
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Visual formatting
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

print("=" * 65)
print("WEEK 5: COMPREHENSIVE DATA SCIENCE SYNTHESIS & STRATEGY PIPELINE")
print("=" * 65)

# ----------------------------------------------------
# Visualization 1: Project Maturity & Analytics Roadmap
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
weeks = [
    "Week 1:\nAcquisition & Cleaning",
    "Week 2:\nEDA & Storytelling",
    "Week 3:\nHypothesis Testing",
    "Week 4:\nML Modeling",
    "Week 5:\nStrategic Synthesis",
]
impact_scores = [65, 78, 85, 96, 99]

ax.plot(
    weeks,
    impact_scores,
    marker="o",
    color="#1f4e79",
    linewidth=2.5,
    markersize=8,
)
for i, txt in enumerate(impact_scores):
  ax.annotate(
      f"{txt}% Maturity",
      (weeks[i], impact_scores[i] + 1.5),
      ha="center",
      fontweight="bold",
      color="#1f4e79",
      fontsize=9,
  )

ax.set_ylim(50, 105)
ax.set_title(
    "End-to-End Data Science Project Maturity & Impact Framework",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
ax.set_ylabel("Analytics Maturity Index (%)")
plt.tight_layout()
plt.savefig("w5_plot1_project_maturity.png")
plt.close()
print("Saved: w5_plot1_project_maturity.png")

# ----------------------------------------------------
# Visualization 2: Strategic Priority Matrix (Effort vs ROI)
# ----------------------------------------------------
plt.figure(figsize=(7.5, 4.5), dpi=300)
recommendations = [
    (
        "Personalized Cashback Programs",
        3.2,
        8.8,
        "High Impact / Low-Med Effort",
    ),
    ("Enterprise Dedicated Pipeline", 4.5, 9.2, "High Impact / Med Effort"),
    ("Mobile-First App Optimization", 2.8, 8.0, "High Impact / Low Effort"),
    ("Standard Discount Promotions", 5.0, 4.2, "Low Impact / Med Effort"),
    ("Proactive ML Retention Alerting", 4.0, 9.5, "High Impact / Med Effort"),
]
df_rec = pd.DataFrame(
    recommendations,
    columns=[
        "Initiative",
        "Implementation_Effort",
        "Projected_ROI_Score",
        "Category",
    ],
)

sns.scatterplot(
    data=df_rec,
    x="Implementation_Effort",
    y="Projected_ROI_Score",
    hue="Category",
    s=250,
    palette="viridis",
)
for i in range(len(df_rec)):
  plt.annotate(
      df_rec["Initiative"][i],
      (
          df_rec["Implementation_Effort"][i] + 0.1,
          df_rec["Projected_ROI_Score"][i] - 0.1,
      ),
      fontsize=8.5,
      fontweight="bold",
  )

plt.axvline(3.8, color="gray", linestyle="--", alpha=0.5)
plt.axhline(6.5, color="gray", linestyle="--", alpha=0.5)
plt.title(
    "Strategic Recommendations: Value Creation vs. Implementation Effort",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Implementation Complexity / Effort (1-10 Scale)")
plt.ylabel("Projected Commercial ROI (1-10 Scale)")
plt.xlim(1.5, 7.0)
plt.ylim(2.5, 11.0)
plt.legend(loc="lower left", fontsize=8.5)
plt.tight_layout()
plt.savefig("w5_plot2_strategy_matrix.png")
plt.close()
print("Saved: w5_plot2_strategy_matrix.png")

print(
    "\nWeek 5 comprehensive reporting assets and visual artifacts generated"
    " successfully."
)