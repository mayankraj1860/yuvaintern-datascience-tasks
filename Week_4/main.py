"""
Week 4 Task: Machine Learning Model Development and Evaluation
Internship Domain: Data Science with Python
Platform: Yuva Intern
Author: Mayank Raj
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Visual aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

# ==========================================
# 1. DATASET GENERATION / PREPARATION
# ==========================================
np.random.seed(42)
n_samples = 1000

age = np.random.normal(38, 12, n_samples).clip(18, 70).astype(int)
income = (age * 1200 + np.random.normal(20000, 10000, n_samples)).clip(
    20000, 140000
)
spending_score = np.random.randint(1, 100, n_samples)
monthly_visits = np.random.poisson(8, n_samples).clip(1, 25)
discount_applied = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])
satisfaction_rating = np.random.choice(
    [1, 2, 3, 4, 5], n_samples, p=[0.15, 0.20, 0.30, 0.20, 0.15]
)

# Churn generation
z = (
    0.8
    - 0.035 * spending_score
    - 0.18 * monthly_visits
    - 0.60 * (satisfaction_rating - 3)
    + 0.015 * (age - 35)
    + np.random.normal(0, 0.6, n_samples)
)
prob = 1 / (1 + np.exp(-z))
churn = (prob > 0.5).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Annual_Income": income,
    "Spending_Score": spending_score,
    "Monthly_Visits": monthly_visits,
    "Discount_Applied": discount_applied,
    "Satisfaction_Rating": satisfaction_rating,
    "Churn": churn,
})

df.to_csv("customer_churn_ml_dataset.csv", index=False)

print("=" * 60)
print("WEEK 4: MACHINE LEARNING PIPELINE EXECUTION")
print("=" * 60)
print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['Churn'].value_counts(normalize=True)}")

# ==========================================
# 2. PREPROCESSING & TRAIN-TEST SPLIT
# ==========================================
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODEL TRAINING & BENCHMARKING
# ==========================================
# Baseline: Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
y_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]

# Ensemble: Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation Summary ---")
print(
    f"Logistic Regression: Accuracy={accuracy_score(y_test, y_pred_lr):.3f},"
    f" AUC={roc_auc_score(y_test, y_proba_lr):.3f}"
)
print(
    f"Random Forest:       Accuracy={accuracy_score(y_test, y_pred_rf):.3f},"
    f" AUC={roc_auc_score(y_test, y_proba_rf):.3f}"
)

# ==========================================
# 4. VISUALIZATIONS & METRICS PLOTTING
# ==========================================
# 1. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_rf = confusion_matrix(y_test, y_pred_rf)

sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
axes[0].set_title(
    "Logistic Regression Confusion Matrix\n(Accuracy:"
    f" {accuracy_score(y_test, y_pred_lr)*100:.1f}%)",
    fontsize=11,
    fontweight="bold",
)
axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")
axes[0].set_xticklabels(["Retained (0)", "Churned (1)"])
axes[0].set_yticklabels(["Retained (0)", "Churned (1)"])

sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens", ax=axes[1], cbar=False)
axes[1].set_title(
    "Random Forest Confusion Matrix\n(Accuracy:"
    f" {accuracy_score(y_test, y_pred_rf)*100:.1f}%)",
    fontsize=11,
    fontweight="bold",
)
axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")
axes[1].set_xticklabels(["Retained (0)", "Churned (1)"])
axes[1].set_yticklabels(["Retained (0)", "Churned (1)"])

plt.tight_layout()
plt.savefig("w4_plot1_confusion_matrices.png")
plt.close()
print("Saved: w4_plot1_confusion_matrices.png")

# 2. ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(7, 4.5), dpi=300)
plt.plot(
    fpr_lr,
    tpr_lr,
    color="#2980b9",
    lw=2,
    label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_proba_lr):.3f})",
)
plt.plot(
    fpr_rf,
    tpr_rf,
    color="#27ae60",
    lw=2.5,
    label=f"Random Forest (AUC = {roc_auc_score(y_test, y_proba_rf):.3f})",
)
plt.plot(
    [0, 1],
    [0, 1],
    color="#7f8c8d",
    linestyle="--",
    label="Random Baseline (AUC = 0.500)",
)
plt.title(
    "Receiver Operating Characteristic (ROC) Curve Comparison",
    fontsize=12,
    fontweight="bold",
)
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Recall)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("w4_plot2_roc_curves.png")
plt.close()
print("Saved: w4_plot2_roc_curves.png")

# 3. Feature Importance
plt.figure(figsize=(7, 4), dpi=300)
feat_imp = pd.Series(
    rf_model.feature_importances_, index=X.columns
).sort_values(ascending=True)
feat_imp.plot(kind="barh", color="#34495e")
plt.title(
    "Random Forest Feature Importance Analysis", fontsize=12, fontweight="bold"
)
plt.xlabel("Gini Relative Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("w4_plot3_feature_importance.png")
plt.close()
print("Saved: w4_plot3_feature_importance.png")

print("\nModel pipeline completed successfully.")