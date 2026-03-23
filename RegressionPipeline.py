"""
Regression Pipeline with scikit-learn
======================================
Dataset : California Housing (built-in sklearn dataset)
Target  : Median house value (in $100,000s)
Models  : Linear Regression, Ridge, Random Forest, Gradient Boosting
Metrics : RMSE, MAE, R²
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  California Housing — Regression Pipeline")
print("=" * 55)

housing = fetch_california_housing(as_frame=True)
df = housing.frame                    # full DataFrame (features + target)

print(f"\n Dataset shape : {df.shape}")
print(f" Target column : MedHouseVal (median house value × $100k)")
print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe().round(2))

# Check for missing values
missing = df.isnull().sum().sum()
print(f"\n Missing values: {missing}")

# ─────────────────────────────────────────────
# 2. SPLIT FEATURES & TARGET
# ─────────────────────────────────────────────
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

print(f"\n Features ({X.shape[1]}): {list(X.columns)}")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT  (80 / 20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n Train size : {X_train.shape[0]:,} rows")
print(f" Test size  : {X_test.shape[0]:,} rows")

# ─────────────────────────────────────────────
# 4. DEFINE MODELS (each wrapped in a Pipeline)
# ─────────────────────────────────────────────
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        )),
    ]),
}

# ─────────────────────────────────────────────
# 5. TRAIN & EVALUATE ALL MODELS
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("  Training & Evaluating Models")
print("─" * 55)

results = {}

for name, pipeline in models.items():
    # — Train
    pipeline.fit(X_train, y_train)

    # — Predict
    y_pred = pipeline.predict(X_test)

    # — Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # — 5-fold CV R² on training set
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=5, scoring="r2", n_jobs=-1)

    results[name] = {
        "RMSE"   : rmse,
        "MAE"    : mae,
        "R²"     : r2,
        "CV R²"  : cv_scores.mean(),
        "CV Std" : cv_scores.std(),
        "y_pred" : y_pred,
    }

    print(f"\n {name}")
    print(f"   RMSE    : {rmse:.4f}")
    print(f"   MAE     : {mae:.4f}")
    print(f"   R²      : {r2:.4f}")
    print(f"   CV R²   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────
summary = pd.DataFrame({
    name: {
        "RMSE"  : v["RMSE"],
        "MAE"   : v["MAE"],
        "R²"    : v["R²"],
        "CV R²" : v["CV R²"],
    }
    for name, v in results.items()
}).T.round(4)

print("\n" + "=" * 55)
print("   Summary Table")
print("=" * 55)
print(summary.to_string())

best_model = summary["R²"].idxmax()
print(f"\n Best model by R²: {best_model}  (R² = {summary.loc[best_model, 'R²']:.4f})")

# ─────────────────────────────────────────────
# 7. FEATURE IMPORTANCE  (best tree model)
# ─────────────────────────────────────────────
best_tree_name = "Gradient Boosting"   # or "Random Forest"
best_tree_pipe = models[best_tree_name]
importances = best_tree_pipe.named_steps["model"].feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print(f"\n Feature importances ({best_tree_name}):")
print(feat_imp.round(4).to_string())

# ─────────────────────────────────────────────
# 8. PLOTS
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Regression Pipeline — California Housing", fontsize=15, fontweight="bold")

# ── Plot 1 : Model comparison (R²)
ax = axes[0, 0]
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = ax.barh(summary.index, summary["R²"], color=colors)
ax.set_xlim(0, 1)
ax.set_xlabel("R² Score (higher = better)")
ax.set_title("Model Comparison — R²")
for bar, val in zip(bars, summary["R²"]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)

# ── Plot 2 : Actual vs Predicted (best model)
ax = axes[0, 1]
y_pred_best = results[best_model]["y_pred"]
ax.scatter(y_test, y_pred_best, alpha=0.3, s=10, color="#4C72B0")
lims = [min(y_test.min(), y_pred_best.min()),
        max(y_test.max(), y_pred_best.max())]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Value")
ax.set_ylabel("Predicted Value")
ax.set_title(f"Actual vs Predicted — {best_model}")
ax.legend(fontsize=8)

# ── Plot 3 : Residuals (best model)
ax = axes[1, 0]
residuals = y_test - y_pred_best
ax.scatter(y_pred_best, residuals, alpha=0.3, s=10, color="#C44E52")
ax.axhline(0, color="black", lw=1.5, linestyle="--")
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Residual (Actual − Predicted)")
ax.set_title(f"Residual Plot — {best_model}")

# ── Plot 4 : Feature Importances
ax = axes[1, 1]
feat_imp.plot(kind="barh", ax=ax, color="#55A868")
ax.set_xlabel("Importance")
ax.set_title(f"Feature Importances — {best_tree_name}")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("regression_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n Plot saved to regression_results.png")

# ─────────────────────────────────────────────
# 9. MAKE A SAMPLE PREDICTION
# ─────────────────────────────────────────────
print("\n" + "─" * 55)
print("   Sample Prediction (first test row)")
print("─" * 55)

sample = X_test.iloc[[0]]
actual = y_test.iloc[0]
predicted = models[best_model].predict(sample)[0]

print(f"Features:\n{sample.to_string()}")
print(f"\nActual value    : ${actual * 100_000:,.0f}")
print(f"Predicted value : ${predicted * 100_000:,.0f}")
print(f"Error           : ${abs(actual - predicted) * 100_000:,.0f}")
