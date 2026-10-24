
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    fbeta_score
)

# STEP 1: LOAD DATASET

df = pd.read_csv("data/featured_telco_churn.csv")

os.makedirs("outputs", exist_ok=True)

sns.set_theme(style="whitegrid")

print("Dataset loaded successfully!")

# =========================================
# GRAPH 1: CHURN BY CONTRACT TYPE
# =========================================

contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

contract_churn = contract_churn.reindex(
    columns=["No", "Yes"]
)

contract_churn.plot(
    kind="bar",
    stacked=True,
    figsize=(9, 5)
)

plt.title("Churn Percentage by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Percentage of Customers")
plt.xticks(rotation=0)
plt.legend(title="Churn")
plt.tight_layout()

plt.savefig(
    "outputs/01_contract_churn.png",
    dpi=300
)
plt.close()

print("Graph 1 saved!")

# =========================================
# GRAPH 2: MONTHLY CHARGES KDE PLOT
# =========================================

plt.figure(figsize=(9, 5))

sns.kdeplot(
    data=df,
    x="MonthlyCharges",
    hue="Churn",
    fill=True,
    common_norm=False
)

plt.title("Monthly Charges Distribution by Churn")
plt.xlabel("Monthly Charges")
plt.ylabel("Density")
plt.tight_layout()

plt.savefig(
    "outputs/02_monthly_charges_kde.png",
    dpi=300
)
plt.close()

print("Graph 2 saved!")

# =========================================
# GRAPH 3: TENURE VIOLIN PLOT
# =========================================

plt.figure(figsize=(10, 6))

sns.violinplot(
    data=df,
    x="InternetService",
    y="tenure",
    hue="Churn",
    split=True,
    inner="quart"
)

plt.title("Tenure Distribution by Internet Service and Churn")
plt.xlabel("Internet Service")
plt.ylabel("Tenure (Months)")
plt.tight_layout()

plt.savefig(
    "outputs/03_tenure_violin.png",
    dpi=300
)
plt.close()

print("Graph 3 saved!")

# =========================================
# PREPARE DATA FOR GRAPHS 4 AND 5
# =========================================

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(
        handle_unknown="ignore"
    ), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ))
])

# Generate out-of-fold probabilities from training data
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

probabilities = cross_val_predict(
    model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

# =========================================
# GRAPH 4: PRECISION-RECALL CURVE
# =========================================

precision, recall, thresholds = precision_recall_curve(
    y_train,
    probabilities
)

plt.figure(figsize=(9, 5))

plt.plot(
    recall,
    precision,
    label="Logistic Regression"
)

# Mark selected thresholds
for t in [0.3, 0.5, 0.7]:
    predictions = (probabilities >= t).astype(int)

    p = (
        ((predictions == 1) & (y_train == 1)).sum()
        / max((predictions == 1).sum(), 1)
    )

    r = (
        ((predictions == 1) & (y_train == 1)).sum()
        / max((y_train == 1).sum(), 1)
    )

    plt.scatter(r, p)
    plt.annotate(
        f"t={t}",
        (r, p),
        xytext=(6, 6),
        textcoords="offset points"
    )

plt.title("Precision-Recall Curve with Thresholds")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()

plt.savefig(
    "outputs/04_precision_recall_curve.png",
    dpi=300
)
plt.close()

print("Graph 4 saved!")

# =========================================
# GRAPH 5: RETENTION COST-BENEFIT CURVE
# =========================================

# Illustrative campaign assumptions
contact_cost = 10
retention_value = 100
campaign_success_rate = 0.30

threshold_values = np.arange(0.1, 1.0, 0.1)

benefits = []

for threshold in threshold_values:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    targeted = predictions.sum()

    true_churners = (
        (predictions == 1) & (y_train == 1)
    ).sum()

    expected_benefit = (
        true_churners
        * campaign_success_rate
        * retention_value
    )

    total_cost = targeted * contact_cost

    net_benefit = expected_benefit - total_cost

    benefits.append(net_benefit)

plt.figure(figsize=(9, 5))

plt.plot(
    threshold_values,
    benefits,
    marker="o"
)

plt.axhline(0, linestyle="--")

plt.title("Estimated Retention Campaign Net Benefit")
plt.xlabel("Decision Threshold")
plt.ylabel("Estimated Net Benefit (Illustrative Units)")
plt.tight_layout()

plt.savefig(
    "outputs/05_cost_benefit_curve.png",
    dpi=300
)
plt.close()

print("Graph 5 saved!")

print("\nAll five visualizations completed!")
