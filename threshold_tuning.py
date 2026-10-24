
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score
)

# STEP 1: LOAD DATASET
df = pd.read_csv("data/featured_telco_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# STEP 2: SEPARATE FEATURES AND TARGET
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# STEP 3: SPLIT DATASET
# Keep 20% for final testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Use part of training data for validation
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

print("Training records:", len(X_fit))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))

# STEP 4: IDENTIFY COLUMN TYPES
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# STEP 5: CREATE PREPROCESSOR
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(
        handle_unknown="ignore"
    ), categorical_features)
])

# STEP 6: CREATE LOGISTIC REGRESSION MODEL
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ))
])

# STEP 7: TRAIN MODEL
model.fit(X_fit, y_fit)

print("\nModel training completed!")

# STEP 8: GET VALIDATION PROBABILITIES
probabilities = model.predict_proba(X_val)[:, 1]

# STEP 9: COMPARE THRESHOLDS
thresholds = np.arange(0.1, 1.0, 0.1)

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    results.append({
        "Threshold": round(float(threshold), 1),
        "Accuracy": accuracy_score(y_val, predictions),
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F2_Score": fbeta_score(
            y_val, predictions,
            beta=2, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

print("\nTHRESHOLD COMPARISON:")
print(results_df.to_string(index=False))

# STEP 10: FIND BEST VALIDATION THRESHOLD
best_index = results_df["F2_Score"].idxmax()
best_row = results_df.loc[best_index]

best_threshold = float(best_row["Threshold"])

print("\nBest validation threshold:", best_threshold)
print("Best validation F2 Score:", best_row["F2_Score"])

# STEP 11: SAVE RESULTS
os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/threshold_comparison.csv",
    index=False
)

# STEP 12: CREATE GRAPH
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="s",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F2_Score"],
    marker="^",
    label="F2 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold = {best_threshold:.1f}"
)

plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Logistic Regression Threshold Tuning")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "outputs/threshold_comparison.png",
    dpi=300
)

plt.show()

print("\nThreshold tuning completed!")
