
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    classification_report,
    confusion_matrix,
    make_scorer
)

# STEP 1: LOAD DATASET

df = pd.read_csv("data/featured_telco_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# STEP 2: SEPARATE INPUT AND TARGET

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# STEP 3: SPLIT DATASET
# Same split as scaling.py

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# STEP 4: IDENTIFY FEATURE TYPES

numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

# STEP 5: DEFINE SCALERS

scalers = {
    "No Scaling": "passthrough",
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

# STEP 6: DEFINE MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# STEP 7: DEFINE CROSS-VALIDATION

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# STEP 8: DEFINE EVALUATION METRICS

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f2": make_scorer(fbeta_score, beta=2)
}

# STEP 9: COMPARE ALL PIPELINES

results = []
pipelines = {}

for scaler_name, scaler in scalers.items():

    for model_name, classifier in models.items():

        print(
            "\nEvaluating:",
            scaler_name, "-", model_name
        )

        preprocessor = ColumnTransformer([
            ("num", scaler, numeric_features),
            ("cat", OneHotEncoder(
                handle_unknown="ignore"
            ), categorical_features)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", classifier)
        ])

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results.append({
            "Scaler": scaler_name,
            "Model": model_name,
            "CV_Accuracy": scores["test_accuracy"].mean(),
            "CV_Precision": scores["test_precision"].mean(),
            "CV_Recall": scores["test_recall"].mean(),
            "CV_F2": scores["test_f2"].mean(),
            "CV_F2_Std": scores["test_f2"].std()
        })

        pipelines[(scaler_name, model_name)] = pipeline

# STEP 10: DISPLAY CV RESULTS

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV_F2",
    ascending=False
)

print("\n5-FOLD CROSS-VALIDATION RESULTS:")
print(results_df.to_string(index=False))

os.makedirs("outputs", exist_ok=True)

results_df.to_csv(
    "outputs/cross_validation_results.csv",
    index=False
)

# STEP 11: SELECT BEST MODEL
# Selection uses training CV results only

best_row = results_df.iloc[0]

best_scaler = best_row["Scaler"]
best_model_name = best_row["Model"]

print("\nBest scaler:", best_scaler)
print("Best model:", best_model_name)

best_pipeline = pipelines[
    (best_scaler, best_model_name)
]

# STEP 12: TRAIN ON FULL TRAINING SET

best_pipeline.fit(X_train, y_train)

# STEP 13: EVALUATE ON HOLD-OUT TEST SET
# Uses the default 0.5 threshold.
# This is a baseline, not the tuned-threshold result.

predictions = best_pipeline.predict(X_test)

print("\nHOLD-OUT TEST RESULTS (DEFAULT THRESHOLD)")

print(
    "Accuracy:",
    accuracy_score(y_test, predictions)
)

print(
    "Precision:",
    precision_score(y_test, predictions, zero_division=0)
)

print(
    "Recall:",
    recall_score(y_test, predictions, zero_division=0)
)

print(
    "F2 Score:",
    fbeta_score(
        y_test, predictions,
        beta=2,
        zero_division=0
    )
)

# STEP 14: CLASSIFICATION REPORT

report = classification_report(
    y_test,
    predictions,
    target_names=["No Churn", "Churn"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT:")
print(report)

with open(
    "outputs/classification_report.txt",
    "w"
) as file:
    file.write(report)

# STEP 15: CONFUSION MATRIX

cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Customer Churn Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nModel evaluation completed!")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)