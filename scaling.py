
import pandas as pd

from sklearn.model_selection import train_test_split
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
    fbeta_score
)

# STEP 1: LOAD DATASET
df = pd.read_csv("data/featured_telco_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# STEP 2: SEPARATE INPUT AND TARGET
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# STEP 3: SPLIT DATASET (80% TRAIN, 20% TEST)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
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

# STEP 7: TRAIN AND COMPARE MODELS
results = []

for scaler_name, scaler in scalers.items():

    for model_name, model in models.items():

        print("\nTraining:", scaler_name, "-", model_name)

        preprocessor = ColumnTransformer([
            ("num", scaler, numeric_features),
            ("cat", OneHotEncoder(
                handle_unknown="ignore"
            ), categorical_features)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        results.append({
            "Scaler": scaler_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F2_Score": fbeta_score(
                y_test, predictions,
                beta=2, zero_division=0
            )
        })

# STEP 8: DISPLAY RESULTS
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON RESULTS:")
print(results_df.to_string(index=False))

# STEP 9: SAVE RESULTS
results_df.to_csv(
    "outputs/scaling_comparison.csv",
    index=False
)

print("\nScaling comparison completed!")
