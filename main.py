
import pandas as pd

# Step 1: Read the dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Step 2: Display first five customers
print("First five rows:")
print(df.head())

# Step 3: Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Step 4: Display column names
print("\nColumn Names:")
print(df.columns)

# Step 5: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 6: Check churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())


# ==================================
# STEP 7: DATA CLEANING
# ==================================

# 1. Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 2. Check missing values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# 3. Handle missing TotalCharges
# Customers with zero tenure may not have accumulated charges
df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isna()),
    "TotalCharges"
] = 0

# Fill any remaining missing TotalCharges with median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# 4. Check duplicate customer IDs
print("\nDuplicate Customer IDs:")
print(df["customerID"].duplicated().sum())

# 5. Check data types
print("\nData Types After Cleaning:")
print(df.dtypes)

# 6. Check final missing values
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nData cleaning completed!")


# ==================================
# STEP 8: OUTLIER DETECTION USING IQR
# ==================================

columns = ["tenure", "MonthlyCharges", "TotalCharges"]

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    print(f"\nColumn: {col}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Number of outliers: {len(outliers)}")


# Save cleaned dataset
df.to_csv("data/cleaned_telco_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")


# =====================================
# STEP 9: FEATURE ENGINEERING
# =====================================

import numpy as np

# 1. TENURE COHORTS
# Group customers based on their tenure

df["TenureCohort"] = pd.cut(
    df["tenure"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12 months", "13-24 months",
            "25-48 months", "49-72 months"]
)

print("\nTenure Cohort Distribution:")
print(df["TenureCohort"].value_counts())


# 2. AVERAGE MONTHLY SPEND RATIO
# Compare actual total charges with expected charges

df["AvgMonthlySpendRatio"] = (
    df["TotalCharges"] /
    (df["MonthlyCharges"] * df["tenure"]).replace(0, np.nan)
)

df["AvgMonthlySpendRatio"] = (
    df["AvgMonthlySpendRatio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

print("\nAverage Monthly Spend Ratio:")
print(df["AvgMonthlySpendRatio"].head())


# 3. SERVICE DENSITY
# Count the number of active services

service_columns = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["ServiceDensity"] = (
    df[service_columns] == "Yes"
).sum(axis=1)

print("\nService Density:")
print(df["ServiceDensity"].head())


# Display newly created features
print("\nNew Features:")
print(df[[
    "TenureCohort",
    "AvgMonthlySpendRatio",
    "ServiceDensity"
]].head())


# Save the updated dataset
df.to_csv(
    "data/featured_telco_churn.csv",
    index=False
)

print("\nFeature engineering completed!")
