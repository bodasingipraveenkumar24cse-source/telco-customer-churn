import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/Telco-Customer-Churn-Cleaned.csv")

print("========== DATASET INFORMATION ==========")

# Number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# First 5 records
print("\nFirst 5 records:")
print(df.head())

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())

# Churn distribution
print("\nChurn distribution:")
print(df["Churn"].value_counts())

# Churn percentage
print("\nChurn percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)