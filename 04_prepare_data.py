import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


# -----------------------------------------
# 1. LOAD DATA
# -----------------------------------------

df = pd.read_csv(
    "data/Telco-Customer-Churn-Cleaned.csv"
)

print("Original shape:")
print(df.shape)


# -----------------------------------------
# 2. REMOVE CUSTOMER ID
# -----------------------------------------

df = df.drop("customerID", axis=1)


# -----------------------------------------
# 3. CONVERT TARGET
# -----------------------------------------

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# -----------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# -----------------------------------------

X = df.drop("Churn", axis=1)

y = df["Churn"]


# -----------------------------------------
# 5. FIND CATEGORICAL COLUMNS
# -----------------------------------------

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


# -----------------------------------------
# 6. FIND NUMERICAL COLUMNS
# -----------------------------------------

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


# -----------------------------------------
# 7. CREATE PREPROCESSOR
# -----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)


# -----------------------------------------
# 8. SPLIT DATA
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining records:")
print(X_train.shape[0])


print("\nTesting records:")
print(X_test.shape[0])


print("\nData preparation completed!")