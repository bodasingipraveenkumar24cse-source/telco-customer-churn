import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv(
    "data/Telco-Customer-Churn-Cleaned.csv"
)


# =========================================
# 2. REMOVE CUSTOMER ID
# =========================================

df = df.drop("customerID", axis=1)


# =========================================
# 3. CONVERT TARGET
# =========================================

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})


# =========================================
# 4. FEATURES AND TARGET
# =========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# =========================================
# 5. IDENTIFY COLUMN TYPES
# =========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# =========================================
# 6. PREPROCESSING
# =========================================

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


# =========================================
# 7. DECISION TREE
# =========================================

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# =========================================
# 8. PIPELINE
# =========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", decision_tree)
    ]
)


# =========================================
# 9. TRAIN / TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================================
# 10. TRAIN
# =========================================

model.fit(X_train, y_train)


# =========================================
# 11. GET FEATURE NAMES
# =========================================

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)


# =========================================
# 12. GET FEATURE IMPORTANCE
# =========================================

importance = (
    model
    .named_steps["classifier"]
    .feature_importances_
)


# =========================================
# 13. CREATE DATAFRAME
# =========================================

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})


# =========================================
# 14. SORT FEATURES
# =========================================

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# =========================================
# 15. DISPLAY TOP FEATURES
# =========================================

print("\n========================================")
print("TOP CHURN DRIVERS")
print("========================================")

print(
    importance_df.head(15).to_string(
        index=False
    )
)


# =========================================
# 16. PLOT TOP 15
# =========================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.title(
    "Top Factors Influencing Customer Churn"
)

plt.tight_layout()

plt.show()