import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv(
    "data/Telco-Customer-Churn-Cleaned.csv"
)

print("Dataset loaded successfully!")
print("Shape:", df.shape)


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
# 4. SEPARATE FEATURES AND TARGET
# =========================================

X = df.drop("Churn", axis=1)
y = df["Churn"]


# =========================================
# 5. IDENTIFY COLUMNS
# =========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)


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
# 7. CREATE DECISION TREE
# =========================================

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)


# =========================================
# 8. CREATE PIPELINE
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


print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# =========================================
# 10. TRAIN MODEL
# =========================================

print("\nTraining Decision Tree...")

model.fit(X_train, y_train)

print("Training completed!")


# =========================================
# 11. MAKE PREDICTIONS
# =========================================

y_pred = model.predict(X_test)


# =========================================
# 12. ACCURACY
# =========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n=================================")
print("MODEL ACCURACY")
print("=================================")

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# =========================================
# 13. CLASSIFICATION REPORT
# =========================================

print("\n=================================")
print("CLASSIFICATION REPORT")
print("=================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Churn",
            "Churn"
        ]
    )
)


# =========================================
# 14. CONFUSION MATRIX
# =========================================

print("\n=================================")
print("CONFUSION MATRIX")
print("=================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# =========================================
# 15. DISPLAY CONFUSION MATRIX
# =========================================

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
).plot()

plt.title("Decision Tree - Confusion Matrix")

plt.show()
# =========================================
# SAVE MODEL
# =========================================

joblib.dump(
    model,
    "models/churn_decision_tree.pkl"
)

print("\nModel saved successfully!")
print("Location: models/churn_decision_tree.pkl")