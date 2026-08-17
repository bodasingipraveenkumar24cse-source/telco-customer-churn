import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_csv(
    "data/Telco-Customer-Churn-Cleaned.csv"
)

print("Dataset loaded:", df.shape)


# =========================================
# 2. SELECT CUSTOMER BEHAVIOR FEATURES
# =========================================

features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X = df[features].copy()


# =========================================
# 3. SCALE FEATURES
# =========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# =========================================
# 4. CREATE K-MEANS MODEL
# =========================================

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)


# =========================================
# 5. TRAIN CLUSTERING MODEL
# =========================================

df["Persona"] = kmeans.fit_predict(X_scaled)


# =========================================
# 6. DISPLAY CLUSTER COUNTS
# =========================================

print("\n========================================")
print("CUSTOMERS IN EACH PERSONA")
print("========================================")

print(
    df["Persona"].value_counts().sort_index()
)


# =========================================
# 7. ANALYZE EACH PERSONA
# =========================================

print("\n========================================")
print("PERSONA CHARACTERISTICS")
print("========================================")

persona_summary = df.groupby("Persona")[
    features
].mean()

print(persona_summary)


# =========================================
# 8. CHURN RATE BY PERSONA
# =========================================

df["ChurnNumeric"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

churn_by_persona = df.groupby(
    "Persona"
)["ChurnNumeric"].mean() * 100

print("\n========================================")
print("CHURN RATE BY PERSONA")
print("========================================")

print(churn_by_persona)


# =========================================
# 9. VISUALIZE PERSONAS
# =========================================

plt.figure(figsize=(9, 6))

plt.scatter(
    df["tenure"],
    df["MonthlyCharges"],
    c=df["Persona"]
)

plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")

plt.title(
    "Customer Personas - Tenure vs Monthly Charges"
)

plt.colorbar(
    label="Persona"
)

plt.tight_layout()

plt.show()
# =========================================
# 10. ASSIGN MEANINGFUL PERSONA NAMES
# =========================================

persona_names = {
    0: "New Budget Customers",
    1: "Long-Term High-Value Customers",
    2: "High-Risk New Customers",
    3: "Long-Term Loyal Budget Customers"
}

df["PersonaName"] = df["Persona"].map(persona_names)


# =========================================
# 11. SAVE PERSONA DATA
# =========================================

df.to_csv(
    "data/customer_personas.csv",
    index=False
)

print("\n========================================")
print("PERSONA DATA SAVED")
print("========================================")

print(
    "Saved to: data/customer_personas.csv"
)
# =========================================
# SAVE K-MEANS MODEL AND SCALER
# =========================================

joblib.dump(
    kmeans,
    "models/persona_kmeans.pkl"
)

joblib.dump(
    scaler,
    "models/persona_scaler.pkl"
)

print("\nPersona model saved successfully!")
print("K-Means model: models/persona_kmeans.pkl")
print("Scaler: models/persona_scaler.pkl")
