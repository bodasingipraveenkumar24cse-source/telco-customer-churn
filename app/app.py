import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/Telco-Customer-Churn-Cleaned.csv")


@st.cache_data
def load_personas():
    return pd.read_csv("data/customer_personas.csv")


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/churn_decision_tree.pkl")


@st.cache_resource
def load_persona_model():
    return joblib.load("models/persona_kmeans.pkl")


@st.cache_resource
def load_persona_scaler():
    return joblib.load("models/persona_scaler.pkl")


# =========================================================
# LOAD EVERYTHING
# =========================================================

df = load_data()
persona_df = load_personas()
model = load_model()
persona_model = load_persona_model()
persona_scaler = load_persona_scaler()


# =========================================================
# PERSONA NAMES
# =========================================================

persona_names = {
    0: "Budget New Customers",
    1: "Loyal High-Value Customers",
    2: "High-Risk Customers",
    3: "Loyal Low-Cost Customers"
}


# =========================================================
# PREPARE PERSONA DATA
# =========================================================

dashboard_personas = persona_df.copy()

dashboard_personas["Persona Name"] = (
    dashboard_personas["Persona"]
    .map(persona_names)
    .fillna(dashboard_personas["Persona"].astype(str))
)

dashboard_personas["ChurnNumeric"] = (
    dashboard_personas["Churn"]
    .map({"No": 0, "Yes": 1})
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Telecom Analytics")

st.sidebar.write(
    "Telecom Churn Driver Discovery & Persona Profiler"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🤖 Predict Churn",
        "👥 Customer Personas",
        "📈 Model Evaluation"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine Learning + Customer Segmentation"
)


# =========================================================
# PAGE 1 - DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.title("📊 Telecom Churn Analytics")

    st.subheader(
        "Telecom Churn Driver Discovery & Persona Profiler"
    )

    st.write(
        "This system analyzes telecom customers, "
        "identifies churn drivers, predicts churn risk "
        "and groups customers into meaningful personas."
    )

    st.divider()

    # =====================================================
    # CUSTOMER METRICS
    # =====================================================

    total_customers = len(df)

    churned_customers = (
        df["Churn"] == "Yes"
    ).sum()

    retained_customers = (
        df["Churn"] == "No"
    ).sum()

    churn_rate = (
        churned_customers / total_customers * 100
    )

    retention_rate = (
        retained_customers / total_customers * 100
    )

    st.subheader("📌 Customer Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )

    with col3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.2f}%"
        )

    with col4:
        st.metric(
            "Retention Rate",
            f"{retention_rate:.2f}%"
        )

    st.divider()

    # =====================================================
    # FINANCIAL METRICS
    # =====================================================

    st.subheader("💰 Customer Financial Metrics")

    avg_monthly = df["MonthlyCharges"].mean()
    avg_total = df["TotalCharges"].mean()
    avg_tenure = df["tenure"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Monthly Charges",
            f"${avg_monthly:.2f}"
        )

    with col2:
        st.metric(
            "Average Total Charges",
            f"${avg_total:.2f}"
        )

    with col3:
        st.metric(
            "Average Tenure",
            f"{avg_tenure:.1f} months"
        )

    st.divider()

    # =====================================================
    # CHURN DISTRIBUTION
    # =====================================================

    st.header("📊 Churn Distribution")

    churn_counts = df["Churn"].value_counts()

    churn_chart = pd.DataFrame({
        "Status": ["No Churn", "Churn"],
        "Customers": [
            churn_counts.get("No", 0),
            churn_counts.get("Yes", 0)
        ]
    })

    st.bar_chart(
        churn_chart.set_index("Status")
    )

    st.divider()

    # =====================================================
    # CONTRACT CHURN
    # =====================================================

    st.header("📋 Churn by Contract Type")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(contract_churn)

    st.divider()

    # =====================================================
    # RISK DISTRIBUTION
    # =====================================================

    st.header("🚦 Customer Risk Distribution")

    dashboard_X = df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    try:

        probabilities = (
            model.predict_proba(dashboard_X)[:, 1] * 100
        )

        high_risk = (probabilities >= 70).sum()

        medium_risk = (
            (probabilities >= 40) &
            (probabilities < 70)
        ).sum()

        low_risk = (probabilities < 40).sum()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🔴 High Risk",
                f"{high_risk:,}"
            )

        with col2:
            st.metric(
                "🟠 Medium Risk",
                f"{medium_risk:,}"
            )

        with col3:
            st.metric(
                "🟢 Low Risk",
                f"{low_risk:,}"
            )

        risk_df = pd.DataFrame({
            "Risk Level": [
                "High Risk",
                "Medium Risk",
                "Low Risk"
            ],
            "Customers": [
                high_risk,
                medium_risk,
                low_risk
            ]
        })

        st.bar_chart(
            risk_df.set_index("Risk Level")
        )

    except Exception as e:

        st.warning(
            f"Risk distribution could not be calculated: {e}"
        )

    st.divider()

    # =====================================================
    # TOP CHURN DRIVERS
    # =====================================================

    st.header("🔥 Top Churn Drivers")

    drivers = pd.DataFrame({
        "Driver": [
            "Month-to-month Contract",
            "Fiber Optic Internet",
            "Tenure",
            "Monthly Charges",
            "Total Charges",
            "Electronic Check",
            "No Tech Support",
            "One-year Contract",
            "No Online Backup",
            "No Online Security"
        ],
        "Importance": [
            0.513973,
            0.163262,
            0.156717,
            0.035922,
            0.035791,
            0.027931,
            0.021377,
            0.009218,
            0.009198,
            0.008620
        ]
    })

    drivers["Importance (%)"] = (
        drivers["Importance"] * 100
    ).round(2)

    st.dataframe(
        drivers,
        width="stretch",
        hide_index=True
    )

    st.bar_chart(
        drivers
        .set_index("Driver")["Importance (%)"]
        .head(6)
    )

    st.info(
        "The strongest identified churn driver is "
        "the month-to-month contract, followed by "
        "Fiber optic internet and customer tenure."
    )

    st.divider()

    # =====================================================
    # PERSONA OVERVIEW
    # =====================================================

    st.header("👥 Customer Personas")

    persona_summary = (
        dashboard_personas
        .groupby("Persona Name")
        .agg(
            Customers=("Persona", "count"),
            Avg_Tenure=("tenure", "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Churn_Rate=("ChurnNumeric", "mean")
        )
        .reset_index()
    )

    persona_summary["Avg_Tenure"] = (
        persona_summary["Avg_Tenure"].round(1)
    )

    persona_summary["Avg_Monthly_Charges"] = (
        persona_summary["Avg_Monthly_Charges"].round(2)
    )

    persona_summary["Avg_Total_Charges"] = (
        persona_summary["Avg_Total_Charges"].round(2)
    )

    persona_summary["Churn_Rate"] = (
        persona_summary["Churn_Rate"] * 100
    ).round(2)

    st.dataframe(
        persona_summary,
        width="stretch",
        hide_index=True
    )

    st.subheader("🔥 Churn Rate by Persona")

    st.bar_chart(
        persona_summary
        .set_index("Persona Name")["Churn_Rate"]
    )

    st.divider()

    # =====================================================
    # BUSINESS INSIGHTS
    # =====================================================

    st.header("💡 Key Business Insights")

    if not persona_summary.empty:

        highest_persona = (
            persona_summary
            .sort_values(
                "Churn_Rate",
                ascending=False
            )
            .iloc[0]
        )

        st.warning(
            f"⚠️ **Highest-risk persona:** "
            f"{highest_persona['Persona Name']} "
            f"with a churn rate of "
            f"{highest_persona['Churn_Rate']:.2f}%."
        )

    st.info(
        "📌 Customers with month-to-month contracts "
        "should receive special retention attention."
    )

    st.info(
        "📌 New customers with short tenure should "
        "receive stronger onboarding and support."
    )

    st.info(
        "📌 Long-term high-value customers should "
        "be protected using loyalty programs."
    )


# =========================================================
# PAGE 2 - PREDICT CHURN
# =========================================================

elif page == "🤖 Predict Churn":

    st.title("🤖 Customer Churn Prediction")

    st.write(
        "Enter customer information to estimate "
        "their churn probability."
    )

    st.divider()

    st.subheader("👤 Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=12
        )

    with col2:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

    st.subheader("💳 Billing Information")

    col4, col5, col6 = st.columns(3)

    with col4:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col5:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col6:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    st.divider()

    if st.button(
        "🔮 Predict Churn",
        width="stretch"
    ):

        customer_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        try:

            prediction = model.predict(
                customer_data
            )[0]

            probability = (
                model.predict_proba(
                    customer_data
                )[0][1] * 100
            )

            # =================================================
            # PERSONA PREDICTION
            # =================================================

            persona_features = customer_data[
                [
                    "tenure",
                    "MonthlyCharges",
                    "TotalCharges"
                ]
            ]

            persona_scaled = persona_scaler.transform(
                persona_features
            )

            persona_prediction = persona_model.predict(
                persona_scaled
            )[0]

            persona_name = persona_names.get(
                persona_prediction,
                f"Persona {persona_prediction}"
            )

            st.divider()

            st.subheader("🎯 Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ HIGH RISK: Customer is likely to churn."
                )

            else:

                st.success(
                    "✅ LOW RISK: Customer is unlikely to churn."
                )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Customer Persona",
                    persona_name
                )

            with col3:

                if probability >= 70:
                    risk_text = "HIGH"
                elif probability >= 40:
                    risk_text = "MEDIUM"
                else:
                    risk_text = "LOW"

                st.metric(
                    "Risk Level",
                    risk_text
                )

            st.divider()

            # =================================================
            # RISK FACTORS
            # =================================================

            st.subheader("🔎 Important Risk Factors")

            risk_factors = []

            if contract == "Month-to-month":
                risk_factors.append(
                    "Month-to-month contract is associated "
                    "with higher churn risk."
                )

            if internet_service == "Fiber optic":
                risk_factors.append(
                    "Customer uses Fiber optic internet."
                )

            if tenure <= 12:
                risk_factors.append(
                    "Customer has relatively short tenure."
                )

            if monthly_charges >= 80:
                risk_factors.append(
                    "Customer has relatively high monthly charges."
                )

            if payment_method == "Electronic check":
                risk_factors.append(
                    "Customer uses electronic check payment."
                )

            if tech_support == "No":
                risk_factors.append(
                    "Customer does not have technical support."
                )

            if risk_factors:

                for factor in risk_factors:
                    st.warning("⚠️ " + factor)

            else:

                st.success(
                    "No major risk indicators were detected."
                )

            st.divider()

            # =================================================
            # RETENTION RECOMMENDATION
            # =================================================

            st.subheader("💡 Recommended Action")

            if probability >= 70:

                if persona_prediction == 2:

                    st.error(
                        "🚨 High-priority retention customer. "
                        "Consider a personalized retention offer, "
                        "contract upgrade and proactive customer support."
                    )

                elif contract == "Month-to-month":

                    st.warning(
                        "⚠️ Offer an attractive longer-term "
                        "contract to encourage customer retention."
                    )

                elif monthly_charges > 80:

                    st.warning(
                        "💰 Review the customer's pricing or "
                        "service package and consider a personalized offer."
                    )

                else:

                    st.warning(
                        "📞 Provide proactive customer support "
                        "and targeted engagement."
                    )

            elif probability >= 40:

                st.warning(
                    "🟠 Medium churn risk. Monitor this customer "
                    "and consider targeted engagement."
                )

            else:

                st.success(
                    "🟢 Low churn risk. Continue normal customer engagement."
                )

            with st.expander("View Customer Information"):

                st.dataframe(
                    customer_data,
                    width="stretch"
                )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.code(str(e))

            st.info(
                "Make sure the saved model expects the same "
                "columns and preprocessing used by this application."
            )


# =========================================================
# PAGE 3 - CUSTOMER PERSONAS
# =========================================================

elif page == "👥 Customer Personas":

    st.title("👥 Customer Persona Profiler")

    st.write(
        "Explore customer groups discovered using "
        "K-Means clustering."
    )

    st.divider()

    # =====================================================
    # PERSONA SUMMARY
    # =====================================================

    st.subheader("📊 Persona Summary")

    persona_summary = (
        dashboard_personas
        .groupby("Persona Name")
        .agg(
            Customers=("Persona", "count"),
            Avg_Tenure=("tenure", "mean"),
            Avg_Monthly_Charges=("MonthlyCharges", "mean"),
            Avg_Total_Charges=("TotalCharges", "mean"),
            Churn_Rate=("ChurnNumeric", "mean")
        )
        .reset_index()
    )

    persona_summary["Avg_Tenure"] = (
        persona_summary["Avg_Tenure"].round(2)
    )

    persona_summary["Avg_Monthly_Charges"] = (
        persona_summary["Avg_Monthly_Charges"].round(2)
    )

    persona_summary["Avg_Total_Charges"] = (
        persona_summary["Avg_Total_Charges"].round(2)
    )

    persona_summary["Churn_Rate"] = (
        persona_summary["Churn_Rate"] * 100
    ).round(2)

    st.dataframe(
        persona_summary,
        width="stretch",
        hide_index=True
    )

    st.divider()

    # =====================================================
    # PERSONA COUNTS
    # =====================================================

    st.subheader("👥 Customers in Each Persona")

    persona_counts = (
        dashboard_personas["Persona Name"]
        .value_counts()
    )

    st.bar_chart(persona_counts)

    st.divider()

    # =====================================================
    # CHURN RATE
    # =====================================================

    st.subheader("🔥 Churn Rate by Persona")

    persona_churn = (
        dashboard_personas
        .groupby("Persona Name")["ChurnNumeric"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(persona_churn)

    st.divider()

    # =====================================================
    # SELECT PERSONA
    # =====================================================

    st.subheader("🎯 Explore a Persona")

    selected_persona = st.selectbox(
        "Select a Persona",
        list(persona_names.values())
    )

    selected_data = (
        dashboard_personas[
            dashboard_personas["Persona Name"] == selected_persona
        ].copy()
    )

    if len(selected_data) > 0:

        count = len(selected_data)

        avg_tenure = selected_data["tenure"].mean()

        avg_monthly = selected_data["MonthlyCharges"].mean()

        churn = (
            selected_data["ChurnNumeric"].mean() * 100
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Customers", f"{count:,}")

        with col2:
            st.metric(
                "Average Tenure",
                f"{avg_tenure:.1f} months"
            )

        with col3:
            st.metric(
                "Avg Monthly Charges",
                f"${avg_monthly:.2f}"
            )

        with col4:
            st.metric(
                "Churn Rate",
                f"{churn:.2f}%"
            )

    st.divider()

    # =====================================================
    # PERSONA DESCRIPTION
    # =====================================================

    st.subheader("🧠 Persona Profile")

    if selected_persona == "Budget New Customers":

        st.info(
            "These customers have relatively short tenure "
            "and lower monthly charges. Focus on onboarding, "
            "affordable plans and early customer support."
        )

    elif selected_persona == "Loyal High-Value Customers":

        st.success(
            "These are valuable long-term customers with "
            "higher spending. Protect them using loyalty "
            "programs and premium support."
        )

    elif selected_persona == "High-Risk Customers":

        st.error(
            "⚠️ This is the highest-risk customer group. "
            "Prioritize proactive retention campaigns."
        )

    elif selected_persona == "Loyal Low-Cost Customers":

        st.success(
            "These customers have long tenure, lower charges "
            "and low churn. Maintain service quality."
        )

    st.divider()

    # =====================================================
    # CUSTOMER FILTER
    # =====================================================

    st.subheader("🔎 Filter Customers")

    churn_filter = st.selectbox(
        "Customer Status",
        [
            "All Customers",
            "Churned Customers",
            "Retained Customers"
        ]
    )

    filtered_customers = selected_data.copy()

    if churn_filter == "Churned Customers":

        filtered_customers = filtered_customers[
            filtered_customers["Churn"] == "Yes"
        ]

    elif churn_filter == "Retained Customers":

        filtered_customers = filtered_customers[
            filtered_customers["Churn"] == "No"
        ]

    st.write(
        f"Showing **{len(filtered_customers):,}** customers."
    )

    display_columns = [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "Churn"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in filtered_customers.columns
    ]

    st.dataframe(
        filtered_customers[available_columns],
        width="stretch",
        hide_index=True
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    csv_data = (
        filtered_customers
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="⬇️ Download Customer List",
        data=csv_data,
        file_name="selected_persona_customers.csv",
        mime="text/csv",
        width="stretch"
    )

    st.divider()

    # =====================================================
    # INDIVIDUAL CUSTOMER
    # =====================================================

    st.header("🔍 Individual Customer Profile")

    if "customerID" in dashboard_personas.columns:

        customer_ids = (
            dashboard_personas["customerID"]
            .astype(str)
            .sort_values()
            .tolist()
        )

        selected_customer = st.selectbox(
            "Select Customer ID",
            customer_ids,
            key="persona_customer_search"
        )

        customer_match = dashboard_personas[
            dashboard_personas["customerID"].astype(str)
            == selected_customer
        ]

        if len(customer_match) > 0:

            customer = customer_match.iloc[0]

            st.subheader("👤 Customer Information")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Customer ID",
                    str(customer["customerID"])
                )

            with col2:
                st.metric(
                    "Tenure",
                    f"{float(customer['tenure']):.0f} months"
                )

            with col3:
                st.metric(
                    "Monthly Charges",
                    f"${float(customer['MonthlyCharges']):.2f}"
                )

            with col4:
                st.metric(
                    "Total Charges",
                    f"${float(customer['TotalCharges']):.2f}"
                )

            st.divider()

            st.subheader("📋 Customer Details")

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Contract:** {customer['Contract']}"
                )

                st.write(
                    f"**Internet Service:** "
                    f"{customer['InternetService']}"
                )

                st.write(
                    f"**Payment Method:** "
                    f"{customer['PaymentMethod']}"
                )

                st.write(
                    f"**Tech Support:** "
                    f"{customer['TechSupport']}"
                )

            with col2:

                st.write(
                    f"**Online Security:** "
                    f"{customer['OnlineSecurity']}"
                )

                st.write(
                    f"**Online Backup:** "
                    f"{customer['OnlineBackup']}"
                )

                st.write(
                    f"**Partner:** {customer['Partner']}"
                )

                st.write(
                    f"**Dependents:** {customer['Dependents']}"
                )

            st.divider()

            st.subheader("🎯 Customer Persona")

            customer_persona = customer["Persona"]

            customer_persona_name = persona_names.get(
                customer_persona,
                f"Persona {customer_persona}"
            )

            st.info(
                f"This customer belongs to "
                f"**{customer_persona_name}**."
            )

            st.subheader("📉 Actual Churn Status")

            if customer["Churn"] == "Yes":

                st.error(
                    "🔴 This customer has churned."
                )

            else:

                st.success(
                    "🟢 This customer has not churned."
                )

            st.divider()

            st.subheader("💡 Retention Recommendation")

            monthly_charge = float(
                customer["MonthlyCharges"]
            )

            contract = str(
                customer["Contract"]
            )

            if customer_persona == 2:

                st.error(
                    "🚨 HIGH PRIORITY\n\n"
                    "This customer belongs to the "
                    "High-Risk Customers persona. "
                    "Consider proactive retention support."
                )

            elif contract == "Month-to-month":

                st.warning(
                    "⚠️ CONTRACT RISK\n\n"
                    "Consider offering a longer-term "
                    "contract with an attractive incentive."
                )

            elif monthly_charge > 80:

                st.warning(
                    "💰 HIGH CHARGES\n\n"
                    "Consider reviewing the service package "
                    "or offering a personalized plan."
                )

            else:

                st.success(
                    "🟢 LOW PRIORITY\n\n"
                    "Continue normal customer engagement."
                )


# =========================================================
# PAGE 4 - MODEL EVALUATION
# =========================================================

elif page == "📈 Model Evaluation":

    st.title("📈 Model Evaluation")

    st.write(
        "Evaluation of the saved Decision Tree "
        "churn prediction model."
    )

    st.divider()

    # =====================================================
    # PREPARE DATA
    # =====================================================

    X = df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        y_pred = model.predict(X_test)

        y_pred = (
            pd.Series(y_pred)
            .astype(int)
            .values
        )

        y_test = (
            pd.Series(y_test)
            .astype(int)
            .values
        )

        # =================================================
        # METRICS
        # =================================================

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=1,
            zero_division=0
        )

        # =================================================
        # PERFORMANCE
        # =================================================

        st.subheader("📊 Model Performance")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{accuracy * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Precision",
                f"{precision * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Recall",
                f"{recall * 100:.2f}%"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{f1 * 100:.2f}%"
            )

        st.divider()

        # =================================================
        # CONFUSION MATRIX
        # =================================================

        st.subheader("🔲 Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        )

        confusion_df = pd.DataFrame(
            cm,
            columns=[
                "Predicted No Churn",
                "Predicted Churn"
            ],
            index=[
                "Actual No Churn",
                "Actual Churn"
            ]
        )

        st.dataframe(
            confusion_df,
            width="stretch"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                f"Correct No Churn predictions: "
                f"**{cm[0][0]}**"
            )

            st.warning(
                f"False Churn predictions: "
                f"**{cm[0][1]}**"
            )

        with col2:

            st.success(
                f"Correct Churn predictions: "
                f"**{cm[1][1]}**"
            )

            st.error(
                f"Missed Churn customers: "
                f"**{cm[1][0]}**"
            )

        st.divider()

        # =================================================
        # CLASSIFICATION REPORT
        # =================================================

        st.subheader("📋 Classification Report")

        report_dict = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "No Churn",
                "Churn"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame({
            "Class": [
                "No Churn",
                "Churn"
            ],
            "Precision": [
                report_dict["No Churn"]["precision"],
                report_dict["Churn"]["precision"]
            ],
            "Recall": [
                report_dict["No Churn"]["recall"],
                report_dict["Churn"]["recall"]
            ],
            "F1 Score": [
                report_dict["No Churn"]["f1-score"],
                report_dict["Churn"]["f1-score"]
            ],
            "Support": [
                int(report_dict["No Churn"]["support"]),
                int(report_dict["Churn"]["support"])
            ]
        })

        display_report = report_df.copy()

        display_report["Precision"] = (
            display_report["Precision"] * 100
        ).round(2)

        display_report["Recall"] = (
            display_report["Recall"] * 100
        ).round(2)

        display_report["F1 Score"] = (
            display_report["F1 Score"] * 100
        ).round(2)

        st.dataframe(
            display_report,
            width="stretch",
            hide_index=True
        )

        st.divider()

        # =================================================
        # MODEL INTERPRETATION
        # =================================================

        st.subheader("🧠 Model Interpretation")

        st.info(
            f"The Decision Tree model achieved "
            f"**{accuracy * 100:.2f}% accuracy** "
            f"on the test dataset."
        )

        st.write(
            f"**Churn Precision:** "
            f"{precision * 100:.2f}%"
        )

        st.write(
            f"**Churn Recall:** "
            f"{recall * 100:.2f}%"
        )

        st.write(
            f"**Churn F1 Score:** "
            f"{f1 * 100:.2f}%"
        )

        st.divider()

        # =================================================
        # CHURN DRIVER ANALYSIS
        # =================================================

        st.subheader("🔥 Top Churn Drivers")

        evaluation_drivers = pd.DataFrame({
            "Feature": [
                "Month-to-month Contract",
                "Fiber Optic Internet",
                "Tenure",
                "Monthly Charges",
                "Total Charges",
                "Electronic Check"
            ],
            "Importance": [
                0.513973,
                0.163262,
                0.156717,
                0.035922,
                0.035791,
                0.027931
            ]
        })

        evaluation_drivers["Importance (%)"] = (
            evaluation_drivers["Importance"] * 100
        ).round(2)

        st.dataframe(
            evaluation_drivers,
            width="stretch",
            hide_index=True
        )

        st.bar_chart(
            evaluation_drivers
            .set_index("Feature")["Importance (%)"]
        )

        st.success(
            "The model shows that contract type, "
            "internet service, and tenure are among "
            "the strongest churn-related factors."
        )

    except Exception as e:

        st.error(
            "Model evaluation failed."
        )

        st.code(str(e))

        st.info(
            "The saved model and cleaned dataset must "
            "use the same preprocessing and feature structure."
        )