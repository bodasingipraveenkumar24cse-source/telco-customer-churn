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
    return pd.read_csv(
        "data/Telco-Customer-Churn-Cleaned.csv"
    )


@st.cache_data
def load_personas():
    return pd.read_csv(
        "data/customer_personas.csv"
    )


# =========================================================
# LOAD CHURN MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/churn_decision_tree.pkl"
    )


# =========================================================
# LOAD PERSONA MODEL
# =========================================================

@st.cache_resource
def load_persona_model():
    return joblib.load(
        "models/persona_kmeans.pkl"
    )


# =========================================================
# LOAD PERSONA SCALER
# =========================================================

@st.cache_resource
def load_persona_scaler():
    return joblib.load(
        "models/persona_scaler.pkl"
    )


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
# PERSONA DESCRIPTIONS
# =========================================================

persona_descriptions = {

    "Budget New Customers":
        "Customers with relatively short tenure and lower "
        "monthly spending. Their early experience is important "
        "for preventing future churn.",

    "Loyal High-Value Customers":
        "Long-term customers with relatively high spending. "
        "They are valuable customers who should be protected "
        "through loyalty programs and premium service.",

    "High-Risk Customers":
        "Customers with the highest churn tendency. "
        "They require proactive retention efforts and "
        "personalized offers.",

    "Loyal Low-Cost Customers":
        "Long-term customers with relatively low spending "
        "and very low churn. Maintaining service quality "
        "is the primary objective."
}


# =========================================================
# PERSONA ACTIONS
# =========================================================

persona_actions = {

    "Budget New Customers":
        "Improve onboarding, provide affordable plans, "
        "and offer early-stage engagement.",

    "Loyal High-Value Customers":
        "Use loyalty rewards, premium support, personalized "
        "offers and long-term contract incentives.",

    "High-Risk Customers":
        "Prioritize immediate retention campaigns, "
        "contract upgrades, discounts and proactive support.",

    "Loyal Low-Cost Customers":
        "Maintain service quality and consider suitable "
        "cross-selling opportunities."
}


# =========================================================
# PERSONA RISK LEVEL
# =========================================================

persona_risk = {

    "Budget New Customers": "🟠 Medium Risk",

    "Loyal High-Value Customers": "🟡 Moderate Risk",

    "High-Risk Customers": "🔴 High Risk",

    "Loyal Low-Cost Customers": "🟢 Low Risk"
}


# =========================================================
# PREPARE PERSONA DATA
# =========================================================

dashboard_personas = persona_df.copy()


dashboard_personas["Persona Name"] = (
    dashboard_personas["Persona"]
    .map(persona_names)
    .fillna(
        dashboard_personas["Persona"].astype(str)
    )
)


dashboard_personas["ChurnNumeric"] = (
    dashboard_personas["Churn"]
    .map({
        "No": 0,
        "Yes": 1
    })
)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "📊 Dashboard",
        "🤖 Predict Churn",
        "👥 Customer Personas",
        "📈 Model Evaluation"
    ]
)


# =========================================================
# PAGE 1 - DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.title(
        "📊 Telecom Churn Analytics"
    )

    st.subheader(
        "Telecom Churn Driver Discovery & Persona Profiler"
    )

    st.write(
        "An analytics system that identifies customer "
        "churn risk, discovers major churn drivers, "
        "and profiles customers into meaningful personas."
    )

    st.divider()


    # =====================================================
    # CUSTOMER STATISTICS
    # =====================================================

    total_customers = len(df)

    churned_customers = len(
        df[df["Churn"] == "Yes"]
    )

    retained_customers = (
        total_customers -
        churned_customers
    )

    churn_rate = (
        churned_customers /
        total_customers
    ) * 100

    retention_rate = (
        retained_customers /
        total_customers
    ) * 100


    st.subheader(
        "📌 Customer Overview"
    )

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

    st.subheader(
        "💰 Customer Financial Metrics"
    )

    avg_monthly_charges = (
        df["MonthlyCharges"].mean()
    )

    avg_total_charges = (
        df["TotalCharges"].mean()
    )

    avg_tenure = (
        df["tenure"].mean()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Average Monthly Charges",
            f"${avg_monthly_charges:.2f}"
        )

    with col2:
        st.metric(
            "Average Total Charges",
            f"${avg_total_charges:.2f}"
        )

    with col3:
        st.metric(
            "Average Tenure",
            f"{avg_tenure:.1f} months"
        )


    st.divider()


    # =====================================================
    # CUSTOMER RISK DISTRIBUTION
    # =====================================================

    st.header(
        "🚦 Customer Risk Distribution"
    )

    dashboard_X = df.drop(
        columns=[
            "customerID",
            "Churn"
        ],
        errors="ignore"
    )

    try:

        dashboard_probabilities = (
            model.predict_proba(
                dashboard_X
            )[:, 1]
            * 100
        )

        high_risk = (
            dashboard_probabilities >= 70
        ).sum()

        medium_risk = (
            (
                dashboard_probabilities >= 40
            )
            &
            (
                dashboard_probabilities < 70
            )
        ).sum()

        low_risk = (
            dashboard_probabilities < 40
        ).sum()


        risk_col1, risk_col2, risk_col3 = (
            st.columns(3)
        )

        with risk_col1:
            st.metric(
                "🔴 High Risk",
                f"{high_risk:,}"
            )

        with risk_col2:
            st.metric(
                "🟠 Medium Risk",
                f"{medium_risk:,}"
            )

        with risk_col3:
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
            risk_df.set_index(
                "Risk Level"
            )
        )

    except Exception:

        st.warning(
            "Risk distribution could not be calculated."
        )


    st.divider()


    # =====================================================
    # CHURN DISTRIBUTION
    # =====================================================

    st.header(
        "📊 Customer Churn Distribution"
    )

    churn_counts = (
        df["Churn"]
        .value_counts()
    )

    churn_display = pd.DataFrame({

        "Customer Status": [
            "No Churn",
            "Churn"
        ],

        "Customers": [
            churn_counts.get("No", 0),
            churn_counts.get("Yes", 0)
        ]

    })

    st.bar_chart(
        churn_display.set_index(
            "Customer Status"
        )
    )


    st.divider()


    # =====================================================
    # CHURN BY CONTRACT
    # =====================================================

    st.header(
        "📋 Churn by Contract Type"
    )

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    st.bar_chart(
        contract_churn
    )


    st.divider()


    # =====================================================
    # TOP CHURN DRIVERS
    # =====================================================

    st.header(
        "🔎 Top Churn Drivers"
    )

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
        drivers[
            [
                "Driver",
                "Importance (%)"
            ]
        ],
        width="stretch",
        hide_index=True
    )

    st.bar_chart(
        drivers
        .set_index("Driver")[
            "Importance (%)"
        ]
        .head(6)
    )

    st.info(
        "The strongest identified churn driver is "
        "the month-to-month contract, followed by "
        "Fiber optic internet service and customer tenure."
    )


    st.divider()


    # =====================================================
    # PERSONA SUMMARY
    # =====================================================

    st.header(
        "👥 Customer Personas"
    )

    persona_summary = (

        dashboard_personas

        .groupby(
            "Persona Name"
        )

        .agg(

            Customers=(
                "Persona",
                "count"
            ),

            Avg_Tenure=(
                "tenure",
                "mean"
            ),

            Avg_Monthly_Charges=(
                "MonthlyCharges",
                "mean"
            ),

            Avg_Total_Charges=(
                "TotalCharges",
                "mean"
            ),

            Churn_Rate=(
                "ChurnNumeric",
                "mean"
            )

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

    st.subheader(
        "📈 Churn Rate by Persona"
    )

    persona_churn = (
        persona_summary
        .set_index(
            "Persona Name"
        )[
            "Churn_Rate"
        ]
    )

    st.bar_chart(
        persona_churn
    )


    st.divider()


    # =====================================================
    # BUSINESS RECOMMENDATIONS
    # =====================================================

    st.header(
        "💡 Business Recommendations"
    )

    st.subheader(
        "1️⃣ High-Risk Customers"
    )

    st.write(
        "Provide early retention offers, onboarding "
        "support, contract incentives and proactive "
        "customer service."
    )

    st.subheader(
        "2️⃣ Budget New Customers"
    )

    st.write(
        "Focus on affordable plans and improving "
        "the early customer experience."
    )

    st.subheader(
        "3️⃣ Loyal High-Value Customers"
    )

    st.write(
        "Protect valuable customers using loyalty "
        "programs, premium support and personalized offers."
    )

    st.subheader(
        "4️⃣ Loyal Low-Cost Customers"
    )

    st.write(
        "Maintain service quality and explore "
        "suitable cross-selling opportunities."
    )


# =========================================================
# PAGE 2 - PREDICT CHURN
# =========================================================

elif page == "🤖 Predict Churn":

    st.title(
        "🤖 Customer Churn Prediction"
    )

    st.write(
        "Enter customer information to predict "
        "their probability of churn."
    )


    # =====================================================
    # CUSTOMER SEARCH
    # =====================================================

    st.subheader(
        "🔍 Search Existing Customer"
    )

    customer_id = st.text_input(
        "Enter Customer ID",
        placeholder="Example: 7590-VHVEG"
    )

    if customer_id:

        existing_customer = df[
            df["customerID"] == customer_id
        ]

        if len(existing_customer) > 0:

            st.success(
                "Customer found!"
            )

            st.dataframe(
                existing_customer,
                width="stretch"
            )

        else:

            st.warning(
                "Customer ID not found in the dataset."
            )


    st.divider()


    # =====================================================
    # CUSTOMER INFORMATION
    # =====================================================

    st.subheader(
        "👤 Customer Information"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [
                0,
                1
            ]
        )

        partner = st.selectbox(
            "Partner",
            [
                "Yes",
                "No"
            ]
        )

        dependents = st.selectbox(
            "Dependents",
            [
                "Yes",
                "No"
            ]
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
            [
                "Yes",
                "No"
            ]
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


    st.subheader(
        "💳 Billing Information"
    )

    col4, col5, col6 = st.columns(3)


    with col4:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
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


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

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


        # =============================================
        # CHURN PREDICTION
        # =============================================

        prediction = model.predict(
            customer_data
        )[0]

        probability = (
            model
            .predict_proba(
                customer_data
            )[0][1]
            * 100
        )


        # =============================================
        # PERSONA PREDICTION
        # =============================================

        persona_features = customer_data[
            [
                "tenure",
                "MonthlyCharges",
                "TotalCharges"
            ]
        ]

        persona_scaled = (
            persona_scaler
            .transform(
                persona_features
            )
        )

        persona_prediction = (
            persona_model
            .predict(
                persona_scaled
            )[0]
        )

        persona_name = persona_names.get(
            persona_prediction,
            f"Persona {persona_prediction}"
        )


        st.divider()


        # =============================================
        # PREDICTION RESULT
        # =============================================

        st.subheader(
            "🎯 Prediction Result"
        )

        if prediction == 1:

            st.error(
                "⚠️ HIGH RISK: Customer is likely to churn."
            )

        else:

            st.success(
                "✅ LOW RISK: Customer is unlikely to churn."
            )


        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Churn Probability",
                f"{probability:.2f}%"
            )

        with result_col2:

            st.metric(
                "Customer Persona",
                persona_name
            )


        # =============================================
        # RISK LEVEL
        # =============================================

        if probability >= 70:

            st.error(
                "🔴 Risk Level: HIGH"
            )

        elif probability >= 40:

            st.warning(
                "🟠 Risk Level: MEDIUM"
            )

        else:

            st.success(
                "🟢 Risk Level: LOW"
            )


        # =============================================
        # RISK FACTORS
        # =============================================

        st.subheader(
            "🔎 Important Risk Factors"
        )

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


        if len(risk_factors) > 0:

            for factor in risk_factors:

                st.warning(
                    "⚠️ " + factor
                )

        else:

            st.success(
                "No major risk indicators were detected."
            )


        # =============================================
        # RETENTION RECOMMENDATION
        # =============================================

        st.subheader(
            "💡 Recommended Action"
        )


        if probability >= 70:

            if persona_prediction == 2:

                st.error(
                    "High-priority retention customer. "
                    "Consider offering a contract upgrade, "
                    "discount, or personalized retention offer."
                )

            elif contract == "Month-to-month":

                st.warning(
                    "Customer is on a month-to-month contract. "
                    "Consider offering an attractive longer-term "
                    "contract option."
                )

            elif monthly_charges > 80:

                st.warning(
                    "Customer has relatively high monthly charges. "
                    "Consider reviewing pricing or offering a "
                    "personalized service package."
                )

            else:

                st.warning(
                    "Customer has elevated churn risk. "
                    "Consider proactive customer support."
                )


        elif probability >= 40:

            st.warning(
                "Medium churn risk. Monitor this customer "
                "and consider targeted engagement."
            )


        else:

            st.success(
                "Low churn risk. Continue normal customer engagement."
            )


        with st.expander(
            "View Customer Information"
        ):

            st.dataframe(
                customer_data,
                width="stretch"
            )


# =========================================================
# PAGE 3 - CUSTOMER PERSONAS
# =========================================================

elif page == "👥 Customer Personas":

    st.title(
        "👥 Customer Persona Profiler"
    )

    st.write(
        "Explore customer groups discovered using "
        "K-Means clustering and analyze individual customers."
    )

    st.divider()


    # =====================================================
    # PERSONA SUMMARY
    # =====================================================

    st.subheader(
        "📊 Persona Summary"
    )

    persona_summary = (

        dashboard_personas

        .groupby(
            "Persona Name"
        )

        .agg(

            Customers=(
                "Persona",
                "count"
            ),

            Avg_Tenure=(
                "tenure",
                "mean"
            ),

            Avg_Monthly_Charges=(
                "MonthlyCharges",
                "mean"
            ),

            Avg_Total_Charges=(
                "TotalCharges",
                "mean"
            ),

            Churn_Rate=(
                "ChurnNumeric",
                "mean"
            )

        )

        .reset_index()

    )


    persona_summary[
        "Avg_Tenure"
    ] = (
        persona_summary[
            "Avg_Tenure"
        ].round(2)
    )


    persona_summary[
        "Avg_Monthly_Charges"
    ] = (
        persona_summary[
            "Avg_Monthly_Charges"
        ].round(2)
    )


    persona_summary[
        "Avg_Total_Charges"
    ] = (
        persona_summary[
            "Avg_Total_Charges"
        ].round(2)
    )


    persona_summary[
        "Churn_Rate"
    ] = (
        persona_summary[
            "Churn_Rate"
        ] * 100
    ).round(2)


    st.dataframe(
        persona_summary,
        width="stretch",
        hide_index=True
    )


    st.divider()


    # =====================================================
    # PERSONA CUSTOMER COUNT
    # =====================================================

    st.subheader(
        "👥 Customers in Each Persona"
    )

    persona_counts = (
        dashboard_personas[
            "Persona Name"
        ]
        .value_counts()
    )

    st.bar_chart(
        persona_counts
    )


    st.divider()


    # =====================================================
    # CHURN RATE BY PERSONA
    # =====================================================

    st.subheader(
        "🔥 Churn Rate by Persona"
    )

    persona_churn = (
        dashboard_personas
        .groupby(
            "Persona Name"
        )["ChurnNumeric"]
        .mean()
        .mul(100)
        .round(2)
    )

    st.bar_chart(
        persona_churn
    )


    st.divider()


    # =====================================================
    # PERSONA COMPARISON
    # =====================================================

    st.subheader(
        "📊 Persona Comparison"
    )

    comparison_df = persona_summary.copy()

    comparison_df["Risk Level"] = (
        comparison_df["Persona Name"]
        .map(persona_risk)
    )

    comparison_df["Recommended Action"] = (
        comparison_df["Persona Name"]
        .map(persona_actions)
    )

    comparison_display = comparison_df[
        [
            "Persona Name",
            "Customers",
            "Avg_Tenure",
            "Avg_Monthly_Charges",
            "Avg_Total_Charges",
            "Churn_Rate",
            "Risk Level",
            "Recommended Action"
        ]
    ].copy()

    comparison_display = comparison_display.rename(
        columns={
            "Persona Name": "Persona",
            "Avg_Tenure": "Avg Tenure",
            "Avg_Monthly_Charges": "Avg Monthly Charges",
            "Avg_Total_Charges": "Avg Total Charges",
            "Churn_Rate": "Churn Rate (%)"
        }
    )

    st.dataframe(
        comparison_display,
        width="stretch",
        hide_index=True
    )


    st.divider()


    # =====================================================
    # SELECT PERSONA
    # =====================================================

    st.subheader(
        "🎯 Explore a Persona"
    )

    selected_persona = st.selectbox(
        "Select a Persona",
        list(persona_names.values())
    )


    selected_persona_data = (
        dashboard_personas[
            dashboard_personas[
                "Persona Name"
            ] == selected_persona
        ]
        .copy()
    )


    # =====================================================
    # PERSONA METRICS
    # =====================================================

    if len(selected_persona_data) > 0:

        selected_customer_count = (
            len(selected_persona_data)
        )

        selected_avg_tenure = (
            selected_persona_data[
                "tenure"
            ].mean()
        )

        selected_avg_monthly = (
            selected_persona_data[
                "MonthlyCharges"
            ].mean()
        )

        selected_avg_total = (
            selected_persona_data[
                "TotalCharges"
            ].mean()
        )

        selected_churn_rate = (
            selected_persona_data[
                "ChurnNumeric"
            ].mean()
            * 100
        )


        # =================================================
        # PERSONA HEADER
        # =================================================

        st.markdown(
            f"### {selected_persona}"
        )

        st.write(
            persona_descriptions.get(
                selected_persona,
                "Customer segment discovered through clustering."
            )
        )


        # =================================================
        # PERSONA METRICS
        # =================================================

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "👥 Customers",
                f"{selected_customer_count:,}"
            )

        with col2:

            st.metric(
                "📅 Average Tenure",
                f"{selected_avg_tenure:.1f} months"
            )

        with col3:

            st.metric(
                "💰 Avg Monthly Charges",
                f"${selected_avg_monthly:.2f}"
            )

        with col4:

            st.metric(
                "🔥 Churn Rate",
                f"{selected_churn_rate:.2f}%"
            )


        st.metric(
            "💵 Avg Total Charges",
            f"${selected_avg_total:.2f}"
        )


        st.divider()


        # =================================================
        # RISK LEVEL
        # =================================================

        st.subheader(
            "🚦 Persona Risk Level"
        )

        risk_level = persona_risk.get(
            selected_persona,
            "Unknown"
        )

        if "High" in risk_level:

            st.error(
                risk_level
            )

        elif "Low" in risk_level:

            st.success(
                risk_level
            )

        else:

            st.warning(
                risk_level
            )


        # =================================================
        # PERSONA DESCRIPTION
        # =================================================

        st.subheader(
            "🧠 Persona Profile"
        )

        st.info(
            persona_descriptions.get(
                selected_persona,
                "No description available."
            )
        )


        # =================================================
        # RECOMMENDED BUSINESS ACTION
        # =================================================

        st.subheader(
            "💡 Recommended Business Action"
        )

        st.success(
            persona_actions.get(
                selected_persona,
                "Monitor this customer segment."
            )
        )


    st.divider()


    # =====================================================
    # FILTER CUSTOMERS
    # =====================================================

    st.subheader(
        "🔎 Filter Customers"
    )

    churn_filter = st.selectbox(
        "Customer Status",
        [
            "All Customers",
            "Churned Customers",
            "Retained Customers"
        ]
    )


    filtered_customers = (
        selected_persona_data.copy()
    )


    if churn_filter == "Churned Customers":

        filtered_customers = (
            filtered_customers[
                filtered_customers[
                    "Churn"
                ] == "Yes"
            ]
        )


    elif churn_filter == "Retained Customers":

        filtered_customers = (
            filtered_customers[
                filtered_customers[
                    "Churn"
                ] == "No"
            ]
        )


    # =====================================================
    # CUSTOMER TABLE
    # =====================================================

    st.subheader(
        "👥 Customers in Selected Persona"
    )

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
        filtered_customers[
            available_columns
        ],
        width="stretch",
        hide_index=True
    )


    # =====================================================
    # DOWNLOAD CUSTOMERS
    # =====================================================

    csv_data = (
        filtered_customers
        .to_csv(
            index=False
        )
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
    # INDIVIDUAL CUSTOMER SEARCH
    # =====================================================

    st.header(
        "🔍 Individual Customer Profile"
    )


    if "customerID" in dashboard_personas.columns:

        customer_ids = (

            dashboard_personas[
                "customerID"
            ]

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
            dashboard_personas[
                "customerID"
            ].astype(str)
            == selected_customer
        ]


        if len(customer_match) > 0:

            customer = (
                customer_match
                .iloc[0]
            )


            st.divider()


            # =============================================
            # CUSTOMER BASIC INFORMATION
            # =============================================

            st.subheader(
                "👤 Customer Information"
            )


            col1, col2, col3, col4 = (
                st.columns(4)
            )


            with col1:

                st.metric(
                    "Customer ID",
                    str(
                        customer["customerID"]
                    )
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


            # =============================================
            # CUSTOMER DETAILS
            # =============================================

            st.subheader(
                "📋 Customer Details"
            )


            col1, col2 = st.columns(2)


            with col1:

                if "Contract" in customer.index:

                    st.write(
                        f"**Contract:** "
                        f"{customer['Contract']}"
                    )


                if "InternetService" in customer.index:

                    st.write(
                        f"**Internet Service:** "
                        f"{customer['InternetService']}"
                    )


                if "PaymentMethod" in customer.index:

                    st.write(
                        f"**Payment Method:** "
                        f"{customer['PaymentMethod']}"
                    )


                if "TechSupport" in customer.index:

                    st.write(
                        f"**Tech Support:** "
                        f"{customer['TechSupport']}"
                    )


            with col2:

                if "OnlineSecurity" in customer.index:

                    st.write(
                        f"**Online Security:** "
                        f"{customer['OnlineSecurity']}"
                    )


                if "OnlineBackup" in customer.index:

                    st.write(
                        f"**Online Backup:** "
                        f"{customer['OnlineBackup']}"
                    )


                if "Partner" in customer.index:

                    st.write(
                        f"**Partner:** "
                        f"{customer['Partner']}"
                    )


                if "Dependents" in customer.index:

                    st.write(
                        f"**Dependents:** "
                        f"{customer['Dependents']}"
                    )


            st.divider()


            # =============================================
            # PERSONA
            # =============================================

            st.subheader(
                "🎯 Customer Persona"
            )


            customer_persona = (
                customer["Persona"]
            )


            customer_persona_name = (
                persona_names.get(
                    customer_persona,
                    f"Persona {customer_persona}"
                )
            )


            st.info(
                f"This customer belongs to: "
                f"**{customer_persona_name}**"
            )


            # =============================================
            # ACTUAL CHURN STATUS
            # =============================================

            st.subheader(
                "📉 Actual Churn Status"
            )


            if customer["Churn"] == "Yes":

                st.error(
                    "🔴 This customer has churned."
                )

            else:

                st.success(
                    "🟢 This customer has not churned."
                )


            st.divider()


            # =============================================
            # RETENTION RECOMMENDATION
            # =============================================

            st.subheader(
                "💡 Retention Recommendation"
            )


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
                    "Consider proactive retention support, "
                    "personalized offers and contract upgrade "
                    "options."
                )


            elif contract == "Month-to-month":

                st.warning(
                    "⚠️ CONTRACT RISK\n\n"
                    "This customer is using a month-to-month "
                    "contract. Consider offering a longer-term "
                    "contract with an attractive incentive."
                )


            elif monthly_charge > 80:

                st.warning(
                    "💰 HIGH CHARGES\n\n"
                    "This customer has relatively high monthly "
                    "charges. Consider reviewing the service "
                    "package or offering a personalized plan."
                )


            else:

                st.success(
                    "🟢 LOW PRIORITY\n\n"
                    "This customer appears relatively stable. "
                    "Continue normal customer engagement."
                )


# =========================================================
# PAGE 4 - MODEL EVALUATION
# =========================================================

elif page == "📈 Model Evaluation":

    st.title(
        "📈 Model Evaluation"
    )

    st.write(
        "Evaluation of the saved Decision Tree "
        "churn prediction model."
    )


    # =====================================================
    # PREPARE DATA
    # =====================================================

    evaluation_df = df.copy()

    X = evaluation_df.drop(
        columns=[
            "customerID",
            "Churn"
        ],
        errors="ignore"
    )

    y = evaluation_df[
        "Churn"
    ].map({
        "No": 0,
        "Yes": 1
    })


    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    y_pred = model.predict(
        X_test
    )

    y_pred = (
        pd.Series(
            y_pred
        )
        .astype(int)
        .values
    )

    y_test = (
        pd.Series(
            y_test
        )
        .astype(int)
        .values
    )


    # =====================================================
    # METRICS
    # =====================================================

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


    # =====================================================
    # MODEL PERFORMANCE
    # =====================================================

    st.subheader(
        "📊 Model Performance"
    )

    col1, col2, col3, col4 = (
        st.columns(4)
    )

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


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    st.subheader(
        "🔲 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            0,
            1
        ]
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


    st.write(
        f"**{cm[0][0]}** customers were correctly "
        "identified as No Churn."
    )

    st.write(
        f"**{cm[1][1]}** customers were correctly "
        "identified as Churn."
    )

    st.write(
        f"**{cm[0][1]}** customers were incorrectly "
        "predicted as Churn."
    )

    st.write(
        f"**{cm[1][0]}** churned customers were "
        "predicted as No Churn."
    )


    st.divider()


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    st.subheader(
        "📋 Classification Report"
    )


    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[
            0,
            1
        ],
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
            report_dict[
                "No Churn"
            ]["precision"],

            report_dict[
                "Churn"
            ]["precision"]
        ],

        "Recall": [
            report_dict[
                "No Churn"
            ]["recall"],

            report_dict[
                "Churn"
            ]["recall"]
        ],

        "F1 Score": [
            report_dict[
                "No Churn"
            ]["f1-score"],

            report_dict[
                "Churn"
            ]["f1-score"]
        ],

        "Support": [
            int(
                report_dict[
                    "No Churn"
                ]["support"]
            ),

            int(
                report_dict[
                    "Churn"
                ]["support"]
            )
        ]

    })


    display_report = (
        report_df.copy()
    )


    display_report[
        "Precision"
    ] = (
        display_report[
            "Precision"
        ] * 100
    ).round(2)


    display_report[
        "Recall"
    ] = (
        display_report[
            "Recall"
        ] * 100
    ).round(2)


    display_report[
        "F1 Score"
    ] = (
        display_report[
            "F1 Score"
        ] * 100
    ).round(2)


    st.dataframe(
        display_report,
        width="stretch",
        hide_index=True
    )


    st.divider()


    # =====================================================
    # MODEL INTERPRETATION
    # =====================================================

    st.subheader(
        "🧠 Model Interpretation"
    )


    st.info(
        f"The model achieved an accuracy of "
        f"{accuracy * 100:.2f}% on the test dataset."
    )


    st.write(
        f"Precision for the Churn class: "
        f"{precision * 100:.2f}%"
    )


    st.write(
        f"Recall for the Churn class: "
        f"{recall * 100:.2f}%"
    )


    st.write(
        f"F1-score for the Churn class: "
        f"{f1 * 100:.2f}%"
    )


    st.success(
        "The evaluation metrics are calculated "
        "from the saved model and test dataset."
    )


    st.divider()


    # =====================================================
    # TOP CHURN DRIVERS
    # =====================================================

    st.subheader(
        "🔎 Top Churn Drivers"
    )


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


    evaluation_drivers[
        "Importance (%)"
    ] = (
        evaluation_drivers[
            "Importance"
        ] * 100
    ).round(2)


    st.dataframe(
        evaluation_drivers[
            [
                "Feature",
                "Importance (%)"
            ]
        ],
        width="stretch",
        hide_index=True
    )


    st.bar_chart(
        evaluation_drivers
        .set_index(
            "Feature"
        )[
            "Importance (%)"
        ]
    )