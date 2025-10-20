import streamlit as st
import requests

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details below to predict if they are likely to churn.")

API_URL=st.text_input("Backend API URL", value="http://127.0.0.1:8000/predict")

with st.form("churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen (1=Yes, 0=No)", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0, step=10.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    try:
        with st.spinner("Calling FastAPI backend..."):
            res = requests.post(API_URL, json=input_data, timeout=30)
            res.raise_for_status()
            result = res.json()

            if "prediction" in result:
                if result["prediction"] == 1:
                    st.error(f"⚠️ Customer likely to **churn**! Probability: {result['probability']*100:.2f}%")
                else:
                    st.success(f"✅ Customer likely to **stay**. Probability: {(1-result['probability'])*100:.2f}%")
            else:
                st.error(f"Backend error: {result.get('error', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")


