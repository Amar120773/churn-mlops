import streamlit as st
import pandas as pd
import mlflow.pyfunc

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Churn Prediction",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = mlflow.pyfunc.load_model("model")
# -----------------------------
# Load dataset columns
# -----------------------------
df = pd.read_csv("churn.csv")
df = df.astype(int)
columns = df.drop("Churn", axis=1).columns

# -----------------------------
# Custom Dark Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    color: #00ADB5;
}
.sub-text {
    color: #94A3B8;
}
.card {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title Section
# -----------------------------
st.markdown('<div class="main-title">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Predict whether a customer will churn based on input data</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Customer Info")

    tenure = st.slider("Tenure", 0, 100, 12)
    monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.slider("Total Charges", 0.0, 10000.0, 2000.0)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Additional Details")

    gender = st.selectbox("Gender", ["Female", "Male"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown('</div>', unsafe_allow_html=True)

# Convert to numeric
gender_male = 1 if gender == "Male" else 0
partner_yes = 1 if partner == "Yes" else 0
dependents_yes = 1 if dependents == "Yes" else 0

# -----------------------------
# Prediction Button
# -----------------------------
st.write("")
if st.button("🔍 Predict Churn"):

    model_input = pd.DataFrame(0, index=[0], columns=columns)

    model_input["tenure"] = tenure
    model_input["MonthlyCharges"] = monthly_charges
    model_input["TotalCharges"] = total_charges
    model_input["gender_Male"] = gender_male
    model_input["Partner_Yes"] = partner_yes
    model_input["Dependents_Yes"] = dependents_yes

    prediction = model.predict(model_input)[0]

    st.write("")

    if prediction == 1:
        st.error("⚠️ High Risk: Customer likely to churn")
    else:
        st.success("✅ Low Risk: Customer likely to stay")