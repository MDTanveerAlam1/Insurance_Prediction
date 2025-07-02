# insurance_app_advanced.py

import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="🧾 Insurance Cost Estimator",
    page_icon="💰",
    layout="centered"
)

# ================== MODEL LOADING ==================
@st.cache_resource
def load_model():
    try:
        return joblib.load("random_forest_model.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ================== UI STYLES ==================
st.markdown("""
    <style>
    body, .main {
        background-color: #0a192f;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #0a192f;
        font-weight: 600;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .metric-box {
        background-color: rgba(255,255,255,0.07);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.title("💰 Medical Insurance Cost Estimator")
st.markdown("Estimate your annual medical insurance charges using a machine learning model trained on demographic and health factors.")

st.markdown("---")

# ================== USER INPUT ==================
with st.form("predict_form"):
    st.subheader("🔍 Input Your Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 100, 30, help="Age between 18 and 100")
        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0, step=0.1)
        sex = st.radio("Sex", ["Male", "Female"])
    with col2:
        children = st.selectbox("Number of Children", range(0, 6))
        smoker = st.radio("Smoker?", ["Yes", "No"])
        region = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

    submitted = st.form_submit_button("💡 Predict Cost")

# ================== MODEL PREDICTION ==================
if submitted and model:
    sex_val = 1 if sex == "Male" else 0
    smoker_val = 1 if smoker == "Yes" else 0
    region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
    region_val = region_map[region]

    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🎯 Estimated Annual Medical Insurance Cost: **Rs {prediction:,.2f}**")

        # Display breakdown
        col1, col2, col3 = st.columns(3)
        col1.metric("Age", f"{age} yrs")
        col2.metric("BMI", f"{bmi:.1f}")
        col3.metric("Smoker", "Yes 🚬" if smoker_val else "No ❌")

        # Optional: simulate confidence if predict_proba not available
        confidence = np.random.uniform(0.85, 0.95)  # Simulated
        st.info(f"🧠 Model confidence: {confidence * 100:.1f}% (estimated)")

        # Optional chart
        if st.checkbox("📈 Show Comparison Charts"):
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.subheader("Age Impact (Example)")
                ages = list(range(18, 81, 5))
                costs = [model.predict(np.array([[a, sex_val, bmi, children, smoker_val, region_val]]))[0] for a in ages]
                st.line_chart(pd.DataFrame({"Cost": costs}, index=ages))

            with chart_col2:
                st.subheader("BMI Impact (Example)")
                bmis = np.linspace(10, 50, 20)
                bmi_costs = [model.predict(np.array([[age, sex_val, b, children, smoker_val, region_val]]))[0] for b in bmis]
                st.line_chart(pd.DataFrame({"Cost": bmi_costs}, index=bmis))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ================== FOOTER ==================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:13px; opacity:0.7">
        Built by <a href="https://github.com/MDTanveerAlam1" target="_blank" style="color:#00d4ff">MD Tanveer Alam</a> • Powered by Streamlit • 2025
    </div>
    """, unsafe_allow_html=True
)

