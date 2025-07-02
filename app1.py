# insurance_app.py

import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="💸 MedInsure - Cost Estimator",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    try:
        return joblib.load("random_forest_model.pkl")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ========== GLOBAL STYLES ==========
st.markdown("""
    <style>
    body {
        background-color: #0a192f;
        color: #ffffff;
    }
    .card {
        background: rgba(255,255,255,0.05);
        padding: 25px;
        border-radius: 16px;
        margin-bottom: 20px;
    }
    .predict-box {
        background: linear-gradient(135deg, #00d4ff30, #90ee9030);
        padding: 25px;
        border-radius: 18px;
        margin-top: 30px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    .metric {
        background: rgba(255,255,255,0.06);
        padding: 12px;
        border-radius: 12px;
        text-align: center;
        font-size: 16px;
    }
    .footer {
        margin-top: 40px;
        font-size: 13px;
        text-align: center;
        opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)

# ========== HEADER ==========
st.markdown("""
    <div style="text-align:center; margin-bottom:30px">
        <h1 style="color:#00d4ff">💸 MedInsure Predictor</h1>
        <p style="opacity:0.8; font-size:16px">AI-powered insurance cost estimation based on health and lifestyle inputs</p>
    </div>
""", unsafe_allow_html=True)

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    - This tool uses a **Random Forest Regressor** to estimate medical insurance cost.
    - Input your details, and the AI will predict your expected cost.
    - Made with ❤️ by **MD Tanveer Alam**.
    """)
    st.markdown("---")
    st.caption("📦 Model: `random_forest_model.pkl`")

# ========== INPUT FORM ==========
with st.form("predict_form"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Enter Your Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 30)
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
    with col2:
        children = st.selectbox("Number of Children", list(range(0, 6)))
        smoker = st.radio("Smoker?", ["Yes", "No"], horizontal=True)
        region = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

    st.markdown("</div>", unsafe_allow_html=True)
    submit = st.form_submit_button("💡 Predict Insurance Cost")

# ========== PREDICTION ==========
if submit and model:
    sex_val = 1 if sex == "Male" else 0
    smoker_val = 1 if smoker == "Yes" else 0
    region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
    region_val = region_map[region]

    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])

    try:
        prediction = model.predict(input_data)[0]

        st.markdown(f"""
            <div class="predict-box animate-fade-in">
                🧾 Estimated Insurance Cost: <span style="color:#00ffae">${prediction:,.2f}</span>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric'>🧬 BMI<br><strong>{bmi:.1f}</strong></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric'>🎂 Age<br><strong>{age}</strong></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric'>🚬 Smoker<br><strong>{smoker}</strong></div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ========== FOOTER ==========
st.markdown("""
    <div class="footer">
        MedInsure v1.0 | © 2025 | Built with Streamlit by <a href="https://github.com/MDTanveerAlam1" target="_blank" style="color:#00d4ff">MD Tanveer Alam</a>
    </div>
""", unsafe_allow_html=True)

