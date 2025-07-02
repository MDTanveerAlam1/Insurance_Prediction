# insurance_app.py

import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="💸 MedInsure - Cost Estimator",
    page_icon="💸",
    layout="wide",
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
        <h1 style="color:#00d4ff">💸 MedInsure</h1>
        <p style="opacity:0.8; font-size:16px">AI-powered Medical Insurance Cost Estimator</p>
    </div>
""", unsafe_allow_html=True)

# ========== NAVIGATION ==========
page = st.sidebar.radio("📁 Navigation", ["🏠 Home", "💡 Predict", "📊 Insights", "ℹ️ About"])

# ========== HOME ==========
if page == "🏠 Home":
    st.markdown("""
        <div class="card">
            <h2>Welcome to MedInsure</h2>
            <p>This app uses a trained machine learning model to predict your annual medical insurance cost based on personal and health-related data.</p>
            <ul>
                <li>Trained on real-world demographic + medical cost data</li>
                <li>Powered by Random Forest Regression (R² ≈ 0.86)</li>
                <li>UI styled for modern clarity and readability</li>
            </ul>
            <p>Use the sidebar to navigate through the app ➡️</p>
        </div>
    """, unsafe_allow_html=True)

# ========== PREDICT ==========
elif page == "💡 Predict":
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

# ========== INSIGHTS ==========
elif page == "📊 Insights":
    st.subheader("📊 Simulated Insights (Age & BMI vs Cost)")

    sex_val = 1
    children = 1
    smoker_val = 0
    region_val = 0

    # Age vs Cost
    ages = list(range(18, 66, 3))
    bmi_val = 27
    costs_by_age = [model.predict(np.array([[a, sex_val, bmi_val, children, smoker_val, region_val]]))[0] for a in ages]
    df_age = pd.DataFrame({'Age': ages, 'Predicted Cost': costs_by_age})
    st.line_chart(df_age.set_index('Age'))

    # BMI vs Cost
    bmis = list(np.linspace(15, 45, 20))
    age_val = 35
    costs_by_bmi = [model.predict(np.array([[age_val, sex_val, b, children, smoker_val, region_val]]))[0] for b in bmis]
    df_bmi = pd.DataFrame({'BMI': bmis, 'Predicted Cost': costs_by_bmi})
    st.line_chart(df_bmi.set_index('BMI'))

# ========== ABOUT ==========
elif page == "ℹ️ About":
    st.markdown("""
        <div class="card">
            <h2>About This App</h2>
            <p>Developed by <strong>MD Tanveer Alam</strong> as part of a machine learning deployment project.</p>
            <ul>
                <li>Language: Python</li>
                <li>Framework: Streamlit</li>
                <li>Model: Random Forest Regressor</li>
            </ul>
            <p>GitHub: <a href="https://github.com/MDTanveerAlam1" target="_blank">MDTanveerAlam1</a></p>
        </div>
    """, unsafe_allow_html=True)

# ========== FOOTER ==========
st.markdown("""
    <div class="footer">
        MedInsure v2.0 | © 2025 | Built by <a href="https://github.com/MDTanveerAlam1" target="_blank" style="color:#00d4ff">MD Tanveer Alam</a>
    </div>
""", unsafe_allow_html=True)
