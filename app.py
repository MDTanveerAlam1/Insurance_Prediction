# app.py

import streamlit as st
import numpy as np
import joblib

# Load ML model
model = joblib.load("random_forest_model.pkl")

# Page config
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")

# --- Sidebar Navigation ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2920/2920277.png", width=100)
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Model Info", "ℹ️ About"])

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #0E4D92;
        color: white;
        font-weight: bold;
    }
    .stMarkdown h1, h2, h3 {
        color: #0E4D92;
    }
    </style>
""", unsafe_allow_html=True)

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center;'>🏥 Medical Insurance Cost Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; color: grey;'>Estimate your insurance cost using Machine Learning</div><br>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.subheader("📝 Enter Patient Details")

        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("🎂 Age", 18, 100, 30)
            sex = st.radio("👤 Sex", ["Male", "Female"])
            bmi = st.slider("⚖️ BMI", 10.0, 50.0, 25.0)
        with col2:
            children = st.number_input("👶 Number of Children", 0, 10, 0)
            smoker = st.radio("🚬 Smoker?", ["Yes", "No"])
            region = st.selectbox("📍 Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

        submit = st.form_submit_button("💡 Predict Cost")

    if submit:
        # Encode input
        sex_val = 1 if sex == "Male" else 0
        smoker_val = 1 if smoker == "Yes" else 0
        region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
        region_val = region_map[region]

        input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])
        prediction = model.predict(input_data)[0]

        st.success(f"💰 **Estimated Medical Insurance Cost: Rs {prediction:,.2f}**")
        with st.expander("📌 Tips to Lower Your Insurance Cost"):
            st.markdown("""
            - Maintain a healthy BMI  
            - Avoid smoking  
            - Opt for preventive checkups  
            - Compare providers before buying plans  
            """)

# --- MODEL INFO PAGE ---
elif page == "📊 Model Info":
    st.markdown("<h2>📊 Model & Dataset Information</h2>", unsafe_allow_html=True)
    st.markdown("""
    This model is trained using a **Random Forest Regressor**.  
    It uses the following input features to predict medical insurance charges:
    - `age` - Age of the individual  
    - `sex` - Gender  
    - `bmi` - Body Mass Index  
    - `children` - Number of dependent children  
    - `smoker` - Smoking status  
    - `region` - Residential region  

    **Why Random Forest?**
    - It handles non-linear data well  
    - Less prone to overfitting  
    - Good performance on small tabular datasets  
    """)

# --- ABOUT PAGE ---
elif page == "ℹ️ About":
    st.markdown("<h2>ℹ️ About This App</h2>", unsafe_allow_html=True)
    st.markdown("""
    This app helps users estimate their medical insurance cost based on simple personal and lifestyle inputs.  
    It uses a machine learning model trained on publicly available insurance data.

    **Technologies Used:**
    - Python 🐍  
    - Streamlit 💻  
    - Scikit-learn ⚙️  
    - Random Forest Model 🌲

    **Developer:**  
    [MD Tanveer Alam](https://github.com/MDTanveerAlam1)  
    [LinkedIn Profile](https://www.linkedin.com/in/md-tanveer-alam-b1ba14258)
    """)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 14px; color: gray'>
Made with ❤️ by <a href='https://github.com/MDTanveerAlam1' target='_blank'>MD Tanveer Alam</a> |
<a href='https://www.linkedin.com/in/md-tanveer-alam-b1ba14258' target='_blank'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
