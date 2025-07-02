# app.py

import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load model
model = joblib.load("random_forest_model.pkl")

# App title and description
st.set_page_config(page_title="Medical Insurance Cost Predictor", layout="centered")
st.title("🏥 Medical Insurance Cost Predictor")
st.markdown("""
Welcome to the Medical Insurance Cost Predictor app!  
Enter patient details below to estimate the insurance cost using a trained Random Forest model.
""")

# Sidebar info
st.sidebar.header("🔍 About")
st.sidebar.markdown("""
This app uses a **Random Forest Regressor** trained on patient data to predict medical insurance charges.  
Created with ❤️ using **Streamlit**.
""")

# Input form
with st.form("prediction_form"):
    st.subheader("Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 30)
        sex = st.selectbox("Sex", ["Male", "Female"])
        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
    with col2:
        children = st.slider("Number of Children", 0, 5, 0)
        smoker = st.selectbox("Smoker?", ["Yes", "No"])
        region = st.selectbox("Region", ["Southeast", "Southwest", "Northeast", "Northwest"])

    submit = st.form_submit_button("Predict 💡")

# Preprocess and predict
if submit:
    # Encode categorical variables
    sex_val = 1 if sex == "Male" else 0
    smoker_val = 1 if smoker == "Yes" else 0
    region_map = {"Southeast": 0, "Southwest": 1, "Northeast": 2, "Northwest": 3}
    region_val = region_map[region]

    # Prepare input
    input_data = np.array([[age, sex_val, bmi, children, smoker_val, region_val]])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Output
    st.success(f"💰 Estimated Medical Insurance Cost: **Rs{prediction:,.2f}**")

    st.info("This is only an estimate based on the trained model. Real costs may vary.")

# Footer
st.markdown("---")
st.markdown("Made by [MD Tanveer Alam](https://github.com/MDTanveerAlam1) | 📍 [LinkedIn](https://www.linkedin.com/in/md-tanveer-alam-b1ba14258)")
