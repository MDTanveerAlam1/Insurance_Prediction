# app.py
import streamlit as st
import joblib
import numpy as np

# Load the saved Random Forest model
model = joblib.load('random_forest_model.pkl')

st.title("Medical Cost Predictor")

# User Inputs
age = st.slider("Age", 18, 100, 30)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
children = st.slider("Number of Children", 0, 5, 0)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

# Encode categorical values (use your model’s encoding logic)
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0
region_map = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
region = region_map[region]

# Create input array
input_data = np.array([[age, sex, bmi, children, smoker, region]])

# Predict
if st.button("Predict Medical Cost"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Medical Cost: Rs {prediction[0]:,.2f}")
