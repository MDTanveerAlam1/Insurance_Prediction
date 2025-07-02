import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# === Load model ===
try:
    model = joblib.load("random_forest_model.pkl")
except Exception as e:
    st.error(f"\u274c Failed to load model: {e}")

# === Page Config ===
st.set_page_config(page_title="Insurance Cost Predictor", layout="wide", initial_sidebar_state="expanded")

# === Theme Toggle ===
theme = st.sidebar.selectbox("\ud83c\udfa8 Choose Theme", ["\u2600\ufe0f Day Mode", "\ud83c\udf19 Night Mode"])

# === Background Gradient ===
def apply_background_gradient(theme):
    if theme == "\u2600\ufe0f Day Mode":
        gradient = """
        <style>
        body {
            background: linear-gradient(to right, #d9a7c7, #fffcdc) !important;
            background-attachment: fixed;
        }
        </style>
        """
    else:
        gradient = """
        <style>
        body {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364) !important;
            background-attachment: fixed;
        }
        </style>
        """
    st.markdown(gradient, unsafe_allow_html=True)

apply_background_gradient(theme)

# === Custom Theme CSS ===
def inject_css(theme_mode):
    if theme_mode == "\u2600\ufe0f Day Mode":
        css = """
        <style>
            body { font-family: 'Segoe UI', sans-serif; }
            .main {
                background-color: rgba(255, 255, 255, 0.9);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0px 4px 25px rgba(0, 0, 0, 0.1);
            }
            h1, h2, h3 { color: #4a148c; font-weight: 700; }
            .stButton>button {
                background-color: #4a148c;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
            }
            .stButton>button:hover {
                background-color: #6a1b9a;
                transform: scale(1.03);
            }
        </style>
        """
    else:
        css = """
        <style>
            body { font-family: 'Segoe UI', sans-serif; color: #eee; }
            .main {
                background-color: rgba(30, 30, 30, 0.9);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0px 4px 25px rgba(255, 255, 255, 0.05);
            }
            h1, h2, h3 { color: #00c6ff; font-weight: 700; }
            .stButton>button {
                background-color: #00c6ff;
                color: #000;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
            }
            .stButton>button:hover {
                background-color: #0072ff;
                color: white;
                transform: scale(1.03);
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

inject_css(theme)

# === Sidebar Navigation ===
menu = st.sidebar.radio("\ud83d\udcc1 Navigation", ["\ud83c\udfe0 Home", "\ud83d\udcca Visualize Data", "\ud83e\udde0 Predict Cost", "\ud83d\udcc4 About App"])

# === Home Page ===
def home():
    st.title("\ud83c\udfe0 Welcome to Insurance Cost Predictor")
    st.markdown("""
    This application predicts **medical insurance charges** based on user inputs.

    ### Features:
    - Built using a **Random Forest** model
    - Modern UI with real-time prediction
    - Day/Night Theme Toggle
    - Interactive Data Visualizations
    """)
    st.image("https://cdn.pixabay.com/photo/2017/01/31/21/23/heart-2026026_1280.png", width=400)

# === Data Visualization Page ===
def visualize_data():
    st.title("\ud83d\udcca Explore the Dataset")
    data = pd.read_csv("insurance.csv")
    data.columns = data.columns.str.strip().str.lower()
    st.subheader("Dataset Overview")
    st.dataframe(data.head(10))

    col1, col2 = st.columns(2)
    with col1:
        if 'charges' in data.columns and 'smoker' in data.columns:
            plot = px.histogram(data, x='charges', nbins=40, color='smoker', title="Distribution of Charges")
            st.plotly_chart(plot, use_container_width=True)
        else:
            st.warning("Columns 'charges' or 'smoker' not found in dataset.")

    with col2:
        if 'region' in data.columns and 'charges' in data.columns:
            plot2 = px.box(data, x='region', y='charges', color='region', title="Charges by Region")
            st.plotly_chart(plot2, use_container_width=True)
        else:
            st.warning("Columns 'region' or 'charges' not found in dataset.")

# === Prediction Page ===
def predict():
    st.title("\ud83e\udde0 Predict Insurance Cost")
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fdfcfb, #e2d1c3);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 5px 25px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    ">
        <h2 style="color:#4a148c; font-weight:700; text-align:center;">\ud83d\udccb Enter Patient Details</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("\ud83e\uddc3 Age", 18, 100, 25)
        bmi = st.number_input("\u2696\ufe0f BMI", 10.0, 60.0, step=0.1)
        children = st.selectbox("\ud83d\udc76 Number of Children", [0, 1, 2, 3, 4, 5])

    with col2:
        sex = st.radio("\ud83e\uddd1 Sex", ["male", "female"], horizontal=True)
        smoker = st.radio("\ud83d\udeac Smoker", ["yes", "no"], horizontal=True)
        region = st.selectbox("\ud83c\udf0d Region", ["southwest", "southeast", "northwest", "northeast"])

    st.markdown("</div>", unsafe_allow_html=True)

    user_input = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])

    user_input_encoded = pd.get_dummies(user_input)
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in user_input_encoded:
            user_input_encoded[col] = 0
    user_input_encoded = user_input_encoded[model_features]

    if st.button("\ud83d\udd2e Predict Cost"):
        prediction = model.predict(user_input_encoded)[0]
        st.markdown(f"""
        <div style="
            background: linear-gradient(to right, #a1c4fd, #c2e9fb);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.4rem;
            font-weight: bold;
            color: #003366;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
        ">
            \ud83d\udcb0 Estimated Insurance Charges: <br> <span style="font-size: 2rem;">Rs {prediction:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

# === About Page ===
def about():
    st.title("\ud83d\udcc4 About this App")
    st.markdown("""
    **Insurance Cost Predictor** is a user-friendly ML dashboard built with:
    - \ud83e\udde0 **Random Forest** for backend model  
    - \ud83d\udda5\ufe0f **Streamlit** for interactive frontend  
    - \ud83d\udcca **Plotly** for modern visualizations

    ### \ud83d\udc65 Project Team:
    - **MD. Tanveer Alam** *(Team Lead)*  
    - Thanmai Yadla  
    - Pushpesh Kumar  
    - Shivani Kumari  
    - Chaitanya Kumar Reddy Padala  
    - Hrithik S. Raveendran  
    - Megha Macchindra Sanap

    ---
    \ud83d\udd17 [GitHub](https://github.com/MDTanveerAlam1)  
    \ud83d\udd17 [LinkedIn](https://www.linkedin.com/in/md-tanveer-alam-b1ba14258)
    """)

# === Routing ===
if menu == "\ud83c\udfe0 Home":
    home()
elif menu == "\ud83d\udcca Visualize Data":
    visualize_data()
elif menu == "\ud83e\udde0 Predict Cost":
    predict()
elif menu == "\ud83d\udcc4 About App":
    about()
