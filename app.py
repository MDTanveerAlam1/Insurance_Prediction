import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px

# Load model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page settings
st.set_page_config(
    page_title="Insurance Cost Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========= THEME TOGGLE =========
theme = st.sidebar.selectbox("🎨 Choose Theme", ["☀️ Day Mode", "🌙 Night Mode"])

# ========= DYNAMIC CSS =========
def inject_css(theme_mode):
    if theme_mode == "☀️ Day Mode":
        css = """
        <style>
            body {
                background: linear-gradient(to right, #d9a7c7, #fffcdc);
                font-family: 'Segoe UI', sans-serif;
            }
            .main {
                background-color: rgba(255, 255, 255, 0.9);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0px 4px 25px rgba(0, 0, 0, 0.1);
                backdrop-filter: blur(6px);
            }
            h1, h2, h3 {
                color: #4a148c;
                font-weight: 700;
            }
            .sidebar .sidebar-content {
                background: linear-gradient(to bottom, #11998e, #38ef7d);
                color: white;
            }
            .stButton>button {
                background-color: #4a148c;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: 600;
                transition: all 0.3s ease;
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
            body {
                background: linear-gradient(to right, #141e30, #243b55);
                font-family: 'Segoe UI', sans-serif;
                color: #eee;
            }
            .main {
                background-color: rgba(30, 30, 30, 0.9);
                padding: 2rem;
                border-radius: 12px;
                box-shadow: 0px 4px 25px rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(6px);
            }
            h1, h2, h3 {
                color: #00c6ff;
                font-weight: 700;
            }
            .sidebar .sidebar-content {
                background: linear-gradient(to bottom, #0f2027, #203a43, #2c5364);
                color: white;
            }
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
                transition: 0.3s;
            }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

inject_css(theme)

# ========= Sidebar Navigation =========
menu = st.sidebar.radio("📁 Navigation", ["🏠 Home", "📊 Visualize Data", "🧠 Predict Cost", "📄 About App"])

# ========= Pages =========
def home():
    st.title("🏠 Welcome to Insurance Cost Predictor")
    st.markdown("""
    This application predicts **medical insurance charges** based on user inputs.

    ### Features:
    - Built using a **Random Forest** model
    - Modern UI with real-time prediction
    - Day/Night Theme Toggle
    - Interactive Data Visualizations
    """)
    st.image("https://cdn.pixabay.com/photo/2017/01/31/21/23/heart-2026026_1280.png", width=400)

def visualize_data():
    st.title("📊 Explore the Dataset")
    data = pd.read_csv("insurance.csv")

    st.subheader("Dataset Overview")
    st.dataframe(data.head(10))

    col1, col2 = st.columns(2)
    with col1:
        plot = px.histogram(data, x='charges', nbins=40, color='smoker', title="Distribution of Charges")
        st.plotly_chart(plot, use_container_width=True)
    with col2:
        plot2 = px.box(data, x='region', y='charges', color='region', title="Charges by Region")
        st.plotly_chart(plot2, use_container_width=True)

def predict():
    st.title("🧠 Predict Insurance Cost")

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #fdfcfb, #e2d1c3);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 5px 25px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    ">
        <h2 style="color:#4a148c; font-weight:700; text-align:center;">📋 Enter Patient Details</h2>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("🧓 Age", 18, 100, 25)
        bmi = st.number_input("⚖️ BMI", 10.0, 60.0, step=0.1)
        children = st.selectbox("👶 Number of Children", [0, 1, 2, 3, 4, 5])

    with col2:
        sex = st.radio("🧍 Sex", ["male", "female"], horizontal=True)
        smoker = st.radio("🚬 Smoker", ["yes", "no"], horizontal=True)
        region = st.selectbox("🌍 Region", ["southwest", "southeast", "northwest", "northeast"])

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

    if st.button("🔮 Predict Cost"):
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
            💰 Estimated Insurance Charges: <br> <span style="font-size: 2rem;">Rs {prediction:,.2f}</span>
        </div>
        """, unsafe_allow_html=True)

def about():
    st.title("📄 About this App")
    st.markdown("""
    **Insurance Cost Predictor** is a user-friendly ML dashboard built with:
    - 🧠 **Random Forest** for backend model  
    - 🖥️ **Streamlit** for interactive frontend  
    - 📊 **Plotly** for modern visualizations

    ### 👥 Project Team:
    - **MD. Tanveer Alam** *(Team Lead)*  
    - Thanmai Yadla  
    - Pushpesh Kumar  
    - Shivani Kumari  
    - Chaitanya Kumar Reddy Padala  
    - Hrithik S. Raveendran  
    - Megha Macchindra Sanap

    ---
    🔗 [GitHub](https://github.com/MDTanveerAlam1)  
    🔗 [LinkedIn](https://www.linkedin.com/in/md-tanveer-alam-b1ba14258)
    """)

# ========= Page Routing =========
if menu == "🏠 Home":
    home()
elif menu == "📊 Visualize Data":
    visualize_data()
elif menu == "🧠 Predict Cost":
    predict()
elif menu == "📄 About App":
    about()
