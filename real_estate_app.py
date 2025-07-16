
import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# App title
st.title('🏠 Real Estate Price Prediction App')

# Sidebar info
st.sidebar.title("🔍 About")
st.sidebar.image("yash_profile.jpg", width=150)
st.sidebar.markdown("**Developer**: Yash Patil")
st.sidebar.markdown("**Project**: Real Estate Price Prediction")
st.sidebar.markdown("[📂 Dataset Source (Google Drive)](https://drive.google.com/file/d/11Ju9jvVQ47Rr_IN76f53YWxcHUi_D6OZ/view?usp=drive_link)")

# Contact info
st.sidebar.markdown("📧 **Email**: [yashpatil877@gmail.com](mailto:yashpatil877@gmail.com)")
st.sidebar.markdown("🔗 **LinkedIn**: [yash-patil-560933b9](https://www.linkedin.com/in/yash-patil-560933b9)")

st.write("Enter the property features below to predict the **house price per unit area (in ₹1,000s)** and the **final house price**.")

# User inputs
age = st.number_input('🏠 House Age (in years)', min_value=0.0, step=0.5)
distance = st.number_input('📍 Distance to the nearest MRT station (in meters)', min_value=0.0, step=10.0)
stores = st.number_input('🛒 Number of convenience stores', min_value=0, step=1)
latitude = st.number_input('🌐 Latitude', format="%.6f")
longitude = st.number_input('🧭 Longitude', format="%.6f")
unit_area = st.number_input('📐 Size of the Property (in square meters)', min_value=1.0, step=1.0)

# Predict button
if st.button('🔍 Predict Price'):
    input_data = np.array([[age, distance, stores, latitude, longitude]])
    input_scaled = scaler.transform(input_data)  # ✅ Apply scaling
    unit_price = model.predict(input_scaled)[0]  # Prediction in ₹1,000s
    final_price = unit_price * unit_area * 1000  # Final price in ₹

    st.info(f"Predicted House Price of Unit Area (₹1,000s): {unit_price:.6f}")
    st.success(f"💰 Final Predicted House Price: ₹{final_price:,.2f}")
