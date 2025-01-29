import pickle
import streamlit as st
import os
import numpy as np  

# Streamlit App Title
st.title("Crop Recommendation App")

# Load Model
file_path = os.path.join("model", "nb_model.pkl")

# Check if directory exists
if not os.path.exists("model"):
    st.error("Model directory not found! Please create a 'model' folder and place 'nb_model.pkl' inside.")
    model = None
elif not os.path.exists(file_path):
    st.error("Model file not found! Please ensure 'nb_model.pkl' is in the 'model' directory.")
    model = None
else:
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None

# Input Fields
nv = st.number_input("Nitrogen (kg/ha)", min_value=1, max_value=100)
phv = st.number_input("Phosphorus (kg/ha)", min_value=30, max_value=100)
pv = st.number_input("Potassium (kg/ha)", min_value=10, max_value=50)
tp = st.number_input("Temperature (°C)", min_value=10, max_value=50)
ht = st.number_input("Humidity (%)", min_value=15, max_value=100)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)  # Fixed pH input
rf = st.number_input("Rainfall (mm)", min_value=50, max_value=250)

# Prediction
if st.button("Recommend"):
    if model is not None:  # Ensure model is loaded
        input_data = np.array([[nv, phv, pv, tp, ht, ph, rf]])
        prediction = model.predict(input_data)
        st.write("Recommended Crop is:", prediction[0])
    else:
        st.error("Model not loaded. Check the file path or model integrity.")
