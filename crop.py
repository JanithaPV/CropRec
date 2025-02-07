import pickle
import streamlit as st
from os import path
import numpy as np

st.title("Crop Recommendation App")

file_name="nb_model.pkl"
with open(path.join(file_name),'rb')as f:
    nb_model=pickle.load(f)

    nv = st.number_input("Nitrogen (kg/ha)", min_value=0, max_value=900)
    
    phv = st.number_input("Phosphorus (kg/ha)", min_value=0, max_value=700)
    pv = st.number_input("Potassium (kg/ha)", min_value=0, max_value=500)
    tp = st.number_input("Temperature (°C)", min_value=0.0, max_value=600.0)
    ht = st.number_input("Humidity (%)", min_value=0.0, max_value=700.0)
    ph = st.number_input("pH", min_value=0.0, max_value=100.0, step=0.1)  # Fixed pH input
    rf = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1250.0)


    if st.button("Recommend"):
        pred=nb_model.predict(np.array([[nv, phv, pv, tp, ht, ph, rf]]))
        st.write("Recommended Crop is:",pred[0])
