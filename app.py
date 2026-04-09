import streamlit as st
import pandas as pd
import joblib

# Load pipeline (with preprocessor)
pipeline = joblib.load("heart_disease_pipeline.pkl")

st.title("Heart Disease Prediction App")

# User input form
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["M", "F"])
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
resting_bp = st.number_input("RestingBP", min_value=80, max_value=200, value=120)
max_hr = st.number_input("MaxHR", min_value=60, max_value=200, value=150)
oldpeak = st.number_input("Oldpeak", min_value=-2.0, max_value=6.0, value=1.0)
exercise_angina = st.selectbox("ExerciseAngina", ["Y", "N"])
fasting_bs = st.selectbox("FastingBS", [0, 1])
chest_pain = st.selectbox("ChestPainType", ["ATA", "NAP", "ASY", "TA"])
st_slope = st.selectbox("ST_Slope", ["Up", "Flat", "Down"])
resting_ecg = st.selectbox("RestingECG", ["Normal", "LVH", "ST"])  # <-- added

# Collect raw inputs into DataFrame
input_data = pd.DataFrame({
    "Age": [age],
    "Sex": [sex],
    "Cholesterol": [cholesterol],
    "RestingBP": [resting_bp],
    "MaxHR": [max_hr],
    "Oldpeak": [oldpeak],
    "ExerciseAngina": [exercise_angina],
    "FastingBS": [fasting_bs],
    "ChestPainType": [chest_pain],
    "ST_Slope": [st_slope],
    "RestingECG": [resting_ecg]   # <-- included here
})

# Prediction
if st.button("Predict"):
    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0]

    st.write("### Prediction:", "Heart Disease" if prediction == 1 else "No Disease")
    st.write("### Probabilities")
    st.write(f"No Disease: {prob[0]:.2f}")
    st.write(f"Disease: {prob[1]:.2f}")

