"""Heart Disease Prediction - Streamlit web app (AIC354, Assignment 2)."""
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")


@st.cache_resource
def load_artifacts():
    """Load the trained SVC and the scaler saved by the notebook (Steps 6.3 and 6.5)."""
    with open(os.path.join(HERE, "heart_svc_trained_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(HERE, "heart_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


# Same fixed category lists as Step 5.1 of the notebook
SEX_OPTIONS = ["Male", "Female"]
CHEST_PAIN_OPTIONS = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]

sex_label_encoder = LabelEncoder().fit(SEX_OPTIONS)
chestpain_label_encoder = LabelEncoder().fit(CHEST_PAIN_OPTIONS)

model, scaler = load_artifacts()

st.title("❤️ Heart Disease Prediction System")
st.write(
    "Enter four details about a patient and the trained Support Vector Classifier "
    "will predict whether heart disease is present. Trained on the UCI Heart Disease "
    "(Cleveland) dataset."
)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=20, max_value=90, value=55, step=1)
    sex = st.selectbox("Sex", SEX_OPTIONS)
with col2:
    max_hr = st.number_input("Maximum heart rate achieved (bpm)", min_value=60, max_value=220, value=150, step=1)
    chest_pain = st.selectbox("Chest pain type", CHEST_PAIN_OPTIONS)

if st.button("Predict"):
    # Step 8.2: feature vector (same column order as the training data)
    user_input = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPain": [chest_pain],
        "MaxHeartRate": [max_hr],
    })
    # Step 8.3: label encoding, then scaling
    user_input["Sex"] = sex_label_encoder.transform(user_input["Sex"])
    user_input["ChestPain"] = chestpain_label_encoder.transform(user_input["ChestPain"])
    scaled = scaler.transform(user_input)
    # Step 8.5: prediction
    prediction = model.predict(scaled)[0]

    if prediction == 1:
        st.error("Prediction: **HEART DISEASE** likely present")
    else:
        st.success("Prediction: **NO HEART DISEASE** predicted")

st.caption("For educational purposes only. This is not a medical diagnosis tool.")
