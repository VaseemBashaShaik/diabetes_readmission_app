import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


import os
import streamlit as st
import joblib

@st.cache_resource
def load_assets():
    model_path = os.path.join("sample_data", "gb_best_model.joblib")
    pre_path = os.path.join("sample_data", "preprocessor.joblib")

    if not os.path.exists(model_path) or not os.path.exists(pre_path):
        st.error("❌ Model or preprocessor file not found in sample_data/.  "
                 "Upload gb_best_model.joblib and preprocessor.joblib to your repo.")
        st.stop()

    model = joblib.load(model_path)
    pre = joblib.load(pre_path)
    return model, pre

# ----------------------------
# 🔧 PAGE CONFIG & STYLING
# ----------------------------
st.set_page_config(
    page_title="🏥 Diabetes Readmission Predictor",
    page_icon="💉",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #E8F1FF 0%, #F9FBFC 100%);
            color: #003366;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-title {
            text-align: center;
            color: #004E89;
            font-size: 2.3rem;
            font-weight: 700;
        }
        .result-box {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 🧠 LOAD MODEL + PREPROCESSOR
# ----------------------------
@st.cache_resource
def load_assets():
    model_path = os.path.join("sample_data", "gb_best_model.joblib")
    pre_path = os.path.join("sample_data", "preprocessor.joblib")

    if not os.path.exists(model_path) or not os.path.exists(pre_path):
        st.error("❌ Model or preprocessor file not found. Please check your sample_data folder.")
        st.stop()

    model = joblib.load(model_path)
    pre = joblib.load(pre_path)
    return model, pre

model, pre = load_assets()
STABLE_THRESHOLD = 0.45

# ----------------------------
# 🏥 HEADER
# ----------------------------
st.markdown("<h1 class='main-title'>Diabetes Readmission Risk Predictor</h1>", unsafe_allow_html=True)
st.write("Predict whether a diabetic patient will be readmitted within 30 days after discharge. Built using a tuned Gradient Boosting model.")

# ----------------------------
# 🧍 SIDEBAR INPUT
# ----------------------------
st.sidebar.header("🧾 Patient Information")

race = st.sidebar.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age_group = st.sidebar.selectbox("Age Group", ["Young", "Middle", "Elderly"])
time_in_hospital = st.sidebar.number_input("Days in Hospital", 1, 14, 4)
num_lab_procedures = st.sidebar.number_input("Lab Procedures", 1, 100, 40)
num_medications = st.sidebar.number_input("Medications", 1, 50, 10)
number_outpatient = st.sidebar.number_input("Outpatient Visits", 0, 20, 0)
number_emergency = st.sidebar.number_input("Emergency Visits", 0, 20, 0)
number_inpatient = st.sidebar.number_input("Inpatient Visits", 0, 20, 0)
total_visits = number_outpatient + number_emergency + number_inpatient
admission_type_id = st.sidebar.selectbox("Admission Type ID", [1,2,3,4,5,6,7,8])
discharge_disposition_id = st.sidebar.selectbox("Discharge Disposition ID", [1,2,3,4,5,6,7,8,9,10])
admission_source_id = st.sidebar.selectbox("Admission Source ID", [1,2,3,4,5,6,7,8,9,10])

predict = st.sidebar.button("🔍 Predict Readmission")

# ----------------------------
# 🔬 PREDICTION
# ----------------------------
if predict:
    st.markdown("---")
    st.subheader("🔬 Prediction Result")

    data = {
        "race": race,
        "gender": gender,
        "age_group": age_group,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "total_visits": total_visits,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id
    }

    df = pd.DataFrame([data])
    for col in pre.feature_names_in_:
        if col not in df.columns:
            df[col] = np.nan
    df = df[pre.feature_names_in_]

    Xp = pre.transform(df)
    prob = model.predict_proba(Xp)[0, 1]
    pred = int(prob >= STABLE_THRESHOLD)

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    if pred == 1:
        st.error(f"🚨 High Risk of Readmission (Probability: {prob:.1%})")
        st.write("⚠️ Suggest closer monitoring and follow-up appointments.")
    else:
        st.success(f"✅ Low Risk of Readmission (Probability: {prob:.1%})")
        st.write("👍 Stable patient condition.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.progress(float(prob))

st.markdown("""
---
👩‍⚕️ *Developed for diabetic patient monitoring and healthcare analytics.*
""")

