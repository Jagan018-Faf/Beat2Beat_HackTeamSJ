import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="CardioInsight Clinical AI Suite",
    layout="wide"
)

# -------------------------------------------------------
# Clean Medical Theme
# -------------------------------------------------------
st.markdown("""
<style>
    body {
        background-color: #f4f8fb;
    }
    h1 {
        color: #003366;
    }
    h2, h3 {
        color: #004080;
    }
    .stButton>button {
        background-color: #003366;
        color: white;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown("""
<h1>CardioInsight Clinical AI Suite</h1>
<p style='color:gray;'>Multi-Stage Cardiovascular Decision Support System</p>
<hr>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar Navigation
# -------------------------------------------------------
st.sidebar.title("Navigation")

module = st.sidebar.radio(
    "Select Module",
    [
        "Stage 1 - Early Screening",
        "Stage 2 - Clinical Evaluation",
        "Stage 3 - ECG Analysis"
    ]
)

patient_id = st.sidebar.text_input("Patient ID", "PT-001")

# ======================================================
# STAGE 1 - EARLY SCREENING (PATIENT FRIENDLY)
# ======================================================

if module == "Stage 1 - Early Screening":

    st.header("Early Cardiovascular Risk Screening")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100)

    with col2:
        height = st.number_input("Height (cm)", 100.0, 220.0)

    with col3:
        weight = st.number_input("Weight (kg)", 30.0, 200.0)

    col4, col5 = st.columns(2)

    with col4:
        systolic = st.number_input("Systolic Blood Pressure (mmHg)", 80, 200)

    with col5:
        diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 130)

    if st.button("Run Screening"):

        height_m = height / 100
        bmi = weight / (height_m ** 2)

        # BMI Classification
        if bmi < 18.5:
            bmi_class = "Underweight"
        elif bmi < 25:
            bmi_class = "Normal"
        elif bmi < 30:
            bmi_class = "Overweight"
        else:
            bmi_class = "Obese"

        # BP Classification
        if systolic >= 140 or diastolic >= 90:
            bp_class = "Hypertension"
        elif systolic >= 130 or diastolic >= 80:
            bp_class = "Elevated"
        else:
            bp_class = "Normal"

        # Simple risk logic
        risk_score = (age * 0.02) + (bmi * 0.03) + (systolic * 0.01)

        st.subheader("Screening Summary")
        st.write(f"Calculated BMI: **{bmi:.2f} ({bmi_class})**")
        st.write(f"Blood Pressure Classification: **{bp_class}**")
        st.write(f"Preliminary Risk Score: **{risk_score:.2f}**")

        if risk_score > 6:
            st.error("Elevated cardiovascular risk detected. Clinical evaluation recommended.")
        else:
            st.success("Low to moderate preliminary risk.")

# ======================================================
# STAGE 2 - CLINICAL EVALUATION
# ======================================================

elif module == "Stage 2 - Clinical Evaluation":

    st.header("Clinical Cardiovascular Evaluation")

    try:
        model = joblib.load("outputs/stage2_model.pkl")
        scaler = joblib.load("outputs/stage2_scaler.pkl")
    except:
        st.error("Stage 2 model files not found in outputs folder.")
        st.stop()

    df = pd.read_csv("data/heart_processed.csv")
    feature_columns = df.drop("HeartDisease", axis=1).columns

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 100)

    with col2:
        gender = st.selectbox("Gender", ["Female", "Male"])

    with col3:
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600)

    col4, col5, col6 = st.columns(3)

    with col4:
        resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 80, 200)

    with col5:
        fasting_bs = st.number_input("Fasting Blood Sugar (mg/dL)", 70, 300)

    with col6:
        max_hr = st.number_input("Maximum Heart Rate Achieved", 60, 220)

    col7, col8 = st.columns(2)

    with col7:
        chest_pain = st.selectbox(
            "Chest Pain Type",
            ["No Pain", "Typical Angina", "Atypical Angina", "Non-Anginal Pain"]
        )

    with col8:
        exercise_angina = st.selectbox("Chest Pain During Exercise?", ["No", "Yes"])

    oldpeak = st.number_input("ST Depression Level", 0.0, 10.0)

    st_slope = st.selectbox("ST Segment Slope", ["Up", "Flat", "Down"])

    if st.button("Run Clinical Prediction"):

        input_dict = dict.fromkeys(feature_columns, 0)

        input_dict["Age"] = age
        input_dict["RestingBP"] = resting_bp
        input_dict["Cholesterol"] = cholesterol
        input_dict["MaxHR"] = max_hr
        input_dict["Oldpeak"] = oldpeak

        input_dict["Sex_M"] = 1 if gender == "Male" else 0
        input_dict["FastingBS"] = 1 if fasting_bs > 120 else 0
        input_dict["ExerciseAngina_Y"] = 1 if exercise_angina == "Yes" else 0

        if chest_pain == "Typical Angina":
            input_dict["ChestPainType_TA"] = 1
        elif chest_pain == "Atypical Angina":
            input_dict["ChestPainType_ATA"] = 1
        elif chest_pain == "Non-Anginal Pain":
            input_dict["ChestPainType_NAP"] = 1

        if st_slope == "Flat":
            input_dict["ST_Slope_Flat"] = 1
        elif st_slope == "Up":
            input_dict["ST_Slope_Up"] = 1

        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Clinical Prediction Result")
        st.write(f"Risk Probability: **{probability*100:.2f}%**")

        if prediction == 1:
            st.error("High likelihood of cardiovascular disease detected.")
        else:
            st.success("Low likelihood of cardiovascular disease.")

# ======================================================
# STAGE 3 - ECG ANALYSIS (UPLOAD BASED)
# ======================================================

elif module == "Stage 3 - ECG Analysis":

    st.header("ECG Anomaly Analysis")

    try:
        model = joblib.load("outputs/stage3_ecg_model.pkl")
        scaler = joblib.load("outputs/stage3_ecg_scaler.pkl")
    except:
        st.error("Stage 3 model files not found in outputs folder.")
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload ECG Signal CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        ecg_df = pd.read_csv(uploaded_file, header=None)
        signal = ecg_df.values.flatten()
        signal = signal[signal != 0]

        if len(signal) == 0:
            st.error("Uploaded file contains no valid ECG signal.")
        else:
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            energy = np.sum(signal**2)

            feature_array = np.array([[mean_val, std_val, energy]])
            scaled_features = scaler.transform(feature_array)

            prediction = model.predict(scaled_features)[0]

            st.subheader("ECG Analysis Result")

            if prediction == -1:
                st.error("ECG anomaly detected.")
            else:
                st.success("Normal ECG pattern detected.")

            st.subheader("ECG Signal Preview")
            st.line_chart(signal)