import streamlit as st
import joblib
import json
import pandas as pd

# Load model, scaler, and expected columns
model = joblib.load("best_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

with open("model_columns.json", "r") as f:
    model_columns = json.load(f)

st.title("🎓 Student Performance Predictor (Study Buddy AI)")

# Input form
with st.form("input_form"):
    st.subheader("Enter Student Details")

    age = st.slider("Age", 15, 30, 21)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    city = st.selectbox("City", ["Urban", "Semi-Urban", "Rural"])
    profession = st.selectbox("Profession", ["Student"])
    degree = st.selectbox("Degree", ["Bachelor", "Master", "PhD"])

    academic_pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
    work_pressure = st.slider("Work Pressure (1-5)", 1, 5, 2)

    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)
    study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
    job_satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 3)

    sleep_duration = st.slider("Sleep Duration (hours)", 0.0, 12.0, 7.0)
    study_hours = st.slider("Work/Study Hours per Day", 0, 16, 5)

    suicidal_thoughts = st.selectbox("Suicidal Thoughts?", ["No", "Yes"])
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
    financial_stress = st.selectbox("Financial Stress", ["Low", "Moderate", "High"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create raw input dictionary
    raw_input = {
        'Age': age,
        'Gender': gender,
        'City': city,
        'Profession': profession,
        'Degree': degree,
        'Academic Pressure': academic_pressure,
        'Work Pressure': work_pressure,
        'CGPA': cgpa,
        'Study Satisfaction': study_satisfaction,
        'Job Satisfaction': job_satisfaction,
        'Sleep Duration': sleep_duration,
        'Work/Study Hours': study_hours,
        'Have you ever had suicidal thoughts ?': suicidal_thoughts,
        'Family History of Mental Illness': family_history,
        'Financial Stress': financial_stress
    }

    # Add derived feature
    raw_input['Total Pressure'] = academic_pressure + work_pressure

    # Convert to DataFrame
    sample_df = pd.DataFrame([raw_input])

    # One-hot encoding
    categorical_features = ['Gender', 'City', 'Profession', 'Degree',
                            'Have you ever had suicidal thoughts ?',
                            'Family History of Mental Illness', 'Financial Stress']
    sample_df = pd.get_dummies(sample_df, columns=categorical_features, drop_first=True)

    # Scale numeric features
    numeric_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA',
                        'Study Satisfaction', 'Job Satisfaction',
                        'Sleep Duration', 'Work/Study Hours', 'Total Pressure']
    sample_df[numeric_features] = scaler.transform(sample_df[numeric_features])

    # Add missing columns with 0s
    missing_cols = [col for col in model_columns if col not in sample_df.columns]
    missing_df = pd.DataFrame(0, index=sample_df.index, columns=missing_cols)
    sample_df = pd.concat([sample_df, missing_df], axis=1)

    # Reorder columns to match training
    sample_df = sample_df[model_columns]

    # Predict
    prediction = model.predict(sample_df)
    predicted_class = prediction[0]

    st.success(f"🎯 **Predicted Performance Category**: `{predicted_class}`")
