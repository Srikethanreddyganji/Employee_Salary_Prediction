import streamlit as st
import pandas as pd
import joblib

model = joblib.load("salary_model.pkl")
label_encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.title("Employee Salary Prediction")

education = st.selectbox("Education", label_encoders["education"].classes_)
occupation = st.selectbox("Occupation", label_encoders["occupation"].classes_)
gender = st.selectbox("Gender", label_encoders["gender"].classes_)
hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)

if st.button("Predict Salary"):
    edu_encoded = label_encoders["education"].transform([education])[0]
    occ_encoded = label_encoders["occupation"].transform([occupation])[0]
    gen_encoded = label_encoders["gender"].transform([gender])[0]

    input_df = pd.DataFrame([{
        "education": edu_encoded,
        "occupation": occ_encoded,
        "gender": gen_encoded,
        "hours-per-week": hours_per_week
    }])

    prediction = model.predict(input_df)
    predicted_class = target_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted Income Class: {predicted_class}")
