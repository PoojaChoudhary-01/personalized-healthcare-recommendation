import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Personalized Healthcare Recommendation App")

uploaded_file = st.file_uploader("Upload your Healthcare dataset CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    st.subheader("Data Visualization")
    col_to_plot = st.selectbox("Select column to visualize", df.columns)

    if col_to_plot:
        fig, ax = plt.subplots()
        if pd.api.types.is_numeric_dtype(df[col_to_plot]):
            sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax)
        else:
            df[col_to_plot].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    st.subheader("Healthcare Recommendation Simulator")

    st.write("Enter your health details to get a personalized recommendation.")

    # Customized input form based on your columns
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Gender", options=df['Gender'].dropna().unique())
    blood_type = st.selectbox("Blood Type", options=df['Blood Type'].dropna().unique())
    medical_condition = st.text_input("Medical Condition (if any)")
    admission_type = st.selectbox("Admission Type", options=df['Admission Type'].dropna().unique())
    billing_amount = st.number_input("Billing Amount", min_value=0.0, value=0.0, step=0.01)

    if st.button("Get Recommendation"):
        # Dummy risk calculation based on age and billing amount
        risk_score = 0
        risk_score += (age / 100)
        risk_score += (billing_amount / 10000)
        risk_score += 0.3 if medical_condition else 0

        if risk_score > 1.5:
            st.error("High risk detected. Please consult a healthcare professional.")
        elif risk_score > 1.0:
            st.warning("Moderate risk. Consider lifestyle changes and regular checkups.")
        else:
            st.success("Low risk. Keep up the healthy lifestyle!")

else:
    st.info("Please upload a CSV file to get started.")

