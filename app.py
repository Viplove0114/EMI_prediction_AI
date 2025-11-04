import streamlit as st
import pandas as pd

# Set the page configuration
st.set_page_config(
    page_title="EMIPredict AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    st.title("🤖 EMIPredict AI")
    st.subheader("Intelligent Financial Risk Assessment Platform")

    st.markdown("""
    Welcome to the EMIPredict AI platform. This tool is built to solve a dual machine learning problem:
    1.  **Classification:** Predicting if a person is eligible for an EMI (Equated Monthly Instalment).
    2.  **Regression:** Predicting the maximum safe EMI amount a person can afford.

    This application uses a dataset of 400,000 financial records to provide data-driven insights.
    """)

    st.image(
        "https://placehold.co/1200x400/1e3a8a/ffffff?text=Financial+Risk+Dashboard",
        caption="Data-Driven Financial Insights"
    )

    st.header("How to Use This App")
    st.markdown("""
    Navigate through the app using the sidebar on the left:

    1.  **🏠 Home:**
        * You are here. This page gives an overview of the project.

    2.  **📊 Data Explorer:**
        * View the distributions of the key financial variables.
        * Analyze how different features (like credit score or salary) correlate with EMI eligibility.

    3.  **🚀 Model Training:**
        * **This is the most important step!** You must run this page first.
        * Click the button to load and process all 400,000 records.
        * This will train all 6 machine learning models (Logistic Regression, Random Forest, XGBoost) and log them to MLflow.
        * This page also saves the necessary artifacts (scaler, encoder, model paths) that the prediction page needs.

    4.  **🔮 Live Prediction:**
        * Once the models are trained, come here to get a real-time risk assessment.
        * Enter a person's financial details into the form.
        * The app will use the best-trained models (XGBoost) to predict their EMI eligibility and maximum affordable EMI instantly.
    """)

    st.warning("⚠️ **Action Required:** Please go to the **🚀 Model Training** page first to train the models before using the **🔮 Live Prediction** page.")

if __name__ == "__main__":
    main()