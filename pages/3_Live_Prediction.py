import streamlit as st
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import os

# Import the custom preprocessing function
from utils import preprocess_for_prediction

st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")

st.title("🔮 Live EMI Risk Prediction")
st.markdown("Enter a person's financial details to predict their EMI eligibility and maximum affordable EMI.")

# --- Helper Function to Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """
    Loads all necessary artifacts: scaler, encoder, feature list, and the two best models.
    """
    # Define paths
    SCALER_PATH = 'standard_scaler.joblib'
    ENCODER_PATH = 'label_encoder.joblib'
    FEATURES_PATH = 'feature_names.txt'
    
    # Check if artifacts exist
    if not all(os.path.exists(p) for p in [SCALER_PATH, ENCODER_PATH, FEATURES_PATH]):
        st.error("Artifacts not found! Please run the '🚀 Model Training' page first.")
        return None

    # Load artifacts
    try:
        scaler = joblib.load(SCALER_PATH)
        le = joblib.load(ENCODER_PATH)
        with open(FEATURES_PATH, 'r') as f:
            feature_names = [line.strip() for line in f]
    except Exception as e:
        st.error(f"Error loading base artifacts: {e}")
        return None

    # Load MLflow models
    # We'll use the "best" models, which we've decided are XGBoost
    try:
        # Load from the registered model name
        model_class = mlflow.pyfunc.load_model(model_uri="models:/XGBoostClassifier/latest")
        model_reg = mlflow.pyfunc.load_model(model_uri="models:/XGBoostRegressor/latest")
    except Exception as e:
        st.error(f"Error loading models from MLflow Registry: {e}")
        st.info("Make sure MLflow is running and models 'XGBoostClassifier' and 'XGBoostRegressor' are registered.")
        return None

    return scaler, le, feature_names, model_class, model_reg

artifacts = load_artifacts()

if artifacts:
    scaler, le, feature_names, model_class, model_reg = artifacts
    st.success("Models and artifacts loaded successfully. Ready for prediction.")

    # --- Create the Input Form ---
    with st.form("prediction_form"):
        st.header("Applicant's Financial Details")
        
        # --- Form Columns ---
        col1, col2, col3 = st.columns(3)
        
        # --- Column 1: Personal Demographics ---
        with col1:
            st.subheader("Personal")
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            family_size = st.number_input("Family Size", min_value=1, max_value=10, value=2)
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=0)

        # --- Column 2: Employment & Income ---
        with col2:
            st.subheader("Employment & Income")
            monthly_salary = st.number_input("Monthly Salary (INR)", min_value=15000, max_value=500000, value=50000, step=1000)
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            company_type = st.selectbox("Company Type", ["Startup", "Mid-size", "MNC", "Small", "Other"])
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])

        # --- Column 3: Financials & Loan ---
        with col3:
            st.subheader("Financials & Loan")
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=750)
            existing_loans = st.selectbox("Existing Loans?", ["Yes", "No"])
            current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0, value=0.0, step=100.0)
            bank_balance = st.number_input("Bank Balance", min_value=0.0, value=100000.0, step=1000.0)
            emergency_fund = st.number_input("Emergency Fund", min_value=0.0, value=50000.0, step=1000.0)
            emi_scenario = st.selectbox("EMI Scenario", ["E-commerce Shopping EMI", "Home Appliances EMI", "Vehicle EMI", "Personal Loan EMI", "Education EMI"])
            requested_amount = st.number_input("Requested Amount", min_value=10000.0, value=100000.0, step=5000.0)
            requested_tenure = st.number_input("Requested Tenure (months)", min_value=3, max_value=84, value=12)

        st.header("Monthly Expenses")
        exp_col1, exp_col2, exp_col3, exp_col4, exp_col5 = st.columns(5)
        
        with exp_col1:
            monthly_rent = st.number_input("Monthly Rent", min_value=0.0, value=10000.0, step=500.0)
        with exp_col2:
            school_fees = st.number_input("School Fees", min_value=0.0, value=0.0, step=500.0)
        with exp_col3:
            college_fees = st.number_input("College Fees", min_value=0.0, value=0.0, step=500.0)
        with exp_col4:
            travel_expenses = st.number_input("Travel Expenses", min_value=0.0, value=2000.0, step=100.0)
        with exp_col5:
            groceries_utilities = st.number_input("Groceries/Utilities", min_value=0.0, value=5000.0, step=500.0)
        
        other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0.0, value=1000.0, step=100.0)

        # Submit button
        submitted = st.form_submit_button("🔮 Predict EMI Eligibility")

    # --- Prediction Logic ---
    if submitted:
        # 1. Collect all inputs into a dictionary
        input_data = {
            'age': age, 'gender': gender, 'marital_status': marital_status, 'education': education,
            'monthly_salary': monthly_salary, 'employment_type': employment_type,
            'years_of_employment': years_of_employment, 'company_type': company_type,
            'house_type': house_type, 'monthly_rent': monthly_rent, 'family_size': family_size,
            'dependents': dependents, 'school_fees': school_fees, 'college_fees': college_fees,
            'travel_expenses': travel_expenses, 'groceries_utilities': groceries_utilities,
            'other_monthly_expenses': other_monthly_expenses, 'existing_loans': existing_loans,
            'current_emi_amount': current_emi_amount, 'credit_score': credit_score,
            'bank_balance': bank_balance, 'emergency_fund': emergency_fund,
            'emi_scenario': emi_scenario, 'requested_amount': requested_amount,
            'requested_tenure': requested_tenure
        }
        
        # 2. Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        with st.spinner("Processing and predicting..."):
            try:
                # 3. Preprocess the data for prediction
                processed_df = preprocess_for_prediction(input_df, scaler, le, feature_names)
                
                # 4. Make predictions
                pred_class_encoded = model_class.predict(processed_df)[0]
                pred_reg_amount = model_reg.predict(processed_df)[0]
                
                # 5. Decode/format results
                pred_class_label = le.inverse_transform([pred_class_encoded])[0]
                
                # Ensure regression prediction isn't negative
                pred_reg_amount = max(0, pred_reg_amount)

                # --- Display Results ---
                st.header("Prediction Results")
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.subheader("EMI Eligibility Assessment")
                    
                    if pred_class_label == 'Eligible':
                        st.success(f"**Status: {pred_class_label}**")
                        st.markdown("This applicant has a low risk profile and is eligible for the EMI.")
                    elif pred_class_label == 'High_Risk':
                        st.warning(f"**Status: {pred_class_label}**")
                        st.markdown("This applicant is a marginal case. Consider approval with higher interest rates or lower amount.")
                    else: # 'Not_Eligible'
                        st.error(f"**Status: {pred_class_label}**")
                        st.markdown("This applicant has a high risk profile and is not recommended for this loan.")
                
                with res_col2:
                    st.subheader("Affordability Assessment")
                    st.metric(
                        label="Maximum Recommended Monthly EMI",
                        value=f"₹ {pred_reg_amount:,.2f}"
                    )
                    st.info("This is the maximum EMI amount the applicant can safely afford based on their financial profile.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e)

else:
    st.error("🚫 Application is not ready.")
    st.info("Please go to the '🚀 Model Training' page and run the pipeline. Once complete, this page will be activated.")
