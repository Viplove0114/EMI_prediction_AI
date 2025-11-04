import streamlit as st
import utils
import joblib
import pandas as pd
import numpy as np
import numpy_financial as npf  
import time

st.set_page_config(
    page_title="Live EMI Prediction",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Live Financial Risk Assessment")
st.markdown("""
Enter a customer's financial details below to get an instant risk assessment.
This predictor uses the best-performing models (XGBoost) trained on over 400,000 records.
""")

# --- 1. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """
    Loads all necessary artifacts from disk.
    Caches resources for performance.
    """
    try:
        classifier = joblib.load('xgb_classifier.joblib')
        regressor = joblib.load('xgb_regressor.joblib')
        le = joblib.load('label_encoder.joblib')
        scaler = joblib.load('standard_scaler.joblib')
        with open('feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f]
            
        return {
            "classifier": classifier,
            "regressor": regressor,
            "le": le,
            "scaler": scaler,
            "feature_names": feature_names
        }
    except FileNotFoundError:
        st.error("Models not found. Please run the 'Model Training' page first!")
        return None

artifacts = load_artifacts()

if artifacts:
    st.success("Prediction models and artifacts loaded successfully!")

    # --- 2. User Input Form ---
    st.header("Enter Customer Details", divider="blue")

    with st.form("prediction_form"):
        
        # --- Column 1: Personal Demographics ---
        st.subheader("Personal Demographics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
        with col2:
            gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])
        with col3:
            marital_status = st.selectbox("Marital Status", options=['Married', 'Single'])
        with col4:
            education = st.selectbox("Education", options=['Graduate', 'Post Graduate', 'Professional', 'High School'])
        
        # --- Column 2: Employment and Income ---
        st.subheader("Employment & Income")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            monthly_salary = st.number_input("Monthly Salary (INR)", min_value=10000, max_value=1000000, value=50000, step=1000)
        with col6:
            employment_type = st.selectbox("Employment Type", options=['Private', 'Government', 'Self-employed'])
        with col7:
            years_of_employment = st.number_input("Years of Employment", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
        with col8:
            company_type = st.selectbox("Company Type", options=['MNC', 'Mid-size', 'Small', 'Startup'])

        # --- Column 3: Housing and Family ---
        st.subheader("Housing & Family")
        col9, col10, col11, col12 = st.columns(4)
        with col9:
            house_type = st.selectbox("House Type", options=['Rented', 'Own', 'Family'])
        with col10:
            monthly_rent = st.number_input("Monthly Rent (if any)", min_value=0, max_value=100000, value=10000, step=500)
        with col11:
            family_size = st.number_input("Family Size", min_value=1, max_value=10, value=2, step=1)
        with col12:
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1, step=1)
            
        # --- Column 4: Monthly Financial Obligations ---
        st.subheader("Monthly Financial Obligations")
        col13, col14, col15, col16, col17 = st.columns(5)
        with col13:
            school_fees = st.number_input("School Fees", min_value=0, max_value=50000, value=0, step=500)
        with col14:
            college_fees = st.number_input("College Fees", min_value=0, max_value=50000, value=0, step=500)
        with col15:
            travel_expenses = st.number_input("Travel Expenses", min_value=0, max_value=20000, value=2000, step=500)
        with col16:
            groceries_utilities = st.number_input("Groceries/Utilities", min_value=0, max_value=30000, value=5000, step=500)
        with col17:
            other_monthly_expenses = st.number_input("Other Expenses", min_value=0, max_value=50000, value=1000, step=500)

        # --- Column 5: Financial Status and Credit History ---
        st.subheader("Financial Status & Credit History")
        col18, col19, col20, col21, col22 = st.columns(5)
        with col18:
            existing_loans = st.selectbox("Existing Loans?", options=['Yes', 'No'])
        with col19:
            current_emi_amount = st.number_input("Current EMI Amount", min_value=0, max_value=100000, value=0, step=500)
        with col20:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=750, step=10)
        with col21:
            bank_balance = st.number_input("Bank Balance", min_value=0, max_value=5000000, value=50000, step=1000)
        with col22:
            emergency_fund = st.number_input("Emergency Fund", min_value=0, max_value=5000000, value=20000, step=1000)

        # --- Column 6: Loan Application Details ---
        st.subheader("Loan Application Details")
        col23, col24, col25 = st.columns(3)
        with col23:
            emi_scenario = st.selectbox("EMI Scenario", options=['E-commerce Shopping EMI', 'Home Appliances EMI', 'Vehicle EMI', 'Personal Loan EMI', 'Education EMI'])
        with col24:
            requested_amount = st.number_input("Requested Amount", min_value=10000, max_value=2000000, value=100000, step=1000)
        with col25:
            requested_tenure = st.number_input("Requested Tenure (Months)", min_value=3, max_value=84, value=12, step=1)
            
        # --- Submission Button ---
        st.divider()
        submit_button = st.form_submit_button("Assess Financial Risk", type="primary", use_container_width=True)


    # --- 3. Prediction Logic ---
    if submit_button and artifacts:
        with st.spinner("Processing customer data and running AI models..."):
            
            # --- a. Create Input DataFrame ---
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
            input_df = pd.DataFrame([input_data])
            
            # --- b. Apply Feature Engineering ---
            # (Must mirror the 'engineer_features' function in utils.py)
            
            # 1. Create financial ratios
            epsilon = 1e-6
            expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
            input_df['total_monthly_expenses'] = input_df[expense_cols].sum(axis=1)
            
            input_df['debt_to_income_ratio'] = (input_df['current_emi_amount'] + input_df['total_monthly_expenses']) / (input_df['monthly_salary'] + epsilon)
            input_df['savings_to_income_ratio'] = (input_df['bank_balance'] + input_df['emergency_fund']) / (input_df['monthly_salary'] + epsilon)
            input_df['expense_to_income_ratio'] = input_df['total_monthly_expenses'] / (input_df['monthly_salary'] + epsilon)
            input_df['emi_to_income_ratio'] = input_df['current_emi_amount'] / (input_df['monthly_salary'] + epsilon)
            input_df['affordability_ratio'] = (input_df['monthly_salary'] - input_df['total_monthly_expenses'] - input_df['current_emi_amount']) / (input_df['monthly_salary'] + epsilon)
            
            input_df.replace([np.inf, -np.inf], 0, inplace=True)

            # 2. Encode categorical columns
            categorical_cols = [
                'gender', 'marital_status', 'education', 'employment_type',
                'company_type', 'house_type', 'existing_loans', 'emi_scenario'
            ]
            input_df_encoded = pd.get_dummies(input_df, columns=categorical_cols, dtype=bool)

            # --- c. Align Columns with Training Data ---
            # (Ensures all columns are present and in the correct order)
            
            # Get the list of feature names from artifacts
            model_feature_names = artifacts["feature_names"]
            
  
            # Reindex the encoded input to match the model's features.
            # This adds missing one-hot columns (e.g., 'gender_Female') and fills them with 0.
            # This is much safer than combine_first and preserves dtypes (int, float, bool).
            aligned_input_df = input_df_encoded.reindex(columns=model_feature_names, fill_value=0)
            
            # Ensure the column order is identical to training
            aligned_input_df = aligned_input_df[model_feature_names]

            # --- d. Scale Numerical Features ---
            # Identify numerical columns to scale (must match training)
            scaler = artifacts["scaler"]
            
      
            numerical_cols_to_scale = scaler.feature_names_in_
            
            # Apply the fitted scaler
            aligned_input_df[numerical_cols_to_scale] = scaler.transform(aligned_input_df[numerical_cols_to_scale])
            
            # --- e.Make Predictions ---
            try:
                # Classification Prediction
                pred_class_encoded = artifacts["classifier"].predict(aligned_input_df)
                pred_class_proba = artifacts["classifier"].predict_proba(aligned_input_df)
                
                # Decode the prediction
                le = artifacts["le"]
                pred_class_label = le.inverse_transform(pred_class_encoded)[0]
                
                # Regression Prediction
                pred_max_emi = artifacts["regressor"].predict(aligned_input_df)[0]
                
                # Get probabilities for each class
                class_probabilities = dict(zip(le.classes_, pred_class_proba[0]))
                
                time.sleep(1) # Simulate complex calculation
                
                # --- f. Display Results ---
                st.header("Financial Risk Assessment Results", divider="green")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.subheader("EMI Eligibility Prediction")
                    if pred_class_label == "Eligible":
                        st.success(f"**Status: {pred_class_label}**")
                        st.markdown("This customer has a low-risk profile and is eligible for the loan.")
                    elif pred_class_label == "High_Risk":
                        st.warning(f"**Status: {pred_class_label}**")
                        st.markdown("This customer is a marginal case. Proceed with caution, recommend higher interest rates, or request further documentation.")
                    else: # Not_Eligible
                        st.error(f"**Status: {pred_class_label}**")
                        st.markdown("This customer has a high-risk profile and is not recommended for the loan.")

                    st.subheader("Prediction Confidence")
                    st.dataframe(pd.DataFrame.from_dict(class_probabilities, orient='index', columns=['Confidence'])
                                 .style.format("{:.1%}"))
                
                with result_col2:
                    st.subheader("Affordability Analysis")
                    st.metric(
                        label="Predicted Max Affordable EMI",
                        value=f"₹ {pred_max_emi:,.2f} / month"
                    )
                    st.info(f"The model estimates this customer can safely afford a maximum monthly EMI of approximately **₹{pred_max_emi:,.2f}**.")
                    
                    # Compare with requested
                    requested_emi = npf.pmt(  # <-- 2. CHANGED np.pmt TO npf.pmt
                        rate=(5/100)/12, # Assuming a 5% APR for calculation
                        nper=requested_tenure,
                        pv=-requested_amount
                    )
                    st.metric(
                        label="Requested Loan's EMI (Approx.)",
                        value=f"₹ {requested_emi:,.2f} / month"
                    )
                    if requested_emi > pred_max_emi:
                        st.warning("The requested loan's EMI is **higher** than the customer's predicted maximum affordability.")
                    else:
                        st.success("The requested loan's EMI is **within** the customer's predicted affordability.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.error("Please ensure all input values are correct. Check for any non-numeric characters in numerical fields.")

elif not artifacts:
    st.warning("Prediction models are not loaded. Please go to the '🚀 Model Training' page and run the training pipeline first.")



