import pandas as pd
import numpy as np
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

# --- Step 1: Data Loading ---
@st.cache_data
def load_data(filepath='emi_prediction_dataset.csv'):
    """Loads the dataset. Using low_memory=False to help with the DtypeWarning."""
    try:
        df = pd.read_csv(filepath, low_memory=False)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {filepath} was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Step 1 (Cont.): Data Cleaning ---
def clean_data(df):
    """
    Applies the full data cleaning pipeline:
    1. Coerces object columns to numeric.
    2. Imputes all missing values.
    """
    df_clean = df.copy()
    
    # 1. Clean Problematic Object Columns
    df_clean['age'] = pd.to_numeric(df_clean['age'], errors='coerce')
    df_clean['monthly_salary'] = pd.to_numeric(df_clean['monthly_salary'], errors='coerce')
    df_clean['bank_balance'] = pd.to_numeric(df_clean['bank_balance'], errors='coerce')

    # 2. Impute Missing Values
    missing_vals = df_clean.isnull().sum()
    cols_with_na = missing_vals[missing_vals > 0].index.tolist()

    if cols_with_na:
        numeric_cols_with_na = [col for col in cols_with_na if df_clean[col].dtype in ['float64', 'int64']]
        categorical_cols_with_na = [col for col in cols_with_na if df_clean[col].dtype == 'object']
        
        if numeric_cols_with_na:
            num_imputer = SimpleImputer(strategy='median')
            df_clean[numeric_cols_with_na] = num_imputer.fit_transform(df_clean[numeric_cols_with_na])
        
        if categorical_cols_with_na:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols_with_na] = cat_imputer.fit_transform(df_clean[categorical_cols_with_na])
            
    return df_clean

# --- Step 3: Feature Engineering ---
def engineer_features(df):
    """
    Applies the full feature engineering pipeline:
    1. Creates new financial ratios.
    2. Cleans 'gender' column.
    3. Encodes binary, ordinal, and nominal columns.
    4. Encodes the target variable.
    """
    df_eng = df.copy()
    epsilon = 1e-6 # For safe division

    # 1. Create New Financial Ratios
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
    df_eng['total_monthly_expenses'] = df_eng[expense_cols].sum(axis=1)
    df_eng['debt_to_income_ratio'] = (df_eng['current_emi_amount'] + df_eng['total_monthly_expenses']) / (df_eng['monthly_salary'] + epsilon)
    df_eng['savings_to_income_ratio'] = (df_eng['bank_balance'] + df_eng['emergency_fund']) / (df_eng['monthly_salary'] + epsilon)
    df_eng['loan_to_income_ratio'] = df_eng['requested_amount'] / (df_eng['monthly_salary'] + epsilon)
    df_eng['dependents_ratio'] = df_eng['dependents'] / (df_eng['family_size'] + epsilon)
    
    # 2. Clean 'gender' column
    df_eng['gender'] = df_eng['gender'].astype(str).str.lower()
    gender_map = {'female': 'Female', 'f': 'Female', 'male': 'Male', 'm': 'Male'}
    df_eng['gender'] = df_eng['gender'].replace(gender_map)
    
    # 3. Encode Categorical Features
    df_eng['existing_loans'] = df_eng['existing_loans'].map({'Yes': 1, 'No': 0})
    
    education_map = {'High School': 1, 'Graduate': 2, 'Post Graduate': 3, 'Professional': 4}
    df_eng['education_encoded'] = df_eng['education'].map(lambda x: education_map.get(x, 1))
    
    nominal_cols = ['gender', 'marital_status', 'employment_type', 
                    'company_type', 'house_type', 'emi_scenario']
    df_encoded = pd.get_dummies(df_eng, columns=nominal_cols, drop_first=True, dtype=int)
    
    # 4. Encode Target Variable
    le = LabelEncoder()
    df_encoded['emi_eligibility_encoded'] = le.fit_transform(df_encoded['emi_eligibility'])
    
    return df_encoded, le

# --- Step 4: Helper Functions for Metrics ---
def eval_regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

def eval_classification_metrics(y_true, y_pred, y_prob, num_classes):
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    if num_classes > 2:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    else:
        roc_auc = roc_auc_score(y_true, y_prob[:, 1], average='macro')
    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc
    }

# --- Prediction Pipeline Function ---
def preprocess_for_prediction(input_data, scaler, le, feature_names):
    """
    Takes a single row of raw input data (as a DataFrame) and
    prepares it for prediction.
    """
    
    # Make a copy
    df_predict = input_data.copy()
    
    # 1. Apply cleaning (handles coerce to NaN)
    # Note: We don't impute, we assume prediction inputs are complete.
    # A more robust app would handle missing prediction inputs.
    df_predict['age'] = pd.to_numeric(df_predict['age'], errors='coerce')
    df_predict['monthly_salary'] = pd.to_numeric(df_predict['monthly_salary'], errors='coerce')
    df_predict['bank_balance'] = pd.to_numeric(df_predict['bank_balance'], errors='coerce')

    # Fill any NaNs created during coercion with 0 or a sensible default
    # This is a simple strategy; median imputation from training would be better
    df_predict = df_predict.fillna(0)

    # 2. Apply feature engineering
    epsilon = 1e-6
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
    df_predict['total_monthly_expenses'] = df_predict[expense_cols].sum(axis=1)
    df_predict['debt_to_income_ratio'] = (df_predict['current_emi_amount'] + df_predict['total_monthly_expenses']) / (df_predict['monthly_salary'] + epsilon)
    df_predict['savings_to_income_ratio'] = (df_predict['bank_balance'] + df_predict['emergency_fund']) / (df_predict['monthly_salary'] + epsilon)
    df_predict['loan_to_income_ratio'] = df_predict['requested_amount'] / (df_predict['monthly_salary'] + epsilon)
    df_predict['dependents_ratio'] = df_predict['dependents'] / (df_predict['family_size'] + epsilon)
    
    # 3. Apply encoding
    df_predict['gender'] = df_predict['gender'].astype(str).str.lower()
    gender_map = {'female': 'Female', 'f': 'Female', 'male': 'Male', 'm': 'Male'}
    df_predict['gender'] = df_predict['gender'].replace(gender_map)
    
    df_predict['existing_loans'] = df_predict['existing_loans'].map({'Yes': 1, 'No': 0})
    
    education_map = {'High School': 1, 'Graduate': 2, 'Post Graduate': 3, 'Professional': 4}
    df_predict['education_encoded'] = df_predict['education'].map(lambda x: education_map.get(x, 1))
    
    nominal_cols = ['gender', 'marital_status', 'employment_type', 
                    'company_type', 'house_type', 'emi_scenario']
    df_processed = pd.get_dummies(df_predict, columns=nominal_cols, drop_first=True, dtype=int)

    # 4. Align Columns
    # Get all columns from the processed data
    processed_cols = df_processed.columns.tolist()
    
    # Create a final DataFrame with all features, initialized to 0
    final_df = pd.DataFrame(columns=feature_names)
    final_df.loc[0] = 0
    
    # Fill in the values we have
    for col in processed_cols:
        if col in feature_names:
            final_df[col] = df_processed[col].values

    # 5. Apply Scaling
    numerical_cols_to_scale = [col for col in scaler.feature_names_in_ if col in final_df.columns]
    final_df[numerical_cols_to_scale] = scaler.transform(final_df[numerical_cols_to_scale])
    
    return final_df