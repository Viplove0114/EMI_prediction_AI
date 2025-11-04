import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

# --- 1. Data Loading ---
@st.cache_data
def load_data(filepath):
    """
    Loads data from a CSV, with error handling.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)
        return df
    except FileNotFoundError:
        st.error(f"Error: The file {filepath} was not found.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- 2. Data Cleaning ---
def clean_data(df):
    """
    Cleans object columns and imputes all missing values.
    """
    # a) Clean object columns to numeric (this creates new NaNs)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['monthly_salary'] = pd.to_numeric(df['monthly_salary'], errors='coerce')
    df['bank_balance'] = pd.to_numeric(df['bank_balance'], errors='coerce')

    # b) Identify ALL columns that now have NaNs
    missing_vals = df.isnull().sum()
    cols_with_na = missing_vals[missing_vals > 0].index.tolist()

    if not cols_with_na:
        return df

    # c) Separate them into numeric and categorical lists
    numeric_cols_with_na = [col for col in cols_with_na if df[col].dtype in ['float64', 'int64']]
    categorical_cols_with_na = [col for col in cols_with_na if df[col].dtype == 'object']

    # d) Apply Imputation
    if numeric_cols_with_na:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols_with_na] = num_imputer.fit_transform(df[numeric_cols_with_na])
    
    if categorical_cols_with_na:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols_with_na] = cat_imputer.fit_transform(df[categorical_cols_with_na])

    return df

# --- 3. Feature Engineering ---
def engineer_features(df):
    """
    Creates new financial ratios, encodes variables, and returns the processed DataFrame,
    LabelEncoder, and the list of final feature names.
    """
    # --- 1. Create New Financial Ratios ---
    epsilon = 1e-6 
    expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                    'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
    df['total_monthly_expenses'] = df[expense_cols].sum(axis=1)
    
    df['debt_to_income_ratio'] = (df['current_emi_amount'] + df['total_monthly_expenses']) / (df['monthly_salary'] + epsilon)
    df['savings_to_income_ratio'] = (df['bank_balance'] + df['emergency_fund']) / (df['monthly_salary'] + epsilon)
    df['expense_to_income_ratio'] = df['total_monthly_expenses'] / (df['monthly_salary'] + epsilon)
    df['emi_to_income_ratio'] = df['current_emi_amount'] / (df['monthly_salary'] + epsilon)
    df['affordability_ratio'] = (df['monthly_salary'] - df['total_monthly_expenses'] - df['current_emi_amount']) / (df['monthly_salary'] + epsilon)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    # --- 2. Clean 'gender' column ---
    df['gender'] = df['gender'].str.lower().map({
        'male': 'Male', 'm': 'Male', 'female': 'Female', 'f': 'Female'
    }).fillna('Other')

    # --- 3. Encode Categorical Columns ---
    categorical_cols = [
        'gender', 'marital_status', 'education', 'employment_type',
        'company_type', 'house_type', 'existing_loans', 'emi_scenario'
    ]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, dtype=bool)

    # --- 4. Label Encode Target Variable ---
    le = LabelEncoder()
    df_encoded['emi_eligibility_encoded'] = le.fit_transform(df_encoded['emi_eligibility'])
    
    # --- 5. Get Final Feature List ---
    target_classification = 'emi_eligibility_encoded'
    target_regression = 'max_monthly_emi'
    
    # Define columns to drop
    cols_to_drop = [
        'emi_eligibility', 'education', # Dropping original 'education' as requested in notebook
        target_classification, target_regression
    ]
    
    # Find all one-hot encoded 'education' columns to drop them too
    education_dummies = [col for col in df_encoded.columns if col.startswith('education_')]
    
    # Combine all columns to be dropped
    features_to_drop = cols_to_drop + education_dummies
    
    # Get the final list of feature names
    all_feature_names = [col for col in df_encoded.columns if col not in features_to_drop]

    # --- THIS IS THE FIX ---
    # Now returns all 3 items
    return df_encoded, le, all_feature_names

# --- 4. Data Splitting & Scaling ---
def split_and_scale_data(df, all_feature_names, le):
    """
    Splits data into train/val/test, scales numerical features, and returns
    a dictionary of data splits and the fitted scaler.
    """
    # 1. Separate Features (X) and Targets (y)
    X = df[all_feature_names]
    y_class = df['emi_eligibility_encoded']
    y_reg = df['max_monthly_emi']
    
    # 2. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # 3. Train-Validation Split (from the 80% train, create 75% train, 25% val)
    # This results in 60% train, 20% val, 20% test overall
    X_train_final, X_val, y_train_class_final, y_val_class, y_train_reg_final, y_val_reg = train_test_split(
        X_train, y_train_class, y_train_reg, test_size=0.25, random_state=42, stratify=y_train_class
    )
    
    # 4. Feature Scaling
    # Identify continuous numerical columns to scale (exclude booleans)
    bool_cols = X_train_final.select_dtypes(include='bool').columns
    all_cols = X_train_final.columns
    numerical_cols_to_scale = [col for col in all_cols if col not in bool_cols]

    scaler = StandardScaler()
    
    # Fit on training data
    X_train_final[numerical_cols_to_scale] = scaler.fit_transform(X_train_final[numerical_cols_to_scale])
    
    # Transform validation and test data
    X_val[numerical_cols_to_scale] = scaler.transform(X_val[numerical_cols_to_scale])
    X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])

    # 5. Return all splits in a dictionary
    data_splits = {
        "X_train_scaled": X_train_final,
        "X_val_scaled": X_val,
        "X_test_scaled": X_test,
        "y_train_class": y_train_class_final,
        "y_val_class": y_val_class,
        "y_test_class": y_test_class,
        "y_train_reg": y_train_reg_final,
        "y_val_reg": y_val_reg,
        "y_test_reg": y_test_reg
    }
    
    return data_splits, scaler

# --- 5. Model Evaluation Metrics ---

def eval_regression_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of regression metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

def eval_classification_metrics(y_true, y_pred, y_prob, num_classes):
    """Calculates and returns a dictionary of classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate precision, recall, f1 for 'macro' average
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Calculate ROC-AUC
    if num_classes > 2:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    else:
        # Fallback for binary
        roc_auc = roc_auc_score(y_true, y_prob[:, 1], average='macro')

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc
    }

