import streamlit as st
import pandas as pd
import numpy as np


import mlflow.sklearn
import mlflow.xgboost
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

# Import all our helper functions
from utils import (
    load_data, clean_data, engineer_features,
    eval_regression_metrics, eval_classification_metrics
)

st.set_page_config(page_title="Model Training", page_icon="🚀", layout="wide")

st.title("🚀 Model Training & MLflow Logging")
st.markdown("""
This page runs the complete end-to-end machine learning pipeline.
1.  **Loads** the `emi_prediction_dataset.csv` file.
2.  **Cleans** all 400,000+ records.
3.  **Engineers** new financial features.
4.  **Splits** data into training, validation, and test sets.
5.  **Saves** the `StandardScaler`, `LabelEncoder`, and `feature_names.txt` for production.
6.  **Trains** all 6 models (Logistic Regression, Linear, RF, XGBoost).
7.  **Logs** all parameters, metrics, and models to MLflow.
""")
st.warning("⚠️ This process will take several minutes. Please be patient.")

# --- MLflow Setup ---
EXPERIMENT_NAME = "EMIPredict_Al_v1"
mlflow.set_experiment(EXPERIMENT_NAME)

# --- Main Training Function ---
def run_training_pipeline():
    progress_bar = st.progress(0)
    st.session_state['training_complete'] = False
    log_area = st.empty()

    def log(message):
        log_area.text_area("Training Logs", value=message, height=300)

    try:
        log("Training started...\n")
        
        # --- Step 1 & 2: Load and Clean Data ---
        log("Step 1/7: Loading and Cleaning Data...")
        df = load_data()
        if df is None:
            raise FileNotFoundError("Dataset file not found.")
        df_clean = clean_data(df)
        progress_bar.progress(15)

        # --- Step 3: Feature Engineering ---
        log("Step 2/7: Engineering Features...")
        df_encoded, le = engineer_features(df_clean)
        # Save the label encoder
        joblib.dump(le, 'label_encoder.joblib')
        log("...LabelEncoder saved as 'label_encoder.joblib'")
        progress_bar.progress(30)

        # --- Step 4: Split Data ---
        log("Step 3/7: Splitting Data...")
        target_classification = 'emi_eligibility_encoded'
        target_regression = 'max_monthly_emi'
        features_to_drop = ['emi_eligibility', 'education', 
                            target_classification, target_regression]
        
        X = df_encoded.drop(columns=features_to_drop)
        y_class = df_encoded[target_classification]
        y_reg = df_encoded[target_regression]
        
        # Save feature names
        all_feature_names = X.columns.tolist()
        with open('feature_names.txt', 'w') as f:
            for item in all_feature_names:
                f.write(f"{item}\n")
        log("...Feature names saved as 'feature_names.txt'")

        y = pd.DataFrame({'class': y_class, 'reg': y_reg})
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y['class']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp['class']
        )
        
        y_train_class, y_train_reg = y_train['class'], y_train['reg']
        y_val_class, y_val_reg = y_val['class'], y_val['reg']
        y_test_class, y_test_reg = y_test['class'], y_test['reg']
        progress_bar.progress(40)

        # --- Step 5: Scale Data ---
        log("Step 4/7: Scaling Data...")
        numerical_cols_to_scale = [
            'age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 
            'family_size', 'dependents', 'school_fees', 'college_fees', 
            'travel_expenses', 'groceries_utilities', 'other_monthly_expenses', 
            'current_emi_amount', 'credit_score', 'bank_balance', 
            'emergency_fund', 'requested_amount', 'requested_tenure', 
            'education_encoded', 'total_monthly_expenses', 'debt_to_income_ratio', 
            'savings_to_income_ratio', 'loan_to_income_ratio', 'dependents_ratio',
            'existing_loans'
        ]
        numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col in X_train.columns]
        
        scaler = StandardScaler()
        
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
        X_val_scaled[numerical_cols_to_scale] = scaler.transform(X_val[numerical_cols_to_scale])
        X_test_scaled[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])
        
        # Save the scaler
        joblib.dump(scaler, 'standard_scaler.joblib')
        log("...StandardScaler saved as 'standard_scaler.joblib'")
        progress_bar.progress(50)

        # --- Step 6 & 7: Model Training ---
        log("Step 5/7: Training Baseline Models...")
        
        # --- Logistic Regression ---
        with mlflow.start_run(run_name="LogisticRegression") as run:
            lr_params = {"model_type": "LogisticRegression", "solver": "lbfgs", "max_iter": 1000, "multi_class": "multinomial", "random_state": 42}
            mlflow.log_params(lr_params)
            
            lr = LogisticRegression(**{k: v for k, v in lr_params.items() if k != 'model_type'})
            lr.fit(X_train_scaled, y_train_class)
            
            y_pred_class_lr = lr.predict(X_val_scaled)
            y_prob_class_lr = lr.predict_proba(X_val_scaled)
            class_metrics_lr = eval_classification_metrics(y_val_class, y_pred_class_lr, y_prob_class_lr, num_classes=len(le.classes_))
            
            mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_lr.items()})
            mlflow.sklearn.log_model(lr, "model", registered_model_name="LogisticRegression")
        
        # --- Linear Regression ---
        with mlflow.start_run(run_name="LinearRegression") as run:
            lin_reg_params = {"model_type": "LinearRegression"}
            mlflow.log_params(lin_reg_params)
            lin_reg = LinearRegression()
            lin_reg.fit(X_train_scaled, y_train_reg)
            y_pred_reg_lin = lin_reg.predict(X_val_scaled)
            reg_metrics_lin = eval_regression_metrics(y_val_reg, y_pred_reg_lin)
            mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_lin.items()})
            mlflow.sklearn.log_model(lin_reg, "model", registered_model_name="LinearRegression")
        
        progress_bar.progress(65)
        log("Step 6/7: Training Random Forest Models...")
        
        # --- Random Forest Classifier ---
        with mlflow.start_run(run_name="RandomForestClassifier") as run:
            rfc_params = {"model_type": "RandomForestClassifier", "n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1}
            mlflow.log_params(rfc_params)
            rfc = RandomForestClassifier(**{k: v for k, v in rfc_params.items() if k != 'model_type'})
            rfc.fit(X_train_scaled, y_train_class)
            y_pred_class_rfc = rfc.predict(X_val_scaled)
            y_prob_class_rfc = rfc.predict_proba(X_val_scaled)
            class_metrics_rfc = eval_classification_metrics(y_val_class, y_pred_class_rfc, y_prob_class_rfc, num_classes=len(le.classes_))
            mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_rfc.items()})
            mlflow.sklearn.log_model(rfc, "model", registered_model_name="RandomForestClassifier")

        # --- Random Forest Regressor ---
        with mlflow.start_run(run_name="RandomForestRegressor") as run:
            rfr_params = {"model_type": "RandomForestRegressor", "n_estimators": 100, "max_depth": 10, "random_state": 42, "n_jobs": -1}
            mlflow.log_params(rfr_params)
            rfr = RandomForestRegressor(**{k: v for k, v in rfr_params.items() if k != 'model_type'})
            rfr.fit(X_train_scaled, y_train_reg)
            y_pred_reg_rfr = rfr.predict(X_val_scaled)
            reg_metrics_rfr = eval_regression_metrics(y_val_reg, y_pred_reg_rfr)
            mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_rfr.items()})
            mlflow.sklearn.log_model(rfr, "model", registered_model_name="RandomForestRegressor")
            
        progress_bar.progress(80)
        log("Step 7/7: Training XGBoost Models...")

        # --- XGBoost Classifier ---
        with mlflow.start_run(run_name="XGBoostClassifier") as run:
            xgbc_params = {"model_type": "XGBoostClassifier", "objective": "multi:softprob", "num_class": len(le.classes_), "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42, "n_jobs": -1, "early_stopping_rounds": 10}
            mlflow.log_params(xgbc_params)
            xgbc_prob = XGBClassifier(**{k: v for k, v in xgbc_params.items() if k != 'model_type'})
            xgbc_prob.fit(X_train_scaled, y_train_class, eval_set=[(X_val_scaled, y_val_class)], verbose=False)
            y_pred_class_xgb = xgbc_prob.predict(X_val_scaled)
            y_prob_class_xgb = xgbc_prob.predict_proba(X_val_scaled)
            class_metrics_xgb = eval_classification_metrics(y_val_class, y_pred_class_xgb, y_prob_class_xgb, num_classes=len(le.classes_))
            mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_xgb.items()})
            mlflow.xgboost.log_model(xgbc_prob, "model", registered_model_name="XGBoostClassifier")
            # Store the run ID for the prediction page
            st.session_state['best_classifier_run_id'] = run.info.run_id

        # --- XGBoost Regressor ---
        with mlflow.start_run(run_name="XGBoostRegressor") as run:
            xgbr_params = {"model_type": "XGBoostRegressor", "objective": "reg:squarederror", "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5, "random_state": 42, "n_jobs": -1, "early_stopping_rounds": 10}
            mlflow.log_params(xgbr_params)
            xgbr = XGBRegressor(**{k: v for k, v in xgbr_params.items() if k != 'model_type'})
            xgbr.fit(X_train_scaled, y_train_reg, eval_set=[(X_val_scaled, y_val_reg)], verbose=False)
            y_pred_reg_xgbr = xgbr.predict(X_val_scaled)
            reg_metrics_xgbr = eval_regression_metrics(y_val_reg, y_pred_reg_xgbr)
            mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_xgbr.items()})
            mlflow.xgboost.log_model(xgbr, "model", registered_model_name="XGBoostRegressor")
            # Store the run ID for the prediction page
            st.session_state['best_regressor_run_id'] = run.info.run_id

        progress_bar.progress(100)
        log("--- TRAINING PIPELINE COMPLETE ---")
        st.session_state['training_complete'] = True
        return True

    except Exception as e:
        st.error(f"An error occurred during training: {e}")
        log(f"An error occurred during training: {e}")
        return False

# --- UI Layout ---
if 'training_complete' not in st.session_state:
    st.session_state['training_complete'] = False

if st.button("🚀 Start Full Training Pipeline"):
    with st.spinner("Running pipeline... This may take 10-15 minutes."):
        success = run_training_pipeline()
        if success:
            st.success("Training Pipeline Completed Successfully!")
            st.balloons()
            st.session_state['artifacts_ready'] = True
            
if st.session_state['training_complete']:
    st.info("Training is complete. The required models and artifacts have been saved. You can now use the '🔮 Live Prediction' page.")

st.header("MLflow Dashboard")
st.markdown(f"""
All models, parameters, and metrics have been logged to the MLflow experiment: '{EXPERIMENT_NAME}',\n To view the dashboard, run this command in your terminal from the project directory:\n mlflow ui \n

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.""")
