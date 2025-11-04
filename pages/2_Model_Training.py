import streamlit as st
import utils  # Import our utility module
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
import time

st.set_page_config(
    page_title="Model Training",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Model Training & MLflow Dashboard")
st.markdown("""
This page runs the complete end-to-end ML pipeline.
1.  **Loads & Cleans Data** (from `utils.py`)
2.  **Engineers Features** (from `utils.py`)
3.  **Splits Data** (60% Train, 20% Validation, 20% Test)
4.  **Trains 6 Models** (3 Classification, 3 Regression)
5.  **Logs all models, parameters, and metrics to MLflow**
6.  **Saves the best models** (`.joblib`) for the prediction page.
""")

st.info("""
**To View Results:**
1.  Open your terminal.
2.  Navigate to this project's folder.
3.  Run: `mlflow ui`
4.  Open `http://127.0.0.1:5000` in your browser.
""")

# --- Main Training Pipeline ---
if st.button("🚀 Start Full Training Pipeline", type="primary", use_container_width=True):
    
    st.header("Step 1: Data Loading & Cleaning", divider="gray")
    with st.spinner("Loading 400,000+ records..."):
        df = utils.load_data("emi_prediction_dataset.csv")
    if df is None:
        st.error("Failed to load data. Halting pipeline.")
    else:
        st.success("Data loaded successfully.")
        
        with st.spinner("Cleaning data and imputing missing values..."):
            df_clean = utils.clean_data(df)
        st.success("Data cleaning complete.")
        
        st.header("Step 2: Feature Engineering", divider="gray")
        with st.spinner("Creating financial ratios and encoding variables..."):
            df_processed, le, all_feature_names = utils.engineer_features(df_clean)
        st.success("Feature engineering complete.")
        
        st.header("Step 3: Data Splitting & Scaling", divider="gray")
        with st.spinner("Splitting data (60/20/20) and applying scaling..."):
            # This variable is named data_splits
            data_splits, scaler = utils.split_and_scale_data(df_processed, all_feature_names, le)
        st.success("Data splitting and scaling complete.")

        # Save artifacts for prediction page
        joblib.dump(le, 'label_encoder.joblib')
        joblib.dump(scaler, 'standard_scaler.joblib')
        with open('feature_names.txt', 'w') as f:
            for item in all_feature_names:
                f.write(f"{item}\n")
        st.success("LabelEncoder, StandardScaler, and FeatureList saved.")

        st.header("Step 4: Model Training & MLflow Logging", divider="blue")
        
        # Set experiment
        EXPERIMENT_NAME = "EMIPredict_Al_v1"
        mlflow.set_experiment(EXPERIMENT_NAME)
        st.write(f"MLflow experiment set to: **{EXPERIMENT_NAME}**")
        
        num_classes = len(le.classes_)
        
        # --- 1. Logistic Regression ---
        st.subheader("Training: 1. LogisticRegression")
        try:
            with mlflow.start_run(run_name="LogisticRegression") as run:
                with st.spinner("Training LogisticRegression..."):
                    lr_params = {
                        "model_type": "LogisticRegression",
                        "solver": "lbfgs",
                        "max_iter": 1000,
                        "multi_class": "multinomial",
                        "random_state": 42
                    }
                    mlflow.log_params(lr_params)
                    
                    lr = LogisticRegression(**{k: v for k, v in lr_params.items() if k != 'model_type'})
                    lr.fit(data_splits["X_train_scaled"], data_splits["y_train_class"])
                    
                    # Evaluate
                    y_pred_class_lr = lr.predict(data_splits["X_val_scaled"])
                    y_prob_class_lr = lr.predict_proba(data_splits["X_val_scaled"])
                    
                    class_metrics_lr = utils.eval_classification_metrics(data_splits["y_val_class"], y_pred_class_lr, y_prob_class_lr, num_classes=num_classes)
                    
                    mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_lr.items()})
                    mlflow.sklearn.log_model(lr, "model")
                st.write(f"✅ LogisticRegression Metrics: `{class_metrics_lr}`")
        except Exception as e:
            st.error(f"Error training LogisticRegression: {e}")

        # --- 2. Linear Regression ---
        st.subheader("Training: 2. LinearRegression")
        try:
            with mlflow.start_run(run_name="LinearRegression") as run:
                with st.spinner("Training LinearRegression..."):
                    lin_reg_params = {"model_type": "LinearRegression"}
                    mlflow.log_params(lin_reg_params)

                    lin_reg = LinearRegression()
                    lin_reg.fit(data_splits["X_train_scaled"], data_splits["y_train_reg"])
                    
                    y_pred_reg_lin = lin_reg.predict(data_splits["X_val_scaled"])
                    
                    reg_metrics_lin = utils.eval_regression_metrics(data_splits["y_val_reg"], y_pred_reg_lin)
                    
                    mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_lin.items()})
                    mlflow.sklearn.log_model(lin_reg, "model")
                st.write(f"✅ LinearRegression Metrics: `{reg_metrics_lin}`")
        except Exception as e:
            st.error(f"Error training LinearRegression: {e}")

        # --- 3. Random Forest Classifier ---
        st.subheader("Training: 3. RandomForestClassifier")
        try:
            with mlflow.start_run(run_name="RandomForestClassifier") as run:
                with st.spinner("Training RandomForestClassifier... (this may take a minute)"):
                    rfc_params = {
                        "model_type": "RandomForestClassifier",
                        "n_estimators": 100,
                        "max_depth": 10,
                        "random_state": 42,
                        "n_jobs": -1
                    }
                    mlflow.log_params(rfc_params)
                    
                    rfc = RandomForestClassifier(**{k: v for k, v in rfc_params.items() if k != 'model_type'})
                    rfc.fit(data_splits["X_train_scaled"], data_splits["y_train_class"])
                    
                    y_pred_class_rfc = rfc.predict(data_splits["X_val_scaled"])
                    y_prob_class_rfc = rfc.predict_proba(data_splits["X_val_scaled"])
                    
                  
                    class_metrics_rfc = utils.eval_classification_metrics(data_splits["y_val_class"], y_pred_class_rfc, y_prob_class_rfc, num_classes=num_classes)
                    
                    mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_rfc.items()})
                    mlflow.sklearn.log_model(rfc, "model")
                st.write(f"✅ RandomForestClassifier Metrics: `{class_metrics_rfc}`")
        except Exception as e:
            st.error(f"Error training RandomForestClassifier: {e}")

        # --- 4. Random Forest Regressor ---
        st.subheader("Training: 4. RandomForestRegressor")
        try:
            with mlflow.start_run(run_name="RandomForestRegressor") as run:
                with st.spinner("Training RandomForestRegressor... (this may take a minute)"):
                    rfr_params = {
                        "model_type": "RandomForestRegressor",
                        "n_estimators": 100,
                        "max_depth": 10,
                        "random_state": 42,
                        "n_jobs": -1
                    }
                    mlflow.log_params(rfr_params)
                    
                    rfr = RandomForestRegressor(**{k: v for k, v in rfr_params.items() if k != 'model_type'})
                    rfr.fit(data_splits["X_train_scaled"], data_splits["y_train_reg"])
                    
                    y_pred_reg_rfr = rfr.predict(data_splits["X_val_scaled"])
                    
       
                    reg_metrics_rfr = utils.eval_regression_metrics(data_splits["y_val_reg"], y_pred_reg_rfr)
                    
                    mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_rfr.items()})
                    mlflow.sklearn.log_model(rfr, "model")
                st.write(f"✅ RandomForestRegressor Metrics: `{reg_metrics_rfr}`")
        except Exception as e:
            st.error(f"Error training RandomForestRegressor: {e}")

        # --- 5. XGBoost Classifier ---
        st.subheader("Training: 5. XGBoostClassifier (Best Model)")
        try:
            with mlflow.start_run(run_name="XGBoostClassifier") as run:
                with st.spinner("Training XGBoostClassifier..."):
                    xgbc_params = {
                        "model_type": "XGBoostClassifier",
                        "objective": "multi:softprob",
                        "num_class": num_classes,
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 5,
                        "random_state": 42,
                        "n_jobs": -1,
                        "early_stopping_rounds": 10
                    }
                    mlflow.log_params(xgbc_params)
                    
                    xgbc_prob = XGBClassifier(**{k: v for k, v in xgbc_params.items() if k != 'model_type'})
                    
                    xgbc_prob.fit(data_splits["X_train_scaled"], data_splits["y_train_class"],
                                  eval_set=[(data_splits["X_val_scaled"], data_splits["y_val_class"])],
                                  verbose=False)
                    
                    y_pred_class_xgb = xgbc_prob.predict(data_splits["X_val_scaled"])
                    y_prob_class_xgb = xgbc_prob.predict_proba(data_splits["X_val_scaled"])
                    
                    
                    class_metrics_xgb = utils.eval_classification_metrics(data_splits["y_val_class"], y_pred_class_xgb, y_prob_class_xgb, num_classes=num_classes)
                    
                    mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_xgb.items()})
                    mlflow.xgboost.log_model(xgbc_prob, "model")
                st.write(f"✅ XGBoostClassifier Metrics: `{class_metrics_xgb}`")
                
                # Save model for cloud deployment
                joblib.dump(xgbc_prob, 'xgb_classifier.joblib')
                st.success("XGBoostClassifier model saved as 'xgb_classifier.joblib'")
        except Exception as e:
            st.error(f"Error training XGBoostClassifier: {e}")

        # --- 6. XGBoost Regressor ---
        st.subheader("Training: 6. XGBoostRegressor (Best Model)")
        try:
            with mlflow.start_run(run_name="XGBoostRegressor") as run:
                with st.spinner("Training XGBoostRegressor..."):
                    xgbr_params = {
                        "model_type": "XGBoostRegressor",
                        "objective": "reg:squarederror",
                        "n_estimators": 100,
                        "learning_rate": 0.1,
                        "max_depth": 5,
                        "random_state": 42,
                        "n_jobs": -1,
                        "early_stopping_rounds": 10
                    }
                    mlflow.log_params(xgbr_params)
                    
                    xgbr = XGBRegressor(**{k: v for k, v in xgbr_params.items() if k != 'model_type'})
                    
                    xgbr.fit(data_splits["X_train_scaled"], data_splits["y_train_reg"],
                             eval_set=[(data_splits["X_val_scaled"], data_splits["y_val_reg"])],
                             verbose=False)
                    
                    y_pred_reg_xgbr = xgbr.predict(data_splits["X_val_scaled"])
                    
                   
                    reg_metrics_xgbr = utils.eval_regression_metrics(data_splits["y_val_reg"], y_pred_reg_xgbr)
                    
                    mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_xgbr.items()})
                    mlflow.xgboost.log_model(xgbr, "model")
                st.write(f"✅ XGBoostRegressor Metrics: `{reg_metrics_xgbr}`")
                
                # Save model for cloud deployment
                joblib.dump(xgbr, 'xgb_regressor.joblib')
                st.success("XGBoostRegressor model saved as 'xgb_regressor.joblib'")
        except Exception as e:
            st.error(f"Error training XGBoostRegressor: {e}")
            
        st.balloons()
        st.header("🎉 --- Full Training Pipeline Complete! --- 🎉", divider="green")
        st.success("All models trained and artifacts saved. Check your MLflow UI!")

