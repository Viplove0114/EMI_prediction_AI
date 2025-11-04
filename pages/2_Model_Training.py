import streamlit as st
import utils  # Import our utility module
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(
    page_title="Model Training",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 Model Training & MLflow Pipeline")
st.markdown("""
This page allows you to run the complete end-to-end model training pipeline.
When you click the button below, the app will:
1.  Load the `emi_prediction_dataset.csv` file.
2.  Clean and preprocess all 400,000+ records.
3.  Engineer new financial features.
4.  Split the data into train, validation, and test sets.
5.  Train **6 different models** (3 for classification, 3 for regression).
6.  Log all model parameters, metrics, and artifacts to **MLflow**.
7.  Save all artifacts (`Scaler`, `Encoder`, `feature_list`) for the prediction page.
8.  **Save the best models (`XGBoost`) as `.joblib` files for Streamlit Cloud deployment.**
""")

# --- MLflow Setup ---
EXPERIMENT_NAME = "EMIPredict_Al_v1"
mlflow.set_experiment(EXPERIMENT_NAME)

st.info(f"MLflow Experiment set to: **`{EXPERIMENT_NAME}`**")
st.markdown("""
**To view your results, run this command in your terminal (for local analysis):**
```bash
mlflow ui
```
Then open `http://127.0.0.1:5000` in your browser.
""")

# --- Training Execution ---
if st.button("🚀 Start Full Training Pipeline", type="primary"):
    
    progress_bar = st.progress(0, text="Pipeline Started...")
    st.session_state['pipeline_run'] = True
    
    try:
        # --- Step 1, 2, 3: Load, Clean, and Engineer ---
        progress_bar.progress(5, text="Loading data...")
        df = utils.load_data("emi_prediction_dataset.csv") # Explicitly pass filename
        if df is None:
            raise Exception("Data loading failed. Check file.")
        
        progress_bar.progress(10, text="Cleaning data...")
        df_clean = utils.clean_data(df)
        
        progress_bar.progress(20, text="Engineering features...")
        df_processed, le, all_feature_names = utils.engineer_features(df_clean)
        
        # --- Step 4: Split and Scale ---
        progress_bar.progress(30, text="Splitting and scaling data...")
        data_splits, scaler = utils.split_and_scale_data(df_processed, all_feature_names, le)
        
        # Save artifacts for prediction page
        joblib.dump(scaler, 'standard_scaler.joblib')
        joblib.dump(le, 'label_encoder.joblib')
        with open('feature_names.txt', 'w') as f:
            for item in all_feature_names:
                f.write(f"{item}\n")
        st.success("Saved Scaler, Encoder, and Feature List.")

        # --- Step 5: Model Training ---
        
        # --- Baseline Models ---
        progress_bar.progress(40, text="Training LogisticRegression...")
        with st.spinner("Training LogisticRegression..."):
            with mlflow.start_run(run_name="LogisticRegression") as run:
                lr_params = {
                    "model_type": "LogisticRegression", "solver": "lbfgs",
                    "max_iter": 1000, "multi_class": "multinomial", "random_state": 42
                }
                mlflow.log_params(lr_params)
                
                lr = LogisticRegression(**{k: v for k, v in lr_params.items() if k != 'model_type'})
                lr.fit(data_splits["X_train_scaled"], data_splits["y_train_class"])
                
                y_pred_class_lr = lr.predict(data_splits["X_val_scaled"])
                y_prob_class_lr = lr.predict_proba(data_splits["X_val_scaled"])
                
                class_metrics_lr = utils.eval_classification_metrics(
                    data_splits["y_val_class"], y_pred_class_lr, y_prob_class_lr, num_classes=len(le.classes_)
                )
                mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_lr.items()})
                mlflow.sklearn.log_model(lr, "model", registered_model_name="LogisticRegression")
                st.write(f"✅ LogisticRegression Metrics: `{class_metrics_lr}`")

        progress_bar.progress(50, text="Training LinearRegression...")
        with st.spinner("Training LinearRegression..."):
            with mlflow.start_run(run_name="LinearRegression") as run:
                lin_reg_params = {"model_type": "LinearRegression"}
                mlflow.log_params(lin_reg_params)

                lin_reg = LinearRegression()
                lin_reg.fit(data_splits["X_train_scaled"], data_splits["y_train_reg"])
                
                y_pred_reg_lin = lin_reg.predict(data_splits["X_val_scaled"])
                reg_metrics_lin = utils.eval_regression_metrics(data_splits["y_val_reg"], y_pred_reg_lin)
                
                mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_lin.items()})
                mlflow.sklearn.log_model(lin_reg, "model", registered_model_name="LinearRegression")
                st.write(f"✅ LinearRegression Metrics: `{reg_metrics_lin}`")

        # --- Random Forest Models ---
        progress_bar.progress(60, text="Training RandomForestClassifier...")
        with st.spinner("Training RandomForestClassifier..."):
            with mlflow.start_run(run_name="RandomForestClassifier") as run:
                rfc_params = {
                    "model_type": "RandomForestClassifier", "n_estimators": 100,
                    "max_depth": 10, "random_state": 42, "n_jobs": -1
                }
                mlflow.log_params(rfc_params)
                
                rfc = RandomForestClassifier(**{k: v for k, v in rfc_params.items() if k != 'model_type'})
                rfc.fit(data_splits["X_train_scaled"], data_splits["y_train_class"])
                
                y_pred_class_rfc = rfc.predict(data_splits["X_val_scaled"])
                y_prob_class_rfc = rfc.predict_proba(data_splits["X_val_scaled"])
                
                class_metrics_rfc = utils.eval_classification_metrics(
                    data_splits["y_val_class"], y_pred_class_rfc, y_prob_class_rfc, num_classes=len(le.classes_)
                )
                mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_rfc.items()})
                mlflow.sklearn.log_model(rfc, "model", registered_model_name="RandomForestClassifier")
                st.write(f"✅ RandomForestClassifier Metrics: `{class_metrics_rfc}`")

        progress_bar.progress(70, text="Training RandomForestRegressor...")
        with st.spinner("Training RandomForestRegressor..."):
            with mlflow.start_run(run_name="RandomForestRegressor") as run:
                rfr_params = {
                    "model_type": "RandomForestRegressor", "n_estimators": 100,
                    "max_depth": 10, "random_state": 42, "n_jobs": -1
                }
                mlflow.log_params(rfr_params)
                
                rfr = RandomForestRegressor(**{k: v for k, v in rfr_params.items() if k != 'model_type'})
                rfr.fit(data_splits["X_train_scaled"], data_splits["y_train_reg"])
                
                y_pred_reg_rfr = rfr.predict(data_splits["X_val_scaled"])
                reg_metrics_rfr = utils.eval_regression_metrics(data_locals["y_val_reg"], y_pred_reg_rfr)
                
                mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_rfr.items()})
                mlflow.sklearn.log_model(rfr, "model", registered_model_name="RandomForestRegressor")
                st.write(f"✅ RandomForestRegressor Metrics: `{reg_metrics_rfr}`")

        # --- XGBoost Models ---
        progress_bar.progress(85, text="Training XGBoostClassifier...")
        with st.spinner("Training XGBoostClassifier... (This may take a minute)"):
            with mlflow.start_run(run_name="XGBoostClassifier") as run:
                xgbc_params = {
                    "model_type": "XGBoostClassifier", "objective": "multi:softprob",
                    "num_class": len(le.classes_), "n_estimators": 100,
                    "learning_rate": 0.1, "max_depth": 5, "random_state": 42,
                    "n_jobs": -1, "early_stopping_rounds": 10
                }
                mlflow.log_params(xgbc_params)
                
                xgbc_prob = XGBClassifier(**{k: v for k, v in xgbc_params.items() if k != 'model_type'})
                xgbc_prob.fit(
                    data_splits["X_train_scaled"], data_splits["y_train_class"],
                    eval_set=[(data_splits["X_val_scaled"], data_splits["y_val_class"])],
                    verbose=False
                )
                
                y_pred_class_xgb = xgbc_prob.predict(data_splits["X_val_scaled"])
                y_prob_class_xgb = xgbc_prob.predict_proba(data_splits["X_val_scaled"])
                
                class_metrics_xgb = utils.eval_classification_metrics(
                    data_splits["y_val_class"], y_pred_class_xgb, y_prob_class_xgb, num_classes=len(le.classes_)
                )
                mlflow.log_metrics({f"val_{k}": v for k, v in class_metrics_xgb.items()})
                mlflow.xgboost.log_model(xgbc_prob, "model", registered_model_name="XGBoostClassifier")
                st.write(f"✅ XGBoostClassifier Metrics: `{class_metrics_xgb}`")
                
                # --- SAVE MODEL FOR DEPLOYMENT ---
                joblib.dump(xgbc_prob, 'xgb_classifier.joblib')
                st.success("XGBoostClassifier model saved as 'xgb_classifier.joblib'")

        progress_bar.progress(95, text="Training XGBoostRegressor...")
        with st.spinner("Training XGBoostRegressor... (This may take a minute)"):
            with mlflow.start_run(run_name="XGBoostRegressor") as run:
                xgbr_params = {
                    "model_type": "XGBoostRegressor", "objective": "reg:squarederror",
                    "n_estimators": 100, "learning_rate": 0.1, "max_depth": 5,
                    "random_state": 42, "n_jobs": -1, "early_stopping_rounds": 10
                }
                mlflow.log_params(xgbr_params)
                
                xgbr = XGBRegressor(**{k: v for k, v in xgbr_params.items() if k != 'model_type'})
                xgbr.fit(
                    data_splits["X_train_scaled"], data_splits["y_train_reg"],
                    eval_set=[(data_splits["X_val_scaled"], data_splits["y_val_reg"])],
                    verbose=False
                )
                
                y_pred_reg_xgbr = xgbr.predict(data_splits["X_val_scaled"])
                reg_metrics_xgbr = utils.eval_regression_metrics(data_splits["y_val_reg"], y_pred_reg_xgbr)
                
                mlflow.log_metrics({f"val_{k}": v for k, v in reg_metrics_xgbr.items()})
                mlflow.xgboost.log_model(xgbr, "model", registered_model_name="XGBoostRegressor")
                st.write(f"✅ XGBoostRegressor Metrics: `{reg_metrics_xgbr}`")

                # --- SAVE MODEL FOR DEPLOYMENT ---
                joblib.dump(xgbr, 'xgb_regressor.joblib')
                st.success("XGBoostRegressor model saved as 'xgb_regressor.joblib'")
        
        progress_bar.progress(100, text="Training Pipeline Complete!")
        st.balloons()
        st.success("All models have been trained, logged to MLflow, and saved for deployment!")

    except FileNotFoundError as e:
        st.error(f"FATAL ERROR: {e}. Make sure `emi_prediction_dataset.csv` is in the root directory.")
    except Exception as e:
        st.error(f"An error occurred during the training pipeline: {e}")
        st.exception(e)

