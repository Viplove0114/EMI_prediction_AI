import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, clean_data  # Import from our utils.py

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("Analyze the distributions and relationships in the financial data.")

# Load and clean data (cached)
@st.cache_data
def load_cleaned_data():
    df = load_data()
    if df is not None:
        df_clean = clean_data(df)
        return df_clean
    return None

df_clean = load_cleaned_data()

if df_clean is not None:
    st.success("Data loaded and cleaned successfully.")

    # --- 1. Target Variable Analysis ---
    st.header("1. Target Variable Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("EMI Eligibility (Classification)")
        fig, ax = plt.subplots()
        sns.countplot(x='emi_eligibility', data=df_clean, ax=ax, palette="viridis")
        ax.set_title("Distribution of EMI Eligibility")
        st.pyplot(fig)

    with col2:
        st.subheader("Max Monthly EMI (Regression)")
        fig, ax = plt.subplots()
        sns.histplot(df_clean['max_monthly_emi'], kde=True, bins=50, ax=ax, color="blue")
        ax.set_title("Distribution of Max Monthly EMI")
        ax.set_xlabel("Max Monthly EMI")
        st.pyplot(fig)

    # --- 2. Bivariate Analysis (Features vs. Classification Target) ---
    st.header("2. Bivariate Analysis (Features vs. EMI Eligibility)")
    
    # Select key features for visualization
    cat_features = ['gender', 'marital_status', 'employment_type', 'existing_loans']
    num_features = ['credit_score', 'monthly_salary', 'bank_balance', 'age']

    st.subheader("Categorical Features vs. Eligibility")
    fig_cat, axes_cat = plt.subplots(2, 2, figsize=(16, 12))
    axes_cat = axes_cat.flatten()
    for i, feature in enumerate(cat_features):
        sns.countplot(data=df_clean, x=feature, hue='emi_eligibility', ax=axes_cat[i], palette="muted")
        axes_cat[i].set_title(f"{feature.replace('_', ' ').title()} vs. Eligibility")
        axes_cat[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_cat)

    st.subheader("Numerical Features vs. Eligibility")
    fig_num, axes_num = plt.subplots(2, 2, figsize=(16, 12))
    axes_num = axes_num.flatten()
    for i, feature in enumerate(num_features):
        sns.boxplot(data=df_clean, x='emi_eligibility', y=feature, ax=axes_num[i], palette="coolwarm")
        axes_num[i].set_title(f"{feature.replace('_', ' ').title()} vs. Eligibility")
    plt.tight_layout()
    st.pyplot(fig_num)
    
    # --- 3. Correlation Heatmap ---
    st.header("3. Numerical Feature Correlation Heatmap")
    
    # Select only numeric columns for correlation
    numeric_df = df_clean.select_dtypes(include=[np.number])
    
    fig_corr, ax_corr = plt.subplots(figsize=(18, 12))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax_corr, annot_kws={"size": 8})
    ax_corr.set_title("Correlation Heatmap of Numerical Features")
    st.pyplot(fig_corr)

else:
    st.error("Failed to load data. Please check the 'emi_prediction_dataset.csv' file.")
