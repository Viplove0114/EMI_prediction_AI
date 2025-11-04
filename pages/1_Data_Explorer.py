import streamlit as st
import utils  # Import our utility module
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Data Explorer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Exploratory Data Analysis (EDA)")
st.markdown("""
This page provides a comprehensive analysis of the 400,000+ financial records.
The data is loaded and cleaned *once* and then cached for performance.
""")

# We now pass the required 'filepath' argument to the load_data function.
@st.cache_data
def load_cleaned_data():
    """
    Loads and cleans the data by calling functions from utils.py
    """
    df = utils.load_data("emi_prediction_dataset.csv") # <-- Correctly passing filepath
    if df is not None:
        df_clean = utils.clean_data(df)
        return df_clean
    return None

# --- Load Data ---
with st.spinner("Loading and cleaning data... This may take a moment."):
    df_clean = load_cleaned_data()

if df_clean is not None:
    st.success("Data loaded and cleaned successfully!")

    # --- Show Raw Data ---
    if st.checkbox("Show Raw Data Sample", value=False):
        st.subheader("Raw Data Sample (First 100 Rows)")
        st.dataframe(df_clean.head(100))

    # --- 1. Target Variable Analysis ---
    st.header("1. Target Variable Analysis", divider="blue")
    st.markdown("""
    Analysis of our two key prediction targets:
    1.  **`emi_eligibility` (Classification):** Is the customer eligible?
    2.  **`max_monthly_emi` (Regression):** What is the max safe EMI?
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # --- Chart 1 ---
        st.subheader("Distribution of EMI Eligibility")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='emi_eligibility', data=df_clean, ax=ax1, palette="viridis", order=df_clean['emi_eligibility'].value_counts().index)
        ax1.set_title("Class Distribution of EMI Eligibility")
        ax1.set_xlabel("Eligibility Status")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)
        st.markdown("""
        **Insight:** We can see the balance of our classes. 'Eligible' is the most common,
        followed by 'Not_Eligible' and 'High_Risk'. This imbalance means 'accuracy' alone
        is not a good metric, and we must use 'precision', 'recall', and 'F1-score'.
        """)

    with col2:
        # --- Chart 2 ---
        st.subheader("Distribution of Max Monthly EMI")
        fig2, ax2 = plt.subplots()
        sns.histplot(df_clean['max_monthly_emi'], kde=True, ax=ax2, bins=50, color="blue")
        ax2.set_title("Distribution of Max Affordable EMI")
        ax2.set_xlabel("Max Monthly EMI (INR)")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
        st.markdown("""
        **Insight:** The distribution is heavily right-skewed. Most customers have a lower
        affordable EMI. This indicates that a `log-transform` might be
        beneficial for linear regression models.
        """)

    # --- 2. Key Numerical Feature Analysis ---
    st.header("2. Key Numerical Feature Analysis", divider="blue")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # --- Chart 3 ---
        st.subheader("EMI Eligibility vs. Credit Score")
        fig3, ax3 = plt.subplots()
        sns.boxplot(x='emi_eligibility', y='credit_score', data=df_clean, ax=ax3, palette="coolwarm")
        ax3.set_title("Credit Score by Eligibility Status")
        ax3.set_xlabel("Eligibility Status")
        ax3.set_ylabel("Credit Score")
        st.pyplot(fig3)
        st.markdown("""
        **Insight:** A very strong predictor. 'Eligible' customers have a significantly
        higher and more consistent credit score. 'Not_Eligible' customers show the lowest.
        """)

    with col4:
        # --- Chart 4 ---
        st.subheader("EMI Eligibility vs. Monthly Salary")
        fig4, ax4 = plt.subplots()
        # Using a log scale for y-axis due to salary skew
        sns.boxplot(x='emi_eligibility', y='monthly_salary', data=df_clean, ax=ax4, palette="coolwarm")
        ax4.set_yscale('log')
        ax4.set_title("Monthly Salary (Log Scale) by Eligibility Status")
        ax4.set_xlabel("Eligibility Status")
        ax4.set_ylabel("Monthly Salary (Log Scale)")
        st.pyplot(fig4)
        st.markdown("""
        **Insight:** 'Eligible' customers clearly have higher salaries, although there is
        significant overlap. The log scale helps visualize this relationship despite outliers.
        """)

    # --- 3. Key Categorical Feature Analysis ---
    st.header("3. Key Categorical Feature Analysis", divider="blue")

    col5, col6 = st.columns(2)

    with col5:
        # --- Chart 5 ---
        st.subheader("EMI Eligibility by Employment Type")
        fig5, ax5 = plt.subplots()
        sns.countplot(data=df_clean, x='employment_type', hue='emi_eligibility', ax=ax5, palette='Spectral')
        ax5.set_title('Eligibility Status by Employment Type')
        ax5.set_xlabel('Employment Type')
        ax5.set_ylabel('Count')
        ax5.legend(title='Eligibility')
        st.pyplot(fig5)
        st.markdown("""
        **Insight:** 'Government' employees appear to have a higher proportion of 'Eligible'
        statuses compared to other types. 'Self-employed' has a noticeable 'High_Risk' segment.
        """)

    with col6:
        # --- Chart 6 ---
        st.subheader("EMI Eligibility by Existing Loans")
        fig6, ax6 = plt.subplots()
        sns.countplot(data=df_clean, x='existing_loans', hue='emi_eligibility', ax=ax6, palette='Spectral')
        ax6.set_title('Eligibility Status by Existing Loan Status')
        ax6.set_xlabel('Has Existing Loans?')
        ax6.set_ylabel('Count')
        ax6.legend(title='Eligibility')
        st.pyplot(fig6)
        st.markdown("""
        **Insight:** As expected, customers with no existing loans are far more likely
        to be 'Eligible'. Having an existing loan significantly increases the chance
        of being 'Not_Eligible' or 'High_Risk'.
        """)

    # --- 4. Bivariate Analysis: Salary vs. Max EMI ---
    st.header("4. Bivariate Analysis", divider="blue")
    
    # --- Chart 7 ---
    st.subheader("Max EMI vs. Monthly Salary (by Eligibility)")
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x='monthly_salary', y='max_monthly_emi', data=df_clean.sample(5000), 
        hue='emi_eligibility', alpha=0.6, ax=ax7, palette="viridis"
    )
    ax7.set_title("Max EMI vs. Monthly Salary (Sample of 5000)")
    ax7.set_xlabel("Monthly Salary (INR)")
    ax7.set_ylabel("Max Monthly EMI (INR)")
    ax7.legend(title="Eligibility")
    st.pyplot(fig7)
    st.markdown("""
    **Insight:** There is a strong, positive linear relationship. As salary increases,
    the maximum safe EMI also increases. The 'Not_Eligible' cluster (in purple) is
    clearly separated at the bottom.
    """)

    # --- 5. Engineered Financial Ratio Analysis ---
    st.header("5. Analysis of Engineered Financial Ratios", divider="blue")
    st.markdown("""
    Here, we re-create the financial ratios from our Feature Engineering step to
    visualize *why* they are such powerful predictors.
    """)

    # Re-create ratios for visualization
    @st.cache_data
    def calculate_ratios(df):
        df_ratios = df.copy()
        epsilon = 1e-6
        expense_cols = ['monthly_rent', 'school_fees', 'college_fees', 
                        'travel_expenses', 'groceries_utilities', 'other_monthly_expenses']
        df_ratios['total_monthly_expenses'] = df_ratios[expense_cols].sum(axis=1)
        
        df_ratios['debt_to_income_ratio'] = (df_ratios['current_emi_amount'] + df_ratios['total_monthly_expenses']) / (df_ratios['monthly_salary'] + epsilon)
        df_ratios['savings_ratio'] = (df_ratios['monthly_salary'] - df_ratios['total_monthly_expenses'] - df_ratios['current_emi_amount']) / (df_ratios['monthly_salary'] + epsilon)
        
        # Clip ratios at reasonable percentile to remove extreme outliers for plotting
        dti_clip = df_ratios['debt_to_income_ratio'].quantile(0.99)
        sr_clip_high = df_ratios['savings_ratio'].quantile(0.99)
        sr_clip_low = df_ratios['savings_ratio'].quantile(0.01)
        
        df_ratios['debt_to_income_ratio'] = df_ratios['debt_to_income_ratio'].clip(0, dti_clip)
        df_ratios['savings_ratio'] = df_ratios['savings_ratio'].clip(sr_clip_low, sr_clip_high)
        
        return df_ratios

    df_with_ratios = calculate_ratios(df_clean)

    col7, col8 = st.columns(2)

    with col7:
        # --- Chart 8 ---
        st.subheader("Debt-to-Income (DTI) Ratio vs. Eligibility")
        fig8, ax8 = plt.subplots()
        sns.boxplot(x='emi_eligibility', y='debt_to_income_ratio', data=df_with_ratios, ax=ax8, palette="Reds")
        ax8.set_title('DTI Ratio by Eligibility Status')
        ax8.set_xlabel('Eligibility Status')
        ax8.set_ylabel('Debt-to-Income Ratio (Clipped)')
        st.pyplot(fig8)
        st.markdown("""
        **Insight:** This is a *critical* predictor. 'Not_Eligible' customers have a
        dangerously high DTI (median > 80%). 'Eligible' customers have a much
        healthier median DTI (around 40-50%).
        """)
        
    with col8:
        # --- Chart 9 ---
        st.subheader("Savings Ratio vs. Eligibility")
        fig9, ax9 = plt.subplots()
        sns.boxplot(x='emi_eligibility', y='savings_ratio', data=df_with_ratios, ax=ax9, palette="Greens")
        ax9.set_title('Savings Ratio by Eligibility Status')
        ax9.set_xlabel('Eligibility Status')
        ax9.set_ylabel('Savings Ratio (Clipped)')
        st.pyplot(fig9)
        st.markdown("""
        **Insight:** This is the inverse of DTI and is equally powerful. 'Eligible'
        customers have a much higher savings ratio (median ~50-60%), while 'Not_Eligible'
        customers have almost no savings (median < 20%).
        """)

    # --- 6. Correlation Heatmap ---
    st.header("6. Correlation Heatmap (with Financial Ratios)", divider="blue")
    st.markdown("""
    A heatmap of the key numerical features *including our new ratios*.
    This helps us see which features are most correlated with our targets.
    """)
    
    with st.spinner("Generating enhanced correlation heatmap..."):
        # --- Chart 10 ---
        fig10, ax10 = plt.subplots(figsize=(20, 16))
        # Select key numerical cols for a cleaner heatmap
        key_num_cols = [
            'age', 'monthly_salary', 'years_of_employment', 'monthly_rent',
            'family_size', 'dependents', 'current_emi_amount', 'credit_score', 
            'bank_balance', 'emergency_fund', 'requested_amount', 'requested_tenure',
            'debt_to_income_ratio', 'savings_ratio', # Add our new ratios
            'max_monthly_emi' # Target
        ]
        
        # Calculate correlation
        corr = df_with_ratios[key_num_cols].corr()
        
        # Create a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        # Plot heatmap
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                    linewidths=.5, ax=ax10, annot_kws={"size": 10})
        ax10.set_title("Correlation Heatmap of Key Features & Ratios", fontsize=16)
        st.pyplot(fig10)
        st.markdown("""
        **Insights:**
        * `max_monthly_emi` (our regression target) has a very strong positive correlation with `monthly_salary` (0.81) and `savings_ratio` (0.87).
        * It has a very strong *negative* correlation with `debt_to_income_ratio` (-0.86).
        * This confirms that our engineered features are the most powerful predictors for our regression task.
        """)

else:
    st.error("Data could not be loaded. Please ensure `emi_prediction_dataset.csv` is in the root directory.")

