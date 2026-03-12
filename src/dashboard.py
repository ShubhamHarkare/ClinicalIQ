import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# --- PAGE CONFIG ---
st.set_page_config(page_title="ClinicalIQ Dashboard", page_icon="🏥", layout="wide")
load_dotenv()

# --- CACHED RESOURCES ---
@st.cache_resource
def load_model_data():
    model = joblib.load("src/xgboost_readmission.pkl")
    features = joblib.load("src/model_features.pkl")
    df = pd.read_csv("data/processed/features.csv")
    return model, features, df

@st.cache_resource
def get_db_engine():
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASS")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "clinicaliq")
    # For SQLAlchemy, we need to url-encode the password in case it has special characters like @
    import urllib.parse
    encoded_pass = urllib.parse.quote_plus(DB_PASS)
    return create_engine(f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

model, model_features, processed_df = load_model_data()
engine = get_db_engine()

# --- UI HEADER ---
st.title("🏥 ClinicalIQ: Patient Readmission Prediction Pipeline")
st.markdown("End-to-end Machine Learning pipeline predicting 30-day hospital readmissions.")
st.divider()

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["🩺 Risk Scorer", "📊 Diagnosis Heatmap", "🧠 Feature Importance (SHAP)", "💻 SQL Explorer"])

# ==========================================
# TAB 1: RISK SCORER
# ==========================================
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        age = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=7)
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"])
        admission_type = st.number_input("Admission Type ID", min_value=1, value=1)
        discharge_id = st.number_input("Discharge ID", min_value=1, value=1)

    with col2:
        st.subheader("🏥 Current Visit & Meds")
        length_of_stay = st.number_input("Length of Stay (Days)", min_value=1, value=3)
        num_lab_procedures = st.number_input("Lab Procedures", min_value=1, value=40)
        num_medications = st.number_input("Medications Administered", min_value=1, value=15)
        num_diagnoses = st.number_input("Diagnoses Count", min_value=1, value=5)
        medication_change_flag = st.selectbox("Dosage Changed During Visit?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")

    with col3:
        st.subheader("📋 Patient History (SQL Derived)")
        total_prior_visits = st.number_input("Total Prior Visits", min_value=0, value=2)
        days_since_last_visit = st.number_input("Days Since Last Visit (999 if none)", min_value=0, value=45)
        diag_readmission_rate = st.slider("Historical Readmission Rate for Primary Diagnosis (%)", 0.0, 100.0, 45.0)
        avg_stay_days = st.number_input("Historic Avg Stay", min_value=0.0, value=3.0)
        avg_medications = st.number_input("Historic Avg Meds", min_value=0.0, value=15.0)
        max_lab_procedures = st.number_input("Historic Max Labs", min_value=0, value=40)

    st.divider()

    if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
        input_data = {
            'admission_type': admission_type, 'discharge_id': discharge_id,
            'length_of_stay': length_of_stay, 'num_lab_procedures': num_lab_procedures,
            'num_medications': num_medications, 'num_diagnoses': num_diagnoses,
            'total_prior_visits': total_prior_visits, 'avg_stay_days': avg_stay_days,
            'avg_medications': avg_medications, 'max_lab_procedures': max_lab_procedures,
            'days_since_last_visit': days_since_last_visit, 
            'diag_readmission_rate': diag_readmission_rate,
            'medication_change_flag': medication_change_flag,
            'age': age, 'gender': gender, 'race': race
        }
        
        df_input = pd.DataFrame([input_data])
        df_encoded = pd.get_dummies(df_input, columns=['age', 'gender', 'race'])
        df_encoded.columns = df_encoded.columns.str.replace(r'[\[\]<]', '_', regex=True)
        df_final = df_encoded.reindex(columns=model_features, fill_value=0).astype(float)
        
        probability = model.predict_proba(df_final)[0][1] * 100
        
        st.subheader("🔮 Prediction Results")
        if probability > 50:
            st.error(f"**HIGH RISK** — {probability:.1f}% probability of 30-day readmission.")
        elif probability > 20:
            st.warning(f"**MODERATE RISK** — {probability:.1f}% probability of 30-day readmission.")
        else:
            st.success(f"**LOW RISK** — {probability:.1f}% probability of 30-day readmission.")
        st.progress(probability / 100.0)

# ==========================================
# TAB 2: DIAGNOSIS HEATMAP
# ==========================================
with tab2:
    st.subheader("Readmission Rates by Diagnosis Group (Level 1 SQL)")
    try:
        query = """
        SELECT d.diagnosis_group,
               COUNT(*) as total_cases,
               ROUND(AVG(CASE WHEN e.readmitted THEN 1.0 ELSE 0.0 END) * 100, 2) as readmission_rate_pct
        FROM encounters e
        JOIN diagnoses d ON e.encounter_id = d.encounter_id
        GROUP BY d.diagnosis_group
        ORDER BY readmission_rate_pct DESC;
        """
        diag_df = pd.read_sql(query, engine)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=diag_df, x='readmission_rate_pct', y='diagnosis_group', palette="rocket", ax=ax)
        ax.set_xlabel("Readmission Rate (%)")
        ax.set_ylabel("Diagnosis Group")
        # Ensure plot looks good in dark mode
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        st.dataframe(diag_df, use_container_width=True)
    except Exception as e:
        st.error(f"Database connection error: {e}")

# ==========================================
# TAB 3: FEATURE IMPORTANCE (SHAP)
# ==========================================
with tab3:
    st.subheader("What is driving the model's predictions?")
    st.markdown("Using SHAP (SHapley Additive exPlanations) to interpret our XGBoost model.")
    
    with st.spinner("Calculating SHAP values..."):
        # Sample data to make SHAP calculation fast for the UI
        X_sample = processed_df.drop(columns=['readmitted', 'encounter_id'], errors='ignore').sample(500, random_state=42)
        X_sample.columns = X_sample.columns.str.replace(r'[\[\]<]', '_', regex=True)
        X_sample = X_sample.reindex(columns=model_features, fill_value=0).astype(float)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        fig.patch.set_facecolor('#0e1117')
        plt.gca().tick_params(colors='white')
        plt.gca().xaxis.label.set_color('white')
        st.pyplot(fig)

# ==========================================
# TAB 4: SQL EXPLORER
# ==========================================
with tab4:
    st.subheader("Live Database Query Explorer")
    st.markdown("Run custom SQL queries directly against your local PostgreSQL container.")
    
    default_query = "SELECT * FROM patients LIMIT 5;"
    user_query = st.text_area("SQL Query", value=default_query, height=150)
    
    if st.button("Run Query", type="secondary"):
        try:
            result_df = pd.read_sql(user_query, engine)
            st.success(f"Query executed successfully. Rows returned: {len(result_df)}")
            st.dataframe(result_df, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")