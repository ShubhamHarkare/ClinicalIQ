import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb

# --- PAGE CONFIG ---
st.set_page_config(page_title="ClinicalIQ | Readmission Risk", page_icon="🏥", layout="wide")

# --- LOAD MODEL & FEATURES ---
@st.cache_resource
def load_model_data():
    model = joblib.load("src/xgboost_readmission.pkl")
    features = joblib.load("src/model_features.pkl")
    return model, features

model, model_features = load_model_data()

# --- UI HEADER ---
st.title("🏥 ClinicalIQ: Readmission Risk Predictor")
st.markdown("Enter patient demographics and encounter details below to predict their 30-day readmission risk.")
st.divider()

# --- INPUT COLUMNS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Demographics")
    age = st.selectbox("Age Group", ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'], index=7)
    gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other", "Unknown"])
    admission_type = st.number_input("Admission Type ID", min_value=1, value=1)
    discharge_id = st.number_input("Discharge ID", min_value=1, value=1)

with col2:
    st.header("🩺 Current Visit")
    length_of_stay = st.number_input("Length of Stay (Days)", min_value=1, value=3)
    num_lab_procedures = st.number_input("Lab Procedures", min_value=1, value=40)
    num_medications = st.number_input("Medications Administered", min_value=1, value=15)
    num_diagnoses = st.number_input("Diagnoses Count", min_value=1, value=5)

with col3:
    st.header("📋 Patient History")
    total_prior_visits = st.number_input("Total Prior Visits", min_value=0, value=0)
    avg_stay_days = st.number_input("Historic Avg Stay (Days)", min_value=0.0, value=0.0)
    avg_medications = st.number_input("Historic Avg Medications", min_value=0.0, value=0.0)
    max_lab_procedures = st.number_input("Historic Max Lab Procedures", min_value=0, value=0)

st.divider()

# --- PREDICTION LOGIC ---
if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
    # 1. Gather raw inputs
    input_data = {
        'admission_type': admission_type, 'discharge_id': discharge_id,
        'length_of_stay': length_of_stay, 'num_lab_procedures': num_lab_procedures,
        'num_medications': num_medications, 'num_diagnoses': num_diagnoses,
        'total_prior_visits': total_prior_visits, 'avg_stay_days': avg_stay_days,
        'avg_medications': avg_medications, 'max_lab_procedures': max_lab_procedures,
        'age': age, 'gender': gender, 'race': race
    }
    
    # 2. Convert to DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 3. One-Hot Encode
    categorical_to_encode = ['age', 'gender', 'race']
    df_encoded = pd.get_dummies(df_input, columns=categorical_to_encode)
    
    # 4. Clean column names (matching our model.py fix)
    df_encoded.columns = df_encoded.columns.str.replace(r'[\[\]<]', '_', regex=True)
    
    # 5. Reindex to force the exact shape the XGBoost model expects
    # Missing columns will be filled with False/0, extra columns dropped
    df_final = df_encoded.reindex(columns=model_features, fill_value=0).astype(float)
    
    # 6. Run Prediction
    probability = model.predict_proba(df_final)[0][1]
    risk_percentage = probability * 100
    
    # 7. Display Results
    st.subheader("🔮 Prediction Results")
    
    if risk_percentage > 50:
        st.error(f"**HIGH RISK** — The patient has a {risk_percentage:.1f}% probability of 30-day readmission.")
    elif risk_percentage > 20:
        st.warning(f"**MODERATE RISK** — The patient has a {risk_percentage:.1f}% probability of 30-day readmission.")
    else:
        st.success(f"**LOW RISK** — The patient has a {risk_percentage:.1f}% probability of 30-day readmission.")
        
    st.progress(probability)
    st.caption("*Disclaimer: This is an MVP analytical tool. Clinical decisions must not rely solely on this prediction.*")