import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "clinicaliq")

RAW_DATA_PATH = "data/raw/diabetic_data.csv"

def get_db_engine():
    encoded_pass = urllib.parse.quote_plus(DB_PASS)
    connection_string = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def map_diagnosis(icd9):
    """Maps raw ICD9 codes to broad diagnosis groups for our Level 1 SQL."""
    if pd.isna(icd9) or icd9 == '?': return 'Unknown'
    if str(icd9).startswith('250'): return 'Diabetes'
    if str(icd9).startswith(('39', '40', '41', '42', '43', '44', '45')): return 'Circulatory'
    if str(icd9).startswith(('46', '47', '48', '49', '50', '51')): return 'Respiratory'
    return 'Other'

def process_and_ingest():
    print("🚀 Starting Phase 1 Ingestion Pipeline...")
    start_time = time.time()
    engine = get_db_engine()

    # 1. Load Data
    print(f"Loading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)
    df.replace('?', None, inplace=True)

    # 2. Patients Table
    print("🗄️ Processing Patients...")
    df['gender'] = df['gender'].replace('Unknown/Invalid', 'Unknown').str[:10]
    df['age'] = df['age'].str[:10]
    
    patients = df[['patient_nbr', 'age', 'gender', 'race', 'admission_type_id', 'discharge_disposition_id']].copy()
    patients.rename(columns={'patient_nbr': 'patient_id', 'admission_type_id': 'admission_type', 'discharge_disposition_id': 'discharge_id'}, inplace=True)
    patients.drop_duplicates(subset=['patient_id'], inplace=True)
    patients.to_sql('patients', engine, if_exists='append', index=False)
    print(f"✅ Loaded {len(patients)} patients.")

    # 3. Encounters Table (With Mock Dates for Window Functions)
    print("🗄️ Processing Encounters...")
    # Sort to ensure chronological order for mock dates
    df = df.sort_values(by=['patient_nbr', 'encounter_id'])
    # Synthesize dates: Start at Jan 1, 2020. Add 45 days per subsequent visit for the same patient.
    df['encounter_date'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(df.groupby('patient_nbr').cumcount() * 45, unit='D')
    
    encounters = df[['encounter_id', 'patient_nbr', 'encounter_date', 'time_in_hospital', 'num_lab_procedures', 'num_medications', 'number_diagnoses', 'readmitted']].copy()
    encounters.rename(columns={'patient_nbr': 'patient_id', 'time_in_hospital': 'length_of_stay', 'number_diagnoses': 'num_diagnoses'}, inplace=True)
    encounters['readmitted'] = encounters['readmitted'].apply(lambda x: True if x == '<30' else False)
    
    encounters.to_sql('encounters', engine, if_exists='append', index=False)
    print(f"✅ Loaded {len(encounters)} encounters.")

    # 4. Diagnoses Table
    print("🗄️ Processing Diagnoses...")
    # Using 'diag_1' as the primary diagnosis to match the schema
    diagnoses = df[['encounter_id', 'diag_1']].copy()
    diagnoses.rename(columns={'diag_1': 'icd9_code'}, inplace=True)
    diagnoses['diagnosis_group'] = diagnoses['icd9_code'].apply(map_diagnosis)
    
    diagnoses.to_sql('diagnoses', engine, if_exists='append', index=False)
    print(f"✅ Loaded {len(diagnoses)} diagnoses.")

    # 5. Medications Table
    print("🗄️ Processing Medications...")
    med_columns = ['metformin', 'repaglinide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone', 'insulin']
    
    # Melt the dataframe so each row is a patient-drug combination
    meds_melted = df.melt(id_vars=['encounter_id'], value_vars=med_columns, var_name='drug_name', value_name='status')
    
    # Filter out 'No' (meaning they weren't on the drug)
    active_meds = meds_melted[meds_melted['status'] != 'No'].copy()
    
    # Check if dosage changed (Up or Down)
    active_meds['dosage_change'] = active_meds['status'].apply(lambda x: True if x in ['Up', 'Down'] else False)
    medications = active_meds[['encounter_id', 'drug_name', 'dosage_change']]
    
    medications.to_sql('medications', engine, if_exists='append', index=False)
    print(f"✅ Loaded {len(medications)} medication records.")

    print(f"🎉 Pipeline complete in {round(time.time() - start_time, 2)} seconds!")

if __name__ == "__main__":
    process_and_ingest()