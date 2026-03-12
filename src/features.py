import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "clinicaliq")

PROCESSED_DATA_PATH = "data/processed/features.csv"

def get_db_engine():
    encoded_pass = urllib.parse.quote_plus(DB_PASS)
    connection_string = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def engineer_features():
    print("🔍 Starting Advanced SQL Feature Engineering (Levels 1, 2, & 3)...")
    start_time = time.time()
    engine = get_db_engine()

    # The Master Query combining all 3 Levels from the README
    sql_query = """
    WITH diagnosis_rates AS (
        -- Level 1: Readmission Rate by Diagnosis Group
        SELECT 
            d.diagnosis_group,
            ROUND(AVG(CASE WHEN e.readmitted THEN 1.0 ELSE 0.0 END) * 100, 2) AS readmission_rate_pct
        FROM encounters e
        JOIN diagnoses d ON e.encounter_id = d.encounter_id
        GROUP BY d.diagnosis_group
    ),
    patient_history AS (
        -- Level 2: Patient History Features via CTEs
        SELECT 
            patient_id,
            COUNT(*) AS total_prior_visits,
            ROUND(AVG(length_of_stay), 2) AS avg_stay_days,
            ROUND(AVG(num_medications), 2) AS avg_medications,
            MAX(num_lab_procedures) AS max_lab_procedures
        FROM encounters
        GROUP BY patient_id
    ),
    rolling_window AS (
        -- Level 3: 30-Day Rolling Window Functions
        SELECT
            encounter_id,
            CAST(encounter_date AS DATE) - CAST(LAG(encounter_date) OVER (
                PARTITION BY patient_id 
                ORDER BY encounter_date
            ) AS DATE) AS days_since_last_visit
        FROM encounters
    ),
    med_changes AS (
        -- Extracting medication change flags from the new medications table
        SELECT 
            encounter_id, 
            MAX(CASE WHEN dosage_change THEN 1 ELSE 0 END) AS med_change_flag
        FROM medications
        GROUP BY encounter_id
    )
    SELECT 
        e.encounter_id,
        p.age,
        p.gender,
        p.race,
        p.admission_type,
        p.discharge_id,
        e.length_of_stay,
        e.num_lab_procedures,
        e.num_medications,
        e.num_diagnoses,
        COALESCE(ph.total_prior_visits, 0) AS total_prior_visits,
        COALESCE(ph.avg_stay_days, 0) AS avg_stay_days,
        COALESCE(ph.avg_medications, 0) AS avg_medications,
        COALESCE(ph.max_lab_procedures, 0) AS max_lab_procedures,
        COALESCE(rw.days_since_last_visit, 999) AS days_since_last_visit,
        COALESCE(dr.readmission_rate_pct, 0) AS diag_readmission_rate,
        COALESCE(mc.med_change_flag, 0) AS medication_change_flag,
        e.readmitted
    FROM encounters e
    JOIN patients p ON e.patient_id = p.patient_id
    LEFT JOIN diagnoses d ON e.encounter_id = d.encounter_id
    LEFT JOIN diagnosis_rates dr ON d.diagnosis_group = dr.diagnosis_group
    LEFT JOIN patient_history ph ON e.patient_id = ph.patient_id
    LEFT JOIN rolling_window rw ON e.encounter_id = rw.encounter_id
    LEFT JOIN med_changes mc ON e.encounter_id = mc.encounter_id;
    """

    print("⚡ Executing Window Functions, CTEs, and JOINs in PostgreSQL...")
    df = pd.read_sql(sql_query, engine)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    # Save to processed folder
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    
    elapsed_time = round(time.time() - start_time, 2)
    print(f"✅ Extracted {len(df)} fully engineered records matching the README!")
    print(f"💾 Saved engineered dataset to {PROCESSED_DATA_PATH}")
    print(f"🎉 Feature engineering complete in {elapsed_time} seconds!")

if __name__ == "__main__":
    engineer_features()