-- Core patient table
CREATE TABLE patients (
    patient_id      SERIAL PRIMARY KEY,
    age             VARCHAR(10),
    gender          VARCHAR(10),
    race            VARCHAR(20),
    admission_type  INTEGER,
    discharge_id    INTEGER,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Encounter table (one patient, many visits)
CREATE TABLE encounters (
    encounter_id        SERIAL PRIMARY KEY,
    patient_id          INTEGER REFERENCES patients(patient_id),
    encounter_date      DATE,
    length_of_stay      INTEGER,
    num_lab_procedures  INTEGER,
    num_medications     INTEGER,
    num_diagnoses       INTEGER,
    readmitted          BOOLEAN,
    created_at          TIMESTAMP DEFAULT NOW()
);

-- Diagnoses table
CREATE TABLE diagnoses (
    diagnosis_id    SERIAL PRIMARY KEY,
    encounter_id    INTEGER REFERENCES encounters(encounter_id),
    icd9_code       VARCHAR(10),
    diagnosis_group VARCHAR(50)
);

-- Medications table
CREATE TABLE medications (
    medication_id   SERIAL PRIMARY KEY,
    encounter_id    INTEGER REFERENCES encounters(encounter_id),
    drug_name       VARCHAR(50),
    dosage_change   BOOLEAN
);