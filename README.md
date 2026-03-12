# ClinicalIQ — Patient Readmission Prediction Pipeline

> An end-to-end Data Science pipeline combining SQL-driven feature engineering with 
> machine learning to predict 30-day hospital readmission risk across 100,000+ patient records.

![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
![PostgreSQL](https://img.shields.io/badge/postgres-%23316192.svg?style=flat&logo=postgresql&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)

---

## 🎯 Problem Statement

Hospital readmissions within 30 days cost the U.S. healthcare system over **$26 billion annually**. 
Identifying high-risk patients before discharge allows clinicians to intervene early — reducing 
costs, improving outcomes, and saving lives.

ClinicalIQ addresses this by building a production-grade data pipeline that:
- Stores and manages clinical data in a **relational PostgreSQL database**
- Engineers meaningful features using **intermediate-to-advanced SQL**
- Trains an **XGBoost classifier** to predict 30-day readmission risk
- Surfaces predictions and insights via an **interactive Streamlit dashboard**

---

## 📊 Dataset

**Diabetes 130-US Hospitals Dataset** — UCI Machine Learning Repository

- 100,000+ patient encounter records from 130 US hospitals (1999–2008)
- 50 features including demographics, diagnoses, medications, and lab results
- Binary target: readmitted within 30 days (Yes / No)
- Publicly available, no IRB approval required
- Used in peer-reviewed clinical informatics research

> Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

---

## 🏗️ Architecture

```
Raw CSV Data
     │
     ▼
PostgreSQL Database          ← Schema design, indexing, constraints
     │
     ▼
SQL Feature Engineering      ← CTEs, Window Functions, JOINs, GROUP BY
     │
     ▼
Python ETL Pipeline          ← SQLAlchemy + pandas extraction
     │
     ▼
Preprocessing & EDA          ← Matplotlib, Seaborn, missing value handling
     │
     ▼
XGBoost Classifier           ← Training, hyperparameter tuning, evaluation
     │
     ▼
Streamlit Dashboard          ← Interactive predictions + clinical insights
     │
     ▼
Docker Container             ← Reproducible, deployable environment
```

---

## 🗄️ Database Schema

```sql
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
```

---

## 🔍 SQL Feature Engineering

### Level 1 — Readmission Rate by Diagnosis Group
```sql
SELECT 
    d.diagnosis_group,
    COUNT(*) AS total_encounters,
    SUM(CASE WHEN e.readmitted THEN 1 ELSE 0 END) AS readmissions,
    ROUND(AVG(CASE WHEN e.readmitted THEN 1.0 ELSE 0.0 END) * 100, 2) 
        AS readmission_rate_pct
FROM encounters e
JOIN diagnoses d ON e.encounter_id = d.encounter_id
GROUP BY d.diagnosis_group
ORDER BY readmission_rate_pct DESC;
```

### Level 2 — Patient History Features via CTEs
```sql
WITH patient_history AS (
    SELECT 
        patient_id,
        COUNT(*)                    AS total_prior_visits,
        AVG(length_of_stay)         AS avg_stay_days,
        AVG(num_medications)        AS avg_medications,
        MAX(num_lab_procedures)     AS max_lab_procedures
    FROM encounters
    GROUP BY patient_id
)
SELECT 
    e.*,
    ph.total_prior_visits,
    ph.avg_stay_days,
    ph.avg_medications,
    ph.max_lab_procedures
FROM encounters e
LEFT JOIN patient_history ph ON e.patient_id = ph.patient_id;
```

### Level 3 — 30-Day Rolling Readmission Window Functions
```sql
SELECT
    encounter_id,
    patient_id,
    encounter_date,
    LAG(encounter_date) OVER (
        PARTITION BY patient_id 
        ORDER BY encounter_date
    ) AS prev_encounter_date,
    encounter_date - LAG(encounter_date) OVER (
        PARTITION BY patient_id 
        ORDER BY encounter_date
    ) AS days_since_last_visit,
    CASE 
        WHEN encounter_date - LAG(encounter_date) OVER (
            PARTITION BY patient_id ORDER BY encounter_date
        ) <= 30 THEN TRUE 
        ELSE FALSE 
    END AS readmitted_within_30_days
FROM encounters
ORDER BY patient_id, encounter_date;
```

---

## 🤖 Machine Learning Pipeline

### Model
- **Algorithm:** XGBoost Classifier
- **Target:** 30-day readmission (binary classification)
- **Evaluation Metrics:** AUC-ROC, F1-Score, Precision, Recall
- **Expected AUC:** 0.80+

### Feature Set (SQL-engineered)
| Feature | Source | SQL Technique |
|---|---|---|
| Total prior visits | Patient history | CTE + COUNT |
| Avg length of stay | Patient history | CTE + AVG |
| Days since last visit | Rolling window | LAG window function |
| Readmission rate by diagnosis | Diagnosis group | JOIN + GROUP BY |
| Medication change flag | Medications table | JOIN + CASE WHEN |
| Number of diagnoses | Encounters table | Direct column |

### Pipeline Steps
```python
# 1. Extract features from PostgreSQL via SQLAlchemy
# 2. Handle missing values + encode categoricals
# 3. Train/test split (80/20, stratified)
# 4. XGBoost with cross-validation
# 5. Hyperparameter tuning via GridSearchCV
# 6. Evaluate: AUC, F1, Confusion Matrix
# 7. SHAP values for feature importance
```

---

## 📈 Dashboard Features (Streamlit)

- **Patient Risk Scorer** — input patient features, get real-time readmission probability
- **Diagnosis Heatmap** — readmission rates across diagnosis groups
- **Feature Importance Chart** — SHAP-based explainability
- **Hospital Trends** — readmission trends over time by admission type
- **SQL Query Explorer** — run live queries against the database from the UI

---

## 📁 Project Structure

```
ClinicalIQ/
├── data/
│   ├── raw/                  # Original UCI dataset
│   └── processed/            # Cleaned, feature-engineered data
├── sql/
│   ├── schema.sql            # Table definitions + indexes
│   ├── features.sql          # Feature engineering queries
│   └── analysis.sql          # EDA SQL queries
├── notebooks/
│   └── EDA.ipynb             # Exploratory data analysis
├── src/
│   ├── ingest.py             # Load CSV → PostgreSQL
│   ├── features.py           # SQLAlchemy feature extraction
│   ├── preprocess.py         # Cleaning + encoding
│   ├── model.py              # XGBoost training + evaluation
│   └── dashboard.py          # Streamlit app
├── tests/
│   └── test_pipeline.py      # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/ShubhamHarkare/ClinicalIQ.git
cd ClinicalIQ

# Run with Docker (recommended)
docker-compose up --build

# Or run locally
pip install -r requirements.txt
python src/ingest.py          # Load data into PostgreSQL
python src/features.py        # Engineer features via SQL
python src/model.py           # Train XGBoost model
streamlit run src/dashboard.py # Launch dashboard
```

---

## 📅 Development Timeline

| Week | Milestone |
|---|---|
| Week 1 | PostgreSQL setup, schema design, data ingestion |
| Week 2 | SQL feature engineering (all 3 levels) |
| Week 3 | Python ETL pipeline + EDA notebook |
| Week 4 | XGBoost model training + evaluation |
| Week 5 | Streamlit dashboard |
| Week 6 | Docker, testing, README, GitHub polish |

---

## 🔮 Future Work

- Add **dbt (data build tool)** for SQL transformation management
- Integrate **Apache Airflow** for pipeline orchestration
- Extend to **multi-class prediction** (readmitted < 30 days / > 30 days / not readmitted)
- Deploy dashboard to **Streamlit Cloud**

---

## 👤 Author

**Shubham Harkare**  
MS Data Science @ University of Michigan  
[LinkedIn](https://linkedin.com/in/shubham-harkare) | [GitHub](https://github.com/ShubhamHarkare)

---

*Dataset: Strack, B., et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International.*
