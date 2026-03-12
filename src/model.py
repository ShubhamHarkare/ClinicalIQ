import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import time
import re

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = "data/processed/features.csv"
MODEL_SAVE_PATH = "src/xgboost_readmission.pkl"
FEATURES_SAVE_PATH = "src/model_features.pkl"

def train_model():
    print("🧠 Starting Phase 3 Model Training Pipeline...")
    start_time = time.time()

    # 1. Load Data
    print(f"Loading fully engineered features from {PROCESSED_DATA_PATH}...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {PROCESSED_DATA_PATH}. Did features.py run successfully?")
        return

    # 2. Preprocessing
    print("⚙️ Preprocessing data...")
    if 'encounter_id' in df.columns:
        df = df.drop(columns=['encounter_id'])
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(exclude=['number']).columns

    df[numeric_cols] = df[numeric_cols].fillna(-1)
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    categorical_to_encode = ['age', 'gender', 'race']
    df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)

    X = df.drop(columns=['readmitted'])
    y = df['readmitted'].astype(int)

    # Clean column names for XGBoost
    X.columns = X.columns.str.replace(r'[\[\]<]', '_', regex=True)

    # 3. Train/Test Split
    print("🔀 Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Hyperparameter Tuning with GridSearchCV
    print("🚀 Running GridSearchCV for Hyperparameter Tuning (this may take a minute)...")
    
    # We use scale_pos_weight to finally address the class imbalance!
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    base_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=ratio # Forces the model to care about the minority class
    )

    # A small grid to keep execution time low for the MVP
    param_grid = {
        'max_depth': [3, 5],
        'n_estimators': [50, 100]
    }

    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        scoring='roc_auc', 
        cv=3, 
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"✨ Best Parameters Found: {grid_search.best_params_}")

    # 5. Evaluation
    print("📊 Evaluating Best Model...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"\n✅ Model AUC-ROC: {auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report (F1, Precision, Recall):")
    print(classification_report(y_test, y_pred))

    # 6. Save Model and Feature Names
    print(f"💾 Saving optimized model to {MODEL_SAVE_PATH}...")
    joblib.dump(best_model, MODEL_SAVE_PATH)
    joblib.dump(list(X.columns), FEATURES_SAVE_PATH)

    elapsed_time = round(time.time() - start_time, 2)
    print(f"🎉 Phase 3 Training complete in {elapsed_time} seconds!")

if __name__ == "__main__":
    train_model()