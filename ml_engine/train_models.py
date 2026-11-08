"""
EduPredict - Machine Learning Training Pipeline
Trains high-performance predictive models using Apache Parquet big data:
1. Student Dropout & Retention Classifier (Random Forest / Gradient Boosting)
2. Academic Performance & GPA Regressor (Gradient Boosting Regressor)
3. Course Demand & Enrollment Forecaster (Trend Regressor)
4. Engagement Anomaly Detector (Isolation Forest)

Saves all trained pipelines and artifacts into `ml_engine/saved_models/`.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    IsolationForest
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    mean_absolute_error
)

SAVED_MODELS_DIR = "ml_engine/saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def train_dropout_model(df):
    print("\n" + "=" * 60)
    print("[MODEL 1/4] Training Student Dropout & Retention Classifier...")
    print("=" * 60)
    
    categorical_features = ["department", "gender", "guardian_education"]
    numeric_features = [
        "age", "semester", "financial_aid", "high_school_gpa",
        "previous_semester_gpa", "cumulative_gpa", "total_credit_hours",
        "failed_courses_count", "attendance_rate", "consecutive_absences",
        "quiz_average", "assignment_average", "midterm_exam_score",
        "final_exam_score", "lms_engagement_score", "lms_weekly_logins",
        "lms_forum_posts", "lms_study_hours_per_week", "lms_late_submissions",
        "lms_downloads_count"
    ]
    
    X = df[categorical_features + numeric_features]
    y = df["dropped_out"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    
    # HistGradientBoostingClassifier provides excellent speed and calibrated probabilities on large datasets
    classifier = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=8,
        l2_regularization=1.5,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    start_t = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_t
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f" -> Training Time: {train_duration:.2f}s")
    print(f" -> Accuracy:  {acc * 100:.2f}%")
    print(f" -> Precision: {prec * 100:.2f}%")
    print(f" -> Recall:    {rec * 100:.2f}%")
    print(f" -> F1-Score:  {f1 * 100:.2f}%")
    print(f" -> ROC-AUC:   {auc * 100:.2f}%")
    
    model_path = os.path.join(SAVED_MODELS_DIR, "dropout_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f" -> Saved Model to: {model_path}")
    
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(auc, 4),
        "features": categorical_features + numeric_features
    }


def train_performance_model(df):
    print("\n" + "=" * 60)
    print("[MODEL 2/4] Training Academic Performance & GPA Regressor...")
    print("=" * 60)
    
    categorical_features = ["department", "gender"]
    numeric_features = [
        "semester", "cumulative_gpa", "previous_semester_gpa", "high_school_gpa",
        "attendance_rate", "quiz_average", "assignment_average",
        "midterm_exam_score", "final_exam_score", "failed_courses_count",
        "lms_engagement_score", "lms_study_hours_per_week"
    ]
    
    X = df[categorical_features + numeric_features]
    y = df["next_semester_gpa"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    
    regressor = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.08,
        max_depth=7,
        l2_regularization=1.0,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    start_t = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_t
    
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f" -> Training Time: {train_duration:.2f}s")
    print(f" -> Mean Absolute Error (MAE): {mae:.4f} GPA points")
    print(f" -> Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f" -> R² Score (Variance Explained): {r2 * 100:.2f}%")
    
    model_path = os.path.join(SAVED_MODELS_DIR, "performance_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f" -> Saved Model to: {model_path}")
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "features": categorical_features + numeric_features
    }


def train_anomaly_model(df):
    print("\n" + "=" * 60)
    print("[MODEL 3/4] Training Engagement Anomaly Detection Model (Isolation Forest)...")
    print("=" * 60)
    
    features = [
        "cumulative_gpa", "attendance_rate", "consecutive_absences",
        "lms_engagement_score", "lms_weekly_logins", "lms_late_submissions"
    ]
    
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso_forest = IsolationForest(
        n_estimators=120,
        contamination=0.05, # Expect ~5% behavioral anomalies
        random_state=42,
        n_jobs=-1
    )
    
    start_t = time.time()
    iso_forest.fit(X_scaled)
    train_duration = time.time() - start_t
    
    # Save both scaler and isolation forest model
    model_artifact = {
        "scaler": scaler,
        "model": iso_forest,
        "features": features
    }
    
    model_path = os.path.join(SAVED_MODELS_DIR, "anomaly_model.joblib")
    joblib.dump(model_artifact, model_path)
    
    print(f" -> Training Time: {train_duration:.2f}s")
    print(f" -> Anomaly Model: IsolationForest (120 Estimators, Contamination=5%)")
    print(f" -> Saved Model to: {model_path}")
    
    return {
        "algorithm": "IsolationForest",
        "contamination": 0.05,
        "features": features
    }


def train_course_demand_model(demand_df_path="data/parquet/course_demand_history.parquet"):
    print("\n" + "=" * 60)
    print("[MODEL 4/4] Training Course Demand & Enrollment Forecaster...")
    print("=" * 60)
    
    df_demand = pd.read_parquet(demand_df_path)
    
    features = ["term_sequence", "target_semester", "pass_rate", "drop_rate", "average_gpa"]
    categorical_features = ["course_code", "department"]
    
    X = df_demand[features + categorical_features]
    y = df_demand["enrolled_count"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )
    
    regressor = Ridge(alpha=1.0)
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    start_t = time.time()
    pipeline.fit(X, y)
    train_duration = time.time() - start_t
    
    y_pred = pipeline.predict(X)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f" -> Training Time: {train_duration:.3f}s")
    print(f" -> Mean Absolute Error: {mae:.2f} students")
    print(f" -> R² Score: {r2 * 100:.2f}%")
    
    model_path = os.path.join(SAVED_MODELS_DIR, "course_demand_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f" -> Saved Model to: {model_path}")
    
    return {
        "mae": round(mae, 2),
        "r2_score": round(r2, 4),
        "features": features + categorical_features
    }


def run_training_pipeline():
    total_start = time.time()
    print("=" * 70)
    print("EDUPREDICT: EXECUTING FULL ML TRAINING PIPELINE (BIG DATA PARQUET)")
    print("=" * 70)
    
    parquet_path = "data/parquet/student_performance_bigdata.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file {parquet_path} not found.")
    
    print(f"Loading Big Data from: {parquet_path}")
    t0 = time.time()
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} records in {time.time() - t0:.2f}s!")
    
    metadata = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_records": len(df),
        "models": {}
    }
    
    metadata["models"]["dropout_classifier"] = train_dropout_model(df)
    metadata["models"]["performance_regressor"] = train_performance_model(df)
    metadata["models"]["anomaly_detector"] = train_anomaly_model(df)
    metadata["models"]["course_demand_forecaster"] = train_course_demand_model()
    
    meta_path = os.path.join(SAVED_MODELS_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("\n" + "=" * 70)
    print(f"ALL 4 ML MODELS TRAINED & SAVED SUCCESSFULLY in {time.time() - total_start:.2f}s!")
    print(f"Metadata exported to: {meta_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
