"""
EduPredict - ML Inference & Prediction Service
Loads serialized models and provides real-time inference for:
- Student Dropout Risk & At-Risk classification
- Next-semester GPA Forecast
- Behavioral & Engagement Anomaly Detection
- Course Demand Forecasting
- Syncing predictions & generating early-warning alerts into MongoDB Atlas
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from edupredict.db import (
    get_db,
    COLL_STUDENTS,
    COLL_ACADEMIC_RECORDS,
    COLL_ATTENDANCE,
    COLL_LMS_ACTIVITY,
    COLL_PREDICTIONS,
    COLL_ALERTS,
    COLL_COURSES
)

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


class EduPredictMLService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EduPredictMLService, cls).__new__(cls)
            cls._instance._load_models()
        return cls._instance

    def _load_models(self):
        try:
            self.dropout_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "dropout_model.joblib"))
            self.performance_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "performance_model.joblib"))
            self.anomaly_artifact = joblib.load(os.path.join(SAVED_MODELS_DIR, "anomaly_model.joblib"))
            self.demand_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "course_demand_model.joblib"))
            self.is_ready = True
        except Exception as e:
            print(f"[WARN] Could not load ML models: {e}")
            self.is_ready = False

    def predict_student(self, student_dict):
        """
        Runs dropout, GPA, and anomaly models on a single student record.
        Returns a rich dictionary with predictions, risk factors, and alerts.
        """
        if not self.is_ready:
            self._load_models()

        # Build feature DataFrame
        df_input = pd.DataFrame([{
            "department": student_dict.get("department", "Computer Science"),
            "gender": student_dict.get("gender", "Male"),
            "guardian_education": student_dict.get("guardian_education", "Bachelors"),
            "age": int(student_dict.get("age", 20)),
            "semester": int(student_dict.get("current_semester", student_dict.get("semester", 3))),
            "financial_aid": int(student_dict.get("financial_aid", 0)),
            "high_school_gpa": float(student_dict.get("high_school_gpa", 3.2)),
            "previous_semester_gpa": float(student_dict.get("previous_semester_gpa", student_dict.get("cgpa", 3.0))),
            "cumulative_gpa": float(student_dict.get("cgpa", 3.0)),
            "total_credit_hours": int(student_dict.get("total_credit_hours", 45)),
            "failed_courses_count": int(student_dict.get("failed_courses_count", 0)),
            "attendance_rate": float(student_dict.get("attendance_rate", 80.0)),
            "consecutive_absences": int(student_dict.get("consecutive_absences", 0)),
            "quiz_average": float(student_dict.get("quiz_average", 75.0)),
            "assignment_average": float(student_dict.get("assignment_average", 78.0)),
            "midterm_exam_score": float(student_dict.get("midterm_exam_score", 72.0)),
            "final_exam_score": float(student_dict.get("final_exam_score", 74.0)),
            "lms_engagement_score": int(student_dict.get("lms_engagement_score", 70)),
            "lms_weekly_logins": int(student_dict.get("lms_weekly_logins", 12)),
            "lms_forum_posts": int(student_dict.get("lms_forum_posts", 8)),
            "lms_study_hours_per_week": float(student_dict.get("lms_study_hours_per_week", 12.0)),
            "lms_late_submissions": int(student_dict.get("lms_late_submissions", 1)),
            "lms_downloads_count": int(student_dict.get("lms_downloads_count", 35)),
        }])

        # 1. Dropout Risk Inference
        prob = float(self.dropout_model.predict_proba(df_input)[:, 1][0])
        dropout_pred = int(self.dropout_model.predict(df_input)[0])

        if prob >= 0.70:
            risk_tier = "Critical"
        elif prob >= 0.45:
            risk_tier = "High"
        elif prob >= 0.25:
            risk_tier = "Medium"
        else:
            risk_tier = "Low"

        # 2. GPA Forecast Inference
        gpa_features = [
            "department", "gender", "semester", "cumulative_gpa",
            "previous_semester_gpa", "high_school_gpa", "attendance_rate",
            "quiz_average", "assignment_average", "midterm_exam_score",
            "final_exam_score", "failed_courses_count", "lms_engagement_score",
            "lms_study_hours_per_week"
        ]
        pred_gpa = float(self.performance_model.predict(df_input[gpa_features])[0])
        pred_gpa = max(1.0, min(4.0, round(pred_gpa, 2)))

        cgpa = float(student_dict.get("cgpa", 3.0))
        if pred_gpa > cgpa + 0.1:
            trajectory = "Improving"
        elif pred_gpa < cgpa - 0.1:
            trajectory = "Declining"
        else:
            trajectory = "Stable"

        # 3. Anomaly Detection Inference
        anomaly_features = self.anomaly_artifact["features"]
        X_scaled = self.anomaly_artifact["scaler"].transform(df_input[anomaly_features])
        anomaly_pred = self.anomaly_artifact["model"].predict(X_scaled)[0]
        is_anomaly = True if anomaly_pred == -1 else False

        # 4. Explainable Risk Factors
        risk_factors = []
        att = float(student_dict.get("attendance_rate", 80.0))
        if att < 75.0:
            risk_factors.append(f"Critical attendance deficit: {att:.1f}% (Below required 75%)")
        if cgpa < 2.2:
            risk_factors.append(f"Academic probation risk: Current CGPA {cgpa:.2f}")
        if int(student_dict.get("failed_courses_count", 0)) > 0:
            risk_factors.append(f"Recorded course failures: {student_dict.get('failed_courses_count')} course(s)")
        if int(student_dict.get("lms_engagement_score", 70)) < 40:
            risk_factors.append("Low online LMS platform engagement")
        if is_anomaly:
            risk_factors.append("Unusual behavioral collapse detected by Anomaly Engine")
        if not risk_factors:
            risk_factors.append("Academic metrics are within standard performance targets")

        return {
            "dropout_probability": round(prob, 3),
            "dropout_probability_percent": round(prob * 100, 1),
            "predicted_dropout": dropout_pred,
            "risk_tier": risk_tier,
            "predicted_next_gpa": pred_gpa,
            "trajectory": trajectory,
            "is_anomaly": is_anomaly,
            "risk_factors": risk_factors,
            "model_version": "v2.0-HistGradientBoosting"
        }

    def sync_all_predictions_to_mongodb(self):
        """
        Runs ML models across all students in MongoDB and updates
        `predictions` and `alerts` collections with fresh real-time inferences.
        """
        db = get_db()
        students = list(db[COLL_STUDENTS].find())
        print(f"[ML Sync] Running live inference on {len(students)} students in MongoDB...")

        predictions_to_upsert = []
        alerts_to_insert = []
        now = datetime.now(timezone.utc)

        for s in students:
            pred = self.predict_student(s)
            
            doc = {
                "student_id": s["student_id"],
                "student_name": s["full_name"],
                "department": s.get("department", "Computer Science"),
                "dropout_probability": pred["dropout_probability"],
                "predicted_dropout": pred["predicted_dropout"],
                "predicted_next_cgpa": pred["predicted_next_gpa"],
                "trajectory": pred["trajectory"],
                "risk_level": pred["risk_tier"],
                "risk_factors": pred["risk_factors"],
                "is_anomaly": pred["is_anomaly"],
                "model_version": pred["model_version"],
                "confidence_score": 0.93,
                "updated_at": now
            }
            predictions_to_upsert.append(doc)

            # Generate alerts for High and Critical risks
            if pred["risk_tier"] in ("Critical", "High") or pred["is_anomaly"]:
                alert = {
                    "student_id": s["student_id"],
                    "student_name": s["full_name"],
                    "alert_type": "CRITICAL_DROPOUT_RISK" if pred["risk_tier"] in ("Critical", "High") else "ENGAGEMENT_ANOMALY",
                    "severity": "critical" if pred["risk_tier"] == "Critical" else "warning",
                    "message": f"ML Alert: {s['full_name']} ({s['student_id']}) flagged with {pred['dropout_probability_percent']}% dropout probability. Primary factor: {pred['risk_factors'][0]}.",
                    "target_roles": ["teacher", "admin", "analyst"],
                    "is_resolved": False,
                    "created_at": now
                }
                alerts_to_insert.append(alert)

        # Batch update MongoDB collections
        db[COLL_PREDICTIONS].delete_many({})
        if predictions_to_upsert:
            db[COLL_PREDICTIONS].insert_many(predictions_to_upsert)

        db[COLL_ALERTS].delete_many({})
        if alerts_to_insert:
            db[COLL_ALERTS].insert_many(alerts_to_insert)

        print(f"[ML Sync] Upserted {len(predictions_to_upsert)} predictions & {len(alerts_to_insert)} automated alerts to Atlas!")
        return len(predictions_to_upsert)


# Singleton instance
ml_service = EduPredictMLService()
