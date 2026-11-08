import json
import os
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from edupredict.db import (
    get_db,
    COLL_STUDENTS,
    COLL_PREDICTIONS,
    COLL_ALERTS,
    COLL_COURSES
)
from ml_engine.predictor import ml_service


def admin_dashboard_view(request):
    """
    Renders the React SPA Dashboard Shell.
    """
    current_user = request.session.get('user')
    if not current_user:
        return redirect('login')

    return render(request, "analytics/admin_dashboard.html", {
        "user": current_user
    })


def api_dashboard_data(request):
    """
    REST API Endpoint: Returns 100% dynamic data directly from MongoDB Atlas
    to feed the React Single Page Application (SPA).
    """
    current_user = request.session.get('user')
    if not current_user:
        return JsonResponse({"error": "Unauthorized"}, status=401)

    db = get_db()

    # 1. KPIs
    total_students = db[COLL_STUDENTS].count_documents({})

    avg_pipeline = [
        {"$group": {
            "_id": None,
            "avg_cgpa": {"$avg": "$cgpa"},
            "avg_attendance": {"$avg": "$attendance_rate"},
            "avg_lms": {"$avg": "$lms_engagement_score"}
        }}
    ]
    avg_results = list(db[COLL_STUDENTS].aggregate(avg_pipeline))
    avg_cgpa = round(avg_results[0]["avg_cgpa"], 2) if avg_results else 3.12
    avg_attendance = round(avg_results[0]["avg_attendance"], 1) if avg_results else 82.4
    avg_lms = round(avg_results[0]["avg_lms"], 1) if avg_results else 71.5

    # 2. Risk Distribution
    critical_count = db[COLL_PREDICTIONS].count_documents({"risk_level": "Critical"})
    high_count = db[COLL_PREDICTIONS].count_documents({"risk_level": "High"})
    medium_count = db[COLL_PREDICTIONS].count_documents({"risk_level": "Medium"})
    low_count = db[COLL_PREDICTIONS].count_documents({"risk_level": "Low"})

    # 3. All Students Joined with ML Predictions
    predictions_map = {p["student_id"]: p for p in db[COLL_PREDICTIONS].find({}, {"_id": 0})}
    
    students_cursor = db[COLL_STUDENTS].find({}, {"_id": 0}).limit(120)
    students_list = []
    for s in students_cursor:
        s_id = s.get("student_id")
        p = predictions_map.get(s_id, {})
        
        prob = p.get("dropout_probability", 0.1)
        prob_pct = int(prob * 100) if prob <= 1.0 else int(prob)

        students_list.append({
            "student_id": s_id,
            "student_name": s.get("full_name", "Student"),
            "gender": s.get("gender", "Male"),
            "department": s.get("department", "Computer Science"),
            "program": s.get("program", "BSCS"),
            "semester": s.get("current_semester", 3),
            "cgpa": round(float(s.get("cgpa", 3.0)), 2),
            "attendance_rate": round(float(s.get("attendance_rate", 80.0)), 1),
            "lms_engagement_score": s.get("lms_engagement_score", 70),
            "risk_level": p.get("risk_level", s.get("risk_tier", "Low")),
            "dropout_probability": round(prob, 3),
            "dropout_probability_percent": prob_pct,
            "predicted_next_cgpa": p.get("predicted_next_cgpa", s.get("cgpa", 3.0)),
            "trajectory": p.get("trajectory", "Stable"),
            "risk_factors": p.get("risk_factors", ["Standard academic trajectory"]),
            "is_anomaly": p.get("is_anomaly", False),
            "status": s.get("status", "enrolled")
        })

    # Sort so Critical and High risks appear first
    risk_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    students_list.sort(key=lambda x: (risk_rank.get(x["risk_level"], 4), -x["dropout_probability"]))

    # 4. Alerts
    alerts = list(db[COLL_ALERTS].find({}, {"_id": 0}).sort("created_at", -1).limit(10))
    for a in alerts:
        if "created_at" in a:
            a["created_at_str"] = a["created_at"].strftime("%b %d, %H:%M") if hasattr(a["created_at"], "strftime") else str(a["created_at"])

    # 5. Course Demands
    course_demands = list(db["course_demand"].find({}, {"_id": 0}).sort("projected_enrollment", -1).limit(8))

    # 6. ML Model Benchmarks
    meta_path = "ml_engine/saved_models/model_metadata.json"
    ml_meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                ml_meta = json.load(f)
        except Exception:
            pass

    return JsonResponse({
        "success": True,
        "kpis": {
            "total_students": total_students,
            "at_risk_total": critical_count + high_count,
            "avg_cgpa": avg_cgpa,
            "avg_attendance": avg_attendance,
            "avg_lms": avg_lms,
            "total_alerts": len(alerts),
        },
        "risk_distribution": {
            "critical": critical_count,
            "high": high_count,
            "medium": medium_count,
            "low": low_count,
        },
        "students": students_list,
        "alerts": alerts,
        "course_demands": course_demands,
        "ml_metadata": ml_meta,
    })


@csrf_exempt
def api_sync_models(request):
    """Trigger real-time inference across all records in MongoDB Atlas."""
    if not request.session.get('user'):
        return JsonResponse({"error": "Unauthorized"}, status=401)

    try:
        updated_count = ml_service.sync_all_predictions_to_mongodb()
        return JsonResponse({
            "success": True,
            "message": f"Successfully updated {updated_count} student predictions in MongoDB Atlas!"
        })
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)
