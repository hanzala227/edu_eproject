"""
EduPredict - Big Data Parquet Dataset Generator
Generates large-scale, high-fidelity educational data stored in Apache Parquet format.
No heavy Hadoop daemon needed; pure vectorized NumPy and Apache PyArrow.
"""

import os
import time
import numpy as np
import pandas as pd

np.random.seed(42)

def generate_student_performance_parquet(num_records=100000, output_path="data/parquet/student_performance_bigdata.parquet"):
    print(f"[1/2] Generating {num_records:,} student performance training records...")
    start_time = time.time()

    departments = ["Computer Science", "Software Engineering", "Data Science", "Information Technology", "Business Analytics"]
    dept_weights = [0.35, 0.25, 0.15, 0.15, 0.10]
    
    genders = ["Male", "Female"]
    gender_weights = [0.55, 0.45]

    guardian_edu = ["High School", "Bachelors", "Masters", "Doctorate"]
    guardian_weights = [0.30, 0.45, 0.20, 0.05]

    # Vectorized synthesis
    student_indices = np.arange(1, num_records + 1)
    student_ids = [f"STU-HIST-{i:06d}" for i in student_indices]
    
    dept_choices = np.random.choice(departments, size=num_records, p=dept_weights)
    gender_choices = np.random.choice(genders, size=num_records, p=gender_weights)
    guardian_choices = np.random.choice(guardian_edu, size=num_records, p=guardian_weights)
    
    semesters = np.random.randint(1, 9, size=num_records)
    ages = 18 + (semesters // 2) + np.random.randint(-1, 3, size=num_records)
    ages = np.clip(ages, 18, 28)
    
    financial_aid = np.random.choice([0, 1], size=num_records, p=[0.70, 0.30])
    high_school_gpa = np.clip(np.random.normal(3.2, 0.5, size=num_records), 2.0, 4.0)
    
    # Latent academic ability factor (Normal distribution)
    academic_aptitude = np.random.normal(0, 1, size=num_records)
    
    # Attendance rate (correlated with aptitude and motivation)
    base_attendance = 78 + (10 * academic_aptitude) + np.random.normal(0, 8, size=num_records)
    attendance_rate = np.clip(np.round(base_attendance, 1), 30.0, 100.0)
    
    # Consecutive absences
    consecutive_absences = np.where(attendance_rate < 60, np.random.randint(4, 12, size=num_records), np.random.randint(0, 3, size=num_records))
    
    # GPA (correlated with aptitude and attendance)
    cumulative_gpa = 2.0 + (0.35 * (academic_aptitude + 2)) + (0.015 * (attendance_rate - 50)) + np.random.normal(0, 0.2, size=num_records)
    cumulative_gpa = np.clip(np.round(cumulative_gpa, 2), 1.5, 4.0)
    
    previous_semester_gpa = np.clip(np.round(cumulative_gpa + np.random.normal(0, 0.25, size=num_records), 2), 1.0, 4.0)
    
    # Course components
    quiz_avg = np.clip(np.round(cumulative_gpa * 23.5 + np.random.normal(0, 6, size=num_records), 1), 20.0, 100.0)
    assignment_avg = np.clip(np.round(cumulative_gpa * 24.0 + np.random.normal(0, 5, size=num_records), 1), 25.0, 100.0)
    midterm_score = np.clip(np.round(cumulative_gpa * 23.0 + np.random.normal(0, 7, size=num_records), 1), 20.0, 100.0)
    final_score = np.clip(np.round(cumulative_gpa * 23.8 + np.random.normal(0, 6, size=num_records), 1), 20.0, 100.0)
    
    # Course failures
    failed_courses = np.where(cumulative_gpa < 2.0, np.random.randint(2, 6, size=num_records),
                    np.where(cumulative_gpa < 2.5, np.random.randint(1, 3, size=num_records), 0))
    
    # Total credit hours attempted
    total_credits = (semesters * 16) + np.random.randint(-4, 5, size=num_records)
    total_credits = np.clip(total_credits, 12, 140)
    
    # LMS engagement metrics
    lms_score = np.clip(np.round(attendance_rate * 0.7 + (cumulative_gpa / 4.0) * 30 + np.random.normal(0, 8, size=num_records)), 10, 100).astype(int)
    weekly_logins = np.clip((lms_score * 0.25 + np.random.randint(0, 5, size=num_records)).astype(int), 1, 35)
    forum_posts = np.clip((lms_score * 0.20 + np.random.randint(0, 4, size=num_records)).astype(int), 0, 40)
    study_hours = np.clip(np.round(lms_score * 0.22 + np.random.normal(0, 2.5, size=num_records), 1), 1.0, 30.0)
    late_submissions = np.where(lms_score < 40, np.random.randint(2, 8, size=num_records), np.random.randint(0, 2, size=num_records))
    downloads_count = np.clip((lms_score * 0.6 + np.random.randint(5, 20, size=num_records)).astype(int), 5, 95)
    
    # Anomaly indicator: Sudden drop in attendance or extreme inactivity despite normal past grades
    is_anomaly = np.where((cumulative_gpa > 2.8) & (attendance_rate < 55), 1,
                 np.where((weekly_logins <= 1) & (semesters >= 2), 1, 0))

    # Real Dropout Probability (Logistic ground-truth simulation)
    # Risk factors: Low attendance (<75), Low GPA (<2.3), Failed courses (>0), Low LMS (<40)
    logit = (
        -3.8
        + (2.5 * (cumulative_gpa < 2.2))
        + (2.2 * (attendance_rate < 70.0))
        + (1.5 * (attendance_rate < 80.0))
        + (1.8 * (failed_courses >= 2))
        + (1.2 * (failed_courses == 1))
        + (1.6 * (lms_score < 45))
        + (1.0 * (consecutive_absences >= 4))
        - (1.2 * (cumulative_gpa > 3.2))
        - (1.0 * (attendance_rate > 90.0))
    )
    dropout_prob = 1.0 / (1.0 + np.exp(-logit))
    dropout_prob = np.clip(np.round(dropout_prob, 3), 0.01, 0.99)
    
    # Target 1: Binary dropout
    dropped_out = (np.random.rand(num_records) < dropout_prob).astype(int)
    
    # Target 2: Next semester GPA (Regression target)
    next_semester_gpa = cumulative_gpa + (0.15 * academic_aptitude) + (0.005 * (attendance_rate - 75)) - (0.15 * failed_courses) + np.random.normal(0, 0.15, size=num_records)
    next_semester_gpa = np.clip(np.round(next_semester_gpa, 2), 1.0, 4.0)

    df = pd.DataFrame({
        "student_id": student_ids,
        "department": dept_choices,
        "gender": gender_choices,
        "guardian_education": guardian_choices,
        "age": ages.astype(np.int32),
        "semester": semesters.astype(np.int32),
        "financial_aid": financial_aid.astype(np.int32),
        "high_school_gpa": high_school_gpa.astype(np.float32),
        "previous_semester_gpa": previous_semester_gpa.astype(np.float32),
        "cumulative_gpa": cumulative_gpa.astype(np.float32),
        "total_credit_hours": total_credits.astype(np.int32),
        "failed_courses_count": failed_courses.astype(np.int32),
        "attendance_rate": attendance_rate.astype(np.float32),
        "consecutive_absences": consecutive_absences.astype(np.int32),
        "quiz_average": quiz_avg.astype(np.float32),
        "assignment_average": assignment_avg.astype(np.float32),
        "midterm_exam_score": midterm_score.astype(np.float32),
        "final_exam_score": final_score.astype(np.float32),
        "lms_engagement_score": lms_score.astype(np.int32),
        "lms_weekly_logins": weekly_logins.astype(np.int32),
        "lms_forum_posts": forum_posts.astype(np.int32),
        "lms_study_hours_per_week": study_hours.astype(np.float32),
        "lms_late_submissions": late_submissions.astype(np.int32),
        "lms_downloads_count": downloads_count.astype(np.int32),
        "is_anomaly": is_anomaly.astype(np.int32),
        "dropout_probability_groundtruth": dropout_prob.astype(np.float32),
        "dropped_out": dropped_out.astype(np.int32),
        "next_semester_gpa": next_semester_gpa.astype(np.float32),
    })

    # Save to Apache Parquet format (using snappy compression)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f" -> Generated {output_path}")
    print(f" -> Total Rows: {len(df):,}")
    print(f" -> Total Features: {df.shape[1]}")
    print(f" -> File Size on Disk: {file_size_mb:.2f} MB (High compression!)")
    print(f" -> Generation Time: {elapsed:.2f} seconds")
    return df


def generate_course_demand_parquet(output_path="data/parquet/course_demand_history.parquet"):
    print("[2/2] Generating historical course demand and enrollment trend Parquet file...")
    start_time = time.time()

    courses = [
        ("CS101", "Introduction to Programming", "Computer Science", 1),
        ("CS102", "Object Oriented Programming", "Computer Science", 2),
        ("CS201", "Data Structures & Algorithms", "Computer Science", 3),
        ("CS202", "Database Management Systems", "Computer Science", 4),
        ("CS301", "Operating Systems", "Computer Science", 5),
        ("CS302", "Computer Networks", "Computer Science", 6),
        ("SE201", "Software Engineering Concepts", "Software Engineering", 3),
        ("SE301", "Software Design & Architecture", "Software Engineering", 5),
        ("DS201", "Statistical Data Analysis", "Data Science", 3),
        ("DS301", "Machine Learning & Predictive Modeling", "Data Science", 5),
        ("DS401", "Big Data Technologies & Hadoop", "Data Science", 7),
        ("MATH101", "Calculus & Analytical Geometry", "Computer Science", 1),
        ("MATH201", "Linear Algebra & Probability", "Computer Science", 2),
        ("BA101", "Principles of Management", "Business Analytics", 1),
        ("BA201", "Business Intelligence & Reporting", "Business Analytics", 4),
    ]

    terms = [f"Term-{y}-{sem}" for y in range(2018, 2026) for sem in ["Spring", "Fall"]]
    
    rows = []
    for term_idx, term in enumerate(terms):
        for code, name, dept, sem in courses:
            base_enrollment = 70 + (sem * 10)
            growth = 1.0 + (term_idx * 0.04) # Steady academic growth year over year
            noise = np.random.normal(1.0, 0.08)
            actual_enrollment = int(base_enrollment * growth * noise)
            actual_enrollment = max(35, actual_enrollment)
            
            pass_rate = np.clip(np.round(np.random.normal(0.82, 0.06), 2), 0.60, 0.98)
            drop_rate = np.clip(np.round(1.0 - pass_rate - np.random.uniform(0.01, 0.05), 2), 0.02, 0.25)
            avg_gpa = np.clip(np.round(np.random.normal(2.95, 0.25), 2), 2.1, 3.8)
            sections = int(np.ceil(actual_enrollment / 40))

            rows.append({
                "term_code": term,
                "term_sequence": term_idx,
                "course_code": code,
                "course_name": name,
                "department": dept,
                "target_semester": sem,
                "enrolled_count": actual_enrollment,
                "sections_opened": sections,
                "pass_rate": pass_rate,
                "drop_rate": drop_rate,
                "average_gpa": avg_gpa
            })

    df_demand = pd.DataFrame(rows)
    df_demand.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

    elapsed = time.time() - start_time
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f" -> Generated {output_path}")
    print(f" -> Total Historical Course Terms: {len(df_demand):,}")
    print(f" -> File Size: {file_size_kb:.2f} KB")
    print(f" -> Generation Time: {elapsed:.2f} seconds")
    return df_demand


if __name__ == "__main__":
    generate_student_performance_parquet(num_records=100000)
    generate_course_demand_parquet()
    print("=" * 60)
    print("ALL BIG DATA PARQUET FILES GENERATED SUCCESSFULLY!")
    print("=" * 60)
