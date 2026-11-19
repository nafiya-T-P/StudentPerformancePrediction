"""
Configuration Module for EduPredict AI.
Defines database paths, model output paths, feature exclusion lists,
and column renaming maps for pandas dataset ingestion.
"""

import os

# Database & Storage Configuration
# Go up one level from modules/ to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "database.db")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "student_model.pkl")
PREDICTIONS_CSV = os.path.join(BASE_DIR, "reports", "academic_predictions.csv")
ACADEMIC_CSV_DEFAULT = os.path.join(BASE_DIR, "datasets", "final_academic_dataset.csv")
SURVEY_CSV_DEFAULT = os.path.join(BASE_DIR, "datasets", "survey_dataset.csv")
# SKIP — still referenced: datasets/merged_behavioral_intelligence.csv
BEHAVIORAL_INTELLIGENCE_CSV = os.path.join(BASE_DIR, "datasets", "merged_behavioral_intelligence.csv")

# Institutional Database Migration Configuration
DB_TYPE = "sqlite"  # Set to "mysql" or "postgresql" for enterprise deployment
DB_HOST = "localhost"
DB_PORT = 3306      # 3306 for MySQL, 5432 for PostgreSQL
DB_NAME = "edupredict"
DB_USER = "root"
DB_PASSWORD = ""

# Demographic bias factors to exclude from ML modeling for ethical AI compliance
BIAS_COLS = {"age", "gender"}

# Exclude target and IDs from features to avoid demographic and identity leakage
EXCLUDE_COLS = {"finalgrade", "studentid", "student_id", "studentname", "student_name", "eligibility", "ia_marks", "id", "name", "timestamp"}

# Render & Performance Settings
PAGINATION_LIMIT = 100
SHAP_MAX_SAMPLE = 500

# Prediction Confidence Thresholds (on Probability %)
CONFIDENCE_HIGH = 80.0
CONFIDENCE_LOW = 40.0
CONFIDENCE_HIGH_RISK = 20.0

# Weighted Wellness System Weights
WELLNESS_WEIGHTS = {
    "motivation": 0.40,
    "lifestyle": 0.35,
    "stress": 0.25,
    "statement": "The EduPredict AI system prioritizes stability, scalability, explainability, fairness, and institutional adaptability over pure model complexity."
}

# Retraining safeguards thresholds
MIN_QUALITY_SCORE = 70.0
MIN_STUDENTS_REQUIRED = 50
MAX_CLASS_IMBALANCE_RATIO = 9.0  # Max ratio of majority/minority class

# Automatic mapping dictionary for column normalization
COLUMN_MAPPING = {
    "studyhours": "studyhours",
    "study_hours": "studyhours",
    "studytime": "studyhours",
    "study_time": "studyhours",
    "attendance": "attendance",
    "attendance_percentage": "attendance",
    "attendancepercentage": "attendance",
    "examscore": "examscore",
    "exam_score": "examscore",
    "internal_marks": "examscore",
    "marks": "examscore",
    "assignmentcompletion": "assignmentcompletion",
    "assignment_completion": "assignmentcompletion",
    "discussions": "discussions",
    "discussion": "discussions",
    "participation": "discussions",
    "onlinecourses": "onlinecourses",
    "online_courses": "onlinecourses",
    "stresslevel": "stresslevel",
    "stress_level": "stresslevel",
    "motivation": "motivation",
    "resources": "resources",
    "resource_access": "resources",
    "extracurricular": "extracurricular",
    "extracurricular_activities": "extracurricular",
    "internet": "internet",
    "internet_access": "internet",
    "learningstyle": "learningstyle",
    "learning_style": "learningstyle",
    "edutech": "edutech",
    "edutech_usage": "edutech",
    "studentid": "studentid",
    "student_id": "studentid",
    "id": "studentid",
    "finalgrade": "finalgrade",
    "final_grade": "finalgrade",
    "grade": "finalgrade",
    "class": "class",
    "semester": "semester",
    "class_semester": "semester",
    "classsemester": "semester"
}

# Mock Mode Configuration
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

# Debug Mode Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

