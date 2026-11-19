"""
Validation Engine Module for EduPredict AI.
Handles schema checks, type consistency checks, outlier/bound anomaly detection,
and statistical drift warnings between baseline and uploaded cohorts.
"""

import pandas as pd
import numpy as np
from . import config

def normalize_columns(df):
    """
    Strips whitespace, lowercases, and removes spaces from column names.
    """
    clean_df = df.copy()
    clean_df.columns = (
        clean_df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "")
        .str.replace("_", "")
    )
    return clean_df

def map_columns(df):
    """
    Maps normalized column names to internal standard keys defined in config.
    """
    # First normalize column names
    df_normalized = normalize_columns(df)
    
    # Generate renaming mapping based on config
    rename_dict = {}
    for col in df_normalized.columns:
        if col in config.COLUMN_MAPPING:
            rename_dict[col] = config.COLUMN_MAPPING[col]
            
    df_mapped = df_normalized.rename(columns=rename_dict)
    return df_mapped

def validate_dataset(df, is_training=False):
    """
    Performs data consistency checks:
    - Missing academic features
    - Invalid value bounds
    - Datatype mismatches
    - Duplicate student IDs
    - Missing target column
    
    Returns a dictionary report:
      {
         'status': 'SUCCESS' | 'WARNING' | 'ERROR',
         'errors': list of str,
         'warnings': list of str,
         'quality_score': float (0 - 100),
         'mapped_df': pd.DataFrame,
         'target_status': 'present' | 'missing'
      }
    """
    errors = []
    warnings = []
    quality_score = 100.0
    
    # 1. Normalize and Map columns
    mapped_df = map_columns(df)
    
    # 2. Check student ID column
    id_col = None
    for c in ["studentid", "student_id"]:
        if c in mapped_df.columns:
            id_col = c
            break
            
    if not id_col:
        warnings.append("Student ID column (e.g. 'StudentID') not detected. Row index will be used to generate IDs.")
        quality_score -= 10.0
    else:
        # Check duplicates
        dup_count = mapped_df[id_col].duplicated().sum()
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate student ID records in dataset.")
            quality_score -= 15.0

    # 3. Check Target Column
    target_col = "finalgrade"
    target_status = "present"
    if target_col not in mapped_df.columns:
        target_status = "missing"
        if is_training:
            errors.append(f"Retraining requires target column '{target_col}' but it was not found.")
            quality_score -= 30.0
        else:
            warnings.append(f"Target column '{target_col}' is missing. (Safe: Prediction-only mode).")
            quality_score -= 5.0
            
    # 4. Check Required Academic Features
    required_features = ["studyhours", "attendance", "examscore", "assignmentcompletion", "discussions"]
    missing_features = [f for f in required_features if f not in mapped_df.columns]
    
    if missing_features:
        if is_training:
            errors.append(f"Training cannot proceed without critical features: {missing_features}")
            quality_score -= 40.0
        else:
            warnings.append(f"Missing core features: {missing_features}. Inference will fall back to default values.")
            quality_score -= 20.0
            
    # 5. Check Datatype mismatches and Null values
    for col in mapped_df.columns:
        # Check nulls
        null_count = mapped_df[col].isnull().sum()
        if null_count > 0:
            warnings.append(f"Column '{col}' has {null_count} missing values. (Will be handled by preprocessing imputation).")
            quality_score -= 5.0
            
        # Check datatypes for academic features
        if col in required_features:
            if not pd.api.types.is_numeric_dtype(mapped_df[col]):
                warnings.append(f"Feature '{col}' is not numeric. Preprocessing will attempt forced numeric conversion.")
                quality_score -= 15.0
                
            # Check range boundaries (anomaly detection)
            if pd.api.types.is_numeric_dtype(mapped_df[col]):
                min_val = mapped_df[col].min()
                max_val = mapped_df[col].max()
                
                if col == "attendance" and (min_val < 0 or max_val > 100):
                    warnings.append("Attendance contains anomalous values outside 0-100% range.")
                    quality_score -= 10.0
                elif col == "studyhours" and (min_val < 0 or max_val > 168):
                    warnings.append("Weekly study hours contains anomalous values outside 0-168 range.")
                    quality_score -= 10.0
                elif col == "examscore" and (min_val < 0 or max_val > 100):
                    warnings.append("Exam Score contains anomalous values outside 0-100 range.")
                    quality_score -= 10.0
                elif col == "assignmentcompletion" and (min_val < 0 or max_val > 100):
                    warnings.append("Assignment completion contains anomalous values outside 0-100% range.")
                    quality_score -= 10.0

    # Ensure quality score does not drop below 0
    quality_score = max(0.0, quality_score)
    
    # Establish overall status
    if errors:
        status = "ERROR"
    elif warnings:
        status = "WARNING"
    else:
        status = "SUCCESS"
        
    return {
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "quality_score": round(quality_score, 1),
        "mapped_df": mapped_df,
        "target_status": target_status
    }

def check_model_drift(training_df, upload_df):
    """
    Compares the upload dataset distributions with training data.
    If the average value of academic features differs by > 20%, it returns a drift warning.
    
    Returns a drift report:
      {
        'drift_detected': bool,
        'features_drift': dict of feature_name -> { 'drift': bool, 'train_mean': float, 'upload_mean': float, 'pct_diff': float }
      }
    """
    features = ["studyhours", "attendance", "examscore", "assignmentcompletion", "discussions"]
    features_drift = {}
    drift_detected = False
    
    # Normalize and map upload dataset columns
    upload_mapped = map_columns(upload_df)
    train_mapped = map_columns(training_df)
    
    for f in features:
        if f in upload_mapped.columns and f in train_mapped.columns:
            # Force numeric
            train_vals = pd.to_numeric(train_mapped[f], errors='coerce').fillna(train_mapped[f].median(numeric_only=True))
            upload_vals = pd.to_numeric(upload_mapped[f], errors='coerce').fillna(upload_mapped[f].median(numeric_only=True))
            
            train_mean = float(train_vals.mean())
            upload_mean = float(upload_vals.mean())
            
            if train_mean == 0:
                pct_diff = 0.0
            else:
                pct_diff = round(((upload_mean - train_mean) / train_mean) * 100.0, 2)
                
            # Define structural drift threshold as 20%
            has_drift = abs(pct_diff) > 20.0
            if has_drift:
                drift_detected = True
                
            features_drift[f] = {
                "drift": has_drift,
                "train_mean": round(train_mean, 2),
                "upload_mean": round(upload_mean, 2),
                "pct_diff": pct_diff
            }
            
    return {
        "drift_detected": drift_detected,
        "features_drift": features_drift
    }
