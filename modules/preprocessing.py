"""
Data Preprocessing Pipeline Module for EduPredict AI.
Cleans outliers, handles missing values, encodes categories safely, and removes demographic bias features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from . import config

def engineer_features(df, train_stats=None):
    df = df.copy()

    # Engagement score
    if "assignmentcompletion" in df.columns and \
       "discussions" in df.columns:
        df["engagement_score"] = (
            df["assignmentcompletion"] * 0.7 +
            df["discussions"]          * 0.3
        )

    # Risk flag — binary indicator for low attendance
    if "attendance" in df.columns:
        df["low_attendance_flag"] = (
            df["attendance"] < 75
        ).astype(int)

    return df

class PreprocessingPipeline:
    """
    Fits and transforms DataFrames for ML training and predictions.
    Excludes config.BIAS_COLS and config.EXCLUDE_COLS automatically.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.medians = {}
        self.modes = {}
        self.iqr_bounds = {}
        self.numeric_features = []
        self.categorical_features = []
        self.features = []

    def fit(self, df):
        """
        Fits scalers, encoders, median values, and IQR bounds on the DataFrame.
        Automatically excludes BIAS_COLS and EXCLUDE_COLS.
        """
        # Ethical AI Compliance: Guard against demographic bias factors
        detected_bias_cols = [c for c in df.columns if c.lower().strip() in config.BIAS_COLS]
        if detected_bias_cols:
            print(f"[Ethical AI Guardrail] Discarded demographic bias features: {detected_bias_cols}")

        # Filter features based on exclusion list
        feature_cols = [
            c for c in df.columns
            if c.lower().strip() not in config.BIAS_COLS
            and c not in config.EXCLUDE_COLS
        ]
        
        # Also ensure target column is not treated as input feature
        if "finalgrade" in feature_cols:
            feature_cols.remove("finalgrade")
            
        X = df[feature_cols].copy()
        
        # Categorical vs Numeric columns mapping
        self.numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.features = X.columns.tolist()
        
        # 1. Save Medians, Modes, and IQR limits
        for col in self.numeric_features:
            self.medians[col] = float(X[col].median())
            # IQR limits for outlier handling
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.iqr_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            
        for col in self.categorical_features:
            self.modes[col] = str(X[col].mode()[0]) if not X[col].mode().empty else 'Unknown'
            
        # 2. Fit Categorical Encoders
        for col in self.categorical_features:
            le = LabelEncoder()
            # Map nan values to mode
            temp_col = X[col].fillna(self.modes[col]).astype(str)
            le.fit(temp_col)
            self.label_encoders[col] = le
            
        # 3. Fit standard scaler
        X_imputed = pd.DataFrame(index=df.index)
        for col in self.numeric_features:
            X_imputed[col] = pd.to_numeric(X[col], errors='coerce').fillna(self.medians[col])
            # Clip outliers (except for attendance to preserve impact)
            if col != "attendance":
                low, high = self.iqr_bounds[col]
                X_imputed[col] = X_imputed[col].clip(low, high)
            
        for col in self.categorical_features:
            le = self.label_encoders[col]
            temp_col = X[col].fillna(self.modes[col]).astype(str)
            classes_set = set(le.classes_)
            temp_col_safe = temp_col.map(lambda s: s if s in classes_set else le.classes_[0])
            X_imputed[col] = le.transform(temp_col_safe)
            
        if len(self.features) > 0:
            self.scaler.fit(X_imputed[self.features])
            
        return self

    def transform(self, df):
        """
        Transforms input DataFrame. Missing values are filled with fitted medians/modes.
        Outputs scaled features.
        """
        # Ensure we have all target features
        feature_cols = self.features
        X = pd.DataFrame(index=df.index)
        
        for col in feature_cols:
            if col in df.columns:
                X[col] = df[col].copy()
            else:
                # Assign default values for columns missing entirely
                if col in self.numeric_features:
                    X[col] = self.medians.get(col, 0.0)
                else:
                    X[col] = self.modes.get(col, 'Unknown')
                    
        # 1. Convert, Impute and Clip
        for col in self.numeric_features:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(self.medians[col])
            if col in self.iqr_bounds and col != "attendance":
                low, high = self.iqr_bounds[col]
                X[col] = X[col].clip(low, high)
                
        for col in self.categorical_features:
            X[col] = X[col].fillna(self.modes[col]).astype(str)
            
        # 2. Encode categorical columns safely (no unseen exception)
        for col in self.categorical_features:
            le = self.label_encoders[col]
            classes_set = set(le.classes_)
            
            # Map unseen level to first class (usually most frequent or default)
            X[col] = X[col].map(lambda s: s if s in classes_set else le.classes_[0])
            X[col] = le.transform(X[col])
            
        # 3. Scale numeric features
        if len(self.features) > 0:
            X_scaled = pd.DataFrame(self.scaler.transform(X), columns=self.features, index=df.index)
            return X_scaled
        else:
            return X

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
