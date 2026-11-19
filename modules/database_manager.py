"""
Database Manager Module for EduPredict AI.
Handles transactional logging of model runs, batch student predictions,
and dataset uploads/metadata histories across SQLite, MySQL, or PostgreSQL.
Uses a factory pattern to select the active institutional database backend.
"""

from abc import ABC, abstractmethod
import sqlite3
import os
import pandas as pd
from contextlib import contextmanager
from datetime import datetime
from . import config

class BaseDatabaseManager(ABC):
    """
    Abstract Base Class for EduPredict AI database operations.
    Enforces a consistent interface across SQLite, MySQL, and PostgreSQL.
    """
    @abstractmethod
    def initialize_database(self):
        """Initializes database schema and tables if they do not exist."""
        pass

    @abstractmethod
    def save_model_run(self, run_id, model_type, metrics):
        """Saves model execution runs and accuracy/F1 evaluation metrics."""
        pass

    @abstractmethod
    def save_predictions(self, run_id, df):
        """Bulk records generated student predictions to the database."""
        pass

    @abstractmethod
    def fetch_student_history(self, student_id):
        """Fetches all past predictions for a specific student, ordered chronologically."""
        pass

    @abstractmethod
    def save_dataset_metadata(self, filename, row_count, col_count, quality_score, model_version):
        """Logs dataset uploaded metadata histories."""
        pass

    @abstractmethod
    def get_latest_dataset_metadata(self):
        """Returns the most recently logged dataset upload metadata."""
        pass

    @abstractmethod
    def get_all_dataset_metadata(self):
        """Returns all logged dataset upload metadata histories."""
        pass

    @abstractmethod
    def get_model_history(self):
        """Fetches all past model training runs."""
        pass


class SQLiteDatabaseManager(BaseDatabaseManager):
    """
    Concrete SQLite Database Manager.
    Used for lightweight local deployments using the file system database.
    """
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(config.DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_database(self):
        with self.get_connection() as conn:
            # Create model runs table to store training history
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_runs (
                    run_id TEXT PRIMARY KEY,
                    model_type TEXT,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    timestamp TEXT
                )
            """)

            # Create predictions table for student longitudinal tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    student_id TEXT,
                    prediction TEXT,
                    probability REAL,
                    risk_level TEXT,
                    confidence_level TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (run_id) REFERENCES model_runs (run_id)
                )
            """)

            # Create dataset upload version tracking table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT,
                    row_count INTEGER,
                    col_count INTEGER,
                    quality_score REAL,
                    model_version TEXT,
                    timestamp TEXT
                )
            """)

    def save_model_run(self, run_id, model_type, metrics):
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO model_runs 
                   (run_id, model_type, accuracy, precision, recall, f1_score, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    model_type,
                    float(metrics.get("Accuracy", 0)),
                    float(metrics.get("Precision", 0)),
                    float(metrics.get("Recall", 0)),
                    float(metrics.get("F1-Score", 0)),
                    datetime.now().isoformat()
                )
            )

    def save_predictions(self, run_id, df):
        with self.get_connection() as conn:
            timestamp = datetime.now().isoformat()
            records = []
            for idx, row in df.iterrows():
                student_id = str(row.get("StudentID", row.get("studentid", f"STU{idx+1:04d}"))).strip().upper()
                prediction = str(row.get("Prediction", "Pass"))
                probability = float(row.get("Probability", 50.0))
                risk_level = str(row.get("Academic_Risk", row.get("risk_level", "Low Risk")))
                confidence_level = str(row.get("Confidence_Level", "Medium Confidence"))

                records.append((run_id, student_id, prediction, probability, risk_level, confidence_level, timestamp))

            conn.executemany(
                """INSERT INTO predictions 
                   (run_id, student_id, prediction, probability, risk_level, confidence_level, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                records
            )

    def fetch_student_history(self, student_id):
        student_id = str(student_id).strip().upper()
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT p.timestamp, p.run_id, p.prediction, p.probability, p.risk_level, 
                          p.confidence_level, r.model_type
                   FROM predictions p
                   LEFT JOIN model_runs r ON p.run_id = r.run_id
                   WHERE p.student_id = ?
                   ORDER BY p.timestamp ASC""",
                (student_id,)
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def save_dataset_metadata(self, filename, row_count, col_count, quality_score, model_version):
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO dataset_metadata 
                   (filename, row_count, col_count, quality_score, model_version, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (filename, int(row_count), int(col_count), float(quality_score), str(model_version), datetime.now().isoformat())
            )

    def get_latest_dataset_metadata(self):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM dataset_metadata ORDER BY timestamp DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_dataset_metadata(self):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM dataset_metadata ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_model_history(self):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM model_runs ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]


class RelationalDatabaseManager(BaseDatabaseManager):
    """
    SQLAlchemy/DBAPI-ready Database Manager for MySQL and PostgreSQL.
    Configured for enterprise institutional databases.
    """
    def __init__(self, db_type):
        self.db_type = db_type.lower()
        self.placeholder = "%s"  # Standard placeholder for MySQL/PostgreSQL DBAPI

    @contextmanager
    def get_connection(self):
        """
        Creates a connection pool to PostgreSQL or MySQL database.
        Falls back gracefully with detailed logs if institutional drivers are missing.
        """
        conn = None
        try:
            if self.db_type == "postgresql":
                import psycopg2
                conn = psycopg2.connect(
                    host=config.DB_HOST,
                    port=config.DB_PORT,
                    database=config.DB_NAME,
                    user=config.DB_USER,
                    password=config.DB_PASSWORD
                )
            else:  # MySQL
                import pymysql
                conn = pymysql.connect(
                    host=config.DB_HOST,
                    port=config.DB_PORT,
                    database=config.DB_NAME,
                    user=config.DB_USER,
                    password=config.DB_PASSWORD,
                    cursorclass=pymysql.cursors.DictCursor
                )
            yield conn
            conn.commit()
        except ImportError:
            # Standard developer environment fallback to SQLite database
            print(f"[Warning] Database driver for '{self.db_type}' not installed. Falling back to SQLite Manager.")
            # Yield from SQLite instead
            sqlite_mgr = SQLiteDatabaseManager()
            with sqlite_mgr.get_connection() as sqlite_conn:
                yield sqlite_conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()

    def initialize_database(self):
        """Creates institutional MySQL/PostgreSQL schemas."""
        # Using dialect-specific auto-increment rules
        identity_clause = "SERIAL" if self.db_type == "postgresql" else "INT AUTO_INCREMENT"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                # 1. Model runs table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS model_runs (
                        run_id VARCHAR(50) PRIMARY KEY,
                        model_type VARCHAR(100),
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        timestamp VARCHAR(50)
                    )
                """)
                # 2. Predictions table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id {identity_clause} PRIMARY KEY,
                        run_id VARCHAR(50),
                        student_id VARCHAR(50),
                        prediction VARCHAR(50),
                        probability REAL,
                        risk_level VARCHAR(50),
                        confidence_level VARCHAR(50),
                        timestamp VARCHAR(50)
                    )
                """)
                # 3. Dataset Metadata table
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS dataset_metadata (
                        id {identity_clause} PRIMARY KEY,
                        filename VARCHAR(255),
                        row_count INTEGER,
                        col_count INTEGER,
                        quality_score REAL,
                        model_version VARCHAR(50),
                        timestamp VARCHAR(50)
                    )
                """)
            except (sqlite3.OperationalError, AttributeError):
                # SQLite fallback handles queries internally
                pass

    def save_model_run(self, run_id, model_type, metrics):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = f"""INSERT INTO model_runs 
                           (run_id, model_type, accuracy, precision, recall, f1_score, timestamp)
                           VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, 
                                   {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})"""
                cursor.execute(query, (
                    run_id, model_type,
                    float(metrics.get("Accuracy", 0)),
                    float(metrics.get("Precision", 0)),
                    float(metrics.get("Recall", 0)),
                    float(metrics.get("F1-Score", 0)),
                    datetime.now().isoformat()
                ))
            except (sqlite3.OperationalError, AttributeError):
                # Fallback implementation executes SQLite style
                SQLiteDatabaseManager().save_model_run(run_id, model_type, metrics)

    def save_predictions(self, run_id, df):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            records = []
            for idx, row in df.iterrows():
                student_id = str(row.get("StudentID", row.get("studentid", f"STU{idx+1:04d}"))).strip().upper()
                prediction = str(row.get("Prediction", "Pass"))
                probability = float(row.get("Probability", 50.0))
                risk_level = str(row.get("Academic_Risk", row.get("risk_level", "Low Risk")))
                confidence_level = str(row.get("Confidence_Level", "Medium Confidence"))
                records.append((run_id, student_id, prediction, probability, risk_level, confidence_level, timestamp))
            
            try:
                query = f"""INSERT INTO predictions 
                           (run_id, student_id, prediction, probability, risk_level, confidence_level, timestamp)
                           VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, 
                                   {self.placeholder}, {self.placeholder}, {self.placeholder}, {self.placeholder})"""
                cursor.executemany(query, records)
            except (sqlite3.OperationalError, AttributeError):
                SQLiteDatabaseManager().save_predictions(run_id, df)

    def fetch_student_history(self, student_id):
        student_id = str(student_id).strip().upper()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = """SELECT p.timestamp, p.run_id, p.prediction, p.probability, p.risk_level, 
                                  p.confidence_level, r.model_type
                           FROM predictions p
                           LEFT JOIN model_runs r ON p.run_id = r.run_id
                           WHERE p.student_id = %s
                           ORDER BY p.timestamp ASC"""
                cursor.execute(query, (student_id,))
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
            except (sqlite3.OperationalError, AttributeError, sqlite3.ProgrammingError):
                return SQLiteDatabaseManager().fetch_student_history(student_id)

    def save_dataset_metadata(self, filename, row_count, col_count, quality_score, model_version):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                query = f"""INSERT INTO dataset_metadata 
                           (filename, row_count, col_count, quality_score, model_version, timestamp)
                           VALUES ({self.placeholder}, {self.placeholder}, {self.placeholder}, 
                                   {self.placeholder}, {self.placeholder}, {self.placeholder})"""
                cursor.execute(query, (filename, int(row_count), int(col_count), float(quality_score), str(model_version), datetime.now().isoformat()))
            except (sqlite3.OperationalError, AttributeError):
                SQLiteDatabaseManager().save_dataset_metadata(filename, row_count, col_count, quality_score, model_version)

    def get_latest_dataset_metadata(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM dataset_metadata ORDER BY timestamp DESC LIMIT 1")
                row = cursor.fetchone()
                return dict(row) if row else None
            except (sqlite3.OperationalError, AttributeError):
                return SQLiteDatabaseManager().get_latest_dataset_metadata()

    def get_all_dataset_metadata(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM dataset_metadata ORDER BY timestamp DESC")
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
            except (sqlite3.OperationalError, AttributeError):
                return SQLiteDatabaseManager().get_all_dataset_metadata()

    def get_model_history(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT * FROM model_runs ORDER BY timestamp DESC")
                rows = cursor.fetchall()
                return [dict(r) for r in rows]
            except (sqlite3.OperationalError, AttributeError):
                return SQLiteDatabaseManager().get_model_history()


class DatabaseManagerFactory:
    """
    Factory class that returns the active Database Manager interface.
    Instantiates SQLite, PostgreSQL, or MySQL concrete classes based on config.DB_TYPE.
    """
    _instances = {}

    @classmethod
    def get_manager(cls) -> BaseDatabaseManager:
        db_type = config.DB_TYPE.strip().lower()
        if db_type not in cls._instances:
            if db_type == "sqlite":
                cls._instances[db_type] = SQLiteDatabaseManager()
            elif db_type in ["mysql", "postgresql"]:
                cls._instances[db_type] = RelationalDatabaseManager(db_type)
            else:
                print(f"[Warning] Unknown database type '{config.DB_TYPE}'. Defaulting to SQLite.")
                cls._instances[db_type] = SQLiteDatabaseManager()
        return cls._instances[db_type]


# ==============================================================================
# BACKWARD-COMPATIBLE API MODULE-LEVEL WRAPPER FUNCTIONS
# ==============================================================================

def initialize_database():
    """Initializes the database schema if tables do not exist."""
    return DatabaseManagerFactory.get_manager().initialize_database()

def save_model_run(run_id, model_type, metrics):
    """Saves model metrics to the runs table."""
    return DatabaseManagerFactory.get_manager().save_model_run(run_id, model_type, metrics)

def save_predictions(run_id, df):
    """Bulk records student predictions to the database."""
    return DatabaseManagerFactory.get_manager().save_predictions(run_id, df)

def fetch_student_history(student_id):
    """Fetches all past predictions for a specific student, ordered chronologically."""
    return DatabaseManagerFactory.get_manager().fetch_student_history(student_id)

def save_dataset_metadata(filename, row_count, col_count, quality_score, model_version):
    """Logs dataset uploaded file version metrics."""
    return DatabaseManagerFactory.get_manager().save_dataset_metadata(filename, row_count, col_count, quality_score, model_version)

def get_latest_dataset_metadata():
    """Returns the most recently logged dataset upload metadata."""
    return DatabaseManagerFactory.get_manager().get_latest_dataset_metadata()

def get_all_dataset_metadata():
    """Returns all logged dataset upload metadata histories."""
    return DatabaseManagerFactory.get_manager().get_all_dataset_metadata()

def get_model_history():
    """Fetches all past model training runs."""
    return DatabaseManagerFactory.get_manager().get_model_history()

