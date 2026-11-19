"""
Data Loader Utility Module for EduPredict AI.
Handles CSV and data file ingestion with robust encoding fallbacks.
"""

import os
import pandas as pd

def load_csv(filepath):
    """
    Safely loads a CSV file with support for multiple encodings.
    Handles UTF-8 BOM automatically and falls back to latin-1 if needed.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at: {filepath}")
        
    try:
        # utf-8-sig handles UTF-8 with Byte Order Mark (BOM) automatically
        df = pd.read_csv(filepath, encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
        except Exception as e:
            raise IOError(f"Could not load file {filepath} with UTF-8 or Latin-1: {e}")
    except Exception as e:
        raise IOError(f"Error loading CSV {filepath}: {e}")
        
    return df
