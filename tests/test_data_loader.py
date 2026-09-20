import os
import sys
import pandas as pd
import pytest

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_spambase, FEATURE_NAMES, TARGET_NAME

def test_load_spambase():
    """
    Tests that load_spambase correctly reads the dataset, separates features and targets,
    assigns correct names, and ensures appropriate numerical types.
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'spambase.data')
    if not os.path.exists(data_path):
        pytest.skip(f"Dataset not found at {data_path}")
        
    X, y = load_spambase(data_path)
    
    # 1. Validate number of samples and features
    assert X.shape[1] == 57, "Expected exactly 57 feature columns in X"
    assert len(X) == len(y), "X and y must have the same number of rows"
    
    # 2. Validate separation types
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y should be a pandas Series"
    
    # 3. Validate column names
    assert list(X.columns) == FEATURE_NAMES, "Feature names in X do not match expected list"
    assert y.name == TARGET_NAME, "Target column name does not match expected name"
    
    # 4. Validate numerical types
    assert pd.api.types.is_numeric_dtype(y), "Target y must be numeric"
    for col in X.columns:
        assert pd.api.types.is_numeric_dtype(X[col]), f"Feature {col} in X must be numeric"
        
    # 5. Validate class labels are only 0 and 1
    assert set(y.dropna().unique()).issubset({0, 1}), "Target labels should be 0 and 1"
