import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_spambase
from src.preprocessing import preprocess_and_reduce_data

def test_preprocessing():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'spambase.data')
    if not os.path.exists(data_path):
        pytest.skip(f"Dataset not found at {data_path}")
        
    X_full, y_full = load_spambase(data_path)
    
    # Pass None to save_dir to avoid saving files during tests
    X_tr, X_te, y_tr, y_te, pca, cum_var = preprocess_and_reduce_data(
        X_full, y_full, n_components=16, n_samples=200, test_size=40, save_dir=None
    )
    
    assert X_tr.shape == (160, 16), "Training features must have shape (160, 16)"
    assert X_te.shape == (40, 16), "Testing features must have shape (40, 16)"
    assert len(y_tr) == 160, "Training labels must have 160 items"
    assert len(y_te) == 40, "Testing labels must have 40 items"
    assert pca.n_components == 16, "PCA must have exactly 16 components"
    assert 0.0 < cum_var <= 1.0, "Cumulative variance must be between 0 and 1"
