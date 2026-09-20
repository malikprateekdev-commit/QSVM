import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.classical_baseline import train_and_evaluate_classical_baseline

def test_classical_baseline(tmp_path):
    """
    Test the classical baseline logic using dummy data with the same dimensions 
    as our strict experiment (160 train, 40 test, 16 features).
    """
    # Create dummy data
    X_train = np.random.rand(160, 16)
    X_test = np.random.rand(40, 16)
    y_train = np.random.randint(0, 2, 160)
    y_test = np.random.randint(0, 2, 40)
    
    # Save to tmp paths
    x_train_path = tmp_path / "X_train_pca.npy"
    x_test_path = tmp_path / "X_test_pca.npy"
    y_train_path = tmp_path / "y_train.npy"
    y_test_path = tmp_path / "y_test.npy"
    
    np.save(x_train_path, X_train)
    np.save(x_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    
    metrics_dir = tmp_path / "metrics"
    tables_dir = tmp_path / "tables"
    
    # Run function
    acc, prec, rec, f1, cm, report = train_and_evaluate_classical_baseline(
        str(x_train_path), str(x_test_path), str(y_train_path), str(y_test_path),
        str(metrics_dir), str(tables_dir)
    )
    
    # Validate bounds
    assert 0 <= acc <= 1
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
    assert 0 <= f1 <= 1
    assert cm.shape == (2, 2)
    
    # Validate file creation
    assert os.path.exists(metrics_dir / "classical_rbf.csv")
    assert os.path.exists(tables_dir / "classical_rbf_confusion_matrix.csv")
    assert os.path.exists(tables_dir / "classical_rbf_predictions.csv")
