import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantum_svm import train_and_evaluate_quantum_svm

def test_quantum_svm_evaluation(tmp_path):
    """
    Test the Quantum SVM logic strictly against expected kernel dimensions (160x160 and 40x160)
    and valid output artifacts.
    """
    # Create strictly sized dummy Gram matrices.
    # Using X @ X.T generates valid symmetric positive semi-definite kernels.
    X_train = np.random.rand(160, 5)
    X_test = np.random.rand(40, 5)
    
    K_train = X_train @ X_train.T
    K_test = X_test @ X_train.T
    
    y_train = np.random.randint(0, 2, 160)
    y_test = np.random.randint(0, 2, 40)
    
    # Save dummy paths
    k_train_path = tmp_path / "dummy_K_train.npy"
    k_test_path = tmp_path / "dummy_K_test.npy"
    y_train_path = tmp_path / "y_train.npy"
    y_test_path = tmp_path / "y_test.npy"
    
    np.save(k_train_path, K_train)
    np.save(k_test_path, K_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    
    metrics_dir = tmp_path / "metrics"
    tables_dir = tmp_path / "tables"
    
    acc, prec, rec, f1, cm, _ = train_and_evaluate_quantum_svm(
        "dummy_enc",
        str(k_train_path), str(k_test_path), str(y_train_path), str(y_test_path),
        str(metrics_dir), str(tables_dir)
    )
    
    assert 0 <= acc <= 1
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
    assert 0 <= f1 <= 1
    assert cm.shape == (2, 2)
    
    assert os.path.exists(metrics_dir / "dummy_enc.csv")
    assert os.path.exists(tables_dir / "dummy_enc_predictions.csv")
    assert os.path.exists(tables_dir / "dummy_enc_confusion_matrix.csv")
