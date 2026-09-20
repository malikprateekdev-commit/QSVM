import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.encodings.angle import fit_angle_scaler, get_encoded_state, NUM_QUBITS

def test_angle_mapping_logic():
    """
    Tests that PCA values are strictly mapped deterministically into the [0, pi] range.
    """
    X_train_dummy = np.random.randn(10, 16) * 100
    scaler = fit_angle_scaler(X_train_dummy)
    
    mapped_train = scaler.transform(X_train_dummy)
    
    # Validate bounds are strictly between 0 and pi
    assert np.all(mapped_train >= 0.0), "Mapped angles must be >= 0"
    assert np.all(mapped_train <= np.pi + 1e-7), "Mapped angles must be <= pi"
    
def test_angle_encoded_state():
    """
    Tests the 16-qubit RY circuit validation.
    """
    angles = np.random.uniform(0, np.pi, size=16)
    state = get_encoded_state(angles)
    
    # Strict 16 qubit validation
    assert NUM_QUBITS == 16, "Must strictly use 16 qubits."
    
    # 16 qubits yield a statevector of size 2^16 = 65536
    assert state.shape == (2**16,), "Statevector must represent exactly 16 qubits."
    
    # In a pure quantum state, sum of squared amplitudes must exactly equal 1
    assert np.isclose(np.sum(np.abs(state)**2), 1.0), "Statevector probabilities must sum to 1."
