import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.encodings.basis import convert_to_binary, get_encoded_state, NUM_QUBITS

def test_basis_conversion_logic():
    """
    Tests that the strict rule (val >= 0 -> 1, val < 0 -> 0) is applied correctly.
    """
    X_dummy = np.array([
        [0.5, -0.1, 0.0, -100, 42]
    ])
    expected = np.array([
        [1, 0, 1, 0, 1]
    ])
    result = convert_to_binary(X_dummy)
    np.testing.assert_array_equal(result, expected)

def test_get_encoded_state():
    """
    Tests that a 16-component array yields a 16-qubit computational basis state.
    """
    # 16 random PCA features
    sample = np.random.randn(16)
    state, binary = get_encoded_state(sample)
    
    # Ensure proper array size
    assert len(binary) == 16, "Binary vector must have 16 bits."
    
    # 16 qubits yield a statevector of size 2^16 = 65536
    assert state.shape == (2**16,), "Statevector must represent exactly 16 qubits."
    
    # In computational basis, sum of squared amplitudes must exactly equal 1
    assert np.isclose(np.sum(np.abs(state)**2), 1.0), "Statevector probabilities must sum to 1."
    
    # Strict 16 qubit validation
    assert NUM_QUBITS == 16, "Must strictly use 16 qubits."
