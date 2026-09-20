import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.encodings.amplitude import normalize_sample, get_encoded_state, NUM_QUBITS

def test_normalization():
    """
    Tests that standard vectors are correctly L2 normalized.
    """
    sample = np.ones(16) * 2.0
    norm_x, norm_val = normalize_sample(sample)
    
    # Original norm should be sqrt(16 * 4) = 8
    assert np.isclose(norm_val, 8.0)
    
    # Sum of squared amplitudes must be exactly 1
    assert np.isclose(np.sum(norm_x**2), 1.0)
    
def test_zero_norm_handling():
    """
    Tests that a zero vector explicitly outputs a uniform superposition
    instead of producing NaNs.
    """
    sample = np.zeros(16)
    norm_x, norm_val = normalize_sample(sample)
    
    assert norm_val == 0.0
    assert not np.any(np.isnan(norm_x))
    
    # Uniform superposition: 1/sqrt(16) = 0.25
    expected = np.ones(16) * 0.25
    np.testing.assert_array_almost_equal(norm_x, expected)

def test_amplitude_encoded_state():
    """
    Tests the strict 4-qubit state preparation validation.
    """
    sample = np.random.randn(16)
    state = get_encoded_state(sample)
    
    # Strict 4 qubit validation
    assert NUM_QUBITS == 4, "Must strictly use 4 qubits."
    assert state.shape == (2**4,), "Statevector must represent exactly 4 qubits (16 values)."
    assert np.isclose(np.sum(np.abs(state)**2), 1.0), "Statevector probabilities must sum to 1."
