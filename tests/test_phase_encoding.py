import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.encodings.phase import fit_phase_scaler, get_encoded_state, NUM_QUBITS

def test_phase_encoded_state_properties():
    """
    Tests the 16-qubit H -> RZ circuit properties and validates the phase assumptions.
    """
    angles = np.random.uniform(0, np.pi, size=16)
    state = get_encoded_state(angles)
    
    # Strict 16 qubit validation
    assert NUM_QUBITS == 16, "Must strictly use 16 qubits."
    assert state.shape == (2**16,), "Statevector must represent exactly 16 qubits."
    
    # Total probability must be 1
    assert np.isclose(np.sum(np.abs(state)**2), 1.0)
    
    # PROOF that immediate computational-basis probabilities do not reveal phase:
    # Because of the H gates, every amplitude should have the exact same magnitude (1/sqrt(2^16))
    expected_prob = 1.0 / (2**16)
    actual_probs = np.abs(state)**2
    assert np.allclose(actual_probs, expected_prob), "All basis probabilities must be uniform, proving phase is hidden."
