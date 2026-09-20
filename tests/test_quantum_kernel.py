import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantum_kernel import calculate_overlap_from_states, validate_compute_uncompute_overlap

def test_overlap_calculation():
    """
    Tests that the core mathematical inner product strictly returns probabilities.
    """
    # Orthogonal states must return 0 overlap
    state1 = np.array([1, 0, 0, 0])
    state2 = np.array([0, 1, 0, 0])
    assert calculate_overlap_from_states(state1, state2) == 0.0
    
    # Identical states must return 1 overlap
    assert calculate_overlap_from_states(state1, state1) == 1.0

def test_compute_uncompute_validation():
    """
    Strictly verifies that the U_x† U_y circuit probability of measuring 
    the all-zero state equals the squared absolute statevector inner product.
    """
    x_val = np.random.uniform(0, np.pi, 16)
    y_val = np.random.uniform(0, np.pi, 16)
    
    prob_zero, overlap = validate_compute_uncompute_overlap("angle", x_val, y_val)
    
    assert prob_zero is not None
    assert overlap is not None
    assert np.isclose(prob_zero, overlap, atol=1e-5), "Compute-uncompute diverges from state overlap."
