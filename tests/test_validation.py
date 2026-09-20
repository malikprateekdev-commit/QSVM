import os
import numpy as np

def test_pca_dimensions():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    x_train = np.load(os.path.join(base_dir, 'results', 'preprocessed', 'X_train_pca.npy'))
    assert x_train.shape[1] == 16, "PCA dimensions should be exactly 16"

def test_amplitude_normalization():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    x_amp = np.load(os.path.join(base_dir, 'results', 'preprocessed', 'X_train_amplitude.npy'))
    norms = np.sum(x_amp**2, axis=1)
    assert np.allclose(norms, 1.0), "Amplitude encoding must be L2 normalized to 1.0"

if __name__ == "__main__":
    test_pca_dimensions()
    test_amplitude_normalization()
    print("All tests passed.")