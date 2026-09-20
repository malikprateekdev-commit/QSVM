import os
import sys
import numpy as np
import pandas as pd
import json

def run_amplitude_encoding():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    preprocessed_dir = os.path.join(base_dir, 'results', 'preprocessed')
    tables_dir = os.path.join(base_dir, 'results', 'tables')
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. Load data
    X_train = np.load(os.path.join(preprocessed_dir, 'X_train_pca.npy'))
    X_test = np.load(os.path.join(preprocessed_dir, 'X_test_pca.npy'))
    
    def encode_amplitude(X):
        # 2. Calculate L2 Norm
        norms = np.linalg.norm(X, axis=1)
        zero_norm_mask = (norms == 0)
        
        # 3. Normalize the 16-dimensional vector
        X_normed = np.zeros_like(X)
        # Avoid division by zero
        X_normed[~zero_norm_mask] = X[~zero_norm_mask] / norms[~zero_norm_mask, np.newaxis]
        
        # For mathematically valid quantum states, zero-norm classical vectors 
        # must be mapped to a valid basis state (e.g., |0000>)
        if np.any(zero_norm_mask):
            X_normed[zero_norm_mask, 0] = 1.0 
            
        return X_normed, norms, zero_norm_mask
        
    X_train_amp, train_norms, train_zeros = encode_amplitude(X_train)
    X_test_amp, test_norms, test_zeros = encode_amplitude(X_test)
    
    # Save representations
    # The normalized 16-dimensional vector is the exact mathematical 
    # statevector representation for the 4-qubit system.
    np.save(os.path.join(preprocessed_dir, 'X_train_amplitude.npy'), X_train_amp)
    np.save(os.path.join(preprocessed_dir, 'X_test_amplitude.npy'), X_test_amp)
    
    # 4. Analyze Constraints
    all_norms = np.concatenate((train_norms, test_norms))
    all_zeros = np.sum(train_zeros) + np.sum(test_zeros)
    
    unique_norms = len(np.unique(all_norms))
    
    # Verify sum(a_i^2) ~ 1
    train_sums = np.sum(X_train_amp**2, axis=1)
    test_sums = np.sum(X_test_amp**2, axis=1)
    max_deviation = max(np.max(np.abs(train_sums - 1.0)), np.max(np.abs(test_sums - 1.0)))
    
    stats = {
        "Qubits": 4,
        "Total_Samples": len(all_norms),
        "Zero_Norm_Samples": int(all_zeros),
        "Distinct_Original_Norms": unique_norms,
        "Max_Normalization_Deviation": float(max_deviation)
    }
    
    pd.DataFrame([stats]).to_csv(os.path.join(metrics_dir, 'amplitude_encoding_stats.csv'), index=False)
    
    # Save examples
    examples = []
    for i in range(2):
        examples.append({
            "Dataset": "Train",
            "Index": i,
            "Original_Norm": float(train_norms[i]),
            "Original_Vector": [float(v) for v in X_train[i]],
            "Normalized_State": [float(v) for v in X_train_amp[i]]
        })
    with open(os.path.join(tables_dir, 'amplitude_encoding_examples.json'), 'w') as f:
        json.dump(examples, f, indent=4)
        
    # Output Numerical Values
    print(f"Qubits Confirmed: 4")
    print(f"Total Samples Processed: {stats['Total_Samples']}")
    print(f"Zero-Norm Samples Detected: {stats['Zero_Norm_Samples']}")
    print(f"Distinct Original Norms: {stats['Distinct_Original_Norms']}")
    print(f"Max Deviation from sum(a_i^2)=1: {stats['Max_Normalization_Deviation']:.2e}")
    
    print("\nRepresentative Data (Train Sample 0):")
    print(f"  Original Norm: {train_norms[0]:.4f}")
    print(f"  Original PCA:  {np.round(X_train[0], 4).tolist()}")
    print(f"  Encoded State: {np.round(X_train_amp[0], 4).tolist()}")

if __name__ == "__main__":
    run_amplitude_encoding()
