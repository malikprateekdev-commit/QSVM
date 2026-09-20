import os
import sys
import json
import numpy as np
import pandas as pd

def convert_to_binary(X):
    """
    Applies the basis encoding rule:
    value >= 0 -> 1
    value < 0  -> 0
    """
    return (X >= 0).astype(int)

def run_basis_encoding():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    preprocessed_dir = os.path.join(base_dir, 'results', 'preprocessed')
    tables_dir = os.path.join(base_dir, 'results', 'tables')
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. Load data
    X_train = np.load(os.path.join(preprocessed_dir, 'X_train_pca.npy'))
    X_test = np.load(os.path.join(preprocessed_dir, 'X_test_pca.npy'))
    
    # 2. Binarize
    X_train_bin = convert_to_binary(X_train)
    X_test_bin = convert_to_binary(X_test)
    
    # Save the transformed arrays for later kernel computation
    np.save(os.path.join(preprocessed_dir, 'X_train_basis.npy'), X_train_bin)
    np.save(os.path.join(preprocessed_dir, 'X_test_basis.npy'), X_test_bin)
    
    # 3. Analyze transformation properties
    X_all = np.vstack((X_train, X_test))
    X_all_bin = np.vstack((X_train_bin, X_test_bin))
    
    total_values = X_all.size
    # A value "changed" if the original float does not strictly equal the binary int
    values_changed = np.sum(X_all != X_all_bin)
    
    # To find unique states and collisions, convert each 16-bit array to a tuple (hashable)
    unique_states = set(tuple(row) for row in X_all_bin)
    num_unique_states = len(unique_states)
    
    total_samples = len(X_all)
    collisions = total_samples - num_unique_states
    
    stats = {
        "Total_Values": total_values,
        "Values_Changed_To_Binary": values_changed,
        "Total_Samples": total_samples,
        "Unique_Binary_States": num_unique_states,
        "Collisions_Count": collisions
    }
    
    # Save metrics
    pd.DataFrame([stats]).to_csv(os.path.join(metrics_dir, 'basis_encoding_stats.csv'), index=False)
    
    # Save representative data
    examples = []
    for i in range(2):
        examples.append({
            "Dataset": "Train",
            "Index": i,
            "Original_Continuous": [float(v) for v in X_train[i]],
            "Binary": [int(v) for v in X_train_bin[i]]
        })
    with open(os.path.join(tables_dir, 'basis_encoding_examples.json'), 'w') as f:
        json.dump(examples, f, indent=4)
        
    # Output numerical facts
    print(f"Total Values: {total_values}")
    print(f"Values Changed (Continuous to Binary): {values_changed}")
    print(f"Total Samples: {total_samples}")
    print(f"Unique Binary States: {num_unique_states}")
    print(f"Collisions (Samples mapped to same state): {collisions}")
    
    print("\nRepresentative Data (Train Sample 0):")
    print(f"  Original: {np.round(X_train[0], 4).tolist()}")
    print(f"  Binary:   {X_train_bin[0].tolist()}")

if __name__ == "__main__":
    run_basis_encoding()
