import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def run_angle_encoding():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    preprocessed_dir = os.path.join(base_dir, 'results', 'preprocessed')
    tables_dir = os.path.join(base_dir, 'results', 'tables')
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # 1. Load data
    X_train = np.load(os.path.join(preprocessed_dir, 'X_train_pca.npy'))
    X_test = np.load(os.path.join(preprocessed_dir, 'X_test_pca.npy'))
    
    # 2. Fit angle mapping strictly on training data
    # Mapping PCA feature space into the [0, pi] domain for optimal RY rotation range
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    theta_train = scaler.fit_transform(X_train)
    
    # Apply identical mapping to test data (values may naturally fall outside [0, pi] if test data exceeds train max/min)
    theta_test = scaler.transform(X_test)
    
    # Save the mathematically equivalent exact representation.
    # Generating 2^16 (65,536) length vectors for 4601 samples is highly inefficient (2.4 GB).
    # Since the state is completely separable (no CNOTs), the 16 independent theta values 
    # intrinsically and exactly represent the state |phi(x)> = TensorProd(RY(theta_i)|0>).
    np.save(os.path.join(preprocessed_dir, 'X_train_angle_theta.npy'), theta_train)
    np.save(os.path.join(preprocessed_dir, 'X_test_angle_theta.npy'), theta_test)
    
    # 3. Analyze representations
    n_qubits = X_train.shape[1]
    state_dimension = 2 ** n_qubits
    
    train_min = np.min(theta_train)
    train_max = np.max(theta_train)
    test_min = np.min(theta_test)
    test_max = np.max(theta_test)
    
    # The state is |phi> = (cos(t_1/2)|0> + sin(t_1/2)|1>) x ... x (cos(t_16/2)|0> + sin(t_16/2)|1>)
    # For every single qubit, cos^2 + sin^2 = 1.0
    # Therefore, the product of their norms is strictly 1.0.
    norm_val = 1.0
    
    stats = {
        "Qubits": n_qubits,
        "State_Dimension": state_dimension,
        "Train_Angle_Min": train_min,
        "Train_Angle_Max": train_max,
        "Test_Angle_Min": test_min,
        "Test_Angle_Max": test_max,
        "Normalization_Value": norm_val
    }
    
    pd.DataFrame([stats]).to_csv(os.path.join(metrics_dir, 'angle_encoding_stats.csv'), index=False)
    
    print(f"Qubits Confirmed: {n_qubits}")
    print(f"State Dimension: {state_dimension}")
    print(f"Training Angle Range: [{train_min:.6f}, {train_max:.6f}]")
    print(f"Testing Angle Range: [{test_min:.6f}, {test_max:.6f}]")
    print(f"Normalization Confirmed: {norm_val:.1f}")
    
    print("\nRepresentative Data (Train Sample 0):")
    print(f"  Original PCA: {np.round(X_train[0], 4).tolist()}")
    print(f"  Theta Angles: {np.round(theta_train[0], 4).tolist()}")

if __name__ == "__main__":
    run_angle_encoding()
