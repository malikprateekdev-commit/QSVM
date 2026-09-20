import os
import sys
import time
import tracemalloc
import numpy as np
import pandas as pd
import pennylane as qml

def run_quantum_kernels():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    preprocessed_dir = os.path.join(base_dir, 'results', 'preprocessed')
    kernels_dir = os.path.join(base_dir, 'results', 'kernels')
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    
    os.makedirs(kernels_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    encodings = ["basis", "angle", "amplitude", "phase"]
    results = []
    
    # Setup PennyLane devices strictly for validation pairs
    dev_4 = qml.device('default.qubit', wires=4)
    dev_16 = qml.device('default.qubit', wires=16)
    
    @qml.qnode(dev_16)
    def basis_circuit(x):
        for i in range(16):
            if x[i] == 1:
                qml.PauliX(wires=i)
        return qml.state()
        
    @qml.qnode(dev_16)
    def angle_circuit(x):
        for i in range(16):
            qml.RY(x[i], wires=i)
        return qml.state()
        
    @qml.qnode(dev_4)
    def amplitude_circuit(x):
        qml.StatePrep(x, wires=range(4))
        return qml.state()
        
    @qml.qnode(dev_16)
    def phase_circuit(x):
        for i in range(16):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)
        return qml.state()
        
    def explicit_overlap(enc, x, y):
        if enc == "basis":
            st_x, st_y = basis_circuit(x), basis_circuit(y)
        elif enc == "angle":
            st_x, st_y = angle_circuit(x), angle_circuit(y)
        elif enc == "amplitude":
            st_x, st_y = amplitude_circuit(x), amplitude_circuit(y)
        elif enc == "phase":
            st_x, st_y = phase_circuit(x), phase_circuit(y)
        return np.abs(np.vdot(st_x, st_y))**2
    
    for enc in encodings:
        tracemalloc.start()
        start_time = time.time()
        
        # Load representations
        if enc == "basis":
            X_train = np.load(os.path.join(preprocessed_dir, 'X_train_basis.npy'))
            X_test = np.load(os.path.join(preprocessed_dir, 'X_test_basis.npy'))
        elif enc == "angle":
            X_train = np.load(os.path.join(preprocessed_dir, 'X_train_angle_theta.npy'))
            X_test = np.load(os.path.join(preprocessed_dir, 'X_test_angle_theta.npy'))
        elif enc == "amplitude":
            X_train = np.load(os.path.join(preprocessed_dir, 'X_train_amplitude.npy'))
            X_test = np.load(os.path.join(preprocessed_dir, 'X_test_amplitude.npy'))
        elif enc == "phase":
            X_train = np.load(os.path.join(preprocessed_dir, 'X_train_phase_theta.npy'))
            X_test = np.load(os.path.join(preprocessed_dir, 'X_test_phase_theta.npy'))
            
        N_train = len(X_train)
        N_test = len(X_test)
        
        # Output Matrices
        K_train = np.zeros((N_train, N_train), dtype=np.float32)
        K_test = np.zeros((N_test, N_train), dtype=np.float32)
        
        # Chunked calculation to explicitly manage memory footprint 
        chunk_size = 500
        
        if enc == "basis":
            for i in range(0, N_train, chunk_size):
                end_i = min(i + chunk_size, N_train)
                # K(x,y) = 1 if exactly equal across all 16 bits, else 0
                match = (X_train[i:end_i, None, :] == X_train[None, :, :]).all(axis=2)
                K_train[i:end_i, :] = match.astype(np.float32)
                
            for i in range(0, N_test, chunk_size):
                end_i = min(i + chunk_size, N_test)
                match = (X_test[i:end_i, None, :] == X_train[None, :, :]).all(axis=2)
                K_test[i:end_i, :] = match.astype(np.float32)
                
        elif enc in ["angle", "phase"]:
            for i in range(0, N_train, chunk_size):
                end_i = min(i + chunk_size, N_train)
                # Overlap |<x|y>|^2 analytically reduces to the product of cos^2(diff/2)
                diff = X_train[i:end_i, None, :] - X_train[None, :, :]
                cos_sq = np.cos(diff / 2.0)**2
                K_train[i:end_i, :] = np.prod(cos_sq, axis=2).astype(np.float32)
                
            for i in range(0, N_test, chunk_size):
                end_i = min(i + chunk_size, N_test)
                diff = X_test[i:end_i, None, :] - X_train[None, :, :]
                cos_sq = np.cos(diff / 2.0)**2
                K_test[i:end_i, :] = np.prod(cos_sq, axis=2).astype(np.float32)
                
        elif enc == "amplitude":
            # State is exactly the 16-D normalized vector, |<x|y>|^2 = (x dot y)^2
            for i in range(0, N_train, chunk_size):
                end_i = min(i + chunk_size, N_train)
                dot = np.dot(X_train[i:end_i], X_train.T)
                K_train[i:end_i, :] = (dot**2).astype(np.float32)
                
            for i in range(0, N_test, chunk_size):
                end_i = min(i + chunk_size, N_test)
                dot = np.dot(X_test[i:end_i], X_train.T)
                K_test[i:end_i, :] = (dot**2).astype(np.float32)
                
        runtime = time.time() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Validation on 20 random pairs
        val_errors = []
        np.random.seed(42)
        idx_pairs = np.random.randint(0, N_train, size=(20, 2))
        for idx1, idx2 in idx_pairs:
            analytic_val = K_train[idx1, idx2]
            explicit_val = explicit_overlap(enc, X_train[idx1], X_train[idx2])
            val_errors.append(np.abs(analytic_val - explicit_val))
            
        max_val_error = float(np.max(val_errors))
        
        # Verification constraints
        sym_error = float(np.max(np.abs(K_train - K_train.T)))
        diag = np.diag(K_train)
        diag_mean = float(np.mean(diag))
        diag_min = float(np.min(diag))
        diag_max = float(np.max(diag))
        
        # Safe saving
        np.save(os.path.join(kernels_dir, f"{enc}_K_train.npy"), K_train)
        np.save(os.path.join(kernels_dir, f"{enc}_K_test.npy"), K_test)
        
        results.append({
            "Encoding": enc,
            "K_train_Shape": f"{K_train.shape}",
            "K_test_Shape": f"{K_test.shape}",
            "Min": float(np.min(K_train)),
            "Max": float(np.max(K_train)),
            "Mean": float(np.mean(K_train)),
            "Std": float(np.std(K_train)),
            "Diag_Mean": diag_mean,
            "Diag_Min": diag_min,
            "Diag_Max": diag_max,
            "Symmetry_Error": sym_error,
            "Validation_Max_Error": max_val_error,
            "Runtime_s": runtime,
            "Peak_Memory_MB": peak_mem / (1024 * 1024),
            "Has_NaN": bool(np.isnan(K_train).any()),
            "Has_Inf": bool(np.isinf(K_train).any())
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(metrics_dir, "quantum_kernels_stats.csv"), index=False)
    
    # Precise Text Output Matrix
    print("=== QUANTUM KERNEL STATS ===\n")
    for row in results:
        print(f"--- Encoding: {row['Encoding'].upper()} ---")
        for k, v in row.items():
            if k != "Encoding":
                print(f"  {k}: {v}")
        print()

if __name__ == "__main__":
    run_quantum_kernels()
