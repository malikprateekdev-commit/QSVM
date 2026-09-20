import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_spambase

def preprocessing_dataset(X, y, test_size=0.2, random_state=42, output_dir="results/preprocessed"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 2. Validation
    assert len(X_train) + len(X_test) == 4601, "Validation Failed: len(train) + len(test) != 4601"
    
    # 3. Save arrays in reusable format
    np.save(os.path.join(output_dir, "X_train_raw.npy"), X_train.values)
    np.save(os.path.join(output_dir, "X_test_raw.npy"), X_test.values)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train.values)
    np.save(os.path.join(output_dir, "y_test.npy"), y_test.values)
    
    # 4. Save metadata and class distributions
    train_dist = y_train.value_counts().to_dict()
    test_dist = y_test.value_counts().to_dict()
    
    metadata = {
        "total_samples": 4601,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": random_state,
        "stratified": True,
        "train_class_counts": {str(k): v for k, v in train_dist.items()},
        "test_class_counts": {str(k): v for k, v in test_dist.items()}
    }
    
    with open(os.path.join(output_dir, "split_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save CSV forms for the table tracking
    pd.DataFrame({
        "Class": list(train_dist.keys()),
        "Train_Count": list(train_dist.values())
    }).to_csv(os.path.join(output_dir, "train_class_distribution.csv"), index=False)
    
    pd.DataFrame({
        "Class": list(test_dist.keys()),
        "Test_Count": list(test_dist.values())
    }).to_csv(os.path.join(output_dir, "test_class_distribution.csv"), index=False)
        
    return metadata

if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'raw', 'spambase.data')
    output_dir = os.path.join(base_dir, 'results', 'preprocessed')
    
    X, y = load_spambase(data_path)
    meta = preprocessing_dataset(X, y, output_dir=output_dir)
    
    print(f"Validation: len(train) + len(test) = {meta['train_samples']} + {meta['test_samples']} = {meta['train_samples'] + meta['test_samples']}")
    print(f"Train Sample Count: {meta['train_samples']}")
    print(f"Test Sample Count: {meta['test_samples']}")
    print(f"Train Class Counts: Class 0: {meta['train_class_counts']['0']}, Class 1: {meta['train_class_counts']['1']}")
    print(f"Test Class Counts: Class 0: {meta['test_class_counts']['0']}, Class 1: {meta['test_class_counts']['1']}")


import os
import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def run_preprocessing():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'raw', 'spambase.data')
    from src.data_loader import load_spambase
    X, y = load_spambase(data_path)
    output_dir = os.path.join(base_dir, 'results', 'preprocessed')
    preprocessing_dataset(X, y, output_dir=output_dir)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    preprocessed_dir = os.path.join(base_dir, 'results', 'preprocessed')
    tables_dir = os.path.join(base_dir, 'results', 'tables')
    
    os.makedirs(tables_dir, exist_ok=True)
    
    # Load raw splits
    X_train_raw = np.load(os.path.join(preprocessed_dir, 'X_train_raw.npy'))
    X_test_raw = np.load(os.path.join(preprocessed_dir, 'X_test_raw.npy'))
    y_train = np.load(os.path.join(preprocessed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(preprocessed_dir, 'y_test.npy'))
    
    # 1. Scale ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # 2. PCA ONLY on scaled training data
    pca = PCA(n_components=16, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Validate Output Shapes
    assert X_train_pca.shape == (len(X_train_raw), 16)
    assert X_test_pca.shape == (len(X_test_raw), 16)
    
    # Explained variance calculation
    evr = pca.explained_variance_ratio_
    cum_evr = np.sum(evr)
    
    # Save artifacts
    np.save(os.path.join(preprocessed_dir, 'X_train_pca.npy'), X_train_pca)
    np.save(os.path.join(preprocessed_dir, 'X_test_pca.npy'), X_test_pca)
    
    with open(os.path.join(preprocessed_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(preprocessed_dir, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)
        
    # Build and save CSV for variance
    var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(16)],
        "Explained_Variance_Ratio": evr
    })
    var_df.to_csv(os.path.join(tables_dir, 'pca_variance.csv'), index=False)
    
    # Print metrics strictly as requested
    print(f"Original Dimension: {X_train_raw.shape[1]}")
    print(f"Final Dimension: {X_train_pca.shape[1]}")
    print(f"Cumulative Explained Variance: {cum_evr:.6f}")
    print(f"X_train_pca Shape: {X_train_pca.shape}")
    print(f"X_test_pca Shape: {X_test_pca.shape}")

if __name__ == "__main__":
    run_preprocessing()

