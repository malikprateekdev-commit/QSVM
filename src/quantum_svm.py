import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def run_quantum_svm():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    preprocessed_dir = os.path.join(base_dir, 'results', 'preprocessed')
    kernels_dir = os.path.join(base_dir, 'results', 'kernels')
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    tables_dir = os.path.join(base_dir, 'results', 'tables')
    models_dir = os.path.join(base_dir, 'results', 'models')
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Load targets
    y_train = np.load(os.path.join(preprocessed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(preprocessed_dir, 'y_test.npy'))
    
    encodings = ["basis", "angle", "amplitude", "phase"]
    
    # Exact deterministic CV setup
    param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for enc in encodings:
        print(f"=== {enc.upper()} ENCODING ===")
        
        # 1. Load unmodified kernel matrices
        K_train = np.load(os.path.join(kernels_dir, f"{enc}_K_train.npy"))
        K_test = np.load(os.path.join(kernels_dir, f"{enc}_K_test.npy"))
        
        svc = SVC(kernel='precomputed', random_state=42)
        
        # 2. Tune on training data ONLY
        grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, return_train_score=False)
        
        start_tune = time.time()
        grid.fit(K_train, y_train)
        tune_time = time.time() - start_tune
        
        selected_C = grid.best_params_['C']
        cv_score = grid.best_score_
        
        pd.DataFrame(grid.cv_results_).to_csv(os.path.join(tables_dir, f"{enc}_cv_results.csv"), index=False)
        
        # 3. Fit final selected model
        final_svc = SVC(kernel='precomputed', C=selected_C, random_state=42)
        
        start_train = time.time()
        final_svc.fit(K_train, y_train)
        train_time = time.time() - start_train
        
        # 4. Predict once on test block
        start_pred = time.time()
        y_pred = final_svc.predict(K_test)
        pred_time = time.time() - start_pred
        
        # 5. Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        
        # Save artifacts securely
        joblib.dump(final_svc, os.path.join(models_dir, f'{enc}_svm_model.joblib'))
        np.save(os.path.join(tables_dir, f'{enc}_predictions.npy'), y_pred)
        
        metrics_df = pd.DataFrame({
            "Metric": [
                "Selected_C", "CV_Score", 
                "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1",
                "Tuning_Runtime_s", "Training_Runtime_s", "Prediction_Runtime_s"
            ],
            "Value": [
                selected_C, cv_score,
                acc, prec, rec, f1,
                tune_time, train_time, pred_time
            ]
        })
        metrics_df.to_csv(os.path.join(metrics_dir, f'{enc}_svm.csv'), index=False)
        
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
        cm_df.to_csv(os.path.join(tables_dir, f'{enc}_confusion_matrix.csv'))
        
        print("METRICS:")
        print(metrics_df.to_string(index=False))
        print("\nCONFUSION MATRIX:")
        print(cm_df.to_string())
        print("\nCLASSIFICATION REPORT:")
        print(report)
        print("-" * 60)

if __name__ == "__main__":
    run_quantum_svm()
