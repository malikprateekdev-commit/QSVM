import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def run_classical_baseline():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    preprocessed_dir = os.path.join(base_dir, 'results', 'preprocessed')
    metrics_dir = os.path.join(base_dir, 'results', 'metrics')
    tables_dir = os.path.join(base_dir, 'results', 'tables')
    models_dir = os.path.join(base_dir, 'results', 'models')
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load Data
    X_train_pca = np.load(os.path.join(preprocessed_dir, 'X_train_pca.npy'))
    X_test_pca = np.load(os.path.join(preprocessed_dir, 'X_test_pca.npy'))
    y_train = np.load(os.path.join(preprocessed_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(preprocessed_dir, 'y_test.npy'))
    
    # 2. Hyperparameter Tuning Setup (Training Data ONLY)
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'gamma': ['scale', 0.001, 0.01, 0.1, 1]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    svc = SVC(kernel='rbf', random_state=42)
    
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, return_train_score=False)
    
    start_tune = time.time()
    grid.fit(X_train_pca, y_train)
    tune_time = time.time() - start_tune
    
    selected_C = grid.best_params_['C']
    selected_gamma = grid.best_params_['gamma']
    cv_score = grid.best_score_
    
    # Save raw CV results
    cv_results_df = pd.DataFrame(grid.cv_results_)
    cv_results_df.to_csv(os.path.join(tables_dir, 'classical_cv_results.csv'), index=False)
    
    # 3. Final Training Configuration
    final_svc = SVC(kernel='rbf', C=selected_C, gamma=selected_gamma, random_state=42)
    
    start_train = time.time()
    final_svc.fit(X_train_pca, y_train)
    train_time = time.time() - start_train
    
    # 4. Final Evaluation (Test Data)
    start_pred = time.time()
    y_pred = final_svc.predict(X_test_pca)
    pred_time = time.time() - start_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # 5. Save Artifacts
    joblib.dump(final_svc, os.path.join(models_dir, 'classical_rbf_model.joblib'))
    
    metrics_df = pd.DataFrame({
        "Metric": [
            "Selected_C", "Selected_gamma", "CV_Score", 
            "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1",
            "Tuning_Runtime_s", "Training_Runtime_s", "Prediction_Runtime_s"
        ],
        "Value": [
            selected_C, selected_gamma, cv_score,
            acc, prec, rec, f1,
            tune_time, train_time, pred_time
        ]
    })
    metrics_df.to_csv(os.path.join(metrics_dir, 'classical_rbf_tuned.csv'), index=False)
    
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Predicted 0", "Predicted 1"])
    cm_df.to_csv(os.path.join(tables_dir, 'classical_confusion_matrix.csv'))
    
    print("\n=== CLASSICAL RBF TUNED METRICS ===")
    print(metrics_df.to_string(index=False))
    
    print("\n=== CONFUSION MATRIX ===")
    print(cm_df.to_string())
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(report)

if __name__ == "__main__":
    run_classical_baseline()
