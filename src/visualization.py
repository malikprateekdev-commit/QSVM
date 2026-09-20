import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_generation():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base_dir, 'results')
    metrics_dir = os.path.join(results_dir, 'metrics')
    tables_dir = os.path.join(results_dir, 'tables')
    figures_dir = os.path.join(results_dir, 'figures')
    kernels_dir = os.path.join(results_dir, 'kernels')
    prep_dir = os.path.join(results_dir, 'preprocessed')
    
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Final metrics table
    methods = ["classical_rbf", "basis", "angle", "amplitude", "phase"]
    qubits_map = {"classical_rbf": 0, "basis": 16, "angle": 16, "amplitude": 4, "phase": 16}
    
    metrics_data = []
    for m in methods:
        fpath = os.path.join(metrics_dir, f"{m}_tuned.csv" if m == "classical_rbf" else f"{m}_svm.csv")
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            row = {"Method": m.capitalize(), "Qubits": qubits_map[m]}
            row["Accuracy"] = float(df[df["Metric"] == "Test_Accuracy"]["Value"].values[0])
            row["Precision"] = float(df[df["Metric"] == "Test_Precision"]["Value"].values[0])
            row["Recall"] = float(df[df["Metric"] == "Test_Recall"]["Value"].values[0])
            row["F1"] = float(df[df["Metric"] == "Test_F1"]["Value"].values[0])
            row["Training_Runtime"] = float(df[df["Metric"] == "Training_Runtime_s"]["Value"].values[0])
            row["Prediction_Runtime"] = float(df[df["Metric"] == "Prediction_Runtime_s"]["Value"].values[0])
            metrics_data.append(row)
            
    final_metrics_df = pd.DataFrame(metrics_data)
    final_metrics_df.to_csv(os.path.join(tables_dir, 'final_metrics.csv'), index=False)
    
    # 2. Encoding table
    encoding_data = []
    for m in ["basis", "angle", "amplitude", "phase"]:
        enc_stats_path = os.path.join(metrics_dir, f"{m}_encoding_stats.csv")
        if os.path.exists(enc_stats_path):
            q_stats = pd.read_csv(os.path.join(metrics_dir, "quantum_kernels_stats.csv"))
            q_row = q_stats[q_stats["Encoding"] == m].iloc[0]
            k_runtime = float(q_row["Runtime_s"])
            
            encoding_data.append({
                "Method": m.capitalize(),
                "Input_Dimension": 16,
                "Qubits": qubits_map[m],
                "Required_Transformation": m.capitalize() + " Mapping",
                "Kernel_Computation_Runtime_s": k_runtime
            })
    encoding_df = pd.DataFrame(encoding_data)
    encoding_df.to_csv(os.path.join(tables_dir, 'encoding_details.csv'), index=False)
    
    # 3. PCA table
    X_train_raw = np.load(os.path.join(prep_dir, 'X_train_raw.npy'))
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    pca = PCA(n_components=16, random_state=42)
    pca.fit(X_train_scaled)
    
    pca_data = []
    cumulative = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        pca_data.append({
            "Component": i+1,
            "Original_Dimensions": 57,
            "PCA_Dimensions": 16,
            "Explained_Variance_Ratio": var,
            "Cumulative_Explained_Variance": cumulative
        })
    pca_df = pd.DataFrame(pca_data)
    pca_df.to_csv(os.path.join(tables_dir, 'pca_details.csv'), index=False)
    
    # 4. Quantum kernel statistics
    y_train = np.load(os.path.join(prep_dir, 'y_train.npy'))
    same_mask = (y_train[:, None] == y_train[None, :])
    diff_mask = ~same_mask
    
    kernel_stats = []
    for m in ["basis", "angle", "amplitude", "phase"]:
        k_path = os.path.join(kernels_dir, f"{m}_K_train.npy")
        if os.path.exists(k_path):
            K = np.load(k_path)
            diag = np.diag(K)
            same_sim = np.mean(K[same_mask])
            diff_sim = np.mean(K[diff_mask])
            
            q_stats = pd.read_csv(os.path.join(metrics_dir, "quantum_kernels_stats.csv"))
            q_row = q_stats[q_stats["Encoding"] == m].iloc[0]
            
            kernel_stats.append({
                "Method": m.capitalize(),
                "Mean": float(np.mean(K)),
                "Std": float(np.std(K)),
                "Min": float(np.min(K)),
                "Max": float(np.max(K)),
                "Diag_Mean": float(np.mean(diag)),
                "Same_Class_Sim": float(same_sim),
                "Diff_Class_Sim": float(diff_sim),
                "Computation_Runtime_s": float(q_row["Runtime_s"])
            })
    kernel_stats_df = pd.DataFrame(kernel_stats)
    kernel_stats_df.to_csv(os.path.join(tables_dir, 'quantum_kernel_similarity_stats.csv'), index=False)
    
    # 5. Figures
    # 5.1 Class distribution
    plt.figure()
    sns.countplot(x=y_train)
    plt.title("Training Set Class Distribution")
    plt.savefig(os.path.join(figures_dir, 'class_distribution.png'))
    plt.close()
    
    # 5.2 PCA
    plt.figure()
    plt.plot(range(1, 17), pca_df["Cumulative_Explained_Variance"], marker='o')
    plt.title("PCA Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.savefig(os.path.join(figures_dir, 'pca_explained_variance.png'))
    plt.close()
    
    # 5.3 Kernel Heatmaps (slice 50x50)
    for m in ["basis", "angle", "amplitude", "phase"]:
        k_path = os.path.join(kernels_dir, f"{m}_K_train.npy")
        if os.path.exists(k_path):
            K = np.load(k_path)[:50, :50]
            plt.figure()
            sns.heatmap(K, cmap="viridis")
            plt.title(f"{m.capitalize()} Kernel Heatmap (50x50 Slice)")
            plt.savefig(os.path.join(figures_dir, f'{m}_kernel_heatmap.png'))
            plt.close()
            
    # 5.4 Same vs Diff class sim
    plt.figure()
    melted = kernel_stats_df.melt(id_vars=["Method"], value_vars=["Same_Class_Sim", "Diff_Class_Sim"], var_name="Similarity_Type", value_name="Similarity")
    sns.barplot(data=melted, x="Method", y="Similarity", hue="Similarity_Type")
    plt.title("Same-Class vs Different-Class Similarity")
    plt.savefig(os.path.join(figures_dir, 'similarity_comparison.png'))
    plt.close()
    
    # 5.5 Classification metrics
    plt.figure(figsize=(10, 6))
    melted_metrics = final_metrics_df.melt(id_vars=["Method"], value_vars=["Accuracy", "Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
    sns.barplot(data=melted_metrics, x="Method", y="Score", hue="Metric")
    plt.title("Classification Metric Comparison")
    plt.ylim(0, 1.05)
    plt.savefig(os.path.join(figures_dir, 'classification_metrics.png'))
    plt.close()
    
    # 5.6 Confusion Matrices
    for m in methods:
        cm_path = os.path.join(tables_dir, f'{m}_confusion_matrix.csv' if m != "classical_rbf" else 'classical_confusion_matrix.csv')
        if os.path.exists(cm_path):
            cm = pd.read_csv(cm_path, index_col=0).values
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f"{m.capitalize()} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.savefig(os.path.join(figures_dir, f'{m}_confusion_matrix.png'))
            plt.close()
            
    # 5.7 Runtime comparison (Training + Prediction + Kernel)
    runtimes = []
    for m in methods:
        row = final_metrics_df[final_metrics_df["Method"] == m.capitalize()].iloc[0]
        t = row["Training_Runtime"] + row["Prediction_Runtime"]
        if m != "classical_rbf":
            k_row = kernel_stats_df[kernel_stats_df["Method"] == m.capitalize()].iloc[0]
            t += k_row["Computation_Runtime_s"]
        runtimes.append({"Method": m.capitalize(), "Total_Runtime_s": t})
    rt_df = pd.DataFrame(runtimes)
    plt.figure()
    sns.barplot(data=rt_df, x="Method", y="Total_Runtime_s")
    plt.title("Total Runtime Comparison (Kernel + Train + Predict)")
    plt.savefig(os.path.join(figures_dir, 'runtime_comparison.png'))
    plt.close()
    
    # 5.8 Qubit comparison
    plt.figure()
    sns.barplot(data=final_metrics_df, x="Method", y="Qubits")
    plt.title("Qubit Count Comparison")
    plt.savefig(os.path.join(figures_dir, 'qubit_comparison.png'))
    plt.close()
    
    # 6. JSON Dump
    final_dict = {
        "Metrics": final_metrics_df.to_dict(orient="records"),
        "Encodings": encoding_df.to_dict(orient="records"),
        "PCA": pca_df.to_dict(orient="records"),
        "Kernel_Stats": kernel_stats_df.to_dict(orient="records")
    }
    with open(os.path.join(results_dir, 'final_results.json'), 'w') as f:
        json.dump(final_dict, f, indent=4)
        
    print("=== FIGURES AND RESULTS GENERATED ===")
    print("FINAL METRICS:")
    print(final_metrics_df.to_string(index=False))
    print("\nKERNEL SIMILARITY:")
    print(kernel_stats_df[['Method', 'Same_Class_Sim', 'Diff_Class_Sim']].to_string(index=False))

if __name__ == "__main__":
    run_generation()
