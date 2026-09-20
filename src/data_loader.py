import os
import pandas as pd
import numpy as np

# Official Spambase documentation feature names
FEATURE_NAMES = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$",
    "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total"
]
TARGET_NAME = "is_spam"

def load_spambase(filepath="data/raw/spambase.data"):
    """
    Loads all records from the raw UCI Spambase dataset.
    Validates the dataset size inherently by pulling the entire file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    column_names = FEATURE_NAMES + [TARGET_NAME]
    
    # Read the data natively
    df = pd.read_csv(filepath, header=None, names=column_names)
    
    # Force numeric constraint and coercion
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Separate exactly 57 features and 1 target
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME].astype(int)
    
    return X, y

def audit_dataset(X, y, output_dir="results/tables"):
    """
    Audits the full dataset for experimental constraints, saving numerical reports.
    Does NOT modify the dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Target Class Validation
    classes = sorted(y.unique().tolist())
    class_counts = y.value_counts()
    class_percentages = y.value_counts(normalize=True) * 100
    
    class_df = pd.DataFrame({
        "Class": class_counts.index,
        "Count": class_counts.values,
        "Percentage": class_percentages.values
    }).sort_values("Class")
    class_df.to_csv(os.path.join(output_dir, "class_distribution.csv"), index=False)
    
    # 2. Constraint Checks
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_missing = X.isnull().sum().sum() + y.isnull().sum()
    n_nan = X.isna().sum().sum() + y.isna().sum()
    n_inf = np.isinf(X).sum().sum()
    
    # Duplicates check (concatenate X and y)
    n_duplicates = pd.concat([X, y], axis=1).duplicated().sum()
    
    # Constant features check
    constant_cols = sum([1 for col in X.columns if X[col].nunique(dropna=False) <= 1])
    
    # Build strict audit report
    audit_data = {
        "Metric": [
            "Total Records", 
            "Total Features", 
            "Target Columns", 
            "Expected Classes (0, 1) Present",
            "Missing Values", 
            "NaN Values", 
            "Infinite Values", 
            "Duplicate Rows",
            "Constant Features"
        ],
        "Value": [
            n_samples,
            n_features,
            1,
            classes == [0, 1],
            n_missing,
            n_nan,
            n_inf,
            n_duplicates,
            constant_cols
        ]
    }
    audit_df = pd.DataFrame(audit_data)
    audit_df.to_csv(os.path.join(output_dir, "full_dataset_audit.csv"), index=False)
    
    # 3. Data types and Ranges
    ranges_df = pd.DataFrame({
        "Feature": X.columns,
        "Type": X.dtypes.astype(str),
        "Min": X.min().values,
        "Max": X.max().values,
        "Missing": X.isnull().sum().values
    })
    ranges_df.to_csv(os.path.join(output_dir, "feature_ranges.csv"), index=False)

    return audit_df, class_df

if __name__ == "__main__":
    # Deterministic execution path evaluation
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(base_dir, 'data', 'raw', 'spambase.data')
    output_path = os.path.join(base_dir, 'results', 'tables')
    
    try:
        X, y = load_spambase(data_path)
        audit_df, class_df = audit_dataset(X, y, output_dir=output_path)
        
        print("\n=== CLASS DISTRIBUTION ===")
        print(class_df.to_string(index=False))
        
        print("\n=== DATASET AUDIT ===")
        print(audit_df.to_string(index=False))
        
    except Exception as e:
        print(f"Error during data loading/auditing: {e}")
