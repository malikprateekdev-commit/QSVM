import sys
import os
import shutil
import glob
import time
import json
import hashlib
import platform

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import data_loader
from src import preprocessing
from src.encodings import basis, angle, amplitude, phase
from src import quantum_kernel
from src import classical_baseline
from src import quantum_svm
from src import visualization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ARCHIVE_DIR = os.path.join(BASE_DIR, 'results_archive')

def get_run_numbers():
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    max_num = 0
    for d in os.listdir(ARCHIVE_DIR):
        if d.startswith('run_'):
            try:
                num = int(d.split('_')[1])
                if num > max_num:
                    max_num = num
            except ValueError:
                pass
    
    active_num = max_num + 1
    
    # Check if results/ actually has data
    has_data = False
    if os.path.exists(RESULTS_DIR):
        for root, dirs, files in os.walk(RESULTS_DIR):
            if files:
                has_data = True
                break
                
    if has_data:
        return max_num, active_num, active_num + 1
    return max_num, max_num, max_num + 1

def hash_source_files():
    hasher = hashlib.sha256()
    # Hash config and src
    files_to_hash = []
    config_path = os.path.join(BASE_DIR, 'config.py')
    if os.path.exists(config_path):
        files_to_hash.append(config_path)
        
    for root, _, files in os.walk(os.path.join(BASE_DIR, 'src')):
        for f in sorted(files):
            if f.endswith('.py') and '__pycache__' not in root:
                files_to_hash.append(os.path.join(root, f))
                
    for fpath in files_to_hash:
        with open(fpath, 'rb') as f:
            hasher.update(f.read())
            
    return hasher.hexdigest()

def archive_active_results(run_num):
    target_archive = os.path.join(ARCHIVE_DIR, f"run_{run_num:02d}")
    print(f"Archiving current active results to: {target_archive}")
    
    if os.path.exists(target_archive):
        print(f"Error: Archive {target_archive} already exists!")
        sys.exit(1)
        
    os.makedirs(target_archive, exist_ok=True)
    
    for item in os.listdir(RESULTS_DIR):
        s = os.path.join(RESULTS_DIR, item)
        d = os.path.join(target_archive, item)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)

def reset_active_results():
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
        
    for d in ['figures', 'kernels', 'metrics', 'models', 'preprocessed', 'tables', 'logs']:
        os.makedirs(os.path.join(RESULTS_DIR, d), exist_ok=True)

def record_metadata(run_num, start_time, src_hash):
    meta = {
        "run_number": run_num,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "python_version": sys.version,
        "platform": platform.platform(),
        "random_state": 42,
        "dataset_size": 4601,
        "pca_components": 16,
        "source_checksum": src_hash,
        "status": "RUNNING"
    }
    with open(os.path.join(RESULTS_DIR, 'logs', 'run_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=4)

def update_metadata_status(status, total_runtime):
    meta_path = os.path.join(RESULTS_DIR, 'logs', 'run_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        meta['status'] = status
        meta['total_runtime_s'] = total_runtime
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=4)

def main():
    start_time = time.time()
    
    # 1. Determine Run Number
    max_archive, active_num, next_run = get_run_numbers()
    
    # 2. Archive active results (which should be the previous run e.g. Run 4)
    if active_num > max_archive:
        archive_active_results(active_num)
    
    # 3. Reset Active Results
    reset_active_results()
    
    # 4. Hash source code
    src_hash = hash_source_files()
    
    # 5. Record Initial Metadata
    record_metadata(next_run, start_time, src_hash)
    
    print(f"=== STARTING RUN {next_run} ===")
    
    try:
        data_loader.load_spambase()
        preprocessing.run_preprocessing()
        
        basis.run_basis_encoding()
        angle.run_angle_encoding()
        amplitude.run_amplitude_encoding()
        phase.run_phase_encoding()
        
        quantum_kernel.run_quantum_kernels()
        
        classical_baseline.run_classical_baseline()
        quantum_svm.run_quantum_svm()
        
        visualization.run_generation()
        
        end_time = time.time()
        update_metadata_status("SUCCESS", end_time - start_time)
        print(f"=== RUN {next_run} COMPLETED SUCCESSFULLY ===")
        print(f"Total Runtime: {end_time - start_time:.2f}s")
        
    except Exception as e:
        end_time = time.time()
        update_metadata_status("FAILED", end_time - start_time)
        print(f"=== RUN {next_run} FAILED ===")
        print(str(e))
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--archive-only':
        max_archive, active_num, next_run = get_run_numbers()
        if active_num > max_archive:
            archive_active_results(active_num)
        reset_active_results()
        print("Archive and reset complete. Exiting early.")
        sys.exit(0)
    main()
