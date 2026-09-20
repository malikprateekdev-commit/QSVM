import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

RANDOM_SEED = 42
TEST_SIZE = 0.20
PCA_COMPONENTS = 16

SVM_PARAM_GRID = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': ['scale', 0.001, 0.01, 0.1, 1]
}
PRECOMPUTED_PARAM_GRID = {'C': [0.01, 0.1, 1, 10, 100]}
