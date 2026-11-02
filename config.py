from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrices"

RANDOM_STATE = 42

TRAIN_LIMIT = 10000
TEST_LIMIT = 2000

# Bu projede yine aynı datasetler
DATASETS = ["fashion_mnist", "cifar10", "kth_tips"]
# HOG sabit olacak
FEATURE_METHOD = "hog"