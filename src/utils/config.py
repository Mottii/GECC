from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

# Paths
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models" / "saved"
ARTIFACTS_DIR = BASE_DIR / "models" / "artifacts"
REPORTS_DIR = PROCESSED_DIR / "reports"

# Dataset
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz"
CANCER_CLASSES = ["BRCA", "KIRC", "COAD", "LUAD", "PRAD"]
NUM_CLASSES = len(CANCER_CLASSES)
INPUT_FEATURES = 20531

# Preprocessing
TOP_K_FEATURES = 2000
TEST_SIZE = 0.2
RANDOM_STATE = 42
RANDOM_SEED = RANDOM_STATE
NORMALIZE = True

# Model
HIDDEN_LAYERS = [1024, 512, 256, 128]
DROPOUT_RATE = 0.4
BATCH_NORM = True
ACTIVATION = "relu"

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100
EARLY_STOPPING = 10
SCHEDULER = "cosine"

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
MODEL_PATH = MODELS_DIR / "deep_net_best.pth"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"
MANIFEST_PATH = ARTIFACTS_DIR / "training_manifest.json"
CV_RESULTS_PATH = REPORTS_DIR / "cv_results.json"
TUNING_RESULTS_PATH = REPORTS_DIR / "tuning_results.json"
CONFIDENCE_THRESHOLD = 0.75
MAX_INPUT_ABS_VALUE = 1_000_000.0


def ensure_project_dirs() -> None:
    """Create runtime directories that are safe to materialize."""
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
