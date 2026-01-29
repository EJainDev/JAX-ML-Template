from pathlib import Path

import jax

MAJOR_VERSION = 0
MINOR_VERSION = 0
PATCH_VERSION = 0
TWEAK_VERSION = 0

ROOT_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "Earthquakes_USGS.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "data.parquet"

SCALER_PATH = PROCESSED_DATA_DIR / "scaler.joblib"

MODELS_DIR = ROOT_DIR / "models"

LOGS_DIR = (
    ROOT_DIR
    / "logs"
    / f"logs_v{MAJOR_VERSION}_{MINOR_VERSION}_{PATCH_VERSION}_{TWEAK_VERSION}"
)

BATCH_SIZE = 32

CHECKPOINT_DIR = LOGS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TEST_RESULTS_DIR = LOGS_DIR / "results"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TENSORBOARD_DIR = LOGS_DIR / "tensorboard"
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_VAL_TEST_SPLIT = (0.9, 0.05, 0.05)

LEARNING_RATE = 0.001
B1 = 0.9
B2 = 0.999
EPSILON = 1e-8

METADATA = {
    "major_version": MAJOR_VERSION,
    "minor_version": MINOR_VERSION,
    "patch_version": PATCH_VERSION,
    "tweak_version": TWEAK_VERSION,
    "batch_size": BATCH_SIZE,
    "model_name": "LinearRegression",
    "train_split": TRAIN_VAL_TEST_SPLIT[0],
    "val_split": TRAIN_VAL_TEST_SPLIT[1],
    "test_split": TRAIN_VAL_TEST_SPLIT[2],
    "seed": 0,
    "learning_rate": LEARNING_RATE,
    "beta1": B1,
    "beta2": B2,
    "epsilon": EPSILON,
    "hidden_sizes": [32],
}

RNG = jax.random.key(METADATA["seed"])  # JAX random key, not flax random key
