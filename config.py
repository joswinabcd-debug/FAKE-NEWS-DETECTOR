"""
TRUTHSCAN AI - Configuration File
Central configuration for paths, model architectures, hyperparameters, and training settings.
"""

import os
import torch

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
TOKENIZER_DIR = os.path.join(BASE_DIR, "tokenizer")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Ensure required directories exist
for directory in [DATA_DIR, PROCESSED_DIR, CHECKPOINTS_DIR, TOKENIZER_DIR, RESULTS_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)

ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")

# Dataset Resolution: Supports both archive/ and data/, and CSV or Excel files
def find_dataset_file(basename: str) -> str:
    """Finds dataset file checking archive/ and data/ directories for .csv, .xlsx, .xls."""
    candidates = [
        os.path.join(ARCHIVE_DIR, f"{basename}.csv"),
        os.path.join(ARCHIVE_DIR, f"{basename}.xlsx"),
        os.path.join(ARCHIVE_DIR, f"{basename}.xls"),
        os.path.join(DATA_DIR, f"{basename}.csv"),
        os.path.join(DATA_DIR, f"{basename}.xlsx"),
        os.path.join(DATA_DIR, f"{basename}.xls"),
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    # Default fallback
    return os.path.join(DATA_DIR, f"{basename}.csv")

FAKE_PATH = find_dataset_file("Fake")
TRUE_PATH = find_dataset_file("True")
FAKE_CSV = FAKE_PATH
TRUE_CSV = TRUE_PATH

# Model Checkpoints
CNN_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "cnn_best.pth")
LSTM_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "lstm_best.pth")

# Tokenizer & Vocabulary
VOCABULARY_FILE = os.path.join(TOKENIZER_DIR, "vocabulary.json")

# Results & Histories
MODEL_COMPARISON_CSV = os.path.join(RESULTS_DIR, "model_comparison.csv")
CNN_HISTORY_FILE = os.path.join(RESULTS_DIR, "cnn_history.json")
LSTM_HISTORY_FILE = os.path.join(RESULTS_DIR, "lstm_history.json")

# Tokenization & Sequence Configuration
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 200
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1

# Dataset Splits (Stratified)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Reproducibility Seed
RANDOM_SEED = 42

# Device Configuration: Auto-detect CUDA GPU, fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Architecture Configurations
CNN_CONFIG = {
    "embedding_dim": 128,
    "num_filters": 100,
    "kernel_sizes": [3, 4, 5],
    "dropout": 0.5,
}

LSTM_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.3,
}

# Training Hyperparameter Presets
TRAINING_MODES = {
    "quick": {
        "samples": 5000,
        "epochs": 2,
        "batch_size": 16,
        "learning_rate": 0.001,
        "description": "Quick Mode (5,000 real ISOT samples, 2 epochs, batch size 16)",
    },
    "normal": {
        "samples": 20000,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "description": "Normal Mode (20,000 real ISOT samples, 5 epochs, batch size 32)",
    },
    "full": {
        "samples": None,  # Use all available data
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "description": "Full Mode (All available ISOT samples, 5 epochs, batch size 32)",
    }
}
