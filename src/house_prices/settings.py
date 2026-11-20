"""Project-wide settings: folders and the random seed.

Folders are resolved from this file, so scripts work from any working directory.
"""

from pathlib import Path

# Seed for every random step (CV folds, train/holdout split, tree models)
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
