"""Central paths and settings for PhishGuard AI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = ROOT / "dataset" / "unified_threat_dataset.csv"
MODELS_DIR = ROOT / "models"
URL_MODEL_PATH = MODELS_DIR / "url_model.joblib"
TEXT_MODEL_PATH = MODELS_DIR / "text_model.joblib"
DB_PATH = ROOT / "database" / "phishguard.db"

RANDOM_STATE = 42

# (minimum probability, label) - checked from highest to lowest
RISK_LEVELS = [(0.85, "Critical"), (0.60, "High"), (0.30, "Medium"), (0.0, "Low")]

# Embedded links are only allowed to raise a text's overall risk by this factor.
LINK_RISK_WEIGHT = 0.85


def risk_level(probability: float) -> str:
    """Map a phishing probability (0-1) to a human readable risk level."""
    for threshold, name in RISK_LEVELS:
        if probability >= threshold:
            return name
    return "Low"
