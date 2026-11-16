"""Load the trained URL model and score a URL."""
import re
from pathlib import Path

import joblib

from .config import URL_MODEL_PATH, risk_level
from .explainability import url_indicators, url_model_drivers
from .url_features import extract_features


class ModelNotFoundError(FileNotFoundError):
    pass


class URLDetector:
    def __init__(self, model_path=URL_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(
                f"URL model not found at {model_path}. Train it first:  python -m src.train_url")
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.feature_names = self.bundle["feature_names"]

    @property
    def info(self) -> dict:
        b = self.bundle
        return {"name": b["model_name"], "trained_at": b["trained_at"],
                "is_synthetic": b["is_synthetic"], "metrics": b["metrics"]["test"]}

    def predict(self, url: str) -> dict:
        url = (url or "").strip()
        if not url:
            raise ValueError("Please enter a URL.")
        if re.search(r"\s", url):
            raise ValueError("A URL cannot contain spaces.")
        features = extract_features(url)
        import pandas as pd
        row = pd.DataFrame([features], columns=self.feature_names)
        p = float(self.model.predict_proba(row)[0, 1])
        return {
            "input": url,
            "label": "Phishing" if p >= 0.5 else "Legitimate",
            "is_phishing": p >= 0.5,
            "probability": round(p, 4),
            "risk_score": int(round(p * 100)),
            "risk_level": risk_level(p),
            "indicators": url_indicators(features),
            "model_drivers": url_model_drivers(
                self.model, self.feature_names, self.bundle["baseline"], features),
            "features": features,
        }
