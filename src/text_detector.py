"""Load the trained text model and score an email / SMS / message.

If a URL detector is supplied, links found in the text are scored too
(multi-layer detection).  Overall probability =
    max(text probability, LINK_RISK_WEIGHT * highest link probability)
so a clean-looking message that carries a clearly malicious link is still escalated.
"""
from pathlib import Path

import joblib

from .config import LINK_RISK_WEIGHT, TEXT_MODEL_PATH, risk_level
from .data_cleaning import clean_text
from .explainability import extract_urls, text_indicators, text_model_drivers
from .url_detector import ModelNotFoundError


class TextDetector:
    def __init__(self, model_path=TEXT_MODEL_PATH, url_detector=None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Text model not found at {model_path}. Train it first:  python -m src.train_text")
        self.bundle = joblib.load(model_path)
        self.vectorizer = self.bundle["vectorizer"]
        self.classifier = self.bundle["classifier"]
        self.url_detector = url_detector

    @property
    def info(self) -> dict:
        b = self.bundle
        return {"name": b["model_name"], "trained_at": b["trained_at"],
                "is_synthetic": b["is_synthetic"], "metrics": b["metrics"]["test"]}

    def predict(self, text: str, channel: str = "message") -> dict:
        text = (text or "").strip()
        if not text:
            raise ValueError("Please enter some text.")
        cleaned = clean_text(text)
        p_text = float(self.classifier.predict_proba(self.vectorizer.transform([cleaned]))[0, 1])

        links = extract_urls(text)
        link_results = []
        if self.url_detector is not None:
            for link in links:
                try:
                    r = self.url_detector.predict(link)
                    link_results.append({"url": link, "probability": r["probability"],
                                         "risk_level": r["risk_level"], "indicators": r["indicators"]})
                except ValueError:
                    continue
        p_link = max((r["probability"] for r in link_results), default=0.0)
        p = max(p_text, LINK_RISK_WEIGHT * p_link)

        return {
            "input": text,
            "channel": channel,
            "label": "Phishing / Scam" if p >= 0.5 else "Legitimate",
            "is_phishing": p >= 0.5,
            "probability": round(p, 4),
            "text_probability": round(p_text, 4),
            "link_probability": round(p_link, 4) if link_results else None,
            "risk_score": int(round(p * 100)),
            "risk_level": risk_level(p),
            "indicators": text_indicators(text, links),
            "model_drivers": text_model_drivers(self.vectorizer, self.classifier, cleaned),
            "links": link_results,
        }
