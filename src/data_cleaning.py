"""Dataset loading, validation, text cleaning and train/val/test splitting."""
import html
import math
import re
from pathlib import Path

import pandas as pd

from .config import RANDOM_STATE

REQUIRED_COLUMNS = ["text", "url", "label", "source_type"]

_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY = re.compile(r"(?:[\u20b9$\u20ac\u00a3]\s?\d[\d,\.]*|\brs\.?\s?\d[\d,\.]*)", re.I)
_LONG_NUM = re.compile(r"\d{10,}")
_NUM = re.compile(r"\d+")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    """Normalise a message for the TF-IDF model.

    URLs, emails, money amounts, phone-like numbers and other digits are
    replaced with placeholder tokens so the model learns patterns
    ("urltoken", "moneytoken") instead of memorising specific values.
    """
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    t = html.unescape(_TAG.sub(" ", str(text)))
    t = _URL.sub(" urltoken ", t)
    t = _EMAIL.sub(" emailtoken ", t)
    t = _MONEY.sub(" moneytoken ", t)
    t = _LONG_NUM.sub(" phonetoken ", t)
    t = _NUM.sub(" numtoken ", t)
    t = _NON_ALPHA.sub(" ", t.lower())
    return _SPACES.sub(" ", t).strip()


def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Create it with:  python build_unified_dataset.py --synthetic   (demo data)\n"
            "            or:  python build_unified_dataset.py --raw ./raw    (real CSVs)")
    df = pd.read_csv(path, keep_default_na=False, dtype={"text": str, "url": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("Column 'label' must contain only 0 (safe) and 1 (malicious).")
    if "source_dataset" not in df.columns:
        df["source_dataset"] = "UNKNOWN"
    return ensure_split(df)


def ensure_split(df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Add a 70/15/15 stratified 'split' column if the dataset has none."""
    if "split" in df.columns and df["split"].isin(["train", "val", "test"]).all():
        return df
    df = df.copy()
    df["split"] = "train"
    for _, grp in df.groupby(["source_type", "label"]):
        idx = grp.sample(frac=1, random_state=seed).index
        n = len(idx)
        df.loc[idx[int(n * .70):int(n * .85)], "split"] = "val"
        df.loc[idx[int(n * .85):], "split"] = "test"
    return df
