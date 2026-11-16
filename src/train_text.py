"""Train the text (email / SMS / scam message) classifier.

    python -m src.train_text
    python -m src.train_text --data my.csv
"""
import argparse
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import DATASET_PATH, RANDOM_STATE, TEXT_MODEL_PATH
from .data_cleaning import clean_text, load_dataset
from .evaluation import evaluate, save_confusion_matrix, save_json


def train(dataset_path=DATASET_PATH, model_path=TEXT_MODEL_PATH, seed=RANDOM_STATE, verbose=True):
    df = load_dataset(dataset_path)
    d = df[df["text"] != ""].reset_index(drop=True)
    if d["label"].nunique() < 2 or len(d) < 50:
        raise ValueError(f"Need at least 50 text rows with both classes (found {len(d)}).")
    d["clean"] = d["text"].map(clean_text)
    d = d[d["clean"] != ""].reset_index(drop=True)

    tr, va, te = (d["split"] == s for s in ("train", "val", "test"))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=30000)
    Xtr = vec.fit_transform(d.loc[tr, "clean"])      # fit on TRAIN only (no leakage)
    Xva, Xte = vec.transform(d.loc[va, "clean"]), vec.transform(d.loc[te, "clean"])
    y = d["label"]

    val_scores, models = {}, {}
    for C in (0.3, 1.0, 3.0, 10.0):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(Xtr, y[tr])
        val_scores[C] = evaluate(y[va], clf.predict_proba(Xva)[:, 1])
        models[C] = clf
        if verbose:
            print(f"[val] LogReg C={C:<5} F1={val_scores[C]['f1']:.4f}  AUC={val_scores[C]['roc_auc']}")
    best_C = max(val_scores, key=lambda c: (val_scores[c]["f1"], val_scores[c]["roc_auc"] or 0))
    clf = models[best_C]

    proba_te = clf.predict_proba(Xte)[:, 1]
    test_metrics = evaluate(y[te], proba_te)
    dte = d[te].reset_index(drop=True)
    per_channel = {ch: evaluate(dte.loc[dte["source_type"] == ch, "label"],
                                proba_te[(dte["source_type"] == ch).to_numpy()])
                   for ch in sorted(dte["source_type"].unique())
                   if (dte["source_type"] == ch).sum() >= 5}

    sources = sorted(d["source_dataset"].unique())
    is_synth = all(str(s).upper().startswith("SYNTHETIC") for s in sources)
    bundle = {
        "kind": "text",
        "vectorizer": vec,
        "classifier": clf,
        "model_name": f"TF-IDF + LogisticRegression (C={best_C})",
        "metrics": {"test": test_metrics, "per_channel": per_channel,
                    "validation": {str(k): v for k, v in val_scores.items()},
                    "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())},
        "data_sources": sources,
        "is_synthetic": is_synth,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = Path(model_path).parent          # metrics/plots go next to the model file
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    summary = {k: v for k, v in bundle.items() if k not in ("vectorizer", "classifier")}
    save_json(summary, out_dir / "text_metrics.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], out_dir / "text_confusion_matrix.png",
                          "Text model")
    if verbose:
        print(f"[test] {bundle['model_name']}: acc={test_metrics['accuracy']} P={test_metrics['precision']} "
              f"R={test_metrics['recall']} F1={test_metrics['f1']} AUC={test_metrics['roc_auc']}")
        for ch, m in per_channel.items():
            print(f"        {ch:8s} n={m['n']:4d}  F1={m['f1']}")
        if is_synth:
            print("WARNING: trained on SYNTHETIC data - metrics do not reflect real-world performance.")
        print(f"Saved -> {model_path}")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATASET_PATH))
    ap.add_argument("--out", default=str(TEXT_MODEL_PATH))
    a = ap.parse_args()
    train(a.data, a.out)
