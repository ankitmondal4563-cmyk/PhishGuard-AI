"""Train the URL phishing classifier.

    python -m src.train_url                     # uses dataset/unified_threat_dataset.csv
    python -m src.train_url --data my.csv
"""
import argparse
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from .config import DATASET_PATH, RANDOM_STATE, URL_MODEL_PATH
from .data_cleaning import load_dataset
from .evaluation import evaluate, save_confusion_matrix, save_json
from .url_features import FEATURE_NAMES, features_dataframe


def train(dataset_path=DATASET_PATH, model_path=URL_MODEL_PATH, seed=RANDOM_STATE, verbose=True):
    df = load_dataset(dataset_path)
    d = df[df["url"] != ""].reset_index(drop=True)
    if d["label"].nunique() < 2 or len(d) < 50:
        raise ValueError(f"Need at least 50 URL rows with both classes (found {len(d)}).")

    X = features_dataframe(d["url"])
    y = d["label"]
    tr, va, te = (d["split"] == s for s in ("train", "val", "test"))

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
    }
    val_scores = {}
    for name, model in candidates.items():
        model.fit(X[tr], y[tr])
        val_scores[name] = evaluate(y[va], model.predict_proba(X[va])[:, 1])
        if verbose:
            print(f"[val] {name:17s} F1={val_scores[name]['f1']:.4f}  AUC={val_scores[name]['roc_auc']}")
    best = max(val_scores, key=lambda n: (val_scores[n]["f1"], val_scores[n]["roc_auc"] or 0))
    model = candidates[best]

    test_metrics = evaluate(y[te], model.predict_proba(X[te])[:, 1])
    sources = sorted(d["source_dataset"].unique())
    is_synth = all(str(s).upper().startswith("SYNTHETIC") for s in sources)

    importance = dict(sorted(zip(FEATURE_NAMES, map(float, model.feature_importances_)),
                             key=lambda kv: -kv[1]))
    bundle = {
        "kind": "url",
        "model": model,
        "model_name": best,
        "feature_names": FEATURE_NAMES,
        # median of benign training rows = neutral baseline for the explanation step
        "baseline": X[tr & (y == 0)].median().to_dict(),
        "metrics": {"test": test_metrics, "validation": val_scores,
                    "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum())},
        "feature_importance": importance,
        "data_sources": sources,
        "is_synthetic": is_synth,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    out_dir = Path(model_path).parent          # metrics/plots go next to the model file
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    summary = {k: v for k, v in bundle.items() if k not in ("model", "baseline")}
    save_json(summary, out_dir / "url_metrics.json")
    save_confusion_matrix(test_metrics["confusion_matrix"], out_dir / "url_confusion_matrix.png",
                          f"URL model ({best})")
    if verbose:
        print(f"[test] {best}: acc={test_metrics['accuracy']} P={test_metrics['precision']} "
              f"R={test_metrics['recall']} F1={test_metrics['f1']} AUC={test_metrics['roc_auc']}")
        if is_synth:
            print("WARNING: trained on SYNTHETIC data - metrics do not reflect real-world performance.")
        print(f"Saved -> {model_path}")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATASET_PATH))
    ap.add_argument("--out", default=str(URL_MODEL_PATH))
    a = ap.parse_args()
    train(a.data, a.out)
