# PhishGuard AI

**An Intelligent Multi-Layer Phishing and Cyber Scam Detection System Using Machine Learning and Natural Language Processing.**

Detects phishing **URLs**, **emails** and **SMS / scam messages**, explains *why* something was flagged, keeps a local scan history and produces downloadable risk reports.

> ## ⚠️ Read this first
> The zip ships with models trained on **synthetic (artificial, template-generated) data**, because no real dataset was included. Metrics on that data prove the software works end to end — **not** that it catches real phishing. In a quick check, the text model scored 100 % on messages built from templates it had seen, but caught only ~36 % of scams written from templates it had never seen. **Train on real datasets (below) before drawing any conclusions or using it for real decisions.**

## Quick start (Windows)
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python build_unified_dataset.py --synthetic     :: demo data (skip if you have real CSVs)
python train_all.py                             :: trains URL + text models into models/
streamlit run app.py
```
Models and a demo dataset are already included, so `streamlit run app.py` works immediately after `pip install`. If loading a model fails after a library upgrade, just re-run `python train_all.py`.

## Using real datasets
1. Put the downloaded CSVs in `./raw/` using the file names in `SOURCES` (top of `build_unified_dataset.py`).
2. Open each CSV and make sure the column names and `malicious_values` in `SOURCES` match it (e.g. `good`/`bad`, `ham`/`spam`, `benign`/`phishing`).
   - UCI "Phishing Websites" uses `-1` = phishing, `1` = legitimate and has **no URL column** — it cannot be used as-is for the URL model.
3. `python build_unified_dataset.py --raw ./raw` → writes `dataset/unified_threat_dataset.csv` (cleans, de-duplicates, 70/15/15 stratified split).
4. `python train_all.py`. The "synthetic data" warnings in the app disappear automatically once `source_dataset` is no longer `SYNTHETIC`.

Tips for honest evaluation: keep the de-duplication on; prefer a time- or source-based hold-out if you can; look at false-positive rate (legit mail flagged) as well as recall.

## How it works
| Layer | Input | Method |
|---|---|---|
| URL | the URL text only (site is never visited) | 29 lexical/host features → Gradient Boosting / Random Forest (best on validation split) |
| Text | email / SMS / message | cleaning → TF-IDF (1-2 grams) → Logistic Regression |
| Multi-layer | links inside a message | each link is scored by the URL model; overall = max(text score, 0.85 × worst link score) |
| Explainability | any scan | rule-based red flags + model drivers (URL: feature perturbation; text: exact word contributions) |
| History | every scan (optional) | local SQLite `database/phishguard.db`, first 500 characters only |

Risk levels: Low < 30 % ≤ Medium < 60 % ≤ High < 85 % ≤ Critical.

## Project structure
```
app.py                       Streamlit UI (Home, URL/Email/SMS scanners, Dashboard, About)
build_unified_dataset.py     merge raw CSVs (or --synthetic) -> dataset/unified_threat_dataset.csv
train_all.py                 train both models
src/
  config.py                  paths, risk levels
  data_cleaning.py           load/validate dataset, clean_text, splits
  synthetic_data.py          synthetic data generator (clearly labelled SYNTHETIC)
  url_features.py            URL feature extraction
  train_url.py / train_text.py   training + evaluation (metrics JSON, confusion matrix PNG)
  url_detector.py / text_detector.py   inference API
  explainability.py          red flags + model drivers
  database.py / report.py    SQLite history / Markdown report
models/                      trained models + metrics
notebooks/data_analysis.ipynb
tests/                       python -m unittest discover -s tests -t .
```

## Tests
```bat
python -m unittest discover -s tests -t .
```
27 tests cover feature extraction, cleaning, database, both detectors and a UI smoke test (UI test uses a fake `streamlit`, so it checks for crashes, not visual layout).

## Known limitations
- URL model sees only the URL string: no page content, WHOIS, domain age or reputation lookups. Shortened links cannot be judged by their text.
- Text model is English-centred; Bengali / Hindi / mixed-script scams need suitable training data.
- Attackers adapt; any static model degrades over time and needs retraining.
- Not a replacement for your own judgement or official bank / government channels.

## Project status
Complete prototype — awaiting training on real datasets.
