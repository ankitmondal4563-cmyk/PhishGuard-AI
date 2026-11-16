"""
Unified Phishing / Spam / Scam Dataset Builder
------------------------------------------------
Sob 8-ta category (URL, Email, SMS, Scam msg, Malicious URL, Website, Spam/Ham,
Fraud msg) ke ekta single CSV te merge kore.

Usage:
    python build_unified_dataset.py --synthetic       # ~3400 row synthetic dataset (pipeline test-er jonno)
    python build_unified_dataset.py --demo            # matro 14 row-er tiny synthetic sample
    python build_unified_dataset.py --raw ./raw       # real downloaded CSV gulo merge kore

Real mode: ./raw folder e CSV gulo rakho, tarpor SOURCES config e column name mapping thik koro.
"""
import argparse, hashlib, re, os
from urllib.parse import urlparse
import pandas as pd

# ---------- 1. Source config: file -> column mapping ----------
# text_col / url_col / label_col tomar downloaded CSV er column name onujayi bodlao.
# malicious_values = je label gulo "bad" (1) hishebe dhora hobe.
SOURCES = [
    dict(file="phishing_urls.csv",       kind="url",     url_col="url",   label_col="label",  malicious_values=["bad", "phishing", "1", "malicious"]),
    dict(file="phishing_emails.csv",     kind="email",   text_col="Email Text", label_col="Email Type", malicious_values=["Phishing Email", "phishing", "1"]),
    dict(file="sms_phishing.csv",        kind="sms",     text_col="TEXT", label_col="LABEL", malicious_values=["smishing", "spam", "1"]),
    dict(file="scam_messages.csv",       kind="message", text_col="message", label_col="label", malicious_values=["scam", "fraud", "1"]),
    dict(file="malicious_urls.csv",      kind="url",     url_col="url",   label_col="type",   malicious_values=["phishing", "malware", "defacement", "1"]),
    # NOTE: UCI "Phishing Websites" uses -1 = phishing, 1 = legitimate (and has NO url column,
    # only 30 pre-computed features). Listing both "1" and "-1" would mark EVERY row malicious.
    # Use a file that actually has a URL column, and check which value means "phishing" in YOUR csv.
    dict(file="phishing_websites.csv",   kind="website", url_col="URL",   label_col="label",  malicious_values=["-1", "phishing"]),
    dict(file="spam_ham_emails.csv",     kind="email",   text_col="text", label_col="label", malicious_values=["spam", "1"]),
    dict(file="fraud_messages.csv",      kind="message", text_col="text", label_col="label", malicious_values=["fraud", "1"]),
]

# ---------- 2. URL feature extraction ----------
SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", "bank", "paypal",
                    "confirm", "password", "signin", "wallet", "free", "bonus"]

def url_features(url: str) -> dict:
    url = str(url) if pd.notna(url) else ""
    host = ""
    try:
        host = urlparse(url if "://" in url else "http://" + url).netloc
    except Exception:
        pass
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "has_at_symbol": int("@" in url),
        "has_ip_host": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", host))),
        "uses_https": int(url.lower().startswith("https")),
        "num_subdomains": max(host.count(".") - 1, 0),
        "suspicious_word_count": sum(w in url.lower() for w in SUSPICIOUS_WORDS),
    }

# ---------- 3. Text feature extraction ----------
def text_features(text: str) -> dict:
    text = str(text) if pd.notna(text) else ""
    return {
        "text_length": len(text),
        "word_count": len(text.split()),
        "num_urls_in_text": len(re.findall(r"https?://\S+|www\.\S+", text)),
        "has_urgency_words": int(bool(re.search(r"urgent|immediately|act now|suspended|expires|last chance|verify now", text, re.I))),
        "exclamation_count": text.count("!"),
        "uppercase_ratio": round(sum(c.isupper() for c in text) / max(len(text), 1), 4),
        "has_phone_or_money": int(bool(re.search(r"\$\d+|\u20b9\s?\d+|\d{10}|prize|lottery|winner", text, re.I))),
    }

# ---------- 4. Normalise one source ----------
def load_source(cfg: dict, raw_dir: str) -> pd.DataFrame:
    path = os.path.join(raw_dir, cfg["file"])
    if not os.path.exists(path):
        print(f"[skip] {cfg['file']} paoa jayni")
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")
    out = pd.DataFrame(index=df.index)   # index diye shuru - na hole scalar "" assign korle 0-row column hoy
    out["text"] = df[cfg["text_col"]] if "text_col" in cfg else ""
    out["url"] = df[cfg["url_col"]] if "url_col" in cfg else ""
    mal = {str(v).lower() for v in cfg["malicious_values"]}
    out["original_label"] = df[cfg["label_col"]].astype(str)
    out["label"] = out["original_label"].str.lower().isin(mal).astype(int)
    out["source_type"] = cfg["kind"]
    out["source_dataset"] = cfg["file"]
    return out

# ---------- 5. Demo data (SYNTHETIC - real dataset noy) ----------
def demo_frame() -> pd.DataFrame:
    rows = [
        ("", "http://secure-paypal-login.verify-account.xyz/update", "url", 1),
        ("", "https://www.wikipedia.org/wiki/Machine_learning", "url", 0),
        ("", "http://192.168.10.5/bank/login.php?user=@admin", "url", 1),
        ("", "https://github.com/pallets/flask", "url", 0),
        ("Dear customer, your account is suspended. Verify now at http://bit.ly/x1 or lose access!", "", "email", 1),
        ("Hi team, attaching the meeting notes from Tuesday. Let me know if I missed anything.", "", "email", 0),
        ("Congratulations! You won a prize of $5000. Reply immediately with your bank details!", "", "sms", 1),
        ("Hey, are we still meeting for lunch at 1pm?", "", "sms", 0),
        ("Your KYC is expiring today. Click www.kyc-update-india.top to avoid block. Act now!", "", "message", 1),
        ("Your OTP for login is 482910. Do not share it with anyone.", "", "message", 0),
        ("", "https://paypa1-secure.com/signin", "website", 1),
        ("", "https://www.bbc.com/news", "website", 0),
        ("WINNER!! Claim your FREE lottery bonus now!!!", "", "email", 1),
        ("Lunch menu for Friday: biriyani, salad and dessert.", "", "email", 0),
    ]
    df = pd.DataFrame(rows, columns=["text", "url", "source_type", "label"])
    df["original_label"] = df["label"].map({1: "malicious", 0: "benign"})
    df["source_dataset"] = "SYNTHETIC_DEMO"
    return df

# ---------- 6. Build ----------
def build(raw_dir: str | None, demo: bool, out_dir: str, synthetic: bool = False):
    if synthetic:
        from src.synthetic_data import make_synthetic_frame   # project root theke chalate hobe
        frames = [make_synthetic_frame()]
    elif demo:
        frames = [demo_frame()]
    else:
        frames = [load_source(c, raw_dir) for c in SOURCES]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit(f"Kono source CSV paoa jayni '{raw_dir}' folder-e. "
                         "CSV rakho, ba --synthetic / --demo diye chalao.")
    df = pd.concat(frames, ignore_index=True)

    # clean + dedup (train/test leakage thamate)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df = df[(df["text"] != "") | (df["url"] != "")]
    df["dedup_key"] = (df["text"].str.lower() + "||" + df["url"].str.lower()).map(
        lambda s: hashlib.md5(s.encode()).hexdigest())
    before = len(df)
    df = df.drop_duplicates("dedup_key").drop(columns="dedup_key").reset_index(drop=True)
    print(f"Duplicates remove: {before - len(df)}")

    # features
    # NOTE: .apply(pd.Series) boro dataset-e onek slow, tai list -> DataFrame
    uf = pd.DataFrame(df["url"].map(url_features).tolist(), index=df.index)
    tf = pd.DataFrame(df["text"].map(text_features).tolist(), index=df.index)
    df = pd.concat([df, uf, tf], axis=1)
    df.insert(0, "id", range(1, len(df) + 1))

    # stratified split 70/15/15
    df["split"] = "train"
    for _, grp in df.groupby(["source_type", "label"]):
        idx = grp.sample(frac=1, random_state=42).index
        n = len(idx)
        df.loc[idx[int(n * .70):int(n * .85)], "split"] = "val"
        df.loc[idx[int(n * .85):], "split"] = "test"

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "unified_threat_dataset.csv")
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows -> {path}")
    print(df.groupby(["source_type", "label"]).size())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="./raw")
    ap.add_argument("--out", default="./dataset")
    ap.add_argument("--demo", action="store_true", help="14-row tiny synthetic sample")
    ap.add_argument("--synthetic", action="store_true", help="larger synthetic dataset (src/synthetic_data.py)")
    a = ap.parse_args()
    build(a.raw, a.demo, a.out, a.synthetic)
