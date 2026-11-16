"""Explain WHY something was flagged.

Two complementary layers:
  * rule-based "indicators" - plain-language red flags (heuristics, for the human reader)
  * model "drivers" - what actually pushed the ML score up/down
      - URL model: perturbation test (reset one feature to a benign baseline, see how
        the probability changes)
      - text model: exact per-word contribution of the linear model (coef x tf-idf)
Indicators are NOT the verdict; the verdict always comes from the trained model.
"""
import re

import numpy as np
import pandas as pd

from .url_features import FEATURE_LABELS, SHORTENERS

# ------------------------------------------------------------------ URL
def url_indicators(f: dict) -> list:
    out = []
    if f["has_ip_host"]:
        out.append("Uses a raw IP address instead of a domain name")
    if f["has_at_symbol"]:
        out.append("Contains '@' - everything before it is ignored by browsers, a common disguise trick")
    if f["brand_mismatch"]:
        out.append("Mentions a well-known brand but the real domain is different (possible impersonation)")
    if f["has_shortener"]:
        out.append("Shortened link - the real destination is hidden")
    if f["has_punycode"]:
        out.append("Punycode (xn--) domain - may use look-alike characters")
    if f["suspicious_tld"]:
        out.append("Domain ending is frequently abused for phishing")
    if f["has_scheme"] and not f["uses_https"]:
        out.append("Not using HTTPS")
    if f["num_subdomains"] >= 3:
        out.append(f"Many subdomains ({f['num_subdomains']}) - often used to bury the real domain")
    if f["suspicious_words_host"] >= 1 and (f["hyphen_in_host"] or f["suspicious_tld"] or f["brand_mismatch"]):
        out.append("Sensitive words (login/verify/secure/...) inside a suspicious-looking domain name")
    if f["suspicious_words_path"] >= 2:
        out.append("Several sensitive words in the path")
    if f["num_hyphens"] >= 3:
        out.append("Many hyphens in the address")
    if f["url_length"] > 90:
        out.append(f"Very long URL ({f['url_length']} characters)")
    if f["embedded_url_in_path"]:
        out.append("Another URL / redirect is embedded inside the address")
    if f["has_port"]:
        out.append("Uses an explicit, non-standard port")
    return out


def url_model_drivers(model, feature_names, baseline, features: dict, top_k: int = 6) -> list:
    """Perturbation-based local explanation (no extra dependency needed)."""
    row = pd.DataFrame([features], columns=feature_names)
    p_full = float(model.predict_proba(row)[0, 1])
    variants = []
    for name in feature_names:
        v = row.copy()
        v[name] = baseline.get(name, 0.0)
        variants.append(v)
    p_without = model.predict_proba(pd.concat(variants, ignore_index=True))[:, 1]
    contrib = p_full - p_without           # >0 : this feature pushed towards "phishing"
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [{"feature": FEATURE_LABELS.get(feature_names[i], feature_names[i]),
             "value": features[feature_names[i]],
             "contribution": round(float(contrib[i]), 4)}
            for i in order if abs(contrib[i]) >= 0.005]


# ------------------------------------------------------------------ text
_URLISH = re.compile(
    r"(?<![@\w.-])(?:https?://|www\.)[^\s<>\"')]+"
    r"|(?<![@\w.-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|in|xyz|top|tk|ml|ga|cf|gq|click|link|work|icu|buzz|online|site|info|cc|ru|co|ly|app|me|io)\b(?:/[^\s<>\"')]*)?",
    re.I)

_RULES = [
    (re.compile(r"\b(urgent|immediately|act now|asap|within \d+ hours?|expires?|expiring|last chance|final notice|suspended|blocked|disconnected|closed within)\b", re.I),
     "Creates urgency or threatens consequences"),
    (re.compile(r"\b(won|winner|prize|lottery|lucky draw|reward points?|cashback|jackpot|free gift|voucher)\b", re.I),
     "Promises a prize, reward or free gift"),
    (re.compile(r"\b(double (your )?(money|returns)|guaranteed (profit|returns?)|crypto|bitcoin|investment scheme)\b|earn\s*[\u20b9$]?\s*[\d,]+\s*(per|a|/)\s*day", re.I),
     "Offers unrealistic earnings or investment returns"),
    (re.compile(r"\b(income tax|customs|police|cbi|rbi|government)\b", re.I),
     "Claims to be a tax / customs / police / government authority"),
    (re.compile(r"\b(kyc|re-?verif\w+|sim (will be )?blocked)\b", re.I),
     "Mentions KYC / re-verification pressure"),
]
_ASKS_SECRET = re.compile(
    r"\b(share|send|submit|enter|confirm|provide|reply with|update)\b[^.\n]{0,50}\b"
    r"(otp|password|pin|cvv|bank details|card (number|details)|aadhaar|account number|passport)", re.I)
_NEGATED_SHARE = re.compile(r"\b(do not|don'?t|never)\s+(share|disclose|give)\b", re.I)
_PHONE_CONTACT = re.compile(r"\b(call|whatsapp|message|contact)\b[^.\n]{0,30}\b[6-9]\d{9}\b", re.I)


def extract_urls(text: str, limit: int = 5) -> list:
    found = []
    for m in _URLISH.finditer(text or ""):
        u = m.group(0).rstrip(".,;:!?)")
        if u and u not in found:
            found.append(u)
    return found[:limit]


def text_indicators(text: str, links: list) -> list:
    text = text or ""
    out = [msg for rx, msg in _RULES if rx.search(text)]
    if _ASKS_SECRET.search(text) and not _NEGATED_SHARE.search(text):
        out.append("Asks you to share or enter sensitive information (OTP / password / bank details)")
    if _PHONE_CONTACT.search(text):
        out.append("Pushes you to contact a phone number / WhatsApp")
    if links:
        out.append(f"Contains {len(links)} link(s)")
        if any(re.sub(r"^https?://", "", l, flags=re.I).split("/")[0].lower() in SHORTENERS for l in links):
            out.append("Contains a shortened link - the real destination is hidden")
    letters = [c for c in text if c.isalpha()]
    if len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.4:
        out.append("Excessive capital letters")
    if text.count("!") >= 3:
        out.append("Many exclamation marks")
    return out


def text_model_drivers(vectorizer, classifier, cleaned: str, top_k: int = 6) -> dict:
    """Exact word-level contributions of the linear model (coef x tf-idf)."""
    X = vectorizer.transform([cleaned])
    if X.nnz == 0:
        return {"towards_phishing": [], "towards_legitimate": []}
    names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]
    items = [(names[j], float(coefs[j] * X[0, j])) for j in X.nonzero()[1]]
    pos = sorted([i for i in items if i[1] > 0], key=lambda i: -i[1])[:top_k]
    neg = sorted([i for i in items if i[1] < 0], key=lambda i: i[1])[:top_k]
    fmt = lambda xs: [{"term": t, "contribution": round(c, 4)} for t, c in xs]
    return {"towards_phishing": fmt(pos), "towards_legitimate": fmt(neg)}
