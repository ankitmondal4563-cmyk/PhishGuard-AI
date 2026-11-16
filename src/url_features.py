"""Lexical / host-based feature extraction for URLs.

The same function is used for training and for live scanning, so the model
always sees identical features.  Nothing here makes a network request.
"""
import math
import re
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "bank",
    "confirm", "password", "wallet", "free", "bonus", "kyc", "refund",
    "billing", "recovery", "reward", "claim", "suspend", "unlock",
]
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc",
}
SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work",
    "icu", "buzz", "loan", "win", "bid", "monster", "rest", "cyou", "zip",
}
# Two-label public suffixes (small built-in list; avoids needing tldextract/network)
SECOND_LEVEL_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "co.in", "org.in",
    "gov.in", "ac.in", "nic.in", "com.bd", "com.br", "co.jp", "com.cn", "co.za",
}
BRANDS = [
    "paypal", "amazon", "google", "microsoft", "apple", "facebook", "instagram",
    "netflix", "whatsapp", "hdfcbank", "sbi", "icici", "axisbank", "paytm",
    "flipkart", "irctc", "linkedin", "dropbox", "binance", "coinbase",
]
_LEET_L = str.maketrans("013457@", "oleasta")    # 1 -> l
_LEET_I = str.maketrans("013457@", "oieasta")    # 1 -> i
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length", "path_depth",
    "num_dots", "num_hyphens", "num_underscores", "num_digits", "digit_ratio",
    "num_special_chars", "num_percent", "num_params",
    "has_at_symbol", "has_ip_host", "has_port", "has_scheme", "uses_https",
    "num_subdomains", "hyphen_in_host", "digits_in_host",
    "suspicious_words_host", "suspicious_words_path",
    "has_shortener", "has_punycode", "suspicious_tld", "brand_mismatch",
    "embedded_url_in_path", "host_entropy",
]

FEATURE_LABELS = {
    "url_length": "URL length",
    "host_length": "Domain length",
    "path_length": "Path length",
    "query_length": "Query-string length",
    "path_depth": "Folder depth",
    "num_dots": "Number of dots",
    "num_hyphens": "Number of hyphens",
    "num_underscores": "Number of underscores",
    "num_digits": "Number of digits",
    "digit_ratio": "Share of digits",
    "num_special_chars": "Special characters",
    "num_percent": "Percent-encoded characters",
    "num_params": "Query parameters",
    "has_at_symbol": "Contains '@'",
    "has_ip_host": "IP address as host",
    "has_port": "Explicit port",
    "has_scheme": "Has http(s):// prefix",
    "uses_https": "Uses HTTPS",
    "num_subdomains": "Number of subdomains",
    "hyphen_in_host": "Hyphen in domain",
    "digits_in_host": "Digits in domain",
    "suspicious_words_host": "Suspicious words in domain",
    "suspicious_words_path": "Suspicious words in path",
    "has_shortener": "URL shortener",
    "has_punycode": "Punycode (xn--) domain",
    "suspicious_tld": "Risky domain ending (TLD)",
    "brand_mismatch": "Brand name on unrelated domain",
    "embedded_url_in_path": "Another URL inside the path",
    "host_entropy": "Domain randomness (entropy)",
}


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in
                (s.count(ch) for ch in set(s)))


def _clean_input(url) -> str:
    if url is None or (isinstance(url, float) and math.isnan(url)):
        return ""
    return str(url).strip()


def parse_url(url):
    """Return (raw, parsed, hostname, netloc_without_port)."""
    raw = _clean_input(url)
    target = raw if _SCHEME.match(raw) else "http://" + raw
    try:
        parsed = urlparse(target)
        host = (parsed.hostname or "").lower()
        netloc = parsed.netloc.lower()
    except ValueError:
        parsed, host, netloc = urlparse("http://"), "", ""
    netloc = re.sub(r":\d+$", "", netloc)
    return raw, parsed, host, netloc


def registered_domain_parts(host: str):
    """Split a hostname into (subdomain_labels, name_label, suffix)."""
    labels = [l for l in host.split(".") if l]
    if len(labels) < 2 or _IPV4.match(host):
        return labels, host, ""
    suffix_len = 2 if ".".join(labels[-2:]) in SECOND_LEVEL_SUFFIXES and len(labels) >= 3 else 1
    suffix = ".".join(labels[-suffix_len:])
    name = labels[-suffix_len - 1]
    subs = labels[: -suffix_len - 1]
    return subs, name, suffix


def brand_mismatch(host: str, netloc: str) -> int:
    """1 if a well-known brand appears in the address but the real
    registered domain is not that brand (e.g. paypal.secure-login.xyz,
    paypa1.com, paypal.com@evil.xyz)."""
    _, name, _ = registered_domain_parts(host)
    for brand in BRANDS:
        if name == brand:
            continue
        for table in (_LEET_L, _LEET_I):
            text = netloc.translate(table)
            tokens = re.split(r"[.\-_@:]", text)
            if brand in tokens or (len(brand) >= 5 and brand in text):
                return 1
    return 0


def extract_features(url) -> dict:
    raw, parsed, host, netloc = parse_url(url)
    path, query = parsed.path or "", parsed.query or ""
    low = raw.lower()
    subs, name, suffix = registered_domain_parts(host)
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    n = max(len(raw), 1)
    digits = sum(c.isdigit() for c in raw)
    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 1

    return {
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "path_depth": path.count("/"),
        "num_dots": raw.count("."),
        "num_hyphens": raw.count("-"),
        "num_underscores": raw.count("_"),
        "num_digits": digits,
        "digit_ratio": round(digits / n, 4),
        "num_special_chars": sum(c in "?=&%#~!$*+,;:" for c in raw),
        "num_percent": raw.count("%"),
        "num_params": len([p for p in query.split("&") if p]),
        "has_at_symbol": int("@" in raw),
        "has_ip_host": int(bool(_IPV4.match(host))),
        "has_port": has_port,
        "has_scheme": int(bool(_SCHEME.match(raw))),
        "uses_https": int(low.startswith("https://")),
        "num_subdomains": len(subs) if not _IPV4.match(host) else 0,
        "hyphen_in_host": int("-" in host),
        "digits_in_host": sum(c.isdigit() for c in host) if not _IPV4.match(host) else 0,
        "suspicious_words_host": sum(w in host for w in SUSPICIOUS_WORDS),
        "suspicious_words_path": sum(w in (path + "?" + query).lower() for w in SUSPICIOUS_WORDS),
        "has_shortener": int(host in SHORTENERS),
        "has_punycode": int("xn--" in host),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "brand_mismatch": brand_mismatch(host, netloc),
        "embedded_url_in_path": int("http" in (path + "?" + query).lower() or "//" in path),
        "host_entropy": round(_entropy(host), 4),
    }


def features_dataframe(urls) -> pd.DataFrame:
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
