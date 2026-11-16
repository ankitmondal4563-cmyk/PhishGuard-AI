"""Synthetic phishing / scam data generator.

IMPORTANT: everything produced here is ARTIFICIAL (template based).  It exists so
the full pipeline (build -> train -> evaluate -> app) can be run and tested
without downloading anything.  Metrics measured on this data say how well the
code works, NOT how well the system detects real-world phishing.  For real
results, train on real datasets (see README -> "Using real datasets").

Both classes contain "hard negatives" so the problem is not trivially separable:
legitimate sites with /login paths, IT notices that say "urgent", real OTP /
bank alerts, benign shortened links, and scams that avoid urgency words.
"""
import random

import pandas as pd

SOURCE_NAME = "SYNTHETIC"

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
_CONS, _VOW = "bcdfghjklmnprstvz", "aeiou"


def _word(rng, lo=2, hi=4):
    return "".join(rng.choice(_CONS) + rng.choice(_VOW) for _ in range(rng.randint(lo, hi)))


def _alnum(rng, n):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(n))


def _hex(rng, n):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _ip(rng):
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def _maybe_drop_scheme(rng, url, p=0.2):
    """Many real datasets store URLs without http(s)://, so mix both styles."""
    if rng.random() < p:
        for s in ("https://", "http://"):
            if url.startswith(s):
                return url[len(s):]
    return url


# --------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------
LEGIT_DOMAINS = [
    "wikipedia.org", "github.com", "bbc.com", "nytimes.com", "stackoverflow.com",
    "amazon.in", "amazon.com", "flipkart.com", "irctc.co.in", "sbi.co.in",
    "hdfcbank.com", "icicibank.com", "paypal.com", "google.com", "microsoft.com",
    "apple.com", "linkedin.com", "netflix.com", "python.org", "mozilla.org",
    "reddit.com", "medium.com", "thehindu.com", "ndtv.com", "coursera.org",
    "khanacademy.org", "iitkgp.ac.in", "wb.gov.in", "india.gov.in", "paytm.com",
    "swiggy.com", "zomato.com", "dropbox.com", "slack.com", "zoom.us",
]
LEGIT_SUB = ["www", "www", "www", "mail", "accounts", "support", "docs", "blog", "help", "m", "en"]
LEGIT_PATHS = [
    "", "/", "/about", "/wiki/Machine_learning", "/news/world-123456",
    "/products/item/48213", "/search?q=python+tutorial", "/login", "/signin",
    "/account/settings", "/help/contact-us", "/orders/track?id=402-3341",
    "/docs/getting-started", "/blog/2026/11/new-release", "/user/profile/edit",
    "/secure/checkout", "/verify-email?token=ab12cd", "/pricing", "/careers/jobs/18273",
]
BRANDS = ["paypal", "amazon", "google", "microsoft", "apple", "netflix", "hdfcbank",
          "sbi", "icici", "paytm", "flipkart", "facebook", "instagram", "whatsapp",
          "irctc", "linkedin"]
WORDS = ["secure", "login", "verify", "account", "update", "confirm", "support",
         "billing", "wallet", "service", "alert", "recovery", "signin", "kyc",
         "refund", "reward", "bonus", "free"]
TLDS_BAD = ["xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work",
            "icu", "buzz", "online", "site", "cc", "ru"]
TLDS_COMMON = ["com", "net", "in", "org", "co", "info"]
LEET = {"l": "1", "o": "0", "e": "3", "a": "4", "s": "5"}


def _tld(rng):
    return rng.choice(TLDS_BAD) if rng.random() < 0.6 else rng.choice(TLDS_COMMON)


def _leet(rng, brand):
    spots = [i for i, c in enumerate(brand) if c in LEET]
    if not spots:
        return brand + "1"
    i = rng.choice(spots)
    return brand[:i] + LEET[brand[i]] + brand[i + 1:]


def legit_url(rng) -> str:
    scheme = "https://" if rng.random() < 0.92 else "http://"
    r = rng.random()
    if r < 0.05:  # benign shortened link
        host, path = rng.choice(["bit.ly", "tinyurl.com"]), "/" + _alnum(rng, 7)
    elif r < 0.30:  # small generic business / personal site
        name = _word(rng, 2, 3) + ("-" + _word(rng, 1, 2) if rng.random() < 0.3 else "")
        host = f"{name}.{rng.choice(['com', 'in', 'org', 'net', 'co.in'])}"
        if rng.random() < 0.4:
            host = "www." + host
        path = rng.choice(["", "/", "/about", "/contact", "/menu", "/blog/" + _word(rng),
                           "/products/" + _word(rng), "/login", "/account"])
    else:  # well-known site
        dom = rng.choice(LEGIT_DOMAINS)
        sub = rng.choice(LEGIT_SUB) if rng.random() < 0.7 else ""
        host = f"{sub}.{dom}" if sub else dom
        path = rng.choice(LEGIT_PATHS)
    if rng.random() < 0.2:
        extra = rng.choice(["utm_source=newsletter", "ref=home", f"id={rng.randint(1, 99999)}", "lang=en"])
        path += ("&" if "?" in path else "?") + extra
    return _maybe_drop_scheme(rng, scheme + host + path)


def phishing_url(rng) -> str:
    brand = rng.choice(BRANDS)
    w1, w2, w3 = rng.sample(WORDS, 3)
    tld = _tld(rng)
    scheme = "https://" if rng.random() < 0.55 else "http://"
    kind = rng.choices(
        ["hyphen", "subdomain", "ip", "leet", "at", "short", "generic", "long", "pathbrand"],
        weights=[22, 14, 8, 12, 5, 6, 12, 9, 12])[0]
    if kind == "hyphen":
        host = f"{brand}-{w1}" + (f"-{w2}" if rng.random() < 0.5 else "") + f".{tld}"
        path = rng.choice([f"/{w2}/login.php", f"/{w3}", "/", f"/{w1}.html",
                           f"/signin?id={rng.randint(1000, 99999)}"])
    elif kind == "subdomain":
        host = f"{w1}.{w2}.{brand}.{_word(rng)}.{tld}"
        path = rng.choice(["/", f"/{w3}/", f"/{w1}.php"])
    elif kind == "ip":
        scheme, host = "http://", _ip(rng)
        path = f"/{brand}/{w1}.php?id={rng.randint(1, 99999)}"
    elif kind == "leet":
        host = _leet(rng, brand) + (f"-{w1}" if rng.random() < 0.4 else "") + f".{tld}"
        path = rng.choice(["/signin", "/login", "/", f"/account/{w2}"])
    elif kind == "at":
        scheme, host, path = "http://", f"{brand}.com@{_word(rng, 3, 4)}.{tld}", f"/{w1}"
    elif kind == "short":
        host, path = rng.choice(["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy"]), "/" + _alnum(rng, 7)
    elif kind == "generic":
        host = (f"{w1}-{w2}-{w3}.{tld}" if rng.random() < 0.6 else f"{w1}{w2}.{tld}")
        path = rng.choice(["/", f"/{_word(rng)}/index.php", "/verify", "/claim?ref=" + _hex(rng, 6)])
    elif kind == "long":
        host = f"{brand}.{w1}-{w2}.{_word(rng, 3, 4)}.{tld}"
        path = f"/{w3}/{_word(rng)}?session={_hex(rng, 16)}&redirect=http://{_word(rng)}.{tld}"
    else:  # brand hidden in the path of an unrelated (compromised) site
        host = f"{_word(rng, 3, 4)}{_word(rng, 1, 2)}.{rng.choice(['com', 'net', 'org', 'in'])}"
        path = f"/wp-includes/{brand}/{w1}.php"
    return _maybe_drop_scheme(rng, scheme + host + path)


# --------------------------------------------------------------------------
# Text messages (email / sms / generic scam message)
# --------------------------------------------------------------------------
BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Paytm", "PhonePe", "Punjab National Bank"]
NAMES = ["Rahul", "Priya", "Amit", "Sneha", "Arjun", "Meera", "Subrata", "Ritu"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
MONTHS = ["September", "October", "November", "December"]
AMOUNTS = ["500", "1,500", "2,000", "5,000", "10,000", "25,000", "50,000", "1,00,000"]

SCAM_TEMPLATES = [
    "Your {bank} KYC is expiring today. Update now at {link} to avoid account block.",
    "URGENT: Your {bank} account has been suspended due to suspicious activity. Verify your identity immediately: {link}",
    "Congratulations! You have won \u20b9{amt} in the lucky draw. Claim your prize now: {link}",
    "You have a pending refund of \u20b9{amt} from the Income Tax Department. Submit your bank details at {link} to receive it.",
    "Your electricity will be disconnected tonight at 9:30 PM due to unpaid bill. Call {phone} immediately.",
    "Your parcel is held at customs. Pay a small fee of \u20b9{amt} here {link} to release the delivery.",
    "Work from home and earn \u20b9{amt} daily. No experience needed. Message us on WhatsApp {phone} to join today.",
    "Invest \u20b9{amt} in our crypto scheme and get double returns in 7 days. Guaranteed profit. Join: {link}",
    "This is {name} from {bank} support. We detected a login from a new device. Share the OTP sent to your phone to secure your account.",
    "Your {bank} credit card reward points worth \u20b9{amt} will expire today. Redeem now {link}",
    "Final notice: your account will be closed within 24 hours. Confirm your password at {link}",
    "I am a barrister handling an unclaimed inheritance of $4.5 million in your family name. Send your bank details and passport copy to proceed.",
    "Hi, I found your CV online. We are hiring part-time staff. Earn \u20b9{amt} per day from home. Reply YES to know more.",
    "Dear customer, a payment of \u20b9{amt} is pending on your {bank} card. Update your card details here: {link}",
    "Your SIM will be blocked in 2 hours. Complete re-verification now at {link} or call {phone}.",
    "Hello {name}, you were selected for a free gift voucher worth \u20b9{amt}. Just pay shipping at {link}.",
    "We noticed unusual sign-in activity. Reset your password using this secure link: {link}",
    "Limited offer! Buy followers and earn cashback of \u20b9{amt} instantly. Pay via {link}",
]
LEGIT_TEMPLATES = [
    "Your OTP for {bank} login is {otp}. Do not share it with anyone. Valid for 10 minutes.",
    "Rs {amt} debited from A/c XX{n4} on {day}. Avl bal Rs {amt2}. If this was not you, call the number on your card.",
    "Your order #{n} has been shipped and will arrive by {day}. Track it here: {link}",
    "Hey, are we still meeting for lunch at {time}?",
    "Hi team, attaching the meeting notes from {day}. Let me know if I missed anything.",
    "Reminder: your appointment with Dr. {name} is on {day} at {time}.",
    "Your {bank} statement for {month} is ready. Log in to the official app to view it.",
    "Urgent: server maintenance tonight at 11 PM. Please save your work and log off before then.",
    "Congratulations on your promotion, {name}! Drinks on Friday.",
    "Your package was delivered to the front desk. Thanks for shopping with us.",
    "Verify your email address to finish setting up your account: {link}",
    "Lunch menu for {day}: biriyani, salad and dessert.",
    "Invoice #{n} for \u20b9{amt} is attached. Payment is due by {day}. Thanks!",
    "Your electricity bill of \u20b9{amt} for {month} has been generated. Pay via the official app before the due date.",
    "Your refund of \u20b9{amt} has been processed and will reflect in 5-7 working days.",
    "Reminder: project review on {day} at {time}. Agenda is in the shared folder.",
    "Happy birthday {name}! Hope you have a wonderful day.",
    "Your cab is arriving in 3 minutes. Driver {name} is on the way.",
    "Class test on {day}. Please revise chapters 4 to 6 and bring your ID card.",
    "Thank you for your payment of \u20b9{amt}. Your receipt is available at {link}",
]
PREFIX = ["", "", "", "Dear customer, ", "Hello, ", "Hi {name}, ", "Notice: "]
SUFFIX = ["", "", "", " Thank you.", " Regards, {sender}", " Please ignore if already done."]
SUBJ_SCAM = ["Action required: verify your account", "Your account has been suspended",
             "You have won a prize", "Final notice", "Update your KYC", "Refund pending",
             "Security alert"]
SUBJ_LEGIT = ["Meeting notes", "Your order has shipped", "Your statement is ready",
              "Lunch on Friday", "Invoice attached", "Appointment reminder",
              "Verify your email", "Scheduled maintenance"]
SIGNOFF = ["Regards,\nCustomer Care", "Thanks,\nSupport Team", "Best,\nTeam",
           "Regards,\nAccounts Department"]


def _slots(rng, link):
    return dict(
        bank=rng.choice(BANKS), name=rng.choice(NAMES), amt=rng.choice(AMOUNTS),
        amt2=rng.choice(AMOUNTS), phone=rng.choice("6789") + "".join(rng.choice("0123456789") for _ in range(9)),
        otp="".join(rng.choice("0123456789") for _ in range(6)), n=rng.randint(10000, 99999),
        n4="".join(rng.choice("0123456789") for _ in range(4)), day=rng.choice(DAYS),
        time=rng.choice(["1 PM", "2:30 PM", "5 PM", "10 AM"]), month=rng.choice(MONTHS),
        link=link, sender=rng.choice(BANKS))


def _perturb(rng, text):
    r = rng.random()
    if r < 0.12:
        return text.lower()
    if r < 0.20:
        return text.replace(".", "").replace(",", "")
    return text


def make_text(rng, malicious: bool):
    """Return (text, channel)."""
    link = phishing_url(rng) if malicious else legit_url(rng)
    template = rng.choice(SCAM_TEMPLATES if malicious else LEGIT_TEMPLATES)
    slots = _slots(rng, link)
    body = (rng.choice(PREFIX) + template + rng.choice(SUFFIX)).format(**slots)
    channel = rng.choices(["email", "sms", "message"], weights=[35, 35, 30])[0]
    if channel == "email":
        subject = rng.choice(SUBJ_SCAM if malicious else SUBJ_LEGIT)
        body = f"Subject: {subject}\n\n{body}\n\n{rng.choice(SIGNOFF)}"
    return _perturb(rng, body), channel


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------
def make_synthetic_frame(n_urls: int = 1600, n_texts: int = 1800, seed: int = 42) -> pd.DataFrame:
    """Balanced synthetic frame in the unified-dataset schema."""
    rng = random.Random(seed)
    rows = []
    for i in range(n_urls):
        bad = i % 2 == 0
        rows.append(("", phishing_url(rng) if bad else legit_url(rng),
                     rng.choices(["url", "website"], weights=[70, 30])[0], int(bad)))
    for i in range(n_texts):
        bad = i % 2 == 0
        text, channel = make_text(rng, bad)
        rows.append((text, "", channel, int(bad)))
    df = pd.DataFrame(rows, columns=["text", "url", "source_type", "label"])
    df["original_label"] = df["label"].map({1: "malicious", 0: "benign"})
    df["source_dataset"] = SOURCE_NAME
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
