from datetime import datetime

import pandas as pd
import streamlit as st

from src import database as db
from src.config import MODELS_DIR
from src.report import build_report
from src.text_detector import TextDetector
from src.url_detector import URLDetector

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# CUSTOM STYLE
# =====================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

h1 {
    color: #17365D;
}

h2, h3 {
    color: #23466b;
}

.stButton button {
    background-color: #1769aa;
    color: white;
    border-radius: 8px;
    border: none;
}

.stButton button:hover {
    background-color: #0e4d80;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# MODEL LOADING (cached)
# =====================================


@st.cache_resource(show_spinner=False)
def load_detectors():
    """Load both models once. Returns (url_detector, text_detector, errors)."""
    errors, url_det, text_det = {}, None, None
    try:
        url_det = URLDetector()
    except Exception as exc:  # missing file OR incompatible pickle
        errors["URL model"] = str(exc)
    try:
        text_det = TextDetector(url_detector=url_det)
    except Exception as exc:
        errors["Text model"] = str(exc)
    return url_det, text_det, errors


url_detector, text_detector, load_errors = load_detectors()
db.init_db()


def synthetic_warning(detector):
    if detector is not None and detector.info["is_synthetic"]:
        st.warning(
            "⚠️ This model was trained on **synthetic demo data**. Its results "
            "show that the software works, not that it can spot real-world "
            "phishing. Train on real datasets before relying on it "
            "(see README → *Using real datasets*)."
        )


def model_missing_error():
    for name, msg in load_errors.items():
        st.error(f"{name} could not be loaded: {msg}")
    st.info("Train the models with:  `python build_unified_dataset.py --synthetic` "
            "then `python train_all.py`")


def render_result(kind, result, detector):
    """Show one scan result (read from session_state so it survives reruns)."""
    level = result["risk_level"]
    banner = {"Critical": st.error, "High": st.error,
              "Medium": st.warning, "Low": st.success}[level]
    banner(f"**{result['label']}**  —  risk level: **{level}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score", f"{result['risk_score']} / 100")
    c2.metric("Risk level", level)
    if result.get("text_probability") is not None:
        link_p = result["link_probability"]
        c3.metric("Text / link score",
                  f"{result['text_probability']:.0%} / "
                  f"{'—' if link_p is None else format(link_p, '.0%')}")
    else:
        c3.metric("Model confidence (phishing)", f"{result['probability']:.0%}")
    st.progress(min(max(result["probability"], 0.0), 1.0))

    st.markdown("#### 🚩 Red flags")
    if result["indicators"]:
        st.markdown("\n".join(f"- {i}" for i in result["indicators"]))
    else:
        st.info("None of the rule-based red flags were triggered.")

    drivers = result.get("model_drivers")
    with st.expander("🔍 Why did the model decide this?"):
        if isinstance(drivers, list):
            if drivers:
                df = pd.DataFrame(drivers)
                st.caption("Positive = pushes towards *phishing*, negative = towards *legitimate*.")
                st.bar_chart(df.set_index("feature")["contribution"])
                st.dataframe(df)
            else:
                st.write("No single feature had a noticeable effect.")
        elif isinstance(drivers, dict):
            left, right = st.columns(2)
            with left:
                st.markdown("**Words pushing towards scam**")
                st.dataframe(pd.DataFrame(drivers["towards_phishing"]))
            with right:
                st.markdown("**Words pushing towards legitimate**")
                st.dataframe(pd.DataFrame(drivers["towards_legitimate"]))

    if result.get("links"):
        st.markdown("#### 🔗 Links found in the message")
        st.dataframe(pd.DataFrame(result["links"])[["url", "risk_level", "probability"]])

    st.download_button(
        "⬇️ Download risk report (.md)",
        data=build_report(kind, result, detector.info),
        file_name=f"phishguard_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
        key=f"download_{kind}",
    )
    st.caption("Automated estimate — not a guarantee. When unsure, don't click or "
               "share anything; check via the organisation's official app or website.")


# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("🛡️ PhishGuard AI")

st.sidebar.caption(
    "Intelligent Cyber Scam Detection"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "URL Scanner",
        "Email Scanner",
        "SMS Scanner",
        "Dashboard",
        "About Project"
    ]
)

st.sidebar.markdown("---")

save_history = st.sidebar.checkbox(
    "Save scans to local history",
    value=True,
    help="Stored only on this computer (first 500 characters of each scan)."
)

if load_errors:
    st.sidebar.error("Models not loaded — see Home page.")
else:
    st.sidebar.success("URL + Text models loaded")

# =====================================
# HOME PAGE
# =====================================

if page == "Home":

    st.title("🛡️ PhishGuard AI")

    st.subheader(
        "Intelligent Multi-Layer Phishing "
        "and Cyber Scam Detection System"
    )

    st.markdown("---")

    st.write(
        "Welcome to PhishGuard AI. "
        "This system identifies "
        "potential phishing URLs, suspicious "
        "emails and scam messages using "
        "Machine Learning and NLP."
    )

    stats = db.get_stats()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Detection Modules", "3")

    with col2:
        st.metric("AI Models Loaded", f"{2 - len(load_errors)} / 2")

    with col3:
        st.metric("Scans Saved", stats["total"])

    with col4:
        st.metric("Flagged as Phishing", stats["flagged"])

    if load_errors:
        model_missing_error()
    else:
        synthetic_warning(url_detector)

    st.markdown("---")

    st.subheader("Available Detection Modules")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("🌐 URL Phishing Detection")

    with col2:
        st.info("📧 Email Phishing Detection")

    with col3:
        st.info("📱 SMS Scam Detection")

    st.success(
        "Select a detection module from the sidebar."
    )

# =====================================
# URL SCANNER
# =====================================

elif page == "URL Scanner":

    st.title("🌐 URL Phishing Scanner")

    st.write(
        "Enter a website URL for analysis. "
        "The URL is analysed from its text only — the site is never opened."
    )

    if url_detector is None:
        model_missing_error()
    else:
        synthetic_warning(url_detector)

        url = st.text_input(
            "Website URL",
            placeholder="https://example.com"
        )

        if st.button("Analyze URL"):

            if not url.strip():
                st.error("Please enter a URL.")

            elif " " in url.strip():
                st.error("URL should not contain spaces.")

            else:
                try:
                    result = url_detector.predict(url)
                    st.session_state["last_url"] = result
                    if save_history:
                        db.log_scan("url", result)
                except ValueError as exc:
                    st.error(str(exc))

        if "last_url" in st.session_state:
            render_result("url", st.session_state["last_url"], url_detector)

# =====================================
# EMAIL SCANNER
# =====================================

elif page == "Email Scanner":

    st.title("📧 Email Phishing Scanner")

    st.write(
        "Paste an email message to analyze. "
        "Links inside the email are checked by the URL model as well."
    )

    if text_detector is None:
        model_missing_error()
    else:
        synthetic_warning(text_detector)

        email_text = st.text_area(
            "Email Content",
            placeholder="Paste email content here...",
            height=200
        )

        if st.button("Analyze Email"):

            if not email_text.strip():
                st.error("Please enter email content.")

            else:
                result = text_detector.predict(email_text, channel="email")
                st.session_state["last_email"] = result
                if save_history:
                    db.log_scan("email", result)

        if "last_email" in st.session_state:
            render_result("email", st.session_state["last_email"], text_detector)

# =====================================
# SMS SCANNER
# =====================================

elif page == "SMS Scanner":

    st.title("📱 SMS Scam Scanner")

    st.write(
        "Enter a suspicious SMS message."
    )

    if text_detector is None:
        model_missing_error()
    else:
        synthetic_warning(text_detector)

        sms_text = st.text_area(
            "SMS Message",
            placeholder="Enter SMS message here...",
            height=150
        )

        if st.button("Analyze SMS"):

            if not sms_text.strip():
                st.error("Please enter an SMS message.")

            else:
                result = text_detector.predict(sms_text, channel="sms")
                st.session_state["last_sms"] = result
                if save_history:
                    db.log_scan("sms", result)

        if "last_sms" in st.session_state:
            render_result("sms", st.session_state["last_sms"], text_detector)

# =====================================
# DASHBOARD
# =====================================

elif page == "Dashboard":

    st.title("📊 Dashboard")

    stats = db.get_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total scans", stats["total"])
    col2.metric("Flagged as phishing", stats["flagged"])
    col3.metric(
        "Flag rate",
        f"{stats['flagged'] / stats['total']:.0%}" if stats["total"] else "—"
    )

    if stats["total"]:
        left, right = st.columns(2)
        with left:
            st.markdown("**Scans by type**")
            st.bar_chart(pd.Series(stats["by_type"], name="scans"))
        with right:
            st.markdown("**Scans by risk level**")
            order = ["Low", "Medium", "High", "Critical"]
            st.bar_chart(pd.Series({k: stats["by_risk"].get(k, 0) for k in order}, name="scans"))

        st.markdown("#### Recent scans")
        history = pd.DataFrame(db.get_history(100))
        history["indicators"] = history["indicators"].map(lambda xs: "; ".join(xs))
        st.dataframe(history[["scanned_at", "scan_type", "label", "probability",
                              "risk_level", "input_text", "indicators"]])

        with st.expander("Delete history"):
            confirm = st.checkbox("Yes, delete all saved scans")
            if st.button("Delete history") and confirm:
                db.clear_history()
                st.success("History deleted.")
    else:
        st.info("No scans saved yet. Run a scan from the sidebar.")

    st.markdown("---")
    st.markdown("### Model performance (held-out test split)")

    for title, det, img in (("URL model", url_detector, "url_confusion_matrix.png"),
                            ("Text model", text_detector, "text_confusion_matrix.png")):
        st.markdown(f"**{title}**")
        if det is None:
            st.warning("Not loaded.")
            continue
        info = det.info
        m = info["metrics"]
        st.caption(f"{info['name']} · trained {info['trained_at']}")
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{m['accuracy']:.1%}")
        b.metric("Precision", f"{m['precision']:.1%}")
        c.metric("Recall", f"{m['recall']:.1%}")
        d.metric("F1", f"{m['f1']:.1%}")
        if (MODELS_DIR / img).exists():
            st.image(str(MODELS_DIR / img), width=320)
        synthetic_warning(det)

# =====================================
# ABOUT PROJECT
# =====================================

elif page == "About Project":

    st.title("About PhishGuard AI")

    st.write("""
    PhishGuard AI is an academic AI and
    Cyber Security project.

    Main Objectives:

    1. Detect phishing URLs.
    2. Identify phishing emails.
    3. Detect suspicious SMS messages.
    4. Analyse suspicious language patterns.
    5. Generate risk assessment reports.

    How it works:

    - **URL layer** — 29 lexical/host features (length, IP host, brand
      impersonation, risky TLD, ...) fed to a tree-ensemble classifier.
      The page itself is never visited.
    - **Text layer** — cleaned text → TF-IDF word/bigram features →
      Logistic Regression. Links found in a message are also scored by the
      URL layer and can raise the overall risk.
    - **Explainability** — rule-based red flags plus the model's own
      per-feature / per-word contributions.
    - **History** — scans are saved to a local SQLite database.

    Technologies:

    - Python
    - Machine Learning
    - Natural Language Processing
    - Scikit-learn
    - Streamlit
    - SQLite
    - Explainable AI
    """)

    st.info(
        "Developed as a Final Year Project."
    )

# =====================================
# FOOTER
# =====================================

st.markdown("---")

st.caption(
    "PhishGuard AI | Academic Project | "
    + datetime.now().strftime("%d-%m-%Y")
)

st.caption(
    "Automated estimates only — never a substitute for your own judgement "
    "or your bank's official channels."
)
