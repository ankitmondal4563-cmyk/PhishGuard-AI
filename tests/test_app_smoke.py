"""Runs app.py page by page against a fake `streamlit` module.

This catches import errors / crashes in the UI code without needing a browser.
It does NOT check how the page looks - run `streamlit run app.py` for that.
"""
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def real_errors(fake):
    """st.error is also used for the High/Critical risk banner - those are not failures."""
    return [m for m in fake.calls["error"] if "risk level" not in m]


class _State(dict):
    __getattr__ = dict.get


def make_fake_streamlit(page, url_value, text_value):
    st = types.ModuleType("streamlit")
    st.calls = {"error": [], "download": []}
    st.session_state = _State()
    st.cache_resource = lambda *a, **k: (lambda f: f)
    st.columns = lambda spec, *a, **k: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.button = lambda *a, **k: True
    st.checkbox = lambda *a, **k: False
    st.text_input = lambda *a, **k: url_value
    st.text_area = lambda *a, **k: text_value
    st.error = lambda msg, *a, **k: st.calls["error"].append(msg)
    st.download_button = lambda *a, **k: st.calls["download"].append(k.get("data"))
    sidebar = MagicMock()
    sidebar.radio.return_value = page
    sidebar.checkbox.return_value = True
    st.sidebar = sidebar
    for name in ("set_page_config", "markdown", "title", "subheader", "write", "metric", "info",
                 "success", "warning", "progress", "caption", "expander", "dataframe",
                 "bar_chart", "image"):
        setattr(st, name, MagicMock())
    return st


class AppSmokeTest(unittest.TestCase):
    def run_page(self, page, url="http://paypa1-secure.xyz/login", text="URGENT verify your account now http://sbi-kyc.top/x"):
        fake = make_fake_streamlit(page, url, text)
        tmp = Path(tempfile.mkdtemp()) / "test.db"
        previous = sys.modules.get("streamlit")
        sys.modules["streamlit"] = fake          # only this one key; patch.dict(sys.modules) would break numpy
        try:
            with patch("src.database.DB_PATH", tmp):
                runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
        finally:
            if previous is None:
                sys.modules.pop("streamlit", None)
            else:
                sys.modules["streamlit"] = previous
        return fake

    def test_every_page_renders(self):
        if not (ROOT / "models" / "url_model.joblib").exists():
            self.skipTest("models not trained")
        for page in ["Home", "URL Scanner", "Email Scanner", "SMS Scanner", "Dashboard", "About Project"]:
            with self.subTest(page=page):
                fake = self.run_page(page)
                self.assertEqual(real_errors(fake), [], f"{page} showed errors: {real_errors(fake)}")

    def test_scanners_produce_report(self):
        if not (ROOT / "models" / "url_model.joblib").exists():
            self.skipTest("models not trained")
        for page in ["URL Scanner", "Email Scanner", "SMS Scanner"]:
            fake = self.run_page(page)
            self.assertEqual(len(fake.calls["download"]), 1)
            self.assertIn("Risk Assessment Report", fake.calls["download"][0])

    def test_empty_input_shows_error(self):
        if not (ROOT / "models" / "url_model.joblib").exists():
            self.skipTest("models not trained")
        for page in ["URL Scanner", "Email Scanner", "SMS Scanner"]:
            fake = self.run_page(page, url="  ", text="   ")
            self.assertTrue(any("Please enter" in m for m in real_errors(fake)), page)
            self.assertEqual(fake.calls["download"], [])


if __name__ == "__main__":
    unittest.main()
