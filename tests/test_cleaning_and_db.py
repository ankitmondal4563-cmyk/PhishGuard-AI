import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from src import database as db
from src.data_cleaning import clean_text, ensure_split, load_dataset


class CleaningTests(unittest.TestCase):
    def test_placeholders(self):
        t = clean_text("Win \u20b95,000 now! Visit http://bad.xyz/x or call 9876543210. Code 4821")
        self.assertIn("moneytoken", t)
        self.assertIn("urltoken", t)
        self.assertIn("phonetoken", t)
        self.assertIn("numtoken", t)
        self.assertNotIn("http", t)

    def test_none_and_nan(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(float("nan")), "")

    def test_html_removed(self):
        self.assertEqual(clean_text("<b>Hello</b>&nbsp;World"), "hello world")

    def test_load_dataset_validation(self):
        tmp = Path(tempfile.mkdtemp())
        with self.assertRaises(FileNotFoundError):
            load_dataset(tmp / "missing.csv")
        pd.DataFrame({"text": ["a"], "label": [1]}).to_csv(tmp / "bad.csv", index=False)
        with self.assertRaises(ValueError):
            load_dataset(tmp / "bad.csv")
        pd.DataFrame({"text": ["a"], "url": [""], "label": [2], "source_type": ["sms"]}).to_csv(tmp / "lab.csv", index=False)
        with self.assertRaises(ValueError):
            load_dataset(tmp / "lab.csv")

    def test_ensure_split_is_stratified_and_deterministic(self):
        df = pd.DataFrame({"source_type": ["sms"] * 100, "label": [0, 1] * 50})
        a, b = ensure_split(df), ensure_split(df)
        self.assertTrue((a["split"] == b["split"]).all())
        self.assertEqual(set(a["split"]), {"train", "val", "test"})
        for s in ("train", "val", "test"):
            self.assertEqual(a[a.split == s]["label"].nunique(), 2)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "t.db"

    def result(self, p=0.9):
        return {"input": "x" * 800, "label": "Phishing", "probability": p,
                "risk_level": "Critical", "indicators": ["a", "b"]}

    def test_log_and_read(self):
        db.log_scan("url", self.result(), self.path)
        db.log_scan("sms", self.result(0.1), self.path)
        hist = db.get_history(10, self.path)
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["scan_type"], "sms")          # newest first
        self.assertEqual(hist[1]["indicators"], ["a", "b"])
        self.assertEqual(len(hist[0]["input_text"]), db.MAX_STORED_CHARS)   # truncated

    def test_stats_and_clear(self):
        db.log_scan("url", self.result(0.9), self.path)
        db.log_scan("url", self.result(0.1), self.path)
        s = db.get_stats(self.path)
        self.assertEqual((s["total"], s["flagged"], s["by_type"]["url"]), (2, 1, 2))
        db.clear_history(self.path)
        self.assertEqual(db.get_stats(self.path)["total"], 0)

    def test_sql_injection_text_is_stored_as_data(self):
        r = self.result()
        r["input"] = "'); DROP TABLE scans;--"
        db.log_scan("sms", r, self.path)
        self.assertEqual(db.get_stats(self.path)["total"], 1)


if __name__ == "__main__":
    unittest.main()
