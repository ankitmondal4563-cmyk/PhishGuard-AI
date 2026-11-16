"""Trains tiny models on freshly generated synthetic data (independent of models/)
and checks the detector API contract + behaviour on obvious cases."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import train_text, train_url
from src.config import risk_level
from src.synthetic_data import make_synthetic_frame
from src.text_detector import TextDetector
from src.url_detector import ModelNotFoundError, URLDetector


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = Path(tempfile.mkdtemp())
        data = tmp / "d.csv"
        make_synthetic_frame(n_urls=500, n_texts=500, seed=7).to_csv(data, index=False)
        train_url.train(data, tmp / "u.joblib", verbose=False)
        train_text.train(data, tmp / "t.joblib", verbose=False)
        cls.url = URLDetector(tmp / "u.joblib")
        cls.text = TextDetector(tmp / "t.joblib", url_detector=cls.url)

    def test_url_contract(self):
        r = self.url.predict("http://paypa1-secure.xyz/login.php")
        for key in ("label", "probability", "risk_score", "risk_level", "indicators", "model_drivers", "features"):
            self.assertIn(key, r)
        self.assertTrue(0 <= r["probability"] <= 1)
        self.assertEqual(r["risk_level"], risk_level(r["probability"]))

    def test_url_obvious_cases(self):
        self.assertTrue(self.url.predict("http://paypa1-secure.xyz/login.php")["is_phishing"])
        self.assertTrue(self.url.predict("http://192.168.10.5/paypal/login.php?id=5")["is_phishing"])
        self.assertFalse(self.url.predict("https://www.wikipedia.org/wiki/Machine_learning")["is_phishing"])
        self.assertFalse(self.url.predict("https://github.com/pallets/flask")["is_phishing"])

    def test_url_validation(self):
        for bad in ["", "   ", "has a space.com"]:
            with self.assertRaises(ValueError):
                self.url.predict(bad)

    def test_text_obvious_cases(self):
        scam = self.text.predict("URGENT: Your SBI account has been suspended. Verify immediately: http://sbi-kyc.top/login", "sms")
        self.assertTrue(scam["is_phishing"])
        self.assertGreaterEqual(len(scam["links"]), 1)
        ok = self.text.predict("Hey, are we still meeting for lunch at 1pm?", "sms")
        self.assertFalse(ok["is_phishing"])

    def test_malicious_link_escalates_overall_risk(self):
        r = self.text.predict("Hello, please see this http://paypa1-secure.xyz/login.php", "email")
        self.assertGreaterEqual(r["probability"], r["text_probability"])
        self.assertIsNotNone(r["link_probability"])

    def test_text_validation_and_drivers(self):
        with self.assertRaises(ValueError):
            self.text.predict("   ")
        r = self.text.predict("Congratulations! You won a prize, claim now", "message")
        self.assertIn("towards_phishing", r["model_drivers"])

    def test_otp_warning_not_flagged_as_secret_request(self):
        r = self.text.predict("Your OTP is 482910. Do not share your OTP with anyone.", "sms")
        self.assertFalse(any("sensitive information" in i for i in r["indicators"]))
        r2 = self.text.predict("Please share your OTP with our agent to continue.", "sms")
        self.assertTrue(any("sensitive information" in i for i in r2["indicators"]))

    def test_missing_model_gives_helpful_error(self):
        with self.assertRaises(ModelNotFoundError) as cm:
            URLDetector(Path(tempfile.mkdtemp()) / "nope.joblib")
        self.assertIn("train_url", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
