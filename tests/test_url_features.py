import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.url_features import FEATURE_NAMES, extract_features, features_dataframe


class UrlFeatureTests(unittest.TestCase):
    def test_feature_vector_is_complete_and_ordered(self):
        f = extract_features("https://www.example.com/a?b=1")
        self.assertEqual(list(f.keys()), FEATURE_NAMES)
        self.assertEqual(list(features_dataframe(["a.com", "b.com"]).columns), FEATURE_NAMES)

    def test_ip_host(self):
        self.assertEqual(extract_features("http://192.168.10.5/login.php")["has_ip_host"], 1)
        self.assertEqual(extract_features("http://example.com")["has_ip_host"], 0)

    def test_at_symbol(self):
        self.assertEqual(extract_features("http://paypal.com@evil.xyz/x")["has_at_symbol"], 1)

    def test_brand_mismatch(self):
        self.assertEqual(extract_features("https://www.paypal.com/signin")["brand_mismatch"], 0)
        self.assertEqual(extract_features("https://sbi.co.in/web")["brand_mismatch"], 0)
        self.assertEqual(extract_features("http://paypa1-secure.xyz/login")["brand_mismatch"], 1)
        self.assertEqual(extract_features("https://login.paypal.com.evil.top/a")["brand_mismatch"], 1)
        self.assertEqual(extract_features("http://paypal.com@evil.xyz/x")["brand_mismatch"], 1)

    def test_https_and_scheme(self):
        self.assertEqual(extract_features("https://a.com")["uses_https"], 1)
        f = extract_features("a.com")
        self.assertEqual((f["has_scheme"], f["uses_https"]), (0, 0))

    def test_subdomain_count_handles_two_label_suffix(self):
        self.assertEqual(extract_features("https://www.bbc.co.uk")["num_subdomains"], 1)
        self.assertEqual(extract_features("https://a.b.c.example.com")["num_subdomains"], 3)

    def test_shortener_and_tld(self):
        self.assertEqual(extract_features("https://bit.ly/abc")["has_shortener"], 1)
        self.assertEqual(extract_features("http://x.top")["suspicious_tld"], 1)

    def test_garbage_input_never_crashes(self):
        for bad in ["", "   ", None, float("nan"), "http://[bad", "://", "a" * 5000, "http://", "ht!tp://x y"]:
            f = extract_features(bad)
            self.assertEqual(list(f.keys()), FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()
