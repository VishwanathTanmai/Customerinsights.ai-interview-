import unittest
from app.reasons import verify, template_reason, build_facts


class TestReasonsVerification(unittest.TestCase):
    def setUp(self):
        self.facts = {
            "specialty": "Dermatology",
            "trx_recent_3m": 35.0,
            "trx_growth_ratio": 1.3,
            "samples_last_9m": 12,
        }
        self.stored = dict(self.facts)

    def test_valid_claim_passes(self):
        text = "Dermatologist with 35 transactions in the last quarter and a growth ratio of 1.3."
        claims = [
            {"field": "specialty", "value": "Dermatology"},
            {"field": "trx_recent_3m", "value": 35.0},
            {"field": "trx_growth_ratio", "value": 1.3},
        ]
        ok, why = verify(text, claims, self.facts, self.stored)
        self.assertTrue(ok, f"Expected pass, got: {why}")
        self.assertIsNone(why)

    def test_banned_word_fails(self):
        text = "This visit will guarantee higher uplift and clinical efficacy for the doctor."
        claims = [{"field": "specialty", "value": "Dermatology"}]
        ok, why = verify(text, claims, self.facts, self.stored)
        self.assertFalse(ok)
        self.assertIn("banned term", why)

    def test_unsupported_number_fails(self):
        text = "Dermatologist wrote 99 prescriptions last month."
        claims = [{"field": "specialty", "value": "Dermatology"}]
        ok, why = verify(text, claims, self.facts, self.stored)
        self.assertFalse(ok)
        self.assertIn("unsupported number", why)

    def test_number_tolerance_passes(self):
        # 35.1 is within max(0.5, 35 * 0.02 = 0.7)
        text = "Prescriber has 35.1 TRx."
        claims = [
            {"field": "specialty", "value": "Dermatology"},
            {"field": "trx_recent_3m", "value": 35.1},
        ]
        ok, why = verify(text, claims, self.facts, self.stored)
        self.assertTrue(ok)

    def test_unknown_field_fails(self):
        text = "Doctor has high score."
        claims = [{"field": "non_existent_field", "value": 100}]
        ok, why = verify(text, claims, self.facts, self.stored)
        self.assertFalse(ok)
        self.assertIn("unknown field", why)

    def test_too_long_fails(self):
        long_text = "word " * 45
        claims = [{"field": "specialty", "value": "Dermatology"}]
        ok, why = verify(long_text, claims, self.facts, self.stored)
        self.assertFalse(ok)
        self.assertIn("too long", why)

    def test_template_reason_fallback(self):
        text, claims = template_reason(self.facts)
        self.assertTrue(text.startswith("Dermatology prescriber"))
        self.assertGreaterEqual(len(claims), 2)
        ok, why = verify(text, claims, self.facts, self.stored)
        self.assertTrue(ok, f"Template fallback should always verify, got error: {why}")

    def test_build_facts(self):
        features = {
            "trx_recent_3m": 25,
            "is_derm": 1,
            "samples_last_9m": 10,
            "patient_panel_size": 300,
        }
        drivers = [
            {"feature": "trx_recent_3m"},
            {"feature": "is_derm"},
            {"feature": "samples_last_9m"},
        ]
        facts = build_facts("Primary Care", features, drivers)
        self.assertEqual(facts["specialty"], "Primary Care")
        self.assertIn("trx_recent_3m", facts)
        self.assertIn("samples_last_9m", facts)
        self.assertNotIn("is_derm", facts, "Internal dummy variable is_derm should be filtered out")


if __name__ == "__main__":
    unittest.main()

