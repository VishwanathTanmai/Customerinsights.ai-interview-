import unittest
import numpy as np
from app.db import connect
from app.models import qini_curve, top_decile_precision, top_decile_incremental, TLearner


class TestModelsAndUplift(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = connect()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_qini_curve_calculation(self):
        # Perfect ranking scenario: all treated who responded ranked first
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        t = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        res = qini_curve(scores, y, t, points=10)
        self.assertIn("coefficient", res)
        self.assertGreater(res["coefficient"], 0, "Perfect uplift should have positive Qini coefficient")
        self.assertEqual(len(res["x"]), len(res["y"]))

    def test_top_decile_precision(self):
        y = np.array([1] * 10 + [0] * 90)
        scores = np.linspace(1, 0, 100)
        prec = top_decile_precision(y, scores)
        self.assertEqual(prec, 1.0, "Top decile should have 100% precision")

    def test_top_decile_incremental(self):
        # 100 HCPs, top 10 have high response under treatment
        y = np.array([5, 5, 4, 4, 1, 1, 0, 0, 0, 0] + [0] * 90)
        t = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [1, 0] * 45)
        scores = np.linspace(1, 0, 100)
        res = top_decile_incremental(scores, y, t, frac=0.1)
        self.assertIsNotNone(res["estimate"])
        self.assertGreater(res["estimate"], 0)

    def test_tlearner_fit_and_predict(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 5))
        t = rng.integers(0, 2, size=200)
        # outcome has positive treatment effect of 2.0
        y = X[:, 0] + 2.0 * t + rng.normal(scale=0.1, size=200)
        w = np.ones(200)
        tl = TLearner().fit(X, y, t, w)
        preds = tl.predict(X)
        self.assertEqual(len(preds), 200)
        # Average treatment effect per quarter / 3
        self.assertGreater(preds.mean(), 0)

    def test_database_model_runs_and_scores(self):
        c = self.conn
        runs = c.execute("SELECT model_type, metrics_json FROM model_run ORDER BY run_id").fetchall()
        types = [r["model_type"] for r in runs]
        self.assertIn("current", types)
        self.assertIn("uplift", types)

        scores_count = c.execute("SELECT count(*) as n FROM hcp_score").fetchone()["n"]
        self.assertEqual(scores_count, 8000, "Should have 4000 scores per model (8000 total)")

        features_count = c.execute("SELECT count(*) as n FROM hcp_feature").fetchone()["n"]
        self.assertEqual(features_count, 4000, "Should have feature snapshots for all 4000 HCPs")


if __name__ == "__main__":
    unittest.main()

