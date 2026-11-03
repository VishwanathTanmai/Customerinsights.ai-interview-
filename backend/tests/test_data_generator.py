import unittest
from app.db import connect


class TestDataGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = connect()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_hcp_count_and_specialties(self):
        c = self.conn
        total = c.execute("SELECT count(*) AS n FROM hcp").fetchone()["n"]
        self.assertEqual(total, 4000, "Should generate exactly 4000 HCPs")

        specs = c.execute("SELECT specialty, count(*) as n FROM hcp GROUP BY specialty").fetchall()
        spec_dict = {r["specialty"]: r["n"] for r in specs}
        self.assertIn("Dermatology", spec_dict)
        self.assertIn("Primary Care", spec_dict)
        self.assertIn("Derm NP/PA", spec_dict)
        self.assertGreater(spec_dict["Dermatology"], 1000)
        self.assertGreater(spec_dict["Primary Care"], 1000)

    def test_territories_and_regions(self):
        c = self.conn
        territory_count = c.execute("SELECT count(*) as n FROM territory").fetchone()["n"]
        self.assertEqual(territory_count, 40, "Should have 40 territories")

        regions = c.execute("SELECT DISTINCT region FROM territory").fetchall()
        self.assertGreaterEqual(len(regions), 3, "Should have multiple regions")

    def test_random_pilot_subset(self):
        c = self.conn
        pilot_count = c.execute("SELECT count(*) as n FROM random_pilot").fetchone()["n"]
        ratio = pilot_count / 4000.0
        self.assertTrue(0.15 <= ratio <= 0.25, f"Random pilot should be 15-25% of universe, got {ratio:.2%}")

    def test_rx_history_completeness(self):
        c = self.conn
        rx_count = c.execute("SELECT count(*) as n FROM rx_history").fetchone()["n"]
        # 4000 HCPs x 15 periods = 60,000 rows
        self.assertEqual(rx_count, 60000, "Should have 15 periods per HCP")

    def test_ground_truth_restricted(self):
        c = self.conn
        gt_count = c.execute("SELECT count(*) as n FROM restricted.ground_truth").fetchone()["n"]
        self.assertEqual(gt_count, 4000, "Restricted ground truth must contain all 4000 HCPs")


if __name__ == "__main__":
    unittest.main()

