import unittest
from fastapi.testclient import TestClient
from app.main import app
from app.config import ADMIN_TOKEN


class TestApiEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data.get("ok"))
        self.assertIn("week", data)

    def test_territories_endpoint(self):
        r = self.client.get("/territories")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("territories", data)
        self.assertIn("regions", data)
        self.assertIn("specialties", data)
        self.assertEqual(len(data["territories"]), 40)
        self.assertGreater(len(data["regions"]), 0)

    def test_hcps_list_endpoint(self):
        r = self.client.get("/hcps?limit=10&flagged_only=true")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("items", data)
        self.assertIn("total", data)
        self.assertIn("flagged_cutoff", data)
        self.assertEqual(len(data["items"]), 10)
        first = data["items"][0]
        self.assertIn("hcp_id", first)
        self.assertIn("display_name", first)
        self.assertIn("uplift", first)
        self.assertIn("call_status", first)
        self.assertIn("band", first)

    def test_hcps_filter_by_territory(self):
        r_terr = self.client.get("/territories").json()
        terr_id = r_terr["territories"][0]["territory_id"]
        r = self.client.get(f"/hcps?territory={terr_id}&flagged_only=false")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        for item in data["items"]:
            self.assertEqual(item["territory_id"], terr_id)

    def test_hcp_detail_endpoint(self):
        # Fetch first HCP from list
        r_list = self.client.get("/hcps?limit=1")
        hcp_id = r_list.json()["items"][0]["hcp_id"]

        r = self.client.get(f"/hcps/{hcp_id}")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["hcp"]["hcp_id"], hcp_id)
        self.assertIn("drivers", data)
        self.assertIn("features", data)
        self.assertIn("rx_history", data)
        self.assertIn("recent_calls", data)

    def test_metrics_endpoint(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("current_model", data)
        self.assertIn("uplift_model", data)
        self.assertIn("verification", data)
        self.assertIn("adoption", data)
        # Check Qini curve is present
        self.assertIn("qini", data["uplift_model"])
        self.assertIn("uplift", data["uplift_model"]["qini"])

    def test_feedback_and_event_logging(self):
        r_list = self.client.get("/hcps?limit=1")
        item = r_list.json()["items"][0]
        hcp_id, rep_id = item["hcp_id"], item["rep_id"]

        # Post feedback
        fb_payload = {
            "hcp_id": hcp_id,
            "rep_id": rep_id,
            "action": "will_call",
            "reason_code": "high_growth_potential"
        }
        r_fb = self.client.post("/feedback", json=fb_payload)
        self.assertEqual(r_fb.status_code, 200)
        self.assertIn("feedback_id", r_fb.json())

        # Post event
        ev_payload = {
            "rep_id": rep_id,
            "event_type": "list_open",
            "hcp_id": hcp_id
        }
        r_ev = self.client.post("/events", json=ev_payload)
        self.assertEqual(r_ev.status_code, 200)
        self.assertIn("event_id", r_ev.json())

    def test_admin_rescore_auth(self):
        # Without token should be 401
        r_fail = self.client.post("/admin/rescore?step=score")
        self.assertEqual(r_fail.status_code, 401)


if __name__ == "__main__":
    unittest.main()

