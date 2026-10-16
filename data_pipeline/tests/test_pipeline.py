"""Mathematical, failure-mode, provenance, API and restart checks."""
import copy
import http.client
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from pipeline import ExposurePipeline, utc_string
from server import ApiServer, PipelineStore, MAX_BODY_BYTES

BASE = datetime(2026, 10, 16, 17, 0, tzinfo=timezone.utc)


def record(start=0, seconds=10, value=100.0, **updates):
    result = {
        "device_id": "dev-1", "firefighter_id": "crew-1", "session_id": "shift-1",
        "sample_start_utc": utc_string(BASE + timedelta(seconds=start)),
        "sample_end_utc": utc_string(BASE + timedelta(seconds=start + seconds)),
        "pm25_ug_m3": value, "latitude_deg": None, "longitude_deg": None,
        "location_time_utc": None, "location_accuracy_m": None, "battery_percent": 80,
        "quality_status": "valid", "source": "sensor",
        "received_at_utc": utc_string(BASE + timedelta(seconds=start + seconds)),
    }
    result.update(updates)
    return result


def run(pipeline, item, now=None):
    return pipeline.process(item, now=now or item["received_at_utc"])


class ExposureMathTests(unittest.TestCase):
    def test_100_for_ten_minutes_equals_1000(self):
        result = run(ExposurePipeline(), record(seconds=600))
        self.assertAlmostEqual(result["observed_exposure_index_ug_min_m3"], 1000)
        self.assertEqual(result["valid_duration_seconds"], 600)
        self.assertEqual(result["observed_time_weighted_pm25_ug_m3"], 100)
        self.assertEqual(result["rolling_5min_mean"], 100)
        self.assertEqual(result["rolling_observed_seconds"], 300)
        self.assertEqual(result["coverage_ratio"], 1)

    def test_unequal_intervals_use_actual_duration(self):
        pipeline = ExposurePipeline()
        run(pipeline, record(seconds=30, value=10))
        result = run(pipeline, record(start=30, seconds=90, value=30))
        self.assertAlmostEqual(result["observed_exposure_index_ug_min_m3"], 50)
        self.assertAlmostEqual(result["observed_time_weighted_pm25_ug_m3"], 25)

    def test_unexpected_interval_is_flagged_but_actual_exposure_preserved(self):
        result = run(ExposurePipeline(), record(seconds=3600, value=100))
        self.assertIn("interval_cadence_mismatch", result["flags"])
        self.assertEqual(result["status"], "insufficient_data")
        self.assertEqual(result["observed_exposure_index_ug_min_m3"], 6000)

    def test_gap_not_filled_and_coverage_accounts_for_it(self):
        pipeline = ExposurePipeline()
        run(pipeline, record())
        result = run(pipeline, record(start=20))
        self.assertAlmostEqual(result["observed_exposure_index_ug_min_m3"], 100 / 3)
        self.assertAlmostEqual(result["coverage_ratio"], 2 / 3)
        self.assertEqual(result["observed_time_weighted_pm25_ug_m3"], 100)
        self.assertIn("data_gap", result["flags"])
        self.assertEqual(result["status"], "insufficient_data")

    def test_rolling_window_clips_partial_interval(self):
        pipeline = ExposurePipeline()
        run(pipeline, record(seconds=200, value=100))
        result = run(pipeline, record(start=200, seconds=200, value=200))
        self.assertAlmostEqual(result["rolling_5min_mean"], (100 * 100 + 200 * 200) / 300)
        self.assertEqual(result["rolling_observed_seconds"], 300)

    def test_invalid_quality_keeps_existing_exposure(self):
        pipeline = ExposurePipeline()
        first = run(pipeline, record())
        bad = run(pipeline, record(start=10, value=None, quality_status="sensor_error"))
        self.assertEqual(bad["observed_exposure_index_ug_min_m3"], first["observed_exposure_index_ug_min_m3"])
        self.assertEqual(bad["valid_duration_seconds"], 10)
        self.assertEqual(bad["status"], "insufficient_data")
        self.assertEqual(bad["coverage_ratio"], 0.5)

    def test_no_valid_data_has_no_average(self):
        result = run(ExposurePipeline(), record(value=None, quality_status="warming_up"))
        self.assertIsNone(result["observed_time_weighted_pm25_ug_m3"])
        self.assertIsNone(result["rolling_5min_mean"])
        self.assertEqual(result["valid_duration_seconds"], 0)
        self.assertEqual(result["coverage_ratio"], 0)

    def test_trend_is_causal_and_uses_observed_points(self):
        pipeline = ExposurePipeline()
        results = [run(pipeline, record(start=index * 60, seconds=60, value=10 + 10 * index)) for index in range(3)]
        self.assertIsNone(results[1]["trend"]["slope_ug_m3_per_min"])
        self.assertAlmostEqual(results[2]["trend"]["slope_ug_m3_per_min"], 10)
        snapshot = copy.deepcopy(results[2])
        run(pipeline, record(start=180, seconds=60, value=500))
        self.assertEqual(results[2], snapshot)

    def test_receipt_time_does_not_replace_sample_time(self):
        result = run(ExposurePipeline(), record(received_at_utc=utc_string(BASE + timedelta(seconds=80))))
        self.assertEqual(result["latest_reading_age"], 70)
        self.assertEqual(result["valid_duration_seconds"], 10)
        self.assertAlmostEqual(result["coverage_ratio"], 10 / 80)
        self.assertEqual(result["status"], "insufficient_data")


class OrderingAndIsolationTests(unittest.TestCase):
    def test_duplicate_not_counted(self):
        pipeline = ExposurePipeline()
        first = run(pipeline, record())
        second = run(pipeline, record())
        self.assertFalse(second["input_accepted"])
        self.assertIn("duplicate_interval", second["flags"])
        self.assertEqual(first["observed_exposure_index_ug_min_m3"], second["observed_exposure_index_ug_min_m3"])

    def test_overlap_not_partially_counted(self):
        pipeline = ExposurePipeline()
        run(pipeline, record())
        result = run(pipeline, record(start=5, seconds=10))
        self.assertIn("overlapping_interval", result["flags"])
        self.assertEqual(result["valid_duration_seconds"], 10)
        self.assertFalse(result["measurement_accumulated"])

    def test_out_of_order_not_counted(self):
        pipeline = ExposurePipeline()
        run(pipeline, record(start=20))
        result = run(pipeline, record(start=0), now=BASE + timedelta(seconds=40))
        self.assertIn("out_of_order", result["flags"])
        self.assertEqual(result["valid_duration_seconds"], 10)

    def test_stream_key_isolates_every_dimension(self):
        pipeline = ExposurePipeline()
        run(pipeline, record(seconds=60))
        for fields in ({"device_id": "dev-2"}, {"firefighter_id": "crew-2"}, {"session_id": "shift-2"}, {"source": "simulated"}, {"source": "replay"}):
            with self.subTest(fields=fields):
                result = run(pipeline, record(seconds=10, **fields))
                self.assertAlmostEqual(result["observed_exposure_index_ug_min_m3"], 100 / 6)
                self.assertEqual(result["valid_duration_seconds"], 10)

    def test_future_interval_never_counted(self):
        result = run(ExposurePipeline(), record(start=10), now=BASE + timedelta(seconds=15))
        self.assertIn("future_sample", result["flags"])
        self.assertFalse(result["measurement_accumulated"])

    def test_future_receipt_rejected(self):
        result = run(ExposurePipeline(), record(received_at_utc=utc_string(BASE + timedelta(seconds=20))), now=BASE + timedelta(seconds=10))
        self.assertIn("future_received_time", result["flags"])
        self.assertFalse(result["input_accepted"])

    def test_receipt_before_sample_end_rejected(self):
        result = run(ExposurePipeline(), record(received_at_utc=utc_string(BASE + timedelta(seconds=5))), now=BASE + timedelta(seconds=10))
        self.assertIn("received_before_sample_end", result["flags"])
        self.assertFalse(result["input_accepted"])

    def test_zero_or_negative_interval_rejected(self):
        for seconds in [0, -10]:
            with self.subTest(seconds=seconds):
                result = run(ExposurePipeline(), record(seconds=seconds), now=BASE)
                self.assertIn("invalid_interval", result["flags"])
                self.assertEqual(result["valid_duration_seconds"], 0)

    def test_latest_recomputes_staleness_without_new_record(self):
        pipeline = ExposurePipeline()
        run(pipeline, record())
        result = pipeline.latest("crew-1", "shift-1", now=BASE + timedelta(seconds=100))
        self.assertEqual(result["latest_reading_age"], 90)
        self.assertEqual(result["status"], "insufficient_data")
        self.assertAlmostEqual(result["coverage_ratio"], 0.1)

    def test_empty_session_lookup_returns_none(self):
        self.assertIsNone(ExposurePipeline().latest("unknown", "unknown"))


class ReportsAndSafetyTests(unittest.TestCase):
    def test_explicit_urgent_overrides_missing_measurement(self):
        result = run(ExposurePipeline(), record(value=None, quality_status="sensor_error", symptoms=[None, 9, "request_help"]))
        self.assertEqual(result["status"], "urgent_assistance")
        self.assertTrue(result["urgent_latched"])
        self.assertIn("malformed_symptoms", result["flags"])
        self.assertEqual(result["health_status"], "reported_concern")
        self.assertIsNone(result["evacuation_required"])

    def test_urgent_persists_after_normal_data_and_empty_reports(self):
        pipeline = ExposurePipeline()
        run(pipeline, record(symptoms=["confusion"]))
        result = run(pipeline, record(start=10, symptoms=[]))
        self.assertEqual(result["status"], "urgent_assistance")
        self.assertEqual(result["withdrawal_recommendation"], "urgent_human_assistance")

    def test_invalid_timestamps_cannot_mask_urgent(self):
        result = ExposurePipeline().process(record(sample_start_utc="broken", symptoms=["loss_of_consciousness"]), now=BASE)
        self.assertEqual(result["status"], "urgent_assistance")
        self.assertFalse(result["measurement_accumulated"])

    def test_missing_device_cannot_erase_attributable_urgent(self):
        pipeline = ExposurePipeline()
        initial = run(pipeline, record(device_id=None, symptoms=["request_help"]))
        later = run(pipeline, record(start=10))
        self.assertEqual(initial["status"], "urgent_assistance")
        self.assertEqual(later["status"], "urgent_assistance")

    def test_missing_person_returns_unattributed_urgent(self):
        result = run(ExposurePipeline(), record(firefighter_id=None, symptoms=["request_help"]))
        self.assertEqual(result["status"], "urgent_assistance")
        self.assertIn("unattributed_urgent_report", result["flags"])

    def test_urgent_shares_devices_but_not_sessions_sources_or_people(self):
        pipeline = ExposurePipeline()
        run(pipeline, record(symptoms=["request_help"]))
        self.assertEqual(run(pipeline, record(device_id="dev-2"))["status"], "urgent_assistance")
        for fields in ({"session_id": "other"}, {"firefighter_id": "other"}, {"source": "simulated"}):
            self.assertNotEqual(run(pipeline, record(**fields))["status"], "urgent_assistance")

    def test_latest_live_urgent_cannot_be_hidden_by_later_simulation(self):
        pipeline = ExposurePipeline()
        run(pipeline, record(symptoms=["request_help"]))
        run(pipeline, record(start=10, source="simulated"))
        result = pipeline.latest("crew-1", "shift-1", now=BASE + timedelta(seconds=20))
        self.assertEqual(result["source"], "sensor")
        self.assertEqual(result["status"], "urgent_assistance")
        self.assertEqual(len(result["available_streams"]), 2)

    def test_simulated_urgent_does_not_contaminate_sensor(self):
        pipeline = ExposurePipeline()
        run(pipeline, record(source="simulated", symptoms=["request_help"]))
        result = run(pipeline, record())
        self.assertEqual(result["status"], "monitoring")
        self.assertFalse(result["urgent_latched"])

    def test_huge_sensor_concentration_is_not_a_health_label(self):
        pipeline = ExposurePipeline()
        for index in range(60):
            result = run(pipeline, record(start=index * 10, value=10000))
        self.assertEqual(result["status"], "monitoring")
        self.assertEqual(result["health_status"], "not_assessed")
        self.assertIsNone(result["evacuation_required"])
        self.assertFalse(result["simulation_rules_active"])
        self.assertFalse(result["clinical_model_available"])
        self.assertTrue(result["clinical_model_unavailable_reason"])

    def test_demo_rules_only_apply_to_simulated(self):
        for source in ("sensor", "replay", "simulated"):
            pipeline = ExposurePipeline()
            for index in range(6):
                result = run(pipeline, record(start=index * 10, value=200, source=source))
            self.assertEqual(result["status"], "check_in" if source == "simulated" else "monitoring")
            self.assertIsNone(result["evacuation_required"])

    def test_nonurgent_report_requires_checkin_not_clinical_label(self):
        result = run(ExposurePipeline(), record(symptoms=["feeling_unwell"]))
        self.assertEqual(result["status"], "check_in")
        self.assertEqual(result["health_status"], "reported_concern")
        self.assertIsNone(result["evacuation_required"])


class MalformedInputTests(unittest.TestCase):
    def test_bad_numeric_values_are_rejected_without_masking_help(self):
        for value in [True, False, -1, float("nan"), float("inf"), "100", {}, [], 10 ** 1000, 1e308]:
            with self.subTest(value_type=type(value).__name__):
                result = run(ExposurePipeline(), record(value=value, symptoms=["request_help"]))
                self.assertFalse(result["measurement_accumulated"])
                self.assertIn("invalid_pm25", result["flags"])
                self.assertEqual(result["status"], "urgent_assistance")
                json.dumps(result, allow_nan=False)

    def test_required_timestamp_formats(self):
        for value in [None, True, "2026-10-16T17:00:00", "not-a-date", "2026-10-16T18:00:00+01:00"]:
            result = run(ExposurePipeline(), record(sample_start_utc=value))
            self.assertFalse(result["measurement_accumulated"])
            self.assertIn("invalid_timestamp", result["flags"])

    def test_invalid_source_and_quality_types_do_not_crash(self):
        for item in [record(source=[]), record(quality_status={}), record(device_id=[]), None, []]:
            result = ExposurePipeline().process(item, now=BASE)
            self.assertEqual(result["status"], "insufficient_data")

    def test_invalid_auxiliary_values_do_not_change_integral(self):
        result = run(ExposurePipeline(), record(latitude_deg=999, battery_percent=True, location_accuracy_m=-1))
        self.assertAlmostEqual(result["observed_exposure_index_ug_min_m3"], 100 / 6)
        self.assertEqual(result["status"], "check_in")
        self.assertIn("invalid_latitude_deg", result["flags"])

    def test_malformed_symptoms_string_still_detects_request(self):
        result = run(ExposurePipeline(), record(symptoms="request_help"))
        self.assertEqual(result["status"], "urgent_assistance")
        self.assertIn("malformed_symptoms", result["flags"])

    def test_exposure_threshold_cannot_enable_unvalidated_alert(self):
        with self.assertRaises(ValueError):
            ExposurePipeline({"exposure_alert_threshold_ug_min_m3": 100})


class PersistenceAndApiTests(unittest.TestCase):
    def setUp(self):
        scratch = BASE_DIR / "work"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="pipeline_test_", dir=scratch)
        self.db = Path(self.temp.name) / "events.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def test_restart_restores_exposure_order_and_urgent_latch(self):
        store = PipelineStore(self.db)
        store.ingest([record(symptoms=["request_help"])], now=BASE + timedelta(seconds=10))
        store.ingest([record(start=10)], now=BASE + timedelta(seconds=20))
        before = store.latest("crew-1", "shift-1", now=BASE + timedelta(seconds=20))
        store.close()
        store = PipelineStore(self.db)
        try:
            after = store.latest("crew-1", "shift-1", now=BASE + timedelta(seconds=20))
            self.assertEqual(before, after)
            self.assertEqual(store.recovered_count, 2)
            duplicate = store.ingest([record(start=10)], now=BASE + timedelta(seconds=20))[0]
            self.assertIn("duplicate_interval", duplicate["flags"])
            self.assertEqual(duplicate["valid_duration_seconds"], 20)
        finally:
            store.close()

    def test_transaction_failure_restores_processor_state(self):
        store = PipelineStore(self.db)
        try:
            with self.assertRaises(ValueError):
                store.ingest([record(), record(start=10, value=float("nan"))], now=BASE + timedelta(seconds=20))
            self.assertIsNone(store.latest("crew-1", "shift-1"))
            self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0], 0)
        finally:
            store.close()

    def test_api_routes_validation_and_private_host_checks(self):
        store = PipelineStore(self.db)
        service = ApiServer(("127.0.0.1", 0), store)
        thread = threading.Thread(target=service.serve_forever, daemon=True)
        thread.start()
        port = service.server_address[1]

        def request(method, path, body=None, headers=None):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                conn.request(method, path, body=body, headers=headers or {})
                response = conn.getresponse()
                return response.status, json.loads(response.read())
            finally:
                conn.close()
        try:
            self.assertEqual(request("GET", "/health")[0], 200)
            self.assertEqual(request("GET", "/health", headers={"Host": "attacker.invalid"})[0], 403)
            self.assertEqual(request("GET", "/health", headers={"Origin": "https://other.invalid"})[0], 403)
            self.assertEqual(request("GET", "/../../secret.txt")[0], 404)
            self.assertEqual(request("GET", "/v1/crew/%2e%2e?session_id=s")[0], 400)
            self.assertEqual(request("GET", "/v1/crew/crew-1")[0], 400)
            self.assertEqual(request("POST", "/v1/readings", "{}", {"Content-Type": "text/plain"})[0], 415)
            self.assertEqual(request("POST", "/v1/readings", "{broken", {"Content-Type": "application/json"})[0], 400)
            self.assertEqual(request("POST", "/v1/readings", '{"x":NaN}', {"Content-Type": "application/json"})[0], 400)
            self.assertEqual(request("POST", "/v1/readings", '{}', {"Content-Type": "application/json", "Content-Length": str(MAX_BODY_BYTES + 1)})[0], 413)
            status, data = request("POST", "/v1/readings", json.dumps({"records": [record(symptoms=["request_help"]), record(start=10)]}), {"Content-Type": "application/json"})
            self.assertEqual(status, 200)
            self.assertEqual(data["count"], 2)
            self.assertEqual(data["results"][1]["status"], "urgent_assistance")
            status, data = request("GET", "/v1/crew/crew-1?session_id=shift-1")
            self.assertEqual(status, 200)
            self.assertEqual(data["status"], "urgent_assistance")
            self.assertIsNone(data["evacuation_required"])
        finally:
            service.shutdown()
            service.server_close()
            thread.join(timeout=3)
            store.close()

    def test_public_bind_is_prohibited(self):
        store = PipelineStore(self.db)
        try:
            with self.assertRaises(ValueError):
                ApiServer(("0.0.0.0", 8766), store)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
