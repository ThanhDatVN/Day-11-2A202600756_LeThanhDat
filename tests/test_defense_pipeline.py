"""Regression tests for every required Assignment 11 pipeline behavior."""
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from defense_pipeline import (  # noqa: E402
    ATTACK_QUERIES,
    EDGE_CASES,
    SAFE_QUERIES,
    DefensePipeline,
    MonitoringAlert,
)


class DefensePipelineTests(unittest.TestCase):
    """Ensure each independent safety layer meets the assignment contract."""

    def test_safe_queries_pass_with_judge_scores(self):
        """Safe banking queries must pass and receive four judge scores."""
        pipeline = DefensePipeline()
        results = [pipeline.process(query, f"safe-{i}") for i, query in enumerate(SAFE_QUERIES)]
        self.assertTrue(all(result.status == "PASS" for result in results))
        self.assertTrue(all(result.judge and result.judge.verdict == "PASS" for result in results))

    def test_all_attacks_block_at_input(self):
        """The seven required attacks must be stopped before model generation."""
        pipeline = DefensePipeline()
        results = [pipeline.process(query, f"attack-{i}") for i, query in enumerate(ATTACK_QUERIES)]
        self.assertTrue(all(result.status == "BLOCKED" for result in results))
        self.assertTrue(all(result.blocked_by == "input_guard" for result in results))

    def test_rate_limit_allows_ten_then_blocks_five(self):
        """Sliding-window rate limiting must enforce the required 10/5 split."""
        pipeline = DefensePipeline()
        results = [pipeline.process(SAFE_QUERIES[0], "rapid-user") for _ in range(15)]
        self.assertEqual(["PASS"] * 10, [result.status for result in results[:10]])
        self.assertEqual(["BLOCKED"] * 5, [result.status for result in results[10:]])
        self.assertTrue(all(result.blocked_by == "rate_limiter" for result in results[10:]))

    def test_edge_cases_are_blocked(self):
        """Malformed, SQL, and off-topic inputs must not reach the responder."""
        pipeline = DefensePipeline()
        results = [pipeline.process(query, f"edge-{i}") for i, query in enumerate(EDGE_CASES)]
        self.assertTrue(all(result.status == "BLOCKED" for result in results))

    def test_output_redaction_and_audit_export(self):
        """Sensitive output must be redacted and every request must be auditable."""
        pipeline = DefensePipeline(lambda _: "Email test@vinbank.com; password: secret; API key sk-demo-key.")
        result = pipeline.process(SAFE_QUERIES[0], "redaction-user")
        self.assertEqual("PASS", result.status)
        self.assertNotIn("test@vinbank.com", result.response)
        self.assertNotIn("secret", result.response)
        self.assertGreaterEqual(result.response.count("[REDACTED]"), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = pipeline.audit.export_json(Path(directory) / "audit.json")
            self.assertEqual(1, len(json.loads(path.read_text(encoding="utf-8"))))

    def test_monitoring_fires_rate_limit_alert(self):
        """Monitoring must alert after five rate-limit hits."""
        pipeline = DefensePipeline()
        for _ in range(15):
            pipeline.process(SAFE_QUERIES[0], "rapid-user")
        monitor = MonitoringAlert()
        metrics = monitor.metrics(pipeline.audit.records)
        self.assertEqual(5, metrics["rate_limit_hits"])
        self.assertTrue(any("rate-limit" in alert for alert in monitor.alerts(metrics)))


if __name__ == "__main__":
    unittest.main()
