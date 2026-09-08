import unittest

from analyze_mega_moe_phases import summarize_capture, summarize_sample


def sample(offset=0, rank=0):
    rows = {
        "0": {"0": 1000, "1": 2000, "3": 5000, "8": 6000,
              "24": 9000, "32": 10000, "33": 12000, "40": 14000,
              "54": 25000, "64": 28000, "80": 30000,
              "128": 16000, "129": 20000, "130": 17000, "131": 22000},
        "1": {"0": 1100, "2": 4000, "128": 18000, "129": 24000},
    }
    return {"rank": rank, "replay": 0, "sm_markers_ns": {
        sm: {slot: value + offset for slot, value in row.items()}
        for sm, row in rows.items()
    }}


class PhaseAnalysisTest(unittest.TestCase):
    def test_independent_rank_clock_offsets_cannot_change_durations(self):
        first = summarize_sample(sample())
        shifted = summarize_sample(sample(987654321000, 9))
        self.assertEqual(first["metrics_us"], shifted["metrics_us"])
        self.assertEqual(first["phase_windows_from_local_entry_us"],
                         shifted["phase_windows_from_local_entry_us"])

    def test_whole_expert_ready_uses_latest_contributing_sm(self):
        result = summarize_sample(sample())
        experts = result["expert_epilogue_observations"]
        self.assertEqual(experts[0]["last_l2_epilogue_fragment_done_us"], 23.0)
        self.assertEqual(experts[1]["last_l2_epilogue_fragment_done_us"], 21.0)
        self.assertEqual(result["metrics_us"]["expert_last_l2_fragment_spread_us"], 2.0)
        self.assertEqual(result["metrics_us"]["earliest_expert_last_fragment_to_first_combine_issue_us"], 6.0)

    def test_absent_remote_markers_are_not_zero_latency(self):
        record = sample()
        record["sm_markers_ns"] = {"0": {"0": 1000, "3": 2000}}
        result = summarize_sample(record)
        self.assertIsNone(result["metrics_us"]["dispatch_issue_to_last_flush_us"])
        self.assertEqual(result["expert_epilogue_observations"], [])

    def test_missing_instrumentation_fails_closed(self):
        record = sample()
        record["sm_markers_ns"] = {}
        with self.assertRaisesRegex(ValueError, "entry marker"):
            summarize_sample(record)

    def test_expert_ready_markers_separate_payload_from_late_headers(self):
        record = sample()
        row = record["sm_markers_ns"]["0"]
        row.update({"99": 26000, "100": 40000, "101": 47000, "55": 49000,
                    "72": 60000, "80": 70000})
        metrics = summarize_sample(record)["metrics_us"]
        self.assertEqual(metrics["first_ready_expert_selection_to_first_combine_issue_us"], 2.0)
        self.assertEqual(metrics["first_combine_issue_to_all_payload_puts_queued_us"], 12.0)
        self.assertEqual(metrics["all_payload_puts_queued_to_payload_local_flush_us"], 7.0)
        self.assertEqual(metrics["payload_local_flush_to_combine_grid1_us"], 2.0)
        row.update({"72": 80000, "80": 90000})
        newer = summarize_sample(record)["metrics_us"]
        for name in ("first_ready_expert_selection_to_first_combine_issue_us",
                     "first_combine_issue_to_all_payload_puts_queued_us",
                     "all_payload_puts_queued_to_payload_local_flush_us",
                     "payload_local_flush_to_combine_grid1_us"):
            self.assertEqual(metrics[name], newer[name])
        old = summarize_sample(sample())["metrics_us"]
        self.assertIsNone(old["first_ready_expert_selection_to_first_combine_issue_us"])
        self.assertIsNone(old["all_payload_puts_queued_to_payload_local_flush_us"])

    def test_cleanup_region_can_include_combine_and_is_not_barrier_cost(self):
        record = sample()
        record["sm_markers_ns"]["0"].update({
            "62": 20000, "56": 30000, "57": 40000,
            "60": 42000, "63": 50000,
        })
        result = summarize_sample(record)
        self.assertNotIn("cleanup_barrier_span_us", result["metrics_us"])
        self.assertEqual(result["metrics_us"]["dispatch_cleanup_region_span_us"], 30.0)
        self.assertEqual(result["metrics_us"]["last_scatter_grid_exit_to_cleanup_rendezvous_exit_us"], 8.0)

    def test_incomplete_expert_marker_pair_is_rejected(self):
        record = sample()
        del record["sm_markers_ns"]["0"]["131"]
        with self.assertRaisesRegex(ValueError, "incomplete readiness pair"):
            summarize_sample(record)

    def test_capture_aggregates_only_precomputed_local_intervals(self):
        result = summarize_capture({
            "shape": {"tokens_per_rank": 32}, "diagnostic_level": 2,
            "routes": {"half_remote": [sample(), sample(999000000, 1)]},
        })
        stats = result["routes"]["half_remote"]["same_gpu_interval_distribution_us"]
        self.assertEqual(stats["dispatch_issue_to_last_flush_us"],
                         {"min": 3.0, "p50": 3.0, "max": 3.0})
        self.assertFalse(result["performance_claim"])


if __name__ == "__main__":
    unittest.main()
