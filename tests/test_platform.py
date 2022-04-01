import numpy as np
import pandas as pd

from unittest import TestCase

from experimentation_tool.platform import Platform


class ExperimentPlatformUnitTest(TestCase):
    def test_sample_size(self):
        # Compare with https://www.evanmiller.org/ab-testing/sample-size.html#!20;80;5;6;1
        sample = Platform.calculate_sample_size(
            alpha=0.05, stats_power=0.8, base_rate=0.2, pct_min_detectable_efect=0.06
        )
        self.assertEquals(round(sample), 17557)

    def test_healthy_buckets(self):
        self.assertTrue(Platform.are_buckets_balanced(9, 10))

    def test_unhealthy_buckets(self):
        self.assertFalse(Platform.are_buckets_balanced(1, 10))

    def test_split_traffic(self):
        ids = np.arange(100**2)
        data = pd.DataFrame({"id": ids})
        experiment_id = "exp12"
        control_group_size = 0.5  # 50/50 split
        Platform.split_traffic(1765984, experiment_id, control_group_size)
        data["bucket"] = data["id"].apply(
            lambda id: Platform.split_traffic(id, experiment_id, control_group_size)
        )
        test_count = data[data.bucket == "t"].bucket.count()
        control_count = data[data.bucket == "c"].bucket.count()
        greater_count = test_count if test_count > control_count else control_count
        self.assertAlmostEqual(test_count, control_count, delta=0.05 * greater_count)

    def test_split_traffic_naive(self):
        ids = np.arange(100**2)
        data = pd.DataFrame({"id": ids})
        data["bucket"] = data["id"].apply(lambda id: Platform.split_traffic_naive(id))
        test_count = data[data.bucket == "t"].bucket.count()
        control_count = data[data.bucket == "c"].bucket.count()
        self.assertNotEqual(test_count, control_count)
