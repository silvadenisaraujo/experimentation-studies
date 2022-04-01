import hashlib
from abc import ABC

import numpy as np
from scipy.stats import chisquare, norm, ttest_ind

from experimentation_tool.experiment import Experiment


class Database(ABC):
    def get_metric_control(self, metric: str, experiment_id: str):
        raise NotImplementedError

    def get_metric_test(self, metric: str, experiment_id: str):
        raise NotImplementedError


class Platform:
    def __init__(self, database: Database):
        self.experiments = []
        self.database = database

    @classmethod
    def calculate_sample_size(
        cls, alpha, stats_power, base_rate, pct_min_detectable_efect
    ) -> float:
        """Calculates the minimum sample size for a valid statistical A/B test.
        Based on https://www.evanmiller.org/ab-testing/sample-size.html

        Args:
            alpha (float): How often you expect the test to fail?
            stats_power (float): How often you expect to correctly map a true positive?
            base_rate (float): Base convertion rate
            pct_min_detectable_efect (float): Minimum relative of the base rate that is detectable

        Returns:
            float: Sample size
        """
        delta = base_rate * pct_min_detectable_efect
        t_alpha2 = norm.ppf(1.0 - alpha / 2)
        t_beta = norm.ppf(stats_power)

        sd1 = np.sqrt(2 * base_rate * (1.0 - base_rate))
        sd2 = np.sqrt(
            base_rate * (1.0 - base_rate)
            + (base_rate + delta) * (1.0 - base_rate - delta)
        )

        return (
            (t_alpha2 * sd1 + t_beta * sd2)
            * (t_alpha2 * sd1 + t_beta * sd2)
            / (delta * delta)
        )

    @classmethod
    def split_traffic(cls, id, experiment_id, control_group_size) -> str:
        """
        Split user traffic into two groups.
        't' for test group, 'c' for control group.
        Based on hash split: http://blog.richardweiss.org/2016/12/25/hash-splits.html

        Args:
            id (int): An stable user id
            experiment_id (int): An unique experiment id
            control_group_size (float): Size of the control group

        Returns:
            str: 't' for test or 'c' for control group
        """
        test_id = str(id) + "-" + str(experiment_id)
        test_id_digest = hashlib.md5(test_id.encode("ascii")).hexdigest()
        test_id_first_digits = test_id_digest[:6]
        test_id_final_int = int(test_id_first_digits, 16)
        ab_split = test_id_final_int / 0xFFFFFF

        if ab_split > control_group_size:
            return "t"
        else:
            return "c"

    @classmethod
    def split_traffic_naive(cls, id: int, prime_n: int = 7) -> str:
        """
        Split user traffic into two groups.
        't' for test group, 'c' for control group.
        Based on a naive modulo arithmetic.

        Args:
            id (int): An stable user id
            prime_n (int): A prime number

        Returns:
            str: 't' for test or 'c' for control group
        """
        ab_split = id % prime_n
        if ab_split > prime_n // 2:
            return "t"
        else:
            return "c"

    def calculate_significance(self, experiment: Experiment) -> float:
        """Calculates the statistical significance of the delta.
        Based on https://www.evanmiller.org/how-not-to-run-an-ab-test.html
        Do not report significance levels until an experiment is over,
        and stop using significance levels to decide whether an experiment should stop or continue.
        Args:
            experiment (Experiment): Experiment

        Returns:
            float: Significance value
        """
        metric_control = self.database.get_metric_control(
            metric=experiment.metric, experiment_id=experiment.id
        )
        metric_test = self.database.get_metric_test(experiment)
        if self.status == "DONE":
            return ttest_ind(metric_control, metric_test).pvalue
        else:
            return -1

    @classmethod
    def are_buckets_balanced(cls, observed: float, expected: float) -> bool:
        """Evaluates if the A/B buckets of an experiment as balanced.
        Based on https://blog.twitter.com/engineering/en_us/a/2015/detecting-and-avoiding-bucket-imbalance-in-ab-tests

        Args:
            observed (float): Number of observations for a bucket
            expected (float): Expected number of observations for a bucket

        Returns:
            bool: Buckets are balanced if chi-square test is between 0 and 1
        """
        return 0 < chisquare([observed], [expected]).statistic < 1

    def calculate_delta(self, experiment: Experiment) -> float:
        """Calculates the delta for a given experiment.
        between the control group and the test group.

        Args:
            experiment (Experiment): An experiment

        Returns:
            float: Delta
        """
        metric_control = self.database.get_metric_control(
            metric=experiment.metric, experiment_id=experiment.id
        )
        metric_test = self.database.get_metric_test(
            metric=experiment.metric, experiment_id=experiment.id
        )
        return metric_control - metric_test
