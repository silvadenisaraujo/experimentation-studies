from dataclasses import dataclass
from typing import Literal


@dataclass
class Experiment:
    experiment_id: int
    name: str
    description: str
    metric: str
    control_metric: str
    traffic_percentage: float
    control_group_size: float
    baseline_converstion_rate: float
    minimum_detectable_effect: float
    status: Literal["DONE", "NEW", "RUNNING"]
    min_significance: float = 0.95
