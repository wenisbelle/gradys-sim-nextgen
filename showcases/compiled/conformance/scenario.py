"""
Conformance protocol: nodes take random actions using every mechanism available to compiled protocols (broadcast,
unicast, timers, timer cancellation, movement, speed changes) and record every callback they receive. Comparing the
recorded traces of two implementations checks that they behave identically, event by event.
"""
from dataclasses import dataclass

import numpy as np

TIMER_IDS = 4
MESSAGE_SIZE = 3


@dataclass
class ConformanceScenario:
    nodes: int = 8
    duration: float = 60.0
    area: float = 80.0
    transmission_range: float = 70.0
    delay: float = 0.0
    update_rate: float = 0.05
    telemetry_decimation: int = 1
    seed: int = 0

    def layout(self):
        rng = np.random.default_rng(self.seed)
        kinds = np.zeros(self.nodes, dtype=np.int64)
        positions = np.zeros((self.nodes, 3))
        positions[:, :2] = rng.uniform(-self.area, self.area, (self.nodes, 2))
        return kinds, positions
