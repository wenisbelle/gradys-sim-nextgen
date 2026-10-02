"""
Data collection: drones loop over waypoint missions, sensors accumulate packets and hand them to drones that ping
them, the ground station collects packets from the drones. Same logic as showcases/simple.
"""
from dataclasses import dataclass

import numpy as np

GROUND, DRONE, SENSOR = 0, 1, 2
"""Node kinds, which are also the indexes of their protocols"""


@dataclass
class DataCollectionScenario:
    drones: int = 10
    sensors: int = 20
    duration: float = 300.0
    area: float = 100.0
    transmission_range: float = 30.0
    delay: float = 0.0
    update_rate: float = 0.01
    telemetry_decimation: int = 1
    speed: float = 5.0
    tolerance: float = 0.5
    waypoints: int = 4
    seed: int = 0

    def layout(self):
        """Node kinds, initial positions and drone missions, all derived from the seed"""
        rng = np.random.default_rng(self.seed)
        kinds = [GROUND] + [DRONE] * self.drones + [SENSOR] * self.sensors
        positions = [(0.0, 0.0, 0.0)] + [(0.0, 0.0, 5.0)] * self.drones
        positions += [(float(x), float(y), 0.0) for x, y in rng.uniform(-self.area, self.area, (self.sensors, 2))]
        missions = np.zeros((self.drones, self.waypoints, 3))
        missions[:, 0] = (0.0, 0.0, 5.0)
        missions[:, 1:, :2] = rng.uniform(-self.area, self.area, (self.drones, self.waypoints - 1, 2))
        missions[:, 1:, 2] = 5.0
        return np.array(kinds, dtype=np.int64), np.array(positions), missions
