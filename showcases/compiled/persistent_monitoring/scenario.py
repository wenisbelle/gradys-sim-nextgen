"""
Persistent monitoring: drones patrol a grid of cells. Each drone keeps its own map of when every cell was last
visited, chooses its next cell by idleness minus a distance penalty, and periodically shares its map with the drones
in range, which merge it into theirs. The quality of a strategy is the time-averaged idleness of the cells.
"""
from dataclasses import dataclass

import numpy as np

DRONE = 0
BROADCAST_TIMER = 0


@dataclass
class PersistentMonitoringScenario:
    drones: int = 10
    cells_x: int = 6
    cells_y: int = 6
    area: float = 150.0
    altitude: float = 10.0
    duration: float = 600.0
    transmission_range: float = 60.0
    delay: float = 0.0
    update_rate: float = 0.01
    telemetry_decimation: int = 1
    speed: float = 8.0
    tolerance: float = 0.5
    distance_weight: float = 1.0
    """Seconds of idleness traded for one meter of travel when choosing the next cell"""
    broadcast_period: float = 2.0
    seed: int = 0

    @property
    def cell_count(self) -> int:
        return self.cells_x * self.cells_y

    def layout(self):
        """Node kinds, initial positions and cell centers"""
        kinds = np.zeros(self.drones, dtype=np.int64)
        positions = np.zeros((self.drones, 3))
        positions[:, 2] = self.altitude
        xs = (np.arange(self.cells_x) + 0.5) / self.cells_x * 2 * self.area - self.area
        ys = (np.arange(self.cells_y) + 0.5) / self.cells_y * 2 * self.area - self.area
        centers = np.array([(x, y, self.altitude) for x in xs for y in ys])
        return kinds, positions, centers


def average_idleness(duration: float, last_visit: np.ndarray, gap_squares: np.ndarray) -> float:
    """
    Time-averaged idleness over all cells. Idleness of a cell grows linearly since its last visit, so each interval
    between visits of length g contributes g^2 / 2.
    """
    final_gaps = duration - last_visit
    return float((gap_squares + final_gaps * final_gaps).sum() / (2 * duration * len(last_visit)))
