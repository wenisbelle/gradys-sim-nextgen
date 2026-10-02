"""
Example: tuning a hyperparameter of the persistent monitoring strategy with the native (C++) mode.

The protocol is compiled once, before starting the workers, and every worker loads the cached module.
Run from the repository root:  python -m showcases.compiled.persistent_monitoring.sweep_cpp
"""
import statistics
import time

from gradysim.native import build
from gradysim.simulator.campaign import Campaign

from .cpp_version import SOURCE, run
from .scenario import PersistentMonitoringScenario

SEEDS = range(8)
DISTANCE_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def evaluate(parameters):
    distance_weight, seed = parameters
    scenario = PersistentMonitoringScenario(distance_weight=distance_weight, seed=seed)
    return run(scenario)["average_idleness"]


def main():
    build(SOURCE)   # compile once, workers reuse the cached module
    candidates = [(weight, seed) for weight in DISTANCE_WEIGHTS for seed in SEEDS]
    start = time.perf_counter()
    with Campaign() as campaign:
        idleness = campaign.map(evaluate, candidates)
    print(f"{len(candidates)} simulations in {time.perf_counter() - start:.1f} s")

    for index, weight in enumerate(DISTANCE_WEIGHTS):
        values = idleness[index * len(SEEDS):(index + 1) * len(SEEDS)]
        print(f"distance_weight={weight:5.2f}  average idleness {statistics.mean(values):6.2f} s "
              f"(std {statistics.stdev(values):.2f})")


if __name__ == "__main__":
    main()
