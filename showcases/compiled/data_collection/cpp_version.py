"""Data collection as native protocols (C++). Same behavior as reference.py."""
from pathlib import Path

from gradysim.native import NativeConfiguration, NativeSimulation, build

from .scenario import DataCollectionScenario

SOURCE = Path(__file__).with_name("data_collection.cpp")


def run(scenario: DataCollectionScenario) -> dict:
    kinds, positions, missions = scenario.layout()
    configuration = NativeConfiguration(
        duration=scenario.duration, transmission_range=scenario.transmission_range, delay=scenario.delay,
        update_rate=scenario.update_rate, telemetry_decimation=scenario.telemetry_decimation, message_size=2,
        timer_ids=1, seed=scenario.seed)
    simulation = NativeSimulation(build(SOURCE), ["ground", "drone", "sensor"], kinds, positions, configuration,
                                  params={"speed": scenario.speed, "tolerance": scenario.tolerance},
                                  arrays={"missions": missions})
    result = simulation.run()
    return {
        "packets": [int(value) for value in result["outputs"]["packets"]],
        "received": [int(value) for value in result["outputs"]["received"]],
        "positions": result["positions"].tolist(),
        "events": result["events"],
    }
