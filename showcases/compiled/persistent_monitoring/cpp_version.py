"""Persistent monitoring as a native protocol (C++). Same behavior as reference.py."""
from pathlib import Path

from gradysim.native import NativeConfiguration, NativeSimulation, build

from .scenario import PersistentMonitoringScenario, average_idleness

SOURCE = Path(__file__).with_name("persistent_monitoring.cpp")


def run(scenario: PersistentMonitoringScenario) -> dict:
    kinds, positions, centers = scenario.layout()
    configuration = NativeConfiguration(
        duration=scenario.duration, transmission_range=scenario.transmission_range, delay=scenario.delay,
        update_rate=scenario.update_rate, telemetry_decimation=scenario.telemetry_decimation,
        message_size=scenario.cell_count, timer_ids=1, seed=scenario.seed)
    simulation = NativeSimulation(
        build(SOURCE), ["patroller"], kinds, positions, configuration,
        params={"distance_weight": scenario.distance_weight, "broadcast_period": scenario.broadcast_period,
                "speed": scenario.speed, "tolerance": scenario.tolerance},
        arrays={"centers": centers})
    result = simulation.run()
    outputs = result["outputs"]
    return {
        "average_idleness": average_idleness(scenario.duration, outputs["last_visit"], outputs["gap_squares"]),
        "last_visit": outputs["last_visit"].tolist(),
        "gap_squares": outputs["gap_squares"].tolist(),
        "visits": [int(value) for value in outputs["visits"]],
        "drone_maps": outputs["drone_maps"].reshape(len(kinds), scenario.cell_count).tolist(),
        "received": [int(value) for value in outputs["received"]],
        "positions": result["positions"].tolist(),
        "events": result["events"],
    }
