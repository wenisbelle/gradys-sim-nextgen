"""Conformance protocol as a native protocol (C++). Same behavior as reference.py."""
from pathlib import Path

from gradysim.native import NativeConfiguration, NativeSimulation, build

from .scenario import MESSAGE_SIZE, TIMER_IDS, ConformanceScenario

SOURCE = Path(__file__).with_name("conformance.cpp")
KINDS = {0: "timer", 1: "packet", 2: "telemetry"}


def run(scenario: ConformanceScenario) -> dict:
    kinds, positions = scenario.layout()
    configuration = NativeConfiguration(
        duration=scenario.duration, transmission_range=scenario.transmission_range, delay=scenario.delay,
        update_rate=scenario.update_rate, telemetry_decimation=scenario.telemetry_decimation,
        message_size=MESSAGE_SIZE, timer_ids=TIMER_IDS, seed=scenario.seed)
    simulation = NativeSimulation(build(SOURCE), ["random_actor"], kinds, positions, configuration,
                                  params={"area": scenario.area})
    result = simulation.run()
    outputs = result["outputs"]
    trace = []
    for time, node, kind, a, b, c in zip(*(outputs[f"trace_{name}"].tolist()
                                            for name in ("time", "node", "kind", "a", "b", "c"))):
        kind = KINDS[int(kind)]
        if kind == "timer":
            trace.append((time, int(node), kind, int(a)))
        else:
            trace.append((time, int(node), kind, [a, b, c]))
    return {"trace": trace, "positions": result["positions"].tolist(), "events": result["events"]}
