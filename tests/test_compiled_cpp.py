"""
The native (C++) mode must produce exactly the same results as the regular Python simulator. Every scenario is
implemented twice, as regular gradysim protocols (reference.py) and as C++ protocols (cpp_version.py), and their
complete outputs are compared: counters, per-node state, final positions at full precision, number of executed
events and, for the conformance protocol, the full trace of callbacks.
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip("pybind11")
if shutil.which(os.environ.get("CXX", "c++")) is None:
    pytest.skip("A C++ compiler is required", allow_module_level=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "showcases"))

from compiled.conformance import cpp_version as conformance_cpp  # noqa: E402
from compiled.conformance import reference as conformance_reference  # noqa: E402
from compiled.conformance.scenario import ConformanceScenario  # noqa: E402
from compiled.data_collection import cpp_version as collection_cpp  # noqa: E402
from compiled.data_collection import reference as collection_reference  # noqa: E402
from compiled.data_collection.scenario import DataCollectionScenario  # noqa: E402
from compiled.node_random import NodeRandom  # noqa: E402
from compiled.persistent_monitoring import cpp_version as monitoring_cpp  # noqa: E402
from compiled.persistent_monitoring import reference as monitoring_reference  # noqa: E402
from compiled.persistent_monitoring.scenario import PersistentMonitoringScenario  # noqa: E402

from gradysim.native import NativeConfiguration, NativeSimulation, build  # noqa: E402

ENGINE_CHECKS = Path(__file__).parent / "native" / "engine_checks.cpp"

CONFORMANCE_CASES = [
    dict(seed=0),
    dict(seed=1, delay=0.3),
    dict(seed=2, telemetry_decimation=4),
    dict(seed=3, nodes=3, transmission_range=40.0),
    dict(seed=4, nodes=40, duration=20.0),       # above the vectorization threshold of the Python simulator
    dict(seed=5, update_rate=0.013, delay=0.07, telemetry_decimation=3),
]

COLLECTION_CASES = [
    dict(seed=0, duration=120.0),
    dict(seed=1, duration=120.0, drones=3, sensors=5),
    dict(seed=2, duration=60.0, drones=30, sensors=40),
    dict(seed=3, duration=120.0, delay=0.05, telemetry_decimation=3, update_rate=0.02),
    dict(seed=4, duration=120.0, transmission_range=50.0, speed=8.0),
]

MONITORING_CASES = [
    dict(seed=0, duration=200.0),
    dict(seed=1, duration=200.0, drones=3, distance_weight=0.2),
    dict(seed=2, duration=100.0, drones=35, cells_x=8, cells_y=8),
    dict(seed=3, duration=200.0, broadcast_period=0.5, delay=0.1, telemetry_decimation=2),
    dict(seed=4, duration=200.0, transmission_range=20.0, distance_weight=5.0, speed=12.0),
]


class TestNativeMatchesPythonSimulator(unittest.TestCase):
    def assert_same(self, reference, result):
        self.assertEqual(reference.keys(), result.keys())
        for key in reference:
            self.assertEqual(reference[key], result[key], f"'{key}' differs")

    def test_conformance(self):
        for case in CONFORMANCE_CASES:
            with self.subTest(**case):
                reference = conformance_reference.run(ConformanceScenario(**case))
                self.assertGreater(len(reference["trace"]), 100)
                self.assert_same(reference, conformance_cpp.run(ConformanceScenario(**case)))

    def test_data_collection(self):
        for case in COLLECTION_CASES:
            with self.subTest(**case):
                reference = collection_reference.run(DataCollectionScenario(**case))
                self.assertGreater(reference["packets"][0], 0)
                self.assert_same(reference, collection_cpp.run(DataCollectionScenario(**case)))

    def test_persistent_monitoring(self):
        for case in MONITORING_CASES:
            with self.subTest(**case):
                reference = monitoring_reference.run(PersistentMonitoringScenario(**case))
                self.assertGreater(sum(reference["visits"]), 0)
                self.assert_same(reference, monitoring_cpp.run(PersistentMonitoringScenario(**case)))


def _run_checks(protocol, nodes=2, params=None):
    module = build(ENGINE_CHECKS)
    return NativeSimulation(module, [protocol], [0] * nodes, [(node, 0, 0) for node in range(nodes)],
                            NativeConfiguration(duration=1.0, mobility=False, seed=7), params=params).run()


class TestNativeEngine(unittest.TestCase):
    def test_random_streams_match_python_implementation(self):
        draws = _run_checks("record_draws", params={"count": 1000})["outputs"]["draws"].tolist()
        reference = NodeRandom(7, 2)
        self.assertEqual(draws, [reference.random(node) for node in range(2) for _ in range(1000)])

    def test_simultaneous_events_run_in_scheduling_order(self):
        order = _run_checks("simultaneous_timers")["outputs"]["order"].tolist()
        self.assertEqual(order, [3, 1, 2, 0, 13, 11, 12, 10])

    def test_sending_to_itself_raises(self):
        with self.assertRaises(ValueError):
            _run_checks("send_to_self")

    def test_timer_in_the_past_raises(self):
        with self.assertRaises(ValueError):
            _run_checks("timer_in_the_past")

    def test_unknown_protocol_raises(self):
        with self.assertRaises(ValueError):
            _run_checks("missing")

    def test_builds_are_cached_by_content(self):
        with tempfile.TemporaryDirectory() as directory:
            first = build(ENGINE_CHECKS, cache_directory=directory)
            libraries = list(Path(directory).iterdir())
            self.assertEqual(len(libraries), 1)
            self.assertIs(build(ENGINE_CHECKS, cache_directory=directory), first)

            changed = Path(directory) / "engine_checks.cpp"
            changed.write_text(ENGINE_CHECKS.read_text() + "\n// changed\n")
            self.assertIsNot(build(changed, cache_directory=directory), first)
            self.assertEqual(len([path for path in Path(directory).iterdir() if path.suffix != ".cpp"]), 2)

    def test_compilation_errors_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "broken.cpp"
            broken.write_text("#include <gradysim_native/module.hpp>\nthis is not C++\n")
            with self.assertRaises(RuntimeError):
                build(broken, cache_directory=directory)
