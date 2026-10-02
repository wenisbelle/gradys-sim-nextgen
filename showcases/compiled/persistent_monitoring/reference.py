"""Persistent monitoring written as regular gradysim protocols. Ground truth for the compiled implementations."""
import json
import logging
import math

import numpy as np

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityConfiguration, MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration

from ..node_random import NodeRandom
from .scenario import PersistentMonitoringScenario, average_idleness


class _Shared:
    scenario: PersistentMonitoringScenario
    centers: list
    random: NodeRandom
    # Ground truth of the visits, used to measure idleness
    last_visit: list
    gap_squares: list
    visits: list


class Patroller(IProtocol):
    def initialize(self):
        self.id = self.provider.get_id()
        scenario = _Shared.scenario
        self.last_visit = [0.0] * scenario.cell_count
        self.received = 0
        self.target = int(_Shared.random.random(self.id) * scenario.cell_count)
        self.provider.send_mobility_command(GotoCoordsMobilityCommand(*_Shared.centers[self.target]))
        self.provider.send_mobility_command(SetSpeedMobilityCommand(scenario.speed))
        self.provider.schedule_timer(
            "share", self.provider.current_time() + scenario.broadcast_period * _Shared.random.random(self.id))

    def choose_next_cell(self, now, x, y, z):
        best, best_score = -1, -math.inf
        for cell, (cx, cy, cz) in enumerate(_Shared.centers):
            if cell == self.target:
                continue
            dx, dy, dz = cx - x, cy - y, cz - z
            distance = math.sqrt(dx * dx + dy * dy + dz * dz)
            score = (now - self.last_visit[cell]) - _Shared.scenario.distance_weight * distance
            if score > best_score:
                best, best_score = cell, score
        return best

    def handle_telemetry(self, telemetry):
        x, y, z = telemetry.current_position
        cx, cy, cz = _Shared.centers[self.target]
        dx, dy, dz = cx - x, cy - y, cz - z
        tolerance = _Shared.scenario.tolerance
        if dx * dx + dy * dy + dz * dz <= tolerance * tolerance:
            now = self.provider.current_time()
            cell = self.target
            gap = now - _Shared.last_visit[cell]
            _Shared.gap_squares[cell] += gap * gap
            _Shared.last_visit[cell] = now
            _Shared.visits[cell] += 1
            self.last_visit[cell] = now
            self.target = self.choose_next_cell(now, x, y, z)
            self.provider.send_mobility_command(GotoCoordsMobilityCommand(*_Shared.centers[self.target]))

    def handle_timer(self, timer):
        self.provider.send_communication_command(BroadcastMessageCommand(json.dumps(self.last_visit)))
        self.provider.schedule_timer("share", self.provider.current_time() + _Shared.scenario.broadcast_period)

    def handle_packet(self, message):
        self.received += 1
        for cell, visited in enumerate(json.loads(message)):
            if visited > self.last_visit[cell]:
                self.last_visit[cell] = visited

    def finish(self):
        pass


def run(scenario: PersistentMonitoringScenario) -> dict:
    kinds, positions, centers = scenario.layout()
    _Shared.scenario = scenario
    _Shared.centers = [tuple(center) for center in centers.tolist()]
    _Shared.random = NodeRandom(scenario.seed, len(kinds))
    _Shared.last_visit = [0.0] * scenario.cell_count
    _Shared.gap_squares = [0.0] * scenario.cell_count
    _Shared.visits = [0] * scenario.cell_count

    logging.disable(logging.CRITICAL)
    builder = SimulationBuilder(SimulationConfiguration(duration=scenario.duration, execution_logging=False))
    builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=scenario.transmission_range,
                                                                 delay=scenario.delay)))
    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler(MobilityConfiguration(update_rate=scenario.update_rate,
                                                              telemetry_decimation=scenario.telemetry_decimation)))
    for position in positions.tolist():
        builder.add_node(Patroller, tuple(position))
    simulation = builder.build()
    simulation.start_simulation()
    logging.disable(logging.NOTSET)

    nodes = [simulation.get_node(node) for node in range(len(kinds))]
    protocols = [node.protocol_encapsulator.protocol for node in nodes]
    last_visit, gap_squares = np.array(_Shared.last_visit), np.array(_Shared.gap_squares)
    return {
        "average_idleness": average_idleness(scenario.duration, last_visit, gap_squares),
        "last_visit": _Shared.last_visit,
        "gap_squares": _Shared.gap_squares,
        "visits": _Shared.visits,
        "drone_maps": [protocol.last_visit for protocol in protocols],
        "received": [protocol.received for protocol in protocols],
        "positions": [list(node.position) for node in nodes],
        "events": simulation._iteration,
    }
