"""Conformance protocol written as a regular gradysim protocol. Ground truth for the compiled implementations."""
import json
import logging

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand, SendMessageCommand
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityConfiguration, MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration

from ..node_random import NodeRandom
from .scenario import TIMER_IDS, ConformanceScenario


class _Shared:
    scenario: ConformanceScenario
    random: NodeRandom
    trace: list


class RandomActor(IProtocol):
    def initialize(self):
        self.id = self.provider.get_id()
        self.sent = 0
        for timer in range(TIMER_IDS):
            self.provider.schedule_timer(str(timer), self.provider.current_time() + 2 * self.random())

    def random(self):
        return _Shared.random.random(self.id)

    def payload(self):
        self.sent += 1
        return [float(self.id), float(self.sent), self.random()]

    def handle_timer(self, timer):
        now = self.provider.current_time()
        timer = int(timer)
        _Shared.trace.append((now, self.id, "timer", timer))
        action = int(self.random() * 6)
        if action == 0:
            self.provider.send_communication_command(BroadcastMessageCommand(json.dumps(self.payload())))
        elif action == 1:
            destination = (self.id + 1 + int(self.random() * (_Shared.scenario.nodes - 1))) % _Shared.scenario.nodes
            self.provider.send_communication_command(SendMessageCommand(json.dumps(self.payload()), destination))
        elif action == 2:
            area = _Shared.scenario.area
            x, y, z = (self.random() * 2 - 1) * area, (self.random() * 2 - 1) * area, self.random() * 20
            self.provider.send_mobility_command(GotoCoordsMobilityCommand(x, y, z))
        elif action == 3:
            self.provider.send_mobility_command(SetSpeedMobilityCommand(1 + self.random() * 20))
        elif action == 4:
            self.provider.cancel_timer(str(int(self.random() * TIMER_IDS)))

        if self.random() < 0.85:
            self.provider.schedule_timer(str(timer), now + 2 * self.random())
        else:
            self.provider.schedule_timer(str((timer + 1) % TIMER_IDS), now + 2 * self.random())

    def handle_packet(self, message):
        payload = json.loads(message)
        sender = int(payload[0])
        _Shared.trace.append((self.provider.current_time(), self.id, "packet", payload))
        if self.random() < 0.3:
            self.provider.send_communication_command(SendMessageCommand(json.dumps(self.payload()), sender))

    def handle_telemetry(self, telemetry):
        _Shared.trace.append((self.provider.current_time(), self.id, "telemetry", list(telemetry.current_position)))

    def finish(self):
        pass


def run(scenario: ConformanceScenario) -> dict:
    kinds, positions = scenario.layout()
    _Shared.scenario = scenario
    _Shared.random = NodeRandom(scenario.seed, len(kinds))
    _Shared.trace = []

    logging.disable(logging.CRITICAL)
    builder = SimulationBuilder(SimulationConfiguration(duration=scenario.duration, execution_logging=False))
    builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=scenario.transmission_range,
                                                                 delay=scenario.delay)))
    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler(MobilityConfiguration(update_rate=scenario.update_rate,
                                                              telemetry_decimation=scenario.telemetry_decimation)))
    for position in positions.tolist():
        builder.add_node(RandomActor, tuple(position))
    simulation = builder.build()
    simulation.start_simulation()
    logging.disable(logging.NOTSET)

    nodes = [simulation.get_node(node) for node in range(len(kinds))]
    return {
        "trace": _Shared.trace,
        "positions": [list(node.position) for node in nodes],
        "events": simulation._iteration,
    }
