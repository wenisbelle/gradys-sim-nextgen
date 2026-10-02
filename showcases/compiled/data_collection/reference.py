"""Data collection written as regular gradysim protocols. Ground truth for the compiled implementations."""
import json
import logging

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand, SetSpeedMobilityCommand
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium
from gradysim.simulator.handler.mobility import MobilityConfiguration, MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration

from ..node_random import NodeRandom
from .scenario import DRONE, GROUND, SENSOR, DataCollectionScenario


class _Shared:
    scenario: DataCollectionScenario
    missions = None
    random: NodeRandom


class _Base(IProtocol):
    def initialize(self):
        self.id = self.provider.get_id()
        self.packets = 0
        self.received = 0
        self.setup()

    def setup(self):
        pass

    def broadcast(self, kind, content):
        self.provider.send_communication_command(BroadcastMessageCommand(json.dumps([kind, content])))

    def handle_timer(self, timer):
        pass

    def handle_telemetry(self, telemetry):
        pass

    def finish(self):
        pass


class Ground(_Base):
    def handle_packet(self, message):
        self.received += 1
        sender, content = json.loads(message)
        if sender == DRONE:
            self.packets += content
            self.broadcast(GROUND, self.packets)


class Drone(_Base):
    def setup(self):
        self.mission = _Shared.missions[self.id - 1]
        self.waypoint = 0
        self.provider.send_mobility_command(GotoCoordsMobilityCommand(*self.mission[0]))
        self.provider.send_mobility_command(SetSpeedMobilityCommand(_Shared.scenario.speed))
        self.provider.schedule_timer("ping", self.provider.current_time() + _Shared.random.random(self.id))

    def handle_timer(self, timer):
        self.broadcast(DRONE, self.packets)
        self.provider.schedule_timer("ping", self.provider.current_time() + _Shared.random.random(self.id))

    def handle_packet(self, message):
        self.received += 1
        sender, content = json.loads(message)
        if sender == GROUND:
            self.packets = 0
        elif sender == SENSOR:
            self.packets += content

    def handle_telemetry(self, telemetry):
        x, y, z = telemetry.current_position
        tx, ty, tz = self.mission[self.waypoint]
        dx, dy, dz = tx - x, ty - y, tz - z
        tolerance = _Shared.scenario.tolerance
        if dx * dx + dy * dy + dz * dz <= tolerance * tolerance:
            self.waypoint = (self.waypoint + 1) % len(self.mission)
            self.provider.send_mobility_command(GotoCoordsMobilityCommand(*self.mission[self.waypoint]))


class Sensor(_Base):
    def setup(self):
        self.packets = 5
        self.provider.schedule_timer("generate", self.provider.current_time() + _Shared.random.random(self.id))

    def handle_timer(self, timer):
        self.packets += 1
        self.provider.schedule_timer("generate", self.provider.current_time() + _Shared.random.random(self.id))

    def handle_packet(self, message):
        self.received += 1
        sender, content = json.loads(message)
        if sender == DRONE:
            self.broadcast(SENSOR, self.packets)
            self.packets = 0


def run(scenario: DataCollectionScenario) -> dict:
    kinds, positions, missions = scenario.layout()
    _Shared.scenario = scenario
    _Shared.missions = [[tuple(waypoint) for waypoint in mission] for mission in missions.tolist()]
    _Shared.random = NodeRandom(scenario.seed, len(kinds))

    logging.disable(logging.CRITICAL)
    builder = SimulationBuilder(SimulationConfiguration(duration=scenario.duration, execution_logging=False))
    builder.add_handler(CommunicationHandler(CommunicationMedium(transmission_range=scenario.transmission_range,
                                                                 delay=scenario.delay)))
    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler(MobilityConfiguration(update_rate=scenario.update_rate,
                                                              telemetry_decimation=scenario.telemetry_decimation)))
    classes = {GROUND: Ground, DRONE: Drone, SENSOR: Sensor}
    for kind, position in zip(kinds.tolist(), positions.tolist()):
        builder.add_node(classes[kind], tuple(position))
    simulation = builder.build()
    simulation.start_simulation()
    logging.disable(logging.NOTSET)

    nodes = [simulation.get_node(node) for node in range(len(kinds))]
    protocols = [node.protocol_encapsulator.protocol for node in nodes]
    return {
        "packets": [protocol.packets for protocol in protocols],
        "received": [protocol.received for protocol in protocols],
        "positions": [list(node.position) for node in nodes],
        "events": simulation._iteration,
    }
