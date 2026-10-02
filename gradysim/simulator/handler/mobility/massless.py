import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from gradysim.protocol.messages.mobility import MobilityCommand, MobilityCommandType
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.position import Position, geo_to_cartesian
from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.interface import INodeHandler
from gradysim.simulator.node import VECTORIZATION_MIN_NODES, Node, PositionSnapshot, square

from .delivery import deliver_telemetry


class MasslessMobilityException(Exception):
    pass


@dataclass
class MasslessMobilityConfiguration:
    """
    Configuration class for the mobility handler
    """

    update_rate: float = 0.01
    """Interval in simulation seconds between mobility updates"""

    default_speed: float = 10
    """This is the default speed of a node in m/s"""

    reference_coordinates: Tuple[float, float, float] = (0, 0, 0)
    """
    These coordinates are used as a reference frame to convert geographical coordinates to cartesian coordinates. They
    will be used as the center of the scene and all geographical coordinates will be converted relative to it.
    """

    telemetry_decimation: int = 1
    """
    Telemetry is sent to the nodes every `telemetry_decimation` mobility updates. The default sends it at every update.
    Telemetry delivery is usually the most expensive part of mobility, increase this if your protocols don't need
    position updates at every `update_rate`.
    """


class MasslessMobilityHandler(INodeHandler):
    """
    Introduces mobility into the simulation. Simulates nodes as massless and dimensionless points that can move
    freely in 3D space. There is no concept of acceleration or inertia, all changes in velocity are instantaneous.

    Works by registering a regular event that updates every node's position based on its target and speed. A protocol
    is capable of altering its own speed and heading by sending mobility commands through its provider. These commands
    will reach the mobility handler which will update the node's target and speed accordingly. Nodes also receive
    telemetry updates containing information pertaining a node's current mobility status.
    """

    @staticmethod
    def get_label() -> str:
        return "mobility"

    _event_loop: EventLoop

    nodes: Dict[int, Node]
    targets: Dict[int, Position]
    speeds: Dict[int, float]

    def __init__(self, configuration: MasslessMobilityConfiguration = MasslessMobilityConfiguration()):
        """
        Constructor for the mobility handler

        Args:
            configuration: Configuration for the mobility handler. If not set all default values will be used.
        """
        if configuration.telemetry_decimation < 1:
            raise MasslessMobilityException("telemetry_decimation must be at least 1")

        self._configuration = configuration
        self.nodes = {}
        self.targets = {}
        self.speeds = {}
        self._injected = False
        self._update_count = 0

        # State mirrored in arrays for the vectorized update, built when first needed. Commands keep it in sync.
        self._node_list: List[Node] = []
        self._rows: Dict[int, int] = {}
        self._positions: Optional[np.ndarray] = None
        self._positions_version = -1
        self._targets_array: Optional[np.ndarray] = None
        self._has_target: Optional[np.ndarray] = None
        self._speeds_array: Optional[np.ndarray] = None

    def inject(self, event_loop: EventLoop):
        self._injected = True
        self._event_loop = event_loop

        event_loop.schedule_event(event_loop.current_time + self._configuration.update_rate,
                                  self._update_movement,
                                  "Mobility")

    def register_node(self, node: Node):
        if not self._injected:
            raise MasslessMobilityException("Error registering node: cannot register nodes while mobility handler "
                                    "is uninitialized.")

        self.nodes[node.id] = node
        self.speeds[node.id] = self._configuration.default_speed
        self._targets_array = None
        self._has_target = None
        self._speeds_array = None

    def _update_movement(self):
        update_rate = self._configuration.update_rate

        # Only nodes with a target move, so they decide whether vectorizing pays off
        if len(self.targets) >= VECTORIZATION_MIN_NODES:
            self._update_positions_vectorized(update_rate)
        elif self.targets:
            self._update_positions(update_rate)

        self._update_count += 1
        if self.nodes and self._update_count % self._configuration.telemetry_decimation == 0:
            deliveries = [(node, Telemetry(current_position=node.position)) for node in self.nodes.values()]

            # Telemetry for every node is delivered by a single event, in node order. Since events with the same
            # timestamp run in scheduling order, this is equivalent to one event per node, but much cheaper.
            self._event_loop.schedule_event(
                self._event_loop.current_time,
                lambda: deliver_telemetry(self._event_loop, deliveries),
                "Mobility telemetry"
            )

        self._event_loop.schedule_event(self._event_loop.current_time + update_rate,
                                        self._update_movement,
                                        "Mobility")

    def _update_positions(self, update_rate: float) -> None:
        for node_id, node in self.nodes.items():
            # If the node has a target update its position
            target = self.targets.get(node_id)
            if target is not None:
                current_position = node.position
                target_vector: Position = (target[0] - current_position[0],
                                           target[1] - current_position[1],
                                           target[2] - current_position[2])
                movement_multiplier = self.speeds[node_id] * update_rate
                distance_delta = math.sqrt(target_vector[0] ** 2 + target_vector[1] ** 2 + target_vector[2] ** 2)

                if movement_multiplier >= distance_delta:
                    node.position = (
                        target[0],
                        target[1],
                        target[2]
                    )
                else:
                    target_vector_multiplier = movement_multiplier / distance_delta

                    node.position = (
                        current_position[0] + target_vector[0] * target_vector_multiplier,
                        current_position[1] + target_vector[1] * target_vector_multiplier,
                        current_position[2] + target_vector[2] * target_vector_multiplier
                    )

    def _build_arrays(self) -> None:
        self._node_list = list(self.nodes.values())
        self._rows = {node.id: row for row, node in enumerate(self._node_list)}
        count = len(self._node_list)
        self._targets_array = np.zeros((count, 3))
        self._has_target = np.zeros(count, dtype=bool)
        for node_id, target in self.targets.items():
            self._targets_array[self._rows[node_id]] = target
            self._has_target[self._rows[node_id]] = True
        self._speeds_array = np.array([self.speeds[node.id] for node in self._node_list], dtype=float)
        self._positions_version = -1

    def _update_positions_vectorized(self, update_rate: float) -> None:
        """
        Same computation as `_update_positions`, performed on all nodes at once. Operations are done in the same
        order so results are identical.
        """
        if self._targets_array is None:
            self._build_arrays()
        nodes = self._node_list

        if self._positions_version != Node.position_version:
            self._positions = np.array([node.position for node in nodes], dtype=float).reshape(-1, 3)
        positions = self._positions
        targets = self._targets_array

        target_vectors = targets - positions
        movement_multiplier = self._speeds_array * update_rate
        distance_delta = np.sqrt(square(target_vectors[:, 0]) + square(target_vectors[:, 1])
                                 + square(target_vectors[:, 2]))

        arrived = self._has_target & (movement_multiplier >= distance_delta)
        moving = self._has_target & ~arrived
        moving_rows = np.flatnonzero(moving)

        new_positions = positions.copy()
        target_vector_multiplier = movement_multiplier[moving_rows] / distance_delta[moving_rows]
        new_positions[moving_rows] = positions[moving_rows] + target_vectors[moving_rows] * target_vector_multiplier[:, None]
        new_positions[arrived] = targets[arrived]

        for row, (x, y, z) in zip(moving_rows.tolist(), new_positions[moving_rows].tolist()):
            nodes[row].position = (x, y, z)
        for row in np.flatnonzero(arrived).tolist():
            node = nodes[row]
            target = self.targets[node.id]
            node.position = (target[0], target[1], target[2])

        self._positions = new_positions
        self._positions_version = Node.position_version
        Node.position_snapshot = PositionSnapshot(Node.position_version, nodes, new_positions)

    def handle_command(self, command: MobilityCommand, node: Node):
        """
        Performs a mobility command. This method is called by the node's 
        provider to transmit it's mobility command to the mobility handler.

        Args:
            command: Command being issued
            node: Node that issued the command
        """
        if node.id not in self.nodes:
            raise MasslessMobilityException("Error handling commands: Cannot handle command from unregistered node")

        if command.command_type == MobilityCommandType.GOTO_COORDS:
            self._goto((command.param_1, command.param_2, command.param_3), node)
        elif command.command_type == MobilityCommandType.GOTO_GEO_COORDS:
            relative_coords = geo_to_cartesian(self._configuration.reference_coordinates,
                                               (command.param_1, command.param_2, command.param_3))
            self._goto(relative_coords, node)
        elif command.command_type == MobilityCommandType.SET_SPEED:
            self.speeds[node.id] = command.param_1
            if self._speeds_array is not None:
                self._speeds_array[self._rows[node.id]] = command.param_1

    def _goto(self, position: Position, node: Node):
        self.targets[node.id] = position
        if self._targets_array is not None:
            row = self._rows[node.id]
            self._targets_array[row] = position
            self._has_target[row] = True

    def _stop(self, node: Node):
        del self.targets[node.id]
        if self._targets_array is not None:
            self._has_target[self._rows[node.id]] = False
