import logging
import random
from dataclasses import dataclass

import numpy as np

from gradysim.simulator.event import EventLoop
from gradysim.simulator.log import node_context
from gradysim.protocol.messages.communication import CommunicationCommand, CommunicationCommandType
from gradysim.simulator.node import VECTORIZATION_MIN_NODES, Node, square
from gradysim.protocol.position import Position
from gradysim.simulator.handler.interface import INodeHandler

from typing import Dict, List, Optional, Tuple


class CommunicationDestination:
    """
    Represents the receiver for incoming communications for a specific node. Mostly
    used for logging.
    """
    node: Node

    def __init__(self, node: Node):
        """
        Creates a communication destination for a specific node. Doesn't need to be
        constructed directly, is used internally in the CommunicationHandler

        Args:
            node: Node owning the destination
        """
        self.node = node
        self._logger = logging.getLogger()

    def receive_message(self, message: str, source: 'CommunicationSource') -> None:
        """
        Function responsible for receiving the message through the communication handler.
        """
        self._logger.debug("Node %s received message from %s", self.node.id, source.node.id)

        self.node.protocol_encapsulator.handle_packet(message)


class CommunicationSource:
    """
    Represents the outwards facing communication interface of a node. Moslty used for logging. Doesn't need to be
    constructed directly, is used internally in the CommunicationHandler
    """
    node: Node

    def __init__(self, node: Node):
        """
        Creates a communication source for a specific node

        Args:
            node: Node owning the source
        """
        self.node = node
        self._logger = logging.getLogger()

    def hand_over_message(self, message: str, endpoint: CommunicationDestination) -> None:
        """
        Function called immediately before the communication handler sends a message. Doesn't deliver the actual
        message

        Args:
            message: Message being delivered
            endpoint: Destination of the message being delivered
        """
        self._logger.debug("Node %s sending message to %s", self.node.id, endpoint.node.id)


class CommunicationException(Exception):
    pass


@dataclass
class CommunicationMedium:
    """
    Conditions through which the messages are delivered. Can influence how and when messages can be delivered.
    """
    transmission_range: float = 60
    """Maximum range in meters for message delivery. Messages destined to nodes outside this range will not be delivered"""

    delay: float = 0
    """Sets a delay in seconds for message delivery, representing network delay. Range is evaluated before the delay is applied"""

    failure_rate: float = 0
    """Failure chance between 0 and 1 for message delivery. 0 represents messages never failing and 1 always fails."""


def can_transmit(source_position: Position, destination_position: Position,
                 medium: CommunicationMedium) -> bool:
    squared_distance = (destination_position[0] - source_position[0]) ** 2 + \
                       (destination_position[1] - source_position[1]) ** 2 + \
                       (destination_position[2] - source_position[2]) ** 2
    in_range = squared_distance <= medium.transmission_range ** 2

    rng = True
    if medium.failure_rate > 0:
        rng = random.random() > medium.failure_rate
    return rng and in_range


class CommunicationHandler(INodeHandler):
    """
    Adds communication to the simulation. Nodes, through their providers, can
    send this handler communication commands that dictate how a message should
    be sent. This message will be delivered to the destination node.

    Messages are transmited through a [medium][gradysim.simulator.handler.communication.CommunicationMedium] that
    determines conditions like communication range and failure rate. Messages can fail to be delivered or 
    be delivered late.
    """
    @staticmethod
    def get_label() -> str:
        return "communication"

    _sources: Dict[int, CommunicationSource]
    _destinations: Dict[int, CommunicationDestination]
    _event_loop: EventLoop

    def __init__(self, communication_medium: CommunicationMedium = CommunicationMedium()):
        """
        Initializes the communication handler.

        Args:
            communication_medium: Configuration of the network conditions. If not set all default values will be used.
        """
        self._injected = False

        self._sources: Dict[int, CommunicationSource] = {}
        self._destinations: Dict[int, CommunicationDestination] = {}

        # Position index used to find broadcast receivers with vectorized distance computations. Rebuilt lazily
        # whenever a node moves or a node is registered.
        self._index_endpoints: List[CommunicationDestination] = []
        self._index_positions: np.ndarray = np.empty((0, 3))
        self._index_version = -1
        self._index_candidate_version = -1
        self._snapshot_nodes: Optional[List[Node]] = None
        self._snapshot_endpoints: Optional[List[CommunicationDestination]] = None

        self.default_medium = communication_medium

    def inject(self, event_loop: EventLoop):
        self._injected = True
        self._event_loop = event_loop

    def register_node(self, node: Node):
        if not self._injected:
            raise CommunicationException("Error registering node: Cannot register node on uninitialized "
                                         "node handler")
        self._sources[node.id] = CommunicationSource(node)
        self._destinations[node.id] = CommunicationDestination(node)
        self._index_version = -1
        self._snapshot_nodes = None

    def handle_command(self,
                       command: CommunicationCommand,
                       sender: Node,
                       medium: CommunicationMedium = None):
        """
        Performs a communication command. This method should be called by the node's
        provider to transmit a communication command to the communication handler.

        Args:
            command: Command being issued
            sender: Node issuing the command
            medium: Optional communication medium to use for this specific command. If not set, the default medium
                    of the handler will be used.
        """
        if sender.id == command.destination:
            raise CommunicationException("Error transmitting message: message destination is equal to sender. Try "
                                         "using schedule_timer.")

        source = self._sources[sender.id]

        # Use the provided medium if available; otherwise fall back to the default handler medium
        if medium is None:
            medium = self.default_medium

        if command.command_type == CommunicationCommandType.BROADCAST:
            self._broadcast_message(command.message, source, medium)
        else:
            destination = command.destination
            if destination is None:
                raise CommunicationException("Error transmitting message: a destination is "
                                             "required when command type SEND is used.")
            if destination not in self._destinations:
                raise CommunicationException(f"Error transmitting message: destination {destination} does not exist.")

            self._transmit_message(command.message, source, self._destinations[destination], medium)

    def _in_range_endpoints(self, source: CommunicationSource, medium: CommunicationMedium) \
            -> List[CommunicationDestination]:
        """
        Returns every registered destination, except the source itself, within the medium's transmission range of
        the source. Destinations are returned in registration order.

        Small simulations are checked one destination at a time. Larger ones use a vectorized distance computation
        over an array of positions, preferably a snapshot published by the mobility handler.
        """
        if len(self._destinations) >= VECTORIZATION_MIN_NODES:
            index = self._position_index()
            if index is not None:
                endpoints, positions = index
                sx, sy, sz = source.node.position
                dx = positions[:, 0] - sx
                dy = positions[:, 1] - sy
                dz = positions[:, 2] - sz
                squared_range = medium.transmission_range ** 2
                squared_distance = dx * dx + dy * dy + dz * dz
                in_range = squared_distance <= squared_range

                # can_transmit squares with Python's `**`, which can differ from `x * x` in the last bit. Distances
                # that close to the range are recomputed exactly the same way, so both always agree.
                borderline = np.flatnonzero(np.abs(squared_distance - squared_range) <= squared_range * 1e-12)
                if borderline.size:
                    in_range[borderline] = (square(dx[borderline]) + square(dy[borderline])
                                            + square(dz[borderline])) <= squared_range

                source_node = source.node
                return [endpoints[row] for row in np.flatnonzero(in_range).tolist()
                        if endpoints[row].node is not source_node]

        source_node = source.node
        sx, sy, sz = source_node.position
        squared_range = medium.transmission_range ** 2
        endpoints = []
        for destination in self._destinations.values():
            x, y, z = destination.node.position
            if (x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2 <= squared_range and destination.node is not source_node:
                endpoints.append(destination)
        return endpoints

    def _position_index(self) -> Optional[Tuple[List[CommunicationDestination], np.ndarray]]:
        """
        Returns destinations and an array with their current positions, row by row, or None when building the
        array isn't worth it.
        """
        snapshot = Node.valid_position_snapshot()
        if snapshot is not None:
            if snapshot.nodes is not self._snapshot_nodes:
                self._snapshot_nodes = snapshot.nodes
                self._snapshot_endpoints = [self._destinations.get(node.id) for node in snapshot.nodes]
                if len(snapshot.nodes) != len(self._destinations) or any(
                        endpoint is None or endpoint.node is not node
                        for endpoint, node in zip(self._snapshot_endpoints, snapshot.nodes)):
                    self._snapshot_endpoints = None
            if self._snapshot_endpoints is not None:
                return self._snapshot_endpoints, snapshot.positions

        if self._index_version != Node.position_version:
            # Gathering positions costs about as much as checking every destination once, so the array is only
            # built when positions are used by more than one broadcast
            if self._index_candidate_version != Node.position_version:
                self._index_candidate_version = Node.position_version
                return None
            self._index_endpoints = list(self._destinations.values())
            self._index_positions = np.array([endpoint.node.position for endpoint in self._index_endpoints],
                                             dtype=float).reshape(-1, 3)
            self._index_version = Node.position_version
        return self._index_endpoints, self._index_positions

    def _broadcast_message(self, message: str, source: CommunicationSource, medium: CommunicationMedium):
        """
        Transmits a message from source to every destination in range through the communication medium.
        """
        failure_rate = medium.failure_rate
        for destination in self._in_range_endpoints(source, medium):
            source.hand_over_message(message, destination)
            if failure_rate > 0 and random.random() <= failure_rate:
                continue
            self._schedule_delivery(message, source, destination, medium)

    def _transmit_message(self, message: str, source: CommunicationSource, destination: CommunicationDestination,
                          medium: CommunicationMedium):
        """
        Transmits a message from source to destination through the communication medium.
        """
        source.hand_over_message(message, destination)

        if can_transmit(source.node.position, destination.node.position, medium):
            self._schedule_delivery(message, source, destination, medium)

    def _schedule_delivery(self, message: str, source: CommunicationSource, destination: CommunicationDestination,
                           medium: CommunicationMedium):
        delay = max(0, medium.delay)
        self._event_loop.schedule_event(
            self._event_loop.current_time + delay,
            lambda: destination.receive_message(message, source),
            node_context(destination.node, "handle_packet")
        )
