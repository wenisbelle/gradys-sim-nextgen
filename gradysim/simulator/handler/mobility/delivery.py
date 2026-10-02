"""
Helpers shared by mobility handlers to deliver telemetry to protocols.
"""

from typing import List, Tuple

from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.simulator.event import EventLoop
from gradysim.simulator.log import node_context
from gradysim.simulator.node import Node


def deliver_telemetry(event_loop: EventLoop, deliveries: List[Tuple[Node, Telemetry]]) -> None:
    """
    Delivers a batch of telemetry messages to their nodes' protocols, in order.

    Args:
        event_loop: Event loop of the simulation, used to keep the logging context up to date
        deliveries: Pairs of node and the telemetry it should receive
    """
    if event_loop.has_context_listener:
        for node, telemetry in deliveries:
            event_loop.scope_context(node_context(node, "handle_telemetry"))
            node.protocol_encapsulator.handle_telemetry(telemetry)
    else:
        for node, telemetry in deliveries:
            node.protocol_encapsulator.handle_telemetry(telemetry)
