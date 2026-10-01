from collections import defaultdict
from typing import Set, Dict

from gradysim.simulator.event import EventLoop, Event
from gradysim.simulator.log import node_context
from gradysim.simulator.node import Node
from gradysim.simulator.handler.interface import INodeHandler


class TimerException(Exception):
    pass


class TimerHandler(INodeHandler):
    """
    Adds timers to the simulation. This allows nodes to set timers
    which is a very important feature when implementing distributed
    algorithms. Nodes can schedule timers through their providers.
    """
    _event_loop: EventLoop

    _timer_id: int
    _pending_timers: Dict[int, Dict[str, Dict[int, Event]]]
    """
    Timers pending for each node. A dict of nodes where the value is a dict of messages, each mapping the
    timer identifiers to their scheduled events.
    """

    def __init__(self):
        """
        Constructs a TimerHandler, no configuration is necessary.
        """
        self._registed_nodes: Set[Node] = set()
        self._timer_id = 0
        self._pending_timers = defaultdict(lambda : defaultdict(dict))

    def get_current_time(self):
        return self._event_loop.current_time

    @staticmethod
    def get_label() -> str:
        return "timer"

    def inject(self, event_loop: EventLoop) -> None:
        self._event_loop = event_loop

    def register_node(self, node: Node) -> None:
        self._registed_nodes.add(node)

    def fire_timer(self, message: str, node: Node, identifier: int):
        """
        Fires a timer. Should be called by the event loop.
        """
        node_timers = self._pending_timers[node.id]
        pending = node_timers.get(message)
        if pending is None or identifier not in pending:
            return

        # Removed before handling so the protocol may freely cancel or re-schedule this same timer
        del pending[identifier]
        if not pending:
            del node_timers[message]
        node.protocol_encapsulator.handle_timer(message)

    def set_timer(self, message: str, timestamp: float, node: Node):
        """
        Sets a timer. Should be called by the nodes' providers. Node needs to be
        registered and timer needs to be set in the future.
        """
        if node not in self._registed_nodes:
            raise TimerException(f"Could not set timer: Node {node.id} not registered")

        if timestamp < self._event_loop.current_time:
            raise TimerException("Could not set timer: Timer cannot be set in the past")

        identifier = self._timer_id
        event = self._event_loop.schedule_event(timestamp,
                                                lambda: self.fire_timer(message, node, identifier),
                                                node_context(node, "handle_timer"))
        self._pending_timers[node.id][message][identifier] = event
        self._timer_id += 1

    def cancel_timer(self, message: str, node: Node):
        """
        Cancels a timer. Should be called by the nodes' providers. Node needs to be
        registered.
        """
        if node not in self._registed_nodes:
            raise TimerException(f"Could not cancel timer: Node {node.id} not registered")

        pending = self._pending_timers[node.id].pop(message, {})
        for event in pending.values():
            self._event_loop.cancel_event(event)