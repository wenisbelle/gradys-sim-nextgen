"""
As an event-based simulator one of the main components in the simulation is an event loop, that's the focus of this
module. Events are compact classes containing a timestamp and a callback. Events are inserted into the event loop which
is organized as a heap to keep the events with the smallest timestamps on top. At every simulation iteration the
simulator class grabs the event with the smallest timestamp and executes its callback.

Events scheduled for the same timestamp are executed in the order they were scheduled (FIFO), which makes simulations
deterministic and reproducible.

Events are created by handlers. Protocols indirectly interact with them through the provider interface they have access
to. These events, when executed, cause effects on the network nodes, mainly observed through calls to the protocol
interface methods like handle_timer.
"""

import heapq
from typing import Callable, List, Optional, Tuple


class Event:
    """
    Class representing a single event. Will be placed inside the simulation loop. Shouldn't be instantiated
    directly, but through the `EventLoop.schedule_event` method.
    """
    __slots__ = ("timestamp", "callback", "context", "pending")

    timestamp: float
    """Simulation time in seconds when the event will fire"""
    callback: Callable
    """Any callable with no parameters. Will be executed when the event fires"""
    context: str
    """
    Context in which the event will be executed. 
    This is used in logging to identify where the callback is being executed.
    """
    pending: bool
    """True while the event is waiting in the event loop, False once it was popped or cancelled"""

    def __init__(self, timestamp: float, callback: Callable, context: str = ""):
        self.timestamp = timestamp
        self.callback = callback
        self.context = context
        self.pending = True

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __repr__(self):
        return f"Event(timestamp={self.timestamp!r}, callback={self.callback!r}, context={self.context!r})"


class EventLoopException(Exception):
    pass


class EventLoop:
    """
    Event loop central to the event-based simulation. Is implemented as a min-heap populated by `Event` instances ordered
    by their timestamps, with ties broken by scheduling order. Generally only the
    [`Simulator`][gradysim.simulator.simulation.Simulator] will call the `pop_event` method, a handler should only need
    to use the `schedule_event` and `cancel_event` methods.
    """
    _event_heap: List[Tuple[float, int, Event]]
    _current_time: float

    def __init__(self):
        """
        Creates an event loop
        """
        self._event_heap = []
        self._current_time = 0
        self._sequence = 0
        self._cancelled = 0
        self._context_listener: Optional[Callable[[str], None]] = None

    def schedule_event(self, timestamp: float, callback: Callable, context: str = "") -> Event:
        """
        Creates an event instance with the information provided as args and inserts it into the event heap.

        Args:
            timestamp: Simulation time in seconds when the event should fire
            callback: Any callable with no arguments
            context: Context where the callable executes. Useful for logging.

        Returns:
            The scheduled event. Can be passed to `cancel_event`.
        """
        if timestamp < self._current_time:
            raise EventLoopException(f"Could not schedule event: tried to schedule at {timestamp} which is "
                                     f"earlier than the current time {self._current_time}. ")

        event = Event(timestamp, callback, context)
        heapq.heappush(self._event_heap, (timestamp, self._sequence, event))
        self._sequence += 1
        return event

    def cancel_event(self, event: Event) -> None:
        """
        Cancels a previously scheduled event so that it is never executed. Cancelling an event that was already
        executed or cancelled has no effect.

        Args:
            event: Event returned by `schedule_event`
        """
        if not event.pending:
            return
        event.pending = False
        self._cancelled += 1

        # Cancelled events are removed lazily. Compact the heap if they become the majority of it
        if self._cancelled > 64 and self._cancelled * 2 > len(self._event_heap):
            self._event_heap = [entry for entry in self._event_heap if entry[2].pending]
            heapq.heapify(self._event_heap)
            self._cancelled = 0

    def _discard_cancelled(self) -> None:
        heap = self._event_heap
        while heap and not heap[0][2].pending:
            heapq.heappop(heap)
            self._cancelled -= 1

    def pop_event(self) -> Event:
        """
        Removes an event from the top of the event heap and returns it. If called when the event heap is empty it will
        raise EventLoopException.

        Returns: The event popped.

        """
        if self._cancelled:
            self._discard_cancelled()

        if not self._event_heap:
            raise EventLoopException("Could not pop event: the event queue is empty")

        event = heapq.heappop(self._event_heap)[2]
        self._current_time = event.timestamp
        event.pending = False
        return event

    def peek_event(self) -> Optional[Event]:
        """
        Peeks at the event at the top of the event heap without removing it. Returns None if the event heap is empty.

        Returns:
            The event at the top of the event heap or None if it's empty
        """
        if self._cancelled:
            self._discard_cancelled()

        if not self._event_heap:
            return None
        return self._event_heap[0][2]

    def clear(self) -> None:
        """
        Clears the event heap
        """
        for entry in self._event_heap:
            entry[2].pending = False
        self._event_heap.clear()
        self._cancelled = 0

    def set_context_listener(self, listener: Optional[Callable[[str], None]]) -> None:
        """
        Registers a function that is notified when `scope_context` is called. Used by the simulator to keep
        logging context up to date.

        Args:
            listener: Function receiving the new context, or None to remove the current listener
        """
        self._context_listener = listener

    def scope_context(self, context: str) -> None:
        """
        Updates the logging context while an event is executing. Useful for events that perform work on behalf of
        several nodes, like batched deliveries, so that logs are still attributed to the right node.

        Args:
            context: New execution context
        """
        if self._context_listener is not None:
            self._context_listener(context)

    @property
    def has_context_listener(self) -> bool:
        """
        Whether anyone is listening to context changes. Lets callers skip building context strings when unused.
        """
        return self._context_listener is not None

    @property
    def current_time(self) -> float:
        """
        Returns the timestamp of the last event to be removed from the heap.
        This represents the current simulation time.

        Returns:
            The current simulation fime
        """
        return self._current_time

    def __len__(self) -> int:
        """
        By calling len() on the EventLoop you can check how many events are queued in the event heap.

        Returns:
            Number of events in the event heap
        """
        return len(self._event_heap) - self._cancelled
