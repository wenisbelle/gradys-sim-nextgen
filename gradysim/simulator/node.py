"""
This is a simple module containing declarations necessary to keep track of the nodes during the simulation
"""

from typing import ClassVar, Generic, List, NamedTuple, Optional, TypeVar

import numpy as np

from gradysim.protocol.interface import IProtocol
from gradysim.encapsulator.interface import IEncapsulator
from gradysim.protocol.position import Position

T = TypeVar("T", bound=IProtocol)


VECTORIZATION_MIN_NODES = 32
"""
Handlers switch from per-node Python code to vectorized numpy code when the simulation has at least this many nodes.
Below it, numpy's fixed per-call overhead makes vectorized code slower.
"""


def square(values: np.ndarray) -> np.ndarray:
    """
    Squares an array exactly like Python's `x ** 2` does for floats.

    Python computes `x ** 2` with the C library's `pow`, which differs from `x * x` (what numpy's `x ** 2` computes)
    in the last bit for a small fraction of values. Vectorized code reproducing scalar code must use this function so
    results are identical.
    """
    return np.float_power(values, 2.0)


class PositionSnapshot(NamedTuple):
    """
    Positions of a group of nodes stored in a contiguous array. Published by handlers that compute positions in
    bulk, like vectorized mobility handlers, so other handlers can reuse them without gathering every node's position.
    """
    version: int
    """Value of `Node.position_version` when the snapshot was taken. The snapshot is only valid while it's unchanged"""
    nodes: List["Node"]
    """Nodes in the snapshot. Row `i` of `positions` is the position of `nodes[i]`"""
    positions: np.ndarray
    """Array of shape (len(nodes), 3) with the nodes' positions. Must not be modified"""


class Node(Generic[T]):
    """
    Represents a node inside the python simulation. Holds the reference to the node's encapsulated protocol.
    This class is accessible to [handlers][gradysim.simulator.handler].
    """
    id: int
    """Node's unique identifier"""

    protocol_encapsulator: IEncapsulator[T]
    """Node's encapsulated protocol"""

    position_version: ClassVar[int] = 0
    """
    Counter incremented every time the position of any node changes. Handlers can compare it with a previously
    seen value to know whether cached position data is still valid.
    """

    position_snapshot: ClassVar[Optional[PositionSnapshot]] = None
    """
    Most recently published snapshot of node positions, see
    [PositionSnapshot][gradysim.simulator.node.PositionSnapshot]. Use `valid_position_snapshot` to read it.
    """

    _position: Position

    @staticmethod
    def valid_position_snapshot() -> Optional[PositionSnapshot]:
        """
        Returns the latest published position snapshot if no node moved since it was taken, None otherwise.
        """
        snapshot = Node.position_snapshot
        if snapshot is not None and snapshot.version == Node.position_version:
            return snapshot
        return None

    @property
    def position(self) -> Position:
        """Node's position inside the simulation"""
        return self._position

    @position.setter
    def position(self, value: Position) -> None:
        self._position = value
        Node.position_version += 1
