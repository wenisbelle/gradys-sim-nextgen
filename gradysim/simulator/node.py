"""
This is a simple module containing declarations necessary to keep track of the nodes during the simulation
"""

from typing import ClassVar, Generic, TypeVar

from gradysim.protocol.interface import IProtocol
from gradysim.encapsulator.interface import IEncapsulator
from gradysim.protocol.position import Position

T = TypeVar("T", bound=IProtocol)


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

    _position: Position

    @property
    def position(self) -> Position:
        """Node's position inside the simulation"""
        return self._position

    @position.setter
    def position(self, value: Position) -> None:
        self._position = value
        Node.position_version += 1
