"""
Logging is an important part of any software. It helps users and developers understand what is happening during the
execution of the program. When running GrADyS-SIM NextGen in prototype-mode logging is automatically configured for you.

The logger annotates the output with timing information and execution context to improve understanding.
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

from gradysim.simulator.node import Node

_SIMULATION_HANDLER_MARKER = "_gradysim_simulation_handler"


class SimulationFormatter(logging.Formatter):
    """
    Custom logging formatter responsible for annotating the simulation logs with useful information about
    timing and execution context.

    The annotation is only built when a log record is actually formatted, so updating the current scope at every
    simulation step is cheap.
    """
    def __init__(self):
        super().__init__("%(message)s")
        self._prefix = ""
        self._iteration: Optional[int] = None
        self._timestamp = 0.0
        self._context = ""

    @property
    def prefix(self) -> str:
        if self._iteration is None:
            return self._prefix
        return f"[it={self._iteration} time={timedelta(seconds=self._timestamp)} | {self._context}] "

    @prefix.setter
    def prefix(self, value: str) -> None:
        self._prefix = value
        self._iteration = None

    def scope(self, iteration: int, timestamp: float, context: str) -> None:
        """
        Updates the current simulation scope used to annotate logs

        Args:
            iteration: Current simulation iteration
            timestamp: Current simulation timestamp in seconds
            context: Context of what's being currently executed in the simulation
        """
        self._iteration = iteration
        self._timestamp = timestamp
        self._context = context

    def scope_context(self, context: str) -> None:
        """
        Updates only the context of the current simulation scope
        """
        self._context = context

    def clear_iteration(self):
        self.prefix = ""

    def format(self, record: logging.LogRecord) -> str:
        log = super().format(record)
        return f"{record.levelname: <8} {self.prefix}{log}"


def setup_simulation_formatter(debug: bool, log_file: Optional[Path]) -> SimulationFormatter:
    """
    Sets up the logger for the simulation. Called before the simulation starts to configure
    the logger. Logging handlers installed by previous simulations are replaced, so creating many
    simulations in the same process doesn't accumulate handlers.

    Args:
        debug: Include DEBUG level logs
        log_file: Configure a logging handler to save logs in a file. Optional.

    Returns:
        The formatter instance. Is returned because it needs to be updated with current simulation information.
    """
    logger = logging.getLogger()

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        if getattr(handler, _SIMULATION_HANDLER_MARKER, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = SimulationFormatter()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    setattr(console_handler, _SIMULATION_HANDLER_MARKER, True)
    logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _SIMULATION_HANDLER_MARKER, True)
        logger.addHandler(file_handler)

    return formatter


def label_node(node: Node) -> str:
    """
    Human-readable label of a node, composed of its protocol name and identifier. The label is cached in the
    node once its protocol is known.
    """
    label = getattr(node, "_label", None)
    if label is not None:
        return label
    try:
        protocol_type_name = node.protocol_encapsulator.protocol.__class__.__name__
    except AttributeError:
        return f"Node {node.id}"
    label = f"{protocol_type_name} {node.id}"
    node._label = label
    return label


def node_context(node: Node, action: str) -> str:
    """
    Execution context of an action performed on a node, used to annotate logs
    """
    return label_node(node) + " " + action
