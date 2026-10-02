"""
Entry point of native (C++) simulations.
"""
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Sequence

import numpy as np


@dataclass
class NativeConfiguration:
    """
    Configuration of a native simulation. Mirrors the configuration of the Python simulator and of its
    communication, timer and massless mobility handlers.
    """
    duration: float
    """Simulation duration in seconds"""
    transmission_range: float = 60
    """Maximum distance in meters for message delivery"""
    delay: float = 0
    """Message delivery delay in seconds"""
    mobility: bool = True
    """Whether nodes can move. Telemetry is only delivered when enabled"""
    update_rate: float = 0.01
    """Interval in seconds between mobility updates"""
    telemetry_decimation: int = 1
    """Telemetry is delivered every `telemetry_decimation` mobility updates"""
    message_size: int = 4
    """Number of float64 values in a message payload"""
    timer_ids: int = 8
    """Timers are identified by integers in [0, timer_ids)"""
    seed: int = 0
    """Seed of the nodes' random streams"""


class NativeSimulation:
    """
    A simulation whose protocols are C++ classes compiled with [build][gradysim.native.build.build].

    Hyperparameters and input data reach the protocols as named numbers (`params`) and arrays (`arrays`), read in
    their constructors through `Setup`. Changing them doesn't require compiling again.
    """

    def __init__(self, module, protocols: Sequence[str], kinds, positions, configuration: NativeConfiguration,
                 params: Optional[Dict[str, float]] = None, arrays: Optional[Dict[str, np.ndarray]] = None):
        """
        Args:
            module: Module returned by `build`
            protocols: Names of the protocols, as registered in the module. Nodes of kind `k` run `protocols[k]`
            kinds: Kind of every node
            positions: Initial position of every node, array of shape (nodes, 3)
            configuration: Simulation configuration
            params: Named numbers available to the protocols, like hyperparameters
            arrays: Named float64 arrays available to the protocols, like waypoints
        """
        self.module = module
        self.protocols = list(protocols)
        self.kinds = np.asarray(kinds, dtype=np.int64)
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.configuration = configuration
        self.params = {name: float(value) for name, value in (params or {}).items()}
        self.arrays = {name: np.ascontiguousarray(value, dtype=np.float64) for name, value in (arrays or {}).items()}
        if self.kinds.shape[0] != self.positions.shape[0]:
            raise ValueError("kinds and positions must have one entry per node")
        if configuration.telemetry_decimation < 1:
            raise ValueError("telemetry_decimation must be at least 1")

    def run(self) -> dict:
        """
        Runs the simulation.

        Returns:
            Dictionary with the number of executed events (`events`), the final simulation time (`time`), the final
            positions of the nodes (`positions`) and the outputs written by the protocols (`outputs`, a dictionary
            of float64 arrays)
        """
        configuration = asdict(self.configuration)
        message_size = configuration.pop("message_size")
        result = self.module.simulate(self.protocols, self.kinds.tolist(), self.positions, configuration,
                                      self.params, self.arrays, message_size)
        return dict(result)
