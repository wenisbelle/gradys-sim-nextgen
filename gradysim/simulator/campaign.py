"""
Helpers to run simulation campaigns: many independent simulation runs, like parameter sweeps, Monte Carlo
repetitions or the evaluations of an optimization algorithm.

A single simulation is executed serially, but independent runs can be spread over every CPU core of the machine.
[run_campaign][gradysim.simulator.campaign.run_campaign] does that using separate processes, so runs don't share
any state and are not limited by Python's GIL.

Example:
    ```python
    from gradysim.simulator.campaign import run_campaign
    from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration

    def run(parameters):
        transmission_range, seed = parameters
        builder = SimulationBuilder(SimulationConfiguration(duration=100, seed=seed, execution_logging=False))
        ...  # Add handlers and nodes using transmission_range
        simulation = builder.build()
        simulation.start_simulation()
        return ...  # Any picklable result, like a metric collected by your protocols

    if __name__ == "__main__":
        parameters = [(transmission_range, seed) for transmission_range in (30, 60, 90) for seed in range(10)]
        results = run_campaign(run, parameters)
    ```

The function executed for each run must be defined at the top level of a module, so it can be sent to the worker
processes. Give each run its own seed through
[SimulationConfiguration.seed][gradysim.simulator.simulation.SimulationConfiguration.seed] to make them reproducible.
"""

import multiprocessing
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Iterable, List, Optional, TypeVar

import numpy as np

P = TypeVar("P")
R = TypeVar("R")


def _reseed_worker() -> None:
    # Worker processes may inherit the random state of their parent. Reseed them from OS entropy so that runs
    # that don't set an explicit seed are still independent from each other.
    random.seed()
    np.random.seed()


def run_campaign(task: Callable[[P], R],
                 parameters: Iterable[P],
                 workers: Optional[int] = None,
                 chunksize: int = 1,
                 mp_context: Optional[multiprocessing.context.BaseContext] = None) -> List[R]:
    """
    Executes `task(parameter)` for every parameter, distributing the runs over worker processes.

    Args:
        task: Function executing a single run. Must be picklable, which means defined at the top level of a module.
        parameters: Parameters of every run
        workers: Number of worker processes. Defaults to the number of CPUs of the machine. If set to 1, runs are
            executed sequentially in the current process, which is useful for debugging.
        chunksize: Number of runs sent to a worker at a time. Increasing it reduces overhead when runs are very short.
        mp_context: Multiprocessing context used to start the workers. Uses Python's default if not set.

    Returns:
        The results of every run, in the same order as `parameters`
    """
    parameters = list(parameters)

    if workers == 1:
        return [task(parameter) for parameter in parameters]

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context, initializer=_reseed_worker) as executor:
        return list(executor.map(task, parameters, chunksize=chunksize))
