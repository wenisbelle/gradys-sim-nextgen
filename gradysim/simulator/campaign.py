"""
Helpers to run simulation campaigns: many independent simulation runs, like parameter sweeps, Monte Carlo
repetitions or the evaluations of an optimization algorithm.

A single simulation is executed serially, but independent runs can be spread over every CPU core of the machine.
[run_campaign][gradysim.simulator.campaign.run_campaign] does that using separate processes, so runs don't share
any state and are not limited by Python's GIL. Iterative workflows, like optimization algorithms that evaluate a batch
of candidates per iteration, should use a [Campaign][gradysim.simulator.campaign.Campaign] to keep the same worker
processes across iterations.

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
from concurrent.futures import Future, ProcessPoolExecutor
from typing import Callable, Iterable, List, Optional, TypeVar

import numpy as np

P = TypeVar("P")
R = TypeVar("R")


def _reseed_worker() -> None:
    # Worker processes may inherit the random state of their parent. Reseed them from OS entropy so that runs
    # that don't set an explicit seed are still independent from each other.
    random.seed()
    np.random.seed()


class Campaign:
    """
    Pool of worker processes that executes simulation runs. Create it once and reuse it for every batch of runs, for
    example every generation of an optimization algorithm: starting worker processes takes much longer than a short
    simulation, so creating a pool for each batch would waste most of the gains of parallelism.

    Use it as a context manager, or call `close` when done.

    Example:
        ```python
        with Campaign(workers=8) as campaign:
            for generation in range(100):
                candidates = optimizer.ask()
                fitness = campaign.map(evaluate, candidates)
                optimizer.tell(candidates, fitness)
        ```

    Optimizers that propose one candidate at a time can use `submit` instead, which returns a
    [Future][concurrent.futures.Future].
    """

    def __init__(self,
                 workers: Optional[int] = None,
                 mp_context: Optional[multiprocessing.context.BaseContext] = None):
        """
        Args:
            workers: Number of worker processes. Defaults to the number of CPUs of the machine. If set to 1, runs are
                executed sequentially in the current process, which is useful for debugging.
            mp_context: Multiprocessing context used to start the workers. Uses Python's default if not set.
        """
        self._executor: Optional[ProcessPoolExecutor] = None
        if workers != 1:
            self._executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp_context,
                                                 initializer=_reseed_worker)

    def map(self, task: Callable[[P], R], parameters: Iterable[P], chunksize: int = 1) -> List[R]:
        """
        Executes `task(parameter)` for every parameter in parallel and waits for all of them.

        Args:
            task: Function executing a single run. Must be picklable, which means defined at the top level of a module.
            parameters: Parameters of every run
            chunksize: Number of runs sent to a worker at a time. Increasing it reduces overhead when runs are very
                short.

        Returns:
            The results of every run, in the same order as `parameters`
        """
        if self._executor is None:
            return [task(parameter) for parameter in parameters]
        return list(self._executor.map(task, parameters, chunksize=chunksize))

    def submit(self, task: Callable[[P], R], parameter: P) -> "Future[R]":
        """
        Starts `task(parameter)` and returns immediately.

        Args:
            task: Function executing a single run. Must be picklable, which means defined at the top level of a module.
            parameter: Parameter of the run

        Returns:
            A future that will hold the result of the run
        """
        if self._executor is None:
            future: Future = Future()
            try:
                future.set_result(task(parameter))
            except Exception as exception:
                future.set_exception(exception)
            return future
        return self._executor.submit(task, parameter)

    def close(self) -> None:
        """
        Waits for pending runs and stops the worker processes.
        """
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def __enter__(self) -> "Campaign":
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()


def run_campaign(task: Callable[[P], R],
                 parameters: Iterable[P],
                 workers: Optional[int] = None,
                 chunksize: int = 1,
                 mp_context: Optional[multiprocessing.context.BaseContext] = None) -> List[R]:
    """
    Executes `task(parameter)` for every parameter, distributing the runs over worker processes. Convenient for a
    single batch of runs; use [Campaign][gradysim.simulator.campaign.Campaign] to execute several batches with the
    same worker processes.

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
    with Campaign(workers, mp_context) as campaign:
        return campaign.map(task, parameters, chunksize)
