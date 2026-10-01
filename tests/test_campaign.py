import logging
import os
import random
import unittest

from gradysim.protocol.interface import IProtocol
from gradysim.simulator.campaign import Campaign, run_campaign
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.simulation import SimulationBuilder, SimulationConfiguration


class RandomTimerProtocol(IProtocol):
    draws = []

    def initialize(self):
        self.provider.schedule_timer("", random.random())

    def handle_timer(self, timer: str):
        RandomTimerProtocol.draws.append((self.provider.get_id(), self.provider.current_time()))

    def handle_packet(self, message: str):
        pass

    def handle_telemetry(self, telemetry):
        pass

    def finish(self):
        pass


def run_seeded(seed):
    RandomTimerProtocol.draws = []
    builder = SimulationBuilder(SimulationConfiguration(seed=seed, execution_logging=False))
    builder.add_handler(TimerHandler())
    for _ in range(5):
        builder.add_node(RandomTimerProtocol, (0, 0, 0))
    builder.build().start_simulation()
    return RandomTimerProtocol.draws


def square(value):
    return value * value


def worker_pid(_):
    return os.getpid()


def divide_by_zero(value):
    return value / 0


class TestCampaign(unittest.TestCase):
    def test_seed_makes_runs_reproducible(self):
        self.assertEqual(run_seeded(1), run_seeded(1))
        self.assertNotEqual(run_seeded(1), run_seeded(2))

    def test_results_keep_parameter_order(self):
        self.assertEqual(run_campaign(square, range(20), workers=2), [value * value for value in range(20)])

    def test_sequential_execution(self):
        self.assertEqual(run_campaign(square, [3, 4], workers=1), [9, 16])

    def test_parallel_runs_match_sequential_runs(self):
        seeds = [10, 11, 12]
        self.assertEqual(run_campaign(run_seeded, seeds, workers=3), [run_seeded(seed) for seed in seeds])

    def test_building_many_simulations_does_not_accumulate_log_handlers(self):
        logger = logging.getLogger()
        for _ in range(5):
            SimulationBuilder(SimulationConfiguration(execution_logging=False)).build()
        simulation_handlers = [handler for handler in logger.handlers
                               if getattr(handler, "_gradysim_simulation_handler", False)]
        self.assertEqual(len(simulation_handlers), 1)

    def test_campaign_reuses_workers_across_batches(self):
        with Campaign(workers=2) as campaign:
            first = campaign.map(worker_pid, range(4))
            second = campaign.map(worker_pid, range(4))
            self.assertEqual(campaign.map(square, range(5)), [0, 1, 4, 9, 16])
        self.assertLessEqual(len(set(first) | set(second)), 2)

    def test_campaign_submit(self):
        with Campaign(workers=2) as campaign:
            futures = [campaign.submit(square, value) for value in range(4)]
            self.assertEqual([future.result() for future in futures], [0, 1, 4, 9])

    def test_sequential_campaign_submit_reports_errors(self):
        with Campaign(workers=1) as campaign:
            self.assertEqual(campaign.submit(square, 3).result(), 9)
            with self.assertRaises(ZeroDivisionError):
                campaign.submit(divide_by_zero, 1).result()
