"""
Vectorized code paths, used for large simulations, must produce exactly the same results as the per-node code paths
used for small simulations.
"""
import random
import unittest

import numpy as np

from gradysim.protocol.messages.communication import BroadcastMessageCommand
from gradysim.protocol.messages.mobility import (
    GotoCoordsMobilityCommand,
    MobilityCommand,
    MobilityCommandType,
    SetSpeedMobilityCommand,
)
from gradysim.simulator.event import EventLoop
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium, can_transmit
from gradysim.simulator.handler.mobility import (
    DynamicVelocityMobilityConfiguration,
    DynamicVelocityMobilityHandler,
    MobilityConfiguration,
    MobilityHandler,
)
from gradysim.simulator.handler.mobility.dynamic_velocity import core
from gradysim.simulator.node import VECTORIZATION_MIN_NODES, Node


class RecordingEncapsulator:
    def __init__(self):
        self.telemetry = []
        self.packets = []

    def handle_telemetry(self, telemetry):
        self.telemetry.append(telemetry)

    def handle_packet(self, message):
        self.packets.append(message)


def create_nodes(count, rng):
    nodes = []
    for identifier in range(count):
        node = Node()
        node.id = identifier
        node.position = (rng.uniform(-100, 100), rng.uniform(-100, 100), rng.choice([0, 10, rng.uniform(0, 20)]))
        node.protocol_encapsulator = RecordingEncapsulator()
        nodes.append(node)
    return nodes


def random_vectors(rng, count):
    values = [rng.choice([0.0, 1.0, -2.5, rng.uniform(-20, 20)]) for _ in range(count * 3)]
    return np.array(values, dtype=float).reshape(count, 3)


class TestDynamicVelocityCore(unittest.TestCase):
    def test_vectorized_functions_match_scalar_functions_exactly(self):
        rng = random.Random(0)
        for _ in range(200):
            current = random_vectors(rng, 64)
            desired = random_vectors(rng, 64)
            dt = rng.choice([0.01, 0.05, 0.1])
            max_acc_xy, max_acc_z = rng.uniform(0, 10), rng.uniform(0, 10)
            tau_xy, tau_z = rng.choice([None, 0.5]), rng.choice([None, 0.8])

            vectorized = core.apply_acceleration_limits_vectorized(current, desired, dt, max_acc_xy, max_acc_z)
            scalar = [core.apply_acceleration_limits(tuple(c), tuple(d), dt, max_acc_xy, max_acc_z)
                      for c, d in zip(current.tolist(), desired.tolist())]
            self.assertEqual(repr(vectorized.tolist()), repr([list(v) for v in scalar]))

            vectorized = core.apply_velocity_tracking_first_order_vectorized(
                current, desired, dt, max_acc_xy, max_acc_z, tau_xy, tau_z)
            scalar = [core.apply_velocity_tracking_first_order(tuple(c), tuple(d), dt, max_acc_xy, max_acc_z,
                                                               tau_xy, tau_z)
                      for c, d in zip(current.tolist(), desired.tolist())]
            self.assertEqual(repr(vectorized.tolist()), repr([list(v) for v in scalar]))

            vectorized = core.apply_velocity_limits_vectorized(current, 10.0, 3.0)
            scalar = [core.apply_velocity_limits(tuple(c), 10.0, 3.0) for c in current.tolist()]
            self.assertEqual(repr(vectorized.tolist()), repr([list(v) for v in scalar]))


def run_massless(count, vectorized, steps=200):
    rng = random.Random(1)
    nodes = create_nodes(count, rng)
    handler = MobilityHandler(MobilityConfiguration(update_rate=0.1))
    handler.inject(EventLoop())
    for node in nodes:
        handler.register_node(node)

    trajectory = []
    for step in range(steps):
        for node in nodes:
            if rng.random() < 0.05:
                handler.handle_command(GotoCoordsMobilityCommand(rng.uniform(-100, 100), rng.choice([0, 50.5]), 10),
                                       node)
            if rng.random() < 0.02:
                handler.handle_command(SetSpeedMobilityCommand(rng.choice([1, 5.5, 20])), node)
        if vectorized:
            handler._update_positions_vectorized(0.1)
        else:
            handler._update_positions(0.1)
        trajectory.append([node.position for node in nodes])
    return trajectory


def run_dynamic_velocity(count, vectorized, tau, steps=200):
    rng = random.Random(2)
    nodes = create_nodes(count, rng)
    handler = DynamicVelocityMobilityHandler(DynamicVelocityMobilityConfiguration(
        update_rate=0.05, max_speed_xy=10, max_speed_z=3, max_acc_xy=4, max_acc_z=2,
        tau_xy=tau, tau_z=tau, send_telemetry=True, telemetry_decimation=3))
    handler.inject(EventLoop())
    for node in nodes:
        handler.register_node(node)

    trajectory = []
    for step in range(steps):
        for node in nodes:
            if rng.random() < 0.05:
                handler.handle_command(MobilityCommand(MobilityCommandType.SET_SPEED, rng.uniform(-15, 15),
                                                       rng.uniform(-15, 15), rng.uniform(-4, 4)), node)
        if vectorized:
            deliveries = handler._mobility_update_vectorized(0.05)
        else:
            deliveries = handler._mobility_update_nodes(0.05)
        trajectory.append(([node.position for node in nodes],
                           [(node.id, telemetry.current_position, telemetry.current_velocity)
                            for node, telemetry in deliveries]))
    return trajectory


class TestVectorizedMobility(unittest.TestCase):
    def test_massless_vectorized_matches_per_node_update(self):
        # repr also distinguishes ints from floats and the sign of zero
        self.assertEqual(repr(run_massless(50, vectorized=True)), repr(run_massless(50, vectorized=False)))

    def test_dynamic_velocity_vectorized_matches_per_node_update(self):
        for tau in (None, 0.5):
            with self.subTest(tau=tau):
                self.assertEqual(repr(run_dynamic_velocity(50, vectorized=True, tau=tau)),
                                 repr(run_dynamic_velocity(50, vectorized=False, tau=tau)))

    def test_massless_telemetry_decimation(self):
        loop = EventLoop()
        handler = MobilityHandler(MobilityConfiguration(update_rate=0.1, telemetry_decimation=3))
        handler.inject(loop)
        node = create_nodes(1, random.Random(0))[0]
        handler.register_node(node)

        while loop.current_time < 1.95:
            loop.pop_event().callback()
        # 19 updates happened, telemetry is delivered on updates 3, 6, ..., 18
        self.assertEqual(len(node.protocol_encapsulator.telemetry), 6)


class TestVectorizedBroadcast(unittest.TestCase):
    def broadcast_receivers(self, count, publish_snapshot):
        rng = random.Random(3)
        nodes = create_nodes(count, rng)
        loop = EventLoop()
        handler = CommunicationHandler(CommunicationMedium(transmission_range=60))
        handler.inject(loop)
        for node in nodes:
            handler.register_node(node)
        if publish_snapshot:
            mobility = MobilityHandler()
            mobility.inject(EventLoop())
            for node in nodes:
                mobility.register_node(node)
            mobility._update_positions_vectorized(0.01)

        for sender in nodes:
            handler.handle_command(BroadcastMessageCommand(f"from {sender.id}"), sender)
            handler.handle_command(BroadcastMessageCommand(f"again from {sender.id}"), sender)
        while len(loop) > 0:
            loop.pop_event().callback()
        return [node.protocol_encapsulator.packets for node in nodes]

    def expected_receivers(self, count):
        nodes = create_nodes(count, random.Random(3))
        received = [[] for _ in nodes]
        for sender in nodes:
            for prefix in ("from", "again from"):
                for node in nodes:
                    distance = sum((a - b) ** 2 for a, b in zip(node.position, sender.position))
                    if node is not sender and distance <= 60 ** 2:
                        received[node.id].append(f"{prefix} {sender.id}")
        return received

    def test_vectorized_broadcast_matches_per_node_check(self):
        count = VECTORIZATION_MIN_NODES * 3
        expected = self.expected_receivers(count)
        self.assertGreater(sum(len(messages) for messages in expected), count)
        for publish_snapshot in (False, True):
            with self.subTest(publish_snapshot=publish_snapshot):
                self.assertEqual(self.broadcast_receivers(count, publish_snapshot), expected)

    def test_vectorized_broadcast_matches_can_transmit_on_range_boundary(self):
        # Distances where `x * x` rounds above Python's `x ** 2`: exactly at the transmission range for
        # can_transmit, but just outside of it if squared naively
        rng = random.Random(4)
        boundary_values = []
        while len(boundary_values) < 5:
            value = rng.uniform(10, 100)
            if value * value > value ** 2:
                boundary_values.append(value)

        for transmission_range in boundary_values:
            loop = EventLoop()
            handler = CommunicationHandler(CommunicationMedium(transmission_range=transmission_range))
            handler.inject(loop)
            nodes = create_nodes(VECTORIZATION_MIN_NODES * 2, rng)
            nodes[0].position = (0.0, 0.0, 0.0)
            nodes[1].position = (transmission_range, 0.0, 0.0)
            nodes[2].position = (0.0, -transmission_range, 0.0)
            for node in nodes:
                handler.register_node(node)

            # Twice, so the second broadcast uses the vectorized position index
            for _ in range(2):
                handler.handle_command(BroadcastMessageCommand("boundary"), nodes[0])
            while len(loop) > 0:
                loop.pop_event().callback()

            for node in nodes[1:]:
                expected = 2 if can_transmit(nodes[0].position, node.position, handler.default_medium) else 0
                self.assertEqual(len(node.protocol_encapsulator.packets), expected)
            self.assertEqual(len(nodes[1].protocol_encapsulator.packets), 2)
