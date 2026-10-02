import unittest

from gradysim.simulator.event import EventLoop, EventLoopException


class TestEventLoop(unittest.TestCase):
    def test_length(self):
        ev_loop = EventLoop()
        ev_loop.schedule_event(1, lambda: 0)
        self.assertEqual(len(ev_loop), 1)
        ev_loop.schedule_event(1, lambda: 0)
        self.assertEqual(len(ev_loop), 2)
        ev_loop.pop_event()
        self.assertEqual(len(ev_loop), 1)
        ev_loop.pop_event()
        self.assertEqual(len(ev_loop), 0)

    def test_scheduling_popping(self):
        ev_loop = EventLoop()

        def callback():
            pass

        ev_loop.schedule_event(1, callback)
        event = ev_loop.pop_event()
        self.assertEqual(event.timestamp, 1)
        self.assertEqual(event.callback, callback)

    def test_invalid_scheduling(self):
        ev_loop = EventLoop()
        with self.assertRaises(EventLoopException):
            ev_loop.schedule_event(-1, lambda: 0)

    def test_invalid_popping(self):
        ev_loop = EventLoop()
        with self.assertRaises(EventLoopException):
            ev_loop.pop_event()

    def test_time_updating(self):
        ev_loop = EventLoop()
        ev_loop.schedule_event(10, lambda: 0)
        self.assertEqual(ev_loop.current_time, 0)
        ev_loop.pop_event()
        self.assertEqual(ev_loop.current_time, 10)

    def test_event_sorting(self):
        ev_loop = EventLoop()
        ev_loop.schedule_event(10, lambda: 0)
        ev_loop.schedule_event(1, lambda: 0)
        ev_loop.schedule_event(4, lambda: 0)
        ev_loop.schedule_event(5, lambda: 0)

        self.assertEqual(ev_loop.pop_event().timestamp, 1)
        self.assertEqual(ev_loop.pop_event().timestamp, 4)
        self.assertEqual(ev_loop.pop_event().timestamp, 5)
        self.assertEqual(ev_loop.pop_event().timestamp, 10)

    def test_same_timestamp_events_are_fifo(self):
        ev_loop = EventLoop()
        order = []
        for i in range(50):
            ev_loop.schedule_event(1, lambda i=i: order.append(i))
        while len(ev_loop) > 0:
            ev_loop.pop_event().callback()
        self.assertEqual(order, list(range(50)))

    def test_cancel_event(self):
        ev_loop = EventLoop()
        ev_loop.schedule_event(1, lambda: 0)
        cancelled = ev_loop.schedule_event(2, lambda: 0)
        ev_loop.schedule_event(3, lambda: 0)

        ev_loop.cancel_event(cancelled)
        ev_loop.cancel_event(cancelled)
        self.assertEqual(len(ev_loop), 2)
        self.assertEqual(ev_loop.pop_event().timestamp, 1)
        self.assertEqual(ev_loop.pop_event().timestamp, 3)
        self.assertEqual(len(ev_loop), 0)
        self.assertIsNone(ev_loop.peek_event())

    def test_cancel_popped_event_has_no_effect(self):
        ev_loop = EventLoop()
        ev_loop.schedule_event(1, lambda: 0)
        ev_loop.schedule_event(2, lambda: 0)
        event = ev_loop.pop_event()
        ev_loop.cancel_event(event)
        self.assertEqual(len(ev_loop), 1)

    def test_many_cancellations_compact_heap(self):
        ev_loop = EventLoop()
        events = [ev_loop.schedule_event(i, lambda: 0) for i in range(1000)]
        for event in events[:900]:
            ev_loop.cancel_event(event)
        self.assertEqual(len(ev_loop), 100)
        self.assertLess(len(ev_loop._event_heap), 1000)
        self.assertEqual(ev_loop.pop_event().timestamp, 900)
