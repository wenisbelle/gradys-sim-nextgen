"""
Per-node random number generator shared by every implementation of the compiled-mode examples.

Each node owns an independent SplitMix64 stream seeded from the simulation seed and the node id. The algorithm is
simple enough to be reproduced bit for bit in plain Python, Numba and C++, so an example produces exactly the same
results whichever way it is executed.
"""

MASK = (1 << 64) - 1
GOLDEN_GAMMA = 0x9E3779B97F4A7C15


def initial_state(seed: int, node: int) -> int:
    """State of a node's stream before its first draw"""
    return (seed * 0x100000001B3 + node * GOLDEN_GAMMA) & MASK


class NodeRandom:
    """Independent uniform random streams, one per node"""

    def __init__(self, seed: int, node_count: int):
        self._states = [initial_state(seed, node) for node in range(node_count)]

    def random(self, node: int) -> float:
        """Next uniform number in [0, 1) of the node's stream"""
        state = (self._states[node] + GOLDEN_GAMMA) & MASK
        self._states[node] = state
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        z = z ^ (z >> 31)
        return (z >> 11) * (1.0 / 9007199254740992.0)
