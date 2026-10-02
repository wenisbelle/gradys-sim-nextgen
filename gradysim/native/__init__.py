"""
Native simulation mode: protocols written in C++ run together with a C++ simulation engine, with the same results as
the Python simulator and 20-30x faster. See COMPILED_MODES.md at the root of the repository.
"""
from .build import build
from .simulation import NativeConfiguration, NativeSimulation

__all__ = ["build", "NativeConfiguration", "NativeSimulation"]
