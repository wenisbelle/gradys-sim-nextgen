# Compiled simulation modes

GrADyS-SIM NextGen simulates protocols written in Python. That's ideal to develop and debug them, but every event
of a simulation (each packet, timer and telemetry update) goes through several layers of Python code. When a
simulation is evaluated thousands of times, for example by an optimization algorithm, that cost dominates.

Compiled modes run **the simulation engine and the protocols together as native code**, with **exactly the same
results** as the Python simulator. They come in two flavors, each in its own branch:

| | Numba mode | C++ mode |
|---|---|---|
| Branch | `compiled-numba` | `compiled-cpp` |
| Package | `gradysim.compiled` | `gradysim.native` |
| Protocols written in | Python functions (the subset Numba compiles) | C++ classes |
| Speed vs the Python simulator | **12-15x** | **35-45x** |
| Compilation | ~5-12 s per protocol set, in every new process | ~6 s per protocol source, cached on disk for every process |
| Debugging | Same code also runs on the Python simulator (`run_python`), with debuggers, prints, visualization | C++ tools (gdb, sanitizers); no Python mode |
| Requirements | `pip install gradysim[compiled]` (numba) | `pip install gradysim[native]` (pybind11) and a C++17 compiler |
| Results | Identical to the Python simulator | Identical to the Python simulator |

Measured on the persistent monitoring and data collection scenarios of `showcases/compiled` (10 drones, 300-600 s
simulated, one core of an i5-13420H). The Python simulator takes about 1 s per run, Numba 0.06-0.09 s, C++
0.02-0.03 s. Combined with [Campaign](gradysim/simulator/campaign.py) to use every core, this is the difference
between a few and a few hundred evaluations per second.

## Why compile the protocols and not only the engine

In the Python simulator, a run of the data collection scenario executes about 122 thousand events in ~1 s: roughly
9 µs per event. Most of that time is spent entering and leaving Python: calling protocol methods, creating message
and telemetry objects, running plugin chains. In compiled modes the same event costs 0.2-0.5 µs.

If only the engine were compiled and protocols stayed in Python, every packet, timer and telemetry delivery would
still call into Python, so most of the cost would remain (we estimate ~1.5x overall). The speedup appears only when
**no Python code runs between events**, which is why protocols are compiled together with the engine.

### What is compiled and what is configuration

Compilation depends on the **code**, not on the values it's given:

| Fixed when compiled (code) | Free on every run (configuration) |
|---|---|
| How a drone chooses its next waypoint | Number of drones, positions, waypoints, grid size |
| What a node does when it receives a message | Transmission range, delay, speeds, timer periods |
| Which messages are exchanged | Hyperparameters like weights and thresholds |
| | Seeds, simulation duration, update rate, telemetry rate |

Decisions taken during the simulation (like each drone choosing where to go from an idleness map it shares with its
neighbors) are part of the code and run normally: compiling the decision rule doesn't fix the decisions. A new
compilation is only needed when a protocol's code changes, or, in Numba mode, when a value of a different type is
given (an `int` where a `float` was expected).

## How it works

Both modes implement the same engine, which reproduces the Python simulator with its `CommunicationHandler`,
`TimerHandler` and `MasslessMobilityHandler`:

```
            ┌──────────────── compiled together ────────────────┐
 Python ──► │ event queue ──► engine ──► protocol handlers       │ ──► results (state / outputs)
 config,    │ (time, then      ├ communication   initialize      │
 params,    │  scheduling      ├ timers          handle_timer    │
 arrays     │  order)          ├ mobility        handle_packet   │
            │                  └ telemetry       handle_telemetry│
            └────────────────────────────────────────────────────┘
```

- **Events** run in timestamp order; events with the same timestamp run in the order they were scheduled.
- **Communication**: `broadcast` delivers to every node within `transmission_range` (in node order), `send` to one
  node if it's in range; both after `delay` seconds. Same range computation as the Python simulator.
- **Timers** are integer ids per node; `cancel_timer` cancels every pending timer with that id.
- **Mobility** is the massless model: `goto` sets a target, the node moves at its speed every `update_rate`
  seconds, and every `telemetry_decimation` updates all nodes receive telemetry.
- **Randomness** comes from per-node streams (SplitMix64) seeded from the simulation seed, so draws don't depend on
  the order nodes execute in, and are identical in every implementation.
- **Floating point** operations are the same as in Python. Notably, Python squares numbers with the C library's
  `pow`, which differs from `x * x` in the last bit for about 0.08% of values; the engines call `pow` the same way,
  and C++ modules are compiled without fused multiply-adds.

Messages are arrays of `message_size` float64 values. Nodes of kind `k` run protocol `k`, and one protocol instance
(or state object) serves every node of its kind, so per-node data lives in arrays indexed by node.

### Mapping from regular protocols

| Python simulator (`self.provider...`) | Numba (`ctx...`) | C++ (`ctx.`) |
|---|---|---|
| `send_communication_command(BroadcastMessageCommand(m))` | `ctx.broadcast(node, payload)` | `ctx.broadcast(node, {values})` |
| `send_communication_command(SendMessageCommand(m, d))` | `ctx.send(node, d, payload)` | `ctx.send(node, d, {values})` |
| `schedule_timer("name", t)` | `ctx.schedule_timer(node, timer_id, t)` | `ctx.schedule_timer(node, timer_id, t)` |
| `cancel_timer("name")` | `ctx.cancel_timer(node, timer_id)` | `ctx.cancel_timer(node, timer_id)` |
| `send_mobility_command(GotoCoordsMobilityCommand(x, y, z))` | `ctx.goto(node, x, y, z)` | `ctx.go_to(node, x, y, z)` |
| `send_mobility_command(SetSpeedMobilityCommand(s))` | `ctx.set_speed(node, s)` | `ctx.set_speed(node, s)` |
| `current_time()` | `ctx.now()` | `ctx.now()` |
| `get_id()` | `node` argument | `node` argument |
| `telemetry.current_position` | `ctx.position(node)` | `ctx.position(node)` (`.x`, `.y`, `.z`) |
| `random.random()` | `ctx.random(node)` | `ctx.random(node)` |
| `handle_packet(self, message)` | `handle_packet(ctx, state, node, sender, payload)` | `handle_packet(Context&, int node, int sender, const Payload&)` |

## Writing a policy in Numba mode

A protocol is a set of plain functions. Don't decorate them with `numba.njit`: the simulation compiles them, and any
helper function they call, together with the engine, and runs the same functions as regular Python in Python mode.

```python
from collections import namedtuple
import math
import numpy as np
from gradysim.compiled import CompiledConfiguration, CompiledProtocol, CompiledSimulation

SHARE = 0   # timer id
State = namedtuple("State", ["last_visit", "target", "centers", "distance_weight"])

def choose_next_cell(state, node, now, x, y, z):          # helpers are compiled automatically
    best, best_score = -1, -math.inf
    for cell in range(state.centers.shape[0]):
        dx, dy, dz = state.centers[cell, 0] - x, state.centers[cell, 1] - y, state.centers[cell, 2] - z
        score = (now - state.last_visit[node, cell]) - state.distance_weight * math.sqrt(dx * dx + dy * dy + dz * dz)
        if cell != state.target[node] and score > best_score:
            best, best_score = cell, score
    return best

def go_to_target(ctx, state, node):
    cell = state.target[node]
    ctx.goto(node, state.centers[cell, 0], state.centers[cell, 1], state.centers[cell, 2])

def initialize(ctx, state, node):
    state.target[node] = int(ctx.random(node) * state.centers.shape[0])
    go_to_target(ctx, state, node)
    ctx.schedule_timer(node, SHARE, ctx.now() + 2.0)

def handle_telemetry(ctx, state, node):
    x, y, z = ctx.position(node)
    cell = state.target[node]
    dx, dy, dz = state.centers[cell, 0] - x, state.centers[cell, 1] - y, state.centers[cell, 2] - z
    if dx * dx + dy * dy + dz * dz <= 0.25:
        state.last_visit[node, cell] = ctx.now()
        state.target[node] = choose_next_cell(state, node, ctx.now(), x, y, z)
        go_to_target(ctx, state, node)

def handle_timer(ctx, state, node, timer):
    ctx.broadcast(node, state.last_visit[node])            # share my idleness map
    ctx.schedule_timer(node, SHARE, ctx.now() + 2.0)

def handle_packet(ctx, state, node, sender, payload):
    for cell in range(state.centers.shape[0]):             # merge the neighbor's map
        state.last_visit[node, cell] = max(state.last_visit[node, cell], payload[cell])

patroller = CompiledProtocol(initialize=initialize, handle_timer=handle_timer,
                             handle_packet=handle_packet, handle_telemetry=handle_telemetry)

drones, centers = 10, np.random.default_rng(0).uniform(-100, 100, (36, 3))
state = State(np.zeros((drones, 36)), np.zeros(drones, dtype=np.int64), centers, 1.0)
simulation = CompiledSimulation([patroller], kinds=[0] * drones, positions=np.zeros((drones, 3)), state=state,
                                configuration=CompiledConfiguration(duration=600, message_size=36, timer_ids=1))
result = simulation.run()          # compiled; results are in `state` and `result`
# simulation.run_python()          # same functions on the Python simulator, with a fresh state
```

Rules:

- **State** is any object Numba accepts, usually a namedtuple of numpy arrays (and numbers). It's shared by all
  nodes and modified in place: use rows indexed by `node`, and create a new state for each run. Numbers inside a
  namedtuple can't be modified; use one-element arrays for counters.
- Code must be in the [subset of Python Numba supports](https://numba.readthedocs.io/en/stable/reference/pysupported.html):
  numbers, numpy arrays, loops, `math`, namedtuples. No dictionaries of objects, strings as data, classes or other
  libraries. Helper functions can't be recursive.
- **Payloads** are float64 arrays of `message_size` values, shared by every receiver: don't modify them.
- To compare with a reference implementation bit for bit, write `x * x` (or `x ** 2` in both) consistently, and draw
  random numbers only with `ctx.random(node)`.
- See `showcases/compiled/*/numba_version.py` for complete examples, and `persistent_monitoring/sweep_numba.py` for
  hyperparameter tuning with `Campaign`.

## Writing a policy in C++ mode

A protocol is a class deriving from `gradysim::Protocol`. Its constructor receives a `Setup` giving access to
hyperparameters (`param`), input arrays (`array`) and outputs returned to Python (`output`).

```cpp
#include <cmath>
#include <limits>
#include <gradysim_native/module.hpp>
using namespace gradysim;

class Patroller : public Protocol {
   public:
    explicit Patroller(Setup &setup)
        : centers_(setup.array("centers")), cells_(centers_.shape[0]),
          distance_weight_(setup.param("distance_weight")),
          last_visit_(setup.node_count, std::vector<double>(cells_, 0.0)), target_(setup.node_count, 0),
          visits_(setup.output("visits")) { visits_.assign(cells_, 0.0); }

    void initialize(Context &ctx, int node) override {
        target_[node] = static_cast<int>(ctx.random(node) * cells_);
        go_to_cell(ctx, node);
        ctx.schedule_timer(node, SHARE, ctx.now() + 2.0);
    }
    void handle_telemetry(Context &ctx, int node) override {
        const Vec3 p = ctx.position(node);
        const int cell = target_[node];
        const double dx = centers_.at(cell, 0) - p.x, dy = centers_.at(cell, 1) - p.y, dz = centers_.at(cell, 2) - p.z;
        if (dx * dx + dy * dy + dz * dz <= 0.25) {
            last_visit_[node][cell] = ctx.now();
            visits_[cell] += 1;
            target_[node] = choose_next_cell(node, ctx.now(), p);
            go_to_cell(ctx, node);
        }
    }
    void handle_timer(Context &ctx, int node, int) override {
        ctx.broadcast(node, last_visit_[node]);
        ctx.schedule_timer(node, SHARE, ctx.now() + 2.0);
    }
    void handle_packet(Context &, int node, int, const Payload &payload) override {
        for (int cell = 0; cell < cells_; ++cell)
            last_visit_[node][cell] = std::max(last_visit_[node][cell], payload[cell]);
    }

   private:
    static constexpr int SHARE = 0;
    int choose_next_cell(int node, double now, const Vec3 &p) const {
        int best = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int cell = 0; cell < cells_; ++cell) {
            const double dx = centers_.at(cell, 0) - p.x, dy = centers_.at(cell, 1) - p.y,
                         dz = centers_.at(cell, 2) - p.z;
            const double score = (now - last_visit_[node][cell]) - distance_weight_ * std::sqrt(dx * dx + dy * dy + dz * dz);
            if (cell != target_[node] && score > best_score) { best = cell; best_score = score; }
        }
        return best;
    }
    void go_to_cell(Context &ctx, int node) {
        const int cell = target_[node];
        ctx.go_to(node, centers_.at(cell, 0), centers_.at(cell, 1), centers_.at(cell, 2));
    }
    const Array &centers_;
    const int cells_;
    const double distance_weight_;
    std::vector<std::vector<double>> last_visit_;
    std::vector<int> target_;
    std::vector<double> &visits_;
};

GRADYSIM_PROTOCOLS({"patroller", make<Patroller>})
```

```python
from gradysim.native import NativeConfiguration, NativeSimulation, build

module = build("patroller.cpp")   # compiled once, cached by content in ~/.cache/gradysim/native
simulation = NativeSimulation(module, ["patroller"], kinds=[0] * 10, positions=np.zeros((10, 3)),
                              configuration=NativeConfiguration(duration=600, message_size=36, timer_ids=1),
                              params={"distance_weight": 1.0}, arrays={"centers": centers})
result = simulation.run()         # result["outputs"]["visits"], result["positions"], result["events"]
```

Rules:

- One instance handles every node of its kind: keep per-node data in vectors indexed by `node`.
- Outputs are float64 vectors. References returned by `setup.output(name)` stay valid for the whole run, and several
  protocols can share an output by using the same name.
- Errors thrown with `std::invalid_argument` (and other standard exceptions) reach Python as exceptions.
- Build modules with `gradysim.native.build`, never by hand: it adds the flags that keep floating point results
  identical to Python (`-fno-builtin-pow -ffp-contract=off`). Call `build` once before starting `Campaign` workers.
- See `showcases/compiled/*/*.cpp` for complete examples, and `persistent_monitoring/sweep_cpp.py` for tuning.

## Using compiled modes in an optimization loop

```python
from gradysim.simulator.campaign import Campaign
from showcases.compiled.persistent_monitoring.cpp_version import run   # or numba_version
from showcases.compiled.persistent_monitoring.scenario import PersistentMonitoringScenario

# `optimizer` is any ask/tell optimizer (CMA-ES, a genetic algorithm, ...)
def evaluate(candidate):                         # module-level, so workers can run it
    distance_weight, seed = candidate
    return run(PersistentMonitoringScenario(distance_weight=distance_weight, seed=seed))["average_idleness"]

with Campaign(workers=8) as campaign:            # workers persist across generations
    for generation in range(100):
        candidates = optimizer.ask()
        optimizer.tell(candidates, campaign.map(evaluate, candidates))
```

Each Numba worker compiles on its first run; each C++ worker loads the module built beforehand. Every later
evaluation runs at compiled speed.

## Adding a new mechanism to the engine

Policies (decision rules, message contents, timers) only need protocol code. A new **mechanism** (an energy model,
a lossy channel, another mobility model) changes the engine and must be added to both the engine and the reference,
so equivalence stays verifiable:

1. **Define the behavior in the Python simulator first**: either use an existing handler, or write the handler
   there. It's the reference every compiled mode must reproduce.
2. **Add it to the engine**:
   - Numba: `gradysim/compiled/engine.py`. State goes in the `Context` jitclass `_spec` and `__init__`; operations
     protocols call become `Context` methods; periodic behavior becomes a new event type handled in the loop of
     `build_runner`. Mirror the operation in `PythonContext` (`gradysim/compiled/python_mode.py`) and add the
     handler to `run_python_mode`.
   - C++: `gradysim/native/include/gradysim_native/engine.hpp`. State goes in `Context` members; operations become
     `Context` methods; periodic behavior becomes a new `EventType` handled in `Context::run`. New configuration
     values go in `Configuration`, `module.hpp` and `NativeConfiguration`.
   - Keep the exact order of operations of the Python handler, including when events are scheduled relative to
     others (it decides the order of simultaneous events) and how floating point values are computed.
3. **Extend the conformance protocol** (`showcases/compiled/conformance`): make the reference and every
   implementation use the new mechanism randomly and record what it observes. Its trace compares every callback.
4. **Run the tests** (`tests/test_compiled_numba.py`, `tests/test_compiled_cpp.py`). Any difference fails them.

Example: a communication failure rate is not implemented yet. It would need a random draw per delivery in the
engine from a dedicated stream, and the Python reference would have to draw from the same stream instead of
Python's global `random`.

## How equivalence is verified

`showcases/compiled` contains three scenarios, each implemented as regular gradysim protocols (`reference.py`) and
in each compiled mode:

- **data collection**: drones on waypoint missions collect packets from sensors and deliver them to a ground
  station (the logic of `showcases/simple`);
- **persistent monitoring**: drones choose their next cell from an idleness map they share with their neighbors;
- **conformance**: nodes take random actions with every mechanism (broadcast, unicast, delays, several timers with
  cancellation and rescheduling, movement, speed changes, telemetry decimation) and record every callback.

The tests run each scenario in 16 configurations (seeds, 3 to 40 nodes, delays, update and telemetry rates, ranges,
hyperparameters) and require **identical** outputs: counters, per-node maps, final positions at full precision,
the number of executed events and, for conformance, the complete trace of callbacks with their times and contents.
The Numba tests also check that Python mode matches. We checked that the tests catch errors by introducing them on
purpose (reversed order of simultaneous events, `x * x` instead of `pow`, ignored delays): each one fails the tests.

To verify a new protocol, run it in both modes and compare their outputs: in Numba mode with `run()` and
`run_python()`; in C++ mode against a Python implementation, as in `reference.py`.

## Limitations

Compiled modes cover the core of the simulator. Not available in them:

- Plugins (mission mobility, dispatcher, statistics, RAFT…): write the equivalent logic in the protocol, as the
  examples do for waypoint missions.
- Dynamic velocity and ArduPilot mobility, communication failure rate, visualization, assertion handler and
  execution logging. Numba's `run_python` can use the visualization handler.
- Python libraries inside protocols, including neural networks. A small trained network can be rewritten as a
  compiled forward pass with exported weights; large ones, or ones being trained, belong in the Python simulator.
- Messages are fixed-size float64 arrays instead of arbitrary strings.

## Choosing

- **Numba** if protocols change often, if you want to debug them in Python and on the regular simulator, or if a
  12-15x speedup is enough. No compiler to install, Python syntax.
- **C++** if evaluations dominate your compute budget: about 3x faster than Numba, compiled once for all processes,
  at the cost of writing and debugging C++.

A protocol can be prototyped in the Python simulator, ported to Numba (its Python mode checks the port), and moved
to C++ when it's stable, using the Numba results as the reference.
