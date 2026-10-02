// GrADyS-SIM NextGen native (C++) simulation engine.
//
// Protocols are C++ classes deriving from gradysim::Protocol. The engine reproduces the Python simulator with its
// communication, timer and massless mobility handlers exactly: events run in timestamp order, ties in scheduling
// order, and floating point operations match Python's. Build modules with gradysim.native.build, which compiles with
// -fno-builtin-pow -ffp-contract=off so the compiler doesn't change floating point results.
#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gradysim {

struct Vec3 {
    double x, y, z;
};

// Read-only view of a message payload. Every receiver of a message shares it.
struct Payload {
    const double *values;
    int size;
    double operator[](int index) const { return values[index]; }
};

// Read-only view of a float64 array given by Python, in row-major order.
struct Array {
    std::vector<double> values;
    std::vector<int64_t> shape;
    double operator[](int64_t index) const { return values[index]; }
    double at(int64_t row, int64_t column) const { return values[row * shape.at(1) + column]; }
    double at(int64_t a, int64_t b, int64_t c) const { return values[(a * shape.at(1) + b) * shape.at(2) + c]; }
};

struct Configuration {
    double duration = 0;
    double transmission_range = 60;
    double delay = 0;
    bool mobility = true;
    double update_rate = 0.01;
    int64_t telemetry_decimation = 1;
    int timer_ids = 8;
    uint64_t seed = 0;
};

using Outputs = std::map<std::string, std::vector<double>>;

// Information available to protocols when they are created: node count, hyperparameters and arrays given by
// Python, and the outputs returned to Python.
class Setup {
   public:
    Setup(int node_count, std::map<std::string, double> params, std::map<std::string, Array> arrays,
          Outputs &outputs)
        : node_count(node_count), params_(std::move(params)), arrays_(std::move(arrays)), outputs_(outputs) {}

    const int node_count;

    double param(const std::string &name) const {
        auto found = params_.find(name);
        if (found == params_.end()) throw std::invalid_argument("Missing parameter: " + name);
        return found->second;
    }

    const Array &array(const std::string &name) const {
        auto found = arrays_.find(name);
        if (found == arrays_.end()) throw std::invalid_argument("Missing array: " + name);
        return found->second;
    }

    // Output returned to Python as a numpy array. References stay valid for the whole run.
    std::vector<double> &output(const std::string &name) { return outputs_[name]; }

   private:
    std::map<std::string, double> params_;
    std::map<std::string, Array> arrays_;
    Outputs &outputs_;
};

class Context;

// Base class of protocols. One instance handles every node of its kind, so per-node state is usually stored in
// vectors indexed by node.
class Protocol {
   public:
    virtual ~Protocol() = default;
    virtual void initialize(Context &, int node) {}
    virtual void handle_timer(Context &, int node, int timer) {}
    virtual void handle_packet(Context &, int node, int sender, const Payload &payload) {}
    virtual void handle_telemetry(Context &, int node) { handles_telemetry = false; }
    virtual void finish(Context &, int node) {}

    // Cleared by the default handle_telemetry, so the engine stops delivering telemetry to protocols ignoring it
    bool handles_telemetry = true;
};

using ProtocolFactory = std::function<std::unique_ptr<Protocol>(Setup &)>;

template <typename T>
std::unique_ptr<Protocol> make(Setup &setup) {
    return std::make_unique<T>(setup);
}

namespace detail {

// Python squares floats with the C library's pow, which differs from x * x in the last bit for some values. Calling
// it through an opaque pointer stops the compiler from replacing it with x * x.
inline double (*volatile libm_pow)(double, double) = std::pow;
inline double square(double value) { return libm_pow(value, 2.0); }

constexpr uint64_t GOLDEN_GAMMA = 0x9E3779B97F4A7C15ULL;

enum EventType : int32_t { MOBILITY, TELEMETRY, TIMER, PACKET };

struct Event {
    double time;
    int64_t sequence;
    int32_t type;
    int32_t a;
    int64_t b;
    int64_t c;
};

struct Later {
    bool operator()(const Event &x, const Event &y) const {
        return x.time > y.time || (x.time == y.time && x.sequence > y.sequence);
    }
};

}  // namespace detail

// Simulation context received by every handler. Holds the simulation state and offers the operations protocols
// can perform.
class Context {
   public:
    Context(const Configuration &configuration, std::vector<int> kinds, const std::vector<Vec3> &positions)
        : configuration_(configuration),
          kinds_(std::move(kinds)),
          positions_(positions),
          known_positions_(positions),
          targets_(positions.size(), Vec3{0, 0, 0}),
          has_target_(positions.size(), 0),
          speeds_(positions.size(), 10.0),
          random_states_(positions.size()),
          timer_generation_(positions.size() * configuration.timer_ids, 0),
          squared_range_(detail::square(configuration.transmission_range)),
          delay_(configuration.delay > 0 ? configuration.delay : 0.0) {
        for (size_t node = 0; node < positions.size(); ++node) {
            random_states_[node] = configuration.seed * 0x100000001B3ULL + node * detail::GOLDEN_GAMMA;
        }
    }

    // Current simulation time in seconds
    double now() const { return time_; }
    // Number of nodes in the simulation
    int node_count() const { return static_cast<int>(kinds_.size()); }
    // Kind of a node, which is the index of its protocol
    int kind(int node) const { return kinds_[node]; }
    // Position of the node as reported by its latest telemetry, or its initial position before any telemetry
    Vec3 position(int node) const { return known_positions_[node]; }

    // Next uniform number in [0, 1) of the node's own random stream (SplitMix64)
    double random(int node) {
        uint64_t z = (random_states_[node] += detail::GOLDEN_GAMMA);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
    }

    // Sends the payload to every node within transmission range of `node`
    void broadcast(int node, const std::vector<double> &payload) {
        const Vec3 source = positions_[node];
        receivers_.clear();
        for (int destination = 0; destination < node_count(); ++destination) {
            if (destination != node && in_range(destination, source)) receivers_.push_back(destination);
        }
        if (receivers_.empty()) return;
        const int64_t message = store(payload, static_cast<int64_t>(receivers_.size()));
        for (int destination : receivers_) push(time_ + delay_, detail::PACKET, destination, node, message);
    }

    // Sends the payload to `destination`, delivered only if it's within transmission range of `node`
    void send(int node, int destination, const std::vector<double> &payload) {
        if (destination == node)
            throw std::invalid_argument("Error transmitting message: message destination is equal to sender");
        if (destination < 0 || destination >= node_count())
            throw std::invalid_argument("Error transmitting message: destination does not exist");
        if (in_range(destination, positions_[node])) {
            push(time_ + delay_, detail::PACKET, destination, node, store(payload, 1));
        }
    }

    // Schedules timer `timer` (an integer in [0, timer_ids)) of `node` to fire at simulation time `time`
    void schedule_timer(int node, int timer, double time) {
        if (time < time_) throw std::invalid_argument("Could not set timer: Timer cannot be set in the past");
        push(time, detail::TIMER, node, timer, generation(node, timer));
    }

    // Cancels every pending timer `timer` of `node`
    void cancel_timer(int node, int timer) { generation(node, timer) += 1; }

    // Starts moving `node` towards the given position at its current speed
    void go_to(int node, double x, double y, double z) {
        targets_[node] = Vec3{x, y, z};
        has_target_[node] = 1;
    }

    // Sets the speed of `node` in m/s
    void set_speed(int node, double speed) { speeds_[node] = speed; }

    // ----------------------------------------------------------------------------------------------------------
    // Engine

    struct Result {
        int64_t events;
        double time;
        std::vector<Vec3> positions;
    };

    Result run(std::vector<Protocol *> &protocols) {
        const int n = node_count();
        if (configuration_.mobility) push(0.0 + configuration_.update_rate, detail::MOBILITY, 0, 0, 0);
        for (int node = 0; node < n; ++node) protocols[kinds_[node]]->initialize(*this, node);

        int64_t update_count = 0;
        while (next_event_live() && queue_.top().time <= configuration_.duration) {
            const detail::Event event = queue_.top();
            queue_.pop();
            ++events_;
            time_ = event.time;

            switch (event.type) {
                case detail::MOBILITY:
                    update_mobility();
                    if (++update_count % configuration_.telemetry_decimation == 0)
                        push(time_, detail::TELEMETRY, 0, 0, 0);
                    push(time_ + configuration_.update_rate, detail::MOBILITY, 0, 0, 0);
                    break;
                case detail::TELEMETRY:
                    for (int node = 0; node < n; ++node) {
                        known_positions_[node] = positions_[node];
                        Protocol *protocol = protocols[kinds_[node]];
                        if (protocol->handles_telemetry) protocol->handle_telemetry(*this, node);
                    }
                    break;
                case detail::TIMER:
                    protocols[kinds_[event.a]]->handle_timer(*this, event.a, static_cast<int>(event.b));
                    break;
                case detail::PACKET: {
                    // Copied because handlers may send messages, which can reallocate the pool
                    delivery_.assign(pool_.begin() + event.c * message_size_,
                                     pool_.begin() + (event.c + 1) * message_size_);
                    release(event.c);
                    const Payload payload{delivery_.data(), message_size_};
                    protocols[kinds_[event.a]]->handle_packet(*this, event.a, static_cast<int>(event.b), payload);
                    break;
                }
            }
        }
        for (int node = 0; node < n; ++node) protocols[kinds_[node]]->finish(*this, node);
        return Result{events_, time_, positions_};
    }

    void set_message_size(int size) { message_size_ = size; }

   private:
    bool in_range(int destination, const Vec3 &source) const {
        const Vec3 &p = positions_[destination];
        return detail::square(p.x - source.x) + detail::square(p.y - source.y) + detail::square(p.z - source.z) <=
               squared_range_;
    }

    int64_t &generation(int node, int timer) {
        if (timer < 0 || timer >= configuration_.timer_ids) throw std::invalid_argument("Timer id out of range");
        return timer_generation_[static_cast<size_t>(node) * configuration_.timer_ids + timer];
    }

    int64_t store(const std::vector<double> &payload, int64_t references) {
        if (static_cast<int>(payload.size()) > message_size_)
            throw std::invalid_argument("Payload is larger than message_size");
        int64_t row;
        if (!free_rows_.empty()) {
            row = free_rows_.back();
            free_rows_.pop_back();
        } else {
            row = static_cast<int64_t>(references_.size());
            references_.push_back(0);
            pool_.resize(pool_.size() + message_size_);
        }
        double *destination = &pool_[row * message_size_];
        for (int index = 0; index < message_size_; ++index)
            destination[index] = index < static_cast<int>(payload.size()) ? payload[index] : 0.0;
        references_[row] = references;
        return row;
    }

    void release(int64_t row) {
        if (--references_[row] == 0) free_rows_.push_back(row);
    }

    void push(double time, int32_t type, int32_t a, int64_t b, int64_t c) {
        queue_.push(detail::Event{time, sequence_++, type, a, b, c});
    }

    // Discards cancelled timers at the top of the queue. Returns whether an event is left.
    bool next_event_live() {
        while (!queue_.empty()) {
            const detail::Event &top = queue_.top();
            if (top.type == detail::TIMER && generation(top.a, static_cast<int>(top.b)) != top.c) {
                queue_.pop();
            } else {
                return true;
            }
        }
        return false;
    }

    // Same computation as MasslessMobilityHandler
    void update_mobility() {
        for (int node = 0; node < node_count(); ++node) {
            if (!has_target_[node]) continue;
            Vec3 &p = positions_[node];
            const Vec3 &t = targets_[node];
            const double vx = t.x - p.x, vy = t.y - p.y, vz = t.z - p.z;
            const double movement = speeds_[node] * configuration_.update_rate;
            const double distance = std::sqrt(detail::square(vx) + detail::square(vy) + detail::square(vz));
            if (movement >= distance) {
                p = t;
            } else {
                const double k = movement / distance;
                p.x = p.x + vx * k;
                p.y = p.y + vy * k;
                p.z = p.z + vz * k;
            }
        }
    }

    Configuration configuration_;
    std::vector<int> kinds_;
    std::vector<Vec3> positions_, known_positions_, targets_;
    std::vector<char> has_target_;
    std::vector<double> speeds_;
    std::vector<uint64_t> random_states_;
    std::vector<int64_t> timer_generation_;
    double squared_range_;
    double delay_;
    double time_ = 0.0;
    int64_t events_ = 0;
    int64_t sequence_ = 0;
    int message_size_ = 4;
    std::priority_queue<detail::Event, std::vector<detail::Event>, detail::Later> queue_;
    std::vector<double> pool_;
    std::vector<int64_t> references_, free_rows_;
    std::vector<int> receivers_;
    std::vector<double> delivery_;
};

}  // namespace gradysim
