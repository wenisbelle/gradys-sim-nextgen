// Data collection as native protocols. Same behavior as reference.py.
#include <gradysim_native/module.hpp>

using namespace gradysim;

namespace {

constexpr double GROUND = 0, DRONE = 1, SENSOR = 2;
constexpr int PING = 0;

// Packet counters and received messages of every node, returned to Python
struct Counters {
    explicit Counters(Setup &setup) : packets(setup.output("packets")), received(setup.output("received")) {
        packets.resize(setup.node_count, 0.0);
        received.resize(setup.node_count, 0.0);
    }
    std::vector<double> &packets;
    std::vector<double> &received;
};

class Ground : public Protocol {
   public:
    explicit Ground(Setup &setup) : counters_(setup) {}

    void handle_packet(Context &ctx, int node, int, const Payload &payload) override {
        counters_.received[node] += 1;
        if (payload[0] == DRONE) {
            counters_.packets[node] += static_cast<int64_t>(payload[1]);
            ctx.broadcast(node, {GROUND, counters_.packets[node]});
        }
    }

   private:
    Counters counters_;
};

class Drone : public Protocol {
   public:
    explicit Drone(Setup &setup)
        : counters_(setup),
          missions_(setup.array("missions")),
          speed_(setup.param("speed")),
          tolerance_(setup.param("tolerance")),
          waypoint_(setup.node_count, 0) {}

    void initialize(Context &ctx, int node) override {
        go_to_waypoint(ctx, node, 0);
        ctx.set_speed(node, speed_);
        ctx.schedule_timer(node, PING, ctx.now() + ctx.random(node));
    }

    void handle_timer(Context &ctx, int node, int) override {
        ctx.broadcast(node, {DRONE, counters_.packets[node]});
        ctx.schedule_timer(node, PING, ctx.now() + ctx.random(node));
    }

    void handle_packet(Context &, int node, int, const Payload &payload) override {
        counters_.received[node] += 1;
        if (payload[0] == GROUND) {
            counters_.packets[node] = 0;
        } else if (payload[0] == SENSOR) {
            counters_.packets[node] += static_cast<int64_t>(payload[1]);
        }
    }

    void handle_telemetry(Context &ctx, int node) override {
        const Vec3 position = ctx.position(node);
        const int drone = node - 1;
        const int waypoint = waypoint_[node];
        const double dx = missions_.at(drone, waypoint, 0) - position.x;
        const double dy = missions_.at(drone, waypoint, 1) - position.y;
        const double dz = missions_.at(drone, waypoint, 2) - position.z;
        if (dx * dx + dy * dy + dz * dz <= tolerance_ * tolerance_) {
            waypoint_[node] = static_cast<int>((waypoint + 1) % missions_.shape[1]);
            go_to_waypoint(ctx, node, waypoint_[node]);
        }
    }

   private:
    void go_to_waypoint(Context &ctx, int node, int waypoint) {
        const int drone = node - 1;
        ctx.go_to(node, missions_.at(drone, waypoint, 0), missions_.at(drone, waypoint, 1),
                  missions_.at(drone, waypoint, 2));
    }

    Counters counters_;
    const Array &missions_;
    const double speed_, tolerance_;
    std::vector<int> waypoint_;
};

class Sensor : public Protocol {
   public:
    explicit Sensor(Setup &setup) : counters_(setup) {}

    void initialize(Context &ctx, int node) override {
        counters_.packets[node] = 5;
        ctx.schedule_timer(node, PING, ctx.now() + ctx.random(node));
    }

    void handle_timer(Context &ctx, int node, int) override {
        counters_.packets[node] += 1;
        ctx.schedule_timer(node, PING, ctx.now() + ctx.random(node));
    }

    void handle_packet(Context &ctx, int node, int, const Payload &payload) override {
        counters_.received[node] += 1;
        if (payload[0] == DRONE) {
            ctx.broadcast(node, {SENSOR, counters_.packets[node]});
            counters_.packets[node] = 0;
        }
    }

   private:
    Counters counters_;
};

}  // namespace

GRADYSIM_PROTOCOLS({"ground", make<Ground>}, {"drone", make<Drone>}, {"sensor", make<Sensor>})
