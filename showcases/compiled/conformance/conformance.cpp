// Conformance protocol as a native protocol. Same behavior as reference.py.
#include <gradysim_native/module.hpp>

using namespace gradysim;

namespace {

constexpr int TIMER_IDS = 4;
constexpr double TIMER_RECORD = 0, PACKET_RECORD = 1, TELEMETRY_RECORD = 2;

class RandomActor : public Protocol {
   public:
    explicit RandomActor(Setup &setup)
        : nodes_(setup.node_count),
          area_(setup.param("area")),
          sent_(setup.node_count, 0),
          time_(setup.output("trace_time")),
          node_(setup.output("trace_node")),
          kind_(setup.output("trace_kind")),
          a_(setup.output("trace_a")),
          b_(setup.output("trace_b")),
          c_(setup.output("trace_c")) {}

    void initialize(Context &ctx, int node) override {
        for (int timer = 0; timer < TIMER_IDS; ++timer) ctx.schedule_timer(node, timer, ctx.now() + 2 * ctx.random(node));
    }

    void handle_timer(Context &ctx, int node, int timer) override {
        const double now = ctx.now();
        record(now, node, TIMER_RECORD, timer, 0, 0);
        const int action = static_cast<int>(ctx.random(node) * 6);
        if (action == 0) {
            ctx.broadcast(node, payload(ctx, node));
        } else if (action == 1) {
            const int destination = (node + 1 + static_cast<int>(ctx.random(node) * (nodes_ - 1))) % nodes_;
            ctx.send(node, destination, payload(ctx, node));
        } else if (action == 2) {
            const double x = (ctx.random(node) * 2 - 1) * area_;
            const double y = (ctx.random(node) * 2 - 1) * area_;
            const double z = ctx.random(node) * 20;
            ctx.go_to(node, x, y, z);
        } else if (action == 3) {
            ctx.set_speed(node, 1 + ctx.random(node) * 20);
        } else if (action == 4) {
            ctx.cancel_timer(node, static_cast<int>(ctx.random(node) * TIMER_IDS));
        }

        if (ctx.random(node) < 0.85) {
            ctx.schedule_timer(node, timer, now + 2 * ctx.random(node));
        } else {
            ctx.schedule_timer(node, (timer + 1) % TIMER_IDS, now + 2 * ctx.random(node));
        }
    }

    void handle_packet(Context &ctx, int node, int sender, const Payload &values) override {
        record(ctx.now(), node, PACKET_RECORD, values[0], values[1], values[2]);
        if (ctx.random(node) < 0.3) ctx.send(node, sender, payload(ctx, node));
    }

    void handle_telemetry(Context &ctx, int node) override {
        const Vec3 p = ctx.position(node);
        record(ctx.now(), node, TELEMETRY_RECORD, p.x, p.y, p.z);
    }

   private:
    std::vector<double> payload(Context &ctx, int node) {
        sent_[node] += 1;
        const double sent = static_cast<double>(sent_[node]);
        return {static_cast<double>(node), sent, ctx.random(node)};
    }

    void record(double time, int node, double kind, double a, double b, double c) {
        time_.push_back(time);
        node_.push_back(node);
        kind_.push_back(kind);
        a_.push_back(a);
        b_.push_back(b);
        c_.push_back(c);
    }

    const int nodes_;
    const double area_;
    std::vector<int64_t> sent_;
    std::vector<double> &time_, &node_, &kind_, &a_, &b_, &c_;
};

}  // namespace

GRADYSIM_PROTOCOLS({"random_actor", make<RandomActor>})
