// Persistent monitoring as a native protocol. Same behavior as reference.py.
#include <cmath>
#include <limits>

#include <gradysim_native/module.hpp>

using namespace gradysim;

namespace {

constexpr int BROADCAST_TIMER = 0;

class Patroller : public Protocol {
   public:
    explicit Patroller(Setup &setup)
        : centers_(setup.array("centers")),
          cells_(static_cast<int>(centers_.shape[0])),
          distance_weight_(setup.param("distance_weight")),
          broadcast_period_(setup.param("broadcast_period")),
          speed_(setup.param("speed")),
          tolerance_(setup.param("tolerance")),
          last_visit_(setup.node_count, std::vector<double>(cells_, 0.0)),
          target_(setup.node_count, 0),
          true_last_visit_(setup.output("last_visit")),
          gap_squares_(setup.output("gap_squares")),
          visits_(setup.output("visits")),
          received_(setup.output("received")),
          drone_maps_(setup.output("drone_maps")) {
        true_last_visit_.assign(cells_, 0.0);
        gap_squares_.assign(cells_, 0.0);
        visits_.assign(cells_, 0.0);
        received_.assign(setup.node_count, 0.0);
    }

    void initialize(Context &ctx, int node) override {
        target_[node] = static_cast<int>(ctx.random(node) * cells_);
        go_to_cell(ctx, node, target_[node]);
        ctx.set_speed(node, speed_);
        ctx.schedule_timer(node, BROADCAST_TIMER, ctx.now() + broadcast_period_ * ctx.random(node));
    }

    void handle_telemetry(Context &ctx, int node) override {
        const Vec3 p = ctx.position(node);
        const int cell = target_[node];
        const double dx = centers_.at(cell, 0) - p.x, dy = centers_.at(cell, 1) - p.y, dz = centers_.at(cell, 2) - p.z;
        if (dx * dx + dy * dy + dz * dz <= tolerance_ * tolerance_) {
            const double now = ctx.now();
            const double gap = now - true_last_visit_[cell];
            gap_squares_[cell] += gap * gap;
            true_last_visit_[cell] = now;
            visits_[cell] += 1;
            last_visit_[node][cell] = now;
            target_[node] = choose_next_cell(node, now, p);
            go_to_cell(ctx, node, target_[node]);
        }
    }

    void handle_timer(Context &ctx, int node, int) override {
        ctx.broadcast(node, last_visit_[node]);
        ctx.schedule_timer(node, BROADCAST_TIMER, ctx.now() + broadcast_period_);
    }

    void handle_packet(Context &, int node, int, const Payload &payload) override {
        received_[node] += 1;
        for (int cell = 0; cell < cells_; ++cell) {
            if (payload[cell] > last_visit_[node][cell]) last_visit_[node][cell] = payload[cell];
        }
    }

    void finish(Context &ctx, int node) override {
        if (node == 0) drone_maps_.clear();
        drone_maps_.insert(drone_maps_.end(), last_visit_[node].begin(), last_visit_[node].end());
    }

   private:
    int choose_next_cell(int node, double now, const Vec3 &p) const {
        int best = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int cell = 0; cell < cells_; ++cell) {
            if (cell == target_[node]) continue;
            const double dx = centers_.at(cell, 0) - p.x, dy = centers_.at(cell, 1) - p.y,
                         dz = centers_.at(cell, 2) - p.z;
            const double score = (now - last_visit_[node][cell]) - distance_weight_ * std::sqrt(dx * dx + dy * dy + dz * dz);
            if (score > best_score) {
                best = cell;
                best_score = score;
            }
        }
        return best;
    }

    void go_to_cell(Context &ctx, int node, int cell) {
        ctx.go_to(node, centers_.at(cell, 0), centers_.at(cell, 1), centers_.at(cell, 2));
    }

    const Array &centers_;
    const int cells_;
    const double distance_weight_, broadcast_period_, speed_, tolerance_;
    std::vector<std::vector<double>> last_visit_;
    std::vector<int> target_;
    std::vector<double> &true_last_visit_, &gap_squares_, &visits_, &received_, &drone_maps_;
};

}  // namespace

GRADYSIM_PROTOCOLS({"patroller", make<Patroller>})
