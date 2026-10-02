// Small protocols used by tests/test_compiled_cpp.py to check the native engine
#include <gradysim_native/module.hpp>

using namespace gradysim;

namespace {

// Records the first draws of every node's random stream
class RecordDraws : public Protocol {
   public:
    explicit RecordDraws(Setup &setup) : draws_(setup.output("draws")), count_(setup.param("count")) {}
    void initialize(Context &ctx, int node) override {
        for (int index = 0; index < count_; ++index) draws_.push_back(ctx.random(node));
    }

   private:
    std::vector<double> &draws_;
    const int count_;
};

// Schedules timers for the same instant and records the order in which they fire
class SimultaneousTimers : public Protocol {
   public:
    explicit SimultaneousTimers(Setup &setup) : order_(setup.output("order")) {}
    void initialize(Context &ctx, int node) override {
        for (int timer : {3, 1, 2, 0}) ctx.schedule_timer(node, timer, 0.5);
    }
    void handle_timer(Context &, int node, int timer) override { order_.push_back(node * 10 + timer); }

   private:
    std::vector<double> &order_;
};

class SendToSelf : public Protocol {
   public:
    explicit SendToSelf(Setup &) {}
    void initialize(Context &ctx, int node) override { ctx.send(node, node, {0.0}); }
};

class TimerInThePast : public Protocol {
   public:
    explicit TimerInThePast(Setup &) {}
    void initialize(Context &ctx, int node) override { ctx.schedule_timer(node, 0, -1.0); }
};

}  // namespace

GRADYSIM_PROTOCOLS({"record_draws", make<RecordDraws>}, {"simultaneous_timers", make<SimultaneousTimers>},
                   {"send_to_self", make<SendToSelf>}, {"timer_in_the_past", make<TimerInThePast>})
