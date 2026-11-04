#pragma once
#include <atomic>
#include <cstdint>

namespace conduit {
    struct list_counters {
        uint64_t draws = 0;
        uint64_t dispatches = 0;
        uint64_t indirect = 0;
        uint64_t barriers = 0;
        uint64_t pipeline_binds = 0;
        uint64_t group_binds = 0;
    };

    struct device_counters {
        std::atomic<uint64_t> draws{0};
        std::atomic<uint64_t> dispatches{0};
        std::atomic<uint64_t> indirect{0};
        std::atomic<uint64_t> barriers{0};
        std::atomic<uint64_t> pipeline_binds{0};
        std::atomic<uint64_t> group_binds{0};
        std::atomic<uint64_t> submits{0};

        void add(const list_counters& c) {
            constexpr auto relaxed = std::memory_order_relaxed;
            draws.fetch_add(c.draws, relaxed);
            dispatches.fetch_add(c.dispatches, relaxed);
            indirect.fetch_add(c.indirect, relaxed);
            barriers.fetch_add(c.barriers, relaxed);
            pipeline_binds.fetch_add(c.pipeline_binds, relaxed);
            group_binds.fetch_add(c.group_binds, relaxed);
        }

        void count_submit() { submits.fetch_add(1, std::memory_order_relaxed); }

        void fill(auto& stats) const {
            constexpr auto relaxed = std::memory_order_relaxed;
            stats.draws = draws.load(relaxed);
            stats.dispatches = dispatches.load(relaxed);
            stats.indirect_draws = indirect.load(relaxed);
            stats.barriers = barriers.load(relaxed);
            stats.pipeline_binds = pipeline_binds.load(relaxed);
            stats.bind_group_binds = group_binds.load(relaxed);
            stats.submits = submits.load(relaxed);
        }
    };
}
