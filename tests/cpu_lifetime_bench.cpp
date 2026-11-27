// CPU-only lifetime/layout benchmark. Compile this same file against each revision's src/include directories.
// CONDUIT_BENCH_BASELINE selects the original DX12 layout scan; retirement always uses the included revision.
#include "common/retire_queue.hpp"
#ifndef CONDUIT_BENCH_BASELINE
#include "common/binding_lookup.hpp"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <random>
#include <span>
#include <vector>

using namespace conduit;
using clock_type = std::chrono::steady_clock;

namespace {
    std::atomic<uint64_t> allocations{0};
}

void* operator new(size_t bytes) {
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void* value = std::malloc(bytes ? bytes : 1))
        return value;
    throw std::bad_alloc();
}
void operator delete(void* value) noexcept { std::free(value); }
void operator delete(void* value, size_t) noexcept { std::free(value); }

namespace {
    double elapsed(clock_type::time_point start) {
        return std::chrono::duration<double, std::nano>(clock_type::now() - start).count();
    }

    double percentile(std::vector<double> values, double fraction) {
        std::sort(values.begin(), values.end());
        return values[std::min(values.size() - 1, static_cast<size_t>(values.size() * fraction))];
    }

    bool lifetime(uint32_t callbacks, bool comma) {
        retire_queue queue;
        uint64_t released = 0;
        constexpr uint64_t warmup = 24;
        constexpr uint64_t frames = 500;
        std::vector<double> samples;
        samples.reserve(frames);
        uint64_t measured_allocations = 0;
        for (uint64_t frame = 0; frame < warmup + frames; frame++) {
            queue_serials submitted{}, done{};
            submitted.fill(frame + 1);
            done.fill(frame > 2 ? frame - 2 : 0);
            const uint64_t before_allocations = allocations.load(std::memory_order_relaxed);
            const auto start = clock_type::now();
            for (uint32_t i = 0; i < callbacks; i++)
                queue.defer([&released] { released++; });
            queue.seal(submitted);
            queue.collect(done);
            queue.collect(done); // begin/end polls may both see the same completed frame.
            const double duration = elapsed(start);
            const uint64_t count = allocations.load(std::memory_order_relaxed) - before_allocations;
            if (frame >= warmup) {
                samples.push_back(duration);
                measured_allocations += count;
            }
        }
        queue.flush_all();
        std::printf("%s{\"scenario\":\"retirement\",\"callbacks_per_frame\":%u,"
                    "\"frames\":%llu,\"median_ns_per_frame\":%.3f,\"p95_ns_per_frame\":%.3f,"
                    "\"allocations_per_frame\":%.3f,\"released\":%llu}",
                    comma ? ",\n" : "", callbacks, static_cast<unsigned long long>(frames),
                    percentile(samples, 0.5), percentile(samples, 0.95),
                    static_cast<double>(measured_allocations) / frames, static_cast<unsigned long long>(released));
        return released == (warmup + frames) * callbacks && queue.pending() == 0;
    }

    void layout(bool sparse) {
        std::vector<bind_layout_entry> entries(64);
        std::vector<uint32_t> queries;
        for (uint32_t i = 0; i < entries.size(); i++) {
            entries[i].binding = sparse && i % 8 == 0 ? UINT32_MAX - i * 17 : i;
            queries.push_back(entries[i].binding);
        }
        std::mt19937 random(0x74983);
        std::shuffle(entries.begin(), entries.end(), random);
        std::shuffle(queries.begin(), queries.end(), random);
#ifndef CONDUIT_BENCH_BASELINE
        binding_lookup lookup;
        lookup.assign(entries);
#endif
        uint64_t checksum = 0;
        constexpr uint32_t samples_count = 30;
        constexpr uint32_t repetitions = 2000;
        std::vector<double> samples;
        samples.reserve(samples_count);
        for (uint32_t sample = 0; sample < samples_count + 4; sample++) {
            const auto start = clock_type::now();
            for (uint32_t repeat = 0; repeat < repetitions; repeat++) {
                std::atomic_signal_fence(std::memory_order_seq_cst);
                for (uint32_t binding : queries) {
#ifdef CONDUIT_BENCH_BASELINE
                    size_t result = static_cast<size_t>(-1);
                    for (size_t i = 0; i < entries.size(); i++)
                        if (entries[i].binding == binding)
                            result = i;
#else
                    const size_t result = lookup.find(binding);
#endif
                    checksum += result;
                }
            }
            const double duration = elapsed(start) / (repetitions * queries.size());
            if (sample >= 4)
                samples.push_back(duration);
        }
        std::printf(",\n{\"scenario\":\"%s\",\"entries\":64,\"median_ns_per_lookup\":%.3f,"
                    "\"p95_ns_per_lookup\":%.3f,\"checksum\":%llu}",
                    sparse ? "sparse_layout" : "dense_layout", percentile(samples, 0.5), percentile(samples, 0.95),
                    static_cast<unsigned long long>(checksum));
    }
}

int main() {
#ifdef CONDUIT_BENCH_BASELINE
    const char* implementation = "baseline";
#else
    const char* implementation = "optimized";
#endif
    std::printf("{\"implementation\":\"%s\",\"unit\":\"CPU only, not FPS\",\"results\":[\n", implementation);
    bool ok = lifetime(64, false);
    ok &= lifetime(512, true);
    ok &= lifetime(4096, true);
    layout(false);
    layout(true);
    std::printf("\n]}\n");
    return ok ? 0 : 1;
}
