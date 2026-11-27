#include "common/binding_lookup.hpp"
#include "common/retire_queue.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using namespace conduit;

namespace {
    int failed = 0;
#define CHECK(condition)                                                                                               \
    do {                                                                                                               \
        if (!(condition)) {                                                                                            \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #condition);                                   \
            failed++;                                                                                                  \
        }                                                                                                              \
    } while (0)

    // The pre-optimization queue's observable rules, with integer callback IDs instead of resource objects.
    struct reference_queue {
        struct item {
            queue_serials serials;
            int id;
        };
        std::vector<item> items;
        std::vector<int> open;
        std::vector<int> released;

        void push(const queue_serials& serials, int id) { items.push_back({serials, id}); }
        void defer(int id) { open.push_back(id); }
        void seal(const queue_serials& serials) {
            for (int id : open)
                push(serials, id);
            open.clear();
        }
        void collect(const queue_serials& done) {
            size_t kept = 0;
            for (const item value : items) {
                bool ready = true;
                for (size_t i = 0; i < queue_count; i++)
                    ready &= done[i] >= value.serials[i];
                if (ready)
                    released.push_back(value.id);
                else
                    items[kept++] = value;
            }
            items.resize(kept);
        }
        void flush() {
            for (const item& value : items)
                released.push_back(value.id);
            released.insert(released.end(), open.begin(), open.end());
            items.clear();
            open.clear();
        }
        size_t pending() const { return items.size() + open.size(); }
    };

    void test_retirement_order() {
        retire_queue queue;
        std::vector<int> released;
        queue_serials zero{};
        queue_serials graphics{};
        graphics[0] = 10;
        queue_serials copy{};
        copy[queue_count - 1] = 2;
        queue.push(graphics, [&] { released.push_back(1); });
        queue.push(zero, [&] { released.push_back(2); });
        queue.push(copy, [&] { released.push_back(3); });
        queue.defer([&] { released.push_back(4); });
        queue.collect(copy);
        CHECK((released == std::vector<int>{2, 3}));
        CHECK(queue.pending() == 2);
        queue.seal(graphics);
        queue.defer([&] { released.push_back(5); });
        queue.collect(graphics);
        CHECK((released == std::vector<int>{2, 3, 1, 4}));
        CHECK(queue.pending() == 1);
        queue.flush_all();
        CHECK((released == std::vector<int>{2, 3, 1, 4, 5}));
        CHECK(queue.pending() == 0);
    }

    void test_retirement_differential() {
        std::mt19937 random(0x31c03e91);
        for (int run = 0; run < 40; run++) {
            retire_queue queue;
            reference_queue reference;
            std::vector<int> released;
            int next = 0;
            for (int step = 0; step < 4000; step++) {
                queue_serials serials{};
                for (auto& serial : serials)
                    serial = random() % 13;
                switch (random() % 6) {
                case 0:
                case 1: {
                    const int id = next++;
                    queue.defer([&, id] { released.push_back(id); });
                    reference.defer(id);
                    break;
                }
                case 2: {
                    const int id = next++;
                    queue.push(serials, [&, id] { released.push_back(id); });
                    reference.push(serials, id);
                    break;
                }
                case 3:
                    queue.seal(serials);
                    reference.seal(serials);
                    break;
                case 4:
                    queue.collect(serials);
                    reference.collect(serials);
                    break;
                default:
                    queue.flush_all();
                    reference.flush();
                    break;
                }
                CHECK(queue.pending() == reference.pending());
                CHECK(released == reference.released);
            }
            queue.flush_all();
            reference.flush();
            CHECK(released == reference.released);
            std::sort(released.begin(), released.end());
            CHECK(released.size() == static_cast<size_t>(next));
            CHECK(std::adjacent_find(released.begin(), released.end()) == released.end());
        }
    }

    void test_reentrant_release() {
        retire_queue queue;
        std::vector<int> released;
        queue_serials zero{};
        queue.push(zero, [&] {
            released.push_back(1);
            CHECK(queue.pending() == 0);
            queue.push(zero, [&] { released.push_back(2); });
            queue.collect(zero);
            queue.defer([&] { released.push_back(4); });
        });
        queue.push(zero, [&] { released.push_back(3); });
        queue.collect(zero);
        CHECK((released == std::vector<int>{1, 2, 3}));
        CHECK(queue.pending() == 1);
        queue.seal(zero);
        queue.collect(zero);
        CHECK((released == std::vector<int>{1, 2, 3, 4}));

        struct captured_resource {
            retire_queue& queue;
            std::vector<int>& released;
            ~captured_resource() {
                // Destruction must also be outside the queue mutex, after all callbacks in the collection ran.
                CHECK((released == std::vector<int>{1, 2, 3, 4, 5, 6}));
                CHECK(queue.pending() == 0);
                queue.defer([&out = released] { out.push_back(7); });
            }
        };
        auto resource = std::make_shared<captured_resource>(queue, released);
        queue.push(zero, [resource, &released] { released.push_back(5); });
        queue_serials other{};
        other[0] = 1;
        queue.push(other, [&released] { released.push_back(6); });
        resource.reset();
        queue.collect(other);
        CHECK(queue.pending() == 1);
        queue.flush_all();
        CHECK((released == std::vector<int>{1, 2, 3, 4, 5, 6, 7}));
    }

    void test_concurrent_retirement() {
        retire_queue queue;
        std::atomic<int> released{0};
        std::atomic<int> active{4};
        std::vector<std::thread> producers;
        for (int thread = 0; thread < 4; thread++)
            producers.emplace_back([&, thread] {
                for (int i = 0; i < 3000; i++) {
                    queue_serials serials{};
                    serials[static_cast<size_t>(thread) % queue_count] = static_cast<uint64_t>(i % 9);
                    if (i & 1)
                        queue.push(serials, [&] { released.fetch_add(1, std::memory_order_relaxed); });
                    else
                        queue.defer([&] { released.fetch_add(1, std::memory_order_relaxed); });
                }
                active.fetch_sub(1, std::memory_order_release);
            });
        queue_serials done{};
        done.fill(9);
        while (active.load(std::memory_order_acquire)) {
            queue.seal(done);
            queue.collect(done);
            std::this_thread::yield();
        }
        for (auto& thread : producers)
            thread.join();
        queue.flush_all();
        CHECK(released == 12000);
        CHECK(queue.pending() == 0);
    }

    size_t linear_find(std::span<const bind_layout_entry> entries, uint32_t binding) {
        size_t result = binding_lookup::missing;
        for (size_t i = 0; i < entries.size(); i++)
            if (entries[i].binding == binding)
                result = i;
        return result;
    }

    void test_binding_lookup() {
        binding_lookup lookup;
        CHECK(lookup.find(0) == binding_lookup::missing);
        CHECK(lookup.find(UINT32_MAX) == binding_lookup::missing);
        std::mt19937 random(0x34a1b507);
        for (int run = 0; run < 1000; run++) {
            std::vector<bind_layout_entry> entries(random() % 150);
            for (auto& entry : entries) {
                switch (random() % 4) {
                case 0:
                    entry.binding = UINT32_MAX;
                    break;
                case 1:
                    entry.binding = random() % 64;
                    break;
                case 2:
                    entry.binding = 64 + random() % 100;
                    break;
                default:
                    entry.binding = random();
                    break;
                }
                entry.count = random() % 8; // Array elements share the layout entry, regardless of count.
            }
            lookup.assign(entries);
            for (const auto& entry : entries)
                CHECK(lookup.find(entry.binding) == linear_find(entries, entry.binding));
            for (uint32_t binding = 0; binding < 200; binding++)
                CHECK(lookup.find(binding) == linear_find(entries, binding));
            CHECK(lookup.find(UINT32_MAX) == linear_find(entries, UINT32_MAX));
        }
        lookup.assign({});
        CHECK(lookup.find(0) == binding_lookup::missing);
        CHECK(lookup.find(UINT32_MAX) == binding_lookup::missing);
    }
}

int main() {
    test_retirement_order();
    test_retirement_differential();
    test_reentrant_release();
    test_concurrent_retirement();
    test_binding_lookup();
    std::printf("conduit common CPU tests: %s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
