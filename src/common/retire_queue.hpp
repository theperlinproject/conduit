#pragma once
#include <conduit/types.hpp>

#include <array>
#include <deque>
#include <functional>
#include <mutex>

namespace conduit {
    using queue_serials = std::array<uint64_t, queue_count>;

    // released once every queue has finished the work submitted before destroy()
    class retire_queue {
    public:
        void push(const queue_serials& submitted, std::function<void()> release) {
            std::lock_guard lock(mutex_);
            items_.push_back({submitted, std::move(release)});
        }

        void collect(const queue_serials& completed) {
            std::vector<std::function<void()>> ready;
            {
                std::lock_guard lock(mutex_);
                while (!items_.empty() && covered(items_.front().serials, completed)) {
                    ready.push_back(std::move(items_.front().release));
                    items_.pop_front();
                }
                for (auto it = items_.begin(); it != items_.end();) {
                    if (covered(it->serials, completed)) {
                        ready.push_back(std::move(it->release));
                        it = items_.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            for (auto& fn : ready)
                fn();
        }

        void flush_all() {
            std::deque<retire_queue::item> all;
            {
                std::lock_guard lock(mutex_);
                all.swap(items_);
            }
            for (auto& item : all)
                item.release();
        }

        size_t pending() const {
            std::lock_guard lock(mutex_);
            return items_.size();
        }

    private:
        struct item {
            queue_serials serials;
            std::function<void()> release;
        };

        static bool covered(const queue_serials& need, const queue_serials& done) {
            for (size_t i = 0; i < queue_count; i++)
                if (done[i] < need[i])
                    return false;
            return true;
        }

        mutable std::mutex mutex_;
        std::deque<item> items_;
    };
}
