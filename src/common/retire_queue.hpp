#pragma once
#include <conduit/types.hpp>

#include <array>
#include <functional>
#include <list>
#include <mutex>
#include <vector>

namespace conduit {
    using queue_serials = std::array<uint64_t, queue_count>;

    // released once every queue has finished the work submitted before destroy(); defer() holds a release back
    // until seal() stamps it with the serials of the frame it was destroyed in, so a resource destroyed while
    // a command list that uses it is still being recorded outlives that list
    class retire_queue {
    public:
        void push(const queue_serials& submitted, std::function<void()> release) {
            std::lock_guard lock(mutex_);
            if (items_.empty() || items_.back().serials != submitted)
                append_batch(submitted);
            items_.back().releases.push_back(std::move(release));
            pending_++;
        }

        void defer(std::function<void()> release) {
            std::lock_guard lock(mutex_);
            open_.push_back(std::move(release));
            pending_++;
        }

        void seal(const queue_serials& submitted) {
            std::lock_guard lock(mutex_);
            if (open_.empty())
                return;
            // Every release made during this frame has the same fence requirements. Keep those requirements
            // once for the whole frame, and exchange buffers with a completed batch instead of copying callbacks.
            append_batch(submitted);
            items_.back().releases.swap(open_);
        }

        void collect(const queue_serials& completed) {
            std::list<batch> running;
            {
                std::lock_guard lock(mutex_);
                // Do not stop at the first unfinished batch: push() can supply unrelated, nonmonotonic serials
                // from different queues. A ready batch must never wait behind an unfinished one.
                for (auto it = items_.begin(); it != items_.end();) {
                    auto current = it++;
                    if (covered(current->serials, completed)) {
                        pending_ -= current->releases.size();
                        running.splice(running.end(), items_, current);
                    }
                }
            }
            invoke(running);
            recycle(running);
        }

        void flush_all() {
            std::list<batch> all;
            std::vector<std::function<void()>> open;
            {
                std::lock_guard lock(mutex_);
                all.splice(all.end(), items_);
                open.swap(open_);
                pending_ = 0;
            }
            invoke(all);
            for (auto& release : open)
                release();
            open.clear();
            recycle(all);
        }

        size_t pending() const {
            std::lock_guard lock(mutex_);
            return pending_;
        }

    private:
        struct batch {
            queue_serials serials;
            std::vector<std::function<void()>> releases;
        };

        // mutex_ is held by the caller. Recycled batches contain no callbacks, only reusable storage.
        void append_batch(const queue_serials& serials) {
            if (spare_.empty())
                items_.emplace_back();
            else
                items_.splice(items_.end(), spare_, spare_.begin());
            items_.back().serials = serials;
        }

        static void invoke(std::list<batch>& batches) {
            for (auto& b : batches)
                for (auto& release : b.releases)
                    release();
        }

        void recycle(std::list<batch>& batches) {
            if (batches.empty())
                return;
            // Both invocation and captured-resource destruction stay outside mutex_: a release may retire
            // another resource, collect another ready batch, or destroy an object that does either. Keep every
            // callback alive until all callbacks in this collection have run, as with the unbatched queue.
            for (auto& b : batches)
                b.releases.clear();
            std::lock_guard lock(mutex_);
            // Bound retained scratch storage even for clients that push thousands of distinct fence values.
            constexpr size_t max_spares = 8;
            while (!batches.empty() && spare_.size() < max_spares)
                spare_.splice(spare_.end(), batches, batches.begin());
            // Any excess empty batches are destroyed by the caller after this lock is released.
        }

        static bool covered(const queue_serials& need, const queue_serials& done) {
            for (size_t i = 0; i < queue_count; i++)
                if (done[i] < need[i])
                    return false;
            return true;
        }

        mutable std::mutex mutex_;
        std::list<batch> items_, spare_;
        std::vector<std::function<void()>> open_;
        size_t pending_ = 0;
    };
}
