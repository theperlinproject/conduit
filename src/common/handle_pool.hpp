#pragma once
#include <conduit/types.hpp>

#include <atomic>
#include <cassert>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace conduit {
    // slots live in fixed pages that never move: get() is lock free, alloc and free take a mutex
    template <class Tag, class T> class handle_pool {
    public:
        using h = handle<Tag>;

        handle_pool() {
            for (auto& page : pages_)
                page.store(nullptr, std::memory_order_relaxed);
        }
        ~handle_pool() {
            for (auto& page : pages_)
                delete[] page.load(std::memory_order_relaxed);
        }
        handle_pool(const handle_pool&) = delete;
        handle_pool& operator=(const handle_pool&) = delete;

        h alloc(T value) {
            std::lock_guard lock(mutex_);
            uint32_t index;
            if (!free_list_.empty()) {
                index = free_list_.back();
                free_list_.pop_back();
            } else {
                index = next_++;
                assert(index < page_size * max_pages);
                const uint32_t page = index / page_size;
                if (!pages_[page].load(std::memory_order_relaxed))
                    pages_[page].store(new handle_pool::slot[page_size], std::memory_order_release);
            }
            handle_pool::slot& slot = slot_at(index);
            slot.value = std::move(value);
            slot.alive.store(true, std::memory_order_release);
            live_++;
            return h{((slot.generation & 0xFFu) << 24) | (index + 1)};
        }

        T* get(h handle) const {
            if (!handle)
                return nullptr;
            const uint32_t index = handle.index() - 1;
            if (index >= page_size * max_pages)
                return nullptr;
            handle_pool::slot* page = pages_[index / page_size].load(std::memory_order_acquire);
            if (!page)
                return nullptr;
            handle_pool::slot& slot = page[index % page_size];
            if (!slot.alive.load(std::memory_order_acquire) || (slot.generation & 0xFFu) != handle.generation())
                return nullptr;
            return &slot.value;
        }

        bool free(h handle, T* out = nullptr) {
            std::lock_guard lock(mutex_);
            T* value = get(handle);
            if (!value)
                return false;
            handle_pool::slot& slot = slot_at(handle.index() - 1);
            if (out)
                *out = std::move(*value);
            slot.value = T{};
            slot.alive.store(false, std::memory_order_release);
            slot.generation++;
            free_list_.push_back(handle.index() - 1);
            live_--;
            return true;
        }

        uint32_t live() const {
            std::lock_guard lock(mutex_);
            return live_;
        }

        template <class Fn> void for_each(Fn&& fn) const {
            std::lock_guard lock(mutex_);
            for (uint32_t i = 0; i < next_; i++) {
                handle_pool::slot& slot = slot_at(i);
                if (slot.alive.load(std::memory_order_acquire))
                    fn(h{((slot.generation & 0xFFu) << 24) | (i + 1)}, slot.value);
            }
        }

    private:
        static constexpr uint32_t page_size = 1024;
        static constexpr uint32_t max_pages = 4096;

        struct slot {
            T value{};
            std::atomic<bool> alive{false};
            uint32_t generation = 1;
        };

        slot& slot_at(uint32_t index) const {
            return pages_[index / page_size].load(std::memory_order_relaxed)[index % page_size];
        }

        mutable std::mutex mutex_;
        std::atomic<slot*> pages_[max_pages];
        std::vector<uint32_t> free_list_;
        uint32_t next_ = 0;
        uint32_t live_ = 0;
    };
}
