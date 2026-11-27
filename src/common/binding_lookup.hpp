#pragma once
#include <conduit/types.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace conduit {
    // A layout is immutable and reused by many groups. Index it once instead of scanning every layout entry
    // for every group entry. Small binding numbers need one array access; arbitrary sparse numbers use a
    // sorted list whose storage is bounded by the number of entries, never by the largest binding number.
    class binding_lookup {
    public:
        static constexpr size_t missing = static_cast<size_t>(-1);

        void assign(std::span<const bind_layout_entry> entries) {
            constexpr size_t direct_limit = 128;
            size_t direct_size = 0;
            for (const auto& entry : entries)
                if (entry.binding < direct_limit)
                    direct_size = std::max(direct_size, static_cast<size_t>(entry.binding) + 1);
            direct_.assign(direct_size, missing);
            sparse_.clear();
            for (size_t i = 0; i < entries.size(); i++) {
                const uint32_t binding = entries[i].binding;
                if (binding < direct_.size())
                    direct_[binding] = i;
                else
                    sparse_.emplace_back(binding, i);
            }
            std::sort(sparse_.begin(), sparse_.end());
            // Keep the original full-scan result for duplicate bindings: the last layout entry wins.
            size_t kept = 0;
            for (const auto& entry : sparse_) {
                if (kept && sparse_[kept - 1].first == entry.first)
                    sparse_[kept - 1].second = entry.second;
                else
                    sparse_[kept++] = entry;
            }
            sparse_.resize(kept);
        }

        size_t find(uint32_t binding) const {
            if (binding < direct_.size())
                return direct_[binding];
            const auto it = std::lower_bound(sparse_.begin(), sparse_.end(), binding,
                                             [](const auto& entry, uint32_t value) { return entry.first < value; });
            return it != sparse_.end() && it->first == binding ? it->second : missing;
        }

    private:
        std::vector<size_t> direct_;
        std::vector<std::pair<uint32_t, size_t>> sparse_;
    };
}
