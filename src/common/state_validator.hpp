#pragma once
#include <conduit/types.hpp>

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace conduit {
    class state_validator {
    public:
        void on_create(texture_handle handle, const texture_desc& desc, state initial = state::undefined);
        void on_create(buffer_handle handle, const buffer_desc& desc);
        void on_destroy(texture_handle handle);
        void on_destroy(buffer_handle handle);

        void barrier(const texture_barrier& barrier);
        void barrier(const buffer_barrier& barrier);

        void expect_texture(texture_handle handle, const subresource_range& range, state want, const char* what);
        void expect_buffer(buffer_handle handle, state want, const char* what);

        struct bound_resource {
            texture_handle texture;
            buffer_handle buffer;
            subresource_range range;
            state want = state::undefined;
        };
        void expect_all(const std::vector<bound_resource>& resources, const char* what);

        size_t error_count() const;
        std::vector<std::string> take_errors();

    private:
        struct texture_info {
            texture_desc desc;
            std::vector<state> states;
            uint32_t layers = 1;
        };
        struct buffer_info {
            buffer_desc desc;
            conduit::state state = conduit::state::undefined;
        };

        void report(std::string message);
        static std::string state_name(state state);
        static uint32_t layer_count(const texture_desc& desc);

        mutable std::mutex mutex_;
        std::unordered_map<uint32_t, texture_info> textures_;
        std::unordered_map<uint32_t, buffer_info> buffers_;
        std::vector<std::string> errors_;
        size_t total_ = 0;
    };
}
