#pragma once
#include <conduit/types.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace conduit {
    enum class resource_kind : uint8_t {
        uniform_buffer,
        storage_buffer,
        storage_buffer_rw,
        sampled_texture,
        storage_texture,
        sampler
    };

    struct shader_resource {
        uint32_t set = 0;
        uint32_t binding = 0;
        resource_kind kind = resource_kind::uniform_buffer;
        texture_dim dim = texture_dim::tex2d;
        bool comparison = false;
        bool depth = false;
        std::string name;
        size_t clause_begin = 0;
        size_t clause_end = 0;
        char register_class = 0;
        uint32_t native_index = 0;
    };

    struct shader_interface {
        std::vector<shader_resource> resources;
        bool uses_push_constants = false;

        const shader_resource* find(uint32_t set, uint32_t binding, resource_kind kind) const;
    };

    shader_interface reflect_shader(shader_format format, std::string_view source);
}
