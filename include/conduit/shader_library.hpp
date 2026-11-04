#pragma once
#include <conduit/types.hpp>

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace conduit {
    enum class shader_target : uint8_t { spirv, dxil, hlsl, msl, wgsl, count };

    struct uniform_member {
        std::string name;
        uint32_t offset = 0;
        uint32_t bytes = 0;
        uint32_t elements = 1;
    };

    // texture n is bound at set 1 binding 2n and its sampler at binding 2n + 1
    struct texture_slot {
        std::string name;
        uint32_t slot = 0;
        bool array = false;
        bool unfilterable = false;
    };

    struct program_info {
        std::string name;
        uint32_t uniform_bytes = 0;
        std::vector<uniform_member> uniforms;

        const uniform_member* find_uniform(std::string_view member) const {
            for (const uniform_member& u : uniforms)
                if (u.name == member)
                    return &u;
            return nullptr;
        }
    };

    class shader_library {
    public:
        bool load(std::span<const uint8_t> pack, std::string* error = nullptr);

        const std::vector<program_info>& programs() const { return programs_; }
        const std::vector<texture_slot>& texture_slots() const { return slots_; }
        const texture_slot* find_texture(std::string_view name) const;
        const program_info* find(std::string_view name) const;
        std::span<const uint8_t> code(const program_info& program, shader_stage stage, shader_target target) const;

        static shader_target target_for(api api);
        static shader_format format_for(shader_target target);
        static const char* entry_for(shader_stage stage, shader_target target);

    private:
        struct blob {
            uint8_t stage = 0;
            uint8_t target = 0;
            uint32_t offset = 0;
            uint32_t bytes = 0;
        };
        std::span<const uint8_t> data_;
        std::vector<program_info> programs_;
        std::vector<texture_slot> slots_;
        std::vector<std::vector<blob>> blobs_;
    };
}
