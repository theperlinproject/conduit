#include <conduit/shader_library.hpp>

#include <cstring>

namespace conduit {
    namespace {
        constexpr uint32_t magic = 0x48534443;
        constexpr uint32_t version = 1;

        struct reader {
            std::span<const uint8_t> bytes;
            size_t at = 0;
            bool ok = true;

            template <class T> T read() {
                T value{};
                if (at + sizeof(T) > bytes.size()) {
                    ok = false;
                    return value;
                }
                std::memcpy(&value, bytes.data() + at, sizeof(T));
                at += sizeof(T);
                return value;
            }

            std::string string() {
                const uint16_t length = read<uint16_t>();
                if (!ok || at + length > bytes.size()) {
                    ok = false;
                    return {};
                }
                std::string out(reinterpret_cast<const char*>(bytes.data() + at), length);
                at += length;
                return out;
            }
        };

        uint8_t stage_id(shader_stage stage) {
            switch (stage) {
            case shader_stage::vertex:
                return 0;
            case shader_stage::fragment:
                return 1;
            default:
                return 2;
            }
        }
    }

    bool shader_library::load(std::span<const uint8_t> pack, std::string* error) {
        auto fail = [&](const char* message) {
            if (error)
                *error = message;
            programs_.clear();
            slots_.clear();
            blobs_.clear();
            data_ = {};
            return false;
        };
        reader header{pack};
        if (header.read<uint32_t>() != magic)
            return fail("not a shader pack");
        if (header.read<uint32_t>() != version)
            return fail("unsupported shader pack version");
        const uint32_t index_bytes = header.read<uint32_t>();
        header.read<uint32_t>();
        if (!header.ok || header.at + index_bytes > pack.size())
            return fail("truncated shader pack");

        reader r{pack.subspan(header.at, index_bytes)};
        const std::span<const uint8_t> area = pack.subspan(header.at + index_bytes);
        slots_.clear();
        const uint16_t slot_count = r.read<uint16_t>();
        for (uint16_t i = 0; i < slot_count && r.ok; i++) {
            texture_slot slot;
            slot.name = r.string();
            slot.slot = r.read<uint8_t>();
            const uint8_t flags = r.read<uint8_t>();
            slot.array = (flags & 1) != 0;
            slot.unfilterable = (flags & 2) != 0;
            slots_.push_back(std::move(slot));
        }
        const uint32_t count = r.read<uint32_t>();
        programs_.clear();
        blobs_.clear();
        for (uint32_t p = 0; p < count && r.ok; p++) {
            program_info info;
            info.name = r.string();
            info.uniform_bytes = r.read<uint32_t>();
            const uint16_t uniform_count = r.read<uint16_t>();
            for (uint16_t u = 0; u < uniform_count && r.ok; u++) {
                uniform_member m;
                m.name = r.string();
                m.offset = r.read<uint32_t>();
                m.bytes = r.read<uint32_t>();
                m.elements = r.read<uint16_t>();
                info.uniforms.push_back(std::move(m));
            }
            std::vector<blob> list;
            const uint8_t stage_count = r.read<uint8_t>();
            for (uint8_t s = 0; s < stage_count && r.ok; s++) {
                const uint8_t stage = r.read<uint8_t>();
                const uint8_t target_count = r.read<uint8_t>();
                for (uint8_t t = 0; t < target_count && r.ok; t++) {
                    blob b;
                    b.stage = stage;
                    b.target = r.read<uint8_t>();
                    b.offset = r.read<uint32_t>();
                    b.bytes = r.read<uint32_t>();
                    if (static_cast<uint64_t>(b.offset) + b.bytes > area.size())
                        return fail("shader blob outside the pack");
                    list.push_back(b);
                }
            }
            programs_.push_back(std::move(info));
            blobs_.push_back(std::move(list));
        }
        if (!r.ok)
            return fail("truncated shader pack index");
        data_ = area;
        return true;
    }

    const texture_slot* shader_library::find_texture(std::string_view name) const {
        for (const texture_slot& s : slots_)
            if (s.name == name)
                return &s;
        return nullptr;
    }

    const program_info* shader_library::find(std::string_view name) const {
        for (const program_info& p : programs_)
            if (p.name == name)
                return &p;
        return nullptr;
    }

    std::span<const uint8_t> shader_library::code(const program_info& program, shader_stage stage,
                                                  shader_target target) const {
        const size_t index = static_cast<size_t>(&program - programs_.data());
        if (index >= programs_.size())
            return {};
        for (const blob& b : blobs_[index])
            if (b.stage == stage_id(stage) && b.target == static_cast<uint8_t>(target))
                return data_.subspan(b.offset, b.bytes);
        return {};
    }

    shader_target shader_library::target_for(conduit::api api) {
        switch (api) {
        case conduit::api::vulkan:
            return shader_target::spirv;
        case conduit::api::d3d12:
            return shader_target::dxil;
        case conduit::api::d3d11:
            return shader_target::hlsl;
        case conduit::api::metal:
            return shader_target::msl;
        case conduit::api::webgpu:
            return shader_target::wgsl;
        // OpenGL ES translates the SPIR-V itself when it builds a pipeline
        case conduit::api::gles:
            return shader_target::spirv;
        default:
            return shader_target::spirv;
        }
    }

    shader_format shader_library::format_for(shader_target target) {
        switch (target) {
        case shader_target::dxil:
            return shader_format::dxil;
        case shader_target::hlsl:
            return shader_format::hlsl;
        case shader_target::msl:
            return shader_format::msl;
        case shader_target::wgsl:
            return shader_format::wgsl;
        default:
            return shader_format::spirv;
        }
    }

    const char* shader_library::entry_for(shader_stage stage, shader_target target) {
        if (target == shader_target::spirv || target == shader_target::dxil)
            return "main";
        switch (stage) {
        case shader_stage::vertex:
            return "vsMain";
        case shader_stage::fragment:
            return "fsMain";
        default:
            return "csMain";
        }
    }
}
