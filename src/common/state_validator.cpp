#include "common/state_validator.hpp"

#include <algorithm>

namespace conduit {
    uint32_t state_validator::layer_count(const texture_desc& desc) {
        switch (desc.dim) {
        case texture_dim::tex2d_array:
            return (std::max)(1u, desc.depth_or_layers);
        case texture_dim::cube:
            return 6;
        default:
            return 1;
        }
    }

    std::string state_validator::state_name(conduit::state state) {
        if (state == conduit::state::undefined)
            return "Undefined";
        static const struct {
            conduit::state bit;
            const char* name;
        } names[] = {
            {conduit::state::copy_src, "CopySrc"},
            {conduit::state::copy_dst, "CopyDst"},
            {conduit::state::vertex_buffer, "VertexBuffer"},
            {conduit::state::index_buffer, "IndexBuffer"},
            {conduit::state::uniform_buffer, "UniformBuffer"},
            {conduit::state::indirect_argument, "IndirectArgument"},
            {conduit::state::shader_read_fragment, "ShaderReadFragment"},
            {conduit::state::shader_read_other, "ShaderReadOther"},
            {conduit::state::shader_write, "ShaderWrite"},
            {conduit::state::color_target, "ColorTarget"},
            {conduit::state::depth_write, "DepthWrite"},
            {conduit::state::depth_read, "DepthRead"},
            {conduit::state::present, "Present"},
            {conduit::state::resolve_src, "ResolveSrc"},
            {conduit::state::resolve_dst, "ResolveDst"},
        };
        std::string out;
        for (const auto& entry : names)
            if (has(state, entry.bit)) {
                if (!out.empty())
                    out += '|';
                out += entry.name;
            }
        return out;
    }

    void state_validator::report(std::string message) {
        total_++;
        if (errors_.size() < 64)
            errors_.push_back(std::move(message));
    }

    void state_validator::on_create(texture_handle handle, const texture_desc& desc, state initial) {
        std::lock_guard lock(mutex_);
        texture_info info;
        info.desc = desc;
        info.layers = layer_count(desc);
        info.states.assign(static_cast<size_t>(std::max(1u, desc.mips)) * info.layers, initial);
        textures_[handle.value] = std::move(info);
    }

    void state_validator::on_create(buffer_handle handle, const buffer_desc& desc) {
        std::lock_guard lock(mutex_);
        buffers_[handle.value] = {desc, state::undefined};
    }

    void state_validator::on_destroy(texture_handle handle) {
        std::lock_guard lock(mutex_);
        textures_.erase(handle.value);
    }

    void state_validator::on_destroy(buffer_handle handle) {
        std::lock_guard lock(mutex_);
        buffers_.erase(handle.value);
    }

    void state_validator::barrier(const texture_barrier& barrier) {
        std::lock_guard lock(mutex_);
        auto it = textures_.find(barrier.texture.value);
        if (it == textures_.end()) {
            report("barrier on a destroyed or unknown texture");
            return;
        }
        texture_info& info = it->second;
        const uint32_t mips = (std::max)(1u, info.desc.mips);
        const uint32_t m0 = barrier.range.base_mip;
        const uint32_t m1 = barrier.range.mip_count ? (std::min)(mips, m0 + barrier.range.mip_count) : mips;
        const uint32_t l0 = barrier.range.base_layer;
        const uint32_t l1 =
            barrier.range.layer_count ? (std::min)(info.layers, l0 + barrier.range.layer_count) : info.layers;
        for (uint32_t m = m0; m < m1; m++)
            for (uint32_t l = l0; l < l1; l++) {
                conduit::state& state = info.states[static_cast<size_t>(l) * mips + m];
                if (barrier.before != conduit::state::undefined && barrier.before != state) {
                    report(std::string("texture '") + (info.desc.name ? info.desc.name : "?") + "' mip " +
                           std::to_string(m) + " layer " + std::to_string(l) + ": barrier says it was " +
                           state_name(barrier.before) + " but it was left " + state_name(state));
                    state = barrier.after;
                    continue;
                }
                state = barrier.after;
            }
    }

    void state_validator::barrier(const buffer_barrier& barrier) {
        std::lock_guard lock(mutex_);
        auto it = buffers_.find(barrier.buffer.value);
        if (it == buffers_.end()) {
            report("barrier on a destroyed or unknown buffer");
            return;
        }
        buffer_info& info = it->second;
        if (info.desc.memory == memory_class::upload || info.desc.memory == memory_class::readback)
            return;
        if (barrier.before != state::undefined && info.state != state::undefined && barrier.before != info.state)
            report(std::string("buffer '") + (info.desc.name ? info.desc.name : "?") + "': barrier says it was " +
                   state_name(barrier.before) + " but it was left " + state_name(info.state));
        info.state = barrier.after;
    }

    void state_validator::expect_texture(texture_handle handle, const subresource_range& range, conduit::state want,
                                         const char* what) {
        std::lock_guard lock(mutex_);
        auto it = textures_.find(handle.value);
        if (it == textures_.end()) {
            report(std::string(what) + ": unknown texture");
            return;
        }
        texture_info& info = it->second;
        const uint32_t mips = (std::max)(1u, info.desc.mips);
        const uint32_t m0 = range.base_mip;
        const uint32_t m1 = range.mip_count ? (std::min)(mips, m0 + range.mip_count) : mips;
        const uint32_t l0 = range.base_layer;
        const uint32_t l1 = range.layer_count ? (std::min)(info.layers, l0 + range.layer_count) : info.layers;
        for (uint32_t m = m0; m < m1; m++)
            for (uint32_t l = l0; l < l1; l++) {
                const conduit::state state = info.states[static_cast<size_t>(l) * mips + m];
                if ((static_cast<uint32_t>(state) & static_cast<uint32_t>(want)) == 0) {
                    report(std::string(what) + ": texture '" + (info.desc.name ? info.desc.name : "?") + "' mip " +
                           std::to_string(m) + " layer " + std::to_string(l) + " is " + state_name(state) + ", needs " +
                           state_name(want));
                    return;
                }
            }
    }

    void state_validator::expect_buffer(buffer_handle handle, state want, const char* what) {
        std::lock_guard lock(mutex_);
        auto it = buffers_.find(handle.value);
        if (it == buffers_.end()) {
            report(std::string(what) + ": unknown buffer");
            return;
        }
        const buffer_info& info = it->second;
        if (info.desc.memory == memory_class::upload || info.desc.memory == memory_class::readback)
            return;
        if (info.state == state::undefined)
            return;
        if ((static_cast<uint32_t>(info.state) & static_cast<uint32_t>(want)) == 0)
            report(std::string(what) + ": buffer '" + (info.desc.name ? info.desc.name : "?") + "' is " +
                   state_name(info.state) + ", needs " + state_name(want));
    }

    void state_validator::expect_all(const std::vector<bound_resource>& resources, const char* what) {
        for (const bound_resource& r : resources) {
            if (r.texture)
                expect_texture(r.texture, r.range, r.want, what);
            else if (r.buffer)
                expect_buffer(r.buffer, r.want, what);
        }
    }

    size_t state_validator::error_count() const {
        std::lock_guard lock(mutex_);
        return total_;
    }

    std::vector<std::string> state_validator::take_errors() {
        std::lock_guard lock(mutex_);
        std::vector<std::string> out;
        out.swap(errors_);
        total_ = 0;
        return out;
    }
}
