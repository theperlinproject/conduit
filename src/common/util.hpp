#pragma once
#include <conduit/types.hpp>

#include <algorithm>

namespace conduit {
    template <class To, class From> To function_cast(From from) {
        return reinterpret_cast<To>(reinterpret_cast<void (*)()>(from));
    }

    constexpr uint64_t align_up(uint64_t value, uint64_t alignment) {
        return (value + alignment - 1) / alignment * alignment;
    }

    constexpr uint32_t mip_extent(uint32_t extent, uint32_t mip) {
        return (std::max)(1u, extent >> mip);
    }

    inline uint32_t layer_count_of(const texture_desc& desc) {
        switch (desc.dim) {
        case texture_dim::tex2d_array:
            return (std::max)(1u, desc.depth_or_layers);
        case texture_dim::cube:
            return 6;
        default:
            return 1;
        }
    }

    inline uint32_t depth_of(const texture_desc& desc) {
        return desc.dim == texture_dim::tex3d ? (std::max)(1u, desc.depth_or_layers) : 1u;
    }

    inline subresource_range resolve_range(const subresource_range& range, const texture_desc& desc) {
        subresource_range out = range;
        const uint32_t mips = (std::max)(1u, desc.mips);
        const uint32_t layers = layer_count_of(desc);
        out.base_mip = (std::min)(out.base_mip, mips - 1);
        out.base_layer = (std::min)(out.base_layer, layers - 1);
        out.mip_count = out.mip_count ? (std::min)(out.mip_count, mips - out.base_mip) : mips - out.base_mip;
        out.layer_count =
            out.layer_count ? (std::min)(out.layer_count, layers - out.base_layer) : layers - out.base_layer;
        return out;
    }

    inline uint32_t row_bytes(conduit::format format, uint32_t width) {
        const format_info& info = get_format_info(format);
        return (width + info.block_width - 1) / info.block_width * info.bytes_per_block;
    }

    inline uint32_t row_blocks(conduit::format format, uint32_t height) {
        const format_info& info = get_format_info(format);
        return (height + info.block_height - 1) / info.block_height;
    }

    inline uint32_t mip_levels_for(uint32_t width, uint32_t height) {
        uint32_t n = 1;
        while ((std::max(width, height) >> n) > 0)
            n++;
        return n;
    }
}
