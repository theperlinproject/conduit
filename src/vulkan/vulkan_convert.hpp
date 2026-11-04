#pragma once
#include <conduit/types.hpp>

#include <volk.h>

namespace conduit::vk {
    inline VkFormat base_format(format f) {
        switch (f) {
        case format::r8_unorm:
            return VK_FORMAT_R8_UNORM;
        case format::rg8_unorm:
            return VK_FORMAT_R8G8_UNORM;
        case format::rgba8_unorm:
            return VK_FORMAT_R8G8B8A8_UNORM;
        case format::rgba8_srgb:
            return VK_FORMAT_R8G8B8A8_SRGB;
        case format::bgra8_unorm:
            return VK_FORMAT_B8G8R8A8_UNORM;
        case format::bgra8_srgb:
            return VK_FORMAT_B8G8R8A8_SRGB;
        case format::r8_uint:
            return VK_FORMAT_R8_UINT;
        case format::r16_uint:
            return VK_FORMAT_R16_UINT;
        case format::r32_uint:
            return VK_FORMAT_R32_UINT;
        case format::rg32_uint:
            return VK_FORMAT_R32G32_UINT;
        case format::rgba32_uint:
            return VK_FORMAT_R32G32B32A32_UINT;
        case format::r16_float:
            return VK_FORMAT_R16_SFLOAT;
        case format::rg16_float:
            return VK_FORMAT_R16G16_SFLOAT;
        case format::rgba16_float:
            return VK_FORMAT_R16G16B16A16_SFLOAT;
        case format::r32_float:
            return VK_FORMAT_R32_SFLOAT;
        case format::rg32_float:
            return VK_FORMAT_R32G32_SFLOAT;
        case format::rgb32_float:
            return VK_FORMAT_R32G32B32_SFLOAT;
        case format::rgba32_float:
            return VK_FORMAT_R32G32B32A32_SFLOAT;
        case format::rgb10a2_unorm:
            return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        case format::r11g11b10_float:
            return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        case format::rg16_unorm:
            return VK_FORMAT_R16G16_UNORM;
        case format::rgba16_unorm:
            return VK_FORMAT_R16G16B16A16_UNORM;
        case format::d16_unorm:
            return VK_FORMAT_D16_UNORM;
        case format::d32_float:
            return VK_FORMAT_D32_SFLOAT;
        case format::d24_unorm_s8_uint:
            return VK_FORMAT_D24_UNORM_S8_UINT;
        case format::d32_float_s8_uint:
            return VK_FORMAT_D32_SFLOAT_S8_UINT;
        case format::bc1_unorm:
            return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case format::bc3_unorm:
            return VK_FORMAT_BC3_UNORM_BLOCK;
        case format::bc7_unorm:
            return VK_FORMAT_BC7_UNORM_BLOCK;
        default:
            return VK_FORMAT_UNDEFINED;
        }
    }

    inline VkImageAspectFlags aspect_of(format f) {
        const format_info& info = get_format_info(f);
        if (info.depth)
            return info.stencil ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_DEPTH_BIT;
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }

    struct state_use {
        VkPipelineStageFlags2 stages = 0;
        VkAccessFlags2 access = 0;
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    constexpr VkPipelineStageFlags2 graphics_only_stages =
        VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    constexpr VkAccessFlags2 graphics_only_access =
        VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    inline state_use use_of(state s, queue_type queue, bool depth_format) {
        state_use use;
        if (s == state::undefined)
            return use;
        VkImageLayout layouts[16];
        int layout_count = 0;
        auto add = [&](VkPipelineStageFlags2 st, VkAccessFlags2 ac, VkImageLayout lay) {
            use.stages |= st;
            use.access |= ac;
            layouts[layout_count++] = lay;
        };
        const VkImageLayout shader_read =
            depth_format ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        constexpr auto shaders = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        if (has(s, state::copy_src))
            add(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        if (has(s, state::copy_dst))
            add(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        if (has(s, state::vertex_buffer))
            add(VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT, VK_IMAGE_LAYOUT_GENERAL);
        if (has(s, state::index_buffer))
            add(VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT, VK_ACCESS_2_INDEX_READ_BIT, VK_IMAGE_LAYOUT_GENERAL);
        if (has(s, state::uniform_buffer))
            add(shaders, VK_ACCESS_2_UNIFORM_READ_BIT, VK_IMAGE_LAYOUT_GENERAL);
        if (has(s, state::indirect_argument))
            add(VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT, VK_IMAGE_LAYOUT_GENERAL);
        if (has(s, state::shader_read_fragment))
            add(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT, shader_read);
        if (has(s, state::shader_read_other))
            add(VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT, shader_read);
        if (has(s, state::shader_write))
            add(shaders, VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL);
        if (has(s, state::color_target))
            add(VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (has(s, state::depth_write))
            add(VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        if (has(s, state::depth_read))
            add(VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        if (has(s, state::present))
            add(0, 0, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        if (has(s, state::resolve_src))
            add(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        if (has(s, state::resolve_dst))
            add(VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        use.layout = layouts[0];
        for (int i = 1; i < layout_count; i++)
            if (layouts[i] != use.layout) {
                use.layout = VK_IMAGE_LAYOUT_GENERAL;
                break;
            }
        if (has(s, state::depth_read) && (has(s, state::shader_read_fragment) || has(s, state::shader_read_other)))
            use.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        if (queue == queue_type::compute) {
            if (use.stages & graphics_only_stages)
                use.stages = (use.stages & ~graphics_only_stages) | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            use.access &= ~graphics_only_access;
        } else if (queue == queue_type::copy) {
            use.stages = use.stages ? VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT : 0;
            use.access &= VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        }
        return use;
    }

    inline VkCompareOp to_vk(compare_op op) {
        switch (op) {
        case compare_op::never:
            return VK_COMPARE_OP_NEVER;
        case compare_op::less:
            return VK_COMPARE_OP_LESS;
        case compare_op::equal:
            return VK_COMPARE_OP_EQUAL;
        case compare_op::less_equal:
            return VK_COMPARE_OP_LESS_OR_EQUAL;
        case compare_op::greater:
            return VK_COMPARE_OP_GREATER;
        case compare_op::not_equal:
            return VK_COMPARE_OP_NOT_EQUAL;
        case compare_op::greater_equal:
            return VK_COMPARE_OP_GREATER_OR_EQUAL;
        default:
            return VK_COMPARE_OP_ALWAYS;
        }
    }

    inline VkStencilOp to_vk(stencil_op op) {
        switch (op) {
        case stencil_op::keep:
            return VK_STENCIL_OP_KEEP;
        case stencil_op::zero:
            return VK_STENCIL_OP_ZERO;
        case stencil_op::replace:
            return VK_STENCIL_OP_REPLACE;
        case stencil_op::increment_clamp:
            return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case stencil_op::decrement_clamp:
            return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case stencil_op::invert:
            return VK_STENCIL_OP_INVERT;
        case stencil_op::increment_wrap:
            return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        default:
            return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        }
    }

    inline VkBlendFactor to_vk(blend_factor f) {
        switch (f) {
        case blend_factor::zero:
            return VK_BLEND_FACTOR_ZERO;
        case blend_factor::one:
            return VK_BLEND_FACTOR_ONE;
        case blend_factor::src_color:
            return VK_BLEND_FACTOR_SRC_COLOR;
        case blend_factor::inv_src_color:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case blend_factor::src_alpha:
            return VK_BLEND_FACTOR_SRC_ALPHA;
        case blend_factor::inv_src_alpha:
            return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case blend_factor::dst_color:
            return VK_BLEND_FACTOR_DST_COLOR;
        case blend_factor::inv_dst_color:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case blend_factor::dst_alpha:
            return VK_BLEND_FACTOR_DST_ALPHA;
        case blend_factor::inv_dst_alpha:
            return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        default:
            return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        }
    }

    inline VkBlendOp to_vk(blend_op op) {
        switch (op) {
        case blend_op::add:
            return VK_BLEND_OP_ADD;
        case blend_op::subtract:
            return VK_BLEND_OP_SUBTRACT;
        case blend_op::reverse_subtract:
            return VK_BLEND_OP_REVERSE_SUBTRACT;
        case blend_op::minimum:
            return VK_BLEND_OP_MIN;
        default:
            return VK_BLEND_OP_MAX;
        }
    }

    inline VkSamplerAddressMode to_vk(address_mode m) {
        switch (m) {
        case address_mode::repeat:
            return VK_SAMPLER_ADDRESS_MODE_REPEAT;
        case address_mode::mirror:
            return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        case address_mode::clamp:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        default:
            return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        }
    }

    inline VkFilter to_vk(filter f) {
        return f == filter::nearest ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
    }

    inline VkShaderStageFlags to_vk(shader_stage s) {
        VkShaderStageFlags out = 0;
        if (s & shader_stage::vertex)
            out |= VK_SHADER_STAGE_VERTEX_BIT;
        if (s & shader_stage::fragment)
            out |= VK_SHADER_STAGE_FRAGMENT_BIT;
        if (s & shader_stage::compute)
            out |= VK_SHADER_STAGE_COMPUTE_BIT;
        return out;
    }

    inline VkImageViewType view_type_of(texture_dim dim) {
        switch (dim) {
        case texture_dim::tex2d:
            return VK_IMAGE_VIEW_TYPE_2D;
        case texture_dim::tex2d_array:
            return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        case texture_dim::cube:
            return VK_IMAGE_VIEW_TYPE_CUBE;
        default:
            return VK_IMAGE_VIEW_TYPE_3D;
        }
    }

    inline VkSampleCountFlagBits samples_of(uint32_t n) {
        switch (n) {
        case 2:
            return VK_SAMPLE_COUNT_2_BIT;
        case 4:
            return VK_SAMPLE_COUNT_4_BIT;
        case 8:
            return VK_SAMPLE_COUNT_8_BIT;
        case 16:
            return VK_SAMPLE_COUNT_16_BIT;
        default:
            return VK_SAMPLE_COUNT_1_BIT;
        }
    }
}
