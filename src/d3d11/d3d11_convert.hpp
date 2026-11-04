#pragma once
#include <conduit/types.hpp>

#include <d3d11_1.h>
#include <dxgiformat.h>

namespace conduit::d3d11 {
    struct format_map {
        DXGI_FORMAT resource = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT view = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT dsv = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT stencil_view = DXGI_FORMAT_UNKNOWN;
    };

    inline format_map map_format(format f) {
        format_map m;
        const auto simple = [&](DXGI_FORMAT x) { m.resource = m.view = x; };
        switch (f) {
        case format::r8_unorm:
            simple(DXGI_FORMAT_R8_UNORM);
            break;
        case format::rg8_unorm:
            simple(DXGI_FORMAT_R8G8_UNORM);
            break;
        case format::rgba8_unorm:
            m.resource = DXGI_FORMAT_R8G8B8A8_TYPELESS;
            m.view = DXGI_FORMAT_R8G8B8A8_UNORM;
            break;
        case format::rgba8_srgb:
            m.resource = DXGI_FORMAT_R8G8B8A8_TYPELESS;
            m.view = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            break;
        case format::bgra8_unorm:
            m.resource = DXGI_FORMAT_B8G8R8A8_TYPELESS;
            m.view = DXGI_FORMAT_B8G8R8A8_UNORM;
            break;
        case format::bgra8_srgb:
            m.resource = DXGI_FORMAT_B8G8R8A8_TYPELESS;
            m.view = DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
            break;
        case format::r8_uint:
            simple(DXGI_FORMAT_R8_UINT);
            break;
        case format::r16_uint:
            simple(DXGI_FORMAT_R16_UINT);
            break;
        case format::r32_uint:
            simple(DXGI_FORMAT_R32_UINT);
            break;
        case format::rg32_uint:
            simple(DXGI_FORMAT_R32G32_UINT);
            break;
        case format::rgba32_uint:
            simple(DXGI_FORMAT_R32G32B32A32_UINT);
            break;
        case format::r16_float:
            simple(DXGI_FORMAT_R16_FLOAT);
            break;
        case format::rg16_float:
            simple(DXGI_FORMAT_R16G16_FLOAT);
            break;
        case format::rgba16_float:
            simple(DXGI_FORMAT_R16G16B16A16_FLOAT);
            break;
        case format::r32_float:
            simple(DXGI_FORMAT_R32_FLOAT);
            break;
        case format::rg32_float:
            simple(DXGI_FORMAT_R32G32_FLOAT);
            break;
        case format::rgb32_float:
            simple(DXGI_FORMAT_R32G32B32_FLOAT);
            break;
        case format::rgba32_float:
            simple(DXGI_FORMAT_R32G32B32A32_FLOAT);
            break;
        case format::rgb10a2_unorm:
            simple(DXGI_FORMAT_R10G10B10A2_UNORM);
            break;
        case format::r11g11b10_float:
            simple(DXGI_FORMAT_R11G11B10_FLOAT);
            break;
        case format::rg16_unorm:
            simple(DXGI_FORMAT_R16G16_UNORM);
            break;
        case format::rgba16_unorm:
            simple(DXGI_FORMAT_R16G16B16A16_UNORM);
            break;
        case format::d16_unorm:
            m.resource = DXGI_FORMAT_R16_TYPELESS;
            m.view = DXGI_FORMAT_R16_UNORM;
            m.dsv = DXGI_FORMAT_D16_UNORM;
            break;
        case format::d32_float:
            m.resource = DXGI_FORMAT_R32_TYPELESS;
            m.view = DXGI_FORMAT_R32_FLOAT;
            m.dsv = DXGI_FORMAT_D32_FLOAT;
            break;
        case format::d24_unorm_s8_uint:
            m.resource = DXGI_FORMAT_R24G8_TYPELESS;
            m.view = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            m.stencil_view = DXGI_FORMAT_X24_TYPELESS_G8_UINT;
            m.dsv = DXGI_FORMAT_D24_UNORM_S8_UINT;
            break;
        case format::d32_float_s8_uint:
            m.resource = DXGI_FORMAT_R32G8X24_TYPELESS;
            m.view = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            m.stencil_view = DXGI_FORMAT_X32_TYPELESS_G8X24_UINT;
            m.dsv = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            break;
        case format::bc1_unorm:
            simple(DXGI_FORMAT_BC1_UNORM);
            break;
        case format::bc3_unorm:
            simple(DXGI_FORMAT_BC3_UNORM);
            break;
        case format::bc7_unorm:
            simple(DXGI_FORMAT_BC7_UNORM);
            break;
        default:
            break;
        }
        return m;
    }

    inline D3D11_COMPARISON_FUNC to_d3d(compare_op op) {
        switch (op) {
        case compare_op::never:
            return D3D11_COMPARISON_NEVER;
        case compare_op::less:
            return D3D11_COMPARISON_LESS;
        case compare_op::equal:
            return D3D11_COMPARISON_EQUAL;
        case compare_op::less_equal:
            return D3D11_COMPARISON_LESS_EQUAL;
        case compare_op::greater:
            return D3D11_COMPARISON_GREATER;
        case compare_op::not_equal:
            return D3D11_COMPARISON_NOT_EQUAL;
        case compare_op::greater_equal:
            return D3D11_COMPARISON_GREATER_EQUAL;
        default:
            return D3D11_COMPARISON_ALWAYS;
        }
    }

    inline D3D11_STENCIL_OP to_d3d(stencil_op op) {
        switch (op) {
        case stencil_op::keep:
            return D3D11_STENCIL_OP_KEEP;
        case stencil_op::zero:
            return D3D11_STENCIL_OP_ZERO;
        case stencil_op::replace:
            return D3D11_STENCIL_OP_REPLACE;
        case stencil_op::increment_clamp:
            return D3D11_STENCIL_OP_INCR_SAT;
        case stencil_op::decrement_clamp:
            return D3D11_STENCIL_OP_DECR_SAT;
        case stencil_op::invert:
            return D3D11_STENCIL_OP_INVERT;
        case stencil_op::increment_wrap:
            return D3D11_STENCIL_OP_INCR;
        default:
            return D3D11_STENCIL_OP_DECR;
        }
    }

    inline D3D11_BLEND to_d3d(blend_factor f, bool alpha) {
        switch (f) {
        case blend_factor::zero:
            return D3D11_BLEND_ZERO;
        case blend_factor::one:
            return D3D11_BLEND_ONE;
        case blend_factor::src_color:
            return alpha ? D3D11_BLEND_SRC_ALPHA : D3D11_BLEND_SRC_COLOR;
        case blend_factor::inv_src_color:
            return alpha ? D3D11_BLEND_INV_SRC_ALPHA : D3D11_BLEND_INV_SRC_COLOR;
        case blend_factor::src_alpha:
            return D3D11_BLEND_SRC_ALPHA;
        case blend_factor::inv_src_alpha:
            return D3D11_BLEND_INV_SRC_ALPHA;
        case blend_factor::dst_color:
            return alpha ? D3D11_BLEND_DEST_ALPHA : D3D11_BLEND_DEST_COLOR;
        case blend_factor::inv_dst_color:
            return alpha ? D3D11_BLEND_INV_DEST_ALPHA : D3D11_BLEND_INV_DEST_COLOR;
        case blend_factor::dst_alpha:
            return D3D11_BLEND_DEST_ALPHA;
        case blend_factor::inv_dst_alpha:
            return D3D11_BLEND_INV_DEST_ALPHA;
        default:
            return D3D11_BLEND_SRC_ALPHA_SAT;
        }
    }

    inline D3D11_BLEND_OP to_d3d(blend_op op) {
        switch (op) {
        case blend_op::add:
            return D3D11_BLEND_OP_ADD;
        case blend_op::subtract:
            return D3D11_BLEND_OP_SUBTRACT;
        case blend_op::reverse_subtract:
            return D3D11_BLEND_OP_REV_SUBTRACT;
        case blend_op::minimum:
            return D3D11_BLEND_OP_MIN;
        default:
            return D3D11_BLEND_OP_MAX;
        }
    }

    inline D3D11_TEXTURE_ADDRESS_MODE to_d3d(address_mode m) {
        switch (m) {
        case address_mode::repeat:
            return D3D11_TEXTURE_ADDRESS_WRAP;
        case address_mode::mirror:
            return D3D11_TEXTURE_ADDRESS_MIRROR;
        case address_mode::clamp:
            return D3D11_TEXTURE_ADDRESS_CLAMP;
        default:
            return D3D11_TEXTURE_ADDRESS_BORDER;
        }
    }

    inline D3D11_FILTER to_d3d(const sampler_desc& s) {
        if (s.max_anisotropy > 1.0f)
            return s.compare ? D3D11_FILTER_COMPARISON_ANISOTROPIC : D3D11_FILTER_ANISOTROPIC;
        UINT bits = (s.min == filter::linear ? 0x10 : 0) | (s.mag == filter::linear ? 0x04 : 0) |
                    (s.mip == filter::linear ? 0x01 : 0);
        if (s.compare)
            bits |= 0x80;
        return static_cast<D3D11_FILTER>(bits);
    }

    inline D3D11_PRIMITIVE_TOPOLOGY to_d3d(topology t) {
        switch (t) {
        case topology::triangle_list:
            return D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        case topology::triangle_strip:
            return D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
        case topology::line_list:
            return D3D11_PRIMITIVE_TOPOLOGY_LINELIST;
        default:
            return D3D11_PRIMITIVE_TOPOLOGY_POINTLIST;
        }
    }
}
