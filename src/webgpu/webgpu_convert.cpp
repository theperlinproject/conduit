#include "webgpu/webgpu_internal.hpp"

namespace conduit::webgpu {
    WGPUTextureFormat to_texture_format(conduit::format format) {
        switch (format) {
        case conduit::format::r8_unorm:
            return WGPUTextureFormat_R8Unorm;
        case conduit::format::rg8_unorm:
            return WGPUTextureFormat_RG8Unorm;
        case conduit::format::rgba8_unorm:
            return WGPUTextureFormat_RGBA8Unorm;
        case conduit::format::rgba8_srgb:
            return WGPUTextureFormat_RGBA8UnormSrgb;
        case conduit::format::bgra8_unorm:
            return WGPUTextureFormat_BGRA8Unorm;
        case conduit::format::bgra8_srgb:
            return WGPUTextureFormat_BGRA8UnormSrgb;
        case conduit::format::r8_uint:
            return WGPUTextureFormat_R8Uint;
        case conduit::format::r16_uint:
            return WGPUTextureFormat_R16Uint;
        case conduit::format::r32_uint:
            return WGPUTextureFormat_R32Uint;
        case conduit::format::rg32_uint:
            return WGPUTextureFormat_RG32Uint;
        case conduit::format::rgba32_uint:
            return WGPUTextureFormat_RGBA32Uint;
        case conduit::format::r16_float:
            return WGPUTextureFormat_R16Float;
        case conduit::format::rg16_float:
            return WGPUTextureFormat_RG16Float;
        case conduit::format::rgba16_float:
            return WGPUTextureFormat_RGBA16Float;
        case conduit::format::r32_float:
            return WGPUTextureFormat_R32Float;
        case conduit::format::rg32_float:
            return WGPUTextureFormat_RG32Float;
        case conduit::format::rgba32_float:
            return WGPUTextureFormat_RGBA32Float;
        case conduit::format::rgb10a2_unorm:
            return WGPUTextureFormat_RGB10A2Unorm;
        case conduit::format::r11g11b10_float:
            return WGPUTextureFormat_RG11B10Ufloat;
        case conduit::format::d16_unorm:
            return WGPUTextureFormat_Depth16Unorm;
        case conduit::format::d32_float:
            return WGPUTextureFormat_Depth32Float;
        case conduit::format::d24_unorm_s8_uint:
            return WGPUTextureFormat_Depth24PlusStencil8;
        case conduit::format::d32_float_s8_uint:
            return WGPUTextureFormat_Depth32FloatStencil8;
        case conduit::format::bc1_unorm:
            return WGPUTextureFormat_BC1RGBAUnorm;
        case conduit::format::bc3_unorm:
            return WGPUTextureFormat_BC3RGBAUnorm;
        case conduit::format::bc7_unorm:
            return WGPUTextureFormat_BC7RGBAUnorm;
        default:
            return WGPUTextureFormat_Undefined;
        }
    }

    conduit::format from_texture_format(WGPUTextureFormat format) {
        switch (format) {
        case WGPUTextureFormat_BGRA8Unorm:
            return conduit::format::bgra8_unorm;
        case WGPUTextureFormat_BGRA8UnormSrgb:
            return conduit::format::bgra8_srgb;
        case WGPUTextureFormat_RGBA8Unorm:
            return conduit::format::rgba8_unorm;
        case WGPUTextureFormat_RGBA8UnormSrgb:
            return conduit::format::rgba8_srgb;
        case WGPUTextureFormat_RGBA16Float:
            return conduit::format::rgba16_float;
        case WGPUTextureFormat_RGB10A2Unorm:
            return conduit::format::rgb10a2_unorm;
        default:
            return conduit::format::unknown;
        }
    }

    WGPUVertexFormat to_vertex_format(conduit::format format) {
        switch (format) {
        case conduit::format::r8_unorm:
            return WGPUVertexFormat_Unorm8;
        case conduit::format::rg8_unorm:
            return WGPUVertexFormat_Unorm8x2;
        case conduit::format::rgba8_unorm:
            return WGPUVertexFormat_Unorm8x4;
        case conduit::format::bgra8_unorm:
            return WGPUVertexFormat_Unorm8x4BGRA;
        case conduit::format::r8_uint:
            return WGPUVertexFormat_Uint8;
        case conduit::format::r16_uint:
            return WGPUVertexFormat_Uint16;
        case conduit::format::r32_uint:
            return WGPUVertexFormat_Uint32;
        case conduit::format::rg32_uint:
            return WGPUVertexFormat_Uint32x2;
        case conduit::format::rgba32_uint:
            return WGPUVertexFormat_Uint32x4;
        case conduit::format::r16_float:
            return WGPUVertexFormat_Float16;
        case conduit::format::rg16_float:
            return WGPUVertexFormat_Float16x2;
        case conduit::format::rgba16_float:
            return WGPUVertexFormat_Float16x4;
        case conduit::format::r32_float:
            return WGPUVertexFormat_Float32;
        case conduit::format::rg32_float:
            return WGPUVertexFormat_Float32x2;
        case conduit::format::rgb32_float:
            return WGPUVertexFormat_Float32x3;
        case conduit::format::rgba32_float:
            return WGPUVertexFormat_Float32x4;
        case conduit::format::rgb10a2_unorm:
            return WGPUVertexFormat_Unorm10_10_10_2;
        case conduit::format::rg16_unorm:
            return WGPUVertexFormat_Unorm16x2;
        case conduit::format::rgba16_unorm:
            return WGPUVertexFormat_Unorm16x4;
        default:
            return static_cast<WGPUVertexFormat>(0);
        }
    }

    WGPUBlendFactor to_blend_factor(blend_factor f) {
        switch (f) {
        case blend_factor::zero:
            return WGPUBlendFactor_Zero;
        case blend_factor::one:
            return WGPUBlendFactor_One;
        case blend_factor::src_color:
            return WGPUBlendFactor_Src;
        case blend_factor::inv_src_color:
            return WGPUBlendFactor_OneMinusSrc;
        case blend_factor::src_alpha:
            return WGPUBlendFactor_SrcAlpha;
        case blend_factor::inv_src_alpha:
            return WGPUBlendFactor_OneMinusSrcAlpha;
        case blend_factor::dst_color:
            return WGPUBlendFactor_Dst;
        case blend_factor::inv_dst_color:
            return WGPUBlendFactor_OneMinusDst;
        case blend_factor::dst_alpha:
            return WGPUBlendFactor_DstAlpha;
        case blend_factor::inv_dst_alpha:
            return WGPUBlendFactor_OneMinusDstAlpha;
        case blend_factor::src_alpha_saturate:
            return WGPUBlendFactor_SrcAlphaSaturated;
        }
        return WGPUBlendFactor_One;
    }

    WGPUBlendOperation to_blend_op(blend_op op) {
        switch (op) {
        case blend_op::add:
            return WGPUBlendOperation_Add;
        case blend_op::subtract:
            return WGPUBlendOperation_Subtract;
        case blend_op::reverse_subtract:
            return WGPUBlendOperation_ReverseSubtract;
        case blend_op::minimum:
            return WGPUBlendOperation_Min;
        case blend_op::maximum:
            return WGPUBlendOperation_Max;
        }
        return WGPUBlendOperation_Add;
    }

    WGPUCompareFunction to_compare(compare_op op) {
        switch (op) {
        case compare_op::never:
            return WGPUCompareFunction_Never;
        case compare_op::less:
            return WGPUCompareFunction_Less;
        case compare_op::equal:
            return WGPUCompareFunction_Equal;
        case compare_op::less_equal:
            return WGPUCompareFunction_LessEqual;
        case compare_op::greater:
            return WGPUCompareFunction_Greater;
        case compare_op::not_equal:
            return WGPUCompareFunction_NotEqual;
        case compare_op::greater_equal:
            return WGPUCompareFunction_GreaterEqual;
        case compare_op::always:
            return WGPUCompareFunction_Always;
        }
        return WGPUCompareFunction_Always;
    }

    WGPUStencilOperation to_stencil_op(stencil_op op) {
        switch (op) {
        case stencil_op::keep:
            return WGPUStencilOperation_Keep;
        case stencil_op::zero:
            return WGPUStencilOperation_Zero;
        case stencil_op::replace:
            return WGPUStencilOperation_Replace;
        case stencil_op::increment_clamp:
            return WGPUStencilOperation_IncrementClamp;
        case stencil_op::decrement_clamp:
            return WGPUStencilOperation_DecrementClamp;
        case stencil_op::invert:
            return WGPUStencilOperation_Invert;
        case stencil_op::increment_wrap:
            return WGPUStencilOperation_IncrementWrap;
        case stencil_op::decrement_wrap:
            return WGPUStencilOperation_DecrementWrap;
        }
        return WGPUStencilOperation_Keep;
    }

    WGPUAddressMode to_address(address_mode mode) {
        switch (mode) {
        case address_mode::repeat:
            return WGPUAddressMode_Repeat;
        case address_mode::mirror:
            return WGPUAddressMode_MirrorRepeat;
        default:
            return WGPUAddressMode_ClampToEdge;
        }
    }

    WGPUFilterMode to_filter(filter f) {
        return f == filter::nearest ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
    }
    WGPUMipmapFilterMode to_mip_filter(filter f) {
        return f == filter::nearest ? WGPUMipmapFilterMode_Nearest : WGPUMipmapFilterMode_Linear;
    }

    WGPUTextureViewDimension to_view_dimension(texture_dim dim) {
        switch (dim) {
        case texture_dim::tex2d:
            return WGPUTextureViewDimension_2D;
        case texture_dim::tex2d_array:
            return WGPUTextureViewDimension_2DArray;
        case texture_dim::cube:
            return WGPUTextureViewDimension_Cube;
        case texture_dim::tex3d:
            return WGPUTextureViewDimension_3D;
        }
        return WGPUTextureViewDimension_2D;
    }
}
