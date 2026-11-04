#include "metal/metal_internal.hpp"

namespace conduit::metal {
    bool g_depth24_stencil8 = false;

    MTLPixelFormat to_pixel_format(format f) {
        switch (f) {
        case format::r8_unorm:
            return MTLPixelFormatR8Unorm;
        case format::rg8_unorm:
            return MTLPixelFormatRG8Unorm;
        case format::rgba8_unorm:
            return MTLPixelFormatRGBA8Unorm;
        case format::rgba8_srgb:
            return MTLPixelFormatRGBA8Unorm_sRGB;
        case format::bgra8_unorm:
            return MTLPixelFormatBGRA8Unorm;
        case format::bgra8_srgb:
            return MTLPixelFormatBGRA8Unorm_sRGB;
        case format::r8_uint:
            return MTLPixelFormatR8Uint;
        case format::r16_uint:
            return MTLPixelFormatR16Uint;
        case format::r32_uint:
            return MTLPixelFormatR32Uint;
        case format::rg32_uint:
            return MTLPixelFormatRG32Uint;
        case format::rgba32_uint:
            return MTLPixelFormatRGBA32Uint;
        case format::r16_float:
            return MTLPixelFormatR16Float;
        case format::rg16_float:
            return MTLPixelFormatRG16Float;
        case format::rgba16_float:
            return MTLPixelFormatRGBA16Float;
        case format::r32_float:
            return MTLPixelFormatR32Float;
        case format::rg32_float:
            return MTLPixelFormatRG32Float;
        case format::rgba32_float:
            return MTLPixelFormatRGBA32Float;
        case format::rgb10a2_unorm:
            return MTLPixelFormatRGB10A2Unorm;
        case format::r11g11b10_float:
            return MTLPixelFormatRG11B10Float;
        case format::rg16_unorm:
            return MTLPixelFormatRG16Unorm;
        case format::rgba16_unorm:
            return MTLPixelFormatRGBA16Unorm;
        case format::d16_unorm:
            return MTLPixelFormatDepth16Unorm;
        case format::d32_float:
            return MTLPixelFormatDepth32Float;
        case format::d24_unorm_s8_uint:
#if TARGET_OS_OSX
            if (g_depth24_stencil8)
                return MTLPixelFormatDepth24Unorm_Stencil8;
#endif
            return MTLPixelFormatDepth32Float_Stencil8;
        case format::d32_float_s8_uint:
            return MTLPixelFormatDepth32Float_Stencil8;
#if TARGET_OS_OSX || (defined(__IPHONE_16_4) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 160400)
        case format::bc1_unorm:
            return MTLPixelFormatBC1_RGBA;
        case format::bc3_unorm:
            return MTLPixelFormatBC3_RGBA;
        case format::bc7_unorm:
            return MTLPixelFormatBC7_RGBAUnorm;
#endif
        default:
            return MTLPixelFormatInvalid;
        }
    }

    MTLVertexFormat to_vertex_format(format f) {
        switch (f) {
        case format::r8_unorm:
            return MTLVertexFormatUCharNormalized;
        case format::rg8_unorm:
            return MTLVertexFormatUChar2Normalized;
        case format::rgba8_unorm:
            return MTLVertexFormatUChar4Normalized;
        case format::bgra8_unorm:
            return MTLVertexFormatUChar4Normalized_BGRA;
        case format::r8_uint:
            return MTLVertexFormatUChar;
        case format::r16_uint:
            return MTLVertexFormatUShort;
        case format::r32_uint:
            return MTLVertexFormatUInt;
        case format::rg32_uint:
            return MTLVertexFormatUInt2;
        case format::rgba32_uint:
            return MTLVertexFormatUInt4;
        case format::r16_float:
            return MTLVertexFormatHalf;
        case format::rg16_float:
            return MTLVertexFormatHalf2;
        case format::rgba16_float:
            return MTLVertexFormatHalf4;
        case format::r32_float:
            return MTLVertexFormatFloat;
        case format::rg32_float:
            return MTLVertexFormatFloat2;
        case format::rgb32_float:
            return MTLVertexFormatFloat3;
        case format::rgba32_float:
            return MTLVertexFormatFloat4;
        case format::rgb10a2_unorm:
            return MTLVertexFormatUInt1010102Normalized;
        case format::rg16_unorm:
            return MTLVertexFormatUShort2Normalized;
        case format::rgba16_unorm:
            return MTLVertexFormatUShort4Normalized;
        default:
            return MTLVertexFormatInvalid;
        }
    }

    MTLCompareFunction to_compare(compare_op op) {
        switch (op) {
        case compare_op::never:
            return MTLCompareFunctionNever;
        case compare_op::less:
            return MTLCompareFunctionLess;
        case compare_op::equal:
            return MTLCompareFunctionEqual;
        case compare_op::less_equal:
            return MTLCompareFunctionLessEqual;
        case compare_op::greater:
            return MTLCompareFunctionGreater;
        case compare_op::not_equal:
            return MTLCompareFunctionNotEqual;
        case compare_op::greater_equal:
            return MTLCompareFunctionGreaterEqual;
        default:
            return MTLCompareFunctionAlways;
        }
    }

    MTLStencilOperation to_stencil_op(stencil_op op) {
        switch (op) {
        case stencil_op::zero:
            return MTLStencilOperationZero;
        case stencil_op::replace:
            return MTLStencilOperationReplace;
        case stencil_op::increment_clamp:
            return MTLStencilOperationIncrementClamp;
        case stencil_op::decrement_clamp:
            return MTLStencilOperationDecrementClamp;
        case stencil_op::invert:
            return MTLStencilOperationInvert;
        case stencil_op::increment_wrap:
            return MTLStencilOperationIncrementWrap;
        case stencil_op::decrement_wrap:
            return MTLStencilOperationDecrementWrap;
        default:
            return MTLStencilOperationKeep;
        }
    }

    MTLBlendFactor to_blend_factor(blend_factor f) {
        switch (f) {
        case blend_factor::zero:
            return MTLBlendFactorZero;
        case blend_factor::one:
            return MTLBlendFactorOne;
        case blend_factor::src_color:
            return MTLBlendFactorSourceColor;
        case blend_factor::inv_src_color:
            return MTLBlendFactorOneMinusSourceColor;
        case blend_factor::src_alpha:
            return MTLBlendFactorSourceAlpha;
        case blend_factor::inv_src_alpha:
            return MTLBlendFactorOneMinusSourceAlpha;
        case blend_factor::dst_color:
            return MTLBlendFactorDestinationColor;
        case blend_factor::inv_dst_color:
            return MTLBlendFactorOneMinusDestinationColor;
        case blend_factor::dst_alpha:
            return MTLBlendFactorDestinationAlpha;
        case blend_factor::inv_dst_alpha:
            return MTLBlendFactorOneMinusDestinationAlpha;
        default:
            return MTLBlendFactorSourceAlphaSaturated;
        }
    }

    MTLBlendOperation to_blend_op(blend_op op) {
        switch (op) {
        case blend_op::subtract:
            return MTLBlendOperationSubtract;
        case blend_op::reverse_subtract:
            return MTLBlendOperationReverseSubtract;
        case blend_op::minimum:
            return MTLBlendOperationMin;
        case blend_op::maximum:
            return MTLBlendOperationMax;
        default:
            return MTLBlendOperationAdd;
        }
    }

    MTLSamplerAddressMode to_address(address_mode m) {
        switch (m) {
        case address_mode::repeat:
            return MTLSamplerAddressModeRepeat;
        case address_mode::mirror:
            return MTLSamplerAddressModeMirrorRepeat;
        case address_mode::border:
#if TARGET_OS_OSX
            return MTLSamplerAddressModeClampToBorderColor;
#else
            return MTLSamplerAddressModeClampToEdge;
#endif
        default:
            return MTLSamplerAddressModeClampToEdge;
        }
    }

    MTLSamplerMinMagFilter to_min_mag(filter f) {
        return f == filter::nearest ? MTLSamplerMinMagFilterNearest : MTLSamplerMinMagFilterLinear;
    }
    MTLSamplerMipFilter to_mip(filter f) {
        return f == filter::nearest ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear;
    }

    MTLPrimitiveType to_primitive(topology t) {
        switch (t) {
        case topology::triangle_strip:
            return MTLPrimitiveTypeTriangleStrip;
        case topology::line_list:
            return MTLPrimitiveTypeLine;
        case topology::point_list:
            return MTLPrimitiveTypePoint;
        default:
            return MTLPrimitiveTypeTriangle;
        }
    }

    MTLTextureType to_texture_type(texture_dim dim, bool multisample, bool array) {
        switch (dim) {
        case texture_dim::tex2d:
            if (multisample)
                return array ? MTLTextureType2DMultisampleArray : MTLTextureType2DMultisample;
            return array ? MTLTextureType2DArray : MTLTextureType2D;
        case texture_dim::tex2d_array:
            return multisample ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
        case texture_dim::cube:
            return MTLTextureTypeCube;
        default:
            return MTLTextureType3D;
        }
    }
}
