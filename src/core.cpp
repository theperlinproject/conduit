#include <conduit/device.hpp>
#include "backends.hpp"

#include <algorithm>

namespace conduit {
    conduit::api api_from_id(std::string_view id) {
        for (conduit::api api : {conduit::api::vulkan, conduit::api::d3d12, conduit::api::d3d11, conduit::api::metal,
                                 conduit::api::webgpu})
            if (id == api_id(api))
                return api;
        if (id == "dx12" || id == "directx12")
            return conduit::api::d3d12;
        if (id == "dx11" || id == "directx11")
            return conduit::api::d3d11;
        if (id == "vk")
            return conduit::api::vulkan;
        if (id == "mtl")
            return conduit::api::metal;
        return conduit::api::none;
    }

    const char* vendor_name(uint32_t vendor_id) {
        switch (vendor_id) {
        case vendor_id::nvidia:
            return "NVIDIA";
        case vendor_id::amd:
            return "AMD";
        case vendor_id::intel:
            return "Intel";
        case vendor_id::apple:
            return "Apple";
        case vendor_id::qualcomm:
            return "Qualcomm";
        case vendor_id::arm:
            return "ARM";
        case vendor_id::microsoft:
            return "Microsoft";
        default:
            return "Unknown";
        }
    }

    const format_info& get_format_info(conduit::format format) {
        static const format_info table[] = {
            /* unknown */ {0, 1, 1, 0, false, false, false, false, false},
            /* r8_unorm */ {1, 1, 1, 1, false, false, false, false, false},
            /* rg8_unorm */ {2, 1, 1, 2, false, false, false, false, false},
            /* rgba8_unorm */ {4, 1, 1, 4, false, false, false, false, false},
            /* rgba8_srgb */ {4, 1, 1, 4, false, false, true, false, false},
            /* bgra8_unorm */ {4, 1, 1, 4, false, false, false, false, false},
            /* bgra8_srgb */ {4, 1, 1, 4, false, false, true, false, false},
            /* r8_uint */ {1, 1, 1, 1, false, false, false, false, false},
            /* r16_uint */ {2, 1, 1, 1, false, false, false, false, false},
            /* r32_uint */ {4, 1, 1, 1, false, false, false, false, false},
            /* rg32_uint */ {8, 1, 1, 2, false, false, false, false, false},
            /* rgba32_uint */ {16, 1, 1, 4, false, false, false, false, false},
            /* r16_float */ {2, 1, 1, 1, false, false, false, false, true},
            /* rg16_float */ {4, 1, 1, 2, false, false, false, false, true},
            /* rgba16_float */ {8, 1, 1, 4, false, false, false, false, true},
            /* r32_float */ {4, 1, 1, 1, false, false, false, false, true},
            /* rg32_float */ {8, 1, 1, 2, false, false, false, false, true},
            /* rgb32_float */ {12, 1, 1, 3, false, false, false, false, true},
            /* rgba32_float */ {16, 1, 1, 4, false, false, false, false, true},
            /* rgb10a2_unorm */ {4, 1, 1, 4, false, false, false, false, false},
            /* r11g11b10_float */ {4, 1, 1, 3, false, false, false, false, true},
            /* rg16_unorm */ {4, 1, 1, 2, false, false, false, false, false},
            /* rgba16_unorm */ {8, 1, 1, 4, false, false, false, false, false},
            /* d16_unorm */ {2, 1, 1, 1, true, false, false, false, false},
            /* d32_float */ {4, 1, 1, 1, true, false, false, false, true},
            /* d24_unorm_s8_uint */ {4, 1, 1, 2, true, true, false, false, false},
            /* d32_float_s8_uint */ {8, 1, 1, 2, true, true, false, false, true},
            /* bc1_unorm */ {8, 4, 4, 4, false, false, false, true, false},
            /* bc3_unorm */ {16, 4, 4, 4, false, false, false, true, false},
            /* bc7_unorm */ {16, 4, 4, 4, false, false, false, true, false},
        };
        static_assert(std::size(table) == static_cast<size_t>(conduit::format::count));
        return table[static_cast<size_t>(format)];
    }

    bool api_compiled_in(conduit::api api) {
        switch (api) {
#if CONDUIT_HAS_VULKAN
        case conduit::api::vulkan:
            return true;
#endif
#if CONDUIT_HAS_D3D12
        case conduit::api::d3d12:
            return true;
#endif
#if CONDUIT_HAS_D3D11
        case conduit::api::d3d11:
            return true;
#endif
#if CONDUIT_HAS_METAL
        case Api::Metal:
            return true;
#endif
#if CONDUIT_HAS_WEBGPU
        case conduit::api::webgpu:
            return true;
#endif
        case conduit::api::none:
            return true;
        default:
            return false;
        }
    }

    std::vector<conduit::api> compiled_apis() {
        std::vector<conduit::api> out;
        for (conduit::api api : {conduit::api::vulkan, conduit::api::d3d12, conduit::api::d3d11, conduit::api::metal,
                                 conduit::api::webgpu})
            if (api_compiled_in(api))
                out.push_back(api);
        return out;
    }

    std::unique_ptr<instance> create_instance(conduit::api api, const instance_options& options) {
        switch (api) {
        case conduit::api::none:
            return backends::create_null_instance(options);
#if CONDUIT_HAS_VULKAN
        case conduit::api::vulkan:
            return backends::create_vulkan_instance(options);
#endif
#if CONDUIT_HAS_D3D12
        case conduit::api::d3d12:
            return backends::create_d3d12_instance(options);
#endif
#if CONDUIT_HAS_D3D11
        case conduit::api::d3d11:
            return backends::create_d3d11_instance(options);
#endif
#if CONDUIT_HAS_METAL
        case Api::Metal:
            return backends::CreateMetalInstance(options);
#endif
#if CONDUIT_HAS_WEBGPU
        case conduit::api::webgpu:
            return backends::create_webgpu_instance(options);
#endif
        default:
            return nullptr;
        }
    }
}
