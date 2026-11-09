#pragma once
#include <conduit/device.hpp>

#ifndef CONDUIT_HAS_VULKAN
#define CONDUIT_HAS_VULKAN 0
#endif
#ifndef CONDUIT_HAS_D3D12
#define CONDUIT_HAS_D3D12 0
#endif
#ifndef CONDUIT_HAS_D3D11
#define CONDUIT_HAS_D3D11 0
#endif
#ifndef CONDUIT_HAS_METAL
#define CONDUIT_HAS_METAL 0
#endif
#ifndef CONDUIT_HAS_WEBGPU
#define CONDUIT_HAS_WEBGPU 0
#endif
#ifndef CONDUIT_HAS_GLES
#define CONDUIT_HAS_GLES 0
#endif

namespace conduit::backends {
    std::unique_ptr<instance> create_null_instance(const instance_options& options);
    std::unique_ptr<instance> create_vulkan_instance(const instance_options& options);
    std::unique_ptr<instance> create_d3d12_instance(const instance_options& options);
    std::unique_ptr<instance> create_d3d11_instance(const instance_options& options);
    std::unique_ptr<instance> create_metal_instance(const instance_options& options);
    std::unique_ptr<instance> create_webgpu_instance(const instance_options& options);
    std::unique_ptr<instance> create_gles_instance(const instance_options& options);
}
