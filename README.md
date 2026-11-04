# conduit

**conduit** is a cross-platform rendering hardware interface for C++23. One API drives Vulkan, Direct3D 12, Direct3D 11,
Metal and WebGPU, with a null backend for headless tests. It also picks the best API and adapter for the machine it runs on.

```cpp
#include <conduit/device.hpp>

auto instance = conduit::create_instance(conduit::api::vulkan);
auto device = instance->create_device({});

conduit::buffer_desc desc;
desc.size = 64 << 10;
desc.usage = conduit::buffer_usage::vertex | conduit::buffer_usage::copy_dst;
conduit::buffer_handle vertices = device->create_buffer(desc);

device->begin_frame();
conduit::command_list* list = device->acquire_command_list(conduit::queue_type::graphics);
list->fill_buffer(vertices, 0, desc.size, 0);
device->submit(conduit::queue_type::graphics, list);
device->end_frame();
```

## Backends

| Backend      | Option            | Default on                  |
|--------------|-------------------|-----------------------------|
| Vulkan       | `CONDUIT_VULKAN`  | Windows, Linux, Android     |
| Direct3D 12  | `CONDUIT_D3D12`   | Windows                     |
| Direct3D 11  | `CONDUIT_D3D11`   | off                         |
| Metal        | `CONDUIT_METAL`   | macOS, iOS                  |
| WebGPU       | `CONDUIT_WEBGPU`  | Emscripten                  |
| Streamline   | `CONDUIT_STREAMLINE` | Windows x64 (DLSS, Reflex, frame generation) |

Other options:

| Option                       | Default | Effect                                              |
|------------------------------|---------|-----------------------------------------------------|
| `CONDUIT_LTO`                | off     | link-time optimization (on in the presets)          |
| `CONDUIT_CLANG_TIDY`         | off     | run `.clang-tidy` on every conduit source as it compiles |
| `CONDUIT_WARNINGS_AS_ERRORS` | off     | `-Werror` / `/WX`                                   |
| `CONDUIT_BUILD_TESTS`        | top level only | tests                                        |
| `CONDUIT_INSTALL`            | top level only | install and package config                   |

Every backend loads its driver at run time. If a loader or driver is missing, `create_instance` returns null instead of
failing to start.

## Headers

| Header                         | Contents                                                     |
|--------------------------------|--------------------------------------------------------------|
| `conduit/types.hpp`            | handles, formats, resource and pipeline descriptions         |
| `conduit/device.hpp`           | `instance`, `device`, `command_list`, `swapchain`            |
| `conduit/render_profile.hpp`   | system probing and `choose_profile` (API, adapter, tier, upscalers) |
| `conduit/shader_library.hpp`   | reads shader packs built by `tools/pack_shaders.py`          |
| `conduit/vendor.hpp`           | upscaling, frame generation and low-latency sessions         |
| `conduit/conduit.hpp`          | all of the above                                             |

## Using it from CMake

```cmake
include(FetchContent)
FetchContent_Declare(conduit GIT_REPOSITORY https://github.com/theperlinproject/conduit.git GIT_TAG main)
FetchContent_MakeAvailable(conduit)
target_link_libraries(app PRIVATE conduit::conduit)
```

You can also use it as a subdirectory (`add_subdirectory(conduit)`), or install it and call `find_package(conduit CONFIG REQUIRED)`.
Dependencies (Vulkan-Headers, volk, VMA, DirectX-Headers, D3D12MA, Streamline, wgpu-native) are fetched at their pinned
revisions. Point `CONDUIT_DEPS_DIR` at a directory of checkouts to build offline.

## Shader packs

Shaders are written in Slang. `tools/pack_shaders.py` compiles each program for every target the enabled backends read
(SPIR-V, DXIL, HLSL, MSL, WGSL) and packs them with their uniform layout. `conduit_add_shader_pack()` wraps it:

```cmake
conduit_add_shader_pack(app_shaders
    OUTPUT ${CMAKE_BINARY_DIR}/app.pack
    SLANGC ${SLANGC}
    SLOTS ${CMAKE_SOURCE_DIR}/shaders/texture_slots.json
    TARGETS spirv msl
    PROGRAMS ${CMAKE_SOURCE_DIR}/shaders/programs)
```

The binding layout is as follows. Uniforms go in set 0, binding 0. Texture slot *n* goes in set 1, binding 2*n*, with its sampler at binding 2*n*+1.

## Building and testing

```sh
cmake --preset linux-clang
cmake --build --preset linux-clang
ctest --preset linux-clang
```

The shader and GPU tests are registered when `slangc` is found (on `PATH`, or in `CONDUIT_DEPS_DIR/slang/bin`). The GPU tests
exit with code 77 (skipped) on machines without a driver.

## Performance

- Draw, bind and barrier recording does not allocate once warmed up: command lists, barrier scratch and descriptor pools are pooled and reused.
- Redundant `set_pipeline`, `set_bind_group` and `set_vertex_buffer` calls are dropped, and dirty bind groups are bound in contiguous runs.
- Statistics are counted per command list and folded in once at submit, so recording on several threads shares no cache lines.
- Handle lookups are lock free: pool pages never move, and a generation in every handle catches stale use.
- Indirect draws use multi-draw indirect and GPU-side draw counts where the driver supports them. Otherwise they fall back to a loop.
- Pipelines compile off the render thread where the API allows it, and pipeline caches can be saved and reloaded.