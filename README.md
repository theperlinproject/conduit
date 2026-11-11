# conduit

conduit is a cross-platform rendering hardware interface for C++23. One API drives Vulkan, Direct3D 12, Direct3D 11,
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
| Vulkan       | `CONDUIT_VULKAN`  | Windows, Linux, Android (1.1 or newer; 1.2 and 1.3 take the fast paths) |
| Direct3D 12  | `CONDUIT_D3D12`   | Windows                     |
| Direct3D 11  | `CONDUIT_D3D11`   | off                         |
| Metal        | `CONDUIT_METAL`   | macOS, iOS                  |
| WebGPU       | `CONDUIT_WEBGPU`  | Emscripten                  |
| OpenGL ES 3 / WebGL 2 | `CONDUIT_GLES` | everywhere except Apple (last resort) |
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

## Vulkan 1.1

Devices that only have Vulkan 1.1 work. Missing features are emulated with extensions or fallbacks (timeline semaphores
through pooled fences, a render pass instead of dynamic rendering, legacy barriers). That code only runs when the feature
is absent, so Vulkan 1.2 and 1.3 devices take exactly the same paths as before. `tools/pack_shaders.py` emits SPIR-V 1.3
so the same pack loads on 1.1 devices.

## OpenGL ES and Firefox

`CONDUIT_GLES` is the last-resort backend. Scoring always ranks it below every other API, so it is only picked when
nothing else works: older phones without Vulkan 1.1, and Firefox, which has no WebGPU (Emscripten builds it as WebGL 2 on
`#canvas`). Shaders come from the `spirv` pack and are translated at pipeline creation with SPIRV-Cross. The context
lives on a worker thread and command lists are replayed there at submit.

Limits: one GL context; no timestamps; no `draw_indexed_indirect_count`; compute, storage buffers and storage images need
ES 3.1 (not WebGL 2); upload buffers must be `flush()`ed before GPU use; each frame ends with one blit to the window. It
has not been run on a GPU yet, so treat it as unverified.

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

- GPU-driven rendering by default where the hardware has it: `device_features::gpu_driven` is set when multi draw
  indirect and a GPU-side draw count are both available (Vulkan with `drawIndirectCount`, Direct3D 12, null), and
  `render_profile::gpu_culling` follows it. `draw_indexed_indirect_count` without driver support runs every slot
  instead of failing, like the backends that never had it: the culling pass zeroes unused slots.
- Pipelines compile off the render thread on Vulkan and Direct3D 12 with `device_options::async_pipelines` (it already
  did on Metal).
  The first bind waits up to 4 ms, after that draws with a pipeline still compiling are skipped instead of stalling
  the frame. `CONDUIT_SYNC_PIPELINES=1` turns it off. `device_stats::pipelines_pending` counts the ones in flight.
- Work graphs on Direct3D 12: `device_features::work_graphs` (tier 1.0, shader model 6.8),
  `device::create_work_graph` from a `lib_6_8` DXIL library with a pipeline layout as its global root signature,
  bound with `set_pipeline` and run with `command_list::dispatch_graph`. Elsewhere `create_work_graph` returns null.

- Draw, bind and barrier recording does not allocate once warmed up: command lists, barrier scratch and descriptor pools are pooled and reused.
- Redundant `set_pipeline`, `set_bind_group` and `set_vertex_buffer` calls are dropped, and dirty bind groups are bound in contiguous runs.
- Statistics are counted per command list and folded in once at submit, so recording on several threads shares no cache lines.
- Handle lookups are lock free: pool pages never move, and a generation in every handle catches stale use.
- Indirect draws use multi-draw indirect and GPU-side draw counts where the driver supports them. Otherwise they fall back to a loop.
- Pipelines compile off the render thread where the API allows it, and pipeline caches can be saved and reloaded.

## Style

`.clang-format` and `.clang-tidy` define the house style: snake_case everywhere, a trailing underscore on private members,
and no `k` prefix on constants.
