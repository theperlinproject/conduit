cmake_minimum_required(VERSION 3.28)

project(conduit
    VERSION 0.1.0
    DESCRIPTION "Cross-platform rendering hardware interface"
    HOMEPAGE_URL "https://github.com/theperlinproject/conduit"
    LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(CONDUIT_TOP_LEVEL ON)
else()
    set(CONDUIT_TOP_LEVEL OFF)
endif()

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GNUInstallDirs)

set(_conduit_windows_x64 OFF)
if(WIN32 AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$" AND NOT CMAKE_CXX_COMPILER_TARGET MATCHES "^(aarch64|arm64)")
    set(_conduit_windows_x64 ON)
endif()
set(_conduit_vulkan_default ON)
if(APPLE OR EMSCRIPTEN)
    set(_conduit_vulkan_default OFF)
endif()
set(_conduit_webgpu_default OFF)
if(EMSCRIPTEN)
    set(_conduit_webgpu_default ON)
endif()

option(CONDUIT_VULKAN "Build the Vulkan backend" ${_conduit_vulkan_default})
option(CONDUIT_D3D12 "Build the Direct3D 12 backend" ${WIN32})
option(CONDUIT_D3D11 "Build the Direct3D 11 backend" OFF)
option(CONDUIT_METAL "Build the Metal backend" ${APPLE})
option(CONDUIT_WEBGPU "Build the WebGPU backend" ${_conduit_webgpu_default})
option(CONDUIT_STREAMLINE "Build the DLSS, Reflex and frame generation layer" ${_conduit_windows_x64})
option(CONDUIT_BUILD_TESTS "Build the conduit tests" ${CONDUIT_TOP_LEVEL})
option(CONDUIT_INSTALL "Generate the install target" ${CONDUIT_TOP_LEVEL})
option(CONDUIT_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(CONDUIT_LTO "Build conduit with link-time optimization" OFF)
option(CONDUIT_CLANG_TIDY "Run clang-tidy while compiling conduit" OFF)
set(CONDUIT_DEPS_DIR "" CACHE PATH "Directory with Vulkan-Headers, volk, vma, d3d12ma, DirectX-Headers, Streamline and wgpu checkouts")

include(conduit_dependencies)
include(conduit_shaders)

set(CONDUIT_SOURCES
    src/core.cpp
    src/shader_library.cpp
    src/render_profile.cpp
    src/vendor/vendor.cpp
    src/common/state_validator.cpp
    src/common/shader_reflect.cpp
    src/null/null_device.cpp)

if(CONDUIT_VULKAN)
    list(APPEND CONDUIT_SOURCES
        src/vulkan/vulkan_instance.cpp
        src/vulkan/vulkan_device.cpp
        src/vulkan/vulkan_commands.cpp
        src/vulkan/vulkan_swapchain.cpp
        src/vulkan/vma_impl.cpp
        "${volk_SOURCE_DIR}/volk.c")
    set_source_files_properties(src/vulkan/vma_impl.cpp "${volk_SOURCE_DIR}/volk.c" PROPERTIES SKIP_LINTING ON)
endif()

if(CONDUIT_D3D12)
    list(APPEND CONDUIT_SOURCES
        src/d3d12/d3d12_instance.cpp
        src/d3d12/d3d12_device.cpp
        src/d3d12/d3d12_commands.cpp
        src/d3d12/d3d12_swapchain.cpp
        src/d3d12/d3d12ma_impl.cpp)
    if(NOT MSVC)
        list(APPEND CONDUIT_SOURCES "${directx_headers_SOURCE_DIR}/src/dxguids.cpp")
    endif()
endif()

if(CONDUIT_D3D11)
    list(APPEND CONDUIT_SOURCES
        src/d3d11/d3d11_instance.cpp
        src/d3d11/d3d11_device.cpp
        src/d3d11/d3d11_commands.cpp)
endif()

set(_conduit_metal_sources
    src/metal/metal_convert.mm
    src/metal/metal_device.mm
    src/metal/metal_commands.mm
    src/metal/metal_instance.mm)
if(CONDUIT_METAL)
    list(APPEND CONDUIT_SOURCES ${_conduit_metal_sources})
    set_source_files_properties(${_conduit_metal_sources}
        PROPERTIES LANGUAGE CXX COMPILE_OPTIONS "-x;objective-c++;-fobjc-arc")
endif()

if(CONDUIT_WEBGPU)
    list(APPEND CONDUIT_SOURCES
        src/webgpu/webgpu_convert.cpp
        src/webgpu/webgpu_device.cpp
        src/webgpu/webgpu_pipeline.cpp
        src/webgpu/webgpu_commands.cpp
        src/webgpu/webgpu_instance.cpp)
endif()

set(_conduit_with_streamline OFF)
if(CONDUIT_STREAMLINE AND WIN32)
    set(_conduit_with_streamline ON)
endif()

add_library(conduit STATIC ${CONDUIT_SOURCES})
add_library(conduit::conduit ALIAS conduit)
set_target_properties(conduit PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_EXTENSIONS OFF
    CXX_SCAN_FOR_MODULES OFF
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON)
target_compile_features(conduit PUBLIC cxx_std_23)
target_include_directories(conduit
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")

set(_conduit_msvc_like OFF)
if(MSVC OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(_conduit_msvc_like ON)
endif()
if(_conduit_msvc_like)
    set(_conduit_warnings /W4 /permissive- /Zc:__cplusplus /utf-8)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        list(APPEND _conduit_warnings /Zc:preprocessor /Zc:inline /bigobj /MP
            /wd4244 /wd4267 /wd4245 /wd4305 /wd4310 /wd4389
            /wd4456 /wd4457 /wd4458 /wd4459
            /wd4127 /wd4201 /wd4324 /wd4701 /wd4703 /wd4702 /wd4706 /wd5054)
    endif()
    if(CONDUIT_WARNINGS_AS_ERRORS)
        list(APPEND _conduit_warnings /WX)
    endif()
else()
    set(_conduit_warnings -Wall -Wextra -Wpedantic -Wno-missing-field-initializers)
    if(CONDUIT_WARNINGS_AS_ERRORS)
        list(APPEND _conduit_warnings -Werror)
    endif()
endif()
target_compile_options(conduit PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${_conduit_warnings}>")

if(CONDUIT_LTO AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"
   AND NOT CMAKE_LINKER MATCHES "lld-link")
    message(STATUS "conduit: link-time optimization with clang-cl needs CMAKE_LINKER=lld-link, building without it")
    set(CONDUIT_LTO OFF)
endif()

if(CONDUIT_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _conduit_ipo OUTPUT _conduit_ipo_error LANGUAGES CXX)
    if(_conduit_ipo)
        set_target_properties(conduit PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            target_link_options(conduit INTERFACE -fuse-ld=lld)
        endif()
    else()
        message(WARNING "conduit: link-time optimization unavailable: ${_conduit_ipo_error}")
    endif()
endif()

if(CONDUIT_CLANG_TIDY)
    find_program(CONDUIT_CLANG_TIDY_EXE NAMES clang-tidy-18 clang-tidy REQUIRED)
    set_target_properties(conduit PROPERTIES CXX_CLANG_TIDY "${CONDUIT_CLANG_TIDY_EXE}")
endif()

if(NOT EMSCRIPTEN)
    find_package(Threads REQUIRED)
    target_link_libraries(conduit PUBLIC Threads::Threads)
endif()

if(WIN32)
    target_compile_definitions(conduit PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

foreach(api VULKAN D3D12 D3D11 METAL WEBGPU)
    if(CONDUIT_${api})
        target_compile_definitions(conduit PUBLIC CONDUIT_HAS_${api}=1)
    endif()
endforeach()

if(CONDUIT_VULKAN)
    target_include_directories(conduit SYSTEM PRIVATE
        "${vulkan_headers_SOURCE_DIR}/include" "${volk_SOURCE_DIR}" "${vma_SOURCE_DIR}/include")
    target_compile_definitions(conduit PRIVATE VK_NO_PROTOTYPES)
    if(UNIX AND NOT APPLE)
        target_link_libraries(conduit PUBLIC ${CMAKE_DL_LIBS})
    endif()
endif()

if(CONDUIT_D3D12)
    target_include_directories(conduit SYSTEM PRIVATE
        "${directx_headers_SOURCE_DIR}/include" "${directx_headers_SOURCE_DIR}/include/directx"
        "${directx_headers_SOURCE_DIR}/include/dxguids" "${d3d12ma_SOURCE_DIR}/include" "${d3d12ma_SOURCE_DIR}/src")
endif()

if(_conduit_with_streamline)
    add_library(conduit_streamline OBJECT src/vendor/streamline.cpp)
    set_target_properties(conduit_streamline PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_SCAN_FOR_MODULES OFF
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        POSITION_INDEPENDENT_CODE ON)
    target_include_directories(conduit_streamline PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_include_directories(conduit_streamline SYSTEM PRIVATE "${streamline_SOURCE_DIR}/include")
    if(MINGW)
        target_include_directories(conduit_streamline SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/vendor/compat")
    endif()
    if(CONDUIT_VULKAN)
        target_include_directories(conduit_streamline SYSTEM PRIVATE "${vulkan_headers_SOURCE_DIR}/include")
    endif()
    target_compile_definitions(conduit_streamline PRIVATE
        CONDUIT_HAS_STREAMLINE=1 NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS
        $<$<BOOL:${CONDUIT_VULKAN}>:CONDUIT_HAS_VULKAN=1 VK_NO_PROTOTYPES>
        $<$<BOOL:${CONDUIT_D3D12}>:CONDUIT_HAS_D3D12=1>
        $<$<BOOL:${CONDUIT_D3D11}>:CONDUIT_HAS_D3D11=1>)
    target_compile_options(conduit_streamline PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${_conduit_warnings}>")
    if(CONDUIT_LTO AND _conduit_ipo)
        set_target_properties(conduit_streamline PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
    if(CONDUIT_CLANG_TIDY)
        set_target_properties(conduit_streamline PROPERTIES CXX_CLANG_TIDY "${CONDUIT_CLANG_TIDY_EXE}")
    endif()
    target_sources(conduit PRIVATE $<TARGET_OBJECTS:conduit_streamline>)
    target_compile_definitions(conduit PUBLIC CONDUIT_HAS_STREAMLINE=1)
endif()

if(CONDUIT_WEBGPU)
    if(EMSCRIPTEN)
        target_compile_options(conduit PUBLIC --use-port=emdawnwebgpu)
        target_link_options(conduit PUBLIC --use-port=emdawnwebgpu)
    else()
        target_include_directories(conduit SYSTEM PRIVATE
            $<BUILD_INTERFACE:${CONDUIT_WGPU_DIR}/include>
            $<BUILD_INTERFACE:${CONDUIT_WGPU_DIR}/include/webgpu>)
        target_link_libraries(conduit PUBLIC "${CONDUIT_WGPU_LIB}")
        if(WIN32)
            target_link_libraries(conduit PUBLIC ws2_32 userenv ntdll bcrypt d3dcompiler opengl32 gdi32 ole32 oleaut32 propsys runtimeobject)
        elseif(APPLE)
            target_link_libraries(conduit PUBLIC "-framework Metal" "-framework QuartzCore" "-framework Foundation"
                "-framework CoreFoundation" "-framework CoreGraphics" "-framework AppKit" "-framework IOSurface" "-framework Security" objc iconv)
        elseif(ANDROID)
            target_link_libraries(conduit PUBLIC android log m ${CMAKE_DL_LIBS})
        elseif(UNIX)
            target_link_libraries(conduit PUBLIC ${CMAKE_DL_LIBS} m)
        endif()
    endif()
endif()

if(CONDUIT_METAL)
    target_link_libraries(conduit PUBLIC "-framework Metal" "-framework QuartzCore" "-framework Foundation" "-framework CoreGraphics")
endif()

if(CONDUIT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(CONDUIT_INSTALL)
    include(CMakePackageConfigHelpers)
    install(TARGETS conduit EXPORT conduit-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(DIRECTORY include/conduit DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES tools/pack_shaders.py DESTINATION ${CMAKE_INSTALL_DATADIR}/conduit)
    install(FILES cmake/conduit_shaders.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/conduit)
    install(EXPORT conduit-targets NAMESPACE conduit:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/conduit)
    configure_package_config_file(cmake/conduit-config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/conduit-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/conduit)
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/conduit-config-version.cmake"
        COMPATIBILITY SameMinorVersion)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/conduit-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/conduit-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/conduit)
endif()
