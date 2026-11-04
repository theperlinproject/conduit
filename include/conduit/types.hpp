#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace conduit {
    enum class api : uint8_t { none, vulkan, d3d12, d3d11, metal, webgpu };

    constexpr const char* api_name(conduit::api api) {
        switch (api) {
        case conduit::api::vulkan:
            return "Vulkan";
        case conduit::api::d3d12:
            return "Direct3D 12";
        case conduit::api::d3d11:
            return "Direct3D 11";
        case conduit::api::metal:
            return "Metal";
        case conduit::api::webgpu:
            return "WebGPU";
        default:
            return "None";
        }
    }

    constexpr const char* api_id(conduit::api api) {
        switch (api) {
        case conduit::api::vulkan:
            return "vulkan";
        case conduit::api::d3d12:
            return "d3d12";
        case conduit::api::d3d11:
            return "d3d11";
        case conduit::api::metal:
            return "metal";
        case conduit::api::webgpu:
            return "webgpu";
        default:
            return "none";
        }
    }

    api api_from_id(std::string_view id);

    // index in the low 24 bits, generation above, so a stale handle never aliases a recycled slot
    template <class Tag> struct handle {
        uint32_t value = 0;
        constexpr explicit operator bool() const { return value != 0; }
        constexpr bool operator==(const handle&) const = default;
        constexpr uint32_t index() const { return value & 0xFFFFFFu; }
        constexpr uint32_t generation() const { return value >> 24; }
    };

    using buffer_handle = handle<struct buffer_tag>;
    using texture_handle = handle<struct texture_tag>;
    using sampler_handle = handle<struct sampler_tag>;
    using shader_handle = handle<struct shader_tag>;
    using bind_layout_handle = handle<struct bind_layout_tag>;
    using bind_group_handle = handle<struct bind_group_tag>;
    using pipeline_layout_handle = handle<struct pipeline_layout_tag>;
    using pipeline_handle = handle<struct pipeline_tag>;
    using fence_handle = handle<struct fence_tag>;
    using query_set_handle = handle<struct query_set_tag>;

    enum class queue_type : uint8_t { graphics, compute, copy, count };
    constexpr size_t queue_count = static_cast<size_t>(queue_type::count);

    enum class format : uint16_t {
        unknown,
        r8_unorm,
        rg8_unorm,
        rgba8_unorm,
        rgba8_srgb,
        bgra8_unorm,
        bgra8_srgb,
        r8_uint,
        r16_uint,
        r32_uint,
        rg32_uint,
        rgba32_uint,
        r16_float,
        rg16_float,
        rgba16_float,
        r32_float,
        rg32_float,
        rgb32_float,
        rgba32_float,
        rgb10a2_unorm,
        r11g11b10_float,
        rg16_unorm,
        rgba16_unorm,
        d16_unorm,
        d32_float,
        d24_unorm_s8_uint,
        d32_float_s8_uint,
        bc1_unorm,
        bc3_unorm,
        bc7_unorm,
        count
    };

    struct format_info {
        uint8_t bytes_per_block;
        uint8_t block_width;
        uint8_t block_height;
        uint8_t channels;
        bool depth;
        bool stencil;
        bool srgb;
        bool compressed;
        bool floating;
    };
    const format_info& get_format_info(format format);
    inline bool is_depth_format(format f) {
        return get_format_info(f).depth;
    }
    inline bool has_stencil(format f) {
        return get_format_info(f).stencil;
    }

    enum class shader_stage : uint8_t {
        none = 0,
        vertex = 1,
        fragment = 2,
        compute = 4,
        all_graphics = vertex | fragment,
        all = vertex | fragment | compute
    };
    constexpr shader_stage operator|(shader_stage a, shader_stage b) {
        return static_cast<shader_stage>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }
    constexpr bool operator&(shader_stage a, shader_stage b) {
        return (static_cast<uint8_t>(a) & static_cast<uint8_t>(b)) != 0;
    }

    enum class buffer_usage : uint32_t {
        none = 0,
        vertex = 1u << 0,
        index = 1u << 1,
        uniform = 1u << 2,
        storage = 1u << 3,
        indirect = 1u << 4,
        copy_src = 1u << 5,
        copy_dst = 1u << 6,
    };
    constexpr buffer_usage operator|(buffer_usage a, buffer_usage b) {
        return static_cast<buffer_usage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    constexpr bool has(buffer_usage v, buffer_usage bit) {
        return (static_cast<uint32_t>(v) & static_cast<uint32_t>(bit)) != 0;
    }

    // upload and readback are persistently mapped, gpu_only is never visible to the cpu
    enum class memory_class : uint8_t { gpu_only, upload, readback, gpu_mappable };

    struct buffer_desc {
        uint64_t size = 0;
        buffer_usage usage = buffer_usage::none;
        memory_class memory = memory_class::gpu_only;
        uint32_t stride = 0;
        bool cross_queue = false;
        const char* name = nullptr;
    };

    enum class texture_dim : uint8_t { tex2d, tex2d_array, cube, tex3d };

    enum class texture_usage : uint32_t {
        none = 0,
        sampled = 1u << 0,
        storage = 1u << 1,
        color_target = 1u << 2,
        depth_target = 1u << 3,
        copy_src = 1u << 4,
        copy_dst = 1u << 5,
    };
    constexpr texture_usage operator|(texture_usage a, texture_usage b) {
        return static_cast<texture_usage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    constexpr bool has(texture_usage v, texture_usage bit) {
        return (static_cast<uint32_t>(v) & static_cast<uint32_t>(bit)) != 0;
    }

    struct texture_desc {
        texture_dim dim = texture_dim::tex2d;
        conduit::format format = conduit::format::unknown;
        uint32_t width = 1;
        uint32_t height = 1;
        uint32_t depth_or_layers = 1;
        uint32_t mips = 1;
        uint32_t samples = 1;
        texture_usage usage = texture_usage::sampled;
        bool cross_queue = false;
        const char* name = nullptr;
        std::array<float, 4> optimized_clear{};
    };

    struct subresource_range {
        uint32_t base_mip = 0;
        uint32_t mip_count = 0;
        uint32_t base_layer = 0;
        uint32_t layer_count = 0;
    };

    enum class filter : uint8_t { nearest, linear };
    enum class address_mode : uint8_t { repeat, mirror, clamp, border };
    enum class compare_op : uint8_t { never, less, equal, less_equal, greater, not_equal, greater_equal, always };

    struct sampler_desc {
        filter min = filter::linear;
        filter mag = filter::linear;
        filter mip = filter::linear;
        address_mode u = address_mode::clamp;
        address_mode v = address_mode::clamp;
        address_mode w = address_mode::clamp;
        float max_anisotropy = 1.0f;
        float lod_bias = 0.0f;
        float min_lod = 0.0f;
        float max_lod = 1000.0f;
        bool compare = false;
        conduit::compare_op compare_op = conduit::compare_op::less_equal;
        std::array<float, 4> border{};
        constexpr bool operator==(const sampler_desc&) const = default;
    };

    enum class shader_format : uint8_t { spirv, dxil, dxbc, msl, wgsl, hlsl };

    struct shader_desc {
        shader_stage stage = shader_stage::vertex;
        shader_format format = shader_format::spirv;
        std::span<const uint8_t> code;
        const char* entry = "main";
        const char* name = nullptr;
    };

    enum class binding_type : uint8_t {
        uniform_buffer,
        uniform_buffer_dynamic,
        storage_buffer,
        storage_buffer_rw,
        sampled_texture,
        storage_texture_rw,
        sampler
    };

    struct bind_layout_entry {
        uint32_t binding = 0;
        binding_type type = binding_type::uniform_buffer;
        shader_stage stages = shader_stage::all_graphics;
        uint32_t count = 1;
        texture_dim view_dim = texture_dim::tex2d;
        bool filterable = true;
    };

    struct bind_layout_desc {
        std::span<const bind_layout_entry> entries;
        const char* name = nullptr;
    };

    struct buffer_binding {
        buffer_handle buffer;
        uint64_t offset = 0;
        uint64_t size = 0;
    };

    struct texture_binding {
        texture_handle texture;
        subresource_range range;
        texture_dim view_dim = texture_dim::tex2d;
        conduit::format format = conduit::format::unknown;
        bool stencil = false;
    };

    struct bind_group_entry {
        uint32_t binding = 0;
        uint32_t array_index = 0;
        buffer_binding buffer;
        texture_binding texture;
        sampler_handle sampler;
    };

    struct bind_group_desc {
        bind_layout_handle layout;
        std::span<const bind_group_entry> entries;
        bool per_frame = false;
        const char* name = nullptr;
    };

    constexpr uint32_t max_bind_groups = 4;
    constexpr uint32_t max_push_constant_bytes = 128;
    constexpr uint32_t max_color_targets = 8;
    constexpr uint32_t max_vertex_streams = 8;
    constexpr uint32_t max_vertex_attributes = 16;
    constexpr uint32_t max_dynamic_offsets = 4;

    struct pipeline_layout_desc {
        std::array<bind_layout_handle, max_bind_groups> sets{};
        uint32_t push_constant_bytes = 0;
        shader_stage push_constant_stages = shader_stage::all_graphics;
        const char* name = nullptr;
    };

    enum class vertex_step : uint8_t { vertex, instance };

    struct vertex_stream_desc {
        uint32_t stride = 0;
        vertex_step step = vertex_step::vertex;
    };

    struct vertex_attribute_desc {
        uint32_t location = 0;
        uint32_t stream = 0;
        conduit::format format = conduit::format::unknown;
        uint32_t offset = 0;
    };

    enum class topology : uint8_t { triangle_list, triangle_strip, line_list, point_list };
    enum class cull_mode : uint8_t { none, back, front };
    enum class front_face : uint8_t { counter_clockwise, clockwise };

    struct raster_state {
        cull_mode cull = cull_mode::none;
        conduit::front_face front_face = conduit::front_face::counter_clockwise;
        bool wireframe = false;
        bool depth_clip = true;
        float depth_bias_constant = 0.0f;
        float depth_bias_slope = 0.0f;
        float depth_bias_clamp = 0.0f;
    };

    enum class stencil_op : uint8_t {
        keep,
        zero,
        replace,
        increment_clamp,
        decrement_clamp,
        invert,
        increment_wrap,
        decrement_wrap
    };

    struct stencil_face {
        stencil_op fail = stencil_op::keep;
        stencil_op depth_fail = stencil_op::keep;
        stencil_op pass = stencil_op::keep;
        compare_op compare = compare_op::always;
    };

    struct depth_stencil_state {
        bool depth_test = false;
        bool depth_write = false;
        compare_op depth_compare = compare_op::less_equal;
        bool stencil_test = false;
        uint8_t stencil_read_mask = 0xFF;
        uint8_t stencil_write_mask = 0xFF;
        stencil_face front;
        stencil_face back;
    };

    enum class blend_factor : uint8_t {
        zero,
        one,
        src_color,
        inv_src_color,
        src_alpha,
        inv_src_alpha,
        dst_color,
        inv_dst_color,
        dst_alpha,
        inv_dst_alpha,
        src_alpha_saturate
    };
    enum class blend_op : uint8_t { add, subtract, reverse_subtract, minimum, maximum };

    enum class color_mask : uint8_t { none = 0, r = 1, g = 2, b = 4, a = 8, rgb = 7, all = 15 };
    constexpr color_mask operator|(color_mask a, color_mask b) {
        return static_cast<color_mask>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
    }
    constexpr bool has(color_mask v, color_mask bit) {
        return (static_cast<uint8_t>(v) & static_cast<uint8_t>(bit)) != 0;
    }

    struct blend_target {
        bool enable = false;
        blend_factor src_color = blend_factor::one;
        blend_factor dst_color = blend_factor::zero;
        blend_op color_op = blend_op::add;
        blend_factor src_alpha = blend_factor::one;
        blend_factor dst_alpha = blend_factor::zero;
        blend_op alpha_op = blend_op::add;
        color_mask write_mask = color_mask::all;
    };

    struct blend_state {
        bool alpha_to_coverage = false;
        std::array<blend_target, max_color_targets> targets{};
    };

    struct graphics_pipeline_desc {
        shader_handle vertex;
        shader_handle fragment;
        pipeline_layout_handle layout;
        std::array<vertex_stream_desc, max_vertex_streams> streams{};
        uint32_t stream_count = 0;
        std::array<vertex_attribute_desc, max_vertex_attributes> attributes{};
        uint32_t attribute_count = 0;
        conduit::topology topology = conduit::topology::triangle_list;
        raster_state raster;
        depth_stencil_state depth_stencil;
        blend_state blend;
        std::array<format, max_color_targets> color_formats{};
        uint32_t color_count = 0;
        format depth_format = format::unknown;
        uint32_t samples = 1;
        const char* name = nullptr;
    };

    struct compute_pipeline_desc {
        shader_handle compute;
        pipeline_layout_handle layout;
        const char* name = nullptr;
    };

    enum class index_type : uint8_t { u16, u32 };

    enum class state : uint32_t {
        undefined = 0,
        copy_src = 1u << 0,
        copy_dst = 1u << 1,
        vertex_buffer = 1u << 2,
        index_buffer = 1u << 3,
        uniform_buffer = 1u << 4,
        indirect_argument = 1u << 5,
        shader_read_fragment = 1u << 6,
        shader_read_other = 1u << 7,
        shader_write = 1u << 8,
        color_target = 1u << 9,
        depth_write = 1u << 10,
        depth_read = 1u << 11,
        present = 1u << 12,
        resolve_src = 1u << 13,
        resolve_dst = 1u << 14,
        shader_read = shader_read_fragment | shader_read_other,
        depth_read_sampled = depth_read | shader_read
    };
    constexpr state operator|(state a, state b) {
        return static_cast<state>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    constexpr bool has(state v, state bit) {
        return (static_cast<uint32_t>(v) & static_cast<uint32_t>(bit)) != 0;
    }
    constexpr bool is_write_state(state s) {
        return has(s, state::copy_dst) || has(s, state::shader_write) || has(s, state::color_target) ||
               has(s, state::depth_write) || has(s, state::resolve_dst);
    }

    struct texture_barrier {
        texture_handle texture;
        state before = state::undefined;
        state after = state::undefined;
        subresource_range range;
        queue_type src_queue = queue_type::graphics;
        queue_type dst_queue = queue_type::graphics;
    };

    struct buffer_barrier {
        buffer_handle buffer;
        state before = state::undefined;
        state after = state::undefined;
        queue_type src_queue = queue_type::graphics;
        queue_type dst_queue = queue_type::graphics;
    };

    enum class load_op : uint8_t { load, clear, dont_care };
    enum class store_op : uint8_t { store, dont_care };

    struct color_attachment {
        texture_handle texture;
        uint32_t mip = 0;
        uint32_t layer = 0;
        load_op load = load_op::clear;
        store_op store = store_op::store;
        std::array<float, 4> clear{};
        texture_handle resolve;
    };

    struct depth_attachment {
        texture_handle texture;
        uint32_t mip = 0;
        uint32_t layer = 0;
        load_op load = load_op::clear;
        store_op store = store_op::store;
        float clear_depth = 1.0f;
        uint8_t clear_stencil = 0;
        bool read_only = false;
    };

    struct rendering_desc {
        std::array<color_attachment, max_color_targets> colors{};
        uint32_t color_count = 0;
        depth_attachment depth;
        const char* name = nullptr;
    };

    struct viewport {
        float x = 0, y = 0, width = 0, height = 0, min_depth = 0.0f, max_depth = 1.0f;
    };

    struct rect {
        int32_t x = 0, y = 0;
        uint32_t width = 0, height = 0;
    };

    struct buffer_texture_copy {
        uint64_t buffer_offset = 0;
        uint32_t buffer_row_pitch = 0;
        uint32_t mip = 0;
        uint32_t layer = 0;
        uint32_t x = 0, y = 0, z = 0;
        uint32_t width = 0, height = 0, depth = 1;
    };

    struct texture_copy {
        uint32_t src_mip = 0, src_layer = 0, src_x = 0, src_y = 0;
        uint32_t dst_mip = 0, dst_layer = 0, dst_x = 0, dst_y = 0;
        uint32_t width = 0, height = 0;
    };

    struct draw_indexed_indirect_command {
        uint32_t index_count;
        uint32_t instance_count;
        uint32_t first_index;
        int32_t vertex_offset;
        uint32_t first_instance;
    };
    static_assert(sizeof(draw_indexed_indirect_command) == 20);

    struct draw_indirect_command {
        uint32_t vertex_count;
        uint32_t instance_count;
        uint32_t first_vertex;
        uint32_t first_instance;
    };
    static_assert(sizeof(draw_indirect_command) == 16);

    struct dispatch_indirect_command {
        uint32_t x, y, z;
    };

    struct query_set_desc {
        uint32_t count = 0;
        const char* name = nullptr;
    };

    enum class present_mode : uint8_t { fifo, mailbox, immediate };

    struct native_window {
        void* window = nullptr;
        void* display = nullptr;
        bool wayland = false;
    };

    struct swapchain_desc {
        native_window window;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t image_count = 3;
        present_mode present = present_mode::fifo;
        bool hdr = false;
        bool allow_sampling = false;
    };

    struct native_handles {
        conduit::api api = conduit::api::none;
        void* instance = nullptr;
        void* physical_device = nullptr;
        void* device = nullptr;
        void* graphics_queue = nullptr;
        void* compute_queue = nullptr;
        uint32_t graphics_family = 0;
        uint32_t compute_family = 0;
        void* adapter = nullptr;
        void* factory = nullptr;
        void* get_instance_proc_addr = nullptr;
        void* get_device_proc_addr = nullptr;
    };

    struct native_texture {
        void* resource = nullptr;
        void* view = nullptr;
        void* memory = nullptr;
        uint32_t state = 0;
        uint32_t native_format = 0;
        uint32_t width = 0, height = 0, mips = 1, layers = 1;
        uint32_t usage = 0, flags = 0;
        uint32_t aspect = 0;
    };

    struct memory_budget {
        uint64_t device_budget = 0;
        uint64_t device_usage = 0;
        uint64_t host_budget = 0;
        uint64_t host_usage = 0;
        bool valid = false;
    };

    struct device_stats {
        uint64_t draws = 0;
        uint64_t dispatches = 0;
        uint64_t indirect_draws = 0;
        uint64_t barriers = 0;
        uint64_t pipeline_binds = 0;
        uint64_t bind_group_binds = 0;
        uint64_t submits = 0;
        uint64_t buffer_bytes = 0;
        uint64_t texture_bytes = 0;
        uint32_t live_buffers = 0;
        uint32_t live_textures = 0;
        uint32_t live_pipelines = 0;
        uint32_t live_bind_groups = 0;
        uint64_t pipelines_pending = 0;
    };
}
