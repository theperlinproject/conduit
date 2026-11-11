#pragma once
#include <conduit/types.hpp>

#include <functional>
#include <memory>
#include <optional>

namespace conduit {
    enum class adapter_type : uint8_t { unknown, discrete, integrated, virtual_gpu, cpu };

    namespace vendor_id {
        constexpr uint32_t nvidia = 0x10DE;
        constexpr uint32_t amd = 0x1002;
        constexpr uint32_t intel = 0x8086;
        constexpr uint32_t apple = 0x106B;
        constexpr uint32_t qualcomm = 0x5143;
        constexpr uint32_t arm = 0x13B5;
        constexpr uint32_t microsoft = 0x1414;
    }

    const char* vendor_name(uint32_t vendor_id);

    struct adapter_info {
        conduit::api api = conduit::api::none;
        uint32_t index = 0;
        std::string name;
        uint32_t vendor_id = 0;
        uint32_t device_id = 0;
        adapter_type type = adapter_type::unknown;
        uint64_t dedicated_vram = 0;
        uint64_t shared_memory = 0;
        uint64_t driver_version = 0;
        std::string driver_name;
        uint32_t api_version = 0;
        uint64_t luid = 0;
        bool usable = true;
        std::string unusable_reason;
    };

    enum class format_use : uint32_t {
        none = 0,
        sampled = 1u << 0,
        sampled_linear = 1u << 1,
        color_target = 1u << 2,
        blendable = 1u << 3,
        depth_target = 1u << 4,
        storage = 1u << 5,
        msaa = 1u << 6,
    };
    constexpr format_use operator|(format_use a, format_use b) {
        return static_cast<format_use>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    }
    constexpr bool has(format_use v, format_use bit) {
        return (static_cast<uint32_t>(v) & static_cast<uint32_t>(bit)) != 0;
    }

    struct device_features {
        bool multi_draw_indirect = false;
        bool draw_indirect_count = false;
        bool draw_indirect_first_instance = false;
        bool async_compute = false;
        bool dedicated_copy_queue = false;
        bool timestamp_queries = false;
        bool pipeline_cache_blob = false;
        bool resizable_bar = false;
        bool unified_memory = false;
        bool texture_compression_bc = false;
        bool sampler_anisotropy = false;
        bool shader_float16 = false;
        bool shader_int64 = false;
        bool shader_float64 = false;
        bool subgroup_ops = false;
        bool mesh_shaders = false;
        bool ray_tracing = false;
        bool variable_rate_shading = false;
        bool cooperative_vector = false;
        bool cooperative_matrix = false;
        bool tearing = false;
        bool hdr_output = false;
        bool enhanced_barriers = false;
        bool bindless = false;
        bool conservative_raster = false;
        bool depth_clip_control = false;
        bool buffer_device_address = false;
        // d3d12 work graphs tier 1.0 (shader model 6.8) or VK_AMDX_shader_enqueue; detection only, no api yet
        bool work_graphs = false;
        // everything gpu-driven rendering needs: compute writes indirect arguments and a gpu-side count
        // chooses how many run (multi draw indirect + draw indirect count + storage buffers in compute)
        bool gpu_driven = false;
    };

    struct device_limits {
        uint32_t max_texture2d = 4096;
        uint32_t max_texture_layers = 256;
        uint32_t max_bind_groups = 4;
        uint32_t max_push_constant_bytes = 128;
        uint32_t max_sampled_textures_per_stage = 16;
        uint32_t max_color_targets = 8;
        uint32_t max_msaa_samples = 1;
        uint32_t min_subgroup_size = 0;
        uint32_t max_subgroup_size = 0;
        uint32_t uniform_buffer_alignment = 256;
        uint32_t storage_buffer_alignment = 256;
        uint32_t copy_row_pitch_alignment = 256;
        uint32_t copy_offset_alignment = 512;
        uint64_t max_buffer_size = 1ull << 30;
        float max_anisotropy = 1.0f;
        float timestamp_period_ns = 1.0f;
    };

    struct device_caps {
        adapter_info adapter;
        device_features features;
        device_limits limits;
        std::array<format_use, static_cast<size_t>(format::count)> formats{};
        format_use format_support(format f) const { return formats[static_cast<size_t>(f)]; }
    };

    struct vendor_hooks {
        void* vk_get_instance_proc_addr = nullptr;
        void* d3d12_create_device = nullptr;
        void* dxgi_create_factory = nullptr;
        std::vector<const char*> instance_extensions;
        std::vector<const char*> device_extensions;
        void* device_feature_chain = nullptr;
    };

    struct instance_options {
        bool validation = false;
        const char* app_name = "conduit";
        vendor_hooks hooks;
    };

    struct device_options {
        uint32_t adapter_index = 0;
        bool validation = false;
        uint32_t frames_in_flight = 2;
        bool state_validation = false;
        std::span<const uint8_t> pipeline_cache_data;
        bool allow_async_compute = true;
        bool async_pipelines = false;
        vendor_hooks hooks;
    };

    struct work_graph_desc {
        std::span<const uint8_t> library;
        const char* program = "graph";
        pipeline_layout_handle layout;
        const char* name = nullptr;
    };

    class swapchain {
    public:
        virtual ~swapchain() = default;

        struct image {
            texture_handle texture;
            uint32_t index = 0;
        };

        virtual bool acquire(image& out) = 0;
        virtual void present() = 0;
        virtual bool resize(uint32_t width, uint32_t height) = 0;
        virtual void set_present_mode(present_mode mode) = 0;
        virtual format get_format() const = 0;
        virtual uint32_t width() const = 0;
        virtual uint32_t height() const = 0;
        virtual present_mode get_present_mode() const = 0;
        virtual uint32_t image_count() const = 0;
        virtual bool is_hdr() const { return false; }
    };

    class command_list {
    public:
        virtual ~command_list() = default;

        virtual queue_type queue() const = 0;

        virtual void barrier(std::span<const texture_barrier> textures, std::span<const buffer_barrier> buffers) = 0;
        void barrier(const texture_barrier& t) { barrier(std::span<const texture_barrier>(&t, 1), {}); }
        void barrier(const buffer_barrier& b) { barrier({}, std::span<const buffer_barrier>(&b, 1)); }
        virtual void uav_barrier(buffer_handle buffer) = 0;
        virtual void uav_barrier(texture_handle texture) = 0;

        virtual void copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src, uint64_t src_offset,
                                 uint64_t size) = 0;
        virtual void copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                            std::span<const buffer_texture_copy> regions) = 0;
        virtual void copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                            std::span<const buffer_texture_copy> regions) = 0;
        virtual void copy_texture(texture_handle dst, texture_handle src, std::span<const texture_copy> regions) = 0;
        virtual void fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) = 0;
        virtual void resolve_texture(texture_handle dst, texture_handle src) = 0;

        virtual void begin_rendering(const rendering_desc& desc) = 0;
        virtual void end_rendering() = 0;

        virtual void set_pipeline(pipeline_handle pipeline) = 0;
        virtual void set_bind_group(uint32_t set, bind_group_handle group,
                                    std::span<const uint32_t> dynamic_offsets = {}) = 0;
        virtual void push_constants(const void* data, uint32_t bytes, uint32_t offset = 0) = 0;
        virtual void set_viewport(const viewport& viewport) = 0;
        virtual void set_scissor(const rect& rect) = 0;
        virtual void set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t offset = 0) = 0;
        virtual void set_index_buffer(buffer_handle buffer, index_type type, uint64_t offset = 0) = 0;
        virtual void set_stencil_reference(uint32_t value) = 0;

        virtual void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                          uint32_t first_instance) = 0;
        virtual void draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                                  int32_t vertex_offset, uint32_t first_instance) = 0;
        virtual void draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws,
                                           uint32_t stride) = 0;
        virtual void draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle count_buffer,
                                                 uint64_t count_offset, uint32_t max_draws, uint32_t stride) = 0;

        virtual void dispatch(uint32_t x, uint32_t y, uint32_t z) = 0;
        // runs the work graph bound with set_pipeline from its entry node 0, fed with records from cpu memory
        // (copied at record time). only where caps().features.work_graphs and create_work_graph succeeded
        virtual void dispatch_graph(const void* records, uint32_t count, uint32_t stride) {
            (void)records;
            (void)count;
            (void)stride;
        }
        virtual void dispatch_indirect(buffer_handle args, uint64_t offset) = 0;

        virtual void write_timestamp(query_set_handle set, uint32_t index) = 0;
        virtual void resolve_queries(query_set_handle set, uint32_t first, uint32_t count, buffer_handle dst,
                                     uint64_t dst_offset) = 0;

        virtual void begin_debug_label(const char* name, uint32_t rgba = 0xFFFFFFFFu) = 0;
        virtual void end_debug_label() = 0;

        virtual void* native_handle() = 0;
        // call after vendor code recorded into the native command list
        virtual void invalidate_bound_state() = 0;
    };

    struct fence_op {
        fence_handle fence;
        uint64_t value = 0;
    };

    struct submit_desc {
        std::span<command_list* const> lists;
        std::span<const fence_op> waits;
        std::span<const fence_op> signals;
    };

    class device {
    public:
        virtual ~device() = default;

        virtual api get_api() const = 0;
        virtual const device_caps& caps() const = 0;
        virtual native_handles native() const = 0;
        virtual native_texture native_of(texture_handle texture, conduit::state state) {
            (void)texture;
            (void)state;
            return {};
        }

        // wraps a native texture (ID3D12Resource*, ID3D11Texture2D*) without taking ownership; on d3d12 it is
        // tracked as starting in the common/present state
        virtual texture_handle import_texture(void* native, const texture_desc& desc) {
            (void)native;
            (void)desc;
            return {};
        }

        virtual buffer_handle create_buffer(const buffer_desc& desc) = 0;
        virtual texture_handle create_texture(const texture_desc& desc) = 0;
        virtual sampler_handle create_sampler(const sampler_desc& desc) = 0;
        virtual shader_handle create_shader(const shader_desc& desc) = 0;
        virtual bind_layout_handle create_bind_layout(const bind_layout_desc& desc) = 0;
        virtual bind_group_handle create_bind_group(const bind_group_desc& desc) = 0;
        virtual pipeline_layout_handle create_pipeline_layout(const pipeline_layout_desc& desc) = 0;
        virtual pipeline_handle create_graphics_pipeline(const graphics_pipeline_desc& desc) = 0;
        virtual pipeline_handle create_compute_pipeline(const compute_pipeline_desc& desc) = 0;
        // a work graph from a DXIL library (lib_6_8) whose global root signature is the given layout: bound like a
        // compute pipeline, run with command_list::dispatch_graph. null where work graphs are not supported
        virtual pipeline_handle create_work_graph(const work_graph_desc& desc) {
            (void)desc;
            return {};
        }
        virtual query_set_handle create_query_set(const query_set_desc& desc) = 0;

        // destruction is deferred until every queue has finished the work submitted so far
        virtual void destroy(buffer_handle h) = 0;
        virtual void destroy(texture_handle h) = 0;
        virtual void destroy(sampler_handle h) = 0;
        virtual void destroy(shader_handle h) = 0;
        virtual void destroy(bind_layout_handle h) = 0;
        virtual void destroy(bind_group_handle h) = 0;
        virtual void destroy(pipeline_layout_handle h) = 0;
        virtual void destroy(pipeline_handle h) = 0;
        virtual void destroy(query_set_handle h) = 0;

        virtual const buffer_desc* get_desc(buffer_handle h) const = 0;
        virtual const texture_desc* get_desc(texture_handle h) const = 0;

        virtual uint8_t* map(buffer_handle buffer) = 0;
        virtual void flush(buffer_handle buffer, uint64_t offset, uint64_t size) = 0;
        virtual void invalidate(buffer_handle buffer, uint64_t offset, uint64_t size) = 0;
        virtual uint64_t gpu_address(buffer_handle buffer) const = 0;

        virtual fence_handle create_fence(uint64_t initial_value = 0) = 0;
        virtual void destroy(fence_handle h) = 0;
        virtual uint64_t fence_value(fence_handle fence) = 0;
        virtual bool wait_fence(fence_handle fence, uint64_t value, uint64_t timeout_ns = ~0ull) = 0;
        virtual void signal_fence(fence_handle fence, uint64_t value) = 0;

        // pooled: valid until passed to submit, recycled once its work completes
        virtual command_list* acquire_command_list(queue_type queue) = 0;
        virtual void submit(queue_type queue, const submit_desc& desc) = 0;
        void submit(queue_type queue, command_list* list) {
            submit(queue, submit_desc{std::span<command_list* const>(&list, 1), {}, {}});
        }

        virtual std::unique_ptr<swapchain> create_swapchain(const swapchain_desc& desc) = 0;

        // blocks until the frame that last used this slot has finished on the gpu
        virtual void begin_frame() = 0;
        virtual void end_frame() = 0;
        virtual uint64_t frame_number() const = 0;
        virtual uint32_t frame_slot() const = 0;
        virtual void wait_idle() = 0;

        virtual memory_budget query_memory_budget() = 0;
        virtual device_stats stats() const = 0;
        virtual std::vector<uint8_t> serialize_pipeline_cache() { return {}; }
        virtual std::string debug_report() { return {}; }

        virtual float timestamp_period_ns() const = 0;
    };

    struct adopt_desc {
        void* device = nullptr;
        void* queue = nullptr;
    };

    class instance {
    public:
        virtual ~instance() = default;
        virtual api get_api() const = 0;
        virtual const std::vector<adapter_info>& adapters() const = 0;
        virtual std::unique_ptr<device> create_device(const device_options& options) = 0;
        // wraps a device the host already created: d3d12 takes ID3D12Device + its direct ID3D12CommandQueue,
        // d3d11 takes ID3D11Device (queue unused). the host keeps ownership; null when the api cannot adopt
        virtual std::unique_ptr<device> adopt_device(const adopt_desc& desc, const device_options& options) {
            (void)desc;
            (void)options;
            return nullptr;
        }
    };

    // null when the api is not compiled in or has no driver on this machine
    std::unique_ptr<instance> create_instance(api api, const instance_options& options = {});
    bool api_compiled_in(api api);
    std::vector<api> compiled_apis();
}
