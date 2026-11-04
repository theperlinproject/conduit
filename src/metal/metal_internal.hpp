#pragma once
#include <chrono>
#include <condition_variable>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <conduit/device.hpp>
#include "common/handle_pool.hpp"
#include "common/shader_reflect.hpp"
#include "common/util.hpp"
#include "backends.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace conduit::metal {
    extern bool g_depth24_stencil8;

    MTLPixelFormat to_pixel_format(format f);
    MTLVertexFormat to_vertex_format(format f);
    MTLCompareFunction to_compare(compare_op op);
    MTLStencilOperation to_stencil_op(stencil_op op);
    MTLBlendFactor to_blend_factor(blend_factor f);
    MTLBlendOperation to_blend_op(blend_op op);
    MTLSamplerAddressMode to_address(address_mode m);
    MTLSamplerMinMagFilter to_min_mag(filter f);
    MTLSamplerMipFilter to_mip(filter f);
    MTLPrimitiveType to_primitive(topology t);
    MTLTextureType to_texture_type(texture_dim dim, bool multisample, bool array);

    struct buffer {
        id<MTLBuffer> native = nil;
        buffer_desc desc;
        std::string name;
        uint64_t size = 0;
        uint8_t* contents = nullptr;
    };

    struct view_key {
        uint8_t dim = 0;
        uint8_t stencil = 0;
        uint16_t format = 0;
        uint32_t base_mip = 0, mip_count = 0, base_layer = 0, layer_count = 0;
        auto operator<=>(const view_key&) const = default;
    };

    struct texture {
        id<MTLTexture> native = nil;
        texture_desc desc;
        std::string name;
        bool external = false;
        std::map<view_key, id<MTLTexture>> views;
    };

    struct sampler {
        id<MTLSamplerState> state = nil;
        sampler_desc desc;
    };

    struct shader {
        shader_stage stage = shader_stage::vertex;
        std::string source;
        std::string entry;
        shader_interface iface;
        uint32_t threads[3] = {1, 1, 1};
    };

    struct bind_layout {
        std::vector<bind_layout_entry> entries;
    };

    struct bind_group {
        bind_layout_handle layout;
        std::vector<bind_group_entry> entries;
    };

    struct pipeline_layout {
        pipeline_layout_desc desc;
    };

    struct stage_slot {
        uint32_t set = 0;
        uint32_t binding = 0;
        resource_kind kind = resource_kind::uniform_buffer;
        texture_dim dim = texture_dim::tex2d;
        uint32_t index = 0;
        uint32_t dynamic_index = ~0u;
    };

    enum stage_index : uint32_t { stage_vertex = 0, stage_fragment = 1, stage_compute = 2 };

    constexpr uint32_t vertex_buffer_base = 30;

    // raised from a completion handler; waiters sleep instead of spinning a core for the whole gpu frame
    struct timeline {
        std::atomic<uint64_t> value{0};
        std::mutex mutex;
        std::condition_variable cv;

        void raise(uint64_t v) {
            uint64_t cur = value.load();
            while (cur < v && !value.compare_exchange_weak(cur, v)) {}
            std::lock_guard lock(mutex);
            cv.notify_all();
        }
        bool wait(uint64_t v, uint64_t timeout_ns) {
            if (value.load(std::memory_order_acquire) >= v)
                return true;
            std::unique_lock lock(mutex);
            const auto ready = [&] { return value.load(std::memory_order_acquire) >= v; };
            if (timeout_ns == ~0ull) {
                cv.wait(lock, ready);
                return true;
            }
            return cv.wait_for(lock, std::chrono::nanoseconds(timeout_ns), ready);
        }
    };

    struct pipeline_build {
        std::atomic<int> state{0};
        std::atomic<bool> waited{false};
        std::atomic<bool> reported{false};
        id<MTLRenderPipelineState> render = nil;
        std::string error;
        timeline done;
    };

    struct pipeline {
        bool compute = false;
        id<MTLRenderPipelineState> render = nil;
        std::shared_ptr<pipeline_build> build;
        id<MTLComputePipelineState> comp = nil;
        id<MTLDepthStencilState> depth_stencil = nil;
        MTLCullMode cull = MTLCullModeNone;
        MTLWinding winding = MTLWindingCounterClockwise;
        MTLTriangleFillMode fill = MTLTriangleFillModeFill;
        MTLDepthClipMode clip = MTLDepthClipModeClip;
        float bias_constant = 0, bias_slope = 0, bias_clamp = 0;
        MTLPrimitiveType primitive = MTLPrimitiveTypeTriangle;
        uint32_t stream_count = 0;
        MTLSize threads = {1, 1, 1};
        std::array<std::vector<stage_slot>, 3> stages;
    };

    struct query {
        uint32_t count = 0;
    };

    struct fence {
        std::shared_ptr<timeline> value = std::make_shared<timeline>();
    };

    class device;

    class cmd_list final : public conduit::command_list {
    public:
        cmd_list(metal::device& device, queue_type queue) : device_(device), queue_(queue) {}
        ~cmd_list() override;

        queue_type queue() const override { return queue_; }
        void begin();
        id<MTLCommandBuffer> finish();

        void barrier(std::span<const texture_barrier> textures, std::span<const buffer_barrier> buffers) override;
        void uav_barrier(buffer_handle) override;
        void uav_barrier(texture_handle) override;
        void copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src, uint64_t src_offset,
                         uint64_t size) override;
        void copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                    std::span<const buffer_texture_copy> regions) override;
        void copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                    std::span<const buffer_texture_copy> regions) override;
        void copy_texture(texture_handle dst, texture_handle src, std::span<const texture_copy> regions) override;
        void fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) override;
        void resolve_texture(texture_handle dst, texture_handle src) override;
        void begin_rendering(const rendering_desc& desc) override;
        void end_rendering() override;
        void set_pipeline(pipeline_handle pipeline) override;
        void set_bind_group(uint32_t set, bind_group_handle group, std::span<const uint32_t> dynamic_offsets) override;
        void push_constants(const void* data, uint32_t bytes, uint32_t offset) override;
        void set_viewport(const viewport& viewport) override;
        void set_scissor(const rect& rect) override;
        void set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t offset) override;
        void set_index_buffer(buffer_handle buffer, index_type type, uint64_t offset) override;
        void set_stencil_reference(uint32_t value) override;
        void draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                  uint32_t first_instance) override;
        void draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset,
                          uint32_t first_instance) override;
        void draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws, uint32_t stride) override;
        void draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle count_buffer,
                                         uint64_t count_offset, uint32_t max_draws, uint32_t stride) override;
        void dispatch(uint32_t x, uint32_t y, uint32_t z) override;
        void dispatch_indirect(buffer_handle args, uint64_t offset) override;
        void write_timestamp(query_set_handle set, uint32_t index) override;
        void resolve_queries(query_set_handle set, uint32_t first, uint32_t count, buffer_handle dst,
                             uint64_t dst_offset) override;
        void begin_debug_label(const char* name, uint32_t rgba) override;
        void end_debug_label() override;
        void* native_handle() override { return (__bridge void*)command_buffer_; }
        void invalidate_bound_state() override {
            dirty_sets_ = 0xF;
            pipeline_dirty_ = true;
            vertex_dirty_ = 0xFF;
        }

    private:
        void end_encoder();
        id<MTLBlitCommandEncoder> blit();
        id<MTLComputeCommandEncoder> compute();
        bool prepare_draw();
        bool bind_sets(const pipeline& p);
        void apply_pipeline_state(const pipeline& p, id<MTLRenderPipelineState> state);

        device& device_;
        queue_type queue_;
        id<MTLCommandBuffer> command_buffer_ = nil;
        id<MTLRenderCommandEncoder> render_ = nil;
        id<MTLComputeCommandEncoder> compute_ = nil;
        id<MTLBlitCommandEncoder> blit_ = nil;

        pipeline_handle pipeline_;
        struct bound_set {
            bind_group_handle group;
            std::array<uint32_t, max_dynamic_offsets> offsets{};
            uint32_t offset_count = 0;
        };
        std::array<bound_set, max_bind_groups> sets_{};
        uint32_t dirty_sets_ = 0;
        bool pipeline_dirty_ = true;
        struct vertex_slot {
            buffer_handle buffer;
            uint64_t offset = 0;
        };
        std::array<vertex_slot, max_vertex_streams> vertex_{};
        uint32_t vertex_dirty_ = 0;
        buffer_handle index_buffer_;
        index_type index_type_ = index_type::u32;
        uint64_t index_offset_ = 0;
        viewport viewport_;
        rect scissor_;
        bool viewport_set_ = false, scissor_set_ = false;
        uint32_t stencil_ref_ = 0;
        uint32_t pass_width_ = 0, pass_height_ = 0;
    };

    class device final : public conduit::device {
    public:
        device(id<MTLDevice> dev, const device_options& options, const adapter_info& info);
        ~device() override;

        bool init();

        api get_api() const override { return api::metal; }
        const device_caps& caps() const override { return caps_; }
        native_handles native() const override;

        buffer_handle create_buffer(const buffer_desc& desc) override;
        texture_handle create_texture(const texture_desc& desc) override;
        sampler_handle create_sampler(const sampler_desc& desc) override;
        shader_handle create_shader(const shader_desc& desc) override;
        bind_layout_handle create_bind_layout(const bind_layout_desc& desc) override;
        bind_group_handle create_bind_group(const bind_group_desc& desc) override;
        pipeline_layout_handle create_pipeline_layout(const pipeline_layout_desc& desc) override;
        pipeline_handle create_graphics_pipeline(const graphics_pipeline_desc& desc) override;
        pipeline_handle create_compute_pipeline(const compute_pipeline_desc& desc) override;
        query_set_handle create_query_set(const query_set_desc& desc) override;

        void destroy(buffer_handle h) override;
        void destroy(texture_handle h) override;
        void destroy(sampler_handle h) override;
        void destroy(shader_handle h) override;
        void destroy(bind_layout_handle h) override;
        void destroy(bind_group_handle h) override;
        void destroy(pipeline_layout_handle h) override;
        void destroy(pipeline_handle h) override;
        void destroy(query_set_handle h) override;
        void destroy(fence_handle h) override;

        const buffer_desc* get_desc(buffer_handle h) const override;
        const texture_desc* get_desc(texture_handle h) const override;

        uint8_t* map(buffer_handle buffer) override;
        void flush(buffer_handle, uint64_t, uint64_t) override {}
        void invalidate(buffer_handle, uint64_t, uint64_t) override {}
        uint64_t gpu_address(buffer_handle) const override { return 0; }

        fence_handle create_fence(uint64_t initial_value) override;
        uint64_t fence_value(fence_handle fence) override;
        bool wait_fence(fence_handle fence, uint64_t value, uint64_t timeout_ns) override;
        void signal_fence(fence_handle fence, uint64_t value) override;

        command_list* acquire_command_list(queue_type queue) override;
        void submit(queue_type queue, const submit_desc& desc) override;
        std::unique_ptr<swapchain> create_swapchain(const swapchain_desc& desc) override;

        void begin_frame() override;
        void end_frame() override;
        uint64_t frame_number() const override { return frame_; }
        uint32_t frame_slot() const override {
            return static_cast<uint32_t>(frame_ % (std::max)(1u, options_.frames_in_flight));
        }
        void wait_idle() override;

        memory_budget query_memory_budget() override;
        device_stats stats() const override { return stats_; }
        float timestamp_period_ns() const override { return 1.0f; }

        id<MTLDevice> dev() const { return device_; }
        id<MTLCommandQueue> queue() const { return queue_; }
        const device_options& options() const { return options_; }
        device_stats& mutable_stats() { return stats_; }
        void report_error(const std::string& message);
        id<MTLTexture> view_of(texture& texture, const subresource_range& range, texture_dim dim, format format,
                               bool stencil);
        texture_handle create_surface_texture(format format, uint32_t width, uint32_t height, bool sampled);
        void set_surface_texture(texture_handle handle, id<MTLTexture> texture);
        void resize_surface_texture(texture_handle handle, uint32_t width, uint32_t height);
        void signal_when_done(id<MTLCommandBuffer> after, fence_handle fence, uint64_t value);

        handle_pool<buffer_tag, buffer> buffers_;
        handle_pool<texture_tag, texture> textures_;
        handle_pool<sampler_tag, sampler> samplers_;
        handle_pool<shader_tag, shader> shaders_;
        handle_pool<bind_layout_tag, bind_layout> bind_layouts_;
        handle_pool<bind_group_tag, bind_group> bind_groups_;
        handle_pool<pipeline_layout_tag, pipeline_layout> pipeline_layouts_;
        handle_pool<pipeline_tag, pipeline> pipelines_;
        handle_pool<query_set_tag, query> queries_;
        handle_pool<fence_tag, fence> fences_;

    public:
        std::mutex view_mutex_;

    private:
        void fill_caps();
        id<MTLFunction> compile_stage(const shader& shader, const pipeline_layout_desc& layout,
                                      std::vector<stage_slot>& slots, std::string& error);
        std::string prepare_stage(const shader& shader, const pipeline_layout_desc& layout,
                                  std::vector<stage_slot>& slots, std::string& error);
        id<MTLFunction> load_function(const std::string& source, const std::string& entry, std::string& error);

        id<MTLDevice> device_ = nil;
        id<MTLCommandQueue> queue_ = nil;
        device_options options_;
        device_caps caps_;
        device_stats stats_;
        uint64_t frame_ = 0;
        fence_handle frame_fence_;
        std::mutex list_mutex_;
        std::vector<std::unique_ptr<cmd_list>> lists_;
        std::vector<cmd_list*> free_lists_;
        std::mutex library_mutex_;
        std::unordered_map<std::string, id<MTLLibrary>> libraries_;
        std::mutex error_mutex_;
        dispatch_queue_t compile_queue_ = nullptr;
        dispatch_group_t compile_group_ = nullptr;

    public:
        id<MTLRenderPipelineState> ready_pipeline(const pipeline& p);
        void wait_for_compiles();
    };

    std::unique_ptr<conduit::swapchain> make_swapchain(device& device, const swapchain_desc& desc);
}
