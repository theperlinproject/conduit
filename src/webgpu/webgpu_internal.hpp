#pragma once
#include "backends.hpp"
#include "common/counters.hpp"
#include "common/handle_pool.hpp"
#include "common/shader_reflect.hpp"
#include "common/util.hpp"

#include <webgpu/webgpu.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace conduit::webgpu {
    inline WGPUStringView view(const char* s) {
        return WGPUStringView{s, s ? WGPU_STRLEN : 0};
    }
    inline WGPUStringView view(const std::string& s) {
        return WGPUStringView{s.data(), s.size()};
    }
    inline std::string str(WGPUStringView v) {
        if (!v.data)
            return {};
        return std::string(v.data, v.length == WGPU_STRLEN ? std::char_traits<char>::length(v.data) : v.length);
    }

    inline void release(WGPUBuffer v) {
        wgpuBufferRelease(v);
    }
    inline void release(WGPUTexture v) {
        wgpuTextureRelease(v);
    }
    inline void release(WGPUTextureView v) {
        wgpuTextureViewRelease(v);
    }
    inline void release(WGPUSampler v) {
        wgpuSamplerRelease(v);
    }
    inline void release(WGPUShaderModule v) {
        wgpuShaderModuleRelease(v);
    }
    inline void release(WGPUBindGroupLayout v) {
        wgpuBindGroupLayoutRelease(v);
    }
    inline void release(WGPUBindGroup v) {
        wgpuBindGroupRelease(v);
    }
    inline void release(WGPUPipelineLayout v) {
        wgpuPipelineLayoutRelease(v);
    }
    inline void release(WGPURenderPipeline v) {
        wgpuRenderPipelineRelease(v);
    }
    inline void release(WGPUComputePipeline v) {
        wgpuComputePipelineRelease(v);
    }
    inline void release(WGPUQuerySet v) {
        wgpuQuerySetRelease(v);
    }
    inline void release(WGPUCommandEncoder v) {
        wgpuCommandEncoderRelease(v);
    }
    inline void release(WGPUCommandBuffer v) {
        wgpuCommandBufferRelease(v);
    }
    inline void release(WGPURenderPassEncoder v) {
        wgpuRenderPassEncoderRelease(v);
    }
    inline void release(WGPUComputePassEncoder v) {
        wgpuComputePassEncoderRelease(v);
    }
    inline void release(WGPUQueue v) {
        wgpuQueueRelease(v);
    }
    inline void release(WGPUDevice v) {
        wgpuDeviceRelease(v);
    }
    inline void release(WGPUAdapter v) {
        wgpuAdapterRelease(v);
    }
    inline void release(WGPUInstance v) {
        wgpuInstanceRelease(v);
    }
    inline void release(WGPUSurface v) {
        wgpuSurfaceRelease(v);
    }

    template <class T> class owned {
    public:
        owned() = default;
        explicit owned(T object) : object_(object) {}
        owned(owned&& o) noexcept : object_(o.object_) { o.object_ = nullptr; }
        owned& operator=(owned&& o) noexcept {
            if (this != &o) {
                reset();
                object_ = o.object_;
                o.object_ = nullptr;
            }
            return *this;
        }
        owned(const owned&) = delete;
        owned& operator=(const owned&) = delete;
        ~owned() { reset(); }

        T get() const { return object_; }
        explicit operator bool() const { return object_ != nullptr; }
        void reset() {
            if (object_) {
                release(object_);
                object_ = nullptr;
            }
        }
        T detach() {
            T t = object_;
            object_ = nullptr;
            return t;
        }

    private:
        T object_ = nullptr;
    };

    WGPUTextureFormat to_texture_format(format format);
    format from_texture_format(WGPUTextureFormat format);
    WGPUVertexFormat to_vertex_format(format format);
    WGPUBlendFactor to_blend_factor(blend_factor f);
    WGPUBlendOperation to_blend_op(blend_op op);
    WGPUCompareFunction to_compare(compare_op op);
    WGPUStencilOperation to_stencil_op(stencil_op op);
    WGPUAddressMode to_address(address_mode mode);
    WGPUFilterMode to_filter(filter f);
    WGPUMipmapFilterMode to_mip_filter(filter f);
    WGPUTextureViewDimension to_view_dimension(texture_dim dim);

    struct buffer {
        owned<WGPUBuffer> native;
        buffer_desc desc;
        std::string name;
        std::vector<uint8_t> shadow;
        uint64_t gpu_size = 0;
    };

    struct view_key {
        uint32_t base_mip = 0, mip_count = 0, base_layer = 0, layer_count = 0;
        uint16_t dim = 0;
        uint16_t format = 0;
        uint8_t aspect = 0;
        auto operator<=>(const view_key&) const = default;
    };

    struct texture {
        owned<WGPUTexture> native;
        bool external = false;
        texture_desc desc;
        std::string name;
        std::map<view_key, owned<WGPUTextureView>> views;
        std::mutex view_mutex;

        texture() = default;
        texture(texture&& o) noexcept
            : native(std::move(o.native)), external(o.external), desc(o.desc), name(std::move(o.name)),
              views(std::move(o.views)) {
            desc.name = nullptr;
        }
        texture& operator=(texture&& o) noexcept {
            native = std::move(o.native);
            external = o.external;
            desc = o.desc;
            name = std::move(o.name);
            views = std::move(o.views);
            return *this;
        }
    };

    struct sampler {
        owned<WGPUSampler> native;
        sampler_desc desc;
        bool linear = false;
    };

    struct shader {
        shader_desc desc;
        std::string source;
        std::string entry;
        owned<WGPUShaderModule> module;
        shader_interface iface;
    };

    struct bind_layout {
        std::vector<bind_layout_entry> entries;
    };

    // native groups are made at first bind, from whichever thread records it: the usual single native layout
    // is published lock free, making one and further layouts take one of the device's group locks
    struct bind_group {
        bind_layout_handle layout;
        std::vector<bind_group_entry> entries;
        bool per_frame = false;
        std::atomic<WGPUBindGroupLayout> first_layout{nullptr};
        owned<WGPUBindGroup> first;
        std::map<WGPUBindGroupLayout, owned<WGPUBindGroup>> others;

        bind_group() = default;
        bind_group(bind_group&& o) noexcept
            : layout(o.layout), entries(std::move(o.entries)), per_frame(o.per_frame),
              first_layout(o.first_layout.load(std::memory_order_relaxed)), first(std::move(o.first)),
              others(std::move(o.others)) {}
        bind_group& operator=(bind_group&& o) noexcept {
            layout = o.layout;
            entries = std::move(o.entries);
            per_frame = o.per_frame;
            first_layout.store(o.first_layout.load(std::memory_order_relaxed), std::memory_order_relaxed);
            first = std::move(o.first);
            others = std::move(o.others);
            return *this;
        }
    };

    // a native group of a per-frame bind group, kept for the next frame's group with the same contents. it holds
    // a reference on each resource it names, so no new object can take one of those addresses while it lives
    struct cached_group {
        WGPUBindGroupLayout layout = nullptr;
        std::vector<WGPUBindGroupEntry> entries;
        owned<WGPUBindGroup> group;
        std::vector<owned<WGPUBuffer>> buffers;
        std::vector<owned<WGPUTextureView>> views;
        std::vector<owned<WGPUSampler>> samplers;
        uint64_t used = 0;
    };

    struct pipeline_layout {
        pipeline_layout_desc desc;
    };

    struct native_set {
        owned<WGPUBindGroupLayout> layout;
        struct entry {
            bind_layout_entry rhi;
            bool depth = false;
            bool comparison = false;
        };
        std::vector<entry> entries;
        std::vector<uint32_t> dynamic_pick;
        uint32_t rhi_dynamic_count = 0;
        owned<WGPUBindGroup> blank;
    };

    struct pipeline {
        bool compute = false;
        owned<WGPURenderPipeline> render;
        owned<WGPUComputePipeline> comp;
        pipeline_layout_handle layout_handle;
        owned<WGPUPipelineLayout> layout;
        std::array<std::shared_ptr<native_set>, max_bind_groups> sets;
        uint32_t stream_count = 0;
        uint32_t set_count = 0;
        // stride of the single per-vertex stream when the vertex shader never reads vertex_index: a new offset
        // into the same buffer can then move first_vertex instead of rebinding the buffer
        uint32_t shift_stride = 0;
    };

    struct query {
        owned<WGPUQuerySet> set;
        uint32_t count = 0;
    };

    struct fence {
        std::shared_ptr<std::atomic<uint64_t>> value = std::make_shared<std::atomic<uint64_t>>(0);
    };

    class webgpu_device;

    class webgpu_command_list final : public command_list {
    public:
        webgpu_command_list(webgpu_device& device, queue_type queue) : device_(device), queue_(queue) {}
        ~webgpu_command_list() override;

        queue_type queue() const override { return queue_; }
        void begin();
        WGPUCommandBuffer finish();

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
        bool prepare_dispatch();
        void write_timestamp(query_set_handle set, uint32_t index) override;
        void resolve_queries(query_set_handle set, uint32_t first, uint32_t count, buffer_handle dst,
                             uint64_t dst_offset) override;
        void begin_debug_label(const char* name, uint32_t rgba) override;
        void end_debug_label() override;
        void* native_handle() override { return encoder_.get(); }
        void invalidate_bound_state() override;
        const list_counters& counters() const { return counters_; }

    private:
        void end_pass();
        void forget_sent();
        bool prepare_draw();
        bool set_native_group(uint32_t set, const native_set* ns);
        void bind_index_buffer(const buffer& b);
        bool unshift_vertices();

        webgpu_device& device_;
        queue_type queue_;
        owned<WGPUCommandEncoder> encoder_;
        owned<WGPURenderPassEncoder> render_;
        owned<WGPUComputePassEncoder> compute_;
        pipeline_handle pipeline_;
        struct bound_set {
            bind_group_handle group;
            std::array<uint32_t, max_dynamic_offsets> offsets{};
            uint32_t offset_count = 0;
        };
        std::array<bound_set, max_bind_groups> sets_{};
        uint32_t dirty_sets_ = 0;
        bool pipeline_dirty_ = true;
        // what the open pass was last given: webgpu keeps groups bound across setPipeline, and a call equal to
        // the last one is dropped. groups are matched by handle and native layout, never by the raw pointer of
        // a group that may since have been freed and its address reused
        struct sent_set {
            bind_group_handle group;
            WGPUBindGroupLayout layout = nullptr;
            WGPUBindGroup native = nullptr;
            std::array<uint32_t, max_dynamic_offsets> offsets{};
            uint32_t offset_count = 0;
        };
        std::array<sent_set, max_bind_groups> sent_sets_{};
        struct vertex_slot {
            buffer_handle buffer;
            uint64_t offset = 0;
            bool operator==(const vertex_slot&) const = default;
        };
        std::array<vertex_slot, max_vertex_streams> vertex_{}, sent_vertex_{};
        uint32_t vertex_dirty_ = 0;
        uint32_t vertex_shift_ = 0;
        buffer_handle index_buffer_;
        index_type index_type_ = index_type::u32;
        uint64_t index_offset_ = 0;
        bool index_dirty_ = false;
        buffer_handle sent_index_buffer_;
        index_type sent_index_type_ = index_type::u32;
        uint64_t sent_index_offset_ = 0;
        viewport viewport_, sent_viewport_;
        bool viewport_set_ = false, viewport_dirty_ = false, viewport_sent_ = false;
        rect scissor_, sent_scissor_;
        bool scissor_set_ = false, scissor_dirty_ = false, scissor_sent_ = false;
        uint32_t stencil_ref_ = 0;
        bool stencil_dirty_ = false;
        uint32_t pass_width_ = 0, pass_height_ = 0;
        bool warned_push_ = false;
        list_counters counters_;
    };

    class webgpu_device final : public device {
    public:
        webgpu_device(owned<WGPUInstance> instance, owned<WGPUAdapter> adapter, owned<WGPUDevice> device,
                      const device_options& options, const adapter_info& info, bool map_sync);
        ~webgpu_device() override;

        bool init();

        api get_api() const override { return api::webgpu; }
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
        void flush(buffer_handle buffer, uint64_t offset, uint64_t size) override;
        void invalidate(buffer_handle buffer, uint64_t offset, uint64_t size) override;
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
        device_stats stats() const override;
        float timestamp_period_ns() const override { return 1.0f; }

        WGPUDevice dev() const { return device_.get(); }
        WGPUQueue queue() const { return queue_.get(); }
        WGPUInstance inst() const { return instance_.get(); }
        WGPUAdapter adapter() const { return adapter_.get(); }
        const device_options& options() const { return options_; }
        bool pump(const std::function<bool()>& done, uint64_t timeout_ns);
        void report_error(const std::string& message);

        WGPUTextureView view_of(texture& texture, const subresource_range& range, texture_dim dim, format format,
                                bool stencil, bool attachment);
        WGPUBindGroup native_group_for(bind_group_handle group, const native_set& set);
        std::shared_ptr<native_set> build_native_set(const pipeline_layout_desc& layout, uint32_t set,
                                                     std::span<const shader* const> shaders,
                                                     shader_stage stages_present);
        WGPUSampler nearest_sampler() const { return nearest_sampler_.get(); }
        WGPUBindGroup empty_group() const { return empty_group_.get(); }
        void set_surface_texture(texture_handle handle, WGPUTexture texture);
        texture_handle create_surface_texture(format format, uint32_t width, uint32_t height, bool sampled);
        void resize_surface_texture(texture_handle handle, uint32_t width, uint32_t height);

    private:
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

        friend class webgpu_command_list;
        void fill_caps();
        std::shared_ptr<native_set> cached_layout(const std::string& key,
                                                  const std::function<std::shared_ptr<native_set>()>& make);
        WGPUBindGroup find_cached_group(uint64_t key, WGPUBindGroupLayout layout,
                                        std::span<const WGPUBindGroupEntry> entries);
        void cache_group(uint64_t key, WGPUBindGroupLayout layout, std::span<const WGPUBindGroupEntry> entries,
                         WGPUBindGroup group);

        owned<WGPUInstance> instance_;
        owned<WGPUAdapter> adapter_;
        owned<WGPUDevice> device_;
        owned<WGPUQueue> queue_;
        owned<WGPUSampler> nearest_sampler_;
        owned<WGPUBindGroupLayout> empty_layout_;
        owned<WGPUBindGroup> empty_group_;
        device_options options_;
        device_caps caps_;
        device_counters counters_;
        std::atomic<int64_t> buffer_bytes_{0}, texture_bytes_{0};
        std::atomic<int32_t> live_buffers_{0}, live_textures_{0}, live_pipelines_{0}, live_groups_{0};
        bool map_sync_ = true;
        uint64_t frame_ = 0;
        fence_handle frame_fence_;
        std::atomic<uint64_t> submitted_frames_{0};
        // native only: the last submission of each frame in flight, to block on exactly that one
        std::atomic<uint64_t> last_submission_{0};
        std::vector<uint64_t> frame_submissions_;
        std::vector<std::unique_ptr<webgpu_command_list>> lists_;
        std::vector<webgpu_command_list*> free_lists_;
        std::mutex list_mutex_;
        std::mutex layout_mutex_;
        std::unordered_map<std::string, std::shared_ptr<native_set>> layout_cache_;
        std::array<std::mutex, 16> group_locks_;
        std::mutex group_cache_mutex_;
        std::unordered_map<uint64_t, cached_group> group_cache_;
        uint64_t group_cache_frame_ = 0;
        std::mutex error_mutex_;
        std::string first_error_;
        bool has_feature_timestamp_ = false;
    };

    std::unique_ptr<instance> make_instance(const instance_options& options);
}
