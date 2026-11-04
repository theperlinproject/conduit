#pragma once
#include <conduit/device.hpp>
#include "common/counters.hpp"
#include "common/handle_pool.hpp"
#include "common/retire_queue.hpp"
#include "common/state_validator.hpp"
#include "common/util.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_6.h>
#include <directx/d3d12.h>

#if __has_include(<winrt/base.h>)
#define WINRT_LEAN_AND_MEAN
#include <winrt/base.h>
#else
#include "d3d12/winrt_shim.hpp"
#endif

#include "D3D12MemAlloc.h"

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace conduit::d3d12 {
    struct library {
        HMODULE d3d12 = nullptr;
        HMODULE dxgi = nullptr;
        PFN_D3D12_CREATE_DEVICE create_device = nullptr;
        PFN_D3D12_GET_DEBUG_INTERFACE get_debug_interface = nullptr;
        PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE serialize_root_signature = nullptr;
        HRESULT(WINAPI* create_factory2)(UINT, REFIID, void**) = nullptr;
    };

    struct instance_state {
        library lib;
        winrt::com_ptr<IDXGIFactory4> factory;
        bool validation = false;
        bool tearing = false;
        std::mutex message_mutex;
        std::vector<std::string> messages;
        size_t error_count = 0;
    };

    class cpu_descriptor_pool {
    public:
        bool init(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t capacity);
        uint32_t alloc();
        void free(uint32_t index);
        D3D12_CPU_DESCRIPTOR_HANDLE handle(uint32_t index) const {
            return {cpu_start_.ptr + static_cast<SIZE_T>(index) * increment_};
        }

    private:
        winrt::com_ptr<ID3D12DescriptorHeap> heap_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_start_{};
        uint32_t increment_ = 0;
        uint32_t capacity_ = 0;
        uint32_t next_ = 0;
        std::vector<uint32_t> free_list_;
        std::mutex mutex_;
    };

    class range_heap {
    public:
        bool init(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t capacity);
        bool alloc(uint32_t count, uint32_t& offset);
        void free(uint32_t offset, uint32_t count);
        D3D12_CPU_DESCRIPTOR_HANDLE cpu(uint32_t index) const {
            return {cpu_start_.ptr + static_cast<SIZE_T>(index) * increment_};
        }
        D3D12_GPU_DESCRIPTOR_HANDLE gpu(uint32_t index) const {
            return {gpu_start_.ptr + static_cast<UINT64>(index) * increment_};
        }
        ID3D12DescriptorHeap* heap() const { return heap_.get(); }
        uint32_t used() const { return used_; }

    private:
        winrt::com_ptr<ID3D12DescriptorHeap> heap_;
        D3D12_CPU_DESCRIPTOR_HANDLE cpu_start_{};
        D3D12_GPU_DESCRIPTOR_HANDLE gpu_start_{};
        uint32_t increment_ = 0;
        uint32_t used_ = 0;
        std::vector<std::pair<uint32_t, uint32_t>> free_ranges_;
        std::mutex mutex_;
    };

    struct buffer_rec {
        winrt::com_ptr<ID3D12Resource> resource;
        winrt::com_ptr<D3D12MA::Allocation> alloc;
        buffer_desc desc;
        std::string name;
        uint8_t* mapped = nullptr;
        D3D12_GPU_VIRTUAL_ADDRESS address = 0;
        D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    };

    struct view_set {
        std::unordered_map<uint64_t, uint32_t> rtv;
        std::unordered_map<uint64_t, uint32_t> dsv;
    };

    struct texture_rec {
        winrt::com_ptr<ID3D12Resource> resource;
        winrt::com_ptr<D3D12MA::Allocation> alloc;
        texture_desc desc;
        std::string name;
        DXGI_FORMAT resource_format = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT view_format = DXGI_FORMAT_UNKNOWN;
        DXGI_FORMAT dsv_format = DXGI_FORMAT_UNKNOWN;
        uint32_t planes = 1;
        uint32_t mips = 1;
        uint32_t layers = 1;
        bool external = false;
        bool simultaneous = false;
        std::vector<D3D12_RESOURCE_STATES> states;
        view_set views;
    };

    struct sampler_rec {
        sampler_desc desc;
    };

    struct shader_rec {
        std::vector<uint8_t> code;
        shader_stage stage = shader_stage::vertex;
    };

    struct bind_layout_rec {
        std::vector<bind_layout_entry> entries;
        std::vector<uint32_t> table_offset;
        uint32_t table_count = 0;
        uint32_t sampler_count = 0;
        uint32_t dynamic_count = 0;
        std::vector<uint32_t> dynamic_bindings;
    };

    struct sampler_table {
        uint32_t offset = 0;
        uint32_t count = 0;
        uint32_t refs = 0;
        std::vector<uint32_t> key;
    };

    struct bind_group_rec {
        bind_layout_handle layout;
        uint32_t table_offset = 0;
        uint32_t table_count = 0;
        int sampler_table = -1;
        uint32_t sampler_offset = 0;
        bool per_frame = false;
        uint32_t dynamic_count = 0;
        std::array<D3D12_GPU_VIRTUAL_ADDRESS, max_dynamic_offsets> dynamic_base{};
        std::vector<state_validator::bound_resource> bound;
    };

    struct pipeline_layout_rec {
        winrt::com_ptr<ID3D12RootSignature> root_signature;
        uint64_t hash = 0;
        int push_param = -1;
        uint32_t push_bytes = 0;
        std::array<bind_layout_handle, max_bind_groups> sets{};
        std::array<int, max_bind_groups> table_param{};
        std::array<int, max_bind_groups> sampler_param{};
        std::array<std::array<int, max_dynamic_offsets>, max_bind_groups> dynamic_param{};
    };

    struct pipeline_rec {
        winrt::com_ptr<ID3D12PipelineState> pso;
        winrt::com_ptr<ID3D12RootSignature> root_signature;
        pipeline_layout_handle layout_handle;
        bool compute = false;
        D3D12_PRIMITIVE_TOPOLOGY topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        std::array<uint32_t, max_vertex_streams> strides{};
        uint32_t stream_count = 0;
        uint32_t color_count = 0;
        std::array<format, max_color_targets> color_formats{};
        format depth_format = format::unknown;
        uint32_t push_bytes = 0;
    };

    struct query_rec {
        winrt::com_ptr<ID3D12QueryHeap> heap;
        uint32_t count = 0;
    };

    struct fence_rec {
        winrt::com_ptr<ID3D12Fence> fence;
    };

    struct queue_info {
        winrt::com_ptr<ID3D12CommandQueue> queue;
        winrt::com_ptr<ID3D12Fence> fence;
        std::atomic<uint64_t> serial{0};
        std::mutex mutex;
    };

    class device;

    class cmd_list final : public command_list {
    public:
        cmd_list(d3d12::device& device, queue_type queue, winrt::com_ptr<ID3D12CommandAllocator> allocator,
                 winrt::com_ptr<ID3D12GraphicsCommandList> list)
            : device_(device), queue_(queue), allocator_(std::move(allocator)), list_(std::move(list)) {}

        queue_type queue() const override { return queue_; }
        void begin();
        void end();
        ID3D12GraphicsCommandList* native() const { return list_.get(); }
        ID3D12CommandAllocator* allocator() const { return allocator_.get(); }

        void barrier(std::span<const texture_barrier> textures, std::span<const buffer_barrier> buffers) override;
        void uav_barrier(buffer_handle buffer) override;
        void uav_barrier(texture_handle texture) override;
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
        void* native_handle() override { return list_.get(); }
        void invalidate_bound_state() override;

        int open_labels() const { return open_labels_; }
        const list_counters& counters() const { return counters_; }

    private:
        void flush_bindings();
        void fail(const std::string& message);
        void transition(ID3D12Resource* resource, UINT subresource, D3D12_RESOURCE_STATES before,
                        D3D12_RESOURCE_STATES after);

        device& device_;

        int open_labels_ = 0;
        list_counters counters_;
        std::vector<D3D12_RESOURCE_BARRIER> barrier_scratch_;
        queue_type queue_;
        winrt::com_ptr<ID3D12CommandAllocator> allocator_;
        winrt::com_ptr<ID3D12GraphicsCommandList> list_;

        pipeline_rec pipeline_;
        pipeline_layout_handle layout_handle_;
        bool have_pipeline_ = false;
        bool in_rendering_ = false;
        bool pass_broken_ = false;
        std::array<bind_group_handle, max_bind_groups> groups_{};
        std::array<std::array<uint32_t, max_dynamic_offsets>, max_bind_groups> dynamic_offsets_{};
        std::array<uint32_t, max_bind_groups> dynamic_counts_{};
        uint32_t dirty_sets_ = 0;
        std::array<uint32_t, max_push_constant_bytes / 4> push_{};
        uint32_t push_used_ = 0;
        bool push_dirty_ = false;
        std::array<D3D12_VERTEX_BUFFER_VIEW, max_vertex_streams> vbuffers_{};
        uint32_t vertex_mask_ = 0;
        uint32_t vertex_dirty_ = 0;
        bool have_index_ = false;
        D3D12_INDEX_BUFFER_VIEW index_view_{};
        std::array<format, max_color_targets> color_formats_{};
        uint32_t color_count_ = 0;
        format depth_format_ = format::unknown;
        struct resolve_pair {
            texture_handle src, dst;
        };
        std::vector<resolve_pair> pending_resolves_;
        bool heaps_set_ = false;
        bool recording_ = false;
        bool invalid_ = false;
        uint32_t target_width_ = 0, target_height_ = 0;
        uint32_t stencil_ref_ = 0;
        ID3D12RootSignature* bound_root_ = nullptr;
        bool bound_compute_ = false;
        D3D12_VIEWPORT viewport_{};
        D3D12_RECT scissor_{};
        bool have_viewport_ = false, have_scissor_ = false;
        std::vector<std::pair<winrt::com_ptr<ID3D12Resource>, winrt::com_ptr<D3D12MA::Allocation>>> temps_;
    };

    struct thread_pools {
        struct per_queue {
            std::vector<std::unique_ptr<cmd_list>> lists;
            size_t used = 0;
        };
        std::vector<std::array<per_queue, queue_count>> slots;
    };

    class device final : public conduit::device {
    public:
        device(std::shared_ptr<instance_state> inst, winrt::com_ptr<IDXGIAdapter1> adapter, const adapter_info& info,
               const device_options& options);
        ~device() override;
        bool init(std::string& error);

        api get_api() const override { return api::d3d12; }
        const device_caps& caps() const override { return caps_; }
        native_handles native() const override;
        native_texture native_of(texture_handle texture, state state) override;

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
        uint64_t gpu_address(buffer_handle buffer) const override;

        fence_handle create_fence(uint64_t initial_value) override;
        uint64_t fence_value(fence_handle fence) override;
        bool wait_fence(fence_handle fence, uint64_t value, uint64_t timeout_ns) override;
        void signal_fence(fence_handle fence, uint64_t value) override;

        command_list* acquire_command_list(queue_type queue) override;
        void submit(queue_type queue, const submit_desc& desc) override;
        std::unique_ptr<conduit::swapchain> create_swapchain(const swapchain_desc& desc) override;

        void begin_frame() override;
        void end_frame() override;
        uint64_t frame_number() const override { return frame_; }
        uint32_t frame_slot() const override { return slot_; }
        void wait_idle() override;

        memory_budget query_memory_budget() override;
        device_stats stats() const override;
        std::vector<uint8_t> serialize_pipeline_cache() override;
        std::string debug_report() override;
        float timestamp_period_ns() const override { return caps_.limits.timestamp_period_ns; }

        texture_handle wrap_swapchain_image(ID3D12Resource* resource, const texture_desc& desc);
        void destroy_wrapped(texture_handle h);
        D3D12_CPU_DESCRIPTOR_HANDLE render_view(texture_rec& tex, uint32_t mip, uint32_t layer, bool depth,
                                                bool read_only = false);
        ID3D12CommandSignature* signature(bool dispatch, uint32_t stride);
        void retire(std::function<void()> fn);
        void report_error(const std::string& message);
        queue_serials completed();
        void check_removed(HRESULT hr, const char* what);
        void signal_serial(queue_type q);

        instance_state& instance_info() const { return *inst_; }
        queue_info& direct_queue() { return queues_[0]; }
        uint32_t frames_in_flight() const { return options_.frames_in_flight; }

    private:
        winrt::com_ptr<ID3D12Device> device_;
        winrt::com_ptr<ID3D12Device4> device4_;
        winrt::com_ptr<IDXGIAdapter1> adapter_;
        winrt::com_ptr<D3D12MA::Allocator> allocator_;
        std::shared_ptr<instance_state> inst_;
        device_caps caps_;
        device_options options_;
        state_validator validator_;
        bool use_validator_ = false;
        bool uma_ = false;

        handle_pool<buffer_tag, buffer_rec> buffers_;
        handle_pool<texture_tag, texture_rec> textures_;
        handle_pool<sampler_tag, sampler_rec> samplers_;
        handle_pool<shader_tag, shader_rec> shaders_;
        handle_pool<bind_layout_tag, bind_layout_rec> bind_layouts_;
        handle_pool<bind_group_tag, bind_group_rec> bind_groups_;
        handle_pool<pipeline_layout_tag, pipeline_layout_rec> pipeline_layouts_;
        handle_pool<pipeline_tag, pipeline_rec> pipelines_;
        handle_pool<query_set_tag, query_rec> queries_;
        handle_pool<fence_tag, fence_rec> fences_;

        std::array<queue_info, queue_count> queues_;
        device_counters counters_;
        std::atomic<int64_t> buffer_bytes_{0}, texture_bytes_{0};
        std::atomic<int32_t> live_buffers_{0}, live_textures_{0}, live_pipelines_{0}, live_groups_{0};

        cpu_descriptor_pool rtv_pool_, dsv_pool_;
        range_heap resource_heap_, sampler_heap_;
        std::mutex view_mutex_;

        friend class cmd_list;

        thread_pools& pools_for_this_thread();
        void reset_slot(uint32_t s);
        void fill_caps();
        void write_sampler_desc(D3D12_CPU_DESCRIPTOR_HANDLE dst, const sampler_desc& d);
        int acquire_sampler_table(const std::vector<uint32_t>& key, const std::vector<sampler_desc>& descs);
        void release_sampler_table(int index);

        std::mutex sampler_mutex_;
        std::vector<std::pair<sampler_desc, sampler_handle>> sampler_cache_;
        std::vector<sampler_table> sampler_tables_;

        std::mutex signature_mutex_;
        std::unordered_map<uint64_t, winrt::com_ptr<ID3D12CommandSignature>> signatures_;

        winrt::com_ptr<ID3D12PipelineLibrary> pipeline_library_;
        std::mutex library_mutex_;

        std::mutex pools_mutex_;
        std::unordered_map<std::thread::id, std::unique_ptr<thread_pools>> thread_pools_;

        struct frame_groups {
            std::vector<bind_group_handle> groups;
        };
        std::vector<frame_groups> frame_groups_;
        std::mutex frame_group_mutex_;

        retire_queue retire_;
        uint64_t frame_ = 0;
        uint32_t slot_ = 0;
        std::vector<queue_serials> slot_serials_;
        winrt::com_ptr<ID3D12InfoQueue> info_queue_;
        winrt::handle wait_event_;
        std::mutex wait_mutex_;
    };

    std::unique_ptr<conduit::swapchain> make_swapchain(device& device, const swapchain_desc& desc);
}
