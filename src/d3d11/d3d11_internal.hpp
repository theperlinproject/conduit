#pragma once
#include <conduit/device.hpp>
#include "common/handle_pool.hpp"
#include "common/shader_reflect.hpp"
#include "common/util.hpp"
#include "d3d11/d3d11_convert.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d11_1.h>
#include <d3dcommon.h>
#include <dxgi1_6.h>

#if __has_include(<winrt/base.h>)
#define WINRT_LEAN_AND_MEAN
#include <winrt/base.h>
#else
#include "d3d12/winrt_shim.hpp"
#endif

#include <array>
#include <deque>
#include <map>
#include <memory>
#include <tuple>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace conduit::d3d11 {
    using compile_fn = HRESULT(WINAPI*)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR, LPCSTR,
                                        UINT, UINT, ID3DBlob**, ID3DBlob**);

    struct library {
        HMODULE d3d11 = nullptr;
        HMODULE dxgi = nullptr;
        HMODULE compiler = nullptr;
        PFN_D3D11_CREATE_DEVICE create_device = nullptr;
        HRESULT(WINAPI* create_factory1)(REFIID, void**) = nullptr;
        compile_fn compile = nullptr;
    };

    struct instance_state {
        library lib;
        winrt::com_ptr<IDXGIFactory2> factory;
        bool validation = false;
        bool tearing = false;
    };

    struct pending_readback {
        winrt::com_ptr<ID3D11Texture2D> staging;
        uint64_t buffer_offset = 0;
        uint32_t buffer_row_pitch = 0;
        uint32_t row_bytes = 0;
        uint32_t rows = 0;
    };

    struct buffer {
        winrt::com_ptr<ID3D11Buffer> main;
        winrt::com_ptr<ID3D11Buffer> constant;
        buffer_desc desc;
        std::string name;
        std::vector<uint8_t> shadow;
        uint64_t gpu_size = 0;
        bool structured = false;
        std::map<std::tuple<uint64_t, uint64_t, uint8_t>, winrt::com_ptr<ID3D11View>> views;
        std::vector<pending_readback> pending;
    };

    struct view_key {
        uint8_t kind = 0;
        uint8_t dim = 0;
        uint16_t format = 0;
        uint32_t base_mip = 0, mip_count = 0, base_layer = 0, layer_count = 0;
        auto operator<=>(const view_key&) const = default;
    };

    struct texture {
        winrt::com_ptr<ID3D11Resource> resource;
        texture_desc desc;
        std::string name;
        format_map formats;
        bool external = false;
        std::map<view_key, winrt::com_ptr<ID3D11View>> views;
    };

    struct sampler {
        winrt::com_ptr<ID3D11SamplerState> state;
        sampler_desc desc;
    };

    struct shader {
        shader_stage stage = shader_stage::vertex;
        std::string source;
        std::string entry;
        shader_interface iface;
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
        char register_class = 't';
        uint32_t slot = 0;
        uint32_t dynamic_index = ~0u;
    };

    enum stage_index : uint32_t { stage_vertex = 0, stage_pixel = 1, stage_compute = 2 };

    struct pipeline {
        bool compute = false;
        winrt::com_ptr<ID3D11VertexShader> vs;
        winrt::com_ptr<ID3D11PixelShader> ps;
        winrt::com_ptr<ID3D11ComputeShader> cs;
        winrt::com_ptr<ID3D11InputLayout> input_layout;
        winrt::com_ptr<ID3D11RasterizerState> raster;
        winrt::com_ptr<ID3D11BlendState> blend;
        winrt::com_ptr<ID3D11DepthStencilState> depth_stencil;
        D3D11_PRIMITIVE_TOPOLOGY topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        std::array<UINT, max_vertex_streams> strides{};
        uint32_t stream_count = 0;
        std::array<std::vector<stage_slot>, 3> stages;
    };

    struct query {
        uint32_t count = 0;
    };

    struct fence {
        uint64_t value = 0;
        std::deque<std::pair<uint64_t, winrt::com_ptr<ID3D11Query>>> pending;
    };

    class device;

    class cmd_list final : public conduit::command_list {
    public:
        cmd_list(device& device, queue_type queue);
        ~cmd_list() override;

        bool init();
        queue_type queue() const override { return queue_; }
        void begin();
        ID3D11CommandList* finish();
        void release_finished() { finished_ = nullptr; }

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
        void* native_handle() override { return ctx_.get(); }
        void invalidate_bound_state() override {
            dirty_sets_ = 0xF;
            pipeline_dirty_ = true;
            vertex_dirty_ = 0xFF;
            index_dirty_ = true;
        }

    private:
        bool prepare_draw();
        bool bind_sets(const pipeline& p);
        void clear_bindings();
        void unbind_targets();
        void copy_to_constant(buffer& b, uint64_t offset, uint64_t size);

        device& device_;
        queue_type queue_;
        winrt::com_ptr<ID3D11DeviceContext> ctx_;
        winrt::com_ptr<ID3D11DeviceContext1> ctx1_;
        winrt::com_ptr<ID3D11CommandList> finished_;
        bool recorded_ = false;

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
        bool index_dirty_ = false;
        viewport viewport_;
        rect scissor_;
        bool viewport_set_ = false, scissor_set_ = false;
        uint32_t stencil_ref_ = 0;
        uint32_t pass_width_ = 0, pass_height_ = 0;
        bool in_pass_ = false;
        std::array<uint32_t, 3> bound_srvs_{};
        uint32_t bound_uavs_ = 0;
        struct resolve {
            texture_handle dst, src;
        };
        std::vector<resolve> resolves_;
        winrt::com_ptr<ID3D11Buffer> mirrors_[3][14];
        struct upload_mirror {
            winrt::com_ptr<ID3D11Buffer> buffer;
            UINT bytes = 0;
            bool valid = false;
            std::vector<uint8_t> last;
        };
        upload_mirror upload_mirrors_[3][14];
    };

    class device final : public conduit::device {
    public:
        device(std::shared_ptr<instance_state> instance, winrt::com_ptr<IDXGIAdapter1> adapter,
               const adapter_info& info, const device_options& options);
        ~device() override;

        bool init(std::string& error);

        api get_api() const override { return api::d3d11; }
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
        device_stats stats() const override { return stats_; }
        float timestamp_period_ns() const override { return 1.0f; }

        ID3D11Device* dev() const { return device_.get(); }
        ID3D11Device1* dev1() const { return device1_.get(); }
        ID3D11DeviceContext* imm() const { return immediate_.get(); }
        instance_state& inst() const { return *inst_; }
        const device_options& options() const { return options_; }
        device_stats& mutable_stats() { return stats_; }
        bool constant_offsetting() const { return constant_offsetting_; }
        void report_error(const std::string& message);
        std::unique_lock<std::mutex> lock_context() { return std::unique_lock<std::mutex>(ctx_mutex_); }

        ID3D11ShaderResourceView* srv_of(texture& texture, const subresource_range& range, texture_dim dim,
                                         format format, bool stencil);
        ID3D11UnorderedAccessView* uav_of(texture& texture, const subresource_range& range, texture_dim dim);
        ID3D11RenderTargetView* rtv_of(texture& texture, uint32_t mip, uint32_t layer);
        ID3D11DepthStencilView* dsv_of(texture& texture, uint32_t mip, uint32_t layer, bool read_only);
        ID3D11ShaderResourceView* buffer_srv(buffer& buffer, uint64_t offset, uint64_t size);
        ID3D11UnorderedAccessView* buffer_uav(buffer& buffer, uint64_t offset, uint64_t size);
        texture_handle wrap_backbuffer(ID3D11Texture2D* texture, const texture_desc& desc);
        void set_backbuffer(texture_handle handle, ID3D11Texture2D* texture);
        void release_backbuffer(texture_handle handle);

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

        friend class cmd_list;
        void fill_caps();
        bool compile_stage(const shader& shader, const pipeline_layout_desc& layout, uint32_t stage_index,
                           std::vector<stage_slot>& slots, winrt::com_ptr<ID3DBlob>& blob, std::string& error);
        void poll_fence(fence& fence);
        void mark_submitted(fence_handle fence, uint64_t value);

        std::shared_ptr<instance_state> inst_;
        winrt::com_ptr<IDXGIAdapter1> adapter_;
        winrt::com_ptr<ID3D11Device> device_;
        winrt::com_ptr<ID3D11Device1> device1_;
        winrt::com_ptr<ID3D11DeviceContext> immediate_;
        device_options options_;
        device_caps caps_;
        device_stats stats_;
        bool constant_offsetting_ = false;
        uint64_t frame_ = 0;
        fence_handle frame_fence_;

        std::mutex ctx_mutex_;
        std::mutex view_mutex_;
        std::mutex fence_mutex_;
        std::mutex list_mutex_;
        std::mutex shader_cache_mutex_;
        std::vector<std::unique_ptr<cmd_list>> lists_;
        std::vector<cmd_list*> free_lists_;
        std::vector<winrt::com_ptr<ID3D11Query>> free_queries_;
        std::unordered_map<std::string, winrt::com_ptr<ID3DBlob>> blob_cache_;
        std::mutex error_mutex_;
        std::vector<uint8_t> zeros_;
    };

    std::unique_ptr<conduit::swapchain> make_swapchain(device& device, const swapchain_desc& desc);
}
