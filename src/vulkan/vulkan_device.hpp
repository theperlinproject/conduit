#pragma once
#include <conduit/device.hpp>
#include "common/counters.hpp"
#include "common/handle_pool.hpp"
#include "common/retire_queue.hpp"
#include "common/state_validator.hpp"
#include "common/util.hpp"

#include <volk.h>
#include <vk_mem_alloc.h>

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace conduit::vk {
    struct instance_state {
        VkInstance instance = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        uint32_t api_version = VK_API_VERSION_1_1;
        bool validation = false;
        bool debug_utils = false;
        bool surface = false;
        bool color_space_ext = false;
        bool surface_caps2 = false;
        std::vector<std::string> enabled_extensions;
        std::mutex message_mutex;
        std::vector<std::string> messages;
        size_t error_count = 0;
        ~instance_state();
    };

    struct physical_info {
        VkPhysicalDevice device = VK_NULL_HANDLE;
        VkPhysicalDeviceProperties props{};
        std::vector<std::string> extensions;
        bool has(const char* name) const;
    };

    struct buffer_rec {
        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        buffer_desc desc;
        std::string name;
        uint8_t* mapped = nullptr;
        VkDeviceAddress address = 0;
        bool coherent = true;
    };

    struct texture_rec {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation alloc = nullptr;
        texture_desc desc;
        std::string name;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        VkImageUsageFlags usage = 0;
        VkImageCreateFlags create_flags = 0;
        uint32_t layers = 1;
        bool external = false;
        std::unordered_map<uint64_t, VkImageView> views;
    };

    struct sampler_rec {
        VkSampler sampler = VK_NULL_HANDLE;
        sampler_desc desc;
    };

    struct shader_rec {
        VkShaderModule module = VK_NULL_HANDLE;
        shader_stage stage = shader_stage::vertex;
        std::string entry;
    };

    struct bind_layout_rec {
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        std::vector<bind_layout_entry> entries;
        uint32_t dynamic_count = 0;
    };

    struct bind_group_rec {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        bind_layout_handle layout;
        bool per_frame = false;
        uint32_t dynamic_count = 0;
        std::vector<state_validator::bound_resource> bound;
    };

    struct pipeline_layout_rec {
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkShaderStageFlags push_stages = 0;
        uint32_t push_bytes = 0;
        std::array<bind_layout_handle, max_bind_groups> sets{};
    };

    struct pipeline_rec {
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
        pipeline_layout_handle layout_handle;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkShaderStageFlags push_stages = 0;
        uint32_t push_bytes = 0;
        uint32_t color_count = 0;
        std::array<format, max_color_targets> color_formats{};
        format depth_format = format::unknown;
        uint32_t stream_count = 0;
        bool compute = false;
    };

    struct query_rec {
        VkQueryPool pool = VK_NULL_HANDLE;
        uint32_t count = 0;
    };

    struct fence_rec {
        VkSemaphore semaphore = VK_NULL_HANDLE;
    };

    struct queue_info {
        VkQueue queue = VK_NULL_HANDLE;
        uint32_t family = 0;
        std::mutex* mutex = nullptr;
        VkSemaphore timeline = VK_NULL_HANDLE;
        std::atomic<uint64_t> serial{0};
        bool shared = false;
    };

    class device;

    class cmd_list final : public command_list {
    public:
        cmd_list(vk::device& device, queue_type queue, VkCommandBuffer cmd)
            : device_(device), queue_(queue), cmd_(cmd) {}

        queue_type queue() const override { return queue_; }
        void begin();
        void end();
        VkCommandBuffer cmd() const { return cmd_; }

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
        void* native_handle() override { return cmd_; }
        void invalidate_bound_state() override;

        int open_labels() const { return open_labels_; }
        const list_counters& counters() const { return counters_; }

    private:
        void flush_bindings();
        void fail(const std::string& message);

        device& device_;

        int open_labels_ = 0;
        list_counters counters_;
        std::vector<VkImageMemoryBarrier2> barrier_images_;
        std::vector<VkBufferMemoryBarrier2> barrier_buffers_;
        queue_type queue_;
        VkCommandBuffer cmd_;

        pipeline_rec pipeline_;
        bool have_pipeline_ = false;
        bool in_rendering_ = false;
        std::array<bind_group_handle, max_bind_groups> groups_{};
        std::array<std::array<uint32_t, max_dynamic_offsets>, max_bind_groups> dynamic_offsets_{};
        std::array<uint32_t, max_bind_groups> dynamic_counts_{};
        uint32_t dirty_sets_ = 0;
        std::array<uint8_t, max_push_constant_bytes> push_{};
        bool push_dirty_ = false;
        uint32_t push_used_ = 0;
        std::array<format, max_color_targets> color_formats_{};
        uint32_t color_count_ = 0;
        format depth_format_ = format::unknown;
        pipeline_handle pipeline_handle_;
        bool have_index_ = false;
        uint32_t vertex_mask_ = 0;
        uint32_t target_width_ = 0, target_height_ = 0;
        bool invalid_ = false;
        VkViewport viewport_{};
        VkRect2D scissor_{};
        bool have_viewport_ = false, have_scissor_ = false;
        std::array<VkBuffer, max_vertex_streams> vbuffers_{};
        std::array<VkDeviceSize, max_vertex_streams> voffsets_{};
        VkBuffer ibuffer_ = VK_NULL_HANDLE;
        VkDeviceSize ioffset_ = 0;
        VkIndexType itype_ = VK_INDEX_TYPE_UINT32;
        uint32_t stencil_ref_ = 0;
    };

    struct thread_pools {
        struct per_queue {
            VkCommandPool pool = VK_NULL_HANDLE;
            std::vector<std::unique_ptr<cmd_list>> lists;
            size_t used = 0;
        };
        std::vector<std::array<per_queue, queue_count>> slots;
    };

    class swapchain;

    class device final : public conduit::device {
    public:
        device(std::shared_ptr<instance_state> inst, const physical_info& phys, const adapter_info& adapter,
               const device_options& options);
        ~device() override;
        bool init(std::string& error);

        api get_api() const override { return api::vulkan; }
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
        void flush(buffer_handle buffer, uint64_t offset, uint64_t size) override;
        void invalidate(buffer_handle buffer, uint64_t offset, uint64_t size) override;
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

        VkFormat format_to_vk(format f) const;
        VkImageView view(texture_rec& tex, const subresource_range& range, texture_dim dim, format format,
                         bool stencil);
        VkImageView attachment_view(texture_handle h, uint32_t mip, uint32_t layer);
        texture_handle wrap_swapchain_image(VkImage image, const texture_desc& desc, VkFormat format);
        void destroy_wrapped(texture_handle h);
        void name_object(VkObjectType type, uint64_t handle, const char* name);
        void retire(std::function<void()> fn);
        void report_error(const std::string& message);
        queue_serials completed();
        void set_pending_acquire(VkSemaphore s) { pending_acquire_ = s; }
        void submit_present(VkSemaphore render_done);
        VkQueue present_queue() const { return queues_[0].queue; }
        std::mutex* present_mutex() const { return queues_[0].mutex; }

    private:
        VolkDeviceTable fn_{};
        VkDevice device_ = VK_NULL_HANDLE;
        VkPhysicalDevice physical_ = VK_NULL_HANDLE;
        VmaAllocator allocator_ = nullptr;
        std::shared_ptr<instance_state> inst_;
        physical_info phys_;
        device_caps caps_;
        device_options options_;
        state_validator validator_;
        bool use_validator_ = false;

        bool dynamic_rendering_ext_ = false;
        bool host_query_reset_ = false;
        bool depth_clip_enable_ = false;
        bool swapchain_ext_ = false;
        bool d24_supported_ = true;
        VkPhysicalDeviceMemoryProperties mem_props_{};

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
        mutable device_stats stats_;
        device_counters counters_;
        std::atomic<int64_t> buffer_bytes_{0}, texture_bytes_{0};
        std::atomic<int32_t> live_buffers_{0}, live_textures_{0}, live_pipelines_{0}, live_groups_{0};

        friend class cmd_list;
        friend class swapchain;

        bool pick_queues(std::string& error);
        thread_pools& pools_for_this_thread();
        void reset_slot(uint32_t s);
        VkDescriptorPool new_descriptor_pool(bool freeable);
        VkDescriptorSet allocate_set(VkDescriptorSetLayout layout, bool per_frame, VkDescriptorPool& pool_out);
        void fill_caps(const adapter_info& adapter);

        std::mutex view_mutex_;
        std::mutex sampler_mutex_;
        std::vector<std::pair<sampler_desc, sampler_handle>> sampler_cache_;
        std::array<std::mutex, 3> queue_mutexes_;
        VkPipelineCache pipeline_cache_ = VK_NULL_HANDLE;
        std::vector<VkDeviceQueueCreateInfo> queue_create_infos_;
        std::array<float, 8> queue_priorities_{};
        std::array<uint32_t, queue_count> queue_index_{};
        VkDescriptorSetLayout empty_set_layout_ = VK_NULL_HANDLE;
        VkImageView make_view(texture_rec& tex, const subresource_range& range, VkImageViewType type, VkFormat format,
                              VkImageAspectFlags aspect);

        std::mutex descriptor_mutex_;
        std::vector<VkDescriptorPool> persistent_pools_;
        struct frame_descriptors {
            std::vector<VkDescriptorPool> pools;
            size_t used = 0;
            std::vector<bind_group_handle> groups;
        };
        std::vector<frame_descriptors> frame_descriptors_;

        std::mutex pools_mutex_;
        std::unordered_map<std::thread::id, std::unique_ptr<thread_pools>> thread_pools_;

        retire_queue retire_;
        uint64_t frame_ = 0;
        uint32_t slot_ = 0;
        std::vector<queue_serials> slot_serials_;
        std::atomic<VkSemaphore> pending_acquire_{VK_NULL_HANDLE};
        std::atomic<bool> lost_{false};
    };

    std::unique_ptr<instance> create_instance(const instance_options& options);
    std::unique_ptr<conduit::swapchain> make_swapchain(device& device, const swapchain_desc& desc);
}
