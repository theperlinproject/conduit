#pragma once
#include <conduit/device.hpp>
#include "common/counters.hpp"
#include "common/handle_pool.hpp"
#include "common/util.hpp"
#include "gles/gles_gl.hpp"
#include "gles/gles_platform.hpp"

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

// The OpenGL ES backend.
//
// A GL context belongs to one thread at a time, and conduit is used from several. So the context lives on a worker
// thread of its own: creating, destroying and submitting hand work to it, command lists are recorded into a compact
// stream anywhere and replayed there, and a frame's fence is signalled when its stream has been replayed. Where there
// are no threads (Emscripten) the same calls run inline.
namespace conduit::gles {
    class gl_thread {
    public:
        gl_thread() = default;
        gl_thread(const gl_thread&) = delete;
        gl_thread& operator=(const gl_thread&) = delete;
        ~gl_thread() { stop(); }

        void start() {
#ifndef __EMSCRIPTEN__
            thread_ = std::thread([this] { run(); });
            id_ = thread_.get_id();
#endif
        }

        void stop() {
#ifndef __EMSCRIPTEN__
            if (!thread_.joinable())
                return;
            {
                std::lock_guard lock(mutex_);
                quit_ = true;
            }
            cv_.notify_all();
            thread_.join();
#endif
        }

        bool on_thread() const {
#ifdef __EMSCRIPTEN__
            return true;
#else
            return std::this_thread::get_id() == id_;
#endif
        }

        // runs in order after everything posted before it
        void post(std::function<void()> fn) {
#ifdef __EMSCRIPTEN__
            fn();
#else
            if (!thread_.joinable() || on_thread()) {
                fn();
                return;
            }
            {
                std::lock_guard lock(mutex_);
                queue_.push_back(std::move(fn));
            }
            cv_.notify_one();
#endif
        }

        // runs in order and waits for the result
        template <class F> auto call(F&& fn) -> std::invoke_result_t<F> {
#ifdef __EMSCRIPTEN__
            return fn();
#else
            using result = std::invoke_result_t<F>;
            if (!thread_.joinable() || on_thread())
                return fn();
            auto task = std::make_shared<std::packaged_task<result()>>(std::forward<F>(fn));
            std::future<result> future = task->get_future();
            post([task] { (*task)(); });
            return future.get();
#endif
        }

    private:
#ifndef __EMSCRIPTEN__
        void run() {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock lock(mutex_);
                    cv_.wait(lock, [this] { return quit_ || !queue_.empty(); });
                    if (queue_.empty())
                        return;
                    job = std::move(queue_.front());
                    queue_.pop_front();
                }
                job();
            }
        }

        std::thread thread_;
        std::thread::id id_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<std::function<void()>> queue_;
        bool quit_ = false;
#endif
    };

    struct format_gl {
        GLenum internal = 0;
        GLenum format = 0;
        GLenum type = 0;
        uint8_t bytes = 0;
        bool compressed = false;
        bool integer = false;
        bool swap_rb = false;
        // needs an extension: filled in once the context says what it has
        bool needs_s3tc = false, needs_bptc = false, needs_norm16 = false;
    };
    const format_gl& format_of(format f);

    struct buffer {
        GLuint gl = 0;
        buffer_desc desc;
        std::string name;
        uint64_t size = 0;
        // upload and readback buffers are plain memory on the cpu: the application writes into it through map() and
        // flush() copies it across, and readbacks land in it
        std::vector<uint8_t> shadow;
    };

    struct texture {
        GLuint gl = 0;
        GLenum target = GL_TEXTURE_2D;
        bool renderbuffer = false;
        texture_desc desc;
        std::string name;
        format_gl fmt;
        int levels = 1;
        // sampling state that lives on the texture and is only touched on the GL thread
        int base_level = 0;
        int max_level = 1000;
    };

    struct sampler {
        GLuint gl = 0;
        // the same sampler with every filter set to nearest, for textures GL will not filter (depth, 32-bit float)
        GLuint gl_nearest = 0;
        sampler_desc desc;
    };

    struct shader {
        shader_stage stage = shader_stage::vertex;
        std::vector<uint32_t> words;
        std::string entry;
        std::string name;
    };

    struct bind_layout {
        std::vector<bind_layout_entry> entries;
    };

    struct bind_group {
        bind_layout_handle layout;
        std::vector<bind_group_entry> entries;
        // entry index by binding number, -1 when the group has none
        std::vector<int32_t> by_binding;

        const bind_group_entry* find(uint32_t binding) const {
            return binding < by_binding.size() && by_binding[binding] >= 0 ? &entries[by_binding[binding]] : nullptr;
        }
    };

    // where one descriptor of a pipeline layout ends up in GL
    struct layout_slot {
        bool present = false;
        binding_type type = binding_type::uniform_buffer;
        // uniform and storage block binding point, or image unit
        uint32_t index = 0;
        uint32_t dynamic_index = ~0u;
        texture_dim dim = texture_dim::tex2d;
    };

    struct pipeline_layout {
        pipeline_layout_desc desc;
        std::array<std::vector<layout_slot>, max_bind_groups> sets;
        uint32_t uniform_blocks = 0;
        uint32_t storage_blocks = 0;
        uint32_t images = 0;
        uint32_t push_slot = ~0u;
        uint32_t push_bytes = 0;
        const layout_slot* find(uint32_t set, uint32_t binding) const {
            if (set >= sets.size() || binding >= sets[set].size() || !sets[set][binding].present)
                return nullptr;
            return &sets[set][binding];
        }
    };

    struct blend_gl {
        bool enable = false;
        GLenum src_rgb = GL_ONE, dst_rgb = GL_ZERO, src_alpha = GL_ONE, dst_alpha = GL_ZERO;
        GLenum op_rgb = GL_FUNC_ADD, op_alpha = GL_FUNC_ADD;
        uint8_t mask = 15;
        bool operator==(const blend_gl&) const = default;
    };

    struct stencil_gl {
        GLenum fail = GL_KEEP, depth_fail = GL_KEEP, pass = GL_KEEP, func = GL_ALWAYS;
        bool operator==(const stencil_gl&) const = default;
    };

    // everything a graphics pipeline sets, in GL terms
    struct pipeline_state {
        GLenum cull_face = 0;
        GLenum front_face = GL_CW;
        bool offset = false;
        float offset_slope = 0.0f, offset_units = 0.0f;
        bool depth_test = false, depth_write = false;
        GLenum depth_func = GL_LEQUAL;
        bool stencil_test = false;
        uint8_t stencil_read = 0xFF, stencil_write = 0xFF;
        stencil_gl front, back;
        bool alpha_to_coverage = false;
        uint32_t blend_count = 0;
        std::array<blend_gl, max_color_targets> blend{};
    };

    struct vertex_attribute_gl {
        uint32_t location = 0;
        uint32_t stream = 0;
        GLint size = 4;
        GLenum type = GL_FLOAT;
        bool normalized = false;
        bool integer = false;
        uint32_t offset = 0;
    };

    // one sampler2D of the translated program: a texture and the sampler it is used with
    struct combined_sampler {
        uint32_t unit = 0;
        uint32_t set = 0, binding = 0;
        uint32_t sampler_set = 0, sampler_binding = 0;
        bool dummy_sampler = false;
        GLenum target = GL_TEXTURE_2D;
    };

    struct pipeline {
        bool compute = false;
        GLuint program = 0;
        pipeline_layout_handle layout;
        GLenum topology = GL_TRIANGLES;
        pipeline_state state;
        std::array<vertex_attribute_gl, max_vertex_attributes> attributes{};
        uint32_t attribute_count = 0;
        std::array<vertex_stream_desc, max_vertex_streams> streams{};
        uint32_t stream_count = 0;
        std::vector<combined_sampler> samplers;
        uint32_t color_count = 0;
        bool front_first = true;
    };

    struct query {
        uint32_t count = 0;
    };

    struct fence_state {
        uint64_t value = 0;
    };

    // what the context turned out to support
    struct gl_features {
        uint32_t major = 3, minor = 0;
        uint32_t glsl_version = 300;
        bool compute = false;           // ES 3.1
        bool indirect = false;          // ES 3.1
        bool storage_images = false;    // ES 3.1
        bool indexed_blend = false;     // ES 3.2 or EXT_draw_buffers_indexed
        bool copy_image = false;        // ES 3.2 or EXT_copy_image
        bool debug_groups = false;
        bool color_float = false;       // 32-bit float color attachments
        bool half_float_color = false;
        bool float_blend = false;
        bool float_linear = false;
        bool anisotropy = false;
        bool s3tc = false, bptc = false, norm16 = false;
        bool border_clamp = false;
        bool map_range = false;
        bool vsync_off = false;
    };

    class device;

    class cmd_list final : public conduit::command_list {
    public:
        cmd_list(device& device, queue_type queue) : device_(device), queue_(queue) {}

        queue_type queue() const override { return queue_; }
        void begin();
        // hands the recorded stream over and leaves the list empty
        std::vector<uint8_t> take() {
            std::vector<uint8_t> out;
            out.swap(stream_);
            return out;
        }
        const list_counters& counters() const { return counters_; }

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
        void* native_handle() override { return nullptr; }
        void invalidate_bound_state() override;

    private:
        struct no_payload {};
        struct stream_header {
            uint16_t op;
            uint16_t pad;
            uint32_t extra;
        };

        // header, the payload, then `extra_bytes` of variable length data
        template <class T>
        void emit(uint16_t op, const T& payload, const void* extra = nullptr, size_t extra_bytes = 0) {
            const stream_header header{op, 0, static_cast<uint32_t>(extra_bytes)};
            const size_t at = stream_.size();
            stream_.resize(at + sizeof(header) + sizeof(T) + extra_bytes);
            std::memcpy(stream_.data() + at, &header, sizeof(header));
            std::memcpy(stream_.data() + at + sizeof(header), &payload, sizeof(T));
            if (extra_bytes)
                std::memcpy(stream_.data() + at + sizeof(header) + sizeof(T), extra, extra_bytes);
        }
        void emit_op(uint16_t op) { emit(op, no_payload{}); }

        device& device_;
        queue_type queue_;
        std::vector<uint8_t> stream_;
        list_counters counters_;
    };

    // the framebuffer a render pass draws into
    struct framebuffer {
        GLuint id = 0;
        uint32_t color_count = 0;
        GLenum depth_attachment = 0;
        std::array<uint64_t, max_color_targets + 1> key{};
    };

    struct attrib_state {
        bool valid = false;
        GLuint buffer = 0;
        uint64_t offset = 0;
        GLsizei stride = 0;
        GLint size = 0;
        GLenum type = 0;
        bool normalized = false;
        bool integer = false;
        GLuint divisor = 0;
    };

    // what the GL thread believes the context is set to, so a pass that repeats itself costs no calls. a value of
    // ~0u means unknown
    struct gl_cache {
        GLuint program = ~0u;
        GLuint draw_framebuffer = ~0u;
        GLuint read_framebuffer = ~0u;
        bool scissor_test = false;
        pipeline_state state;
        bool state_valid = false;
        bool read_only_depth = false;
        GLenum active_unit = ~0u;
        std::array<GLuint, 48> unit_texture{};
        std::array<GLenum, 48> unit_target{};
        std::array<GLuint, 48> unit_sampler{};
        std::array<attrib_state, max_vertex_attributes> attribs{};
        uint32_t enabled_attribs = 0;
        GLuint array_buffer = ~0u;
        GLuint element_buffer = ~0u;
        GLfloat depth_near = -1.0f, depth_far = -1.0f;
        int unpack_row = -1, pack_row = -1;
        gl_cache() { reset(); }
        void reset() {
            program = draw_framebuffer = read_framebuffer = ~0u;
            scissor_test = false;
            state = pipeline_state{};
            state_valid = false;
            active_unit = ~0u;
            unit_texture.fill(~0u);
            unit_target.fill(~0u);
            unit_sampler.fill(~0u);
            attribs.fill(attrib_state{});
            enabled_attribs = 0;
            array_buffer = element_buffer = ~0u;
            depth_near = depth_far = -1.0f;
            unpack_row = pack_row = -1;
        }
    };

    class device final : public conduit::device {
    public:
        device(const adapter_info& info, const device_options& options);
        ~device() override;

        bool init(std::string& error);

        api get_api() const override { return api::gles; }
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
        std::string debug_report() override;
        float timestamp_period_ns() const override { return 1.0f; }

        // ---- shared with the command list replay and the swapchain ----
        gl_thread& thread() { return thread_; }
        gl_context& context() { return *context_; }
        const gl_api& gl() const { return gl_; }
        const gl_features& features() const { return features_; }
        void report_error(const std::string& message);
        void report_once(const std::string& message);

        buffer* buffer_of(buffer_handle h) const { return buffers_.get(h); }
        texture* texture_of(texture_handle h) const { return textures_.get(h); }
        sampler* sampler_of(sampler_handle h) const { return samplers_.get(h); }
        bind_group* group_of(bind_group_handle h) const { return bind_groups_.get(h); }
        bind_layout* layout_of(bind_layout_handle h) const { return bind_layouts_.get(h); }
        pipeline_layout* pipeline_layout_of(pipeline_layout_handle h) const { return pipeline_layouts_.get(h); }
        pipeline* pipeline_of(pipeline_handle h) const { return pipelines_.get(h); }

        gl_cache& cache() { return cache_; }
        GLuint vao() const { return vao_; }
        GLuint default_sampler() const { return default_sampler_; }
        // the framebuffer for a pass, built the first time it is asked for
        const framebuffer* framebuffer_for(const std::array<uint64_t, max_color_targets + 1>& key,
                                           const rendering_desc& desc, bool resolve_targets);
        GLuint scratch_framebuffer(bool read) const { return read ? scratch_read_ : scratch_draw_; }
        void bind_framebuffer(GLenum target, GLuint fbo);
        // one slot of the ring that push constants are written into
        GLuint push_ring() const { return push_ring_; }
        uint64_t push_alloc(uint32_t bytes);
        void present_texture(texture_handle texture, gl_surface& surface, present_mode mode);
        void execute(std::vector<std::vector<uint8_t>> streams, std::vector<fence_op> waits,
                     std::vector<fence_op> signals);
        void replay(const std::vector<uint8_t>& stream);
        void apply_cache_reset() { cache_.reset(); }
        device_counters& counters() { return counters_; }
        void signal_now(fence_handle fence, uint64_t value);
        bool wait_now(fence_handle fence, uint64_t value, uint64_t timeout_ns);

        // bind a texture to a unit, skipping the call when the unit already holds it
        void use_texture(uint32_t unit, GLenum target, GLuint texture);
        void use_program(GLuint program);
        void use_scissor_test(bool enable);
        // forget what the caches know about a GL buffer that is about to go away
        void forget_buffer(GLuint gl);
        // the state that is unknown after something else used the context
        void resync();
        // a unit nothing else samples from, for uploads and storage creation
        uint32_t scratch_unit() const { return max_texture_units_ - 1; }
        void drain_errors();

    private:
        friend class cmd_list;
        bool query_features(std::string& error);
        void fill_caps();
        void release_all();
        bool translate_and_link(pipeline& out, const graphics_pipeline_desc* graphics,
                                const compute_pipeline_desc* compute, std::string& error);
        pipeline_handle build_pipeline(const graphics_pipeline_desc* graphics, const compute_pipeline_desc* compute);
        void delete_framebuffers_using(GLuint gl, bool renderbuffer);
        bool format_supported(format f) const;

        gl_thread thread_;
        std::unique_ptr<gl_context> context_;
        gl_api gl_;
        gl_features features_;
        std::string extensions_;
        device_options options_;
        device_caps caps_;

        handle_pool<buffer_tag, buffer> buffers_;
        handle_pool<texture_tag, texture> textures_;
        handle_pool<sampler_tag, sampler> samplers_;
        handle_pool<shader_tag, shader> shaders_;
        handle_pool<bind_layout_tag, bind_layout> bind_layouts_;
        handle_pool<bind_group_tag, bind_group> bind_groups_;
        handle_pool<pipeline_layout_tag, pipeline_layout> pipeline_layouts_;
        handle_pool<pipeline_tag, pipeline> pipelines_;
        handle_pool<query_set_tag, query> queries_;
        handle_pool<fence_tag, fence_state> fences_;

        // GL thread only
        gl_cache cache_;
        GLuint vao_ = 0;
        GLuint default_sampler_ = 0;
        GLuint scratch_read_ = 0, scratch_draw_ = 0;
        GLuint push_ring_ = 0;
        uint64_t push_ring_size_ = 0, push_ring_pos_ = 0;
        std::map<std::array<uint64_t, max_color_targets + 1>, framebuffer> framebuffers_;

        std::mutex fence_mutex_;
        std::condition_variable fence_cv_;
        std::mutex list_mutex_;
        std::vector<std::unique_ptr<cmd_list>> lists_;
        std::vector<cmd_list*> free_lists_;
        std::mutex error_mutex_;
        std::vector<std::string> errors_;
        std::unordered_map<std::string, bool> reported_;

        device_counters counters_;
        std::atomic<uint64_t> buffer_bytes_{0};
        std::atomic<uint64_t> texture_bytes_{0};
        std::atomic<uint32_t> live_buffers_{0};
        std::atomic<uint32_t> live_textures_{0};
        uint64_t frame_ = 0;
        fence_handle frame_fence_;
        bool closing_ = false;
        uint32_t max_uniform_bindings_ = 24;
        uint32_t max_texture_units_ = 32;
    };

    std::unique_ptr<conduit::swapchain> make_swapchain(device& device, const swapchain_desc& desc);

    // ---- SPIR-V to GLSL ES ----
    struct translated_sampler {
        std::string name;
        uint32_t set = 0, binding = 0;
        uint32_t sampler_set = 0, sampler_binding = 0;
        bool dummy_sampler = false;
        GLenum target = GL_TEXTURE_2D;
    };

    struct translated_block {
        std::string name;
        uint32_t index = 0;
    };

    struct translated_stage {
        std::string glsl;
        std::vector<translated_sampler> samplers;
        std::vector<translated_block> uniform_blocks;
        bool needs_storage = false;
        bool uses_push = false;
        uint32_t push_bytes = 0;
    };

    bool translate_stage(const std::vector<uint32_t>& words, shader_stage stage, const std::string& entry,
                         uint32_t glsl_version, const pipeline_layout& layout, translated_stage& out,
                         std::string& error);
}
