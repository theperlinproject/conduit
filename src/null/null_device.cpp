#include "backends.hpp"
#include "common/handle_pool.hpp"
#include "common/retire_queue.hpp"
#include "common/state_validator.hpp"
#include "common/util.hpp"

#include <cstring>
#include <functional>
#include <mutex>

namespace conduit {
    namespace {
        struct null_buffer {
            buffer_desc desc;
            std::string name;
            std::vector<uint8_t> data;
        };

        struct null_texture {
            texture_desc desc;
            std::string name;
            std::vector<std::vector<std::vector<uint8_t>>> data;
            bool external = false;
        };

        struct null_sampler {
            sampler_desc desc;
        };
        struct null_shader {
            shader_desc desc;
            std::vector<uint8_t> code;
        };
        struct null_bind_layout {
            std::vector<bind_layout_entry> entries;
        };
        struct null_bind_group {
            bind_layout_handle layout;
            std::vector<bind_group_entry> entries;
            std::vector<bind_layout_entry> layout_entries;
        };
        struct null_pipeline_layout {
            pipeline_layout_desc desc;
        };
        struct null_pipeline {
            bool compute = false;
            graphics_pipeline_desc graphics;
            compute_pipeline_desc compute_desc;
        };
        struct null_query {
            uint32_t count = 0;
        };
        struct null_fence {
            uint64_t value = 0;
        };

        class null_device;

        class null_command_list final : public command_list {
        public:
            null_command_list(null_device& device, queue_type queue) : device_(device), queue_(queue) {}

            queue_type queue() const override { return queue_; }
            void reset() {
                ops_.clear();
                in_rendering_ = false;
                pipeline_ = {};
                bound_sets_ = {};
                index_buffer_ = {};
                vertex_buffers_ = {};
                color_formats_.fill(format::unknown);
                color_count_ = 0;
                depth_format_ = format::unknown;
            }
            void run_ops();

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
            void set_bind_group(uint32_t set, bind_group_handle group,
                                std::span<const uint32_t> dynamic_offsets) override;
            void push_constants(const void* data, uint32_t bytes, uint32_t offset) override;
            void set_viewport(const viewport&) override {}
            void set_scissor(const rect&) override {}
            void set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t offset) override;
            void set_index_buffer(buffer_handle buffer, index_type type, uint64_t offset) override;
            void set_stencil_reference(uint32_t) override {}
            void draw(uint32_t, uint32_t, uint32_t, uint32_t) override { check_draw("Draw"); }
            void draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t,
                              uint32_t) override;
            void draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws,
                                       uint32_t stride) override;
            void draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle count_buffer,
                                             uint64_t count_offset, uint32_t max_draws, uint32_t stride) override;
            void dispatch(uint32_t, uint32_t, uint32_t) override;
            void dispatch_indirect(buffer_handle args, uint64_t offset) override;
            void write_timestamp(query_set_handle, uint32_t) override {}
            void resolve_queries(query_set_handle, uint32_t, uint32_t, buffer_handle, uint64_t) override {}
            void begin_debug_label(const char*, uint32_t) override { label_depth_++; }
            void end_debug_label() override;
            void* native_handle() override { return nullptr; }
            void invalidate_bound_state() override {}
            int open_labels() const { return label_depth_; }

        private:
            void check_draw(const char* what);
            void error(const std::string& message);

            null_device& device_;
            queue_type queue_;
            int label_depth_ = 0;
            std::vector<std::function<void()>> ops_;
            bool in_rendering_ = false;
            pipeline_handle pipeline_;
            std::array<bind_group_handle, max_bind_groups> bound_sets_{};
            std::array<std::array<uint32_t, max_dynamic_offsets>, max_bind_groups> dynamic_offsets_{};
            buffer_handle index_buffer_;
            std::array<buffer_handle, max_vertex_streams> vertex_buffers_{};
            std::array<format, max_color_targets> color_formats_{};
            uint32_t color_count_ = 0;
            format depth_format_ = format::unknown;
            uint32_t push_bytes_set_ = 0;
        };

        class null_swapchain;

        class null_device final : public device {
        public:
            explicit null_device(const device_options& options) : options_(options) {
                caps_.adapter.api = api::none;
                caps_.adapter.name = "Null (software reference)";
                caps_.adapter.type = adapter_type::cpu;
                caps_.adapter.vendor_id = 0;
                device_features& f = caps_.features;
                f.multi_draw_indirect = f.draw_indirect_count = f.draw_indirect_first_instance = true;
                f.async_compute = f.dedicated_copy_queue = f.timestamp_queries = true;
                f.sampler_anisotropy = true;
                f.texture_compression_bc = true;
                f.enhanced_barriers = true;
                caps_.limits.max_texture2d = 16384;
                caps_.limits.max_texture_layers = 2048;
                caps_.limits.max_msaa_samples = 4;
                caps_.limits.max_anisotropy = 16.0f;
                caps_.limits.max_sampled_textures_per_stage = 128;
                for (size_t i = 1; i < caps_.formats.size(); i++)
                    caps_.formats[i] = format_use::sampled | format_use::sampled_linear | format_use::color_target |
                                       format_use::blendable | format_use::depth_target | format_use::storage |
                                       format_use::msaa;
            }

            ~null_device() override { retire_.flush_all(); }

            api get_api() const override { return api::none; }
            const device_caps& caps() const override { return caps_; }
            native_handles native() const override { return {}; }

            buffer_handle create_buffer(const buffer_desc& desc) override {
                null_buffer b;
                b.desc = desc;
                b.name = desc.name ? desc.name : "";
                b.data.assign(desc.size, 0);
                stats_.buffer_bytes += desc.size;
                stats_.live_buffers++;
                buffer_handle h = buffers_.alloc(std::move(b));
                buffers_.get(h)->desc.name = buffers_.get(h)->name.c_str();
                validator_.on_create(h, *get_desc(h));
                return h;
            }

            texture_handle create_texture(const texture_desc& desc) override {
                null_texture t;
                t.desc = desc;
                t.name = desc.name ? desc.name : "";
                t.desc.mips = (std::max)(1u, desc.mips);
                const uint32_t layers = layer_count_of(t.desc);
                t.data.resize(layers);
                uint64_t bytes = 0;
                for (auto& layer : t.data) {
                    layer.resize(t.desc.mips);
                    for (uint32_t m = 0; m < t.desc.mips; m++) {
                        const uint32_t w = mip_extent(desc.width, m), h = mip_extent(desc.height, m);
                        layer[m].assign(static_cast<size_t>(row_bytes(desc.format, w)) * row_blocks(desc.format, h) *
                                            depth_of(desc),
                                        0);
                        bytes += layer[m].size();
                    }
                }
                stats_.texture_bytes += bytes;
                stats_.live_textures++;
                texture_handle h = textures_.alloc(std::move(t));
                textures_.get(h)->desc.name = textures_.get(h)->name.c_str();
                validator_.on_create(h, *get_desc(h));
                return h;
            }

            sampler_handle create_sampler(const sampler_desc& desc) override {
                return samplers_.alloc(null_sampler{desc});
            }
            shader_handle create_shader(const shader_desc& desc) override {
                null_shader s;
                s.desc = desc;
                s.code.assign(desc.code.begin(), desc.code.end());
                return shaders_.alloc(std::move(s));
            }
            bind_layout_handle create_bind_layout(const bind_layout_desc& desc) override {
                return bind_layouts_.alloc(null_bind_layout{{desc.entries.begin(), desc.entries.end()}});
            }
            bind_group_handle create_bind_group(const bind_group_desc& desc) override {
                null_bind_layout* layout = bind_layouts_.get(desc.layout);
                if (!layout) {
                    report_device_error("CreateBindGroup: unknown layout");
                    return {};
                }
                null_bind_group g;
                g.layout = desc.layout;
                g.entries.assign(desc.entries.begin(), desc.entries.end());
                g.layout_entries = layout->entries;
                for (const bind_layout_entry& le : layout->entries) {
                    uint32_t found = 0;
                    for (const bind_group_entry& e : g.entries)
                        if (e.binding == le.binding)
                            found++;
                    if (found < le.count)
                        report_device_error("CreateBindGroup: binding " + std::to_string(le.binding) +
                                            " is not fully populated");
                }
                stats_.live_bind_groups++;
                return bind_groups_.alloc(std::move(g));
            }
            pipeline_layout_handle create_pipeline_layout(const pipeline_layout_desc& desc) override {
                if (desc.push_constant_bytes > max_push_constant_bytes)
                    report_device_error("CreatePipelineLayout: push constants exceed the limit");
                return pipeline_layouts_.alloc(null_pipeline_layout{desc});
            }
            pipeline_handle create_graphics_pipeline(const graphics_pipeline_desc& desc) override {
                if (!pipeline_layouts_.get(desc.layout))
                    report_device_error("CreateGraphicsPipeline: unknown layout");
                if (!shaders_.get(desc.vertex))
                    report_device_error("CreateGraphicsPipeline: missing vertex shader");
                for (uint32_t i = 0; i < desc.attribute_count; i++)
                    if (desc.attributes[i].stream >= desc.stream_count)
                        report_device_error("CreateGraphicsPipeline: attribute references a missing stream");
                null_pipeline p;
                p.graphics = desc;
                stats_.live_pipelines++;
                return pipelines_.alloc(p);
            }
            pipeline_handle create_compute_pipeline(const compute_pipeline_desc& desc) override {
                if (!shaders_.get(desc.compute))
                    report_device_error("CreateComputePipeline: missing shader");
                null_pipeline p;
                p.compute = true;
                p.compute_desc = desc;
                stats_.live_pipelines++;
                return pipelines_.alloc(p);
            }
            query_set_handle create_query_set(const query_set_desc& desc) override {
                return queries_.alloc(null_query{desc.count});
            }

            void destroy(buffer_handle h) override {
                null_buffer* b = buffers_.get(h);
                if (!b)
                    return;
                stats_.buffer_bytes -= b->desc.size;
                stats_.live_buffers--;
                validator_.on_destroy(h);
                retire([this, h] { buffers_.free(h); });
            }
            void destroy(texture_handle h) override {
                null_texture* t = textures_.get(h);
                if (!t)
                    return;
                for (auto& layer : t->data)
                    for (auto& mip : layer)
                        stats_.texture_bytes -= mip.size();
                stats_.live_textures--;
                validator_.on_destroy(h);
                retire([this, h] { textures_.free(h); });
            }
            void destroy(sampler_handle h) override {
                retire([this, h] { samplers_.free(h); });
            }
            void destroy(shader_handle h) override {
                retire([this, h] { shaders_.free(h); });
            }
            void destroy(bind_layout_handle h) override {
                retire([this, h] { bind_layouts_.free(h); });
            }
            void destroy(bind_group_handle h) override {
                if (bind_groups_.get(h))
                    stats_.live_bind_groups--;
                retire([this, h] { bind_groups_.free(h); });
            }
            void destroy(pipeline_layout_handle h) override {
                retire([this, h] { pipeline_layouts_.free(h); });
            }
            void destroy(pipeline_handle h) override {
                if (pipelines_.get(h))
                    stats_.live_pipelines--;
                retire([this, h] { pipelines_.free(h); });
            }
            void destroy(query_set_handle h) override {
                retire([this, h] { queries_.free(h); });
            }

            const buffer_desc* get_desc(buffer_handle h) const override {
                null_buffer* b = buffers_.get(h);
                return b ? &b->desc : nullptr;
            }
            const texture_desc* get_desc(texture_handle h) const override {
                null_texture* t = textures_.get(h);
                return t ? &t->desc : nullptr;
            }

            uint8_t* map(buffer_handle h) override {
                null_buffer* b = buffers_.get(h);
                if (!b || b->desc.memory == memory_class::gpu_only)
                    return nullptr;
                return b->data.data();
            }
            void flush(buffer_handle, uint64_t, uint64_t) override {}
            void invalidate(buffer_handle, uint64_t, uint64_t) override {}
            uint64_t gpu_address(buffer_handle h) const override { return h.value * 0x100000000ull; }

            fence_handle create_fence(uint64_t initial) override {
                fence_handle h = fences_.alloc(null_fence{});
                fences_.get(h)->value = initial;
                return h;
            }
            void destroy(fence_handle h) override { fences_.free(h); }
            uint64_t fence_value(fence_handle h) override {
                null_fence* f = fences_.get(h);
                return f ? f->value : 0;
            }
            bool wait_fence(fence_handle h, uint64_t value, uint64_t) override { return fence_value(h) >= value; }
            void signal_fence(fence_handle h, uint64_t value) override {
                if (null_fence* f = fences_.get(h))
                    f->value = std::max<uint64_t>(f->value, value);
            }

            command_list* acquire_command_list(queue_type queue) override;
            void submit(queue_type queue, const submit_desc& desc) override;

            std::unique_ptr<swapchain> create_swapchain(const swapchain_desc& desc) override;

            void begin_frame() override {
                frame_++;
                retire_.collect(submitted_);
            }
            void end_frame() override { retire_.collect(submitted_); }
            uint64_t frame_number() const override { return frame_; }
            uint32_t frame_slot() const override {
                return static_cast<uint32_t>(frame_ % (std::max)(1u, options_.frames_in_flight));
            }
            void wait_idle() override { retire_.collect(submitted_); }

            memory_budget query_memory_budget() override {
                memory_budget m;
                m.valid = true;
                m.device_budget = 8ull << 30;
                m.device_usage = stats_.buffer_bytes + stats_.texture_bytes;
                return m;
            }
            device_stats stats() const override { return stats_; }
            float timestamp_period_ns() const override { return 1.0f; }

            state_validator& validator() { return validator_; }
            void report_device_error(const std::string& message) { device_errors_.push_back(message); }
            std::vector<std::string> take_device_errors() { return std::move(device_errors_); }

            void retire(std::function<void()> fn) { retire_.push(submitted_, std::move(fn)); }

        private:
            handle_pool<buffer_tag, null_buffer> buffers_;
            handle_pool<texture_tag, null_texture> textures_;
            handle_pool<sampler_tag, null_sampler> samplers_;
            handle_pool<shader_tag, null_shader> shaders_;
            handle_pool<bind_layout_tag, null_bind_layout> bind_layouts_;
            handle_pool<bind_group_tag, null_bind_group> bind_groups_;
            handle_pool<pipeline_layout_tag, null_pipeline_layout> pipeline_layouts_;
            handle_pool<pipeline_tag, null_pipeline> pipelines_;
            handle_pool<query_set_tag, null_query> queries_;
            handle_pool<fence_tag, null_fence> fences_;
            device_stats stats_;
            queue_serials submitted_{};
            std::vector<std::unique_ptr<null_command_list>> lists_;
            std::vector<null_command_list*> free_lists_[queue_count];

            friend class null_command_list;
            friend class null_swapchain;
            device_options options_;
            device_caps caps_;
            state_validator validator_;
            retire_queue retire_;
            std::vector<std::string> device_errors_;
            uint64_t frame_ = 0;
        };

        void null_command_list::error(const std::string& message) {
            device_.report_device_error(message);
        }

        void null_command_list::end_debug_label() {
            if (--label_depth_ < 0)
                error("EndDebugLabel without BeginDebugLabel");
        }

        void null_command_list::barrier(std::span<const texture_barrier> textures,
                                        std::span<const buffer_barrier> buffers) {
            for (const texture_barrier& b : textures) {
                if (in_rendering_)
                    error("barrier recorded inside a rendering pass");
                device_.validator().barrier(b);
                device_.stats_.barriers++;
            }
            for (const buffer_barrier& b : buffers) {
                if (in_rendering_)
                    error("barrier recorded inside a rendering pass");
                device_.validator().barrier(b);
                device_.stats_.barriers++;
            }
        }

        void null_command_list::uav_barrier(buffer_handle) {
            device_.stats_.barriers++;
        }
        void null_command_list::uav_barrier(texture_handle) {
            device_.stats_.barriers++;
        }

        void null_command_list::copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src,
                                            uint64_t src_offset, uint64_t size) {
            device_.validator().expect_buffer(src, state::copy_src, "CopyBuffer src");
            device_.validator().expect_buffer(dst, state::copy_dst, "CopyBuffer dst");
            ops_.emplace_back([this, dst, dst_offset, src, src_offset, size] {
                null_buffer* d = device_.buffers_.get(dst);
                null_buffer* s = device_.buffers_.get(src);
                if (!d || !s || dst_offset + size > d->data.size() || src_offset + size > s->data.size()) {
                    device_.report_device_error("CopyBuffer out of range");
                    return;
                }
                std::memmove(d->data.data() + dst_offset, s->data.data() + src_offset, size);
            });
        }

        void null_command_list::copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                                       std::span<const buffer_texture_copy> regions) {
            device_.validator().expect_buffer(src, state::copy_src, "CopyBufferToTexture src");
            std::vector<buffer_texture_copy> copy(regions.begin(), regions.end());
            for (const auto& r : copy)
                device_.validator().expect_texture(dst, {r.mip, 1, r.layer, 1}, state::copy_dst,
                                                   "CopyBufferToTexture dst");
            ops_.emplace_back([this, dst, src, copy] {
                null_texture* t = device_.textures_.get(dst);
                null_buffer* b = device_.buffers_.get(src);
                if (!t || !b) {
                    device_.report_device_error("CopyBufferToTexture: bad handle");
                    return;
                }
                const format_info& fi = get_format_info(t->desc.format);
                for (const buffer_texture_copy& r : copy) {
                    auto& mip = t->data[r.layer][r.mip];
                    const uint32_t mip_w = mip_extent(t->desc.width, r.mip);
                    const uint32_t tex_row = row_bytes(t->desc.format, mip_w);
                    const uint32_t row_size = row_bytes(t->desc.format, r.width);
                    const uint32_t pitch = r.buffer_row_pitch ? r.buffer_row_pitch : row_size;
                    const uint32_t rows = row_blocks(t->desc.format, r.height);
                    for (uint32_t z = 0; z < r.depth; z++)
                        for (uint32_t row = 0; row < rows; row++) {
                            const uint64_t src_off = r.buffer_offset + (static_cast<uint64_t>(z) * rows + row) * pitch;
                            const uint64_t dst_off =
                                (static_cast<uint64_t>(r.z + z) *
                                     row_blocks(t->desc.format, mip_extent(t->desc.height, r.mip)) +
                                 r.y / fi.block_height + row) *
                                    tex_row +
                                (r.x / fi.block_width) * fi.bytes_per_block;
                            if (src_off + row_size > b->data.size() || dst_off + row_size > mip.size()) {
                                device_.report_device_error("CopyBufferToTexture out of range");
                                return;
                            }
                            std::memcpy(mip.data() + dst_off, b->data.data() + src_off, row_size);
                        }
                }
            });
        }

        void null_command_list::copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                                       std::span<const buffer_texture_copy> regions) {
            device_.validator().expect_buffer(dst, state::copy_dst, "CopyTextureToBuffer dst");
            std::vector<buffer_texture_copy> copy(regions.begin(), regions.end());
            for (const auto& r : copy)
                device_.validator().expect_texture(src, {r.mip, 1, r.layer, 1}, state::copy_src,
                                                   "CopyTextureToBuffer src");
            ops_.emplace_back([this, dst, src, copy] {
                null_texture* t = device_.textures_.get(src);
                null_buffer* b = device_.buffers_.get(dst);
                if (!t || !b) {
                    device_.report_device_error("CopyTextureToBuffer: bad handle");
                    return;
                }
                const format_info& fi = get_format_info(t->desc.format);
                for (const buffer_texture_copy& r : copy) {
                    auto& mip = t->data[r.layer][r.mip];
                    const uint32_t mip_w = mip_extent(t->desc.width, r.mip);
                    const uint32_t tex_row = row_bytes(t->desc.format, mip_w);
                    const uint32_t row_size = row_bytes(t->desc.format, r.width);
                    const uint32_t pitch = r.buffer_row_pitch ? r.buffer_row_pitch : row_size;
                    const uint32_t rows = row_blocks(t->desc.format, r.height);
                    for (uint32_t z = 0; z < r.depth; z++)
                        for (uint32_t row = 0; row < rows; row++) {
                            const uint64_t dst_off = r.buffer_offset + (static_cast<uint64_t>(z) * rows + row) * pitch;
                            const uint64_t src_off =
                                (static_cast<uint64_t>(r.z + z) *
                                     row_blocks(t->desc.format, mip_extent(t->desc.height, r.mip)) +
                                 r.y / fi.block_height + row) *
                                    tex_row +
                                (r.x / fi.block_width) * fi.bytes_per_block;
                            if (dst_off + row_size > b->data.size() || src_off + row_size > mip.size()) {
                                device_.report_device_error("CopyTextureToBuffer out of range");
                                return;
                            }
                            std::memcpy(b->data.data() + dst_off, mip.data() + src_off, row_size);
                        }
                }
            });
        }

        void null_command_list::copy_texture(texture_handle dst, texture_handle src,
                                             std::span<const texture_copy> regions) {
            std::vector<texture_copy> copy(regions.begin(), regions.end());
            for (const auto& r : copy) {
                device_.validator().expect_texture(src, {r.src_mip, 1, r.src_layer, 1}, state::copy_src,
                                                   "CopyTexture src");
                device_.validator().expect_texture(dst, {r.dst_mip, 1, r.dst_layer, 1}, state::copy_dst,
                                                   "CopyTexture dst");
            }
            ops_.emplace_back([this, dst, src, copy] {
                null_texture* d = device_.textures_.get(dst);
                null_texture* s = device_.textures_.get(src);
                if (!d || !s || d->desc.format != s->desc.format) {
                    device_.report_device_error("CopyTexture: bad handle or format mismatch");
                    return;
                }
                const format_info& fi = get_format_info(s->desc.format);
                for (const texture_copy& r : copy) {
                    auto& sm = s->data[r.src_layer][r.src_mip];
                    auto& dm = d->data[r.dst_layer][r.dst_mip];
                    const uint32_t s_row = row_bytes(s->desc.format, mip_extent(s->desc.width, r.src_mip));
                    const uint32_t d_row = row_bytes(d->desc.format, mip_extent(d->desc.width, r.dst_mip));
                    const uint32_t row_size = row_bytes(s->desc.format, r.width);
                    for (uint32_t y = 0; y < r.height; y++) {
                        const uint64_t so =
                            static_cast<uint64_t>(r.src_y + y) * s_row + r.src_x / fi.block_width * fi.bytes_per_block;
                        const uint64_t d_o =
                            static_cast<uint64_t>(r.dst_y + y) * d_row + r.dst_x / fi.block_width * fi.bytes_per_block;
                        if (so + row_size > sm.size() || d_o + row_size > dm.size()) {
                            device_.report_device_error("CopyTexture out of range");
                            return;
                        }
                        std::memcpy(dm.data() + d_o, sm.data() + so, row_size);
                    }
                }
            });
        }

        void null_command_list::fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) {
            device_.validator().expect_buffer(dst, state::copy_dst, "FillBuffer dst");
            ops_.emplace_back([this, dst, offset, size, value] {
                null_buffer* b = device_.buffers_.get(dst);
                if (!b || offset + size > b->data.size()) {
                    device_.report_device_error("FillBuffer out of range");
                    return;
                }
                for (uint64_t i = 0; i + 4 <= size; i += 4)
                    std::memcpy(b->data.data() + offset + i, &value, 4);
            });
        }

        void null_command_list::resolve_texture(texture_handle dst, texture_handle src) {
            device_.validator().expect_texture(src, {}, state::resolve_src, "ResolveTexture src");
            device_.validator().expect_texture(dst, {}, state::resolve_dst, "ResolveTexture dst");
        }

        void null_command_list::begin_rendering(const rendering_desc& desc) {
            if (in_rendering_)
                error("BeginRendering inside a rendering pass");
            in_rendering_ = true;
            color_count_ = desc.color_count;
            for (uint32_t i = 0; i < desc.color_count; i++) {
                const color_attachment& a = desc.colors[i];
                const texture_desc* td = device_.get_desc(a.texture);
                if (!td) {
                    error("BeginRendering: unknown colour attachment");
                    continue;
                }
                color_formats_[i] = td->format;
                if (!has(td->usage, texture_usage::color_target))
                    error("BeginRendering: texture '" + std::string(td->name) + "' lacks ColorTarget usage");
                device_.validator().expect_texture(a.texture, {a.mip, 1, a.layer, 1}, state::color_target,
                                                   "BeginRendering colour");
            }
            depth_format_ = format::unknown;
            if (desc.depth.texture) {
                const texture_desc* td = device_.get_desc(desc.depth.texture);
                if (!td) {
                    error("BeginRendering: unknown depth attachment");
                } else {
                    depth_format_ = td->format;
                    if (!has(td->usage, texture_usage::depth_target))
                        error("BeginRendering: texture '" + std::string(td->name) + "' lacks DepthTarget usage");
                    device_.validator().expect_texture(desc.depth.texture, {desc.depth.mip, 1, desc.depth.layer, 1},
                                                       desc.depth.read_only ? state::depth_read : state::depth_write,
                                                       "BeginRendering depth");
                }
            }
        }

        void null_command_list::end_rendering() {
            if (!in_rendering_)
                error("EndRendering without BeginRendering");
            in_rendering_ = false;
        }

        void null_command_list::set_pipeline(pipeline_handle handle) {
            null_pipeline* p = device_.pipelines_.get(handle);
            if (!p) {
                error("SetPipeline: unknown pipeline");
                return;
            }
            if (p->compute == in_rendering_)
                error(p->compute ? "compute pipeline set inside a rendering pass"
                                 : "graphics pipeline set outside a rendering pass");
            device_.stats_.pipeline_binds++;
            pipeline_ = handle;
        }

        void null_command_list::set_bind_group(uint32_t set, bind_group_handle group,
                                               std::span<const uint32_t> offsets) {
            null_bind_group* g = device_.bind_groups_.get(group);
            if (!g || set >= max_bind_groups) {
                error("SetBindGroup: unknown group or set index");
                return;
            }
            uint32_t dynamic_count = 0;
            std::vector<state_validator::bound_resource> bound;
            for (const bind_layout_entry& le : g->layout_entries) {
                if (le.type == binding_type::uniform_buffer_dynamic)
                    dynamic_count++;
                for (const bind_group_entry& e : g->entries) {
                    if (e.binding != le.binding)
                        continue;
                    state_validator::bound_resource r;
                    r.range = e.texture.range;
                    switch (le.type) {
                    case binding_type::sampled_texture:
                        r.texture = e.texture.texture;
                        r.want = state::shader_read | state::depth_read_sampled;
                        break;
                    case binding_type::storage_texture_rw:
                        r.texture = e.texture.texture;
                        r.want = state::shader_write;
                        break;
                    case binding_type::storage_buffer_rw:
                        r.buffer = e.buffer.buffer;
                        r.want = state::shader_write;
                        break;
                    case binding_type::storage_buffer:
                        r.buffer = e.buffer.buffer;
                        r.want = state::shader_read;
                        break;
                    case binding_type::uniform_buffer:
                    case binding_type::uniform_buffer_dynamic:
                        r.buffer = e.buffer.buffer;
                        r.want = state::uniform_buffer;
                        break;
                    case binding_type::sampler:
                        break;
                    }
                    if (r.texture || r.buffer)
                        bound.push_back(r);
                }
            }
            if (dynamic_count != offsets.size())
                error("SetBindGroup: expected " + std::to_string(dynamic_count) + " dynamic offsets, got " +
                      std::to_string(offsets.size()));
            device_.validator().expect_all(bound, "SetBindGroup");
            bound_sets_[set] = group;
            device_.stats_.bind_group_binds++;
        }

        void null_command_list::push_constants(const void*, uint32_t bytes, uint32_t offset) {
            if (offset + bytes > max_push_constant_bytes)
                error("PushConstants exceed the limit");
            push_bytes_set_ = (std::max)(push_bytes_set_, offset + bytes);
        }

        void null_command_list::set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t) {
            if (slot >= max_vertex_streams) {
                error("SetVertexBuffer: slot out of range");
                return;
            }
            device_.validator().expect_buffer(buffer, state::vertex_buffer, "SetVertexBuffer");
            vertex_buffers_[slot] = buffer;
        }

        void null_command_list::set_index_buffer(buffer_handle buffer, index_type, uint64_t) {
            device_.validator().expect_buffer(buffer, state::index_buffer, "SetIndexBuffer");
            index_buffer_ = buffer;
        }

        void null_command_list::check_draw(const char* what) {
            if (!in_rendering_)
                error(std::string(what) + " outside a rendering pass");
            null_pipeline* p = device_.pipelines_.get(pipeline_);
            if (!p || p->compute) {
                error(std::string(what) + " without a graphics pipeline");
                return;
            }
            const graphics_pipeline_desc& g = p->graphics;
            if (g.color_count != color_count_)
                error(std::string(what) + ": pipeline has " + std::to_string(g.color_count) +
                      " colour targets, pass has " + std::to_string(color_count_));
            else
                for (uint32_t i = 0; i < color_count_; i++)
                    if (g.color_formats[i] != color_formats_[i])
                        error(std::string(what) + ": colour target " + std::to_string(i) +
                              " format differs from the pipeline's");
            if (g.depth_format != depth_format_)
                error(std::string(what) + ": depth format differs from the pipeline's");
            for (uint32_t i = 0; i < g.stream_count; i++)
                if (!vertex_buffers_[i])
                    error(std::string(what) + ": vertex stream " + std::to_string(i) + " has no buffer");
            if (null_pipeline_layout* layout = device_.pipeline_layouts_.get(g.layout))
                for (uint32_t s = 0; s < max_bind_groups; s++)
                    if (layout->desc.sets[s] && !bound_sets_[s])
                        error(std::string(what) + ": bind group " + std::to_string(s) + " not set");
            device_.stats_.draws++;
        }

        void null_command_list::draw_indexed(uint32_t, uint32_t, uint32_t, int32_t, uint32_t) {
            check_draw("DrawIndexed");
            if (!index_buffer_)
                error("DrawIndexed without an index buffer");
        }

        void null_command_list::draw_indexed_indirect(buffer_handle args, uint64_t, uint32_t, uint32_t) {
            check_draw("DrawIndexedIndirect");
            device_.validator().expect_buffer(args, state::indirect_argument, "DrawIndexedIndirect args");
            if (!index_buffer_)
                error("DrawIndexedIndirect without an index buffer");
            device_.stats_.indirect_draws++;
        }

        void null_command_list::draw_indexed_indirect_count(buffer_handle args, uint64_t, buffer_handle count_buffer,
                                                            uint64_t, uint32_t, uint32_t) {
            check_draw("DrawIndexedIndirectCount");
            device_.validator().expect_buffer(args, state::indirect_argument, "DrawIndexedIndirectCount args");
            device_.validator().expect_buffer(count_buffer, state::indirect_argument, "DrawIndexedIndirectCount count");
            if (!index_buffer_)
                error("DrawIndexedIndirectCount without an index buffer");
            device_.stats_.indirect_draws++;
        }

        void null_command_list::dispatch(uint32_t, uint32_t, uint32_t) {
            null_pipeline* p = device_.pipelines_.get(pipeline_);
            if (in_rendering_)
                error("Dispatch inside a rendering pass");
            if (!p || !p->compute)
                error("Dispatch without a compute pipeline");
            device_.stats_.dispatches++;
        }

        void null_command_list::dispatch_indirect(buffer_handle args, uint64_t) {
            dispatch(0, 0, 0);
            device_.validator().expect_buffer(args, state::indirect_argument, "DispatchIndirect args");
        }

        void null_command_list::run_ops() {
            for (auto& op : ops_)
                op();
            ops_.clear();
        }

        command_list* null_device::acquire_command_list(queue_type queue) {
            auto& pool = free_lists_[static_cast<size_t>(queue)];
            if (!pool.empty()) {
                null_command_list* list = pool.back();
                pool.pop_back();
                list->reset();
                return list;
            }
            lists_.push_back(std::make_unique<null_command_list>(*this, queue));
            return lists_.back().get();
        }

        void null_device::submit(queue_type queue, const submit_desc& desc) {
            for (const fence_op& w : desc.waits)
                if (fence_value(w.fence) < w.value)
                    report_device_error("Submit waits on a fence value that was never signalled (would deadlock)");
            for (command_list* list : desc.lists) {
                auto* null = static_cast<null_command_list*>(list);
                if (null->open_labels() != 0)
                    report_device_error("command list submitted with an open debug label");
                null->run_ops();
                free_lists_[static_cast<size_t>(queue)].push_back(null);
            }
            submitted_[static_cast<size_t>(queue)]++;
            stats_.submits++;
            for (const fence_op& s : desc.signals)
                signal_fence(s.fence, s.value);
        }

        class null_swapchain final : public swapchain {
        public:
            null_swapchain(null_device& device, const swapchain_desc& desc) : device_(device), desc_(desc) { build(); }
            ~null_swapchain() override { free(); }

            bool acquire(image& out) override {
                if (width_ == 0 || height_ == 0)
                    return false;
                index_ = (index_ + 1) % images_.size();
                out.texture = images_[index_];
                out.index = index_;
                return true;
            }
            void present() override {}
            bool resize(uint32_t w, uint32_t h) override {
                if (w == width_ && h == height_)
                    return true;
                free();
                desc_.width = w;
                desc_.height = h;
                build();
                return w && h;
            }
            void set_present_mode(present_mode mode) override { desc_.present = mode; }
            format get_format() const override { return format::bgra8_unorm; }
            uint32_t width() const override { return width_; }
            uint32_t height() const override { return height_; }
            present_mode get_present_mode() const override { return desc_.present; }
            uint32_t image_count() const override { return static_cast<uint32_t>(images_.size()); }

        private:
            void build() {
                width_ = desc_.width;
                height_ = desc_.height;
                if (!width_ || !height_)
                    return;
                for (uint32_t i = 0; i < (std::max)(2u, desc_.image_count); i++) {
                    texture_desc td;
                    td.format = format::bgra8_unorm;
                    td.width = width_;
                    td.height = height_;
                    td.usage = texture_usage::color_target | texture_usage::copy_src |
                               (desc_.allow_sampling ? texture_usage::sampled : texture_usage::none);
                    td.name = "backbuffer";
                    images_.push_back(device_.create_texture(td));
                }
            }
            void free() {
                for (texture_handle h : images_)
                    device_.destroy(h);
                images_.clear();
                index_ = 0;
            }

            null_device& device_;
            swapchain_desc desc_;
            std::vector<texture_handle> images_;
            uint32_t width_ = 0, height_ = 0, index_ = 0;
        };

        std::unique_ptr<swapchain> null_device::create_swapchain(const swapchain_desc& desc) {
            return std::make_unique<null_swapchain>(*this, desc);
        }

        class null_instance final : public instance {
        public:
            null_instance() {
                adapter_info a;
                a.api = api::none;
                a.name = "Null (software reference)";
                a.type = adapter_type::cpu;
                adapters_.push_back(a);
            }
            api get_api() const override { return api::none; }
            const std::vector<adapter_info>& adapters() const override { return adapters_; }
            std::unique_ptr<device> create_device(const device_options& options) override {
                return std::make_unique<null_device>(options);
            }

        private:
            std::vector<adapter_info> adapters_;
        };
    }

    namespace backends {
        std::unique_ptr<instance> create_null_instance(const instance_options&) {
            return std::make_unique<null_instance>();
        }
    }

    size_t null_device_error_count(device& device);
    std::vector<std::string> null_device_take_errors(device& device);

    size_t null_device_error_count(conduit::device& device) {
        auto* null = dynamic_cast<null_device*>(&device);
        return null ? null->validator().error_count() : 0;
    }

    std::vector<std::string> null_device_take_errors(conduit::device& device) {
        std::vector<std::string> out;
        if (auto* null = dynamic_cast<null_device*>(&device)) {
            out = null->validator().take_errors();
            for (auto& e : null->take_device_errors())
                out.push_back(std::move(e));
        }
        return out;
    }
}
