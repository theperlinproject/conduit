#include "gles/gles_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace conduit::gles {
    namespace {
        // gl takes offsets into the bound buffer as pointers
        const void* gl_offset(uint64_t offset) {
            return reinterpret_cast<const void*>(static_cast<uintptr_t>(offset)); // NOLINT(performance-no-int-to-ptr)
        }

        enum op : uint16_t {
            op_memory_barrier,
            op_copy_buffer,
            op_buffer_to_texture,
            op_texture_to_buffer,
            op_copy_texture,
            op_fill_buffer,
            op_resolve_texture,
            op_begin_rendering,
            op_end_rendering,
            op_set_pipeline,
            op_set_bind_group,
            op_push_constants,
            op_viewport,
            op_scissor,
            op_vertex_buffer,
            op_index_buffer,
            op_stencil_reference,
            op_draw,
            op_draw_indexed,
            op_draw_indexed_indirect,
            op_draw_indexed_indirect_count,
            op_dispatch,
            op_dispatch_indirect,
            op_resolve_queries,
            op_debug_begin,
            op_debug_end,
            op_invalidate,
        };

        struct p_barrier {
            GLbitfield bits;
        };
        struct p_copy_buffer {
            buffer_handle dst, src;
            uint64_t dst_offset, src_offset, size;
        };
        struct p_buffer_texture {
            texture_handle texture;
            buffer_handle buffer;
            uint32_t count;
        };
        struct p_copy_texture {
            texture_handle dst, src;
            uint32_t count;
        };
        struct p_fill {
            buffer_handle dst;
            uint64_t offset, size;
            uint32_t value;
        };
        struct p_resolve {
            texture_handle dst, src;
        };
        struct p_begin {
            rendering_desc desc;
        };
        struct p_pipeline {
            pipeline_handle pipeline;
        };
        struct p_group {
            uint32_t set;
            bind_group_handle group;
            uint32_t count;
        };
        struct p_push {
            uint32_t offset, bytes;
        };
        struct p_vertex {
            uint32_t slot;
            buffer_handle buffer;
            uint64_t offset;
        };
        struct p_index {
            buffer_handle buffer;
            index_type type;
            uint64_t offset;
        };
        struct p_u32 {
            uint32_t value;
        };
        struct p_draw {
            uint32_t vertex_count, instance_count, first_vertex, first_instance;
        };
        struct p_draw_indexed {
            uint32_t index_count, instance_count, first_index;
            int32_t vertex_offset;
            uint32_t first_instance;
        };
        struct p_indirect {
            buffer_handle args;
            uint64_t offset;
            uint32_t max_draws, stride;
        };
        struct p_dispatch {
            uint32_t x, y, z;
        };
        struct p_dispatch_indirect {
            buffer_handle args;
            uint64_t offset;
        };
        struct p_queries {
            buffer_handle dst;
            uint64_t offset;
            uint32_t count;
        };

        GLbitfield barrier_bits(state before, state after) {
            if (!has(before, state::shader_write))
                return 0;
            GLbitfield bits = 0;
            if (has(after, state::vertex_buffer))
                bits |= GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT;
            if (has(after, state::index_buffer))
                bits |= GL_ELEMENT_ARRAY_BARRIER_BIT;
            if (has(after, state::uniform_buffer))
                bits |= GL_UNIFORM_BARRIER_BIT;
            if (has(after, state::indirect_argument))
                bits |= GL_COMMAND_BARRIER_BIT;
            if (has(after, state::shader_read) || has(after, state::shader_write))
                bits |= GL_TEXTURE_FETCH_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                        GL_SHADER_STORAGE_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT;
            if (has(after, state::copy_src) || has(after, state::copy_dst))
                bits |= GL_BUFFER_UPDATE_BARRIER_BIT | GL_TEXTURE_UPDATE_BARRIER_BIT | GL_PIXEL_BUFFER_BARRIER_BIT;
            if (has(after, state::color_target) || has(after, state::depth_write) || has(after, state::depth_read) ||
                has(after, state::present) || has(after, state::resolve_src) || has(after, state::resolve_dst))
                bits |= GL_FRAMEBUFFER_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT;
            return bits ? bits : static_cast<GLbitfield>(GL_ALL_BARRIER_BITS);
        }

        uint16_t float_to_half(float f) {
            uint32_t bits;
            std::memcpy(&bits, &f, 4);
            const uint32_t sign = (bits >> 16) & 0x8000u;
            int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mantissa = bits & 0x7FFFFFu;
            if (((bits >> 23) & 0xFF) == 0xFF)
                return static_cast<uint16_t>(sign | 0x7C00u | (mantissa ? 0x200u : 0u));
            if (exponent >= 31)
                return static_cast<uint16_t>(sign | 0x7C00u);
            if (exponent <= 0) {
                if (exponent < -10)
                    return static_cast<uint16_t>(sign);
                mantissa |= 0x800000u;
                const uint32_t shift = static_cast<uint32_t>(14 - exponent);
                return static_cast<uint16_t>(sign | ((mantissa + (1u << (shift - 1))) >> shift));
            }
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | ((mantissa + 0x1000u) >> 13));
        }
    }

    // ---------------------------------------------------------------- recording

    void cmd_list::begin() {
        stream_.clear();
        counters_ = {};
    }

    void cmd_list::barrier(std::span<const texture_barrier> textures, std::span<const buffer_barrier> buffers) {
        counters_.barriers += textures.size() + buffers.size();
        if (!device_.features().compute && !device_.features().storage_images)
            return;
        GLbitfield bits = 0;
        for (const texture_barrier& t : textures)
            bits |= barrier_bits(t.before, t.after);
        for (const buffer_barrier& b : buffers)
            bits |= barrier_bits(b.before, b.after);
        if (bits)
            emit(op_memory_barrier, p_barrier{bits});
    }

    void cmd_list::uav_barrier(buffer_handle) {
        counters_.barriers++;
        if (device_.features().compute)
            emit(op_memory_barrier, p_barrier{static_cast<GLbitfield>(GL_ALL_BARRIER_BITS)});
    }

    void cmd_list::uav_barrier(texture_handle) {
        counters_.barriers++;
        if (device_.features().compute)
            emit(op_memory_barrier, p_barrier{static_cast<GLbitfield>(GL_ALL_BARRIER_BITS)});
    }

    void cmd_list::copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src, uint64_t src_offset,
                               uint64_t size) {
        emit(op_copy_buffer, p_copy_buffer{dst, src, dst_offset, src_offset, size});
    }

    void cmd_list::copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        emit(op_buffer_to_texture, p_buffer_texture{dst, src, static_cast<uint32_t>(regions.size())}, regions.data(),
             regions.size_bytes());
    }

    void cmd_list::copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        emit(op_texture_to_buffer, p_buffer_texture{src, dst, static_cast<uint32_t>(regions.size())}, regions.data(),
             regions.size_bytes());
    }

    void cmd_list::copy_texture(texture_handle dst, texture_handle src, std::span<const texture_copy> regions) {
        emit(op_copy_texture, p_copy_texture{dst, src, static_cast<uint32_t>(regions.size())}, regions.data(),
             regions.size_bytes());
    }

    void cmd_list::fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) {
        emit(op_fill_buffer, p_fill{dst, offset, size, value});
    }

    void cmd_list::resolve_texture(texture_handle dst, texture_handle src) {
        emit(op_resolve_texture, p_resolve{dst, src});
    }

    void cmd_list::begin_rendering(const rendering_desc& desc) {
        emit(op_begin_rendering, p_begin{desc});
    }

    void cmd_list::end_rendering() {
        emit_op(op_end_rendering);
    }

    void cmd_list::set_pipeline(pipeline_handle pipeline) {
        counters_.pipeline_binds++;
        emit(op_set_pipeline, p_pipeline{pipeline});
    }

    void cmd_list::set_bind_group(uint32_t set, bind_group_handle group, std::span<const uint32_t> dynamic_offsets) {
        counters_.group_binds++;
        emit(op_set_bind_group, p_group{set, group, static_cast<uint32_t>(dynamic_offsets.size())},
             dynamic_offsets.data(), dynamic_offsets.size_bytes());
    }

    void cmd_list::push_constants(const void* data, uint32_t bytes, uint32_t offset) {
        emit(op_push_constants, p_push{offset, bytes}, data, bytes);
    }

    void cmd_list::set_viewport(const viewport& v) {
        emit(op_viewport, v);
    }

    void cmd_list::set_scissor(const rect& r) {
        emit(op_scissor, r);
    }

    void cmd_list::set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t offset) {
        emit(op_vertex_buffer, p_vertex{slot, buffer, offset});
    }

    void cmd_list::set_index_buffer(buffer_handle buffer, index_type type, uint64_t offset) {
        emit(op_index_buffer, p_index{buffer, type, offset});
    }

    void cmd_list::set_stencil_reference(uint32_t value) {
        emit(op_stencil_reference, p_u32{value});
    }

    void cmd_list::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                        uint32_t first_instance) {
        counters_.draws++;
        emit(op_draw, p_draw{vertex_count, instance_count, first_vertex, first_instance});
    }

    void cmd_list::draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                                int32_t vertex_offset, uint32_t first_instance) {
        counters_.draws++;
        emit(op_draw_indexed, p_draw_indexed{index_count, instance_count, first_index, vertex_offset, first_instance});
    }

    void cmd_list::draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws, uint32_t stride) {
        counters_.indirect++;
        emit(op_draw_indexed_indirect, p_indirect{args, offset, max_draws, stride});
    }

    void cmd_list::draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle, uint64_t,
                                               uint32_t max_draws, uint32_t stride) {
        counters_.indirect++;
        emit(op_draw_indexed_indirect_count, p_indirect{args, offset, max_draws, stride});
    }

    void cmd_list::dispatch(uint32_t x, uint32_t y, uint32_t z) {
        counters_.dispatches++;
        emit(op_dispatch, p_dispatch{x, y, z});
    }

    void cmd_list::dispatch_indirect(buffer_handle args, uint64_t offset) {
        counters_.dispatches++;
        emit(op_dispatch_indirect, p_dispatch_indirect{args, offset});
    }

    void cmd_list::write_timestamp(query_set_handle, uint32_t) {}

    void cmd_list::resolve_queries(query_set_handle, uint32_t, uint32_t count, buffer_handle dst, uint64_t dst_offset) {
        // there are no timestamps in OpenGL ES 3.x: the results read as zero
        emit(op_resolve_queries, p_queries{dst, dst_offset, count});
    }

    void cmd_list::begin_debug_label(const char* name, uint32_t) {
        const size_t n = std::strlen(name);
        emit(op_debug_begin, p_u32{static_cast<uint32_t>(n)}, name, n);
    }

    void cmd_list::end_debug_label() {
        emit_op(op_debug_end);
    }

    void cmd_list::invalidate_bound_state() {
        emit_op(op_invalidate);
    }

    // ---------------------------------------------------------------- device helpers

    void device::forget_buffer(GLuint gl) {
        for (attrib_state& a : cache_.attribs)
            if (a.valid && a.buffer == gl)
                a.valid = false;
        if (cache_.array_buffer == gl)
            cache_.array_buffer = ~0u;
        if (cache_.element_buffer == gl)
            cache_.element_buffer = ~0u;
    }

    void device::resync() {
        cache_.reset();
        gl_.glBindVertexArray(vao_);
        gl_.glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        gl_.glPixelStorei(GL_PACK_ALIGNMENT, 1);
    }

    uint64_t device::push_alloc(uint32_t bytes) {
        const uint64_t alignment = std::max<uint64_t>(caps_.limits.uniform_buffer_alignment, 16);
        uint64_t pos = align_up(push_ring_pos_, alignment);
        if (pos + bytes > push_ring_size_) {
            // a fresh store: what is still queued keeps reading the old one
            gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, push_ring_);
            gl_.glBufferData(GL_COPY_WRITE_BUFFER, static_cast<GLsizeiptr>(push_ring_size_), nullptr, GL_STREAM_DRAW);
            pos = 0;
        }
        push_ring_pos_ = pos + bytes;
        return pos;
    }

    const framebuffer* device::framebuffer_for(const std::array<uint64_t, max_color_targets + 1>& key,
                                               const rendering_desc& desc, bool) {
        const auto found = framebuffers_.find(key);
        if (found != framebuffers_.end())
            return &found->second;

        framebuffer fb;
        fb.key = key;
        gl_.glGenFramebuffers(1, &fb.id);
        bind_framebuffer(GL_FRAMEBUFFER, fb.id);
        const auto attach = [&](GLenum point, const texture& t, uint32_t mip, uint32_t layer) {
            if (t.renderbuffer) {
                gl_.glFramebufferRenderbuffer(GL_FRAMEBUFFER, point, GL_RENDERBUFFER, t.gl);
            } else if (t.target == GL_TEXTURE_2D) {
                gl_.glFramebufferTexture2D(GL_FRAMEBUFFER, point, GL_TEXTURE_2D, t.gl, static_cast<GLint>(mip));
            } else if (t.target == GL_TEXTURE_CUBE_MAP) {
                gl_.glFramebufferTexture2D(GL_FRAMEBUFFER, point, GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer, t.gl,
                                           static_cast<GLint>(mip));
            } else {
                gl_.glFramebufferTextureLayer(GL_FRAMEBUFFER, point, t.gl, static_cast<GLint>(mip),
                                              static_cast<GLint>(layer));
            }
        };
        GLenum buffers[max_color_targets];
        for (uint32_t i = 0; i < desc.color_count; i++) {
            const color_attachment& c = desc.colors[i];
            const texture* t = textures_.get(c.texture);
            if (t)
                attach(GL_COLOR_ATTACHMENT0 + i, *t, c.mip, c.layer);
            buffers[i] = GL_COLOR_ATTACHMENT0 + i;
        }
        fb.color_count = desc.color_count;
        if (desc.color_count) {
            gl_.glDrawBuffers(static_cast<GLsizei>(desc.color_count), buffers);
            gl_.glReadBuffer(GL_COLOR_ATTACHMENT0);
        } else {
            const GLenum none = GL_NONE;
            gl_.glDrawBuffers(1, &none);
            gl_.glReadBuffer(GL_NONE);
        }
        if (const texture* d = textures_.get(desc.depth.texture)) {
            fb.depth_attachment = has_stencil(d->desc.format) ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;
            attach(fb.depth_attachment, *d, desc.depth.mip, desc.depth.layer);
        }
        const GLenum status = gl_.glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE)
            report_error("a render pass framebuffer is incomplete (status 0x" + [&] {
                char text[16];
                std::snprintf(text, sizeof(text), "%X", status);
                return std::string(text);
            }() + "): the attachments' formats or sizes do not go together");
        return &framebuffers_.emplace(key, fb).first->second;
    }

    void device::present_texture(texture_handle handle, gl_surface& surface, present_mode mode) {
        const texture* t = textures_.get(handle);
        if (!t || t->renderbuffer)
            return;
        if (!surface.bind())
            return;
        surface.set_interval(mode == present_mode::immediate && features_.vsync_off ? 0 : 1);
        uint32_t dw = 0, dh = 0;
        surface.extent(dw, dh);

        use_scissor_test(false);
        bind_framebuffer(GL_READ_FRAMEBUFFER, scratch_read_);
        gl_.glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->gl, 0);
        gl_.glReadBuffer(GL_COLOR_ATTACHMENT0);
        bind_framebuffer(GL_DRAW_FRAMEBUFFER, 0);
        // the image is stored top row first, the window wants the bottom row first
        const GLint sw = static_cast<GLint>(t->desc.width), sh = static_cast<GLint>(t->desc.height);
        gl_.glBlitFramebuffer(0, 0, sw, sh, 0, static_cast<GLint>(dh), static_cast<GLint>(dw), 0, GL_COLOR_BUFFER_BIT,
                              sw == static_cast<GLint>(dw) && sh == static_cast<GLint>(dh) ? GL_NEAREST : GL_LINEAR);
        gl_.glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        surface.swap();
    }

    // ---------------------------------------------------------------- replay

    namespace {
        class replayer {
        public:
            explicit replayer(device& d) : dev_(d), gl_(d.gl()), cache_(d.cache()) {}

            void run(const std::vector<uint8_t>& stream) {
                size_t at = 0;
                while (at + 8 <= stream.size()) {
                    uint16_t code;
                    uint32_t extra;
                    std::memcpy(&code, stream.data() + at, 2);
                    std::memcpy(&extra, stream.data() + at + 4, 4);
                    at += 8;
                    const uint8_t* payload = stream.data() + at;
                    size_t payload_bytes = 0;
                    dispatch(code, payload, extra, payload_bytes);
                    at += payload_bytes + extra;
                }
                if (in_pass_)
                    end_pass();
            }

        private:
            template <class T> static T read(const uint8_t* p) {
                T v;
                std::memcpy(&v, p, sizeof(T));
                return v;
            }

            template <class T> T take(const uint8_t* p, size_t& bytes) {
                bytes = sizeof(T);
                return read<T>(p);
            }

            void dispatch(uint16_t code, const uint8_t* p, uint32_t extra, size_t& bytes) {
                switch (code) {
                case op_memory_barrier: {
                    const auto a = take<p_barrier>(p, bytes);
                    if (gl_.glMemoryBarrier)
                        gl_.glMemoryBarrier(a.bits);
                    break;
                }
                case op_copy_buffer:
                    copy_buffer(take<p_copy_buffer>(p, bytes));
                    break;
                case op_buffer_to_texture: {
                    const auto a = take<p_buffer_texture>(p, bytes);
                    buffer_to_texture(a, regions_at<buffer_texture_copy>(p + bytes, a.count));
                    break;
                }
                case op_texture_to_buffer: {
                    const auto a = take<p_buffer_texture>(p, bytes);
                    texture_to_buffer(a, regions_at<buffer_texture_copy>(p + bytes, a.count));
                    break;
                }
                case op_copy_texture: {
                    const auto a = take<p_copy_texture>(p, bytes);
                    copy_texture(a, regions_at<texture_copy>(p + bytes, a.count));
                    break;
                }
                case op_fill_buffer:
                    fill_buffer(take<p_fill>(p, bytes));
                    break;
                case op_resolve_texture: {
                    const auto a = take<p_resolve>(p, bytes);
                    resolve(a.dst, a.src);
                    break;
                }
                case op_begin_rendering:
                    begin_pass(take<p_begin>(p, bytes).desc);
                    break;
                case op_end_rendering:
                    bytes = 1;
                    end_pass();
                    break;
                case op_set_pipeline:
                    set_pipeline(take<p_pipeline>(p, bytes).pipeline);
                    break;
                case op_set_bind_group: {
                    const auto a = take<p_group>(p, bytes);
                    set_group(a, p + bytes);
                    break;
                }
                case op_push_constants: {
                    const auto a = take<p_push>(p, bytes);
                    if (a.offset + a.bytes <= push_.size()) {
                        std::memcpy(push_.data() + a.offset, p + bytes, a.bytes);
                        push_dirty_ = true;
                    }
                    break;
                }
                case op_viewport:
                    set_viewport(take<viewport>(p, bytes));
                    break;
                case op_scissor:
                    set_scissor(take<rect>(p, bytes));
                    break;
                case op_vertex_buffer: {
                    const auto a = take<p_vertex>(p, bytes);
                    if (a.slot < vertex_.size()) {
                        vertex_[a.slot] = {a.buffer, a.offset};
                        attribs_dirty_ = true;
                    }
                    break;
                }
                case op_index_buffer: {
                    const auto a = take<p_index>(p, bytes);
                    index_ = a.buffer;
                    index_type_ = a.type;
                    index_offset_ = a.offset;
                    index_dirty_ = true;
                    break;
                }
                case op_stencil_reference:
                    stencil_ref_ = take<p_u32>(p, bytes).value;
                    stencil_dirty_ = true;
                    break;
                case op_draw:
                    draw(take<p_draw>(p, bytes));
                    break;
                case op_draw_indexed:
                    draw_indexed(take<p_draw_indexed>(p, bytes));
                    break;
                case op_draw_indexed_indirect:
                    draw_indirect(take<p_indirect>(p, bytes));
                    break;
                case op_draw_indexed_indirect_count:
                    bytes = sizeof(p_indirect);
                    dev_.report_once(
                        "draw_indexed_indirect_count is not available in OpenGL ES: the draws are skipped");
                    break;
                case op_dispatch: {
                    const auto a = take<p_dispatch>(p, bytes);
                    if (prepare_dispatch())
                        gl_.glDispatchCompute(a.x, a.y, a.z);
                    break;
                }
                case op_dispatch_indirect: {
                    const auto a = take<p_dispatch_indirect>(p, bytes);
                    const buffer* b = dev_.buffer_of(a.args);
                    if (b && b->gl && prepare_dispatch() && gl_.glDispatchComputeIndirect) {
                        gl_.glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, b->gl);
                        gl_.glDispatchComputeIndirect(static_cast<GLintptr>(a.offset));
                    }
                    break;
                }
                case op_resolve_queries:
                    resolve_queries(take<p_queries>(p, bytes));
                    break;
                case op_debug_begin: {
                    const auto a = take<p_u32>(p, bytes);
                    if (dev_.features().debug_groups) {
                        const std::string name(reinterpret_cast<const char*>(p + bytes), a.value);
                        gl_.glPushDebugGroup(0x824A, 0, static_cast<GLsizei>(name.size()), name.c_str());
                    }
                    break;
                }
                case op_debug_end:
                    bytes = 1;
                    if (dev_.features().debug_groups)
                        gl_.glPopDebugGroup();
                    break;
                case op_invalidate:
                    bytes = 1;
                    dev_.resync();
                    pipeline_dirty_ = true;
                    attribs_dirty_ = true;
                    index_dirty_ = true;
                    sets_dirty_ = 0xF;
                    samplers_dirty_ = true;
                    if (in_pass_)
                        rebind_pass();
                    break;
                default:
                    (void)extra;
                    dev_.report_error("corrupt command stream");
                    bytes = 0;
                    break;
                }
            }

            template <class T> static std::span<const T> regions_at(const uint8_t* p, uint32_t count) {
                // the stream is byte packed: copy so the elements are aligned
                thread_local std::vector<T> scratch;
                scratch.resize(count);
                if (count)
                    std::memcpy(scratch.data(), p, sizeof(T) * count);
                return {scratch.data(), count};
            }

            // ------------------------------------------------ buffers

            void copy_buffer(const p_copy_buffer& a) {
                buffer* d = dev_.buffer_of(a.dst);
                buffer* s = dev_.buffer_of(a.src);
                if (!d || !s)
                    return;
                const uint64_t size = std::min(
                    {a.size, d->size - std::min(a.dst_offset, d->size), s->size - std::min(a.src_offset, s->size)});
                if (!size)
                    return;
                if (!s->shadow.empty() && s->desc.memory != memory_class::readback) {
                    // the shadow is what the application wrote: it needs no flush to be copied
                    if (d->gl) {
                        gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, d->gl);
                        gl_.glBufferSubData(GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(a.dst_offset),
                                            static_cast<GLsizeiptr>(size), s->shadow.data() + a.src_offset);
                    }
                    if (!d->shadow.empty())
                        std::memcpy(d->shadow.data() + a.dst_offset, s->shadow.data() + a.src_offset, size);
                    return;
                }
                if (d->gl && s->gl) {
                    gl_.glBindBuffer(GL_COPY_READ_BUFFER, s->gl);
                    gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, d->gl);
                    gl_.glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                                            static_cast<GLintptr>(a.src_offset), static_cast<GLintptr>(a.dst_offset),
                                            static_cast<GLsizeiptr>(size));
                    return;
                }
                if (!d->gl && !d->shadow.empty() && s->gl) {
                    // gpu to readback memory
                    if (!dev_.features().map_range || !gl_.glMapBufferRange) {
                        dev_.report_once("copying a GPU buffer into a readback buffer needs buffer mapping");
                        return;
                    }
                    gl_.glBindBuffer(GL_COPY_READ_BUFFER, s->gl);
                    void* mapped = gl_.glMapBufferRange(GL_COPY_READ_BUFFER, static_cast<GLintptr>(a.src_offset),
                                                        static_cast<GLsizeiptr>(size), GL_MAP_READ_BIT);
                    if (mapped) {
                        std::memcpy(d->shadow.data() + a.dst_offset, mapped, size);
                        gl_.glUnmapBuffer(GL_COPY_READ_BUFFER);
                    }
                }
            }

            void fill_buffer(const p_fill& a) {
                buffer* b = dev_.buffer_of(a.dst);
                if (!b)
                    return;
                const uint64_t offset = std::min(a.offset, b->size);
                const uint64_t size = std::min(a.size ? a.size : b->size, b->size - offset) & ~3ull;
                if (!size)
                    return;
                std::vector<uint32_t> words(size / 4, a.value);
                if (b->gl) {
                    gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, b->gl);
                    gl_.glBufferSubData(GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(offset),
                                        static_cast<GLsizeiptr>(size), words.data());
                }
                if (!b->shadow.empty())
                    std::memcpy(b->shadow.data() + offset, words.data(), size);
            }

            void resolve_queries(const p_queries& a) {
                buffer* b = dev_.buffer_of(a.dst);
                if (!b)
                    return;
                const uint64_t size =
                    std::min<uint64_t>(static_cast<uint64_t>(a.count) * 8, b->size - std::min(a.offset, b->size));
                if (!size)
                    return;
                std::vector<uint8_t> zeros(size, 0);
                if (b->gl) {
                    gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, b->gl);
                    gl_.glBufferSubData(GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(a.offset),
                                        static_cast<GLsizeiptr>(size), zeros.data());
                }
                if (!b->shadow.empty())
                    std::memset(b->shadow.data() + a.offset, 0, size);
            }

            // ------------------------------------------------ textures

            void set_unpack_row(int pixels) {
                if (cache_.unpack_row == pixels)
                    return;
                cache_.unpack_row = pixels;
                gl_.glPixelStorei(GL_UNPACK_ROW_LENGTH, pixels);
            }

            void set_pack_row(int pixels) {
                if (cache_.pack_row == pixels)
                    return;
                cache_.pack_row = pixels;
                gl_.glPixelStorei(GL_PACK_ROW_LENGTH, pixels);
            }

            void buffer_to_texture(const p_buffer_texture& a, std::span<const buffer_texture_copy> regions) {
                texture* t = dev_.texture_of(a.texture);
                buffer* b = dev_.buffer_of(a.buffer);
                if (!t || !b || t->renderbuffer)
                    return;
                const format_gl& fg = t->fmt;
                const format_info& info = get_format_info(t->desc.format);
                dev_.use_texture(dev_.scratch_unit(), t->target, t->gl);
                for (const buffer_texture_copy& r : regions) {
                    const uint32_t w = r.width ? r.width : mip_extent(t->desc.width, r.mip) - r.x;
                    const uint32_t h = r.height ? r.height : mip_extent(t->desc.height, r.mip) - r.y;
                    const uint32_t depth = std::max(r.depth, 1u);
                    const uint32_t tight = row_bytes(t->desc.format, w);
                    const uint32_t pitch = r.buffer_row_pitch ? r.buffer_row_pitch : tight;
                    const uint32_t rows = row_blocks(t->desc.format, h);
                    const uint64_t needed = static_cast<uint64_t>(pitch) * rows * depth;
                    const uint8_t* source = nullptr;
                    std::vector<uint8_t> repacked;
                    bool from_buffer_object = false;
                    if (!b->shadow.empty() && b->desc.memory != memory_class::readback) {
                        if (r.buffer_offset + needed > b->shadow.size() + (pitch - tight)) {
                            dev_.report_error("copy_buffer_to_texture reads past the end of the buffer");
                            continue;
                        }
                        source = b->shadow.data() + r.buffer_offset;
                    } else if (b->gl) {
                        from_buffer_object = true;
                    } else {
                        continue;
                    }

                    // tight rows, byte order fixed up, whenever the data cannot go straight from where it is
                    const bool pitch_ok = fg.compressed ? pitch == tight : (pitch % info.bytes_per_block == 0);
                    if (source && (!pitch_ok || fg.swap_rb)) {
                        repacked.resize(static_cast<size_t>(tight) * rows * depth);
                        for (uint32_t z = 0; z < depth; z++)
                            for (uint32_t y = 0; y < rows; y++) {
                                const uint8_t* from = source + (static_cast<size_t>(z) * rows + y) * pitch;
                                uint8_t* to = repacked.data() + (static_cast<size_t>(z) * rows + y) * tight;
                                std::memcpy(to, from, tight);
                                if (fg.swap_rb)
                                    for (uint32_t x = 0; x + 3 < tight; x += 4)
                                        std::swap(to[x], to[x + 2]);
                            }
                        source = repacked.data();
                        set_unpack_row(0);
                    } else if (!fg.compressed) {
                        set_unpack_row(pitch == tight ? 0 : static_cast<int>(pitch / info.bytes_per_block));
                    } else {
                        set_unpack_row(0);
                    }

                    const void* data = source;
                    if (from_buffer_object) {
                        gl_.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, b->gl);
                        data = gl_offset(r.buffer_offset);
                    }
                    upload(*t, r, w, h, depth, data, static_cast<GLsizei>(static_cast<size_t>(tight) * rows));
                    if (from_buffer_object)
                        gl_.glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                }
                set_unpack_row(0);
            }

            void upload(const texture& t, const buffer_texture_copy& r, uint32_t w, uint32_t h, uint32_t depth,
                        const void* data, GLsizei slice_bytes) {
                const format_gl& fg = t.fmt;
                const GLint mip = static_cast<GLint>(r.mip);
                switch (t.target) {
                case GL_TEXTURE_2D:
                case GL_TEXTURE_CUBE_MAP: {
                    const GLenum target =
                        t.target == GL_TEXTURE_CUBE_MAP ? GL_TEXTURE_CUBE_MAP_POSITIVE_X + r.layer : GL_TEXTURE_2D;
                    if (fg.compressed)
                        gl_.glCompressedTexSubImage2D(target, mip, static_cast<GLint>(r.x), static_cast<GLint>(r.y),
                                                      static_cast<GLsizei>(w), static_cast<GLsizei>(h), fg.internal,
                                                      slice_bytes, data);
                    else
                        gl_.glTexSubImage2D(target, mip, static_cast<GLint>(r.x), static_cast<GLint>(r.y),
                                            static_cast<GLsizei>(w), static_cast<GLsizei>(h), fg.format, fg.type, data);
                    break;
                }
                default: {
                    const GLint z = t.target == GL_TEXTURE_3D ? static_cast<GLint>(r.z) : static_cast<GLint>(r.layer);
                    if (fg.compressed)
                        gl_.glCompressedTexSubImage3D(t.target, mip, static_cast<GLint>(r.x), static_cast<GLint>(r.y),
                                                      z, static_cast<GLsizei>(w), static_cast<GLsizei>(h),
                                                      static_cast<GLsizei>(depth), fg.internal,
                                                      slice_bytes * static_cast<GLsizei>(depth), data);
                    else
                        gl_.glTexSubImage3D(t.target, mip, static_cast<GLint>(r.x), static_cast<GLint>(r.y), z,
                                            static_cast<GLsizei>(w), static_cast<GLsizei>(h),
                                            static_cast<GLsizei>(depth), fg.format, fg.type, data);
                    break;
                }
                }
            }

            // attaches one subresource of a texture to a scratch framebuffer for reading or writing
            void attach_scratch(bool read, const texture& t, uint32_t mip, uint32_t layer) {
                const GLenum target = read ? GL_READ_FRAMEBUFFER : GL_DRAW_FRAMEBUFFER;
                dev_.bind_framebuffer(target, dev_.scratch_framebuffer(read));
                const bool depth = is_depth_format(t.desc.format);
                const GLenum point =
                    depth ? (has_stencil(t.desc.format) ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT)
                          : GL_COLOR_ATTACHMENT0;
                // clear the attachment points that could be left from an earlier use
                for (const GLenum other : {GL_COLOR_ATTACHMENT0, GL_DEPTH_STENCIL_ATTACHMENT})
                    if (other != point)
                        gl_.glFramebufferTexture2D(target, other, GL_TEXTURE_2D, 0, 0);
                if (t.renderbuffer)
                    gl_.glFramebufferRenderbuffer(target, point, GL_RENDERBUFFER, t.gl);
                else if (t.target == GL_TEXTURE_2D)
                    gl_.glFramebufferTexture2D(target, point, GL_TEXTURE_2D, t.gl, static_cast<GLint>(mip));
                else if (t.target == GL_TEXTURE_CUBE_MAP)
                    gl_.glFramebufferTexture2D(target, point, GL_TEXTURE_CUBE_MAP_POSITIVE_X + layer, t.gl,
                                               static_cast<GLint>(mip));
                else
                    gl_.glFramebufferTextureLayer(target, point, t.gl, static_cast<GLint>(mip),
                                                  static_cast<GLint>(layer));
                if (read) {
                    gl_.glReadBuffer(depth ? GL_NONE : GL_COLOR_ATTACHMENT0);
                } else {
                    const GLenum buffers = depth ? GL_NONE : GL_COLOR_ATTACHMENT0;
                    gl_.glDrawBuffers(1, &buffers);
                }
            }

            void detach_scratch() {
                for (const bool read : {true, false}) {
                    const GLenum target = read ? GL_READ_FRAMEBUFFER : GL_DRAW_FRAMEBUFFER;
                    dev_.bind_framebuffer(target, dev_.scratch_framebuffer(read));
                    gl_.glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
                    gl_.glFramebufferTexture2D(target, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
                }
            }

            void copy_texture(const p_copy_texture& a, std::span<const texture_copy> regions) {
                texture* d = dev_.texture_of(a.dst);
                texture* s = dev_.texture_of(a.src);
                if (!d || !s)
                    return;
                if (d->fmt.compressed || s->fmt.compressed) {
                    dev_.report_once("copy_texture between compressed textures is not available in OpenGL ES");
                    return;
                }
                const bool depth = is_depth_format(s->desc.format);
                for (const texture_copy& r : regions) {
                    const uint32_t w = r.width ? r.width : mip_extent(s->desc.width, r.src_mip) - r.src_x;
                    const uint32_t h = r.height ? r.height : mip_extent(s->desc.height, r.src_mip) - r.src_y;
                    if (dev_.features().copy_image && !s->renderbuffer && !d->renderbuffer) {
                        const auto z_of = [](const texture& t, uint32_t layer) {
                            return t.target == GL_TEXTURE_2D ? 0 : static_cast<GLint>(layer);
                        };
                        gl_.glCopyImageSubData(
                            s->gl, s->target, static_cast<GLint>(r.src_mip), static_cast<GLint>(r.src_x),
                            static_cast<GLint>(r.src_y), z_of(*s, r.src_layer), d->gl, d->target,
                            static_cast<GLint>(r.dst_mip), static_cast<GLint>(r.dst_x), static_cast<GLint>(r.dst_y),
                            z_of(*d, r.dst_layer), static_cast<GLsizei>(w), static_cast<GLsizei>(h), 1);
                        continue;
                    }
                    attach_scratch(true, *s, r.src_mip, r.src_layer);
                    attach_scratch(false, *d, r.dst_mip, r.dst_layer);
                    dev_.use_scissor_test(false);
                    gl_.glBlitFramebuffer(
                        static_cast<GLint>(r.src_x), static_cast<GLint>(r.src_y), static_cast<GLint>(r.src_x + w),
                        static_cast<GLint>(r.src_y + h), static_cast<GLint>(r.dst_x), static_cast<GLint>(r.dst_y),
                        static_cast<GLint>(r.dst_x + w), static_cast<GLint>(r.dst_y + h),
                        depth ? (GL_DEPTH_BUFFER_BIT | (has_stencil(s->desc.format) ? GL_STENCIL_BUFFER_BIT : 0))
                              : GL_COLOR_BUFFER_BIT,
                        GL_NEAREST);
                }
                detach_scratch();
            }

            void resolve(texture_handle dst, texture_handle src) {
                texture* d = dev_.texture_of(dst);
                texture* s = dev_.texture_of(src);
                if (!d || !s)
                    return;
                resolve_into(*d, *s, 0, 0);
            }

            void resolve_into(const texture& d, const texture& s, uint32_t dst_mip, uint32_t dst_layer) {
                const bool depth = is_depth_format(s.desc.format);
                attach_scratch(true, s, 0, 0);
                attach_scratch(false, d, dst_mip, dst_layer);
                dev_.use_scissor_test(false);
                const GLint w = static_cast<GLint>(s.desc.width), h = static_cast<GLint>(s.desc.height);
                gl_.glBlitFramebuffer(
                    0, 0, w, h, 0, 0, w, h,
                    depth ? (GL_DEPTH_BUFFER_BIT | (has_stencil(s.desc.format) ? GL_STENCIL_BUFFER_BIT : 0))
                          : GL_COLOR_BUFFER_BIT,
                    GL_NEAREST);
                detach_scratch();
            }

            void texture_to_buffer(const p_buffer_texture& a, std::span<const buffer_texture_copy> regions) {
                texture* t = dev_.texture_of(a.texture);
                buffer* b = dev_.buffer_of(a.buffer);
                if (!t || !b || b->shadow.empty())
                    return;
                if (t->renderbuffer || t->fmt.compressed || is_depth_format(t->desc.format)) {
                    dev_.report_once(
                        "reading back multisampled, compressed or depth textures is not available in OpenGL ES");
                    return;
                }
                const format_info& info = get_format_info(t->desc.format);
                const uint32_t channels = std::max<uint32_t>(info.channels, 1);
                enum class family { bytes, floats, ints, none } fam = family::none;
                switch (t->desc.format) {
                case format::r8_unorm:
                case format::rg8_unorm:
                case format::rgba8_unorm:
                case format::rgba8_srgb:
                case format::bgra8_unorm:
                case format::bgra8_srgb:
                    fam = family::bytes;
                    break;
                case format::r16_float:
                case format::rg16_float:
                case format::rgba16_float:
                case format::r32_float:
                case format::rg32_float:
                case format::rgba32_float:
                    fam = family::floats;
                    break;
                case format::r8_uint:
                case format::r16_uint:
                case format::r32_uint:
                case format::rg32_uint:
                case format::rgba32_uint:
                    fam = family::ints;
                    break;
                default:
                    break;
                }
                if (fam == family::none) {
                    dev_.report_once("this texture format cannot be read back in OpenGL ES");
                    return;
                }
                const uint32_t component_bytes = info.bytes_per_block / channels;
                for (const buffer_texture_copy& r : regions) {
                    const uint32_t w = r.width ? r.width : mip_extent(t->desc.width, r.mip) - r.x;
                    const uint32_t h = r.height ? r.height : mip_extent(t->desc.height, r.mip) - r.y;
                    const uint32_t pitch = r.buffer_row_pitch ? r.buffer_row_pitch : w * info.bytes_per_block;
                    attach_scratch(true, *t, r.mip, r.layer);
                    set_pack_row(0);
                    std::vector<uint32_t> wide(static_cast<size_t>(w) * h * 4);
                    if (fam == family::bytes)
                        gl_.glReadPixels(static_cast<GLint>(r.x), static_cast<GLint>(r.y), static_cast<GLsizei>(w),
                                         static_cast<GLsizei>(h), GL_RGBA, GL_UNSIGNED_BYTE, wide.data());
                    else if (fam == family::floats)
                        gl_.glReadPixels(static_cast<GLint>(r.x), static_cast<GLint>(r.y), static_cast<GLsizei>(w),
                                         static_cast<GLsizei>(h), GL_RGBA, GL_FLOAT, wide.data());
                    else
                        gl_.glReadPixels(static_cast<GLint>(r.x), static_cast<GLint>(r.y), static_cast<GLsizei>(w),
                                         static_cast<GLsizei>(h), GL_RGBA_INTEGER, GL_UNSIGNED_INT, wide.data());
                    for (uint32_t y = 0; y < h; y++) {
                        const uint64_t at = r.buffer_offset + static_cast<uint64_t>(y) * pitch;
                        if (at + static_cast<uint64_t>(w) * info.bytes_per_block > b->shadow.size())
                            break;
                        uint8_t* row = b->shadow.data() + at;
                        for (uint32_t x = 0; x < w; x++) {
                            uint8_t* px = row + static_cast<size_t>(x) * info.bytes_per_block;
                            if (fam == family::bytes) {
                                const uint8_t* src = reinterpret_cast<const uint8_t*>(wide.data()) +
                                                     (static_cast<size_t>(y) * w + x) * 4;
                                uint8_t rgba[4] = {src[0], src[1], src[2], src[3]};
                                if (t->fmt.swap_rb)
                                    std::swap(rgba[0], rgba[2]);
                                std::memcpy(px, rgba, channels);
                                continue;
                            }
                            const uint32_t* src = wide.data() + (static_cast<size_t>(y) * w + x) * 4;
                            for (uint32_t c = 0; c < channels; c++) {
                                if (fam == family::floats) {
                                    float f;
                                    std::memcpy(&f, &src[c], 4);
                                    if (component_bytes == 2) {
                                        const uint16_t half = float_to_half(f);
                                        std::memcpy(px + c * 2, &half, 2);
                                    } else {
                                        std::memcpy(px + c * 4, &f, 4);
                                    }
                                } else if (component_bytes == 1) {
                                    px[c] = static_cast<uint8_t>(src[c]);
                                } else if (component_bytes == 2) {
                                    const uint16_t v = static_cast<uint16_t>(src[c]);
                                    std::memcpy(px + c * 2, &v, 2);
                                } else {
                                    std::memcpy(px + c * 4, &src[c], 4);
                                }
                            }
                        }
                    }
                }
                detach_scratch();
            }

            // ------------------------------------------------ render passes

            static uint64_t attachment_key(const texture& t, uint32_t mip, uint32_t layer) {
                return static_cast<uint64_t>(t.gl) | (t.renderbuffer ? 1ull << 32 : 0) |
                       (static_cast<uint64_t>(mip & 0xFF) << 33) | (static_cast<uint64_t>(layer & 0xFFFF) << 41) |
                       (1ull << 57);
            }

            void begin_pass(const rendering_desc& desc) {
                if (in_pass_)
                    end_pass();
                desc_ = desc;
                in_pass_ = true;
                std::array<uint64_t, max_color_targets + 1> key{};
                pass_width_ = pass_height_ = 0;
                const uint32_t count = std::min(desc.color_count, max_color_targets);
                desc_.color_count = count;
                for (uint32_t i = 0; i < count; i++) {
                    const texture* t = dev_.texture_of(desc.colors[i].texture);
                    if (!t) {
                        dev_.report_error("begin_rendering: a color attachment is gone");
                        in_pass_ = false;
                        return;
                    }
                    key[i] = attachment_key(*t, desc.colors[i].mip, desc.colors[i].layer);
                    if (!pass_width_) {
                        pass_width_ = mip_extent(t->desc.width, desc.colors[i].mip);
                        pass_height_ = mip_extent(t->desc.height, desc.colors[i].mip);
                    }
                }
                if (const texture* d = dev_.texture_of(desc.depth.texture)) {
                    key[max_color_targets] = attachment_key(*d, desc.depth.mip, desc.depth.layer);
                    if (!pass_width_) {
                        pass_width_ = mip_extent(d->desc.width, desc.depth.mip);
                        pass_height_ = mip_extent(d->desc.height, desc.depth.mip);
                    }
                }
                if (!pass_width_) {
                    dev_.report_error("begin_rendering without attachments");
                    in_pass_ = false;
                    return;
                }
                pass_key_ = key;
                const framebuffer* fb = dev_.framebuffer_for(key, desc_, false);
                pass_fbo_ = fb->id;
                dev_.bind_framebuffer(GL_FRAMEBUFFER, pass_fbo_);
                cache_.read_only_depth = desc.depth.read_only;
                read_only_depth_ = desc.depth.read_only;
                pipeline_dirty_ = true;

                clear_and_discard_at_start();

                viewport_ = {0, 0, static_cast<float>(pass_width_), static_cast<float>(pass_height_), 0.0f, 1.0f};
                scissor_ = {0, 0, pass_width_, pass_height_};
                apply_viewport();
                apply_scissor();
            }

            void clear_and_discard_at_start() {
                bool clearing = false;
                GLenum discard[max_color_targets + 1];
                GLsizei discards = 0;
                for (uint32_t i = 0; i < desc_.color_count; i++) {
                    clearing = clearing || desc_.colors[i].load == load_op::clear;
                    if (desc_.colors[i].load == load_op::dont_care)
                        discard[discards++] = GL_COLOR_ATTACHMENT0 + i;
                }
                const texture* depth = dev_.texture_of(desc_.depth.texture);
                const bool depth_clear = depth && desc_.depth.load == load_op::clear;
                clearing = clearing || depth_clear;
                if (depth && desc_.depth.load == load_op::dont_care)
                    discard[discards++] =
                        has_stencil(depth->desc.format) ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;
                if (discards)
                    gl_.glInvalidateFramebuffer(GL_FRAMEBUFFER, discards, discard);
                if (!clearing)
                    return;

                // clears honour the scissor test and the write masks: open both up, and make the next draw set its own
                dev_.use_scissor_test(false);
                gl_.glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
                gl_.glDepthMask(GL_TRUE);
                gl_.glStencilMaskSeparate(GL_FRONT_AND_BACK, 0xFF);
                cache_.state_valid = false;
                for (uint32_t i = 0; i < desc_.color_count; i++) {
                    const color_attachment& c = desc_.colors[i];
                    if (c.load != load_op::clear)
                        continue;
                    const texture* t = dev_.texture_of(c.texture);
                    if (t && t->fmt.integer) {
                        const GLuint v[4] = {static_cast<GLuint>(std::max(c.clear[0], 0.0f)),
                                             static_cast<GLuint>(std::max(c.clear[1], 0.0f)),
                                             static_cast<GLuint>(std::max(c.clear[2], 0.0f)),
                                             static_cast<GLuint>(std::max(c.clear[3], 0.0f))};
                        gl_.glClearBufferuiv(GL_COLOR, static_cast<GLint>(i), v);
                    } else {
                        gl_.glClearBufferfv(GL_COLOR, static_cast<GLint>(i), c.clear.data());
                    }
                }
                if (depth_clear) {
                    if (has_stencil(depth->desc.format))
                        gl_.glClearBufferfi(GL_DEPTH_STENCIL, 0, desc_.depth.clear_depth, desc_.depth.clear_stencil);
                    else
                        gl_.glClearBufferfv(GL_DEPTH, 0, &desc_.depth.clear_depth);
                }
            }

            void rebind_pass() {
                const framebuffer* fb = dev_.framebuffer_for(pass_key_, desc_, false);
                pass_fbo_ = fb->id;
                dev_.bind_framebuffer(GL_FRAMEBUFFER, pass_fbo_);
                apply_viewport();
                apply_scissor();
            }

            void end_pass() {
                if (!in_pass_)
                    return;
                in_pass_ = false;
                GLenum discard[max_color_targets + 1];
                GLsizei discards = 0;
                for (uint32_t i = 0; i < desc_.color_count; i++) {
                    const color_attachment& c = desc_.colors[i];
                    if (const texture* resolve_to = dev_.texture_of(c.resolve)) {
                        if (const texture* from = dev_.texture_of(c.texture)) {
                            dev_.bind_framebuffer(GL_READ_FRAMEBUFFER, pass_fbo_);
                            gl_.glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
                            attach_scratch_draw_only(*resolve_to);
                            dev_.use_scissor_test(false);
                            const GLint w = static_cast<GLint>(from->desc.width),
                                        h = static_cast<GLint>(from->desc.height);
                            gl_.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
                            gl_.glReadBuffer(GL_COLOR_ATTACHMENT0);
                        }
                    }
                    if (c.store == store_op::dont_care)
                        discard[discards++] = GL_COLOR_ATTACHMENT0 + i;
                }
                if (const texture* depth = dev_.texture_of(desc_.depth.texture))
                    if (desc_.depth.store == store_op::dont_care || desc_.depth.read_only)
                        discard[discards++] =
                            has_stencil(depth->desc.format) ? GL_DEPTH_STENCIL_ATTACHMENT : GL_DEPTH_ATTACHMENT;
                if (discards) {
                    dev_.bind_framebuffer(GL_FRAMEBUFFER, pass_fbo_);
                    gl_.glInvalidateFramebuffer(GL_FRAMEBUFFER, discards, discard);
                }
                cache_.read_only_depth = false;
                read_only_depth_ = false;
            }

            void attach_scratch_draw_only(const texture& t) {
                dev_.bind_framebuffer(GL_DRAW_FRAMEBUFFER, dev_.scratch_framebuffer(false));
                if (t.renderbuffer)
                    gl_.glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, t.gl);
                else if (t.target == GL_TEXTURE_2D)
                    gl_.glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t.gl, 0);
                else
                    gl_.glFramebufferTextureLayer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, t.gl, 0, 0);
                const GLenum buffer = GL_COLOR_ATTACHMENT0;
                gl_.glDrawBuffers(1, &buffer);
            }

            void set_viewport(const viewport& v) {
                viewport_ = v;
                if (in_pass_)
                    apply_viewport();
            }

            void apply_viewport() {
                // the pass renders upside down, so window rows are image rows and the viewport maps straight across
                const float h = std::fabs(viewport_.height);
                const float y = viewport_.height < 0 ? viewport_.y + viewport_.height : viewport_.y;
                gl_.glViewport(static_cast<GLint>(viewport_.x), static_cast<GLint>(y),
                               static_cast<GLsizei>(viewport_.width), static_cast<GLsizei>(h));
                if (cache_.depth_near != viewport_.min_depth || cache_.depth_far != viewport_.max_depth) {
                    cache_.depth_near = viewport_.min_depth;
                    cache_.depth_far = viewport_.max_depth;
                    gl_.glDepthRangef(viewport_.min_depth, viewport_.max_depth);
                }
            }

            void set_scissor(const rect& r) {
                scissor_ = r;
                if (in_pass_)
                    apply_scissor();
            }

            void apply_scissor() {
                dev_.use_scissor_test(true);
                gl_.glScissor(scissor_.x, scissor_.y, static_cast<GLsizei>(scissor_.width),
                              static_cast<GLsizei>(scissor_.height));
            }

            // ------------------------------------------------ pipeline state

            void set_pipeline(pipeline_handle handle) {
                const pipeline* p = dev_.pipeline_of(handle);
                if (!p) {
                    dev_.report_error("set_pipeline: the pipeline is gone");
                    pipeline_ = nullptr;
                    return;
                }
                if (p == pipeline_)
                    return;
                const pipeline* previous = pipeline_;
                pipeline_ = p;
                pipeline_dirty_ = true;
                attribs_dirty_ = true;
                samplers_dirty_ = true;
                if (!previous || !(previous->layout == p->layout)) {
                    sets_dirty_ = 0xF;
                    push_dirty_ = true;
                }
            }

            void apply_state(const pipeline_state& n) {
                gl_cache& c = cache_;
                pipeline_state e = n;
                if (read_only_depth_) {
                    e.depth_write = false;
                    e.stencil_write = 0;
                }
                const bool all = !c.state_valid;
                const pipeline_state& o = c.state;
                if (all || o.cull_face != e.cull_face) {
                    if (!e.cull_face) {
                        gl_.glDisable(GL_CULL_FACE);
                    } else {
                        if (all || !o.cull_face)
                            gl_.glEnable(GL_CULL_FACE);
                        gl_.glCullFace(e.cull_face);
                    }
                }
                if (all || o.front_face != e.front_face)
                    gl_.glFrontFace(e.front_face);
                if (all || o.offset != e.offset) {
                    if (e.offset)
                        gl_.glEnable(GL_POLYGON_OFFSET_FILL);
                    else
                        gl_.glDisable(GL_POLYGON_OFFSET_FILL);
                }
                if (e.offset && (all || o.offset_slope != e.offset_slope || o.offset_units != e.offset_units))
                    gl_.glPolygonOffset(e.offset_slope, e.offset_units);
                if (all || o.depth_test != e.depth_test) {
                    if (e.depth_test)
                        gl_.glEnable(GL_DEPTH_TEST);
                    else
                        gl_.glDisable(GL_DEPTH_TEST);
                }
                if (all || o.depth_func != e.depth_func)
                    gl_.glDepthFunc(e.depth_func);
                if (all || o.depth_write != e.depth_write)
                    gl_.glDepthMask(e.depth_write ? GL_TRUE : GL_FALSE);
                if (all || o.stencil_test != e.stencil_test) {
                    if (e.stencil_test)
                        gl_.glEnable(GL_STENCIL_TEST);
                    else
                        gl_.glDisable(GL_STENCIL_TEST);
                }
                if (all || !(o.front == e.front) || !(o.back == e.back) || o.stencil_read != e.stencil_read)
                    stencil_dirty_ = true;
                if (all || !(o.front == e.front)) {
                    gl_.glStencilOpSeparate(GL_FRONT, e.front.fail, e.front.depth_fail, e.front.pass);
                }
                if (all || !(o.back == e.back)) {
                    gl_.glStencilOpSeparate(GL_BACK, e.back.fail, e.back.depth_fail, e.back.pass);
                }
                if (all || o.stencil_write != e.stencil_write)
                    gl_.glStencilMaskSeparate(GL_FRONT_AND_BACK, e.stencil_write);
                if (all || o.alpha_to_coverage != e.alpha_to_coverage) {
                    if (e.alpha_to_coverage)
                        gl_.glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
                    else
                        gl_.glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
                }
                apply_blend(all, o, e);
                c.state = e;
                c.state_valid = true;
            }

            void apply_blend(bool all, const pipeline_state& o, const pipeline_state& e) {
                if (!e.blend_count)
                    return;
                if (dev_.features().indexed_blend) {
                    for (uint32_t i = 0; i < e.blend_count; i++) {
                        const blend_gl& n = e.blend[i];
                        if (!all && i < o.blend_count && o.blend[i] == n)
                            continue;
                        if (n.enable)
                            gl_.glEnablei(GL_BLEND, i);
                        else
                            gl_.glDisablei(GL_BLEND, i);
                        gl_.glBlendFuncSeparatei(i, n.src_rgb, n.dst_rgb, n.src_alpha, n.dst_alpha);
                        gl_.glBlendEquationSeparatei(i, n.op_rgb, n.op_alpha);
                        gl_.glColorMaski(i, n.mask & 1, (n.mask >> 1) & 1, (n.mask >> 2) & 1, (n.mask >> 3) & 1);
                    }
                    return;
                }
                const blend_gl& n = e.blend[0];
                if (!all && o.blend_count && o.blend[0] == n)
                    return;
                if (n.enable)
                    gl_.glEnable(GL_BLEND);
                else
                    gl_.glDisable(GL_BLEND);
                gl_.glBlendFuncSeparate(n.src_rgb, n.dst_rgb, n.src_alpha, n.dst_alpha);
                gl_.glBlendEquationSeparate(n.op_rgb, n.op_alpha);
                gl_.glColorMask(n.mask & 1, (n.mask >> 1) & 1, (n.mask >> 2) & 1, (n.mask >> 3) & 1);
            }

            // ------------------------------------------------ descriptors

            void set_group(const p_group& a, const uint8_t* offsets) {
                if (a.set >= max_bind_groups)
                    return;
                bound& b = sets_[a.set];
                b.group = a.group;
                b.offset_count = std::min<uint32_t>(a.count, max_dynamic_offsets);
                for (uint32_t i = 0; i < b.offset_count; i++)
                    std::memcpy(&b.offsets[i], offsets + i * 4, 4);
                sets_dirty_ |= 1u << a.set;
                samplers_dirty_ = true;
            }

            void bind_buffers(uint32_t set, const pipeline_layout& layout) {
                const bound& b = sets_[set];
                const bind_group* g = dev_.group_of(b.group);
                if (!g)
                    return;
                const std::vector<layout_slot>& slots = layout.sets[set];
                for (uint32_t binding = 0; binding < slots.size(); binding++) {
                    const layout_slot& slot = slots[binding];
                    if (!slot.present)
                        continue;
                    const bind_group_entry* e = g->find(binding);
                    if (!e)
                        continue;
                    switch (slot.type) {
                    case binding_type::uniform_buffer:
                    case binding_type::uniform_buffer_dynamic: {
                        buffer* buf = dev_.buffer_of(e->buffer.buffer);
                        if (!buf || !buf->size)
                            continue;
                        uint64_t offset = e->buffer.offset;
                        if (slot.dynamic_index != ~0u && slot.dynamic_index < b.offset_count)
                            offset += b.offsets[slot.dynamic_index];
                        offset = std::min(offset, buf->size - 16);
                        uint64_t size = e->buffer.size ? e->buffer.size : buf->size - offset;
                        size = std::min(size, buf->size - offset);
                        bind_uniform(slot.index, *buf, offset, size);
                        break;
                    }
                    case binding_type::storage_buffer:
                    case binding_type::storage_buffer_rw: {
                        buffer* buf = dev_.buffer_of(e->buffer.buffer);
                        if (!buf || !buf->gl)
                            continue;
                        const uint64_t offset = std::min(e->buffer.offset, buf->size);
                        const uint64_t size =
                            e->buffer.size ? std::min(e->buffer.size, buf->size - offset) : buf->size - offset;
                        gl_.glBindBufferRange(GL_SHADER_STORAGE_BUFFER, slot.index, buf->gl,
                                              static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size));
                        break;
                    }
                    case binding_type::storage_texture_rw: {
                        texture* t = dev_.texture_of(e->texture.texture);
                        if (!t || t->renderbuffer || !gl_.glBindImageTexture)
                            continue;
                        const bool layered = t->target != GL_TEXTURE_2D;
                        gl_.glBindImageTexture(slot.index, t->gl, static_cast<GLint>(e->texture.range.base_mip),
                                               layered, static_cast<GLint>(e->texture.range.base_layer), GL_READ_WRITE,
                                               t->fmt.internal);
                        break;
                    }
                    default:
                        break;
                    }
                }
            }

            void bind_uniform(uint32_t index, buffer& buf, uint64_t offset, uint64_t size) {
                size = std::max<uint64_t>(size, 16);
                const uint64_t alignment = std::max<uint32_t>(dev_.caps().limits.uniform_buffer_alignment, 1);
                const bool upload_class = !buf.shadow.empty() && buf.desc.memory != memory_class::readback;
                GLuint source = buf.gl;
                GLintptr at = static_cast<GLintptr>(offset);
                if (offset % alignment != 0) {
                    // an offset GL cannot bind at: bind a copy
                    const uint64_t slot = dev_.push_alloc(static_cast<uint32_t>(align_up(size, 16)));
                    gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, dev_.push_ring());
                    if (upload_class) {
                        gl_.glBufferSubData(GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(slot),
                                            static_cast<GLsizeiptr>(size), buf.shadow.data() + offset);
                    } else {
                        gl_.glBindBuffer(GL_COPY_READ_BUFFER, buf.gl);
                        gl_.glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                                                static_cast<GLintptr>(offset), static_cast<GLintptr>(slot),
                                                static_cast<GLsizeiptr>(size));
                    }
                    source = dev_.push_ring();
                    at = static_cast<GLintptr>(slot);
                } else if (upload_class) {
                    // what the application wrote is in the shadow: constant data is small, so it goes up every time
                    gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, buf.gl);
                    gl_.glBufferSubData(GL_COPY_WRITE_BUFFER, at, static_cast<GLsizeiptr>(size),
                                        buf.shadow.data() + offset);
                }
                gl_.glBindBufferRange(GL_UNIFORM_BUFFER, index, source, at, static_cast<GLsizeiptr>(size));
            }

            // the preferred sampler object for a texture: GL leaves filtered reads of depth, integer and wide float
            // textures incomplete, so those get the nearest variant
            GLuint sampler_for(const sampler* s, const texture& t) {
                if (!s)
                    return dev_.default_sampler();
                const format_info& info = get_format_info(t.desc.format);
                const bool wide_float =
                    info.floating && !info.depth && info.bytes_per_block / std::max<uint8_t>(info.channels, 1) == 4;
                const bool unfilterable =
                    t.fmt.integer || (info.depth && !s->desc.compare) || (wide_float && !dev_.features().float_linear);
                return unfilterable ? s->gl_nearest : s->gl;
            }

            void bind_samplers() {
                for (const combined_sampler& c : pipeline_->samplers) {
                    const bind_group* g = dev_.group_of(sets_[c.set].group);
                    const bind_group_entry* te = g ? g->find(c.binding) : nullptr;
                    texture* t = te ? dev_.texture_of(te->texture.texture) : nullptr;
                    if (!t || t->renderbuffer) {
                        dev_.use_texture(c.unit, c.target, 0);
                        continue;
                    }
                    dev_.use_texture(c.unit, t->target, t->gl);
                    const subresource_range& range = te->texture.range;
                    const int base = static_cast<int>(std::min<uint32_t>(range.base_mip, t->levels - 1));
                    const int last = range.mip_count
                                         ? std::min(base + static_cast<int>(range.mip_count) - 1, t->levels - 1)
                                         : t->levels - 1;
                    if (t->base_level != base) {
                        gl_.glTexParameteri(t->target, GL_TEXTURE_BASE_LEVEL, base);
                        t->base_level = base;
                    }
                    if (t->max_level != last) {
                        gl_.glTexParameteri(t->target, GL_TEXTURE_MAX_LEVEL, last);
                        t->max_level = last;
                    }
                    const sampler* s = nullptr;
                    if (!c.dummy_sampler) {
                        const bind_group* sg = dev_.group_of(sets_[c.sampler_set].group);
                        const bind_group_entry* se = sg ? sg->find(c.sampler_binding) : nullptr;
                        s = se ? dev_.sampler_of(se->sampler) : nullptr;
                    }
                    const GLuint object = c.dummy_sampler ? dev_.default_sampler() : sampler_for(s, *t);
                    if (c.unit < cache_.unit_sampler.size() && cache_.unit_sampler[c.unit] != object) {
                        cache_.unit_sampler[c.unit] = object;
                        gl_.glBindSampler(c.unit, object);
                    }
                }
            }

            bool bind_resources() {
                const pipeline_layout* layout = dev_.pipeline_layout_of(pipeline_->layout);
                if (!layout)
                    return false;
                if (sets_dirty_) {
                    for (uint32_t s = 0; s < max_bind_groups; s++)
                        if ((sets_dirty_ & (1u << s)) && !layout->sets[s].empty())
                            bind_buffers(s, *layout);
                    sets_dirty_ = 0;
                }
                if (samplers_dirty_) {
                    bind_samplers();
                    samplers_dirty_ = false;
                }
                if (push_dirty_ && layout->push_slot != ~0u) {
                    const uint32_t bytes = static_cast<uint32_t>(align_up(std::max(layout->push_bytes, 16u), 16));
                    const uint64_t at = dev_.push_alloc(bytes);
                    gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, dev_.push_ring());
                    gl_.glBufferSubData(GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(at), static_cast<GLsizeiptr>(bytes),
                                        push_.data());
                    gl_.glBindBufferRange(GL_UNIFORM_BUFFER, layout->push_slot, dev_.push_ring(),
                                          static_cast<GLintptr>(at), static_cast<GLsizeiptr>(bytes));
                    push_dirty_ = false;
                }
                return true;
            }

            // ------------------------------------------------ drawing

            void set_arrays(int64_t base_vertex, int64_t base_instance) {
                gl_cache& c = cache_;
                uint32_t used = 0;
                for (uint32_t i = 0; i < pipeline_->attribute_count; i++) {
                    const vertex_attribute_gl& a = pipeline_->attributes[i];
                    const buffer* b = dev_.buffer_of(vertex_[a.stream].buffer);
                    if (!b || !b->gl || a.location >= c.attribs.size())
                        continue;
                    const vertex_stream_desc& stream = pipeline_->streams[a.stream];
                    const bool per_instance = stream.step == vertex_step::instance;
                    // there is no base vertex or base instance before ES 3.2 and its extensions: start the arrays later
                    const int64_t base = per_instance ? base_instance : base_vertex;
                    const int64_t offset =
                        static_cast<int64_t>(vertex_[a.stream].offset) + a.offset + base * stream.stride;
                    if (offset < 0)
                        continue;
                    used |= 1u << a.location;
                    attrib_state& s = c.attribs[a.location];
                    const GLuint divisor = per_instance ? 1 : 0;
                    if (!(c.enabled_attribs & (1u << a.location))) {
                        gl_.glEnableVertexAttribArray(a.location);
                        c.enabled_attribs |= 1u << a.location;
                    }
                    if (!s.valid || s.buffer != b->gl || s.offset != static_cast<uint64_t>(offset) ||
                        s.stride != static_cast<GLsizei>(stream.stride) || s.size != a.size || s.type != a.type ||
                        s.normalized != a.normalized || s.integer != a.integer) {
                        if (c.array_buffer != b->gl) {
                            gl_.glBindBuffer(GL_ARRAY_BUFFER, b->gl);
                            c.array_buffer = b->gl;
                        }
                        const void* pointer = gl_offset(offset);
                        if (a.integer)
                            gl_.glVertexAttribIPointer(a.location, a.size, a.type, static_cast<GLsizei>(stream.stride),
                                                       pointer);
                        else
                            gl_.glVertexAttribPointer(a.location, a.size, a.type, a.normalized ? GL_TRUE : GL_FALSE,
                                                      static_cast<GLsizei>(stream.stride), pointer);
                        s.valid = true;
                        s.buffer = b->gl;
                        s.offset = static_cast<uint64_t>(offset);
                        s.stride = static_cast<GLsizei>(stream.stride);
                        s.size = a.size;
                        s.type = a.type;
                        s.normalized = a.normalized;
                        s.integer = a.integer;
                    }
                    if (s.divisor != divisor) {
                        gl_.glVertexAttribDivisor(a.location, divisor);
                        s.divisor = divisor;
                    }
                }
                const uint32_t stale = c.enabled_attribs & ~used;
                for (uint32_t l = 0; l < c.attribs.size(); l++)
                    if (stale & (1u << l)) {
                        gl_.glDisableVertexAttribArray(l);
                        c.enabled_attribs &= ~(1u << l);
                    }
                applied_base_vertex_ = base_vertex;
                applied_base_instance_ = base_instance;
                attribs_dirty_ = false;
            }

            bool prepare_draw(int64_t base_vertex, int64_t base_instance) {
                if (!in_pass_ || !pipeline_ || pipeline_->compute) {
                    dev_.report_once("a draw was recorded outside a render pass or without a graphics pipeline");
                    return false;
                }
                if (pipeline_dirty_) {
                    dev_.use_program(pipeline_->program);
                    apply_state(pipeline_->state);
                    pipeline_dirty_ = false;
                    stencil_dirty_ = true;
                }
                if (!bind_resources())
                    return false;
                if (pipeline_->state.stencil_test && stencil_dirty_) {
                    const pipeline_state& s = pipeline_->state;
                    gl_.glStencilFuncSeparate(GL_FRONT, s.front.func, static_cast<GLint>(stencil_ref_), s.stencil_read);
                    gl_.glStencilFuncSeparate(GL_BACK, s.back.func, static_cast<GLint>(stencil_ref_), s.stencil_read);
                    stencil_dirty_ = false;
                }
                if (attribs_dirty_ || base_vertex != applied_base_vertex_ || base_instance != applied_base_instance_)
                    set_arrays(base_vertex, base_instance);
                return true;
            }

            void draw(const p_draw& a) {
                if (!prepare_draw(0, a.first_instance))
                    return;
                if (a.instance_count == 1)
                    gl_.glDrawArrays(pipeline_->topology, static_cast<GLint>(a.first_vertex),
                                     static_cast<GLsizei>(a.vertex_count));
                else
                    gl_.glDrawArraysInstanced(pipeline_->topology, static_cast<GLint>(a.first_vertex),
                                              static_cast<GLsizei>(a.vertex_count),
                                              static_cast<GLsizei>(a.instance_count));
            }

            bool bind_index_buffer() {
                const buffer* b = dev_.buffer_of(index_);
                if (!b || !b->gl) {
                    dev_.report_once("an indexed draw was recorded without an index buffer");
                    return false;
                }
                if (index_dirty_ || cache_.element_buffer != b->gl) {
                    gl_.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, b->gl);
                    cache_.element_buffer = b->gl;
                    index_dirty_ = false;
                }
                return true;
            }

            void draw_indexed(const p_draw_indexed& a) {
                if (!prepare_draw(a.vertex_offset, a.first_instance) || !bind_index_buffer())
                    return;
                const bool wide = index_type_ == index_type::u32;
                const uint64_t offset = index_offset_ + static_cast<uint64_t>(a.first_index) * (wide ? 4 : 2);
                const void* at = gl_offset(offset);
                const GLenum type = wide ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT;
                if (a.instance_count == 1)
                    gl_.glDrawElements(pipeline_->topology, static_cast<GLsizei>(a.index_count), type, at);
                else
                    gl_.glDrawElementsInstanced(pipeline_->topology, static_cast<GLsizei>(a.index_count), type, at,
                                                static_cast<GLsizei>(a.instance_count));
            }

            void draw_indirect(const p_indirect& a) {
                if (!dev_.features().indirect || !gl_.glDrawElementsIndirect) {
                    dev_.report_once("indirect draws need OpenGL ES 3.1");
                    return;
                }
                const buffer* args = dev_.buffer_of(a.args);
                if (!args || !args->gl || !prepare_draw(0, 0) || !bind_index_buffer())
                    return;
                gl_.glBindBuffer(GL_DRAW_INDIRECT_BUFFER, args->gl);
                const GLenum type = index_type_ == index_type::u32 ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT;
                const uint32_t stride = a.stride ? a.stride : sizeof(draw_indexed_indirect_command);
                // one draw per call: ES has no multi draw indirect
                for (uint32_t i = 0; i < a.max_draws; i++)
                    gl_.glDrawElementsIndirect(pipeline_->topology, type,
                                               gl_offset(a.offset + static_cast<uint64_t>(i) * stride));
            }

            bool prepare_dispatch() {
                if (!pipeline_ || !pipeline_->compute) {
                    dev_.report_once("a dispatch was recorded without a compute pipeline");
                    return false;
                }
                if (pipeline_dirty_) {
                    dev_.use_program(pipeline_->program);
                    pipeline_dirty_ = false;
                }
                return bind_resources();
            }

            struct bound {
                bind_group_handle group;
                std::array<uint32_t, max_dynamic_offsets> offsets{};
                uint32_t offset_count = 0;
            };
            struct vertex_slot {
                buffer_handle buffer;
                uint64_t offset = 0;
            };

            device& dev_;
            const gl_api& gl_;
            gl_cache& cache_;

            const pipeline* pipeline_ = nullptr;
            bool pipeline_dirty_ = true;
            std::array<bound, max_bind_groups> sets_{};
            uint32_t sets_dirty_ = 0xF;
            bool samplers_dirty_ = true;
            std::array<uint8_t, max_push_constant_bytes> push_{};
            bool push_dirty_ = true;
            std::array<vertex_slot, max_vertex_streams> vertex_{};
            bool attribs_dirty_ = true;
            int64_t applied_base_vertex_ = 0, applied_base_instance_ = 0;
            buffer_handle index_;
            index_type index_type_ = index_type::u32;
            uint64_t index_offset_ = 0;
            bool index_dirty_ = true;
            uint32_t stencil_ref_ = 0;
            bool stencil_dirty_ = true;
            viewport viewport_;
            rect scissor_;
            bool in_pass_ = false;
            bool read_only_depth_ = false;
            rendering_desc desc_;
            std::array<uint64_t, max_color_targets + 1> pass_key_{};
            GLuint pass_fbo_ = 0;
            uint32_t pass_width_ = 0, pass_height_ = 0;
        };
    }

    void device::replay(const std::vector<uint8_t>& stream) {
        replayer r(*this);
        r.run(stream);
    }
}
