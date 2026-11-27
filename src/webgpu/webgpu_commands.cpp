#include "webgpu/webgpu_internal.hpp"

#include <cstdio>

#if !defined(__EMSCRIPTEN__)
#include <webgpu/wgpu.h>
#endif

namespace conduit::webgpu {
    namespace {
        WGPULoadOp to_load(load_op op) {
            return op == load_op::load ? WGPULoadOp_Load : WGPULoadOp_Clear;
        }
        WGPUStoreOp to_store(store_op op) {
            return op == store_op::store ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
        }

        // at most four offsets: a plain loop, where std::equal becomes a memcmp call on every draw
        bool same_offsets(const uint32_t* a, const uint32_t* b, uint32_t count) {
            for (uint32_t i = 0; i < count; i++)
                if (a[i] != b[i])
                    return false;
            return true;
        }
    }

    webgpu_command_list::~webgpu_command_list() = default;

    void webgpu_command_list::begin() {
        render_.reset();
        compute_.reset();
        encoder_.reset();
        encoder_ = owned<WGPUCommandEncoder>(wgpuDeviceCreateCommandEncoder(device_.dev(), nullptr));
        pipeline_ = {};
        sets_ = {};
        vertex_ = {};
        index_buffer_ = {};
        dirty_sets_ = 0xF;
        pipeline_dirty_ = true;
        vertex_dirty_ = 0xFF;
        index_dirty_ = true;
        viewport_set_ = scissor_set_ = false;
        viewport_dirty_ = scissor_dirty_ = false;
        stencil_dirty_ = false;
        pass_width_ = pass_height_ = 0;
        counters_ = {};
        forget_sent();
    }

    void webgpu_command_list::forget_sent() {
        sent_sets_ = {};
        sent_vertex_ = {};
        vertex_shift_ = 0;
        sent_index_buffer_ = {};
        viewport_sent_ = scissor_sent_ = false;
    }

    void webgpu_command_list::invalidate_bound_state() {
        dirty_sets_ = 0xF;
        pipeline_dirty_ = true;
        vertex_dirty_ = 0xFF;
        index_dirty_ = true;
        viewport_dirty_ = viewport_set_;
        scissor_dirty_ = scissor_set_;
        stencil_dirty_ = true;
        forget_sent();
    }

    WGPUCommandBuffer webgpu_command_list::finish() {
        end_pass();
        if (!encoder_)
            return nullptr;
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder_.get(), nullptr);
        encoder_.reset();
        return cb;
    }

    void webgpu_command_list::end_pass() {
        if (render_) {
            wgpuRenderPassEncoderEnd(render_.get());
            render_.reset();
        }
        if (compute_) {
            wgpuComputePassEncoderEnd(compute_.get());
            compute_.reset();
        }
    }

    void webgpu_command_list::barrier(std::span<const texture_barrier> textures,
                                      std::span<const buffer_barrier> buffers) {
        counters_.barriers += textures.size() + buffers.size();
    }
    void webgpu_command_list::uav_barrier(buffer_handle) {
        counters_.barriers++;
    }
    void webgpu_command_list::uav_barrier(texture_handle) {
        counters_.barriers++;
    }

    void webgpu_command_list::copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src,
                                          uint64_t src_offset, uint64_t size) {
        end_pass();
        buffer* d = device_.buffers_.get(dst);
        buffer* s = device_.buffers_.get(src);
        if (!d || !s || !size) {
            device_.report_error("CopyBuffer: bad handle");
            return;
        }
        if ((dst_offset | src_offset) & 3) {
            device_.report_error("CopyBuffer: offsets must be multiples of 4");
            return;
        }
        uint64_t n = align_up(size, 4);
        if (dst_offset + n > d->gpu_size || src_offset + n > s->gpu_size)
            n = size & ~3ull;
        if (!n)
            return;
        wgpuCommandEncoderCopyBufferToBuffer(encoder_.get(), s->native.get(), src_offset, d->native.get(), dst_offset,
                                             n);
    }

    void webgpu_command_list::copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                                     std::span<const buffer_texture_copy> regions) {
        end_pass();
        texture* t = device_.textures_.get(dst);
        buffer* b = device_.buffers_.get(src);
        if (!t || !b) {
            device_.report_error("CopyBufferToTexture: bad handle");
            return;
        }
        const bool depth = is_depth_format(t->desc.format);
        const bool array = t->desc.dim != texture_dim::tex3d;
        for (const buffer_texture_copy& r : regions) {
            WGPUTexelCopyBufferInfo bi{};
            bi.buffer = b->native.get();
            bi.layout.offset = r.buffer_offset;
            bi.layout.bytesPerRow = r.buffer_row_pitch ? r.buffer_row_pitch : row_bytes(t->desc.format, r.width);
            bi.layout.rowsPerImage = row_blocks(t->desc.format, r.height);
            WGPUTexelCopyTextureInfo ti{};
            ti.texture = t->native.get();
            ti.mipLevel = r.mip;
            ti.origin = {r.x, r.y, array ? r.layer : r.z};
            ti.aspect = depth ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
            const WGPUExtent3D size{r.width, r.height, array ? 1u : (std::max)(1u, r.depth)};
            wgpuCommandEncoderCopyBufferToTexture(encoder_.get(), &bi, &ti, &size);
        }
    }

    void webgpu_command_list::copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                                     std::span<const buffer_texture_copy> regions) {
        end_pass();
        texture* t = device_.textures_.get(src);
        buffer* b = device_.buffers_.get(dst);
        if (!t || !b || !t->native) {
            device_.report_error("CopyTextureToBuffer: bad handle");
            return;
        }
        const bool depth = is_depth_format(t->desc.format);
        const bool array = t->desc.dim != texture_dim::tex3d;
        for (const buffer_texture_copy& r : regions) {
            WGPUTexelCopyBufferInfo bi{};
            bi.buffer = b->native.get();
            bi.layout.offset = r.buffer_offset;
            bi.layout.bytesPerRow = r.buffer_row_pitch ? r.buffer_row_pitch : row_bytes(t->desc.format, r.width);
            bi.layout.rowsPerImage = row_blocks(t->desc.format, r.height);
            WGPUTexelCopyTextureInfo ti{};
            ti.texture = t->native.get();
            ti.mipLevel = r.mip;
            ti.origin = {r.x, r.y, array ? r.layer : r.z};
            ti.aspect = depth ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
            const WGPUExtent3D size{r.width, r.height, array ? 1u : (std::max)(1u, r.depth)};
            wgpuCommandEncoderCopyTextureToBuffer(encoder_.get(), &ti, &bi, &size);
        }
    }

    void webgpu_command_list::copy_texture(texture_handle dst, texture_handle src,
                                           std::span<const texture_copy> regions) {
        end_pass();
        texture* d = device_.textures_.get(dst);
        texture* s = device_.textures_.get(src);
        if (!d || !s || !d->native || !s->native) {
            device_.report_error("CopyTexture: bad handle");
            return;
        }
        for (const texture_copy& r : regions) {
            WGPUTexelCopyTextureInfo si{};
            si.texture = s->native.get();
            si.mipLevel = r.src_mip;
            si.origin = {r.src_x, r.src_y, r.src_layer};
            si.aspect = WGPUTextureAspect_All;
            WGPUTexelCopyTextureInfo di{};
            di.texture = d->native.get();
            di.mipLevel = r.dst_mip;
            di.origin = {r.dst_x, r.dst_y, r.dst_layer};
            di.aspect = WGPUTextureAspect_All;
            const WGPUExtent3D size{r.width, r.height, 1};
            wgpuCommandEncoderCopyTextureToTexture(encoder_.get(), &si, &di, &size);
        }
    }

    void webgpu_command_list::fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) {
        end_pass();
        buffer* b = device_.buffers_.get(dst);
        if (!b) {
            device_.report_error("FillBuffer: bad handle");
            return;
        }
        if (value != 0) {
            device_.report_error("FillBuffer: webgpu can only clear to zero");
            return;
        }
        wgpuCommandEncoderClearBuffer(encoder_.get(), b->native.get(), offset,
                                      size ? align_up(size, 4) : WGPU_WHOLE_SIZE);
    }

    void webgpu_command_list::resolve_texture(texture_handle dst, texture_handle src) {
        end_pass();
        texture* d = device_.textures_.get(dst);
        texture* s = device_.textures_.get(src);
        if (!d || !s) {
            device_.report_error("ResolveTexture: bad handle");
            return;
        }
        WGPURenderPassColorAttachment ca{};
        ca.view = device_.view_of(*s, {}, texture_dim::tex2d, format::unknown, false, true);
        ca.resolveTarget = device_.view_of(*d, {}, texture_dim::tex2d, format::unknown, false, true);
        ca.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        ca.loadOp = WGPULoadOp_Load;
        ca.storeOp = WGPUStoreOp_Discard;
        WGPURenderPassDescriptor rd{};
        rd.label = view("resolve");
        rd.colorAttachmentCount = 1;
        rd.colorAttachments = &ca;
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder_.get(), &rd);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }

    void webgpu_command_list::begin_rendering(const rendering_desc& desc) {
        end_pass();
        std::array<WGPURenderPassColorAttachment, max_color_targets> colors{};
        pass_width_ = pass_height_ = 0;
        for (uint32_t i = 0; i < desc.color_count; i++) {
            const color_attachment& a = desc.colors[i];
            texture* t = device_.textures_.get(a.texture);
            if (!t || !t->native) {
                device_.report_error("BeginRendering: colour attachment " + std::to_string(i) + " is gone");
                return;
            }
            const texture_dim dim = texture_dim::tex2d;
            colors[i].view = device_.view_of(*t, {a.mip, 1, a.layer, 1}, dim, format::unknown, false, true);
            colors[i].depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
            colors[i].loadOp = to_load(a.load);
            colors[i].storeOp = to_store(a.store);
            colors[i].clearValue = {a.clear[0], a.clear[1], a.clear[2], a.clear[3]};
            if (a.resolve)
                if (texture* r = device_.textures_.get(a.resolve))
                    colors[i].resolveTarget = device_.view_of(*r, {}, dim, format::unknown, false, true);
            if (!pass_width_) {
                pass_width_ = mip_extent(t->desc.width, a.mip);
                pass_height_ = mip_extent(t->desc.height, a.mip);
            }
        }
        WGPURenderPassDepthStencilAttachment depth{};
        bool have_depth = false;
        if (desc.depth.texture) {
            texture* t = device_.textures_.get(desc.depth.texture);
            if (!t || !t->native) {
                device_.report_error("BeginRendering: depth attachment is gone");
                return;
            }
            depth.view = device_.view_of(*t, {desc.depth.mip, 1, desc.depth.layer, 1}, texture_dim::tex2d,
                                         format::unknown, false, true);
            if (desc.depth.read_only) {
                depth.depthReadOnly = true;
                depth.stencilReadOnly = has_stencil(t->desc.format);
            } else {
                depth.depthLoadOp = to_load(desc.depth.load);
                depth.depthStoreOp = to_store(desc.depth.store);
                depth.depthClearValue = desc.depth.clear_depth;
                if (has_stencil(t->desc.format)) {
                    depth.stencilLoadOp = to_load(desc.depth.load);
                    depth.stencilStoreOp = to_store(desc.depth.store);
                    depth.stencilClearValue = desc.depth.clear_stencil;
                }
            }
            have_depth = true;
            if (!pass_width_) {
                pass_width_ = mip_extent(t->desc.width, desc.depth.mip);
                pass_height_ = mip_extent(t->desc.height, desc.depth.mip);
            }
        }
        WGPURenderPassDescriptor rd{};
        rd.label = view(desc.name);
        rd.colorAttachmentCount = desc.color_count;
        rd.colorAttachments = colors.data();
        rd.depthStencilAttachment = have_depth ? &depth : nullptr;
        render_ = owned<WGPURenderPassEncoder>(wgpuCommandEncoderBeginRenderPass(encoder_.get(), &rd));
        invalidate_bound_state();
        // a pass starts with a zero stencil reference
        stencil_dirty_ = stencil_ref_ != 0;
    }

    void webgpu_command_list::end_rendering() {
        end_pass();
    }

    void webgpu_command_list::set_pipeline(pipeline_handle p) {
        // the same pipeline again keeps every group it already has bound
        if (p == pipeline_ && !pipeline_dirty_)
            return;
        pipeline_ = p;
        pipeline_dirty_ = true;
        dirty_sets_ = 0xF;
    }

    void webgpu_command_list::set_bind_group(uint32_t set, bind_group_handle group,
                                             std::span<const uint32_t> dynamic_offsets) {
        if (set >= max_bind_groups)
            return;
        bound_set& s = sets_[set];
        const uint32_t count = static_cast<uint32_t>(std::min<size_t>(dynamic_offsets.size(), max_dynamic_offsets));
        if (s.group == group && s.offset_count == count &&
            same_offsets(dynamic_offsets.data(), s.offsets.data(), count))
            return;
        s.group = group;
        s.offset_count = count;
        for (uint32_t i = 0; i < count; i++)
            s.offsets[i] = dynamic_offsets[i];
        dirty_sets_ |= 1u << set;
        counters_.group_binds++;
    }

    void webgpu_command_list::push_constants(const void*, uint32_t, uint32_t) {
        if (!warned_push_) {
            warned_push_ = true;
            device_.report_error("PushConstants: webgpu has none; per-draw data belongs in a uniform buffer");
        }
    }

    void webgpu_command_list::set_viewport(const viewport& v) {
        viewport_ = v;
        viewport_set_ = viewport_dirty_ = true;
    }

    void webgpu_command_list::set_scissor(const rect& r) {
        scissor_ = r;
        scissor_set_ = scissor_dirty_ = true;
    }

    void webgpu_command_list::set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t offset) {
        if (slot >= max_vertex_streams)
            return;
        vertex_[slot] = {buffer, offset};
        vertex_dirty_ |= 1u << slot;
    }

    void webgpu_command_list::set_index_buffer(buffer_handle buffer, index_type type, uint64_t offset) {
        index_buffer_ = buffer;
        index_type_ = type;
        index_offset_ = offset;
        index_dirty_ = true;
    }

    void webgpu_command_list::set_stencil_reference(uint32_t value) {
        stencil_ref_ = value;
        stencil_dirty_ = true;
    }

    bool webgpu_command_list::set_native_group(uint32_t s, const native_set* ns) {
        sent_set& sent = sent_sets_[s];
        WGPUBindGroup group = nullptr;
        bind_group_handle handle;
        WGPUBindGroupLayout layout = nullptr;
        std::array<uint32_t, max_dynamic_offsets> picked{};
        uint32_t picked_count = 0;
        if (!ns) {
            group = device_.empty_group();
        } else if (ns->entries.empty()) {
            group = ns->blank.get();
        } else {
            handle = sets_[s].group;
            layout = ns->layout.get();
            // the same group under the same layout keeps the native group resolved when it was last sent
            if (handle == sent.group && layout == sent.layout && device_.bind_groups_.get(handle))
                group = sent.native;
            else
                group = device_.native_group_for(handle, *ns);
            for (uint32_t i = 0; i < ns->dynamic_pick.size() && i < max_dynamic_offsets; i++) {
                const uint32_t from = ns->dynamic_pick[i];
                picked[picked_count++] = from < sets_[s].offset_count ? sets_[s].offsets[from] : 0;
            }
        }
        if (!group) {
            device_.report_error(std::string(render_ ? "draw" : "Dispatch") + ": set " + std::to_string(s) +
                                 " is not bound");
            return false;
        }
        if (group == sent.native && handle == sent.group && layout == sent.layout &&
            picked_count == sent.offset_count && same_offsets(picked.data(), sent.offsets.data(), picked_count))
            return true;
        if (render_)
            wgpuRenderPassEncoderSetBindGroup(render_.get(), s, group, picked_count, picked.data());
        else
            wgpuComputePassEncoderSetBindGroup(compute_.get(), s, group, picked_count, picked.data());
        sent = {handle, layout, group, picked, picked_count};
        return true;
    }

    bool webgpu_command_list::prepare_draw() {
        if (!render_) {
            device_.report_error("draw outside a rendering pass");
            return false;
        }
        const pipeline* p = device_.pipelines_.get(pipeline_);
        if (!p || p->compute || !p->render) {
            device_.report_error("draw without a graphics pipeline");
            return false;
        }
        if (pipeline_dirty_) {
            wgpuRenderPassEncoderSetPipeline(render_.get(), p->render.get());
            pipeline_dirty_ = false;
            counters_.pipeline_binds++;
            // a shifted stream 0 is only right for the pipeline that allowed it
            vertex_shift_ = 0;
            vertex_dirty_ |= 1;
        }
        for (uint32_t s = 0; s < p->set_count; s++)
            if ((dirty_sets_ & (1u << s)) && !set_native_group(s, p->sets[s].get()))
                return false;
        dirty_sets_ = 0;
        for (uint32_t i = 0; i < p->stream_count; i++) {
            if (!(vertex_dirty_ & (1u << i)))
                continue;
            buffer* b = device_.buffers_.get(vertex_[i].buffer);
            if (!b) {
                device_.report_error("draw: vertex stream " + std::to_string(i) + " is not bound");
                return false;
            }
            if (i == 0)
                vertex_shift_ = 0;
            if (vertex_[i] == sent_vertex_[i])
                continue;
            // the usual per-draw step through one buffer: the bound range already covers it, so only
            // first_vertex moves, and the draw sends one call instead of two
            if (i == 0 && p->shift_stride && vertex_[0].buffer == sent_vertex_[0].buffer &&
                vertex_[0].offset > sent_vertex_[0].offset) {
                const uint64_t delta = vertex_[0].offset - sent_vertex_[0].offset;
                if (delta % p->shift_stride == 0 && delta / p->shift_stride <= 0x7FFFFFFFull) {
                    vertex_shift_ = static_cast<uint32_t>(delta / p->shift_stride);
                    continue;
                }
            }
            wgpuRenderPassEncoderSetVertexBuffer(render_.get(), i, b->native.get(), vertex_[i].offset,
                                                 b->gpu_size - vertex_[i].offset);
            sent_vertex_[i] = vertex_[i];
        }
        vertex_dirty_ = 0;
        if (viewport_dirty_ && viewport_set_) {
            viewport v = viewport_;
            if (pass_width_) {
                v.width = (std::min)(v.width, static_cast<float>(pass_width_) - v.x);
                v.height = (std::min)(v.height, static_cast<float>(pass_height_) - v.y);
            }
            const viewport& o = sent_viewport_;
            if (!viewport_sent_ || v.x != o.x || v.y != o.y || v.width != o.width || v.height != o.height ||
                v.min_depth != o.min_depth || v.max_depth != o.max_depth) {
                wgpuRenderPassEncoderSetViewport(render_.get(), v.x, v.y, v.width, v.height, v.min_depth, v.max_depth);
                sent_viewport_ = v;
                viewport_sent_ = true;
            }
            viewport_dirty_ = false;
        }
        if (scissor_dirty_ && scissor_set_) {
            rect r = scissor_;
            r.x = (std::max)(r.x, 0);
            r.y = (std::max)(r.y, 0);
            if (pass_width_) {
                r.width = static_cast<uint32_t>(
                    std::max<int64_t>(0, std::min<int64_t>(r.width, static_cast<int64_t>(pass_width_) - r.x)));
                r.height = static_cast<uint32_t>(
                    std::max<int64_t>(0, std::min<int64_t>(r.height, static_cast<int64_t>(pass_height_) - r.y)));
            }
            const rect& o = sent_scissor_;
            if (!scissor_sent_ || r.x != o.x || r.y != o.y || r.width != o.width || r.height != o.height) {
                wgpuRenderPassEncoderSetScissorRect(render_.get(), static_cast<uint32_t>(r.x),
                                                    static_cast<uint32_t>(r.y), r.width, r.height);
                sent_scissor_ = r;
                scissor_sent_ = true;
            }
            scissor_dirty_ = false;
        }
        if (stencil_dirty_) {
            wgpuRenderPassEncoderSetStencilReference(render_.get(), stencil_ref_);
            stencil_dirty_ = false;
        }
        return true;
    }

    bool webgpu_command_list::unshift_vertices() {
        if (!vertex_shift_)
            return true;
        buffer* b = device_.buffers_.get(vertex_[0].buffer);
        if (!b) {
            device_.report_error("draw: vertex stream 0 is not bound");
            return false;
        }
        wgpuRenderPassEncoderSetVertexBuffer(render_.get(), 0, b->native.get(), vertex_[0].offset,
                                             b->gpu_size - vertex_[0].offset);
        sent_vertex_[0] = vertex_[0];
        vertex_shift_ = 0;
        return true;
    }

    void webgpu_command_list::bind_index_buffer(const buffer& b) {
        if (index_buffer_ != sent_index_buffer_ || index_type_ != sent_index_type_ ||
            index_offset_ != sent_index_offset_) {
            wgpuRenderPassEncoderSetIndexBuffer(render_.get(), b.native.get(),
                                                index_type_ == index_type::u16 ? WGPUIndexFormat_Uint16
                                                                               : WGPUIndexFormat_Uint32,
                                                index_offset_, b.gpu_size - index_offset_);
            sent_index_buffer_ = index_buffer_;
            sent_index_type_ = index_type_;
            sent_index_offset_ = index_offset_;
        }
        index_dirty_ = false;
    }

    void webgpu_command_list::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                                   uint32_t first_instance) {
        if (!prepare_draw())
            return;
        if (static_cast<uint64_t>(first_vertex) + vertex_shift_ > 0xFFFFFFFFull && !unshift_vertices())
            return;
        wgpuRenderPassEncoderDraw(render_.get(), vertex_count, instance_count, first_vertex + vertex_shift_,
                                  first_instance);
        counters_.draws++;
    }

    void webgpu_command_list::draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                                           int32_t vertex_offset, uint32_t first_instance) {
        if (!prepare_draw())
            return;
        if (index_dirty_) {
            buffer* b = device_.buffers_.get(index_buffer_);
            if (!b) {
                device_.report_error("DrawIndexed: no index buffer");
                return;
            }
            bind_index_buffer(*b);
        }
        if (static_cast<int64_t>(vertex_offset) + vertex_shift_ > 0x7FFFFFFF && !unshift_vertices())
            return;
        wgpuRenderPassEncoderDrawIndexed(render_.get(), index_count, instance_count, first_index,
                                         static_cast<int32_t>(vertex_offset + static_cast<int64_t>(vertex_shift_)),
                                         first_instance);
        counters_.draws++;
    }

    void webgpu_command_list::draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws,
                                                    uint32_t stride) {
        if (!prepare_draw())
            return;
        buffer* a = device_.buffers_.get(args);
        buffer* ib = device_.buffers_.get(index_buffer_);
        if (!a || !ib) {
            device_.report_error("DrawIndexedIndirect: bad buffer");
            return;
        }
        if (index_dirty_)
            bind_index_buffer(*ib);
        // the arguments live on the gpu, so the buffer has to be where they expect it
        if (!unshift_vertices())
            return;
#if !defined(__EMSCRIPTEN__)
        if (max_draws > 1 && stride == sizeof(draw_indexed_indirect_command) &&
            device_.caps().features.multi_draw_indirect) {
            wgpuRenderPassEncoderMultiDrawIndexedIndirect(render_.get(), a->native.get(), offset, max_draws);
            counters_.indirect += max_draws;
            return;
        }
#endif
        for (uint32_t i = 0; i < max_draws; i++)
            wgpuRenderPassEncoderDrawIndexedIndirect(render_.get(), a->native.get(),
                                                     offset + static_cast<uint64_t>(i) * stride);
        counters_.indirect += max_draws;
    }

    void webgpu_command_list::draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle count,
                                                          uint64_t count_offset, uint32_t max_draws, uint32_t stride) {
#if !defined(__EMSCRIPTEN__)
        buffer* c = device_.buffers_.get(count);
        if (c && stride == sizeof(draw_indexed_indirect_command) && device_.caps().features.draw_indirect_count) {
            buffer* a = device_.buffers_.get(args);
            buffer* ib = device_.buffers_.get(index_buffer_);
            if (!prepare_draw())
                return;
            if (!a || !ib) {
                device_.report_error("DrawIndexedIndirectCount: bad buffer");
                return;
            }
            if (index_dirty_)
                bind_index_buffer(*ib);
            if (!unshift_vertices())
                return;
            wgpuRenderPassEncoderMultiDrawIndexedIndirectCount(render_.get(), a->native.get(), offset, c->native.get(),
                                                               count_offset, max_draws);
            counters_.indirect += max_draws;
            return;
        }
#else
        (void)count;
        (void)count_offset;
#endif
        draw_indexed_indirect(args, offset, max_draws, stride);
    }

    void webgpu_command_list::dispatch(uint32_t x, uint32_t y, uint32_t z) {
        if (!prepare_dispatch())
            return;
        wgpuComputePassEncoderDispatchWorkgroups(compute_.get(), x, y, z);
        counters_.dispatches++;
    }

    bool webgpu_command_list::prepare_dispatch() {
        const pipeline* p = device_.pipelines_.get(pipeline_);
        if (!p || !p->compute || !p->comp) {
            device_.report_error("Dispatch without a compute pipeline");
            return false;
        }
        if (!compute_) {
            end_pass();
            compute_ = owned<WGPUComputePassEncoder>(wgpuCommandEncoderBeginComputePass(encoder_.get(), nullptr));
            dirty_sets_ = 0xF;
            pipeline_dirty_ = true;
            forget_sent();
        }
        if (pipeline_dirty_) {
            wgpuComputePassEncoderSetPipeline(compute_.get(), p->comp.get());
            pipeline_dirty_ = false;
        }
        for (uint32_t s = 0; s < p->set_count; s++)
            if ((dirty_sets_ & (1u << s)) && !set_native_group(s, p->sets[s].get()))
                return false;
        dirty_sets_ = 0;
        return true;
    }

    void webgpu_command_list::dispatch_indirect(buffer_handle args, uint64_t offset) {
        buffer* a = device_.buffers_.get(args);
        if (!a) {
            device_.report_error("DispatchIndirect: bad buffer");
            return;
        }
        // binds like dispatch does, without the empty direct dispatch it used to record first
        if (!prepare_dispatch())
            return;
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_.get(), a->native.get(), offset);
        counters_.dispatches++;
    }

    void webgpu_command_list::write_timestamp(query_set_handle, uint32_t) {}
    void webgpu_command_list::resolve_queries(query_set_handle, uint32_t, uint32_t, buffer_handle, uint64_t) {}
    void webgpu_command_list::begin_debug_label(const char*, uint32_t) {}
    void webgpu_command_list::end_debug_label() {}
}
