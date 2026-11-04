#include "webgpu/webgpu_internal.hpp"

#include <cstdio>

namespace conduit::webgpu {
    namespace {
        WGPULoadOp to_load(load_op op) {
            return op == load_op::load ? WGPULoadOp_Load : WGPULoadOp_Clear;
        }
        WGPUStoreOp to_store(store_op op) {
            return op == store_op::store ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
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
        device_.mutable_stats().barriers += textures.size() + buffers.size();
    }
    void webgpu_command_list::uav_barrier(buffer_handle) {
        device_.mutable_stats().barriers++;
    }
    void webgpu_command_list::uav_barrier(texture_handle) {
        device_.mutable_stats().barriers++;
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
        dirty_sets_ = 0xF;
        pipeline_dirty_ = true;
        vertex_dirty_ = 0xFF;
        index_dirty_ = true;
        viewport_dirty_ = viewport_set_;
        scissor_dirty_ = scissor_set_;
        stencil_dirty_ = true;
    }

    void webgpu_command_list::end_rendering() {
        end_pass();
    }

    void webgpu_command_list::set_pipeline(pipeline_handle p) {
        pipeline_ = p;
        pipeline_dirty_ = true;
        dirty_sets_ = 0xF;
    }

    void webgpu_command_list::set_bind_group(uint32_t set, bind_group_handle group,
                                             std::span<const uint32_t> dynamic_offsets) {
        if (set >= max_bind_groups)
            return;
        bound_set& s = sets_[set];
        s.group = group;
        s.offset_count = static_cast<uint32_t>(std::min<size_t>(dynamic_offsets.size(), max_dynamic_offsets));
        for (uint32_t i = 0; i < s.offset_count; i++)
            s.offsets[i] = dynamic_offsets[i];
        dirty_sets_ |= 1u << set;
        device_.mutable_stats().bind_group_binds++;
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
            device_.mutable_stats().pipeline_binds++;
        }
        for (uint32_t s = 0; s < p->set_count; s++) {
            if (!(dirty_sets_ & (1u << s)))
                continue;
            native_set* ns = p->sets[s].get();
            WGPUBindGroup group = nullptr;
            std::array<uint32_t, max_dynamic_offsets> picked{};
            uint32_t picked_count = 0;
            if (!ns) {
                group = device_.empty_group();
            } else if (ns->entries.empty()) {
                if (!ns->blank) {
                    WGPUBindGroupDescriptor gd{};
                    gd.layout = ns->layout.get();
                    ns->blank = owned<WGPUBindGroup>(wgpuDeviceCreateBindGroup(device_.dev(), &gd));
                }
                group = ns->blank.get();
            } else {
                group = device_.native_group_for(sets_[s].group, *ns);
                for (uint32_t i = 0; i < ns->dynamic_pick.size() && i < max_dynamic_offsets; i++) {
                    const uint32_t from = ns->dynamic_pick[i];
                    picked[picked_count++] = from < sets_[s].offset_count ? sets_[s].offsets[from] : 0;
                }
            }
            if (!group) {
                device_.report_error("draw: set " + std::to_string(s) + " is not bound");
                return false;
            }
            wgpuRenderPassEncoderSetBindGroup(render_.get(), s, group, picked_count, picked.data());
        }
        dirty_sets_ = 0;
        for (uint32_t i = 0; i < p->stream_count; i++) {
            if (!(vertex_dirty_ & (1u << i)))
                continue;
            buffer* b = device_.buffers_.get(vertex_[i].buffer);
            if (!b) {
                device_.report_error("draw: vertex stream " + std::to_string(i) + " is not bound");
                return false;
            }
            wgpuRenderPassEncoderSetVertexBuffer(render_.get(), i, b->native.get(), vertex_[i].offset,
                                                 b->gpu_size - vertex_[i].offset);
        }
        vertex_dirty_ = 0;
        if (viewport_dirty_ && viewport_set_) {
            viewport v = viewport_;
            if (pass_width_) {
                v.width = (std::min)(v.width, static_cast<float>(pass_width_) - v.x);
                v.height = (std::min)(v.height, static_cast<float>(pass_height_) - v.y);
            }
            wgpuRenderPassEncoderSetViewport(render_.get(), v.x, v.y, v.width, v.height, v.min_depth, v.max_depth);
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
            wgpuRenderPassEncoderSetScissorRect(render_.get(), static_cast<uint32_t>(r.x), static_cast<uint32_t>(r.y),
                                                r.width, r.height);
            scissor_dirty_ = false;
        }
        if (stencil_dirty_) {
            wgpuRenderPassEncoderSetStencilReference(render_.get(), stencil_ref_);
            stencil_dirty_ = false;
        }
        return true;
    }

    void webgpu_command_list::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                                   uint32_t first_instance) {
        if (!prepare_draw())
            return;
        wgpuRenderPassEncoderDraw(render_.get(), vertex_count, instance_count, first_vertex, first_instance);
        device_.mutable_stats().draws++;
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
            wgpuRenderPassEncoderSetIndexBuffer(render_.get(), b->native.get(),
                                                index_type_ == index_type::u16 ? WGPUIndexFormat_Uint16
                                                                               : WGPUIndexFormat_Uint32,
                                                index_offset_, b->gpu_size - index_offset_);
            index_dirty_ = false;
        }
        wgpuRenderPassEncoderDrawIndexed(render_.get(), index_count, instance_count, first_index, vertex_offset,
                                         first_instance);
        device_.mutable_stats().draws++;
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
        if (index_dirty_) {
            wgpuRenderPassEncoderSetIndexBuffer(render_.get(), ib->native.get(),
                                                index_type_ == index_type::u16 ? WGPUIndexFormat_Uint16
                                                                               : WGPUIndexFormat_Uint32,
                                                index_offset_, ib->gpu_size - index_offset_);
            index_dirty_ = false;
        }
        for (uint32_t i = 0; i < max_draws; i++)
            wgpuRenderPassEncoderDrawIndexedIndirect(render_.get(), a->native.get(),
                                                     offset + static_cast<uint64_t>(i) * stride);
        device_.mutable_stats().indirect_draws += max_draws;
    }

    void webgpu_command_list::draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle, uint64_t,
                                                          uint32_t max_draws, uint32_t stride) {
        draw_indexed_indirect(args, offset, max_draws, stride);
    }

    void webgpu_command_list::dispatch(uint32_t x, uint32_t y, uint32_t z) {
        const pipeline* p = device_.pipelines_.get(pipeline_);
        if (!p || !p->compute || !p->comp) {
            device_.report_error("Dispatch without a compute pipeline");
            return;
        }
        if (!compute_) {
            end_pass();
            compute_ = owned<WGPUComputePassEncoder>(wgpuCommandEncoderBeginComputePass(encoder_.get(), nullptr));
            dirty_sets_ = 0xF;
            pipeline_dirty_ = true;
        }
        if (pipeline_dirty_) {
            wgpuComputePassEncoderSetPipeline(compute_.get(), p->comp.get());
            pipeline_dirty_ = false;
        }
        for (uint32_t s = 0; s < p->set_count; s++) {
            if (!(dirty_sets_ & (1u << s)))
                continue;
            native_set* ns = p->sets[s].get();
            WGPUBindGroup group = ns ? device_.native_group_for(sets_[s].group, *ns) : device_.empty_group();
            std::array<uint32_t, max_dynamic_offsets> picked{};
            uint32_t picked_count = 0;
            if (ns)
                for (uint32_t i = 0; i < ns->dynamic_pick.size() && i < max_dynamic_offsets; i++)
                    picked[picked_count++] =
                        ns->dynamic_pick[i] < sets_[s].offset_count ? sets_[s].offsets[ns->dynamic_pick[i]] : 0;
            if (!group) {
                device_.report_error("Dispatch: set " + std::to_string(s) + " is not bound");
                return;
            }
            wgpuComputePassEncoderSetBindGroup(compute_.get(), s, group, picked_count, picked.data());
        }
        dirty_sets_ = 0;
        wgpuComputePassEncoderDispatchWorkgroups(compute_.get(), x, y, z);
        device_.mutable_stats().dispatches++;
    }

    void webgpu_command_list::dispatch_indirect(buffer_handle args, uint64_t offset) {
        buffer* a = device_.buffers_.get(args);
        if (!a) {
            device_.report_error("DispatchIndirect: bad buffer");
            return;
        }
        dispatch(0, 0, 0);
        if (compute_)
            wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute_.get(), a->native.get(), offset);
    }

    void webgpu_command_list::write_timestamp(query_set_handle, uint32_t) {}
    void webgpu_command_list::resolve_queries(query_set_handle, uint32_t, uint32_t, buffer_handle, uint64_t) {}
    void webgpu_command_list::begin_debug_label(const char*, uint32_t) {}
    void webgpu_command_list::end_debug_label() {}
}
