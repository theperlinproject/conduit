#include "metal/metal_internal.hpp"

#include <algorithm>
#include <cstring>

namespace conduit::metal {
    namespace {
        MTLOrigin origin(uint32_t x, uint32_t y, uint32_t z) {
            return MTLOriginMake(x, y, z);
        }
        MTLSize size(uint32_t w, uint32_t h, uint32_t d) {
            return MTLSizeMake(w, h, d);
        }

        uint32_t slice_of(const texture_desc& d, uint32_t layer) {
            return d.dim == texture_dim::tex3d ? 0 : layer;
        }
    }

    cmd_list::~cmd_list() {
        end_encoder();
    }

    void cmd_list::begin() {
        end_encoder();
        command_buffer_ = [device_.queue() commandBuffer];
        pipeline_ = {};
        sets_ = {};
        vertex_ = {};
        index_buffer_ = {};
        dirty_sets_ = 0xF;
        pipeline_dirty_ = true;
        vertex_dirty_ = 0xFF;
        viewport_set_ = scissor_set_ = false;
        pass_width_ = pass_height_ = 0;
    }

    id<MTLCommandBuffer> cmd_list::finish() {
        end_encoder();
        id<MTLCommandBuffer> cb = command_buffer_;
        command_buffer_ = nil;
        return cb;
    }

    void cmd_list::end_encoder() {
        if (render_) {
            [render_ endEncoding];
            render_ = nil;
        }
        if (compute_) {
            [compute_ endEncoding];
            compute_ = nil;
        }
        if (blit_) {
            [blit_ endEncoding];
            blit_ = nil;
        }
    }

    id<MTLBlitCommandEncoder> cmd_list::blit() {
        if (!blit_) {
            end_encoder();
            blit_ = [command_buffer_ blitCommandEncoder];
        }
        return blit_;
    }

    id<MTLComputeCommandEncoder> cmd_list::compute() {
        if (!compute_) {
            end_encoder();
            compute_ = [command_buffer_ computeCommandEncoder];
            dirty_sets_ = 0xF;
            pipeline_dirty_ = true;
        }
        return compute_;
    }

    void cmd_list::barrier(std::span<const texture_barrier> textures, std::span<const buffer_barrier> buffers) {
        device_.mutable_stats().barriers += textures.size() + buffers.size();
    }
    void cmd_list::uav_barrier(buffer_handle) {
        device_.mutable_stats().barriers++;
    }
    void cmd_list::uav_barrier(texture_handle) {
        device_.mutable_stats().barriers++;
    }

    void cmd_list::copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src, uint64_t src_offset,
                               uint64_t size) {
        buffer* d = device_.buffers_.get(dst);
        buffer* s = device_.buffers_.get(src);
        if (!d || !s || !size) {
            device_.report_error("CopyBuffer: bad handle");
            return;
        }
        size = std::min<uint64_t>(
            {size, d->size - (std::min)(dst_offset, d->size), s->size - (std::min)(src_offset, s->size)});
        if (!size)
            return;
        [blit() copyFromBuffer:s->native
                  sourceOffset:src_offset
                      toBuffer:d->native
             destinationOffset:dst_offset
                          size:size];
    }

    void cmd_list::copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        texture* t = device_.textures_.get(dst);
        buffer* b = device_.buffers_.get(src);
        if (!t || !b) {
            device_.report_error("CopyBufferToTexture: bad handle");
            return;
        }
        for (const buffer_texture_copy& r : regions) {
            const uint32_t row_pitch = r.buffer_row_pitch ? r.buffer_row_pitch : row_bytes(t->desc.format, r.width);
            const uint32_t image_pitch = row_pitch * row_blocks(t->desc.format, r.height);
            const uint32_t depth = (std::max)(1u, r.depth);
            [blit() copyFromBuffer:b->native
                       sourceOffset:r.buffer_offset
                  sourceBytesPerRow:row_pitch
                sourceBytesPerImage:image_pitch
                         sourceSize:size(r.width, r.height, depth)
                          toTexture:t->native
                   destinationSlice:slice_of(t->desc, r.layer)
                   destinationLevel:r.mip
                  destinationOrigin:origin(r.x, r.y, r.z)];
        }
    }

    void cmd_list::copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        texture* t = device_.textures_.get(src);
        buffer* b = device_.buffers_.get(dst);
        if (!t || !b) {
            device_.report_error("CopyTextureToBuffer: bad handle");
            return;
        }
        for (const buffer_texture_copy& r : regions) {
            const uint32_t row_pitch = r.buffer_row_pitch ? r.buffer_row_pitch : row_bytes(t->desc.format, r.width);
            const uint32_t image_pitch = row_pitch * row_blocks(t->desc.format, r.height);
            [blit() copyFromTexture:t->native
                             sourceSlice:slice_of(t->desc, r.layer)
                             sourceLevel:r.mip
                            sourceOrigin:origin(r.x, r.y, r.z)
                              sourceSize:size(r.width, r.height, (std::max)(1u, r.depth))
                                toBuffer:b->native
                       destinationOffset:r.buffer_offset
                  destinationBytesPerRow:row_pitch
                destinationBytesPerImage:image_pitch];
        }
    }

    void cmd_list::copy_texture(texture_handle dst, texture_handle src, std::span<const texture_copy> regions) {
        texture* d = device_.textures_.get(dst);
        texture* s = device_.textures_.get(src);
        if (!d || !s) {
            device_.report_error("CopyTexture: bad handle");
            return;
        }
        for (const texture_copy& r : regions) {
            [blit() copyFromTexture:s->native
                        sourceSlice:slice_of(s->desc, r.src_layer)
                        sourceLevel:r.src_mip
                       sourceOrigin:origin(r.src_x, r.src_y, 0)
                         sourceSize:size(r.width, r.height, 1)
                          toTexture:d->native
                   destinationSlice:slice_of(d->desc, r.dst_layer)
                   destinationLevel:r.dst_mip
                  destinationOrigin:origin(r.dst_x, r.dst_y, 0)];
        }
    }

    void cmd_list::fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) {
        buffer* b = device_.buffers_.get(dst);
        if (!b) {
            device_.report_error("FillBuffer: bad handle");
            return;
        }
        if (!size || offset + size > b->size)
            size = b->size - (std::min)(offset, b->size);
        if (!size)
            return;
        const uint8_t byte = static_cast<uint8_t>(value & 0xFF);
        const bool uniform = value == byte * 0x01010101u;
        if (uniform) {
            [blit() fillBuffer:b->native range:NSMakeRange(offset, size) value:byte];
            return;
        }
        id<MTLBuffer> staging = [device_.dev() newBufferWithLength:size options:MTLResourceStorageModeShared];
        uint32_t* words = static_cast<uint32_t*>([staging contents]);
        for (uint64_t i = 0; i + 4 <= size; i += 4)
            words[i / 4] = value;
        [blit() copyFromBuffer:staging sourceOffset:0 toBuffer:b->native destinationOffset:offset size:size];
    }

    void cmd_list::resolve_texture(texture_handle dst, texture_handle src) {
        texture* d = device_.textures_.get(dst);
        texture* s = device_.textures_.get(src);
        if (!d || !s) {
            device_.report_error("ResolveTexture: bad handle");
            return;
        }
        end_encoder();
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        if (is_depth_format(s->desc.format)) {
            device_.report_error("ResolveTexture: depth resolve is not supported on metal");
            return;
        }
        pass.colorAttachments[0].texture = s->native;
        pass.colorAttachments[0].resolveTexture = d->native;
        pass.colorAttachments[0].loadAction = MTLLoadActionLoad;
        pass.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
        id<MTLRenderCommandEncoder> enc = [command_buffer_ renderCommandEncoderWithDescriptor:pass];
        [enc endEncoding];
    }

    void cmd_list::begin_rendering(const rendering_desc& desc) {
        end_encoder();
        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass_width_ = pass_height_ = 0;
        for (uint32_t i = 0; i < desc.color_count; i++) {
            const color_attachment& a = desc.colors[i];
            texture* t = device_.textures_.get(a.texture);
            if (!t || !t->native) {
                device_.report_error("BeginRendering: colour attachment " + std::to_string(i) + " is gone");
                return;
            }
            MTLRenderPassColorAttachmentDescriptor* ca = pass.colorAttachments[i];
            ca.texture = t->native;
            if (t->desc.dim == texture_dim::tex3d) {
                ca.depthPlane = a.layer;
            } else {
                ca.slice = a.layer;
            }
            ca.level = a.mip;
            ca.loadAction = a.load == load_op::clear  ? MTLLoadActionClear
                            : a.load == load_op::load ? MTLLoadActionLoad
                                                      : MTLLoadActionDontCare;
            ca.clearColor = MTLClearColorMake(a.clear[0], a.clear[1], a.clear[2], a.clear[3]);
            const bool store = a.store == store_op::store;
            if (a.resolve) {
                texture* r = device_.textures_.get(a.resolve);
                if (r && r->native) {
                    ca.resolveTexture = r->native;
                    ca.storeAction =
                        store ? MTLStoreActionStoreAndMultisampleResolve : MTLStoreActionMultisampleResolve;
                } else {
                    ca.storeAction = store ? MTLStoreActionStore : MTLStoreActionDontCare;
                }
            } else {
                ca.storeAction = store ? MTLStoreActionStore : MTLStoreActionDontCare;
            }
            if (!pass_width_) {
                pass_width_ = mip_extent(t->desc.width, a.mip);
                pass_height_ = mip_extent(t->desc.height, a.mip);
            }
        }
        if (desc.depth.texture) {
            texture* t = device_.textures_.get(desc.depth.texture);
            if (!t || !t->native) {
                device_.report_error("BeginRendering: depth attachment is gone");
                return;
            }
            const bool stencil = has_stencil(t->desc.format);
            const MTLLoadAction load = desc.depth.read_only                ? MTLLoadActionLoad
                                       : desc.depth.load == load_op::clear ? MTLLoadActionClear
                                       : desc.depth.load == load_op::load  ? MTLLoadActionLoad
                                                                           : MTLLoadActionDontCare;
            const MTLStoreAction store = desc.depth.read_only || desc.depth.store == store_op::store
                                             ? MTLStoreActionStore
                                             : MTLStoreActionDontCare;
            MTLRenderPassDepthAttachmentDescriptor* da = pass.depthAttachment;
            da.texture = t->native;
            da.level = desc.depth.mip;
            da.slice = desc.depth.layer;
            da.loadAction = load;
            da.storeAction = store;
            da.clearDepth = desc.depth.clear_depth;
            if (stencil) {
                MTLRenderPassStencilAttachmentDescriptor* sa = pass.stencilAttachment;
                sa.texture = t->native;
                sa.level = desc.depth.mip;
                sa.slice = desc.depth.layer;
                sa.loadAction = load;
                sa.storeAction = store;
                sa.clearStencil = desc.depth.clear_stencil;
            }
            if (!pass_width_) {
                pass_width_ = mip_extent(t->desc.width, desc.depth.mip);
                pass_height_ = mip_extent(t->desc.height, desc.depth.mip);
            }
        }
        render_ = [command_buffer_ renderCommandEncoderWithDescriptor:pass];
        if (desc.name)
            render_.label = [NSString stringWithUTF8String:desc.name];
        dirty_sets_ = 0xF;
        pipeline_dirty_ = true;
        vertex_dirty_ = 0xFF;

        const bool had_viewport = viewport_set_, had_scissor = scissor_set_;
        set_viewport(had_viewport ? viewport_
                                  : viewport{0, 0, static_cast<float>(pass_width_), static_cast<float>(pass_height_),
                                             0.0f, 1.0f});
        set_scissor(had_scissor ? scissor_ : rect{0, 0, pass_width_, pass_height_});
        viewport_set_ = had_viewport;
        scissor_set_ = had_scissor;
    }

    void cmd_list::end_rendering() {
        if (render_) {
            [render_ endEncoding];
            render_ = nil;
        }
    }

    void cmd_list::set_pipeline(pipeline_handle p) {
        pipeline_ = p;
        pipeline_dirty_ = true;
        dirty_sets_ = 0xF;
    }

    void cmd_list::set_bind_group(uint32_t set, bind_group_handle group, std::span<const uint32_t> dynamic_offsets) {
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

    void cmd_list::push_constants(const void*, uint32_t, uint32_t) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            device_.report_error("PushConstants: not implemented on metal; per-draw data belongs in a uniform buffer");
        }
    }

    void cmd_list::set_viewport(const viewport& v) {
        viewport_ = v;
        viewport_set_ = true;
        if (!render_)
            return;
        viewport c = v;
        if (pass_width_) {
            c.width = (std::min)(c.width, static_cast<float>(pass_width_) - c.x);
            c.height = (std::min)(c.height, static_cast<float>(pass_height_) - c.y);
        }
        const MTLViewport d{c.x, c.y, c.width, c.height, c.min_depth, c.max_depth};
        [render_ setViewport:d];
    }

    void cmd_list::set_scissor(const rect& r) {
        scissor_ = r;
        scissor_set_ = true;
        if (!render_)
            return;
        const int64_t left = (std::max)(r.x, 0);
        const int64_t top = (std::max)(r.y, 0);
        int64_t right = left + r.width, bottom = top + r.height;
        if (pass_width_) {
            right = std::min<int64_t>(right, pass_width_);
            bottom = std::min<int64_t>(bottom, pass_height_);
        }
        const NSUInteger x = static_cast<NSUInteger>(std::min<int64_t>(left, pass_width_ ? pass_width_ - 1 : left));
        const NSUInteger y = static_cast<NSUInteger>(std::min<int64_t>(top, pass_height_ ? pass_height_ - 1 : top));
        const MTLScissorRect d{x, y, static_cast<NSUInteger>(std::max<int64_t>(1, right - static_cast<int64_t>(x))),
                               static_cast<NSUInteger>(std::max<int64_t>(1, bottom - static_cast<int64_t>(y)))};
        [render_ setScissorRect:d];
    }

    void cmd_list::set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t offset) {
        if (slot >= max_vertex_streams)
            return;
        vertex_[slot] = {buffer, offset};
        vertex_dirty_ |= 1u << slot;
    }

    void cmd_list::set_index_buffer(buffer_handle buffer, index_type type, uint64_t offset) {
        index_buffer_ = buffer;
        index_type_ = type;
        index_offset_ = offset;
    }

    void cmd_list::set_stencil_reference(uint32_t value) {
        stencil_ref_ = value;
        if (render_)
            [render_ setStencilReferenceValue:value];
    }

    bool cmd_list::bind_sets(const pipeline& p) {
        const uint32_t first = p.compute ? stage_compute : stage_vertex;
        const uint32_t last = p.compute ? stage_compute : stage_fragment;
        for (uint32_t s = 0; s < max_bind_groups; s++) {
            if (!(dirty_sets_ & (1u << s)))
                continue;
            const bound_set& bs = sets_[s];
            const bind_group* group = device_.bind_groups_.get(bs.group);
            for (uint32_t stage = first; stage <= last; stage++) {
                for (const stage_slot& slot : p.stages[stage]) {
                    if (slot.set != s)
                        continue;
                    if (!group) {
                        device_.report_error("draw: set " + std::to_string(s) + " is not bound");
                        return false;
                    }
                    const bind_group_entry* e = nullptr;
                    for (const bind_group_entry& candidate : group->entries)
                        if (candidate.binding == slot.binding && candidate.array_index == 0) {
                            e = &candidate;
                            break;
                        }
                    if (!e) {
                        device_.report_error("bind group lacks binding " + std::to_string(slot.binding));
                        return false;
                    }
                    switch (slot.kind) {
                    case resource_kind::uniform_buffer:
                    case resource_kind::storage_buffer:
                    case resource_kind::storage_buffer_rw: {
                        buffer* b = device_.buffers_.get(e->buffer.buffer);
                        if (!b) {
                            device_.report_error("bind group buffer at binding " + std::to_string(slot.binding) +
                                                 " is gone");
                            return false;
                        }
                        uint64_t offset = e->buffer.offset;
                        if (slot.dynamic_index != ~0u && slot.dynamic_index < bs.offset_count)
                            offset += bs.offsets[slot.dynamic_index];
                        if (stage == stage_vertex)
                            [render_ setVertexBuffer:b->native offset:offset atIndex:slot.index];
                        else if (stage == stage_fragment)
                            [render_ setFragmentBuffer:b->native offset:offset atIndex:slot.index];
                        else
                            [compute_ setBuffer:b->native offset:offset atIndex:slot.index];
                        break;
                    }
                    case resource_kind::sampled_texture:
                    case resource_kind::storage_texture: {
                        texture* t = device_.textures_.get(e->texture.texture);
                        id<MTLTexture> view =
                            t ? device_.view_of(*t, e->texture.range, slot.dim, e->texture.format, e->texture.stencil)
                              : nil;
                        if (!view) {
                            device_.report_error("bind group texture at binding " + std::to_string(slot.binding) +
                                                 " is gone");
                            return false;
                        }
                        if (stage == stage_vertex)
                            [render_ setVertexTexture:view atIndex:slot.index];
                        else if (stage == stage_fragment)
                            [render_ setFragmentTexture:view atIndex:slot.index];
                        else
                            [compute_ setTexture:view atIndex:slot.index];
                        break;
                    }
                    case resource_kind::sampler: {
                        sampler* sm = device_.samplers_.get(e->sampler);
                        if (!sm) {
                            device_.report_error("bind group sampler at binding " + std::to_string(slot.binding) +
                                                 " is gone");
                            return false;
                        }
                        if (stage == stage_vertex)
                            [render_ setVertexSamplerState:sm->state atIndex:slot.index];
                        else if (stage == stage_fragment)
                            [render_ setFragmentSamplerState:sm->state atIndex:slot.index];
                        else
                            [compute_ setSamplerState:sm->state atIndex:slot.index];
                        break;
                    }
                    }
                }
            }
        }
        dirty_sets_ = 0;
        return true;
    }

    void cmd_list::apply_pipeline_state(const pipeline& p, id<MTLRenderPipelineState> state) {
        [render_ setRenderPipelineState:state];
        if (p.depth_stencil)
            [render_ setDepthStencilState:p.depth_stencil];
        [render_ setCullMode:p.cull];
        [render_ setFrontFacingWinding:p.winding];
        [render_ setTriangleFillMode:p.fill];
        [render_ setDepthClipMode:p.clip];
        [render_ setDepthBias:p.bias_constant slopeScale:p.bias_slope clamp:p.bias_clamp];
        [render_ setStencilReferenceValue:stencil_ref_];
    }

    bool cmd_list::prepare_draw() {
        if (!render_) {
            device_.report_error("draw outside a rendering pass");
            return false;
        }
        const pipeline* p = device_.pipelines_.get(pipeline_);
        if (!p || p->compute) {
            device_.report_error("draw without a graphics pipeline");
            return false;
        }
        if (pipeline_dirty_) {
            id<MTLRenderPipelineState> state = device_.ready_pipeline(*p);
            if (!state)
                return false;
            apply_pipeline_state(*p, state);
            pipeline_dirty_ = false;
            device_.mutable_stats().pipeline_binds++;
        }
        if (!bind_sets(*p))
            return false;
        for (uint32_t i = 0; i < p->stream_count; i++) {
            if (!(vertex_dirty_ & (1u << i)))
                continue;
            buffer* b = device_.buffers_.get(vertex_[i].buffer);
            if (!b) {
                device_.report_error("draw: vertex stream " + std::to_string(i) + " is not bound");
                return false;
            }
            [render_ setVertexBuffer:b->native offset:vertex_[i].offset atIndex:vertex_buffer_base - i];
        }
        vertex_dirty_ = 0;
        return true;
    }

    void cmd_list::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                        uint32_t first_instance) {
        if (!prepare_draw())
            return;
        const pipeline* p = device_.pipelines_.get(pipeline_);
        [render_ drawPrimitives:p->primitive
                    vertexStart:first_vertex
                    vertexCount:vertex_count
                  instanceCount:instance_count
                   baseInstance:first_instance];
        device_.mutable_stats().draws++;
    }

    void cmd_list::draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                                int32_t vertex_offset, uint32_t first_instance) {
        if (!prepare_draw())
            return;
        buffer* b = device_.buffers_.get(index_buffer_);
        if (!b) {
            device_.report_error("DrawIndexed: no index buffer");
            return;
        }
        const pipeline* p = device_.pipelines_.get(pipeline_);
        const uint64_t width = index_type_ == index_type::u16 ? 2 : 4;
        [render_ drawIndexedPrimitives:p->primitive
                            indexCount:index_count
                             indexType:index_type_ == index_type::u16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                           indexBuffer:b->native
                     indexBufferOffset:index_offset_ + first_index * width
                         instanceCount:instance_count
                            baseVertex:vertex_offset
                          baseInstance:first_instance];
        device_.mutable_stats().draws++;
    }

    void cmd_list::draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws, uint32_t stride) {
        if (!prepare_draw())
            return;
        buffer* a = device_.buffers_.get(args);
        buffer* ib = device_.buffers_.get(index_buffer_);
        if (!a || !ib) {
            device_.report_error("DrawIndexedIndirect: bad buffer");
            return;
        }
        const pipeline* p = device_.pipelines_.get(pipeline_);
        for (uint32_t i = 0; i < max_draws; i++)
            [render_ drawIndexedPrimitives:p->primitive
                                 indexType:index_type_ == index_type::u16 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                               indexBuffer:ib->native
                         indexBufferOffset:index_offset_
                            indirectBuffer:a->native
                      indirectBufferOffset:offset + static_cast<uint64_t>(i) * stride];
        device_.mutable_stats().indirect_draws += max_draws;
    }

    void cmd_list::draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle, uint64_t,
                                               uint32_t max_draws, uint32_t stride) {
        // the count lives on the gpu and an indirect call cannot read it: run every slot, the culling pass zeroes the unused ones
        draw_indexed_indirect(args, offset, max_draws, stride);
    }

    void cmd_list::dispatch(uint32_t x, uint32_t y, uint32_t z) {
        if (render_) {
            device_.report_error("Dispatch inside a rendering pass");
            return;
        }
        const pipeline* p = device_.pipelines_.get(pipeline_);
        if (!p || !p->compute) {
            device_.report_error("Dispatch without a compute pipeline");
            return;
        }
        id<MTLComputeCommandEncoder> enc = compute();
        if (pipeline_dirty_) {
            [enc setComputePipelineState:p->comp];
            pipeline_dirty_ = false;
        }
        if (!bind_sets(*p))
            return;
        [enc dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:p->threads];
        device_.mutable_stats().dispatches++;
    }

    void cmd_list::dispatch_indirect(buffer_handle args, uint64_t offset) {
        if (render_) {
            device_.report_error("DispatchIndirect inside a rendering pass");
            return;
        }
        const pipeline* p = device_.pipelines_.get(pipeline_);
        buffer* a = device_.buffers_.get(args);
        if (!p || !p->compute || !a) {
            device_.report_error("DispatchIndirect: bad pipeline or buffer");
            return;
        }
        id<MTLComputeCommandEncoder> enc = compute();
        if (pipeline_dirty_) {
            [enc setComputePipelineState:p->comp];
            pipeline_dirty_ = false;
        }
        if (!bind_sets(*p))
            return;
        [enc dispatchThreadgroupsWithIndirectBuffer:a->native
                               indirectBufferOffset:offset
                              threadsPerThreadgroup:p->threads];
        device_.mutable_stats().dispatches++;
    }

    void cmd_list::write_timestamp(query_set_handle, uint32_t) {}
    void cmd_list::resolve_queries(query_set_handle, uint32_t, uint32_t, buffer_handle, uint64_t) {}

    void cmd_list::begin_debug_label(const char*, uint32_t) {}
    void cmd_list::end_debug_label() {}
}
