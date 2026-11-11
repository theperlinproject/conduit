#include "d3d12/d3d12_device.hpp"

#include "d3d12/d3d12_convert.hpp"

#include <algorithm>
#include <cstring>

namespace conduit::d3d12 {
    namespace {
        D3D12_RESOURCE_BARRIER make_transition(ID3D12Resource* resource, UINT subresource, D3D12_RESOURCE_STATES before,
                                               D3D12_RESOURCE_STATES after) {
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition.pResource = resource;
            b.Transition.Subresource = subresource;
            b.Transition.StateBefore = before;
            b.Transition.StateAfter = after;
            return b;
        }

        void transition_texture(std::vector<D3D12_RESOURCE_BARRIER>& out, texture_rec& t, const subresource_range& r,
                                D3D12_RESOURCE_STATES after) {
            if (t.simultaneous)
                return;
            const bool whole =
                r.base_mip == 0 && r.mip_count == t.mips && r.base_layer == 0 && r.layer_count == t.layers;
            if (whole && std::all_of(t.states.begin(), t.states.end(),
                                     [&](D3D12_RESOURCE_STATES s) { return s == t.states[0]; })) {
                if (t.states[0] != after) {
                    out.push_back(
                        make_transition(t.resource.get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, t.states[0], after));
                    std::fill(t.states.begin(), t.states.end(), after);
                }
                return;
            }
            for (uint32_t plane = 0; plane < t.planes; plane++)
                for (uint32_t layer = r.base_layer; layer < r.base_layer + r.layer_count; layer++)
                    for (uint32_t mip = r.base_mip; mip < r.base_mip + r.mip_count; mip++) {
                        const UINT sub = subresource_index(mip, layer, plane, t.mips, t.layers);
                        D3D12_RESOURCE_STATES& cur = t.states[sub];
                        if (cur == after)
                            continue;
                        out.push_back(make_transition(t.resource.get(), sub, cur, after));
                        cur = after;
                    }
        }

        void transition_buffer(std::vector<D3D12_RESOURCE_BARRIER>& out, buffer_rec& b, D3D12_RESOURCE_STATES after) {
            if (b.state == after)
                return;
            out.push_back(make_transition(b.resource.get(), 0, b.state, after));
            b.state = after;
        }

        subresource_range whole_texture(const texture_rec& t) {
            return {0, t.mips, 0, t.layers};
        }
    }

    void cmd_list::fail(const std::string& message) {
        device_.report_error(message);
    }

    void cmd_list::transition(ID3D12Resource* resource, UINT subresource, D3D12_RESOURCE_STATES before,
                              D3D12_RESOURCE_STATES after) {
        const D3D12_RESOURCE_BARRIER b = make_transition(resource, subresource, before, after);
        list_->ResourceBarrier(1, &b);
    }

    void cmd_list::begin() {
        if (recording_)
            list_->Close();
        temps_.clear();
        list_->Reset(allocator_.get(), nullptr);
        recording_ = true;

        pipeline_ = {};
        layout_handle_ = {};
        have_pipeline_ = false;
        pipeline_pending_ = false;
        in_rendering_ = false;
        groups_ = {};
        dynamic_offsets_ = {};
        dynamic_counts_ = {};
        dirty_sets_ = 0;
        push_.fill(0);
        push_dirty_ = false;
        push_used_ = 0;
        color_formats_.fill(format::unknown);
        color_count_ = 0;
        depth_format_ = format::unknown;
        have_index_ = false;
        vertex_mask_ = 0;
        vertex_dirty_ = 0;
        vbuffers_ = {};
        index_view_ = {};
        pending_resolves_.clear();
        heaps_set_ = false;
        invalid_ = false;
        bound_root_ = nullptr;
        bound_compute_ = false;
        have_viewport_ = have_scissor_ = false;
        stencil_ref_ = 0;
        open_labels_ = 0;
        counters_ = {};
    }

    void cmd_list::end() {
        if (!recording_)
            return;
        if (in_rendering_) {
            fail("command list ended inside a rendering pass");
            end_rendering();
        }
        list_->Close();
        recording_ = false;
    }

    void cmd_list::barrier(std::span<const texture_barrier> textures, std::span<const buffer_barrier> buffers) {
        if (in_rendering_)
            fail("barrier recorded inside a rendering pass");
        std::vector<D3D12_RESOURCE_BARRIER>& out = barrier_scratch_;
        out.clear();

        for (const texture_barrier& tb : textures) {
            texture_rec* tex = device_.textures_.get(tb.texture);
            if (!tex) {
                fail("barrier on an unknown texture");
                continue;
            }
            if (device_.use_validator_)
                device_.validator_.barrier(tb);
            const bool handoff = tb.src_queue != tb.dst_queue;
            // queue handoff goes through common: the releasing list parks the resource there, the acquiring one moves it on
            const D3D12_RESOURCE_STATES target =
                handoff && queue_ == tb.src_queue ? D3D12_RESOURCE_STATE_COMMON : to_states(tb.after, queue_);
            transition_texture(out, *tex, resolve_range(tb.range, tex->desc), target);
        }

        for (const buffer_barrier& bb : buffers) {
            buffer_rec* buf = device_.buffers_.get(bb.buffer);
            if (!buf) {
                fail("barrier on an unknown buffer");
                continue;
            }
            if (device_.use_validator_)
                device_.validator_.barrier(bb);
            if (buf->desc.memory == memory_class::upload || buf->desc.memory == memory_class::readback ||
                buf->desc.cross_queue)
                continue;
            const bool handoff = bb.src_queue != bb.dst_queue;
            const D3D12_RESOURCE_STATES target =
                handoff && queue_ == bb.src_queue ? D3D12_RESOURCE_STATE_COMMON : to_states(bb.after, queue_);
            transition_buffer(out, *buf, target);
        }

        if (out.empty())
            return;
        list_->ResourceBarrier(static_cast<UINT>(out.size()), out.data());
        counters_.barriers += out.size();
    }

    void cmd_list::uav_barrier(buffer_handle buffer) {
        buffer_rec* b = device_.buffers_.get(buffer);
        if (!b) {
            fail("UavBarrier: unknown buffer");
            return;
        }
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = b->resource.get();
        list_->ResourceBarrier(1, &barrier);
        counters_.barriers++;
    }

    void cmd_list::uav_barrier(texture_handle texture) {
        texture_rec* t = device_.textures_.get(texture);
        if (!t) {
            fail("UavBarrier: unknown texture");
            return;
        }
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        barrier.UAV.pResource = t->resource.get();
        list_->ResourceBarrier(1, &barrier);
        counters_.barriers++;
    }

    void cmd_list::copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src, uint64_t src_offset,
                               uint64_t size) {
        buffer_rec* d = device_.buffers_.get(dst);
        buffer_rec* s = device_.buffers_.get(src);
        if (!d || !s) {
            fail("CopyBuffer: unknown buffer");
            return;
        }
        if (device_.use_validator_) {
            device_.validator_.expect_buffer(src, state::copy_src, "CopyBuffer src");
            device_.validator_.expect_buffer(dst, state::copy_dst, "CopyBuffer dst");
        }
        list_->CopyBufferRegion(d->resource.get(), dst_offset, s->resource.get(), src_offset, size);
    }

    namespace {
        bool footprint(const texture_rec& t, const buffer_texture_copy& r, D3D12_PLACED_SUBRESOURCE_FOOTPRINT& out,
                       std::string& why) {
            uint32_t pitch = r.buffer_row_pitch ? r.buffer_row_pitch : row_bytes(t.desc.format, r.width);
            if (pitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT != 0) {
                why = "row pitch " + std::to_string(pitch) +
                      " is not a multiple of 256 (caps.limits.copyRowPitchAlignment)";
                return false;
            }
            if (r.buffer_offset % D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT != 0) {
                why = "buffer offset is not a multiple of 512 (caps.limits.copyOffsetAlignment)";
                return false;
            }
            out.Offset = r.buffer_offset;
            out.Footprint.Format = t.view_format != DXGI_FORMAT_UNKNOWN ? t.view_format : t.resource_format;
            out.Footprint.Width = r.width;
            out.Footprint.Height = r.height;
            out.Footprint.Depth = (std::max)(1u, r.depth);
            out.Footprint.RowPitch = pitch;
            return true;
        }
    }

    void cmd_list::copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        texture_rec* t = device_.textures_.get(dst);
        buffer_rec* b = device_.buffers_.get(src);
        if (!t || !b) {
            fail("CopyBufferToTexture: unknown resource");
            return;
        }
        for (const buffer_texture_copy& r : regions) {
            if (device_.use_validator_) {
                device_.validator_.expect_buffer(src, state::copy_src, "CopyBufferToTexture src");
                device_.validator_.expect_texture(dst, {r.mip, 1, r.layer, 1}, state::copy_dst,
                                                  "CopyBufferToTexture dst");
            }
            D3D12_TEXTURE_COPY_LOCATION s{};
            s.pResource = b->resource.get();
            s.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            std::string why;
            if (!footprint(*t, r, s.PlacedFootprint, why)) {
                fail("CopyBufferToTexture: " + why);
                continue;
            }
            D3D12_TEXTURE_COPY_LOCATION d{};
            d.pResource = t->resource.get();
            d.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            d.SubresourceIndex = subresource_index(r.mip, r.layer, 0, t->mips, t->layers);
            list_->CopyTextureRegion(&d, r.x, r.y, r.z, &s, nullptr);
        }
    }

    void cmd_list::copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        texture_rec* t = device_.textures_.get(src);
        buffer_rec* b = device_.buffers_.get(dst);
        if (!t || !b) {
            fail("CopyTextureToBuffer: unknown resource");
            return;
        }
        for (const buffer_texture_copy& r : regions) {
            if (device_.use_validator_) {
                device_.validator_.expect_buffer(dst, state::copy_dst, "CopyTextureToBuffer dst");
                device_.validator_.expect_texture(src, {r.mip, 1, r.layer, 1}, state::copy_src,
                                                  "CopyTextureToBuffer src");
            }
            D3D12_TEXTURE_COPY_LOCATION d{};
            d.pResource = b->resource.get();
            d.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            std::string why;
            if (!footprint(*t, r, d.PlacedFootprint, why)) {
                fail("CopyTextureToBuffer: " + why);
                continue;
            }
            D3D12_TEXTURE_COPY_LOCATION s{};
            s.pResource = t->resource.get();
            s.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            s.SubresourceIndex = subresource_index(r.mip, r.layer, 0, t->mips, t->layers);
            const D3D12_BOX box{r.x, r.y, r.z, r.x + r.width, r.y + r.height, r.z + (std::max)(1u, r.depth)};
            list_->CopyTextureRegion(&d, 0, 0, 0, &s, &box);
        }
    }

    void cmd_list::copy_texture(texture_handle dst, texture_handle src, std::span<const texture_copy> regions) {
        texture_rec* d = device_.textures_.get(dst);
        texture_rec* s = device_.textures_.get(src);
        if (!d || !s) {
            fail("CopyTexture: unknown texture");
            return;
        }
        for (const texture_copy& r : regions) {
            if (device_.use_validator_) {
                device_.validator_.expect_texture(src, {r.src_mip, 1, r.src_layer, 1}, state::copy_src,
                                                  "CopyTexture src");
                device_.validator_.expect_texture(dst, {r.dst_mip, 1, r.dst_layer, 1}, state::copy_dst,
                                                  "CopyTexture dst");
            }
            D3D12_TEXTURE_COPY_LOCATION dl{};
            dl.pResource = d->resource.get();
            dl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            dl.SubresourceIndex = subresource_index(r.dst_mip, r.dst_layer, 0, d->mips, d->layers);
            D3D12_TEXTURE_COPY_LOCATION sl{};
            sl.pResource = s->resource.get();
            sl.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            sl.SubresourceIndex = subresource_index(r.src_mip, r.src_layer, 0, s->mips, s->layers);
            const D3D12_BOX box{r.src_x, r.src_y, 0, r.src_x + r.width, r.src_y + r.height, 1};
            list_->CopyTextureRegion(&dl, r.dst_x, r.dst_y, 0, &sl, &box);
        }
    }

    void cmd_list::fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) {
        buffer_rec* b = device_.buffers_.get(dst);
        if (!b) {
            fail("FillBuffer: unknown buffer");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(dst, state::copy_dst, "FillBuffer dst");
        if (size == ~0ull || size == 0)
            size = b->desc.size - offset;

        // no fill command in d3d12: stage the pattern in an upload buffer and copy it
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = size;
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12MA::ALLOCATION_DESC ad{};
        ad.HeapType = D3D12_HEAP_TYPE_UPLOAD;
        winrt::com_ptr<ID3D12Resource> res;
        winrt::com_ptr<D3D12MA::Allocation> alloc;
        if (FAILED(device_.allocator_->CreateResource(&ad, &rd, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, alloc.put(),
                                                      winrt::guid_of<ID3D12Resource>(), res.put_void()))) {
            fail("FillBuffer: staging allocation failed");
            return;
        }
        void* mapped = nullptr;
        D3D12_RANGE none{0, 0};
        if (FAILED(res->Map(0, &none, &mapped))) {
            fail("FillBuffer: staging map failed");
            return;
        }
        auto* words = static_cast<uint32_t*>(mapped);
        std::fill(words, words + size / 4, value);
        res->Unmap(0, nullptr);
        list_->CopyBufferRegion(b->resource.get(), offset, res.get(), 0, size);
        temps_.emplace_back(std::move(res), std::move(alloc));
    }

    void cmd_list::resolve_texture(texture_handle dst, texture_handle src) {
        texture_rec* d = device_.textures_.get(dst);
        texture_rec* s = device_.textures_.get(src);
        if (!d || !s) {
            fail("ResolveTexture: unknown texture");
            return;
        }
        if (device_.use_validator_) {
            device_.validator_.expect_texture(src, {}, state::resolve_src, "ResolveTexture src");
            device_.validator_.expect_texture(dst, {}, state::resolve_dst, "ResolveTexture dst");
        }
        list_->ResolveSubresource(d->resource.get(), 0, s->resource.get(), 0, d->view_format);
    }

    void cmd_list::begin_rendering(const rendering_desc& desc) {
        if (in_rendering_)
            fail("BeginRendering inside a rendering pass");
        D3D12_CPU_DESCRIPTOR_HANDLE rtvs[max_color_targets]{};
        D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
        bool have_depth = false;
        bool attachment_missing = false;
        pass_broken_ = false;
        target_width_ = target_height_ = 0;
        color_count_ = desc.color_count;
        color_formats_.fill(format::unknown);
        depth_format_ = format::unknown;
        pending_resolves_.clear();

        for (uint32_t i = 0; i < desc.color_count; i++) {
            const color_attachment& a = desc.colors[i];
            texture_rec* t = device_.textures_.get(a.texture);
            if (!t) {
                fail("BeginRendering: unknown colour attachment");
                attachment_missing = true;
                continue;
            }
            if (!has(t->desc.usage, texture_usage::color_target))
                fail("BeginRendering: '" + t->name + "' lacks ColorTarget usage");
            if (device_.use_validator_)
                device_.validator_.expect_texture(a.texture, {a.mip, 1, a.layer, 1}, state::color_target,
                                                  "BeginRendering colour");
            rtvs[i] = device_.render_view(*t, a.mip, a.layer, false);
            if (a.load == load_op::clear)
                list_->ClearRenderTargetView(rtvs[i], a.clear.data(), 0, nullptr);
            if (a.resolve)
                pending_resolves_.push_back({a.texture, a.resolve});
            color_formats_[i] = t->desc.format;
            if (!target_width_) {
                target_width_ = mip_extent(t->desc.width, a.mip);
                target_height_ = mip_extent(t->desc.height, a.mip);
            }
        }

        if (desc.depth.texture) {
            const depth_attachment& a = desc.depth;
            texture_rec* t = device_.textures_.get(a.texture);
            if (!t) {
                fail("BeginRendering: unknown depth attachment");
                attachment_missing = true;
            } else {
                if (!has(t->desc.usage, texture_usage::depth_target))
                    fail("BeginRendering: '" + t->name + "' lacks DepthTarget usage");
                if (device_.use_validator_)
                    device_.validator_.expect_texture(a.texture, {a.mip, 1, a.layer, 1},
                                                      a.read_only ? state::depth_read : state::depth_write,
                                                      "BeginRendering depth");
                dsv = device_.render_view(*t, a.mip, a.layer, true, a.read_only);
                have_depth = true;
                if (!a.read_only && a.load == load_op::clear) {
                    D3D12_CLEAR_FLAGS flags = D3D12_CLEAR_FLAG_DEPTH;
                    if (has_stencil(t->desc.format))
                        flags |= D3D12_CLEAR_FLAG_STENCIL;
                    list_->ClearDepthStencilView(dsv, flags, a.clear_depth, a.clear_stencil, 0, nullptr);
                }
                depth_format_ = t->desc.format;
                if (!target_width_) {
                    target_width_ = mip_extent(t->desc.width, a.mip);
                    target_height_ = mip_extent(t->desc.height, a.mip);
                }
            }
        }

        if (attachment_missing) {
            pass_broken_ = true;
            in_rendering_ = true;
            have_pipeline_ = false;
            return;
        }
        list_->OMSetRenderTargets(desc.color_count, rtvs, FALSE, have_depth ? &dsv : nullptr);
        in_rendering_ = true;

        set_viewport({0, 0, static_cast<float>(target_width_), static_cast<float>(target_height_), 0.0f, 1.0f});
        set_scissor({0, 0, target_width_, target_height_});
        have_pipeline_ = false;
        dirty_sets_ = 0;
        for (uint32_t i = 0; i < max_bind_groups; i++)
            if (groups_[i])
                dirty_sets_ |= 1u << i;
        push_dirty_ = push_used_ != 0;
        vertex_dirty_ = vertex_mask_;
    }

    void cmd_list::end_rendering() {
        if (!in_rendering_) {
            fail("EndRendering without BeginRendering");
            return;
        }
        in_rendering_ = false;
        if (pass_broken_) {
            pass_broken_ = false;
            pending_resolves_.clear();
            return;
        }

        for (const resolve_pair& p : pending_resolves_) {
            texture_rec* s = device_.textures_.get(p.src);
            texture_rec* d = device_.textures_.get(p.dst);
            if (!s || !d)
                continue;
            const std::vector<D3D12_RESOURCE_STATES> saved_src = s->states;
            const std::vector<D3D12_RESOURCE_STATES> saved_dst = d->states;
            std::vector<D3D12_RESOURCE_BARRIER> in, back;
            transition_texture(in, *s, whole_texture(*s), D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            transition_texture(in, *d, whole_texture(*d), D3D12_RESOURCE_STATE_RESOLVE_DEST);
            if (!in.empty())
                list_->ResourceBarrier(static_cast<UINT>(in.size()), in.data());
            list_->ResolveSubresource(d->resource.get(), 0, s->resource.get(), 0, d->view_format);
            for (size_t i = 0; i < saved_src.size(); i++)
                if (s->states[i] != saved_src[i]) {
                    back.push_back(
                        make_transition(s->resource.get(), static_cast<UINT>(i), s->states[i], saved_src[i]));
                    s->states[i] = saved_src[i];
                }
            for (size_t i = 0; i < saved_dst.size(); i++)
                if (d->states[i] != saved_dst[i]) {
                    back.push_back(
                        make_transition(d->resource.get(), static_cast<UINT>(i), d->states[i], saved_dst[i]));
                    d->states[i] = saved_dst[i];
                }
            if (!back.empty())
                list_->ResourceBarrier(static_cast<UINT>(back.size()), back.data());
        }
        pending_resolves_.clear();
    }

    void cmd_list::set_pipeline(pipeline_handle handle) {
        pipeline_rec* p = device_.pipelines_.get(handle);
        if (!p) {
            fail("SetPipeline: unknown pipeline");
            return;
        }
        if (pass_broken_)
            return;
        if (p->compute == in_rendering_)
            fail(p->compute ? "compute pipeline set inside a rendering pass"
                            : "graphics pipeline set outside a rendering pass");
        const winrt::com_ptr<ID3D12PipelineState> pso = device_.ready_pipeline(*p);
        if (!pso && !p->graph) {
            // still compiling (or failed): skip draws until the next pipeline rather than stall the frame
            have_pipeline_ = false;
            pipeline_pending_ = true;
            return;
        }
        pipeline_pending_ = false;
        if (have_pipeline_ && !invalid_ && pipeline_.pso.get() == pso.get() && pipeline_.graph.get() == p->graph.get())
            return;
        pipeline_ = *p;
        pipeline_.pso = pso;
        pipeline_.build.reset();
        layout_handle_ = p->layout_handle;
        have_pipeline_ = true;
        if (pso)
            list_->SetPipelineState(pso.get());
        if (!p->compute)
            list_->IASetPrimitiveTopology(p->topology);
        if (bound_root_ != p->root_signature.get() || bound_compute_ != p->compute) {
            if (!heaps_set_ && queue_ != queue_type::copy) {
                ID3D12DescriptorHeap* heaps[2] = {device_.resource_heap_.heap(), device_.sampler_heap_.heap()};
                list_->SetDescriptorHeaps(2, heaps);
                heaps_set_ = true;
            }
            if (p->compute)
                list_->SetComputeRootSignature(p->root_signature.get());
            else
                list_->SetGraphicsRootSignature(p->root_signature.get());
            bound_root_ = p->root_signature.get();
            bound_compute_ = p->compute;
            for (uint32_t i = 0; i < max_bind_groups; i++)
                if (groups_[i])
                    dirty_sets_ |= 1u << i;
            push_dirty_ = push_used_ != 0;
        }
        vertex_dirty_ = vertex_mask_;
        counters_.pipeline_binds++;
    }

    void cmd_list::set_bind_group(uint32_t set, bind_group_handle group, std::span<const uint32_t> offsets) {
        bind_group_rec* g = device_.bind_groups_.get(group);
        if (!g || set >= max_bind_groups) {
            fail("SetBindGroup: unknown group or set index");
            return;
        }
        if (offsets.size() != g->dynamic_count)
            fail("SetBindGroup: expected " + std::to_string(g->dynamic_count) + " dynamic offsets, got " +
                 std::to_string(offsets.size()));
        if (device_.use_validator_)
            device_.validator_.expect_all(g->bound, "SetBindGroup");
        const uint32_t count = std::min<uint32_t>(static_cast<uint32_t>(offsets.size()), max_dynamic_offsets);
        if (groups_[set] == group && dynamic_counts_[set] == count &&
            std::equal(offsets.begin(), offsets.begin() + count, dynamic_offsets_[set].begin()))
            return;
        groups_[set] = group;
        dynamic_counts_[set] = count;
        std::copy_n(offsets.begin(), count, dynamic_offsets_[set].begin());
        dirty_sets_ |= 1u << set;
        counters_.group_binds++;
    }

    void cmd_list::push_constants(const void* data, uint32_t bytes, uint32_t offset) {
        if (offset + bytes > max_push_constant_bytes) {
            fail("PushConstants exceed the limit");
            return;
        }
        std::memcpy(reinterpret_cast<uint8_t*>(push_.data()) + offset, data, bytes);
        push_used_ = (std::max)(push_used_, offset + bytes);
        push_dirty_ = true;
    }

    void cmd_list::set_viewport(const viewport& v) {
        viewport_ = {v.x, v.y, v.width, v.height, v.min_depth, v.max_depth};
        have_viewport_ = true;
        list_->RSSetViewports(1, &viewport_);
    }

    void cmd_list::set_scissor(const rect& r) {
        scissor_ = {r.x, r.y, r.x + static_cast<LONG>(r.width), r.y + static_cast<LONG>(r.height)};
        have_scissor_ = true;
        list_->RSSetScissorRects(1, &scissor_);
    }

    void cmd_list::set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t offset) {
        buffer_rec* b = device_.buffers_.get(buffer);
        if (!b || slot >= max_vertex_streams) {
            fail("SetVertexBuffer: unknown buffer or slot");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(buffer, state::vertex_buffer, "SetVertexBuffer");
        vbuffers_[slot].BufferLocation = b->address + offset;
        vbuffers_[slot].SizeInBytes = static_cast<UINT>(b->desc.size - offset);
        vertex_mask_ |= 1u << slot;
        vertex_dirty_ |= 1u << slot;
    }

    void cmd_list::set_index_buffer(buffer_handle buffer, index_type type, uint64_t offset) {
        buffer_rec* b = device_.buffers_.get(buffer);
        if (!b) {
            fail("SetIndexBuffer: unknown buffer");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(buffer, state::index_buffer, "SetIndexBuffer");
        index_view_.BufferLocation = b->address + offset;
        index_view_.SizeInBytes = static_cast<UINT>(b->desc.size - offset);
        index_view_.Format = type == index_type::u16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT;
        have_index_ = true;
        list_->IASetIndexBuffer(&index_view_);
    }

    void cmd_list::set_stencil_reference(uint32_t value) {
        stencil_ref_ = value;
        list_->OMSetStencilRef(value);
    }

    void cmd_list::invalidate_bound_state() {
        invalid_ = true;
        heaps_set_ = false;
        bound_root_ = nullptr;
        for (uint32_t i = 0; i < max_bind_groups; i++)
            if (groups_[i])
                dirty_sets_ |= 1u << i;
        push_dirty_ = push_used_ != 0;
        vertex_dirty_ = vertex_mask_;
    }

    void cmd_list::flush_bindings() {
        if (!have_pipeline_)
            return;
        if (invalid_) {
            invalid_ = false;
            if (pipeline_.pso)
                list_->SetPipelineState(pipeline_.pso.get());
            if (!heaps_set_ && queue_ != queue_type::copy) {
                ID3D12DescriptorHeap* heaps[2] = {device_.resource_heap_.heap(), device_.sampler_heap_.heap()};
                list_->SetDescriptorHeaps(2, heaps);
                heaps_set_ = true;
            }
            if (pipeline_.compute)
                list_->SetComputeRootSignature(pipeline_.root_signature.get());
            else
                list_->SetGraphicsRootSignature(pipeline_.root_signature.get());
            bound_root_ = pipeline_.root_signature.get();
            bound_compute_ = pipeline_.compute;
            if (in_rendering_) {
                list_->IASetPrimitiveTopology(pipeline_.topology);
                if (have_viewport_)
                    list_->RSSetViewports(1, &viewport_);
                if (have_scissor_)
                    list_->RSSetScissorRects(1, &scissor_);
                list_->OMSetStencilRef(stencil_ref_);
                if (have_index_)
                    list_->IASetIndexBuffer(&index_view_);
            }
        }

        if (vertex_dirty_ && !pipeline_.compute) {
            for (uint32_t i = 0; i < max_vertex_streams; i++) {
                if (!(vertex_dirty_ & (1u << i)))
                    continue;
                vbuffers_[i].StrideInBytes = pipeline_.strides[i];
                list_->IASetVertexBuffers(i, 1, &vbuffers_[i]);
            }
            vertex_dirty_ = 0;
        }

        pipeline_layout_rec* layout = device_.pipeline_layouts_.get(layout_handle_);
        if (!layout)
            return;
        if (dirty_sets_) {
            for (uint32_t i = 0; i < max_bind_groups; i++) {
                if (!(dirty_sets_ & (1u << i)))
                    continue;
                if (!layout->sets[i])
                    continue;
                bind_group_rec* g = device_.bind_groups_.get(groups_[i]);
                if (!g) {
                    fail("draw with a destroyed or missing bind group at set " + std::to_string(i));
                    continue;
                }
                if (layout->table_param[i] >= 0) {
                    const auto gpu = device_.resource_heap_.gpu(g->table_offset);
                    if (pipeline_.compute)
                        list_->SetComputeRootDescriptorTable(layout->table_param[i], gpu);
                    else
                        list_->SetGraphicsRootDescriptorTable(layout->table_param[i], gpu);
                }
                if (layout->sampler_param[i] >= 0) {
                    const auto gpu = device_.sampler_heap_.gpu(g->sampler_offset);
                    if (pipeline_.compute)
                        list_->SetComputeRootDescriptorTable(layout->sampler_param[i], gpu);
                    else
                        list_->SetGraphicsRootDescriptorTable(layout->sampler_param[i], gpu);
                }
                for (uint32_t k = 0; k < dynamic_counts_[i] && k < g->dynamic_count; k++) {
                    const int param = layout->dynamic_param[i][k];
                    if (param < 0)
                        continue;
                    const D3D12_GPU_VIRTUAL_ADDRESS address = g->dynamic_base[k] + dynamic_offsets_[i][k];
                    if (pipeline_.compute)
                        list_->SetComputeRootConstantBufferView(param, address);
                    else
                        list_->SetGraphicsRootConstantBufferView(param, address);
                }
            }
            dirty_sets_ = 0;
        }
        if (push_dirty_ && layout->push_param >= 0) {
            const UINT words = layout->push_bytes / 4;
            if (pipeline_.compute)
                list_->SetComputeRoot32BitConstants(layout->push_param, words, push_.data(), 0);
            else
                list_->SetGraphicsRoot32BitConstants(layout->push_param, words, push_.data(), 0);
            push_dirty_ = false;
        }
    }

    void cmd_list::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                        uint32_t first_instance) {
        if (pipeline_pending_)
            return;
        if (!have_pipeline_ || !in_rendering_) {
            fail("Draw without a pipeline or outside a rendering pass");
            return;
        }
        flush_bindings();
        list_->DrawInstanced(vertex_count, instance_count, first_vertex, first_instance);
        counters_.draws++;
    }

    void cmd_list::draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                                int32_t vertex_offset, uint32_t first_instance) {
        if (pipeline_pending_)
            return;
        if (!have_pipeline_ || !in_rendering_ || !have_index_) {
            fail("DrawIndexed without a pipeline, index buffer or rendering pass");
            return;
        }
        flush_bindings();
        list_->DrawIndexedInstanced(index_count, instance_count, first_index, vertex_offset, first_instance);
        counters_.draws++;
    }

    void cmd_list::draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws, uint32_t stride) {
        if (pipeline_pending_)
            return;
        buffer_rec* a = device_.buffers_.get(args);
        if (!a || !have_pipeline_ || !in_rendering_ || !have_index_) {
            fail("DrawIndexedIndirect: missing buffer, pipeline, index buffer or rendering pass");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(args, state::indirect_argument, "DrawIndexedIndirect args");
        ID3D12CommandSignature* sig = device_.signature(false, stride);
        if (!sig) {
            fail("DrawIndexedIndirect: command signature creation failed");
            return;
        }
        flush_bindings();
        list_->ExecuteIndirect(sig, max_draws, a->resource.get(), offset, nullptr, 0);
        counters_.draws += max_draws;
        counters_.indirect++;
    }

    void cmd_list::draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle count_buffer,
                                               uint64_t count_offset, uint32_t max_draws, uint32_t stride) {
        if (pipeline_pending_)
            return;
        buffer_rec* a = device_.buffers_.get(args);
        buffer_rec* c = device_.buffers_.get(count_buffer);
        if (!a || !c || !have_pipeline_ || !in_rendering_ || !have_index_) {
            fail("DrawIndexedIndirectCount: missing buffer, pipeline, index buffer or rendering pass");
            return;
        }
        if (device_.use_validator_) {
            device_.validator_.expect_buffer(args, state::indirect_argument, "DrawIndexedIndirectCount args");
            device_.validator_.expect_buffer(count_buffer, state::indirect_argument, "DrawIndexedIndirectCount count");
        }
        ID3D12CommandSignature* sig = device_.signature(false, stride);
        if (!sig) {
            fail("DrawIndexedIndirectCount: command signature creation failed");
            return;
        }
        flush_bindings();
        list_->ExecuteIndirect(sig, max_draws, a->resource.get(), offset, c->resource.get(), count_offset);
        counters_.indirect++;
    }

    void cmd_list::dispatch(uint32_t x, uint32_t y, uint32_t z) {
        if (!have_pipeline_ || in_rendering_) {
            fail("Dispatch without a compute pipeline or inside a rendering pass");
            return;
        }
        flush_bindings();
        list_->Dispatch(x, y, z);
        counters_.dispatches++;
    }

    void cmd_list::dispatch_graph(const void* records, uint32_t count, uint32_t stride) {
        if (!have_pipeline_ || !pipeline_.graph || in_rendering_) {
            fail("DispatchGraph without a work graph or inside a rendering pass");
            return;
        }
        const auto list10 = list_.try_as<ID3D12GraphicsCommandList10>();
        if (!list10) {
            fail("DispatchGraph: the command list has no ID3D12GraphicsCommandList10 (old runtime)");
            return;
        }
        flush_bindings();
        D3D12_SET_PROGRAM_DESC sp{};
        sp.Type = D3D12_PROGRAM_TYPE_WORK_GRAPH;
        sp.WorkGraph.ProgramIdentifier = pipeline_.program;
        // the backing memory is initialized by the first program set that uses it, later sets keep its state
        const bool first = pipeline_.backing_ready && !pipeline_.backing_ready->exchange(true);
        sp.WorkGraph.Flags = first ? D3D12_SET_WORK_GRAPH_FLAG_INITIALIZE : D3D12_SET_WORK_GRAPH_FLAG_NONE;
        sp.WorkGraph.BackingMemory = pipeline_.backing_range;
        list10->SetProgram(&sp);
        D3D12_DISPATCH_GRAPH_DESC dg{};
        dg.Mode = D3D12_DISPATCH_MODE_NODE_CPU_INPUT;
        dg.NodeCPUInput.EntrypointIndex = 0;
        dg.NodeCPUInput.NumRecords = count;
        dg.NodeCPUInput.pRecords = records;
        dg.NodeCPUInput.RecordStrideInBytes = stride;
        list10->DispatchGraph(&dg);
        // SetProgram replaced the pipeline state: the next set_pipeline has to bind again
        have_pipeline_ = false;
        counters_.dispatches++;
    }

    void cmd_list::dispatch_indirect(buffer_handle args, uint64_t offset) {
        buffer_rec* a = device_.buffers_.get(args);
        if (!a || !have_pipeline_ || in_rendering_) {
            fail("DispatchIndirect: missing buffer or compute pipeline");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(args, state::indirect_argument, "DispatchIndirect args");
        ID3D12CommandSignature* sig = device_.signature(true, 0);
        if (!sig) {
            fail("DispatchIndirect: command signature creation failed");
            return;
        }
        flush_bindings();
        list_->ExecuteIndirect(sig, 1, a->resource.get(), offset, nullptr, 0);
        counters_.dispatches++;
    }

    void cmd_list::write_timestamp(query_set_handle set, uint32_t index) {
        query_rec* q = device_.queries_.get(set);
        if (!q || index >= q->count) {
            fail("WriteTimestamp: bad query set or index");
            return;
        }
        list_->EndQuery(q->heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
    }

    void cmd_list::resolve_queries(query_set_handle set, uint32_t first, uint32_t count, buffer_handle dst,
                                   uint64_t dst_offset) {
        query_rec* q = device_.queries_.get(set);
        buffer_rec* b = device_.buffers_.get(dst);
        if (!q || !b || first + count > q->count) {
            fail("ResolveQueries: bad query set or buffer");
            return;
        }
        list_->ResolveQueryData(q->heap.get(), D3D12_QUERY_TYPE_TIMESTAMP, first, count, b->resource.get(), dst_offset);
    }

    void cmd_list::begin_debug_label(const char* name, uint32_t) {
        open_labels_++;
        list_->BeginEvent(1, name, static_cast<UINT>(std::strlen(name) + 1));
    }

    void cmd_list::end_debug_label() {
        if (--open_labels_ < 0) {
            fail("EndDebugLabel without BeginDebugLabel");
            open_labels_ = 0;
            return;
        }
        list_->EndEvent();
    }
}
