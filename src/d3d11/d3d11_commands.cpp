#include "d3d11/d3d11_internal.hpp"

#include <algorithm>
#include <bit>
#include <cstring>

namespace conduit::d3d11 {
    namespace {
        constexpr uint32_t max_srv_slots = 128;
        constexpr uint64_t constant_window = 4096ull * 16;

        UINT subresource(const texture_desc& d, uint32_t mip, uint32_t layer) {
            return mip + layer * (std::max)(1u, d.mips);
        }
    }

    cmd_list::cmd_list(d3d11::device& device, queue_type queue) : device_(device), queue_(queue) {}

    cmd_list::~cmd_list() = default;

    bool cmd_list::init() {
        if (FAILED(device_.dev()->CreateDeferredContext(0, ctx_.put())))
            return false;
        ctx1_ = ctx_.try_as<ID3D11DeviceContext1>();
        return true;
    }

    void cmd_list::begin(queue_type queue) {
        queue_ = queue;
        finished_ = nullptr;
        recorded_ = false;
        in_pass_ = false;
        pipeline_ = {};
        sets_ = {};
        vertex_ = {};
        index_buffer_ = {};
        dirty_sets_ = 0xF;
        pipeline_dirty_ = true;
        vertex_dirty_ = 0xFF;
        index_dirty_ = false;
        viewport_set_ = scissor_set_ = false;
        pass_width_ = pass_height_ = 0;
        bound_srvs_ = {};
        bound_uavs_ = 0;
        offsets_dirty_ = 0;
        resolved_ = {};
        reset_bound();
        resolves_.clear();
        counters_ = {};
        for (auto& stage_mirrors : upload_mirrors_)
            for (upload_mirror& m : stage_mirrors)
                m.valid = false;
    }

    ID3D11CommandList* cmd_list::finish() {
        if (!recorded_)
            return nullptr;
        if (in_pass_)
            end_rendering();
        if (FAILED(ctx_->FinishCommandList(FALSE, finished_.put())))
            return nullptr;
        return finished_.get();
    }

    void cmd_list::barrier(std::span<const texture_barrier> textures, std::span<const buffer_barrier> buffers) {
        counters_.barriers += textures.size() + buffers.size();
    }
    void cmd_list::uav_barrier(buffer_handle) {
        counters_.barriers++;
    }
    void cmd_list::uav_barrier(texture_handle) {
        counters_.barriers++;
    }

    void cmd_list::copy_to_constant(buffer& b, uint64_t offset, uint64_t size) {
        if (!b.constant)
            return;
        const D3D11_BOX box{static_cast<UINT>(offset), 0, 0, static_cast<UINT>(offset + size), 1, 1};
        ctx_->CopySubresourceRegion(b.constant.get(), 0, static_cast<UINT>(offset), 0, 0, b.main.get(), 0, &box);
    }

    void cmd_list::copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src, uint64_t src_offset,
                               uint64_t size) {
        recorded_ = true;
        buffer* d = device_.buffers_.get(dst);
        buffer* s = device_.buffers_.get(src);
        if (!d || !s || !size) {
            device_.report_error("CopyBuffer: bad handle");
            return;
        }
        size = std::min<uint64_t>({size, d->gpu_size - (std::min)(dst_offset, d->gpu_size),
                                   s->gpu_size - (std::min)(src_offset, s->gpu_size)});
        if (!size)
            return;
        const D3D11_BOX box{static_cast<UINT>(src_offset), 0, 0, static_cast<UINT>(src_offset + size), 1, 1};
        ctx_->CopySubresourceRegion(d->main.get(), 0, static_cast<UINT>(dst_offset), 0, 0, s->main.get(), 0, &box);
        copy_to_constant(*d, dst_offset, size);
    }

    void cmd_list::copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        recorded_ = true;
        texture* t = device_.textures_.get(dst);
        buffer* b = device_.buffers_.get(src);
        if (!t || !b || b->shadow.empty()) {
            device_.report_error("CopyBufferToTexture: bad handle or a source without cpu data");
            return;
        }
        for (const buffer_texture_copy& r : regions) {
            const uint32_t row_pitch = r.buffer_row_pitch ? r.buffer_row_pitch : row_bytes(t->desc.format, r.width);
            const uint32_t depth_pitch = row_pitch * row_blocks(t->desc.format, r.height);
            const uint32_t depth = (std::max)(1u, r.depth);
            if (r.buffer_offset + static_cast<uint64_t>(depth_pitch) * depth > b->shadow.size()) {
                device_.report_error("CopyBufferToTexture: region reads past the source buffer");
                continue;
            }
            const bool volume = t->desc.dim == texture_dim::tex3d;
            const D3D11_BOX box{r.x, r.y, r.z, r.x + r.width, r.y + r.height, r.z + depth};
            const bool whole = r.x == 0 && r.y == 0 && r.z == 0 && r.width >= mip_extent(t->desc.width, r.mip) &&
                               r.height >= mip_extent(t->desc.height, r.mip) &&
                               (!volume || depth >= (std::max)(1u, depth_of(t->desc) >> r.mip));
            ctx_->UpdateSubresource(t->resource.get(), subresource(t->desc, r.mip, volume ? 0 : r.layer),
                                    whole ? nullptr : &box, b->shadow.data() + r.buffer_offset, row_pitch, depth_pitch);
        }
    }

    void cmd_list::copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        recorded_ = true;
        texture* t = device_.textures_.get(src);
        buffer* b = device_.buffers_.get(dst);
        if (!t || !b || b->desc.memory != memory_class::readback) {
            device_.report_error("CopyTextureToBuffer: bad handle or a destination that is not a readback buffer");
            return;
        }
        const format_info& info = get_format_info(t->desc.format);
        for (const buffer_texture_copy& r : regions) {
            D3D11_TEXTURE2D_DESC sd{};
            sd.Width = static_cast<UINT>(align_up(r.width, info.block_width));
            sd.Height = static_cast<UINT>(align_up(r.height, info.block_height));
            sd.MipLevels = 1;
            sd.ArraySize = 1;
            sd.Format = t->formats.resource;
            sd.SampleDesc.Count = 1;
            sd.Usage = D3D11_USAGE_STAGING;
            sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            pending_readback pending;
            pending.staging = device_.take_staging(sd);
            if (!pending.staging) {
                device_.report_error("CopyTextureToBuffer: could not create the staging texture");
                continue;
            }
            const D3D11_BOX box{r.x, r.y, 0, r.x + r.width, r.y + r.height, 1};
            const bool whole = r.x == 0 && r.y == 0 && r.width >= mip_extent(t->desc.width, r.mip) &&
                               r.height >= mip_extent(t->desc.height, r.mip);
            ctx_->CopySubresourceRegion(pending.staging.get(), 0, 0, 0, 0, t->resource.get(),
                                        subresource(t->desc, r.mip, r.layer), whole || info.depth ? nullptr : &box);
            pending.buffer_offset = r.buffer_offset;
            pending.row_bytes = row_bytes(t->desc.format, r.width);
            pending.buffer_row_pitch = r.buffer_row_pitch ? r.buffer_row_pitch : pending.row_bytes;
            pending.rows = row_blocks(t->desc.format, r.height);
            std::lock_guard lock(device_.view_mutex_);
            b->pending.push_back(std::move(pending));
        }
    }

    void cmd_list::copy_texture(texture_handle dst, texture_handle src, std::span<const texture_copy> regions) {
        recorded_ = true;
        texture* d = device_.textures_.get(dst);
        texture* s = device_.textures_.get(src);
        if (!d || !s) {
            device_.report_error("CopyTexture: bad handle");
            return;
        }
        const bool depth = is_depth_format(s->desc.format);
        for (const texture_copy& r : regions) {
            const D3D11_BOX box{r.src_x, r.src_y, 0, r.src_x + r.width, r.src_y + r.height, 1};
            ctx_->CopySubresourceRegion(d->resource.get(), subresource(d->desc, r.dst_mip, r.dst_layer), r.dst_x,
                                        r.dst_y, 0, s->resource.get(), subresource(s->desc, r.src_mip, r.src_layer),
                                        depth ? nullptr : &box);
        }
    }

    void cmd_list::fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) {
        recorded_ = true;
        buffer* b = device_.buffers_.get(dst);
        if (!b) {
            device_.report_error("FillBuffer: bad handle");
            return;
        }
        if (!size || offset + size > b->gpu_size)
            size = b->gpu_size - (std::min)(offset, b->gpu_size);
        if (!size)
            return;
        // a raw view clears on the gpu, so nothing is built on the cpu or copied into the command list
        if (has(b->desc.usage, buffer_usage::storage) && b->desc.memory != memory_class::readback && !b->structured &&
            (offset & 3) == 0 && (size & 3) == 0)
            if (ID3D11UnorderedAccessView* uav = device_.buffer_uav(*b, offset, size)) {
                const UINT values[4] = {value, value, value, value};
                ctx_->ClearUnorderedAccessViewUint(uav, values);
                copy_to_constant(*b, offset, size);
                return;
            }
        thread_local std::vector<uint8_t> data;
        data.assign(static_cast<size_t>(size), 0);
        if (value)
            for (size_t i = 0; i + 4 <= data.size(); i += 4)
                std::memcpy(data.data() + i, &value, 4);
        const D3D11_BOX box{static_cast<UINT>(offset), 0, 0, static_cast<UINT>(offset + size), 1, 1};
        ctx_->UpdateSubresource(b->main.get(), 0, &box, data.data(), 0, 0);
        copy_to_constant(*b, offset, size);
    }

    void cmd_list::resolve_texture(texture_handle dst, texture_handle src) {
        recorded_ = true;
        texture* d = device_.textures_.get(dst);
        texture* s = device_.textures_.get(src);
        if (!d || !s) {
            device_.report_error("ResolveTexture: bad handle");
            return;
        }
        ctx_->ResolveSubresource(d->resource.get(), 0, s->resource.get(), 0, d->formats.view);
    }

    void cmd_list::clear_bindings() {
        static ID3D11ShaderResourceView* const null_srvs[max_srv_slots] = {};
        static ID3D11UnorderedAccessView* const null_uavs[8] = {};
        if (bound_srvs_[stage_vertex])
            ctx_->VSSetShaderResources(0, bound_srvs_[stage_vertex], null_srvs);
        if (bound_srvs_[stage_pixel])
            ctx_->PSSetShaderResources(0, bound_srvs_[stage_pixel], null_srvs);
        if (bound_srvs_[stage_compute])
            ctx_->CSSetShaderResources(0, bound_srvs_[stage_compute], null_srvs);
        if (bound_uavs_)
            ctx_->CSSetUnorderedAccessViews(0, (std::min)(bound_uavs_, 8u), null_uavs, nullptr);
        bound_srvs_ = {};
        bound_uavs_ = 0;
        for (stage_bound& b : bound_) {
            b.srvs = {};
            b.srv_begin = cached_srvs;
            b.srv_end = 0;
        }
    }

    void cmd_list::reset_bound() {
        bound_ = {};
        state_bound_ = {};
    }

    void cmd_list::bind_constant(uint32_t stage, uint32_t slot, ID3D11Buffer* cb, bool offset_call, UINT first,
                                 UINT constants) {
        if (!offset_call) {
            first = 0;
            constants = ~0u;
        }
        if (slot < cached_constants) {
            stage_bound& b = bound_[stage];
            if (b.constants[slot] == cb && b.first[slot] == first && b.count[slot] == constants)
                return;
            b.constants[slot] = cb;
            b.first[slot] = first;
            b.count[slot] = constants;
        }
        switch (stage) {
        case stage_vertex:
            if (offset_call)
                ctx1_->VSSetConstantBuffers1(slot, 1, &cb, &first, &constants);
            else
                ctx_->VSSetConstantBuffers(slot, 1, &cb);
            break;
        case stage_pixel:
            if (offset_call)
                ctx1_->PSSetConstantBuffers1(slot, 1, &cb, &first, &constants);
            else
                ctx_->PSSetConstantBuffers(slot, 1, &cb);
            break;
        default:
            if (offset_call)
                ctx1_->CSSetConstantBuffers1(slot, 1, &cb, &first, &constants);
            else
                ctx_->CSSetConstantBuffers(slot, 1, &cb);
            break;
        }
    }

    void cmd_list::bind_srv(uint32_t stage, uint32_t slot, ID3D11ShaderResourceView* srv) {
        if (slot < cached_srvs) {
            stage_bound& b = bound_[stage];
            if (b.srvs[slot] == srv)
                return;
            b.srvs[slot] = srv;
            if (stage != stage_compute) {
                b.srv_begin = (std::min)(b.srv_begin, slot);
                b.srv_end = (std::max)(b.srv_end, slot + 1);
                return;
            }
        }
        if (stage == stage_vertex)
            ctx_->VSSetShaderResources(slot, 1, &srv);
        else if (stage == stage_pixel)
            ctx_->PSSetShaderResources(slot, 1, &srv);
        else
            ctx_->CSSetShaderResources(slot, 1, &srv);
    }

    void cmd_list::bind_sampler(uint32_t stage, uint32_t slot, ID3D11SamplerState* state) {
        if (slot < cached_samplers) {
            stage_bound& b = bound_[stage];
            if (b.samplers[slot] == state)
                return;
            b.samplers[slot] = state;
            if (stage != stage_compute) {
                b.sampler_begin = (std::min)(b.sampler_begin, slot);
                b.sampler_end = (std::max)(b.sampler_end, slot + 1);
                return;
            }
        }
        if (stage == stage_vertex)
            ctx_->VSSetSamplers(slot, 1, &state);
        else if (stage == stage_pixel)
            ctx_->PSSetSamplers(slot, 1, &state);
        else
            ctx_->CSSetSamplers(slot, 1, &state);
    }

    // slots inside a range that did not change are sent with the value the context already holds, or null where
    // the shadow was reset, which only clears a slot no set of the current pipeline uses
    void cmd_list::flush_graphics_binds() {
        stage_bound& v = bound_[stage_vertex];
        stage_bound& f = bound_[stage_pixel];
        if (v.srv_begin < v.srv_end)
            ctx_->VSSetShaderResources(v.srv_begin, v.srv_end - v.srv_begin, v.srvs.data() + v.srv_begin);
        if (f.srv_begin < f.srv_end)
            ctx_->PSSetShaderResources(f.srv_begin, f.srv_end - f.srv_begin, f.srvs.data() + f.srv_begin);
        if (v.sampler_begin < v.sampler_end)
            ctx_->VSSetSamplers(v.sampler_begin, v.sampler_end - v.sampler_begin, v.samplers.data() + v.sampler_begin);
        if (f.sampler_begin < f.sampler_end)
            ctx_->PSSetSamplers(f.sampler_begin, f.sampler_end - f.sampler_begin, f.samplers.data() + f.sampler_begin);
        for (stage_bound* b : {&v, &f}) {
            b->srv_begin = cached_srvs;
            b->srv_end = 0;
            b->sampler_begin = cached_samplers;
            b->sampler_end = 0;
        }
    }

    void cmd_list::apply_pipeline(const pipeline& p) {
        state_bound& b = state_bound_;
        const bool all = b.fresh;
        if (all || b.input_layout != p.input_layout.get())
            ctx_->IASetInputLayout(b.input_layout = p.input_layout.get());
        if (all || b.topology != p.topology)
            ctx_->IASetPrimitiveTopology(b.topology = p.topology);
        if (all || b.vs != p.vs.get())
            ctx_->VSSetShader(b.vs = p.vs.get(), nullptr, 0);
        if (all || b.ps != p.ps.get())
            ctx_->PSSetShader(b.ps = p.ps.get(), nullptr, 0);
        if (all || b.raster != p.raster.get())
            ctx_->RSSetState(b.raster = p.raster.get());
        if (all || b.blend != p.blend.get())
            ctx_->OMSetBlendState(b.blend = p.blend.get(), nullptr, 0xFFFFFFFFu);
        if (all || b.depth_stencil != p.depth_stencil.get() || b.stencil_ref != stencil_ref_) {
            b.depth_stencil = p.depth_stencil.get();
            b.stencil_ref = stencil_ref_;
            ctx_->OMSetDepthStencilState(b.depth_stencil, stencil_ref_);
        }
        b.fresh = false;
    }

    void cmd_list::begin_rendering(const rendering_desc& desc) {
        recorded_ = true;
        if (in_pass_)
            end_rendering();
        clear_bindings();
        std::array<ID3D11RenderTargetView*, max_color_targets> rtvs{};
        pass_width_ = pass_height_ = 0;
        for (uint32_t i = 0; i < desc.color_count; i++) {
            const color_attachment& a = desc.colors[i];
            texture* t = device_.textures_.get(a.texture);
            ID3D11RenderTargetView* rtv = t ? device_.rtv_of(*t, a.mip, a.layer) : nullptr;
            if (!rtv) {
                device_.report_error("BeginRendering: colour attachment " + std::to_string(i) + " is gone");
                return;
            }
            rtvs[i] = rtv;
            if (a.load == load_op::clear)
                ctx_->ClearRenderTargetView(rtv, a.clear.data());
            if (a.resolve)
                resolves_.push_back({a.resolve, a.texture});
            if (!pass_width_) {
                pass_width_ = mip_extent(t->desc.width, a.mip);
                pass_height_ = mip_extent(t->desc.height, a.mip);
            }
        }
        ID3D11DepthStencilView* dsv = nullptr;
        if (desc.depth.texture) {
            texture* t = device_.textures_.get(desc.depth.texture);
            dsv = t ? device_.dsv_of(*t, desc.depth.mip, desc.depth.layer, desc.depth.read_only) : nullptr;
            if (!dsv) {
                device_.report_error("BeginRendering: depth attachment is gone");
                return;
            }
            if (!desc.depth.read_only && desc.depth.load == load_op::clear)
                ctx_->ClearDepthStencilView(dsv,
                                            D3D11_CLEAR_DEPTH | (has_stencil(t->desc.format) ? D3D11_CLEAR_STENCIL : 0),
                                            desc.depth.clear_depth, desc.depth.clear_stencil);
            if (!pass_width_) {
                pass_width_ = mip_extent(t->desc.width, desc.depth.mip);
                pass_height_ = mip_extent(t->desc.height, desc.depth.mip);
            }
        }
        ctx_->OMSetRenderTargets(desc.color_count, rtvs.data(), dsv);
        in_pass_ = true;
        dirty_sets_ = 0xF;
        pipeline_dirty_ = true;
        vertex_dirty_ = 0xFF;
        index_dirty_ = index_buffer_.value != 0;

        const bool had_viewport = viewport_set_, had_scissor = scissor_set_;
        set_viewport(had_viewport ? viewport_
                                  : viewport{0, 0, static_cast<float>(pass_width_), static_cast<float>(pass_height_),
                                             0.0f, 1.0f});
        set_scissor(had_scissor ? scissor_ : rect{0, 0, pass_width_, pass_height_});
        viewport_set_ = had_viewport;
        scissor_set_ = had_scissor;
    }

    void cmd_list::end_rendering() {
        if (!in_pass_)
            return;
        ctx_->OMSetRenderTargets(0, nullptr, nullptr);
        for (const resolve& r : resolves_) {
            texture* d = device_.textures_.get(r.dst);
            texture* s = device_.textures_.get(r.src);
            if (d && s)
                ctx_->ResolveSubresource(d->resource.get(), 0, s->resource.get(), 0, d->formats.view);
        }
        resolves_.clear();
        in_pass_ = false;
    }

    void cmd_list::set_pipeline(pipeline_handle p) {
        if (p == pipeline_)
            return;
        pipeline_ = p;
        pipeline_dirty_ = true;
        dirty_sets_ = 0xF;
        vertex_dirty_ = 0xFF;
    }

    void cmd_list::set_bind_group(uint32_t set, bind_group_handle group, std::span<const uint32_t> dynamic_offsets) {
        if (set >= max_bind_groups)
            return;
        bound_set& s = sets_[set];
        const uint32_t bit = 1u << set;
        const uint32_t count = static_cast<uint32_t>(std::min<size_t>(dynamic_offsets.size(), max_dynamic_offsets));
        counters_.group_binds++;
        if (s.group == group && !(dirty_sets_ & bit)) {
            if (count == s.offset_count &&
                std::equal(dynamic_offsets.begin(), dynamic_offsets.begin() + count, s.offsets.begin()))
                return;
            s.offset_count = count;
            std::copy_n(dynamic_offsets.begin(), count, s.offsets.begin());
            offsets_dirty_ |= bit;
            return;
        }
        s.group = group;
        s.offset_count = count;
        std::copy_n(dynamic_offsets.begin(), count, s.offsets.begin());
        dirty_sets_ |= bit;
    }

    void cmd_list::push_constants(const void*, uint32_t, uint32_t) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            device_.report_error("PushConstants: d3d11 has none; per-draw data belongs in a uniform buffer");
        }
    }

    void cmd_list::set_viewport(const viewport& v) {
        recorded_ = true;
        viewport_ = v;
        viewport_set_ = true;
        if (!in_pass_)
            return;
        viewport c = v;
        if (pass_width_) {
            c.width = (std::min)(c.width, static_cast<float>(pass_width_) - c.x);
            c.height = (std::min)(c.height, static_cast<float>(pass_height_) - c.y);
        }
        const D3D11_VIEWPORT d{c.x, c.y, c.width, c.height, c.min_depth, c.max_depth};
        ctx_->RSSetViewports(1, &d);
    }

    void cmd_list::set_scissor(const rect& r) {
        recorded_ = true;
        scissor_ = r;
        scissor_set_ = true;
        if (!in_pass_)
            return;
        const int64_t left = (std::max)(r.x, 0);
        const int64_t top = (std::max)(r.y, 0);
        int64_t right = left + r.width, bottom = top + r.height;
        if (pass_width_) {
            right = std::min<int64_t>(right, pass_width_);
            bottom = std::min<int64_t>(bottom, pass_height_);
        }
        const D3D11_RECT d{static_cast<LONG>(left), static_cast<LONG>(top), static_cast<LONG>(std::max(left, right)),
                           static_cast<LONG>(std::max(top, bottom))};
        ctx_->RSSetScissorRects(1, &d);
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
        index_dirty_ = true;
    }

    void cmd_list::set_stencil_reference(uint32_t value) {
        stencil_ref_ = value;
        if (const pipeline* p = device_.pipelines_.get(pipeline_);
            p && !p->compute && in_pass_ && !pipeline_dirty_ && state_bound_.stencil_ref != value) {
            state_bound_.depth_stencil = p->depth_stencil.get();
            state_bound_.stencil_ref = value;
            ctx_->OMSetDepthStencilState(state_bound_.depth_stencil, value);
        }
    }

    bool cmd_list::bind_uniform(uniform_range& u, const uniform_range* same, const bound_set& set) {
        // a stage reading the bytes another stage of this set just had copied into a mirror binds that mirror
        if (same && same->mirror) {
            u.mirror = same->mirror;
            bind_constant(u.stage, u.slot, u.mirror, false, 0, 0);
            return true;
        }
        buffer* b = device_.buffers_.get(u.buffer);
        if (!b) {
            device_.report_error("bind group buffer at binding " + std::to_string(u.binding) + " is gone");
            return false;
        }
        const uint64_t offset = u.base + (u.dynamic < set.offset_count ? set.offsets[u.dynamic] : 0);
        const uint64_t size = u.size ? u.size : b->gpu_size - (std::min)(offset, b->gpu_size);
        u.mirror = nullptr;
        if (b->constant && ctx1_ && (offset & 255) == 0 && offset < b->gpu_size) {
            const uint64_t span = std::min<uint64_t>(align_up(size, 256), b->gpu_size - offset);
            bind_constant(u.stage, u.slot, b->constant.get(), true, static_cast<UINT>(offset / 16),
                          static_cast<UINT>(std::min<uint64_t>(span, constant_window) / 16));
            return true;
        }
        if (!b->shadow.empty() && b->desc.memory != memory_class::gpu_only &&
            b->desc.memory != memory_class::readback) {
            upload_mirror& um = upload_mirrors_[u.stage][u.slot];
            const uint64_t avail = b->shadow.size() > offset ? b->shadow.size() - offset : 0;
            const uint64_t want = (std::min)({size, avail, constant_window});
            const UINT n = static_cast<UINT>(want & ~15ull);
            if (!n) {
                device_.report_error("uniform buffer range is empty at binding " + std::to_string(u.binding));
                return false;
            }
            if (!um.buffer || um.bytes != n) {
                D3D11_BUFFER_DESC md{};
                md.ByteWidth = n;
                md.Usage = D3D11_USAGE_DEFAULT;
                md.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
                um.buffer = nullptr;
                if (FAILED(device_.dev()->CreateBuffer(&md, nullptr, um.buffer.put())))
                    return false;
                um.bytes = n;
                um.valid = false;
            }
            const uint8_t* src = b->shadow.data() + offset;
            if (!um.valid || std::memcmp(um.last.data(), src, n) != 0) {
                ctx_->UpdateSubresource(um.buffer.get(), 0, nullptr, src, 0, 0);
                um.last.assign(src, src + n);
                um.valid = true;
            }
            u.mirror = um.buffer.get();
        } else {
            winrt::com_ptr<ID3D11Buffer>& mirror = mirrors_[u.stage][u.slot];
            if (!mirror) {
                D3D11_BUFFER_DESC md{};
                md.ByteWidth = static_cast<UINT>(constant_window);
                md.Usage = D3D11_USAGE_DEFAULT;
                md.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
                if (FAILED(device_.dev()->CreateBuffer(&md, nullptr, mirror.put())))
                    return false;
            }
            const uint64_t n = std::min<uint64_t>(
                {align_up(size, 16), constant_window, b->gpu_size - (std::min)(offset, b->gpu_size)});
            const D3D11_BOX box{static_cast<UINT>(offset), 0, 0, static_cast<UINT>(offset + n), 1, 1};
            ctx_->CopySubresourceRegion(mirror.get(), 0, 0, 0, 0, b->main.get(), 0, &box);
            u.mirror = mirror.get();
        }
        bind_constant(u.stage, u.slot, u.mirror, false, 0, 0);
        return true;
    }

    bool cmd_list::bind_sets(const pipeline& p) {
        const uint32_t first = p.compute ? stage_compute : stage_vertex;
        const uint32_t last = p.compute ? stage_compute : stage_pixel;
        for (uint32_t moved = offsets_dirty_ & ~dirty_sets_; moved;) {
            const uint32_t s = static_cast<uint32_t>(std::countr_zero(moved));
            moved &= moved - 1;
            const bound_set& bs = sets_[s];
            resolved_set& r = resolved_[s];
            if (!r.valid || r.pipeline != pipeline_ || r.group != bs.group) {
                dirty_sets_ |= 1u << s;
                continue;
            }
            for (uint32_t i = 0; i < r.count; i++) {
                uniform_range& u = r.constants[i];
                if (u.dynamic != no_dynamic && !bind_uniform(u, u.same < i ? &r.constants[u.same] : nullptr, bs)) {
                    dirty_sets_ |= 1u << s;
                    offsets_dirty_ = 0;
                    return false;
                }
            }
        }
        offsets_dirty_ = 0;
        for (uint32_t s = 0; s < max_bind_groups; s++) {
            if (!(dirty_sets_ & (1u << s)))
                continue;
            const bound_set& bs = sets_[s];
            const bind_group* group = device_.bind_groups_.get(bs.group);
            resolved_set& r = resolved_[s];
            r.pipeline = pipeline_;
            r.group = bs.group;
            r.valid = true;
            r.count = 0;
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
                    case resource_kind::uniform_buffer: {
                        uniform_range u;
                        u.stage = static_cast<uint8_t>(stage);
                        u.slot = static_cast<uint8_t>(slot.slot);
                        u.dynamic = slot.dynamic_index < max_dynamic_offsets ? static_cast<uint8_t>(slot.dynamic_index)
                                                                             : no_dynamic;
                        u.binding = slot.binding;
                        u.buffer = e->buffer.buffer;
                        u.base = e->buffer.offset;
                        u.size = e->buffer.size;
                        const uniform_range* same = nullptr;
                        for (uint32_t i = 0; i < r.count && !same; i++) {
                            const uniform_range& o = r.constants[i];
                            if (o.buffer == u.buffer && o.base == u.base && o.size == u.size &&
                                o.dynamic == u.dynamic) {
                                same = &o;
                                u.same = static_cast<uint8_t>(i);
                            }
                        }
                        if (r.count == r.constants.size()) {
                            r.valid = false;
                            if (!bind_uniform(u, same, bs))
                                return false;
                            break;
                        }
                        r.constants[r.count] = u;
                        if (!bind_uniform(r.constants[r.count], same, bs))
                            return false;
                        r.count++;
                        break;
                    }
                    case resource_kind::storage_buffer:
                    case resource_kind::storage_buffer_rw:
                    case resource_kind::sampled_texture:
                    case resource_kind::storage_texture: {
                        if (slot.register_class == 'u') {
                            if (stage != stage_compute)
                                break;
                            ID3D11UnorderedAccessView* uav = nullptr;
                            if (slot.kind == resource_kind::storage_texture) {
                                texture* t = device_.textures_.get(e->texture.texture);
                                uav = t ? device_.uav_of(*t, e->texture.range, slot.dim) : nullptr;
                            } else {
                                buffer* b = device_.buffers_.get(e->buffer.buffer);
                                uav = b ? device_.buffer_uav(*b, e->buffer.offset, e->buffer.size) : nullptr;
                            }
                            if (!uav) {
                                device_.report_error("bind group storage resource at binding " +
                                                     std::to_string(slot.binding) + " is gone");
                                return false;
                            }
                            ctx_->CSSetUnorderedAccessViews(slot.slot, 1, &uav, nullptr);
                            bound_uavs_ = (std::max)(bound_uavs_, slot.slot + 1);
                            // binding a resource for writing pulls it out of every stage's read slots
                            for (stage_bound& sb : bound_)
                                sb.srvs = {};
                            break;
                        }
                        ID3D11ShaderResourceView* srv = nullptr;
                        if (slot.kind == resource_kind::sampled_texture ||
                            slot.kind == resource_kind::storage_texture) {
                            if (texture* t = device_.textures_.get(e->texture.texture)) {
                                const bind_group::texture_view* v =
                                    group->views.empty() ? nullptr : &group->views[e - group->entries.data()];
                                srv = v && v->srv && v->dim == slot.dim
                                          ? v->srv.get()
                                          : device_.srv_of(*t, e->texture.range, slot.dim, e->texture.format,
                                                           e->texture.stencil);
                            }
                        } else {
                            buffer* b = device_.buffers_.get(e->buffer.buffer);
                            srv = b ? device_.buffer_srv(*b, e->buffer.offset, e->buffer.size) : nullptr;
                        }
                        if (!srv) {
                            device_.report_error("bind group resource at binding " + std::to_string(slot.binding) +
                                                 " is gone");
                            return false;
                        }
                        bind_srv(stage, slot.slot, srv);
                        bound_srvs_[stage] = (std::max)(bound_srvs_[stage], slot.slot + 1);
                        break;
                    }
                    case resource_kind::sampler: {
                        sampler* sm = device_.samplers_.get(e->sampler);
                        if (!sm) {
                            device_.report_error("bind group sampler at binding " + std::to_string(slot.binding) +
                                                 " is gone");
                            return false;
                        }
                        bind_sampler(stage, slot.slot, sm->state.get());
                        break;
                    }
                    }
                }
            }
        }
        dirty_sets_ = 0;
        if (!p.compute)
            flush_graphics_binds();
        return true;
    }

    bool cmd_list::bind_vertex_streams(const pipeline& p, int64_t max_shift) {
        state_bound& sb = state_bound_;
        const bool rebase = max_shift >= 0 && !p.vertex_id;
        std::array<ID3D11Buffer*, max_vertex_streams> bufs{};
        for (uint32_t i = 0; i < p.stream_count; i++) {
            const uint32_t bit = 1u << i;
            const bool needed = (vertex_dirty_ & bit) || (vertex_shift_ && (p.per_vertex & bit));
            if (!needed && !(rebase && (p.per_vertex & bit)))
                continue;
            if (buffer* b = device_.buffers_.get(vertex_[i].buffer)) {
                bufs[i] = b->main.get();
            } else if (needed) {
                device_.report_error("draw: vertex stream " + std::to_string(i) + " is not bound");
                return false;
            }
        }
        // the per-vertex streams can stay where they are when all of them moved by the same whole number of
        // vertices from what the context holds
        bool keep = rebase && p.per_vertex;
        int64_t shift = -1;
        for (uint32_t i = 0; keep && i < p.stream_count; i++) {
            if (!(p.per_vertex & (1u << i)))
                continue;
            const UINT stride = p.strides[i];
            const UINT offset = static_cast<UINT>(vertex_[i].offset);
            if (!bufs[i] || bufs[i] != sb.vertex[i] || sb.stride[i] != stride || !stride || offset < sb.offset[i] ||
                (offset - sb.offset[i]) % stride) {
                keep = false;
                break;
            }
            const int64_t d = (offset - sb.offset[i]) / stride;
            keep = (shift < 0 || d == shift) && d <= max_shift;
            shift = d;
        }
        vertex_shift_ = keep ? static_cast<uint32_t>(shift) : 0;
        for (uint32_t i = 0; i < p.stream_count; i++) {
            if (!bufs[i] || (keep && (p.per_vertex & (1u << i))))
                continue;
            const UINT stride = p.strides[i];
            const UINT offset = static_cast<UINT>(vertex_[i].offset);
            if (sb.vertex[i] == bufs[i] && sb.stride[i] == stride && sb.offset[i] == offset)
                continue;
            sb.vertex[i] = bufs[i];
            sb.stride[i] = stride;
            sb.offset[i] = offset;
            ctx_->IASetVertexBuffers(i, 1, &bufs[i], &stride, &offset);
        }
        vertex_dirty_ = 0;
        return true;
    }

    // max_shift is how far the draw's start vertex may move, negative where it cannot (indirect arguments)
    bool cmd_list::prepare_draw(int64_t max_shift) {
        recorded_ = true;
        if (!in_pass_) {
            device_.report_error("draw outside a rendering pass");
            return false;
        }
        const pipeline* p = device_.pipelines_.get(pipeline_);
        if (!p || p->compute) {
            device_.report_error("draw without a graphics pipeline");
            return false;
        }
        if (pipeline_dirty_) {
            apply_pipeline(*p);
            pipeline_dirty_ = false;
            counters_.pipeline_binds++;
        }
        if ((dirty_sets_ | offsets_dirty_) && !bind_sets(*p))
            return false;
        if (vertex_dirty_ || (vertex_shift_ && vertex_shift_ > max_shift))
            return bind_vertex_streams(*p, max_shift);
        return true;
    }

    void cmd_list::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                        uint32_t first_instance) {
        if (!prepare_draw(int64_t{UINT32_MAX} - first_vertex))
            return;
        ctx_->DrawInstanced(vertex_count, instance_count, first_vertex + vertex_shift_, first_instance);
        counters_.draws++;
    }

    void cmd_list::draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                                int32_t vertex_offset, uint32_t first_instance) {
        if (!prepare_draw(int64_t{INT32_MAX} - vertex_offset))
            return;
        if (index_dirty_) {
            buffer* b = device_.buffers_.get(index_buffer_);
            if (!b) {
                device_.report_error("DrawIndexed: no index buffer");
                return;
            }
            ctx_->IASetIndexBuffer(b->main.get(),
                                   index_type_ == index_type::u16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT,
                                   static_cast<UINT>(index_offset_));
            index_dirty_ = false;
        }
        ctx_->DrawIndexedInstanced(index_count, instance_count, first_index,
                                   vertex_offset + static_cast<int32_t>(vertex_shift_), first_instance);
        counters_.draws++;
    }

    void cmd_list::draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws, uint32_t stride) {
        if (!prepare_draw(-1))
            return;
        buffer* a = device_.buffers_.get(args);
        buffer* ib = device_.buffers_.get(index_buffer_);
        if (!a || !ib) {
            device_.report_error("DrawIndexedIndirect: bad buffer");
            return;
        }
        if (index_dirty_) {
            ctx_->IASetIndexBuffer(ib->main.get(),
                                   index_type_ == index_type::u16 ? DXGI_FORMAT_R16_UINT : DXGI_FORMAT_R32_UINT,
                                   static_cast<UINT>(index_offset_));
            index_dirty_ = false;
        }
        for (uint32_t i = 0; i < max_draws; i++)
            ctx_->DrawIndexedInstancedIndirect(a->main.get(),
                                               static_cast<UINT>(offset + static_cast<uint64_t>(i) * stride));
        counters_.indirect += max_draws;
    }

    void cmd_list::draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle, uint64_t,
                                               uint32_t max_draws, uint32_t stride) {
        draw_indexed_indirect(args, offset, max_draws, stride);
    }

    void cmd_list::dispatch(uint32_t x, uint32_t y, uint32_t z) {
        recorded_ = true;
        const pipeline* p = device_.pipelines_.get(pipeline_);
        if (!p || !p->compute) {
            device_.report_error("Dispatch without a compute pipeline");
            return;
        }
        if (pipeline_dirty_) {
            if (state_bound_.fresh || state_bound_.cs != p->cs.get())
                ctx_->CSSetShader(state_bound_.cs = p->cs.get(), nullptr, 0);
            pipeline_dirty_ = false;
        }
        if (!bind_sets(*p))
            return;
        ctx_->Dispatch(x, y, z);
        counters_.dispatches++;
        if (bound_uavs_) {
            static ID3D11UnorderedAccessView* const null_uavs[8] = {};
            ctx_->CSSetUnorderedAccessViews(0, (std::min)(bound_uavs_, 8u), null_uavs, nullptr);
            bound_uavs_ = 0;
            dirty_sets_ = 0xF;
        }
    }

    void cmd_list::dispatch_indirect(buffer_handle args, uint64_t offset) {
        recorded_ = true;
        const pipeline* p = device_.pipelines_.get(pipeline_);
        buffer* a = device_.buffers_.get(args);
        if (!p || !p->compute || !a) {
            device_.report_error("DispatchIndirect: bad pipeline or buffer");
            return;
        }
        if (pipeline_dirty_) {
            if (state_bound_.fresh || state_bound_.cs != p->cs.get())
                ctx_->CSSetShader(state_bound_.cs = p->cs.get(), nullptr, 0);
            pipeline_dirty_ = false;
        }
        if (!bind_sets(*p))
            return;
        ctx_->DispatchIndirect(a->main.get(), static_cast<UINT>(offset));
        counters_.dispatches++;
        if (bound_uavs_) {
            static ID3D11UnorderedAccessView* const null_uavs[8] = {};
            ctx_->CSSetUnorderedAccessViews(0, (std::min)(bound_uavs_, 8u), null_uavs, nullptr);
            bound_uavs_ = 0;
            dirty_sets_ = 0xF;
        }
    }

    void cmd_list::write_timestamp(query_set_handle, uint32_t) {}
    void cmd_list::resolve_queries(query_set_handle, uint32_t, uint32_t, buffer_handle, uint64_t) {}
    void cmd_list::begin_debug_label(const char*, uint32_t) {}
    void cmd_list::end_debug_label() {}
}
