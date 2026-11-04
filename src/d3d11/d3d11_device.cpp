#include <cstdlib>
#include "d3d11/d3d11_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>

namespace conduit::d3d11 {
    namespace {
        constexpr UINT optimize_full = 1u << 15;

        bool is_integer_format(format f) {
            return f == format::r8_uint || f == format::r16_uint || f == format::r32_uint || f == format::rg32_uint ||
                   f == format::rgba32_uint;
        }

        int class_index(char c) {
            return c == 'b' ? 0 : c == 't' ? 1 : c == 's' ? 2 : 3;
        }
    }

    device::device(std::shared_ptr<instance_state> instance, winrt::com_ptr<IDXGIAdapter1> adapter_in,
                   const adapter_info& info, const device_options& opts)
        : inst_(std::move(instance)), adapter_(std::move(adapter_in)), options_(opts) {
        caps_.adapter = info;
        options_.frames_in_flight = std::clamp(options_.frames_in_flight, 1u, 4u);
    }

    device::~device() {
        if (!device_)
            return;
        wait_idle();
        lists_.clear();
        free_queries_.clear();
        if (immediate_) {
            immediate_->ClearState();
            immediate_->Flush();
        }
    }

    void device::report_error(const std::string& message) {
        std::lock_guard lock(error_mutex_);
        std::fprintf(stderr, "[d3d11] %s\n", message.c_str());
    }

    native_handles device::native() const {
        native_handles n;
        n.api = api::d3d11;
        n.device = device_.get();
        n.adapter = adapter_.get();
        n.factory = inst_->factory.get();
        return n;
    }

    bool device::init(std::string& error) {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
        if (inst_->validation)
            flags |= D3D11_CREATE_DEVICE_DEBUG;
        const D3D_FEATURE_LEVEL both[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        const D3D_FEATURE_LEVEL only11[] = {D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL got{};
        HRESULT hr = E_FAIL;
        for (const bool debug : {true, false}) {
            if (debug && !(flags & D3D11_CREATE_DEVICE_DEBUG))
                continue;
            const UINT f = debug ? flags : (flags & ~static_cast<UINT>(D3D11_CREATE_DEVICE_DEBUG));
            for (const auto& levels :
                 {std::span<const D3D_FEATURE_LEVEL>(both), std::span<const D3D_FEATURE_LEVEL>(only11)}) {
                device_ = nullptr;
                immediate_ = nullptr;
                hr = inst_->lib.create_device(adapter_.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, f, levels.data(),
                                              static_cast<UINT>(levels.size()), D3D11_SDK_VERSION, device_.put(), &got,
                                              immediate_.put());
                if (SUCCEEDED(hr))
                    break;
            }
            if (SUCCEEDED(hr))
                break;
        }
        if (FAILED(hr)) {
            error = "D3D11CreateDevice failed";
            return false;
        }
        device1_ = device_.try_as<ID3D11Device1>();
        if (device1_) {
            D3D11_FEATURE_DATA_D3D11_OPTIONS options11{};
            if (SUCCEEDED(device_->CheckFeatureSupport(D3D11_FEATURE_D3D11_OPTIONS, &options11, sizeof(options11))))
                constant_offsetting_ =
                    options11.ConstantBufferOffsetting != FALSE && std::getenv("PERLIN_D3D11_CB_OFFSET") != nullptr;
        }
        fill_caps();
        frame_fence_ = create_fence(0);
        return true;
    }

    void device::fill_caps() {
        device_limits& l = caps_.limits;
        l.max_texture2d = 16384;
        l.max_texture_layers = 2048;
        l.max_bind_groups = max_bind_groups;
        l.max_push_constant_bytes = 0;
        l.max_sampled_textures_per_stage = 128;
        l.max_color_targets = 8;
        l.uniform_buffer_alignment = 256;
        l.storage_buffer_alignment = 16;
        l.copy_row_pitch_alignment = 4;
        l.copy_offset_alignment = 4;
        l.max_buffer_size = 1ull << 30;
        l.max_anisotropy = 16.0f;
        l.max_msaa_samples = 1;
        for (const UINT samples : {8u, 4u, 2u}) {
            UINT levels = 0;
            if (SUCCEEDED(device_->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, samples, &levels)) &&
                levels > 0) {
                l.max_msaa_samples = samples;
                break;
            }
        }

        device_features& f = caps_.features;
        f.sampler_anisotropy = true;
        f.texture_compression_bc = true;
        f.depth_clip_control = true;
        f.draw_indirect_first_instance = true;
        f.unified_memory = caps_.adapter.type == adapter_type::integrated || caps_.adapter.type == adapter_type::cpu;
        f.tearing = inst_->tearing;
        f.multi_draw_indirect = false;
        f.draw_indirect_count = false;
        f.async_compute = false;
        f.dedicated_copy_queue = false;
        f.timestamp_queries = false;

        for (uint32_t i = 1; i < static_cast<uint32_t>(format::count); i++) {
            const format fmt = static_cast<format>(i);
            const format_map fm = map_format(fmt);
            if (fm.view == DXGI_FORMAT_UNKNOWN)
                continue;
            UINT s = 0;
            device_->CheckFormatSupport(fm.view, &s);
            format_use use = format_use::none;
            if (s & D3D11_FORMAT_SUPPORT_SHADER_SAMPLE)
                use = use | format_use::sampled |
                      (is_integer_format(fmt) ? format_use::none : format_use::sampled_linear);
            if (s & D3D11_FORMAT_SUPPORT_RENDER_TARGET)
                use = use | format_use::color_target;
            if (s & D3D11_FORMAT_SUPPORT_BLENDABLE)
                use = use | format_use::blendable;
            if (s & D3D11_FORMAT_SUPPORT_TYPED_UNORDERED_ACCESS_VIEW)
                use = use | format_use::storage;
            if (s & D3D11_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET)
                use = use | format_use::msaa;
            if (fm.dsv != DXGI_FORMAT_UNKNOWN) {
                UINT d = 0;
                device_->CheckFormatSupport(fm.dsv, &d);
                if (d & D3D11_FORMAT_SUPPORT_DEPTH_STENCIL)
                    use = use | format_use::depth_target;
                if (d & D3D11_FORMAT_SUPPORT_MULTISAMPLE_RENDERTARGET)
                    use = use | format_use::msaa;
            }
            caps_.formats[i] = use;
        }
    }

    buffer_handle device::create_buffer(const buffer_desc& desc) {
        buffer b;
        b.desc = desc;
        b.name = desc.name ? desc.name : "";
        const bool storage = has(desc.usage, buffer_usage::storage);
        const bool indirect = has(desc.usage, buffer_usage::indirect);
        b.structured = storage && desc.stride > 0 && !indirect;
        uint64_t size = align_up(std::max<uint64_t>(desc.size, 16), 16);
        if (has(desc.usage, buffer_usage::uniform))
            size = align_up(size, 256);
        if (b.structured)
            size = align_up(size, desc.stride);
        b.gpu_size = size;

        D3D11_BUFFER_DESC bd{};
        bd.ByteWidth = static_cast<UINT>(size);
        if (desc.memory == memory_class::readback) {
            bd.Usage = D3D11_USAGE_STAGING;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        } else {
            bd.Usage = D3D11_USAGE_DEFAULT;
            if (has(desc.usage, buffer_usage::vertex))
                bd.BindFlags |= D3D11_BIND_VERTEX_BUFFER;
            if (has(desc.usage, buffer_usage::index))
                bd.BindFlags |= D3D11_BIND_INDEX_BUFFER;
            if (storage)
                bd.BindFlags |= D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
            if (indirect)
                bd.MiscFlags |= D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
            if (b.structured) {
                bd.MiscFlags |= D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
                bd.StructureByteStride = desc.stride;
            } else if (storage) {
                bd.MiscFlags |= D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
            }
        }
        if (FAILED(device_->CreateBuffer(&bd, nullptr, b.main.put())))
            return {};

        if (has(desc.usage, buffer_usage::uniform) && constant_offsetting_ && desc.memory != memory_class::readback) {
            D3D11_BUFFER_DESC cd{};
            cd.ByteWidth = static_cast<UINT>(size);
            cd.Usage = D3D11_USAGE_DEFAULT;
            cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            if (FAILED(device_->CreateBuffer(&cd, nullptr, b.constant.put())))
                return {};
        }
        if (desc.memory != memory_class::gpu_only)
            b.shadow.assign(size, 0);
        stats_.buffer_bytes += size * (b.constant ? 2 : 1);
        stats_.live_buffers++;
        const buffer_handle h = buffers_.alloc(std::move(b));
        buffers_.get(h)->desc.name = buffers_.get(h)->name.c_str();
        return h;
    }

    texture_handle device::create_texture(const texture_desc& desc) {
        const format_map fm = map_format(desc.format);
        if (fm.resource == DXGI_FORMAT_UNKNOWN) {
            report_error("CreateTexture: unsupported format " + std::to_string(static_cast<int>(desc.format)));
            return {};
        }
        texture t;
        t.desc = desc;
        t.desc.mips = (std::max)(1u, desc.mips);
        t.desc.samples = (std::max)(1u, desc.samples);
        t.formats = fm;
        t.name = desc.name ? desc.name : "";

        UINT bind = 0;
        if (has(desc.usage, texture_usage::sampled))
            bind |= D3D11_BIND_SHADER_RESOURCE;
        if (has(desc.usage, texture_usage::color_target))
            bind |= D3D11_BIND_RENDER_TARGET;
        if (has(desc.usage, texture_usage::depth_target))
            bind |= D3D11_BIND_DEPTH_STENCIL;
        if (has(desc.usage, texture_usage::storage))
            bind |= D3D11_BIND_UNORDERED_ACCESS;

        if (desc.dim == texture_dim::tex3d) {
            D3D11_TEXTURE3D_DESC td{};
            td.Width = desc.width;
            td.Height = desc.height;
            td.Depth = (std::max)(1u, desc.depth_or_layers);
            td.MipLevels = t.desc.mips;
            td.Format = fm.resource;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = bind;
            winrt::com_ptr<ID3D11Texture3D> tex;
            if (FAILED(device_->CreateTexture3D(&td, nullptr, tex.put())))
                return {};
            t.resource.copy_from(tex.get());
        } else {
            D3D11_TEXTURE2D_DESC td{};
            td.Width = desc.width;
            td.Height = desc.height;
            td.MipLevels = t.desc.samples > 1 ? 1 : t.desc.mips;
            td.ArraySize = layer_count_of(t.desc);
            td.Format = fm.resource;
            td.SampleDesc.Count = t.desc.samples;
            td.Usage = D3D11_USAGE_DEFAULT;
            td.BindFlags = bind;
            td.MiscFlags = desc.dim == texture_dim::cube ? D3D11_RESOURCE_MISC_TEXTURECUBE : 0;
            winrt::com_ptr<ID3D11Texture2D> tex;
            if (FAILED(device_->CreateTexture2D(&td, nullptr, tex.put())))
                return {};
            t.resource.copy_from(tex.get());
        }
        uint64_t bytes = 0;
        for (uint32_t m = 0; m < t.desc.mips; m++)
            bytes += static_cast<uint64_t>(row_bytes(desc.format, mip_extent(desc.width, m))) *
                     row_blocks(desc.format, mip_extent(desc.height, m)) * layer_count_of(t.desc) * depth_of(t.desc);
        stats_.texture_bytes += bytes * t.desc.samples;
        stats_.live_textures++;
        const texture_handle h = textures_.alloc(std::move(t));
        textures_.get(h)->desc.name = textures_.get(h)->name.c_str();
        return h;
    }

    sampler_handle device::create_sampler(const sampler_desc& desc) {
        sampler s;
        s.desc = desc;
        D3D11_SAMPLER_DESC sd{};
        sd.Filter = to_d3d(desc);
        sd.AddressU = to_d3d(desc.u);
        sd.AddressV = to_d3d(desc.v);
        sd.AddressW = to_d3d(desc.w);
        sd.MipLODBias = desc.lod_bias;
        sd.MaxAnisotropy = static_cast<UINT>(std::clamp(desc.max_anisotropy, 1.0f, 16.0f));
        sd.ComparisonFunc = desc.compare ? to_d3d(desc.compare_op) : D3D11_COMPARISON_NEVER;
        for (int i = 0; i < 4; i++)
            sd.BorderColor[i] = desc.border[i];
        sd.MinLOD = desc.min_lod;
        sd.MaxLOD = (std::min)(desc.max_lod, 1000.0f);
        if (FAILED(device_->CreateSamplerState(&sd, s.state.put())))
            return {};
        return samplers_.alloc(std::move(s));
    }

    shader_handle device::create_shader(const shader_desc& desc) {
        if (desc.format != shader_format::hlsl) {
            report_error("CreateShader: d3d11 takes hlsl source");
            return {};
        }
        shader s;
        s.stage = desc.stage;
        s.source.assign(reinterpret_cast<const char*>(desc.code.data()), desc.code.size());
        s.entry = desc.entry ? desc.entry : "main";
        s.iface = reflect_shader(shader_format::hlsl, s.source);
        return shaders_.alloc(std::move(s));
    }

    bind_layout_handle device::create_bind_layout(const bind_layout_desc& desc) {
        bind_layout l;
        l.entries.assign(desc.entries.begin(), desc.entries.end());
        std::sort(l.entries.begin(), l.entries.end(),
                  [](const bind_layout_entry& a, const bind_layout_entry& b) { return a.binding < b.binding; });
        return bind_layouts_.alloc(std::move(l));
    }

    bind_group_handle device::create_bind_group(const bind_group_desc& desc) {
        if (!bind_layouts_.get(desc.layout)) {
            report_error("CreateBindGroup: unknown layout");
            return {};
        }
        bind_group g;
        g.layout = desc.layout;
        g.entries.assign(desc.entries.begin(), desc.entries.end());
        stats_.live_bind_groups++;
        return bind_groups_.alloc(std::move(g));
    }

    pipeline_layout_handle device::create_pipeline_layout(const pipeline_layout_desc& desc) {
        if (desc.push_constant_bytes)
            report_error("CreatePipelineLayout: d3d11 has no push constants");
        pipeline_layout l;
        l.desc = desc;
        l.desc.name = nullptr;
        return pipeline_layouts_.alloc(l);
    }

    query_set_handle device::create_query_set(const query_set_desc& desc) {
        query q;
        q.count = desc.count;
        return queries_.alloc(q);
    }

    void device::destroy(buffer_handle h) {
        buffer b;
        if (!buffers_.free(h, &b))
            return;
        stats_.buffer_bytes -= b.gpu_size * (b.constant ? 2 : 1);
        stats_.live_buffers--;
    }

    void device::destroy(texture_handle h) {
        texture t;
        if (!textures_.free(h, &t))
            return;
        if (!t.external)
            stats_.live_textures--;
    }

    void device::destroy(sampler_handle h) {
        samplers_.free(h);
    }
    void device::destroy(shader_handle h) {
        shaders_.free(h);
    }
    void device::destroy(bind_layout_handle h) {
        bind_layouts_.free(h);
    }
    void device::destroy(bind_group_handle h) {
        if (bind_groups_.free(h))
            stats_.live_bind_groups--;
    }
    void device::destroy(pipeline_layout_handle h) {
        pipeline_layouts_.free(h);
    }
    void device::destroy(pipeline_handle h) {
        if (pipelines_.free(h))
            stats_.live_pipelines--;
    }
    void device::destroy(query_set_handle h) {
        queries_.free(h);
    }
    void device::destroy(fence_handle h) {
        fences_.free(h);
    }

    const buffer_desc* device::get_desc(buffer_handle h) const {
        const buffer* b = buffers_.get(h);
        return b ? &b->desc : nullptr;
    }

    const texture_desc* device::get_desc(texture_handle h) const {
        const texture* t = textures_.get(h);
        return t ? &t->desc : nullptr;
    }

    uint8_t* device::map(buffer_handle handle) {
        buffer* b = buffers_.get(handle);
        return b && !b->shadow.empty() ? b->shadow.data() : nullptr;
    }

    void device::flush(buffer_handle handle, uint64_t offset, uint64_t size) {
        buffer* b = buffers_.get(handle);
        if (!b || b->shadow.empty() || b->desc.memory == memory_class::readback)
            return;
        const uint64_t lo = offset & ~3ull;
        const uint64_t hi = std::min<uint64_t>(b->gpu_size, align_up(offset + size, 4));
        if (hi <= lo)
            return;
        D3D11_BOX box{static_cast<UINT>(lo), 0, 0, static_cast<UINT>(hi), 1, 1};
        auto lock = lock_context();
        immediate_->UpdateSubresource(b->main.get(), 0, &box, b->shadow.data() + lo, 0, 0);
        if (b->constant)
            immediate_->CopySubresourceRegion(b->constant.get(), 0, static_cast<UINT>(lo), 0, 0, b->main.get(), 0,
                                              &box);
    }

    void device::invalidate(buffer_handle handle, uint64_t, uint64_t) {
        buffer* b = buffers_.get(handle);
        if (!b || b->desc.memory != memory_class::readback)
            return;
        auto lock = lock_context();
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (SUCCEEDED(immediate_->Map(b->main.get(), 0, D3D11_MAP_READ, 0, &mapped))) {
            std::memcpy(b->shadow.data(), mapped.pData, b->shadow.size());
            immediate_->Unmap(b->main.get(), 0);
        }
        std::vector<pending_readback> pending;
        {
            std::lock_guard views(view_mutex_);
            pending.swap(b->pending);
        }
        for (pending_readback& p : pending) {
            if (FAILED(immediate_->Map(p.staging.get(), 0, D3D11_MAP_READ, 0, &mapped)))
                continue;
            for (uint32_t y = 0; y < p.rows; y++) {
                const uint64_t at = p.buffer_offset + static_cast<uint64_t>(y) * p.buffer_row_pitch;
                if (at + p.row_bytes <= b->shadow.size())
                    std::memcpy(b->shadow.data() + at,
                                static_cast<const uint8_t*>(mapped.pData) + static_cast<size_t>(y) * mapped.RowPitch,
                                p.row_bytes);
            }
            immediate_->Unmap(p.staging.get(), 0);
        }
    }

    ID3D11ShaderResourceView* device::srv_of(texture& t, const subresource_range& range, texture_dim dim,
                                             conduit::format format, bool stencil) {
        const subresource_range r = resolve_range(range, t.desc);
        view_key key;
        key.kind = 0;
        key.dim = static_cast<uint8_t>(dim);
        key.format = static_cast<uint16_t>(format) | (stencil ? 0x8000 : 0);
        key.base_mip = r.base_mip;
        key.mip_count = r.mip_count;
        key.base_layer = r.base_layer;
        key.layer_count = r.layer_count;
        std::lock_guard lock(view_mutex_);
        if (const auto it = t.views.find(key); it != t.views.end())
            return static_cast<ID3D11ShaderResourceView*>(it->second.get());

        D3D11_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = format == conduit::format::unknown ? (stencil ? t.formats.stencil_view : t.formats.view)
                                                      : map_format(format).view;
        const bool ms = t.desc.samples > 1;
        switch (dim) {
        case texture_dim::tex2d:
            if (ms) {
                d.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
            } else if (r.base_layer > 0) {
                d.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
                d.Texture2DArray = {r.base_mip, r.mip_count, r.base_layer, 1};
            } else {
                d.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                d.Texture2D = {r.base_mip, r.mip_count};
            }
            break;
        case texture_dim::tex2d_array:
            if (ms) {
                d.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY;
                d.Texture2DMSArray = {r.base_layer, r.layer_count};
            } else {
                d.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DARRAY;
                d.Texture2DArray = {r.base_mip, r.mip_count, r.base_layer, r.layer_count};
            }
            break;
        case texture_dim::cube:
            d.ViewDimension = D3D11_SRV_DIMENSION_TEXTURECUBE;
            d.TextureCube = {r.base_mip, r.mip_count};
            break;
        case texture_dim::tex3d:
            d.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
            d.Texture3D = {r.base_mip, r.mip_count};
            break;
        }
        winrt::com_ptr<ID3D11ShaderResourceView> view;
        if (FAILED(device_->CreateShaderResourceView(t.resource.get(), &d, view.put())))
            return nullptr;
        t.views[key].copy_from(view.get());
        return view.get();
    }

    ID3D11UnorderedAccessView* device::uav_of(texture& t, const subresource_range& range, texture_dim dim) {
        const subresource_range r = resolve_range(range, t.desc);
        view_key key;
        key.kind = 3;
        key.dim = static_cast<uint8_t>(dim);
        key.base_mip = r.base_mip;
        key.base_layer = r.base_layer;
        key.layer_count = r.layer_count;
        std::lock_guard lock(view_mutex_);
        if (const auto it = t.views.find(key); it != t.views.end())
            return static_cast<ID3D11UnorderedAccessView*>(it->second.get());
        D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = t.formats.view;
        if (t.desc.dim == texture_dim::tex3d) {
            d.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
            d.Texture3D = {r.base_mip, 0, (std::max)(1u, depth_of(t.desc) >> r.base_mip)};
        } else if (dim == texture_dim::tex2d && r.base_layer == 0 && layer_count_of(t.desc) == 1) {
            d.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2D;
            d.Texture2D = {r.base_mip};
        } else {
            d.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE2DARRAY;
            d.Texture2DArray = {r.base_mip, r.base_layer, dim == texture_dim::tex2d ? 1u : r.layer_count};
        }
        winrt::com_ptr<ID3D11UnorderedAccessView> view;
        if (FAILED(device_->CreateUnorderedAccessView(t.resource.get(), &d, view.put())))
            return nullptr;
        t.views[key].copy_from(view.get());
        return view.get();
    }

    ID3D11RenderTargetView* device::rtv_of(texture& t, uint32_t mip, uint32_t layer) {
        view_key key;
        key.kind = 1;
        key.base_mip = mip;
        key.base_layer = layer;
        std::lock_guard lock(view_mutex_);
        if (const auto it = t.views.find(key); it != t.views.end())
            return static_cast<ID3D11RenderTargetView*>(it->second.get());
        D3D11_RENDER_TARGET_VIEW_DESC d{};
        d.Format = t.formats.view;
        const bool ms = t.desc.samples > 1;
        const bool array = layer_count_of(t.desc) > 1 || layer > 0;
        if (t.desc.dim == texture_dim::tex3d) {
            d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE3D;
            d.Texture3D = {mip, layer, ~0u};
        } else if (ms) {
            d.ViewDimension = array ? D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY : D3D11_RTV_DIMENSION_TEXTURE2DMS;
            if (array)
                d.Texture2DMSArray = {layer, 1};
        } else if (array) {
            d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2DARRAY;
            d.Texture2DArray = {mip, layer, 1};
        } else {
            d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            d.Texture2D = {mip};
        }
        winrt::com_ptr<ID3D11RenderTargetView> view;
        if (FAILED(device_->CreateRenderTargetView(t.resource.get(), &d, view.put())))
            return nullptr;
        t.views[key].copy_from(view.get());
        return view.get();
    }

    ID3D11DepthStencilView* device::dsv_of(texture& t, uint32_t mip, uint32_t layer, bool read_only) {
        view_key key;
        key.kind = read_only ? 5 : 2;
        key.base_mip = mip;
        key.base_layer = layer;
        std::lock_guard lock(view_mutex_);
        if (const auto it = t.views.find(key); it != t.views.end())
            return static_cast<ID3D11DepthStencilView*>(it->second.get());
        D3D11_DEPTH_STENCIL_VIEW_DESC d{};
        d.Format = t.formats.dsv;
        if (read_only)
            d.Flags = D3D11_DSV_READ_ONLY_DEPTH | (has_stencil(t.desc.format) ? D3D11_DSV_READ_ONLY_STENCIL : 0);
        const bool ms = t.desc.samples > 1;
        const bool array = layer_count_of(t.desc) > 1 || layer > 0;
        if (ms) {
            d.ViewDimension = array ? D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY : D3D11_DSV_DIMENSION_TEXTURE2DMS;
            if (array)
                d.Texture2DMSArray = {layer, 1};
        } else if (array) {
            d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2DARRAY;
            d.Texture2DArray = {mip, layer, 1};
        } else {
            d.ViewDimension = D3D11_DSV_DIMENSION_TEXTURE2D;
            d.Texture2D = {mip};
        }
        winrt::com_ptr<ID3D11DepthStencilView> view;
        if (FAILED(device_->CreateDepthStencilView(t.resource.get(), &d, view.put())))
            return nullptr;
        t.views[key].copy_from(view.get());
        return view.get();
    }

    ID3D11ShaderResourceView* device::buffer_srv(buffer& b, uint64_t offset, uint64_t size) {
        if (!size || offset + size > b.gpu_size)
            size = b.gpu_size - offset;
        const auto key = std::make_tuple(offset, size, uint8_t{0});
        std::lock_guard lock(view_mutex_);
        if (const auto it = b.views.find(key); it != b.views.end())
            return static_cast<ID3D11ShaderResourceView*>(it->second.get());
        D3D11_SHADER_RESOURCE_VIEW_DESC d{};
        if (b.structured) {
            d.Format = DXGI_FORMAT_UNKNOWN;
            d.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            d.Buffer.FirstElement = static_cast<UINT>(offset / b.desc.stride);
            d.Buffer.NumElements = static_cast<UINT>(size / b.desc.stride);
        } else {
            d.Format = DXGI_FORMAT_R32_TYPELESS;
            d.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
            d.BufferEx.FirstElement = static_cast<UINT>(offset / 4);
            d.BufferEx.NumElements = static_cast<UINT>(size / 4);
            d.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
        }
        winrt::com_ptr<ID3D11ShaderResourceView> view;
        if (FAILED(device_->CreateShaderResourceView(b.main.get(), &d, view.put())))
            return nullptr;
        b.views[key].copy_from(view.get());
        return view.get();
    }

    ID3D11UnorderedAccessView* device::buffer_uav(buffer& b, uint64_t offset, uint64_t size) {
        if (!size || offset + size > b.gpu_size)
            size = b.gpu_size - offset;
        const auto key = std::make_tuple(offset, size, uint8_t{1});
        std::lock_guard lock(view_mutex_);
        if (const auto it = b.views.find(key); it != b.views.end())
            return static_cast<ID3D11UnorderedAccessView*>(it->second.get());
        D3D11_UNORDERED_ACCESS_VIEW_DESC d{};
        d.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        if (b.structured) {
            d.Format = DXGI_FORMAT_UNKNOWN;
            d.Buffer.FirstElement = static_cast<UINT>(offset / b.desc.stride);
            d.Buffer.NumElements = static_cast<UINT>(size / b.desc.stride);
        } else {
            d.Format = DXGI_FORMAT_R32_TYPELESS;
            d.Buffer.FirstElement = static_cast<UINT>(offset / 4);
            d.Buffer.NumElements = static_cast<UINT>(size / 4);
            d.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
        }
        winrt::com_ptr<ID3D11UnorderedAccessView> view;
        if (FAILED(device_->CreateUnorderedAccessView(b.main.get(), &d, view.put())))
            return nullptr;
        b.views[key].copy_from(view.get());
        return view.get();
    }

    texture_handle device::wrap_backbuffer(ID3D11Texture2D* texture, const texture_desc& desc) {
        d3d11::texture t;
        t.resource.copy_from(texture);
        t.desc = desc;
        t.formats = map_format(desc.format);
        t.external = true;
        t.name = "swapchain image";
        return textures_.alloc(std::move(t));
    }

    void device::set_backbuffer(texture_handle handle, ID3D11Texture2D* texture) {
        d3d11::texture* t = textures_.get(handle);
        if (!t)
            return;
        std::lock_guard lock(view_mutex_);
        t->views.clear();
        t->resource.copy_from(texture);
    }

    void device::release_backbuffer(texture_handle handle) {
        destroy(handle);
    }

    bool device::compile_stage(const shader& sh, const pipeline_layout_desc& layout, uint32_t stage_index,
                               std::vector<stage_slot>& slots, winrt::com_ptr<ID3DBlob>& blob, std::string& error) {
        std::vector<const shader_resource*> resources;
        resources.reserve(sh.iface.resources.size());
        for (const shader_resource& r : sh.iface.resources)
            resources.push_back(&r);
        std::sort(resources.begin(), resources.end(), [](const shader_resource* a, const shader_resource* b) {
            return std::tie(a->set, a->binding, a->register_class) < std::tie(b->set, b->binding, b->register_class);
        });

        const uint32_t limits[4] = {14, 128, 16, stage_index == stage_compute ? 64u : 8u};
        uint32_t next[4] = {0, 0, 0, 0};
        struct edit {
            size_t begin, end;
            std::string text;
        };
        std::vector<edit> edits;
        for (const shader_resource* r : resources) {
            const int ci = class_index(r->register_class);
            if (next[ci] >= limits[ci]) {
                error = std::string("too many '") + r->register_class + "' registers in " + sh.entry;
                return false;
            }
            stage_slot s;
            s.set = r->set;
            s.binding = r->binding;
            s.kind = r->kind;
            s.dim = r->dim;
            s.register_class = r->register_class;
            s.slot = next[ci]++;
            if (r->set < max_bind_groups)
                if (const bind_layout* l = bind_layouts_.get(layout.sets[r->set])) {
                    uint32_t dynamic = 0;
                    for (const bind_layout_entry& e : l->entries) {
                        if (e.binding == r->binding) {
                            if (e.type == binding_type::uniform_buffer_dynamic)
                                s.dynamic_index = dynamic;
                            break;
                        }
                        if (e.type == binding_type::uniform_buffer_dynamic)
                            dynamic++;
                    }
                }
            slots.push_back(s);
            edits.push_back({r->clause_begin, r->clause_end,
                             std::string("register(") + r->register_class + std::to_string(s.slot) + ")"});
        }

        std::sort(edits.begin(), edits.end(), [](const edit& a, const edit& b) { return a.begin > b.begin; });
        std::string patched = sh.source;
        for (const edit& e : edits)
            patched.replace(e.begin, e.end - e.begin, e.text);

        const char* profile = stage_index == stage_vertex ? "vs_5_0" : stage_index == stage_pixel ? "ps_5_0" : "cs_5_0";
        const std::string key = std::string(profile) + '|' + sh.entry + '|' + patched;
        {
            std::lock_guard lock(shader_cache_mutex_);
            if (const auto it = blob_cache_.find(key); it != blob_cache_.end()) {
                blob = it->second;
                return true;
            }
        }
        if (!inst_->lib.compile) {
            error = "d3dcompiler_47.dll is not available";
            return false;
        }
        winrt::com_ptr<ID3DBlob> errors;
        const HRESULT hr = inst_->lib.compile(patched.data(), patched.size(), sh.entry.c_str(), nullptr, nullptr,
                                              sh.entry.c_str(), profile, optimize_full, 0, blob.put(), errors.put());
        if (FAILED(hr)) {
            error = errors ? std::string(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize())
                           : "D3DCompile failed";
            return false;
        }
        std::lock_guard lock(shader_cache_mutex_);
        blob_cache_[key] = blob;
        return true;
    }

    pipeline_handle device::create_graphics_pipeline(const graphics_pipeline_desc& desc) {
        const shader* vs = shaders_.get(desc.vertex);
        const shader* ps = shaders_.get(desc.fragment);
        const pipeline_layout* layout = pipeline_layouts_.get(desc.layout);
        if (!vs || !layout) {
            report_error("CreateGraphicsPipeline: missing shader or layout");
            return {};
        }
        pipeline p;
        std::string error;
        winrt::com_ptr<ID3DBlob> vs_blob, ps_blob;
        if (!compile_stage(*vs, layout->desc, stage_vertex, p.stages[stage_vertex], vs_blob, error) ||
            (ps && !compile_stage(*ps, layout->desc, stage_pixel, p.stages[stage_pixel], ps_blob, error))) {
            report_error(std::string("shader compile failed (") + (desc.name ? desc.name : "?") + "): " + error);
            return {};
        }
        if (FAILED(device_->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr,
                                               p.vs.put())))
            return {};
        if (ps_blob && FAILED(device_->CreatePixelShader(ps_blob->GetBufferPointer(), ps_blob->GetBufferSize(), nullptr,
                                                         p.ps.put())))
            return {};

        std::array<D3D11_INPUT_ELEMENT_DESC, max_vertex_attributes> elements{};
        for (uint32_t i = 0; i < desc.attribute_count; i++) {
            const vertex_attribute_desc& a = desc.attributes[i];
            D3D11_INPUT_ELEMENT_DESC& e = elements[i];
            e.SemanticName = "TEXCOORD";
            e.SemanticIndex = a.location;
            e.Format = map_format(a.format).view;
            e.InputSlot = a.stream;
            e.AlignedByteOffset = a.offset;
            const bool instance = a.stream < desc.stream_count && desc.streams[a.stream].step == vertex_step::instance;
            e.InputSlotClass = instance ? D3D11_INPUT_PER_INSTANCE_DATA : D3D11_INPUT_PER_VERTEX_DATA;
            e.InstanceDataStepRate = instance ? 1 : 0;
        }
        if (desc.attribute_count &&
            FAILED(device_->CreateInputLayout(elements.data(), desc.attribute_count, vs_blob->GetBufferPointer(),
                                              vs_blob->GetBufferSize(), p.input_layout.put()))) {
            report_error(std::string("input layout does not match the vertex shader (") +
                         (desc.name ? desc.name : "?") + ")");
            return {};
        }
        p.stream_count = desc.stream_count;
        for (uint32_t i = 0; i < desc.stream_count; i++)
            p.strides[i] = desc.streams[i].stride;
        p.topology = to_d3d(desc.topology);

        D3D11_RASTERIZER_DESC rd{};
        rd.FillMode = desc.raster.wireframe ? D3D11_FILL_WIREFRAME : D3D11_FILL_SOLID;
        rd.CullMode = desc.raster.cull == cull_mode::none   ? D3D11_CULL_NONE
                      : desc.raster.cull == cull_mode::back ? D3D11_CULL_BACK
                                                            : D3D11_CULL_FRONT;
        rd.FrontCounterClockwise = desc.raster.front_face == front_face::counter_clockwise;
        rd.DepthBias = static_cast<INT>(std::lround(desc.raster.depth_bias_constant));
        rd.DepthBiasClamp = desc.raster.depth_bias_clamp;
        rd.SlopeScaledDepthBias = desc.raster.depth_bias_slope;
        rd.DepthClipEnable = desc.raster.depth_clip;
        rd.ScissorEnable = TRUE;
        rd.MultisampleEnable = desc.samples > 1;
        if (FAILED(device_->CreateRasterizerState(&rd, p.raster.put())))
            return {};

        D3D11_DEPTH_STENCIL_DESC dd{};
        const depth_stencil_state& ds = desc.depth_stencil;
        const bool has_depth = desc.depth_format != format::unknown;
        dd.DepthEnable = has_depth && (ds.depth_test || ds.depth_write);
        dd.DepthWriteMask = ds.depth_write ? D3D11_DEPTH_WRITE_MASK_ALL : D3D11_DEPTH_WRITE_MASK_ZERO;
        dd.DepthFunc = ds.depth_test ? to_d3d(ds.depth_compare) : D3D11_COMPARISON_ALWAYS;
        dd.StencilEnable = has_depth && ds.stencil_test;
        dd.StencilReadMask = ds.stencil_read_mask;
        dd.StencilWriteMask = ds.stencil_write_mask;
        dd.FrontFace = {to_d3d(ds.front.fail), to_d3d(ds.front.depth_fail), to_d3d(ds.front.pass),
                        to_d3d(ds.front.compare)};
        dd.BackFace = {to_d3d(ds.back.fail), to_d3d(ds.back.depth_fail), to_d3d(ds.back.pass), to_d3d(ds.back.compare)};
        if (FAILED(device_->CreateDepthStencilState(&dd, p.depth_stencil.put())))
            return {};

        D3D11_BLEND_DESC bd{};
        bd.AlphaToCoverageEnable = desc.blend.alpha_to_coverage;
        bd.IndependentBlendEnable = TRUE;
        for (uint32_t i = 0; i < max_color_targets; i++) {
            const blend_target& t = desc.blend.targets[i];
            D3D11_RENDER_TARGET_BLEND_DESC& o = bd.RenderTarget[i];
            o.BlendEnable = t.enable;
            o.SrcBlend = t.enable ? to_d3d(t.src_color, false) : D3D11_BLEND_ONE;
            o.DestBlend = t.enable ? to_d3d(t.dst_color, false) : D3D11_BLEND_ZERO;
            o.BlendOp = t.enable ? to_d3d(t.color_op) : D3D11_BLEND_OP_ADD;
            o.SrcBlendAlpha = t.enable ? to_d3d(t.src_alpha, true) : D3D11_BLEND_ONE;
            o.DestBlendAlpha = t.enable ? to_d3d(t.dst_alpha, true) : D3D11_BLEND_ZERO;
            o.BlendOpAlpha = t.enable ? to_d3d(t.alpha_op) : D3D11_BLEND_OP_ADD;
            o.RenderTargetWriteMask = static_cast<UINT8>(t.write_mask);
        }
        if (FAILED(device_->CreateBlendState(&bd, p.blend.put())))
            return {};

        stats_.live_pipelines++;
        return pipelines_.alloc(std::move(p));
    }

    pipeline_handle device::create_compute_pipeline(const compute_pipeline_desc& desc) {
        const shader* cs = shaders_.get(desc.compute);
        const pipeline_layout* layout = pipeline_layouts_.get(desc.layout);
        if (!cs || !layout) {
            report_error("CreateComputePipeline: missing shader or layout");
            return {};
        }
        pipeline p;
        p.compute = true;
        std::string error;
        winrt::com_ptr<ID3DBlob> blob;
        if (!compile_stage(*cs, layout->desc, stage_compute, p.stages[stage_compute], blob, error)) {
            report_error(std::string("shader compile failed (") + (desc.name ? desc.name : "?") + "): " + error);
            return {};
        }
        if (FAILED(device_->CreateComputeShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, p.cs.put())))
            return {};
        stats_.live_pipelines++;
        return pipelines_.alloc(std::move(p));
    }

    fence_handle device::create_fence(uint64_t initial_value) {
        fence f;
        f.value = initial_value;
        return fences_.alloc(std::move(f));
    }

    void device::poll_fence(fence& f) {
        while (!f.pending.empty()) {
            auto& [value, query] = f.pending.front();
            if (immediate_->GetData(query.get(), nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK)
                break;
            f.value = (std::max)(f.value, value);
            free_queries_.push_back(std::move(query));
            f.pending.pop_front();
        }
    }

    void device::mark_submitted(fence_handle handle, uint64_t value) {
        fence* f = fences_.get(handle);
        if (!f)
            return;
        winrt::com_ptr<ID3D11Query> q;
        if (!free_queries_.empty()) {
            q = std::move(free_queries_.back());
            free_queries_.pop_back();
        } else {
            D3D11_QUERY_DESC qd{D3D11_QUERY_EVENT, 0};
            if (FAILED(device_->CreateQuery(&qd, q.put()))) {
                std::lock_guard lock(fence_mutex_);
                f->value = (std::max)(f->value, value);
                return;
            }
        }
        immediate_->End(q.get());
        std::lock_guard lock(fence_mutex_);
        f->pending.emplace_back(value, std::move(q));
    }

    uint64_t device::fence_value(fence_handle fence) {
        d3d11::fence* f = fences_.get(fence);
        if (!f)
            return 0;
        auto lock = lock_context();
        std::lock_guard fence_lock(fence_mutex_);
        poll_fence(*f);
        return f->value;
    }

    void device::signal_fence(fence_handle fence, uint64_t value) {
        if (d3d11::fence* f = fences_.get(fence)) {
            std::lock_guard lock(fence_mutex_);
            f->value = (std::max)(f->value, value);
        }
    }

    bool device::wait_fence(fence_handle fence, uint64_t value, uint64_t timeout_ns) {
        d3d11::fence* f = fences_.get(fence);
        if (!f)
            return true;
        const auto start = std::chrono::steady_clock::now();
        bool flushed = false;
        for (;;) {
            {
                auto lock = lock_context();
                std::lock_guard fence_lock(fence_mutex_);
                poll_fence(*f);
                if (f->value >= value)
                    return true;
                if (!flushed) {
                    immediate_->Flush();
                    flushed = true;
                }
            }
            if (timeout_ns != ~0ull && static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                 std::chrono::steady_clock::now() - start)
                                                                 .count()) >= timeout_ns)
                return false;
            std::this_thread::yield();
        }
    }

    void device::begin_frame() {
        frame_++;
        const uint64_t behind = (std::max)(1u, options_.frames_in_flight);
        if (frame_ > behind)
            wait_fence(frame_fence_, frame_ - behind, 2'000'000'000ull);
    }

    void device::end_frame() {
        auto lock = lock_context();
        mark_submitted(frame_fence_, frame_);
    }

    void device::wait_idle() {
        if (!device_)
            return;
        fence_handle idle = create_fence(0);
        {
            auto lock = lock_context();
            immediate_->Flush();
            mark_submitted(idle, 1);
        }
        wait_fence(idle, 1, 10'000'000'000ull);
        destroy(idle);
    }

    memory_budget device::query_memory_budget() {
        memory_budget m;
        if (auto a3 = adapter_.try_as<IDXGIAdapter3>()) {
            DXGI_QUERY_VIDEO_MEMORY_INFO local{}, host{};
            if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
                SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &host))) {
                m.device_budget = local.Budget;
                m.device_usage = local.CurrentUsage;
                m.host_budget = host.Budget;
                m.host_usage = host.CurrentUsage;
                m.valid = true;
                return m;
            }
        }
        m.device_usage = stats_.buffer_bytes + stats_.texture_bytes;
        return m;
    }

    command_list* device::acquire_command_list(queue_type queue) {
        std::lock_guard lock(list_mutex_);
        cmd_list* list;
        if (!free_lists_.empty()) {
            list = free_lists_.back();
            free_lists_.pop_back();
        } else {
            auto made = std::make_unique<cmd_list>(*this, queue);
            if (!made->init()) {
                report_error("could not create a deferred context");
                return nullptr;
            }
            lists_.push_back(std::move(made));
            list = lists_.back().get();
        }
        list->begin();
        return list;
    }

    void device::submit(queue_type, const submit_desc& desc) {
        std::vector<winrt::com_ptr<ID3D11CommandList>> commands;
        for (conduit::command_list* base : desc.lists) {
            auto* list = static_cast<cmd_list*>(base);
            if (ID3D11CommandList* cl = list->finish()) {
                winrt::com_ptr<ID3D11CommandList> owned;
                owned.copy_from(cl);
                commands.push_back(std::move(owned));
            }
        }
        {
            auto lock = lock_context();
            for (auto& cl : commands)
                immediate_->ExecuteCommandList(cl.get(), FALSE);
            for (const fence_op& op : desc.signals)
                mark_submitted(op.fence, op.value);
        }
        commands.clear();
        {
            std::lock_guard lock(list_mutex_);
            for (conduit::command_list* base : desc.lists) {
                auto* list = static_cast<cmd_list*>(base);
                list->release_finished();
                free_lists_.push_back(list);
            }
        }
        stats_.submits++;
    }

    std::unique_ptr<conduit::swapchain> device::create_swapchain(const swapchain_desc& desc) {
        return make_swapchain(*this, desc);
    }
}
