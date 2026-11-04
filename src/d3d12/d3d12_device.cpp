#include "d3d12/d3d12_device.hpp"

#include "d3d12/d3d12_convert.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace conduit::d3d12 {
    namespace {
        uint64_t fnv(const void* data, size_t size, uint64_t h = 1469598103934665603ull) {
            const auto* p = static_cast<const uint8_t*>(data);
            for (size_t i = 0; i < size; i++)
                h = (h ^ p[i]) * 1099511628211ull;
            return h;
        }

        std::wstring hash_name(const char* prefix, uint64_t hash) {
            wchar_t buf[40];
            swprintf(buf, 40, L"%hs%016llx", prefix, static_cast<unsigned long long>(hash));
            return buf;
        }

        std::wstring widen(const char* s) {
            std::wstring out;
            for (; s && *s; s++)
                out.push_back(static_cast<unsigned char>(*s));
            return out;
        }

        D3D12_SHADER_RESOURCE_VIEW_DESC buffer_srv(uint64_t offset, uint64_t size, uint32_t stride) {
            D3D12_SHADER_RESOURCE_VIEW_DESC d{};
            d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            if (stride) {
                d.Format = DXGI_FORMAT_UNKNOWN;
                d.Buffer.FirstElement = offset / stride;
                d.Buffer.NumElements = static_cast<UINT>(size / stride);
                d.Buffer.StructureByteStride = stride;
            } else {
                d.Format = DXGI_FORMAT_R32_TYPELESS;
                d.Buffer.FirstElement = offset / 4;
                d.Buffer.NumElements = static_cast<UINT>(size / 4);
                d.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            }
            return d;
        }

        D3D12_UNORDERED_ACCESS_VIEW_DESC buffer_uav(uint64_t offset, uint64_t size, uint32_t stride) {
            D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
            d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            if (stride) {
                d.Format = DXGI_FORMAT_UNKNOWN;
                d.Buffer.FirstElement = offset / stride;
                d.Buffer.NumElements = static_cast<UINT>(size / stride);
                d.Buffer.StructureByteStride = stride;
            } else {
                d.Format = DXGI_FORMAT_R32_TYPELESS;
                d.Buffer.FirstElement = offset / 4;
                d.Buffer.NumElements = static_cast<UINT>(size / 4);
                d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            }
            return d;
        }
    }

    buffer_handle device::create_buffer(const buffer_desc& desc) {
        if (desc.size == 0) {
            report_error("CreateBuffer: zero size");
            return {};
        }
        buffer_rec rec;
        rec.desc = desc;
        rec.name = desc.name ? desc.name : "";

        // constant buffer views must span a multiple of 256 bytes
        uint64_t size = desc.size;
        if (has(desc.usage, buffer_usage::uniform))
            size = align_up(size, 256);

        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = size;
        rd.Height = rd.DepthOrArraySize = rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (has(desc.usage, buffer_usage::storage))
            rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        memory_class memory = desc.memory;
        if (memory == memory_class::gpu_mappable)
            memory = memory_class::upload;
        D3D12MA::ALLOCATION_DESC ad{};
        D3D12_RESOURCE_STATES initial = D3D12_RESOURCE_STATE_COMMON;
        switch (memory) {
        case memory_class::upload:
            ad.HeapType = D3D12_HEAP_TYPE_UPLOAD;
            initial = D3D12_RESOURCE_STATE_GENERIC_READ;
            break;
        case memory_class::readback:
            ad.HeapType = D3D12_HEAP_TYPE_READBACK;
            initial = D3D12_RESOURCE_STATE_COPY_DEST;
            break;
        default:
            ad.HeapType = D3D12_HEAP_TYPE_DEFAULT;
            break;
        }
        rec.state = initial;
        if (FAILED(allocator_->CreateResource(&ad, &rd, initial, nullptr, rec.alloc.put(),
                                              winrt::guid_of<ID3D12Resource>(), rec.resource.put_void()))) {
            report_error("CreateBuffer '" + rec.name + "' failed");
            return {};
        }
        if (memory != memory_class::gpu_only) {
            D3D12_RANGE none{0, 0};
            void* ptr = nullptr;
            if (SUCCEEDED(rec.resource->Map(0, memory == memory_class::readback ? nullptr : &none, &ptr)))
                rec.mapped = static_cast<uint8_t*>(ptr);
        }
        rec.address = rec.resource->GetGPUVirtualAddress();
        if (!rec.name.empty())
            rec.resource->SetName(widen(rec.name.c_str()).c_str());
        buffer_bytes_ += static_cast<int64_t>(desc.size);
        live_buffers_++;

        buffer_handle h = buffers_.alloc(std::move(rec));
        buffer_rec* stored = buffers_.get(h);
        stored->desc.name = stored->name.c_str();
        stored->desc.memory = memory;
        if (use_validator_)
            validator_.on_create(h, stored->desc);
        return h;
    }

    texture_handle device::create_texture(const texture_desc& desc) {
        if (desc.format == format::unknown || desc.width == 0 || desc.height == 0) {
            report_error("CreateTexture: invalid description");
            return {};
        }
        texture_rec rec;
        rec.desc = desc;
        rec.desc.mips = (std::max)(1u, desc.mips);
        rec.name = desc.name ? desc.name : "";
        const format_map fm = map_format(desc.format);
        rec.resource_format = fm.resource;
        rec.view_format = fm.view;
        rec.dsv_format = fm.dsv;
        rec.planes = fm.planes;
        rec.mips = rec.desc.mips;
        rec.layers = layer_count_of(desc);

        D3D12_RESOURCE_DESC rd{};
        rd.Dimension =
            desc.dim == texture_dim::tex3d ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rd.Width = desc.width;
        rd.Height = desc.height;
        rd.DepthOrArraySize = static_cast<UINT16>(desc.dim == texture_dim::tex3d ? depth_of(desc) : rec.layers);
        rd.MipLevels = static_cast<UINT16>(rec.mips);
        rd.Format = fm.resource;
        rd.SampleDesc.Count = (std::max)(1u, desc.samples);
        rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        const bool depth = fm.dsv != DXGI_FORMAT_UNKNOWN;
        if (has(desc.usage, texture_usage::color_target))
            rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (has(desc.usage, texture_usage::depth_target))
            rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        if (has(desc.usage, texture_usage::storage))
            rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        if (desc.cross_queue && !depth && desc.samples <= 1) {
            rd.Flags |= D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
            rec.simultaneous = true;
        }
        if (depth && has(desc.usage, texture_usage::depth_target) && !has(desc.usage, texture_usage::sampled))
            rd.Flags |= D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;

        D3D12_CLEAR_VALUE clear{};
        const D3D12_CLEAR_VALUE* p_clear = nullptr;
        if (has(desc.usage, texture_usage::depth_target)) {
            clear.Format = fm.dsv;
            clear.DepthStencil = {1.0f, 0};
            p_clear = &clear;
        } else if (has(desc.usage, texture_usage::color_target)) {
            clear.Format = fm.view;
            std::memcpy(clear.Color, desc.optimized_clear.data(), sizeof(float) * 4);
            p_clear = &clear;
        }

        D3D12MA::ALLOCATION_DESC ad{};
        ad.HeapType = D3D12_HEAP_TYPE_DEFAULT;
        if (p_clear)
            ad.Flags = D3D12MA::ALLOCATION_FLAG_COMMITTED;
        if (FAILED(allocator_->CreateResource(&ad, &rd, D3D12_RESOURCE_STATE_COMMON, p_clear, rec.alloc.put(),
                                              winrt::guid_of<ID3D12Resource>(), rec.resource.put_void()))) {
            report_error("CreateTexture '" + rec.name + "' failed");
            return {};
        }
        if (!rec.name.empty())
            rec.resource->SetName(widen(rec.name.c_str()).c_str());
        rec.states.assign(static_cast<size_t>(rec.mips) * (desc.dim == texture_dim::tex3d ? 1 : rec.layers) *
                              rec.planes,
                          D3D12_RESOURCE_STATE_COMMON);

        uint64_t bytes = 0;
        for (uint32_t m = 0; m < rec.mips; m++)
            bytes += static_cast<uint64_t>(row_bytes(desc.format, mip_extent(desc.width, m))) *
                     row_blocks(desc.format, mip_extent(desc.height, m)) * depth_of(desc) * rec.layers *
                     (std::max)(1u, desc.samples);
        texture_bytes_ += static_cast<int64_t>(bytes);
        live_textures_++;

        texture_handle h = textures_.alloc(std::move(rec));
        texture_rec* stored = textures_.get(h);
        stored->desc.name = stored->name.c_str();
        if (use_validator_)
            validator_.on_create(h, stored->desc);
        return h;
    }

    texture_handle device::wrap_swapchain_image(ID3D12Resource* resource, const texture_desc& desc) {
        texture_rec rec;
        rec.resource.copy_from(resource);
        rec.desc = desc;
        rec.name = "backbuffer";
        const format_map fm = map_format(desc.format);
        rec.resource_format = fm.resource;
        rec.view_format = fm.view;
        rec.mips = rec.layers = rec.planes = 1;
        rec.external = true;
        rec.states.assign(1, D3D12_RESOURCE_STATE_COMMON);
        texture_handle h = textures_.alloc(std::move(rec));
        texture_rec* stored = textures_.get(h);
        stored->desc.name = stored->name.c_str();
        if (use_validator_)
            validator_.on_create(h, stored->desc);
        return h;
    }

    void device::destroy_wrapped(texture_handle h) {
        texture_rec rec;
        if (!textures_.free(h, &rec))
            return;
        if (use_validator_)
            validator_.on_destroy(h);
        for (auto& [k, v] : rec.views.rtv)
            rtv_pool_.free(v);
    }

    D3D12_CPU_DESCRIPTOR_HANDLE device::render_view(texture_rec& tex, uint32_t mip, uint32_t layer, bool depth,
                                                    bool read_only) {
        const uint64_t key = mip | (static_cast<uint64_t>(layer) << 8) | (read_only ? 1ull << 41 : 0);
        std::lock_guard lock(view_mutex_);
        auto& map = depth ? tex.views.dsv : tex.views.rtv;
        auto it = map.find(key);
        if (it != map.end())
            return depth ? dsv_pool_.handle(it->second) : rtv_pool_.handle(it->second);
        const bool ms = tex.desc.samples > 1;
        if (depth) {
            const uint32_t index = dsv_pool_.alloc();
            D3D12_DEPTH_STENCIL_VIEW_DESC dv{};
            dv.Format = tex.dsv_format;
            if (read_only)
                dv.Flags = D3D12_DSV_FLAG_READ_ONLY_DEPTH |
                           (tex.planes > 1 ? D3D12_DSV_FLAG_READ_ONLY_STENCIL : D3D12_DSV_FLAG_NONE);
            dv.ViewDimension = ms ? D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY : D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
            if (ms) {
                dv.Texture2DMSArray.FirstArraySlice = layer;
                dv.Texture2DMSArray.ArraySize = 1;
            } else {
                dv.Texture2DArray.MipSlice = mip;
                dv.Texture2DArray.FirstArraySlice = layer;
                dv.Texture2DArray.ArraySize = 1;
            }
            device_->CreateDepthStencilView(tex.resource.get(), &dv, dsv_pool_.handle(index));
            map.emplace(key, index);
            return dsv_pool_.handle(index);
        }
        const uint32_t index = rtv_pool_.alloc();
        D3D12_RENDER_TARGET_VIEW_DESC rv{};
        rv.Format = tex.view_format;
        rv.ViewDimension = ms ? D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY : D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
        if (ms) {
            rv.Texture2DMSArray.FirstArraySlice = layer;
            rv.Texture2DMSArray.ArraySize = 1;
        } else {
            rv.Texture2DArray.MipSlice = mip;
            rv.Texture2DArray.FirstArraySlice = layer;
            rv.Texture2DArray.ArraySize = 1;
        }
        device_->CreateRenderTargetView(tex.resource.get(), &rv, rtv_pool_.handle(index));
        map.emplace(key, index);
        return rtv_pool_.handle(index);
    }

    sampler_handle device::create_sampler(const sampler_desc& desc) {
        std::lock_guard lock(sampler_mutex_);
        for (auto& [d, h] : sampler_cache_)
            if (d == desc)
                return h;
        sampler_rec rec;
        rec.desc = desc;
        sampler_handle h = samplers_.alloc(rec);
        sampler_cache_.emplace_back(desc, h);
        return h;
    }

    void device::write_sampler_desc(D3D12_CPU_DESCRIPTOR_HANDLE dst, const sampler_desc& d) {
        D3D12_SAMPLER_DESC sd{};
        sd.Filter = to_d3d(d);
        sd.AddressU = to_d3d(d.u);
        sd.AddressV = to_d3d(d.v);
        sd.AddressW = to_d3d(d.w);
        sd.MipLODBias = d.lod_bias;
        sd.MaxAnisotropy = static_cast<UINT>(std::clamp(d.max_anisotropy, 1.0f, 16.0f));
        sd.ComparisonFunc = d.compare ? to_d3d(d.compare_op) : D3D12_COMPARISON_FUNC_NEVER;
        std::memcpy(sd.BorderColor, d.border.data(), sizeof(float) * 4);
        sd.MinLOD = d.min_lod;
        sd.MaxLOD = d.max_lod;
        device_->CreateSampler(&sd, dst);
    }

    int device::acquire_sampler_table(const std::vector<uint32_t>& key, const std::vector<sampler_desc>& descs) {
        std::lock_guard lock(sampler_mutex_);
        for (size_t i = 0; i < sampler_tables_.size(); i++)
            if (sampler_tables_[i].refs > 0 && sampler_tables_[i].key == key) {
                sampler_tables_[i].refs++;
                return static_cast<int>(i);
            }
        uint32_t offset = 0;
        if (!sampler_heap_.alloc(static_cast<uint32_t>(descs.size()), offset)) {
            report_error("sampler descriptor heap exhausted");
            return -1;
        }
        for (size_t i = 0; i < descs.size(); i++)
            write_sampler_desc(sampler_heap_.cpu(offset + static_cast<uint32_t>(i)), descs[i]);
        sampler_table table;
        table.offset = offset;
        table.count = static_cast<uint32_t>(descs.size());
        table.refs = 1;
        table.key = key;
        for (size_t i = 0; i < sampler_tables_.size(); i++)
            if (sampler_tables_[i].refs == 0) {
                sampler_tables_[i] = std::move(table);
                return static_cast<int>(i);
            }
        sampler_tables_.push_back(std::move(table));
        return static_cast<int>(sampler_tables_.size() - 1);
    }

    void device::release_sampler_table(int index) {
        if (index < 0)
            return;
        std::lock_guard lock(sampler_mutex_);
        sampler_table& t = sampler_tables_[static_cast<size_t>(index)];
        if (--t.refs == 0) {
            sampler_heap_.free(t.offset, t.count);
            t.key.clear();
        }
    }

    shader_handle device::create_shader(const shader_desc& desc) {
        if (desc.format != shader_format::dxil && desc.format != shader_format::dxbc) {
            report_error("CreateShader: expects DXIL");
            return {};
        }
        shader_rec rec;
        rec.code.assign(desc.code.begin(), desc.code.end());
        rec.stage = desc.stage;
        return shaders_.alloc(std::move(rec));
    }

    bind_layout_handle device::create_bind_layout(const bind_layout_desc& desc) {
        bind_layout_rec rec;
        rec.entries.assign(desc.entries.begin(), desc.entries.end());
        for (const bind_layout_entry& e : rec.entries) {
            const uint32_t count = (std::max)(1u, e.count);
            switch (e.type) {
            case binding_type::uniform_buffer_dynamic:
                rec.dynamic_bindings.push_back(e.binding);
                rec.table_offset.push_back(0);
                break;
            case binding_type::sampler:
                rec.table_offset.push_back(rec.sampler_count);
                rec.sampler_count += count;
                break;
            default:
                rec.table_offset.push_back(rec.table_count);
                rec.table_count += count;
                break;
            }
        }
        std::sort(rec.dynamic_bindings.begin(), rec.dynamic_bindings.end());
        rec.dynamic_count = static_cast<uint32_t>(rec.dynamic_bindings.size());
        if (rec.dynamic_count > max_dynamic_offsets) {
            report_error("CreateBindLayout: too many dynamic uniform buffers");
            return {};
        }
        return bind_layouts_.alloc(std::move(rec));
    }

    bind_group_handle device::create_bind_group(const bind_group_desc& desc) {
        bind_layout_rec* layout = bind_layouts_.get(desc.layout);
        if (!layout) {
            report_error("CreateBindGroup: unknown layout");
            return {};
        }
        bind_group_rec rec;
        rec.layout = desc.layout;
        rec.per_frame = desc.per_frame;
        rec.dynamic_count = layout->dynamic_count;
        rec.table_count = layout->table_count;
        if (rec.table_count && !resource_heap_.alloc(rec.table_count, rec.table_offset)) {
            report_error("CreateBindGroup: descriptor heap exhausted");
            return {};
        }

        std::vector<uint32_t> sampler_key(layout->sampler_count, 0);
        std::vector<sampler_desc> sampler_descs(layout->sampler_count);

        for (const bind_group_entry& e : desc.entries) {
            size_t index = SIZE_MAX;
            for (size_t i = 0; i < layout->entries.size(); i++)
                if (layout->entries[i].binding == e.binding)
                    index = i;
            if (index == SIZE_MAX) {
                report_error("CreateBindGroup: binding " + std::to_string(e.binding) + " is not in the layout");
                continue;
            }
            const bind_layout_entry& le = layout->entries[index];
            const uint32_t slot_index = layout->table_offset[index] + e.array_index;

            switch (le.type) {
            case binding_type::uniform_buffer:
            case binding_type::uniform_buffer_dynamic:
            case binding_type::storage_buffer:
            case binding_type::storage_buffer_rw: {
                buffer_rec* b = buffers_.get(e.buffer.buffer);
                if (!b) {
                    report_error("CreateBindGroup: binding " + std::to_string(e.binding) + " has no buffer");
                    continue;
                }
                if (use_validator_) {
                    state_validator::bound_resource br;
                    br.buffer = e.buffer.buffer;
                    br.want = le.type == binding_type::storage_buffer_rw ? state::shader_write
                              : le.type == binding_type::storage_buffer  ? state::shader_read
                                                                         : state::uniform_buffer;
                    rec.bound.push_back(br);
                }
                const uint64_t size = e.buffer.size ? e.buffer.size : b->desc.size - e.buffer.offset;
                if (le.type == binding_type::uniform_buffer_dynamic) {
                    const auto it =
                        std::find(layout->dynamic_bindings.begin(), layout->dynamic_bindings.end(), le.binding);
                    rec.dynamic_base[static_cast<size_t>(it - layout->dynamic_bindings.begin())] =
                        b->address + e.buffer.offset;
                } else if (le.type == binding_type::uniform_buffer) {
                    D3D12_CONSTANT_BUFFER_VIEW_DESC cv{b->address + e.buffer.offset,
                                                       static_cast<UINT>(align_up(size, 256))};
                    device_->CreateConstantBufferView(&cv, resource_heap_.cpu(rec.table_offset + slot_index));
                } else if (le.type == binding_type::storage_buffer) {
                    const auto sv = buffer_srv(e.buffer.offset, size, b->desc.stride);
                    device_->CreateShaderResourceView(b->resource.get(), &sv,
                                                      resource_heap_.cpu(rec.table_offset + slot_index));
                } else {
                    const auto uv = buffer_uav(e.buffer.offset, size, b->desc.stride);
                    device_->CreateUnorderedAccessView(b->resource.get(), nullptr, &uv,
                                                       resource_heap_.cpu(rec.table_offset + slot_index));
                }
                break;
            }
            case binding_type::sampled_texture:
            case binding_type::storage_texture_rw: {
                texture_rec* t = textures_.get(e.texture.texture);
                if (!t) {
                    report_error("CreateBindGroup: binding " + std::to_string(e.binding) + " has no texture");
                    continue;
                }
                if (use_validator_) {
                    state_validator::bound_resource br;
                    br.texture = e.texture.texture;
                    br.range = e.texture.range;
                    br.want =
                        le.type == binding_type::storage_texture_rw ? state::shader_write : state::depth_read_sampled;
                    rec.bound.push_back(br);
                }
                const subresource_range r = resolve_range(e.texture.range, t->desc);
                const texture_dim dim = e.texture.view_dim != texture_dim::tex2d ? e.texture.view_dim : le.view_dim;
                const format_map override = map_format(e.texture.format);
                const DXGI_FORMAT fmt = e.texture.stencil                     ? map_format(t->desc.format).stencil_view
                                        : e.texture.format != format::unknown ? override.view
                                                                              : t->view_format;
                if (le.type == binding_type::sampled_texture) {
                    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
                    sv.Format = fmt;
                    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                    const bool ms = t->desc.samples > 1;
                    switch (dim) {
                    case texture_dim::tex2d:
                        if (ms) {
                            sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS;
                        } else {
                            sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
                            sv.Texture2D = {r.base_mip, r.mip_count, e.texture.stencil ? 1u : 0u, 0.0f};
                        }
                        break;
                    case texture_dim::tex2d_array:
                        sv.ViewDimension =
                            ms ? D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY : D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
                        if (ms)
                            sv.Texture2DMSArray = {r.base_layer, r.layer_count};
                        else
                            sv.Texture2DArray = {
                                r.base_mip, r.mip_count, r.base_layer, r.layer_count, e.texture.stencil ? 1u : 0u,
                                0.0f};
                        break;
                    case texture_dim::cube:
                        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBE;
                        sv.TextureCube = {r.base_mip, r.mip_count, 0.0f};
                        break;
                    case texture_dim::tex3d:
                        sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
                        sv.Texture3D = {r.base_mip, r.mip_count, 0.0f};
                        break;
                    }
                    device_->CreateShaderResourceView(t->resource.get(), &sv,
                                                      resource_heap_.cpu(rec.table_offset + slot_index));
                } else {
                    D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};
                    uv.Format = fmt;
                    switch (dim) {
                    case texture_dim::tex2d:
                        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
                        uv.Texture2D = {r.base_mip, 0};
                        break;
                    case texture_dim::tex3d:
                        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
                        uv.Texture3D = {
                            r.base_mip, 0,
                            depth_of(t->desc) >> r.base_mip ? (std::max)(1u, depth_of(t->desc) >> r.base_mip) : 1u};
                        break;
                    default:
                        uv.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
                        uv.Texture2DArray = {r.base_mip, r.base_layer, r.layer_count, 0};
                        break;
                    }
                    device_->CreateUnorderedAccessView(t->resource.get(), nullptr, &uv,
                                                       resource_heap_.cpu(rec.table_offset + slot_index));
                }
                break;
            }
            case binding_type::sampler: {
                sampler_rec* s = samplers_.get(e.sampler);
                if (!s) {
                    report_error("CreateBindGroup: binding " + std::to_string(e.binding) + " has no sampler");
                    continue;
                }
                const uint32_t at = layout->table_offset[index] + e.array_index;
                sampler_key[at] = e.sampler.value;
                sampler_descs[at] = s->desc;
                break;
            }
            }
        }

        if (layout->sampler_count) {
            rec.sampler_table = acquire_sampler_table(sampler_key, sampler_descs);
            if (rec.sampler_table >= 0) {
                std::lock_guard lock(sampler_mutex_);
                rec.sampler_offset = sampler_tables_[static_cast<size_t>(rec.sampler_table)].offset;
            }
        }
        live_groups_++;
        bind_group_handle h = bind_groups_.alloc(std::move(rec));
        if (desc.per_frame) {
            std::lock_guard lock(frame_group_mutex_);
            frame_groups_[slot_].groups.push_back(h);
        }
        return h;
    }

    pipeline_layout_handle device::create_pipeline_layout(const pipeline_layout_desc& desc) {
        pipeline_layout_rec rec;
        rec.sets = desc.sets;
        rec.table_param.fill(-1);
        rec.sampler_param.fill(-1);
        for (auto& row : rec.dynamic_param)
            row.fill(-1);
        rec.push_bytes = desc.push_constant_bytes;

        std::array<std::vector<D3D12_DESCRIPTOR_RANGE1>, max_bind_groups> table_ranges;
        std::array<std::vector<D3D12_DESCRIPTOR_RANGE1>, max_bind_groups> sampler_ranges;
        for (uint32_t s = 0; s < max_bind_groups; s++) {
            bind_layout_rec* l = bind_layouts_.get(desc.sets[s]);
            if (!l)
                continue;
            table_ranges[s].reserve(l->entries.size());
            sampler_ranges[s].reserve(l->entries.size());
            for (size_t i = 0; i < l->entries.size(); i++) {
                const bind_layout_entry& e = l->entries[i];
                D3D12_DESCRIPTOR_RANGE1 r{};
                r.NumDescriptors = (std::max)(1u, e.count);
                r.BaseShaderRegister = e.binding;
                r.RegisterSpace = s;
                r.OffsetInDescriptorsFromTableStart = l->table_offset[i];
                switch (e.type) {
                case binding_type::uniform_buffer:
                    r.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
                    break;
                case binding_type::storage_buffer:
                case binding_type::sampled_texture:
                    r.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
                    break;
                case binding_type::storage_buffer_rw:
                case binding_type::storage_texture_rw:
                    r.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                    break;
                case binding_type::sampler:
                    r.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER;
                    break;
                case binding_type::uniform_buffer_dynamic:
                    continue;
                }
                (e.type == binding_type::sampler ? sampler_ranges[s] : table_ranges[s]).push_back(r);
            }
        }

        std::vector<D3D12_ROOT_PARAMETER1> params;
        params.reserve(2 + max_bind_groups * (2 + max_dynamic_offsets));
        if (desc.push_constant_bytes) {
            D3D12_ROOT_PARAMETER1 p{};
            p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            p.Constants = {0, 15, desc.push_constant_bytes / 4};
            p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            rec.push_param = static_cast<int>(params.size());
            params.push_back(p);
        }
        for (uint32_t s = 0; s < max_bind_groups; s++) {
            bind_layout_rec* l = bind_layouts_.get(desc.sets[s]);
            if (!l)
                continue;
            if (!table_ranges[s].empty()) {
                D3D12_ROOT_PARAMETER1 p{};
                p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                p.DescriptorTable = {static_cast<UINT>(table_ranges[s].size()), table_ranges[s].data()};
                p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                rec.table_param[s] = static_cast<int>(params.size());
                params.push_back(p);
            }
            if (!sampler_ranges[s].empty()) {
                D3D12_ROOT_PARAMETER1 p{};
                p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                p.DescriptorTable = {static_cast<UINT>(sampler_ranges[s].size()), sampler_ranges[s].data()};
                p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                rec.sampler_param[s] = static_cast<int>(params.size());
                params.push_back(p);
            }
            for (size_t k = 0; k < l->dynamic_bindings.size(); k++) {
                D3D12_ROOT_PARAMETER1 p{};
                p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
                p.Descriptor = {l->dynamic_bindings[k], s, D3D12_ROOT_DESCRIPTOR_FLAG_NONE};
                p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
                rec.dynamic_param[s][k] = static_cast<int>(params.size());
                params.push_back(p);
            }
        }

        D3D12_VERSIONED_ROOT_SIGNATURE_DESC vd{};
        vd.Version = D3D_ROOT_SIGNATURE_VERSION_1_1;
        vd.Desc_1_1.NumParameters = static_cast<UINT>(params.size());
        vd.Desc_1_1.pParameters = params.data();
        vd.Desc_1_1.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        winrt::com_ptr<ID3DBlob> blob, error;
        if (FAILED(inst_->lib.serialize_root_signature(&vd, blob.put(), error.put()))) {
            report_error(std::string("root signature serialization failed: ") +
                         (error ? static_cast<const char*>(error->GetBufferPointer()) : "unknown"));
            return {};
        }
        if (FAILED(device_->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                                winrt::guid_of<ID3D12RootSignature>(),
                                                rec.root_signature.put_void()))) {
            report_error("CreateRootSignature failed (more than 64 root dwords?)");
            return {};
        }
        rec.hash = fnv(blob->GetBufferPointer(), blob->GetBufferSize());
        if (desc.name)
            rec.root_signature->SetName(widen(desc.name).c_str());
        return pipeline_layouts_.alloc(std::move(rec));
    }

    pipeline_handle device::create_graphics_pipeline(const graphics_pipeline_desc& desc) {
        pipeline_layout_rec* layout = pipeline_layouts_.get(desc.layout);
        shader_rec* vs = shaders_.get(desc.vertex);
        shader_rec* fs = shaders_.get(desc.fragment);
        if (!layout || !vs) {
            report_error("CreateGraphicsPipeline: missing layout or vertex shader");
            return {};
        }

        D3D12_GRAPHICS_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = layout->root_signature.get();
        pd.VS = {vs->code.data(), vs->code.size()};
        if (fs)
            pd.PS = {fs->code.data(), fs->code.size()};

        pd.BlendState.AlphaToCoverageEnable = desc.blend.alpha_to_coverage;
        pd.BlendState.IndependentBlendEnable = TRUE;
        for (uint32_t i = 0; i < max_color_targets; i++) {
            const blend_target& t = desc.blend.targets[i];
            D3D12_RENDER_TARGET_BLEND_DESC& rt = pd.BlendState.RenderTarget[i];
            rt.BlendEnable = t.enable;
            rt.SrcBlend = to_d3d(t.src_color, false);
            rt.DestBlend = to_d3d(t.dst_color, false);
            rt.BlendOp = to_d3d(t.color_op);
            rt.SrcBlendAlpha = to_d3d(t.src_alpha, true);
            rt.DestBlendAlpha = to_d3d(t.dst_alpha, true);
            rt.BlendOpAlpha = to_d3d(t.alpha_op);
            rt.RenderTargetWriteMask = static_cast<UINT8>(t.write_mask);
            rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        }
        pd.SampleMask = UINT_MAX;

        D3D12_RASTERIZER_DESC& rs = pd.RasterizerState;
        rs.FillMode = desc.raster.wireframe ? D3D12_FILL_MODE_WIREFRAME : D3D12_FILL_MODE_SOLID;
        rs.CullMode = desc.raster.cull == cull_mode::none   ? D3D12_CULL_MODE_NONE
                      : desc.raster.cull == cull_mode::back ? D3D12_CULL_MODE_BACK
                                                            : D3D12_CULL_MODE_FRONT;
        rs.FrontCounterClockwise = desc.raster.front_face == front_face::counter_clockwise;
        rs.DepthBias = static_cast<INT>(desc.raster.depth_bias_constant);
        rs.DepthBiasClamp = desc.raster.depth_bias_clamp;
        rs.SlopeScaledDepthBias = desc.raster.depth_bias_slope;
        rs.DepthClipEnable = desc.raster.depth_clip;
        rs.MultisampleEnable = desc.samples > 1;

        D3D12_DEPTH_STENCIL_DESC& ds = pd.DepthStencilState;
        ds.DepthEnable = desc.depth_stencil.depth_test;
        ds.DepthWriteMask = desc.depth_stencil.depth_write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
        ds.DepthFunc = to_d3d(desc.depth_stencil.depth_compare);
        ds.StencilEnable = desc.depth_stencil.stencil_test;
        ds.StencilReadMask = desc.depth_stencil.stencil_read_mask;
        ds.StencilWriteMask = desc.depth_stencil.stencil_write_mask;
        auto face = [](const stencil_face& f) {
            return D3D12_DEPTH_STENCILOP_DESC{to_d3d(f.fail), to_d3d(f.depth_fail), to_d3d(f.pass), to_d3d(f.compare)};
        };
        ds.FrontFace = face(desc.depth_stencil.front);
        ds.BackFace = face(desc.depth_stencil.back);

        D3D12_INPUT_ELEMENT_DESC elements[max_vertex_attributes];
        for (uint32_t i = 0; i < desc.attribute_count; i++) {
            const vertex_attribute_desc& a = desc.attributes[i];
            const bool instance = a.stream < desc.stream_count && desc.streams[a.stream].step == vertex_step::instance;
            elements[i] = {"TEXCOORD",
                           a.location,
                           map_format(a.format).view,
                           a.stream,
                           a.offset,
                           instance ? D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA
                                    : D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
                           instance ? 1u : 0u};
        }
        pd.InputLayout = {elements, desc.attribute_count};
        pd.PrimitiveTopologyType = to_d3d_type(desc.topology);
        pd.NumRenderTargets = desc.color_count;
        for (uint32_t i = 0; i < desc.color_count; i++)
            pd.RTVFormats[i] = map_format(desc.color_formats[i]).view;
        if (desc.depth_format != format::unknown)
            pd.DSVFormat = map_format(desc.depth_format).dsv;
        pd.SampleDesc = {(std::max)(1u, desc.samples), 0};

        uint64_t key = fnv(vs->code.data(), vs->code.size());
        if (fs)
            key = fnv(fs->code.data(), fs->code.size(), key);
        key = fnv(&pd.BlendState, sizeof(pd.BlendState), key);
        key = fnv(&pd.RasterizerState, sizeof(pd.RasterizerState), key);
        key = fnv(&pd.DepthStencilState, sizeof(pd.DepthStencilState), key);
        for (uint32_t i = 0; i < desc.attribute_count; i++) {
            const uint32_t v[] = {elements[i].SemanticIndex, static_cast<uint32_t>(elements[i].Format),
                                  elements[i].InputSlot, elements[i].AlignedByteOffset,
                                  static_cast<uint32_t>(elements[i].InputSlotClass)};
            key = fnv(v, sizeof(v), key);
        }
        key = fnv(pd.RTVFormats, sizeof(pd.RTVFormats), key);
        const uint32_t tail[] = {static_cast<uint32_t>(pd.DSVFormat), static_cast<uint32_t>(pd.PrimitiveTopologyType),
                                 pd.SampleDesc.Count, pd.NumRenderTargets};
        key = fnv(tail, sizeof(tail), key);
        key = fnv(&layout->hash, sizeof(layout->hash), key);
        const std::wstring name = hash_name("g", key);

        pipeline_rec rec;
        HRESULT hr = E_FAIL;
        if (pipeline_library_) {
            std::lock_guard lock(library_mutex_);
            hr = pipeline_library_->LoadGraphicsPipeline(name.c_str(), &pd, winrt::guid_of<ID3D12PipelineState>(),
                                                         rec.pso.put_void());
        }
        if (FAILED(hr)) {
            hr = device_->CreateGraphicsPipelineState(&pd, winrt::guid_of<ID3D12PipelineState>(), rec.pso.put_void());
            if (SUCCEEDED(hr) && pipeline_library_) {
                std::lock_guard lock(library_mutex_);
                pipeline_library_->StorePipeline(name.c_str(), rec.pso.get());
            }
        }
        if (FAILED(hr)) {
            {
                char code[32];
                std::snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned>(hr));
                report_error("CreateGraphicsPipelineState failed (" + std::string(code) + ") for '" +
                             std::string(desc.name ? desc.name : "") + "' rt=" + std::to_string(desc.color_count) +
                             " attrs=" + std::to_string(desc.attribute_count));
            }
            return {};
        }
        if (desc.name)
            rec.pso->SetName(widen(desc.name).c_str());
        rec.root_signature = layout->root_signature;
        rec.layout_handle = desc.layout;
        rec.topology = to_d3d(desc.topology);
        for (uint32_t i = 0; i < desc.stream_count; i++)
            rec.strides[i] = desc.streams[i].stride;
        rec.stream_count = desc.stream_count;
        rec.color_count = desc.color_count;
        rec.color_formats = desc.color_formats;
        rec.depth_format = desc.depth_format;
        rec.push_bytes = layout->push_bytes;
        live_pipelines_++;
        return pipelines_.alloc(std::move(rec));
    }

    pipeline_handle device::create_compute_pipeline(const compute_pipeline_desc& desc) {
        pipeline_layout_rec* layout = pipeline_layouts_.get(desc.layout);
        shader_rec* cs = shaders_.get(desc.compute);
        if (!layout || !cs) {
            report_error("CreateComputePipeline: missing layout or shader");
            return {};
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
        pd.pRootSignature = layout->root_signature.get();
        pd.CS = {cs->code.data(), cs->code.size()};
        uint64_t key = fnv(cs->code.data(), cs->code.size());
        key = fnv(&layout->hash, sizeof(layout->hash), key);
        const std::wstring name = hash_name("c", key);

        pipeline_rec rec;
        HRESULT hr = E_FAIL;
        if (pipeline_library_) {
            std::lock_guard lock(library_mutex_);
            hr = pipeline_library_->LoadComputePipeline(name.c_str(), &pd, winrt::guid_of<ID3D12PipelineState>(),
                                                        rec.pso.put_void());
        }
        if (FAILED(hr)) {
            hr = device_->CreateComputePipelineState(&pd, winrt::guid_of<ID3D12PipelineState>(), rec.pso.put_void());
            if (SUCCEEDED(hr) && pipeline_library_) {
                std::lock_guard lock(library_mutex_);
                pipeline_library_->StorePipeline(name.c_str(), rec.pso.get());
            }
        }
        if (FAILED(hr)) {
            report_error("CreateComputePipelineState failed");
            return {};
        }
        if (desc.name)
            rec.pso->SetName(widen(desc.name).c_str());
        rec.compute = true;
        rec.root_signature = layout->root_signature;
        rec.layout_handle = desc.layout;
        rec.push_bytes = layout->push_bytes;
        live_pipelines_++;
        return pipelines_.alloc(std::move(rec));
    }

    query_set_handle device::create_query_set(const query_set_desc& desc) {
        D3D12_QUERY_HEAP_DESC qd{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, desc.count, 0};
        query_rec rec;
        rec.count = desc.count;
        if (FAILED(device_->CreateQueryHeap(&qd, winrt::guid_of<ID3D12QueryHeap>(), rec.heap.put_void()))) {
            report_error("CreateQueryHeap failed");
            return {};
        }
        return queries_.alloc(std::move(rec));
    }

    ID3D12CommandSignature* device::signature(bool dispatch, uint32_t stride) {
        const uint64_t key = (dispatch ? 1ull << 32 : 0) | stride;
        std::lock_guard lock(signature_mutex_);
        auto it = signatures_.find(key);
        if (it != signatures_.end())
            return it->second.get();
        D3D12_INDIRECT_ARGUMENT_DESC arg{};
        arg.Type = dispatch ? D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH : D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
        D3D12_COMMAND_SIGNATURE_DESC sd{};
        sd.ByteStride = dispatch ? sizeof(D3D12_DISPATCH_ARGUMENTS) : stride;
        sd.NumArgumentDescs = 1;
        sd.pArgumentDescs = &arg;
        winrt::com_ptr<ID3D12CommandSignature> sig;
        if (FAILED(device_->CreateCommandSignature(&sd, nullptr, winrt::guid_of<ID3D12CommandSignature>(),
                                                   sig.put_void())))
            return nullptr;
        auto* raw = sig.get();
        signatures_.emplace(key, std::move(sig));
        return raw;
    }

    void device::retire(std::function<void()> release) {
        queue_serials snapshot;
        for (size_t i = 0; i < queue_count; i++)
            snapshot[i] = queues_[i].serial.load();
        retire_.push(snapshot, std::move(release));
    }

    void device::destroy(buffer_handle h) {
        buffer_rec rec;
        if (!buffers_.free(h, &rec))
            return;
        buffer_bytes_ -= static_cast<int64_t>(rec.desc.size);
        live_buffers_--;
        if (use_validator_)
            validator_.on_destroy(h);
        retire([rec = std::move(rec)]() mutable { rec = buffer_rec{}; });
    }

    void device::destroy(texture_handle h) {
        texture_rec rec;
        if (!textures_.free(h, &rec))
            return;
        live_textures_--;
        if (use_validator_)
            validator_.on_destroy(h);
        retire([this, rec = std::move(rec)]() mutable {
            for (auto& [k, v] : rec.views.rtv)
                rtv_pool_.free(v);
            for (auto& [k, v] : rec.views.dsv)
                dsv_pool_.free(v);
            rec = texture_rec{};
        });
    }

    void device::destroy(sampler_handle) {}
    void device::destroy(shader_handle h) {
        shaders_.free(h);
    }
    void device::destroy(bind_layout_handle h) {
        bind_layouts_.free(h);
    }

    void device::destroy(bind_group_handle h) {
        bind_group_rec* g = bind_groups_.get(h);
        if (!g || g->per_frame)
            return;
        bind_group_rec rec;
        bind_groups_.free(h, &rec);
        live_groups_--;
        retire([this, rec] {
            resource_heap_.free(rec.table_offset, rec.table_count);
            release_sampler_table(rec.sampler_table);
        });
    }

    void device::destroy(pipeline_layout_handle h) {
        pipeline_layout_rec rec;
        if (!pipeline_layouts_.free(h, &rec))
            return;
        retire([rec = std::move(rec)]() mutable { rec = pipeline_layout_rec{}; });
    }

    void device::destroy(pipeline_handle h) {
        pipeline_rec rec;
        if (!pipelines_.free(h, &rec))
            return;
        live_pipelines_--;
        retire([rec = std::move(rec)]() mutable { rec = pipeline_rec{}; });
    }

    void device::destroy(query_set_handle h) {
        query_rec rec;
        if (!queries_.free(h, &rec))
            return;
        retire([rec = std::move(rec)]() mutable { rec = query_rec{}; });
    }

    void device::destroy(fence_handle h) {
        fence_rec rec;
        if (!fences_.free(h, &rec))
            return;
        retire([rec = std::move(rec)]() mutable { rec = fence_rec{}; });
    }

    const buffer_desc* device::get_desc(buffer_handle h) const {
        buffer_rec* r = buffers_.get(h);
        return r ? &r->desc : nullptr;
    }

    const texture_desc* device::get_desc(texture_handle h) const {
        texture_rec* r = textures_.get(h);
        return r ? &r->desc : nullptr;
    }

    uint8_t* device::map(buffer_handle h) {
        buffer_rec* r = buffers_.get(h);
        return r ? r->mapped : nullptr;
    }

    uint64_t device::gpu_address(buffer_handle h) const {
        buffer_rec* r = buffers_.get(h);
        return r ? r->address : 0;
    }

    fence_handle device::create_fence(uint64_t initial) {
        fence_rec rec;
        if (FAILED(device_->CreateFence(initial, D3D12_FENCE_FLAG_SHARED, winrt::guid_of<ID3D12Fence>(),
                                        rec.fence.put_void())) &&
            FAILED(device_->CreateFence(initial, D3D12_FENCE_FLAG_NONE, winrt::guid_of<ID3D12Fence>(),
                                        rec.fence.put_void())))
            return {};
        return fences_.alloc(std::move(rec));
    }

    uint64_t device::fence_value(fence_handle h) {
        fence_rec* f = fences_.get(h);
        return f ? f->fence->GetCompletedValue() : 0;
    }

    bool device::wait_fence(fence_handle h, uint64_t value, uint64_t timeout_ns) {
        fence_rec* f = fences_.get(h);
        if (!f)
            return false;
        if (f->fence->GetCompletedValue() >= value)
            return true;
        std::lock_guard lock(wait_mutex_);
        if (FAILED(f->fence->SetEventOnCompletion(value, wait_event_.get())))
            return false;
        const DWORD ms = timeout_ns == ~0ull
                             ? INFINITE
                             : static_cast<DWORD>(std::min<uint64_t>(timeout_ns / 1'000'000ull, 0xFFFFFFFEull));
        return WaitForSingleObject(wait_event_.get(), ms) == WAIT_OBJECT_0;
    }

    void device::signal_fence(fence_handle h, uint64_t value) {
        if (fence_rec* f = fences_.get(h))
            f->fence->Signal(value);
    }

    queue_serials device::completed() {
        queue_serials out{};
        for (size_t i = 0; i < queue_count; i++) {
            if (!queues_[i].fence) {
                out[i] = ~0ull;
                continue;
            }
            out[i] = queues_[i].fence->GetCompletedValue();
        }
        return out;
    }

    thread_pools& device::pools_for_this_thread() {
        std::lock_guard lock(pools_mutex_);
        auto& entry = thread_pools_[std::this_thread::get_id()];
        if (!entry) {
            entry = std::make_unique<thread_pools>();
            entry->slots.resize(options_.frames_in_flight);
        }
        return *entry;
    }

    command_list* device::acquire_command_list(queue_type requested) {
        const queue_type queue =
            (requested == queue_type::compute && !queues_[1].queue) ? queue_type::graphics : requested;
        thread_pools& tp = pools_for_this_thread();
        auto& per = tp.slots[slot_][static_cast<size_t>(queue)];
        if (per.used == per.lists.size()) {
            const D3D12_COMMAND_LIST_TYPE types[queue_count] = {
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY};
            winrt::com_ptr<ID3D12CommandAllocator> alloc;
            winrt::com_ptr<ID3D12GraphicsCommandList> list;
            if (FAILED(device_->CreateCommandAllocator(types[static_cast<size_t>(queue)],
                                                       winrt::guid_of<ID3D12CommandAllocator>(), alloc.put_void())) ||
                FAILED(device_->CreateCommandList(0, types[static_cast<size_t>(queue)], alloc.get(), nullptr,
                                                  winrt::guid_of<ID3D12GraphicsCommandList>(), list.put_void()))) {
                report_error("command list creation failed");
                return nullptr;
            }
            list->Close();
            per.lists.push_back(std::make_unique<cmd_list>(*this, queue, std::move(alloc), std::move(list)));
        }
        cmd_list* list = per.lists[per.used++].get();
        list->begin();
        return list;
    }

    void device::submit(conduit::queue_type requested, const submit_desc& desc) {
        const conduit::queue_type queue_type = (requested == conduit::queue_type::compute && !queues_[1].queue)
                                                   ? conduit::queue_type::graphics
                                                   : requested;
        queue_info& q = queues_[static_cast<size_t>(queue_type)];
        std::vector<ID3D12CommandList*> native;
        native.reserve(desc.lists.size());
        for (command_list* l : desc.lists) {
            auto* list = static_cast<cmd_list*>(l);
            if (list->open_labels() != 0)
                report_error("command list submitted with an open debug label");
            list->end();
            counters_.add(list->counters());
            native.push_back(list->native());
        }
        std::lock_guard lock(q.mutex);
        for (const fence_op& w : desc.waits)
            if (fence_rec* f = fences_.get(w.fence))
                q.queue->Wait(f->fence.get(), w.value);
        if (!native.empty())
            q.queue->ExecuteCommandLists(static_cast<UINT>(native.size()), native.data());
        for (const fence_op& s : desc.signals)
            if (fence_rec* f = fences_.get(s.fence))
                q.queue->Signal(f->fence.get(), s.value);
        const uint64_t serial = q.serial.load() + 1;
        q.queue->Signal(q.fence.get(), serial);
        q.serial.store(serial);
        counters_.count_submit();
    }

    void device::signal_serial(queue_type) {}

    std::unique_ptr<conduit::swapchain> device::create_swapchain(const swapchain_desc& desc) {
        return make_swapchain(*this, desc);
    }

    void device::reset_slot(uint32_t s) {
        {
            std::lock_guard lock(pools_mutex_);
            for (auto& [id, tp] : thread_pools_)
                for (auto& per : tp->slots[s]) {
                    for (size_t i = 0; i < per.used; i++)
                        per.lists[i]->allocator()->Reset();
                    per.used = 0;
                }
        }
        std::vector<bind_group_handle> groups;
        {
            std::lock_guard lock(frame_group_mutex_);
            groups.swap(frame_groups_[s].groups);
        }
        for (bind_group_handle g : groups) {
            bind_group_rec rec;
            if (bind_groups_.free(g, &rec)) {
                live_groups_--;
                resource_heap_.free(rec.table_offset, rec.table_count);
                release_sampler_table(rec.sampler_table);
            }
        }
    }

    void device::begin_frame() {
        frame_++;
        slot_ = static_cast<uint32_t>(frame_ % options_.frames_in_flight);
        const queue_serials& need = slot_serials_[slot_];
        for (size_t i = 0; i < queue_count; i++) {
            if (!need[i] || !queues_[i].fence)
                continue;
            if (queues_[i].fence->GetCompletedValue() >= need[i])
                continue;
            std::lock_guard lock(wait_mutex_);
            queues_[i].fence->SetEventOnCompletion(need[i], wait_event_.get());
            // never reuse a slot whose frame is still running: wait it out unless the device is gone
            DWORD w = WaitForSingleObject(wait_event_.get(), 10'000);
            while (w == WAIT_TIMEOUT && device_ && device_->GetDeviceRemovedReason() == S_OK &&
                   queues_[i].fence->GetCompletedValue() < need[i])
                w = WaitForSingleObject(wait_event_.get(), 10'000);
            if (w != WAIT_OBJECT_0 && queues_[i].fence->GetCompletedValue() < need[i])
                report_error("frame wait did not complete");
        }
        reset_slot(slot_);
        retire_.collect(completed());
    }

    void device::end_frame() {
        for (size_t i = 0; i < queue_count; i++)
            slot_serials_[slot_][i] = queues_[i].serial.load();
        retire_.collect(completed());
    }

    void device::wait_idle() {
        for (size_t i = 0; i < queue_count; i++) {
            if (!queues_[i].fence)
                continue;
            const uint64_t target = queues_[i].serial.load();
            if (queues_[i].fence->GetCompletedValue() >= target)
                continue;
            std::lock_guard lock(wait_mutex_);
            queues_[i].fence->SetEventOnCompletion(target, wait_event_.get());
            WaitForSingleObject(wait_event_.get(), INFINITE);
        }
        retire_.collect(completed());
    }

    device_stats device::stats() const {
        device_stats s;
        counters_.fill(s);
        s.buffer_bytes = static_cast<uint64_t>(std::max<int64_t>(0, buffer_bytes_));
        s.texture_bytes = static_cast<uint64_t>(std::max<int64_t>(0, texture_bytes_));
        s.live_buffers = static_cast<uint32_t>(live_buffers_.load());
        s.live_textures = static_cast<uint32_t>(live_textures_.load());
        s.live_pipelines = static_cast<uint32_t>(live_pipelines_.load());
        s.live_bind_groups = static_cast<uint32_t>(live_groups_.load());
        return s;
    }

    std::vector<uint8_t> device::serialize_pipeline_cache() {
        if (!pipeline_library_)
            return {};
        std::lock_guard lock(library_mutex_);
        const SIZE_T size = pipeline_library_->GetSerializedSize();
        if (!size)
            return {};
        std::vector<uint8_t> data(size);
        if (FAILED(pipeline_library_->Serialize(data.data(), size)))
            return {};
        return data;
    }
}
