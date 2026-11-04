#include "metal/metal_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <tuple>

namespace conduit::metal {
    namespace {
        bool is_integer_format(format f) {
            return f == format::r8_uint || f == format::r16_uint || f == format::r32_uint || f == format::rg32_uint ||
                   f == format::rgba32_uint;
        }

        bool storage_capable(format f) {
            switch (f) {
            case format::r8_unorm:
            case format::rg8_unorm:
            case format::rgba8_unorm:
            case format::bgra8_unorm:
            case format::r8_uint:
            case format::r16_uint:
            case format::r32_uint:
            case format::rg32_uint:
            case format::rgba32_uint:
            case format::r16_float:
            case format::rg16_float:
            case format::rgba16_float:
            case format::r32_float:
            case format::rg32_float:
            case format::rgba32_float:
            case format::rgb10a2_unorm:
            case format::r11g11b10_float:
            case format::rg16_unorm:
            case format::rgba16_unorm:
                return true;
            default:
                return false;
            }
        }

        void parse_threads(const std::string& src, uint32_t out[3]) {
            const size_t at = src.find("// conduit:threads ");
            if (at == std::string::npos)
                return;
            std::sscanf(src.c_str() + at, "// conduit:threads %u %u %u", &out[0], &out[1], &out[2]);
        }
    }

    device::device(id<MTLDevice> dev, const device_options& opts, const adapter_info& info)
        : device_(dev), options_(opts) {
        caps_.adapter = info;
        options_.frames_in_flight = std::clamp(options_.frames_in_flight, 1u, 4u);
        if (options_.async_pipelines) {
            compile_queue_ = dispatch_queue_create(
                "conduit.metal.pipelines",
                dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0));
            compile_group_ = dispatch_group_create();
        }
    }

    device::~device() {
        if (!device_)
            return;
        wait_for_compiles();
        wait_idle();
        lists_.clear();
    }

    void device::report_error(const std::string& message) {
        std::lock_guard lock(error_mutex_);
        std::fprintf(stderr, "[metal] %s\n", message.c_str());
    }

    native_handles device::native() const {
        native_handles n;
        n.api = api::metal;
        n.device = (__bridge void*)device_;
        n.graphics_queue = (__bridge void*)queue_;
        return n;
    }

    bool device::init() {
        queue_ = [device_ newCommandQueue];
        if (!queue_)
            return false;
#if TARGET_OS_OSX
        g_depth24_stencil8 = device_.depth24Stencil8PixelFormatSupported;
#endif
        fill_caps();
        frame_fence_ = create_fence(0);
        return true;
    }

    void device::fill_caps() {
        device_limits& l = caps_.limits;
        const bool apple3 = [device_ supportsFamily:MTLGPUFamilyApple3];
        const bool mac = [device_ supportsFamily:MTLGPUFamilyMac2];
        l.max_texture2d = (apple3 || mac) ? 16384 : 8192;
        l.max_texture_layers = 2048;
        l.max_bind_groups = max_bind_groups;
        l.max_push_constant_bytes = 0;
        l.max_sampled_textures_per_stage = 128;
        l.max_color_targets = 8;
        l.uniform_buffer_alignment = 256;
        l.storage_buffer_alignment = 16;
        l.copy_row_pitch_alignment = 1;
        l.copy_offset_alignment = 16;
        l.max_buffer_size = device_.maxBufferLength;
        l.max_anisotropy = 16.0f;
        l.max_msaa_samples = 1;
        for (const NSUInteger samples : {8u, 4u, 2u})
            if ([device_ supportsTextureSampleCount:samples]) {
                l.max_msaa_samples = static_cast<uint32_t>(samples);
                break;
            }

        device_features& f = caps_.features;
        f.sampler_anisotropy = true;
        f.depth_clip_control = true;
        f.draw_indirect_first_instance = true;
        f.shader_float16 = true;
        f.unified_memory = device_.hasUnifiedMemory;
        f.subgroup_ops = apple3 || mac;
        f.ray_tracing = device_.supportsRaytracing;
        f.hdr_output = true;
        f.tearing = true;
        f.multi_draw_indirect = false;
        f.draw_indirect_count = false;
        f.async_compute = false;
        f.dedicated_copy_queue = false;
        f.timestamp_queries = false;
#if TARGET_OS_OSX
        f.texture_compression_bc = device_.supportsBCTextureCompression;
#else
        f.texture_compression_bc = false;
#endif
        const bool bc = f.texture_compression_bc;

        for (uint32_t i = 1; i < static_cast<uint32_t>(format::count); i++) {
            const format fmt = static_cast<format>(i);
            if (to_pixel_format(fmt) == MTLPixelFormatInvalid)
                continue;
            const format_info& info = get_format_info(fmt);
            format_use use = format_use::sampled;
            if (info.compressed) {
                if (bc)
                    use = use | format_use::sampled_linear;
                else
                    use = format_use::none;
            } else if (info.depth) {
                use = use | format_use::sampled_linear | format_use::depth_target | format_use::msaa;
            } else {
                const bool integer = is_integer_format(fmt);
                if (!integer)
                    use = use | format_use::sampled_linear | format_use::blendable | format_use::msaa;
                use = use | format_use::color_target;
                if (storage_capable(fmt))
                    use = use | format_use::storage;
            }
            caps_.formats[i] = use;
        }
    }

    buffer_handle device::create_buffer(const buffer_desc& desc) {
        buffer b;
        b.desc = desc;
        b.name = desc.name ? desc.name : "";
        uint64_t size = align_up(std::max<uint64_t>(desc.size, 16), 16);
        if (has(desc.usage, buffer_usage::uniform))
            size = align_up(size, 256);
        MTLResourceOptions opts = MTLResourceStorageModePrivate;
        switch (desc.memory) {
        case memory_class::gpu_only:
            break;
        case memory_class::upload:
            opts = MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
            break;
        default:
            opts = MTLResourceStorageModeShared;
            break;
        }
        b.native = [device_ newBufferWithLength:size options:opts];
        if (!b.native)
            return {};
        if (!b.name.empty())
            b.native.label = [NSString stringWithUTF8String:b.name.c_str()];
        b.size = size;
        if (desc.memory != memory_class::gpu_only) {
            b.contents = static_cast<uint8_t*>([b.native contents]);
            std::memset(b.contents, 0, size);
        }
        stats_.buffer_bytes += size;
        stats_.live_buffers++;
        const buffer_handle h = buffers_.alloc(std::move(b));
        buffers_.get(h)->desc.name = buffers_.get(h)->name.c_str();
        return h;
    }

    texture_handle device::create_texture(const texture_desc& desc) {
        const MTLPixelFormat pf = to_pixel_format(desc.format);
        if (pf == MTLPixelFormatInvalid) {
            report_error("CreateTexture: unsupported format " + std::to_string(static_cast<int>(desc.format)));
            return {};
        }
        texture t;
        t.desc = desc;
        t.desc.mips = (std::max)(1u, desc.mips);
        t.desc.samples = (std::max)(1u, desc.samples);
        t.name = desc.name ? desc.name : "";

        MTLTextureDescriptor* td = [MTLTextureDescriptor new];
        const bool ms = t.desc.samples > 1;
        td.textureType = to_texture_type(desc.dim, ms, false);
        td.pixelFormat = pf;
        td.width = desc.width;
        td.height = desc.height;
        td.depth = desc.dim == texture_dim::tex3d ? (std::max)(1u, desc.depth_or_layers) : 1;
        td.arrayLength = desc.dim == texture_dim::tex2d_array ? (std::max)(1u, desc.depth_or_layers) : 1;
        td.mipmapLevelCount = ms ? 1 : t.desc.mips;
        td.sampleCount = t.desc.samples;
        td.storageMode = MTLStorageModePrivate;
        MTLTextureUsage usage = MTLTextureUsageUnknown;
        if (has(desc.usage, texture_usage::sampled))
            usage |= MTLTextureUsageShaderRead;
        if (has(desc.usage, texture_usage::storage))
            usage |= MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        if (has(desc.usage, texture_usage::color_target) || has(desc.usage, texture_usage::depth_target))
            usage |= MTLTextureUsageRenderTarget;
        if (!is_depth_format(desc.format) || has_stencil(desc.format))
            usage |= MTLTextureUsagePixelFormatView;
        td.usage = usage;
        t.native = [device_ newTextureWithDescriptor:td];
        if (!t.native)
            return {};
        if (!t.name.empty())
            t.native.label = [NSString stringWithUTF8String:t.name.c_str()];

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
        MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
        sd.minFilter = to_min_mag(desc.min);
        sd.magFilter = to_min_mag(desc.mag);
        sd.mipFilter = to_mip(desc.mip);
        sd.sAddressMode = to_address(desc.u);
        sd.tAddressMode = to_address(desc.v);
        sd.rAddressMode = to_address(desc.w);
        sd.maxAnisotropy = static_cast<NSUInteger>(std::clamp(desc.max_anisotropy, 1.0f, 16.0f));
        sd.lodMinClamp = desc.min_lod;
        sd.lodMaxClamp = (std::min)(desc.max_lod, 1000.0f);
        if (desc.compare)
            sd.compareFunction = to_compare(desc.compare_op);
#if TARGET_OS_OSX
        if (desc.border[3] < 0.5f)
            sd.borderColor = MTLSamplerBorderColorTransparentBlack;
        else if (desc.border[0] + desc.border[1] + desc.border[2] < 1.5f)
            sd.borderColor = MTLSamplerBorderColorOpaqueBlack;
        else
            sd.borderColor = MTLSamplerBorderColorOpaqueWhite;
#endif
        s.state = [device_ newSamplerStateWithDescriptor:sd];
        if (!s.state)
            return {};
        return samplers_.alloc(std::move(s));
    }

    shader_handle device::create_shader(const shader_desc& desc) {
        if (desc.format != shader_format::msl) {
            report_error("CreateShader: metal takes msl source");
            return {};
        }
        shader s;
        s.stage = desc.stage;
        s.source.assign(reinterpret_cast<const char*>(desc.code.data()), desc.code.size());
        s.entry = desc.entry ? desc.entry : "main";
        s.iface = reflect_shader(shader_format::msl, s.source);
        if (desc.stage == shader_stage::compute)
            parse_threads(s.source, s.threads);
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
            report_error("CreatePipelineLayout: metal has no push constants here");
        pipeline_layout l;
        l.desc = desc;
        l.desc.name = nullptr;
        return pipeline_layouts_.alloc(std::move(l));
    }

    query_set_handle device::create_query_set(const query_set_desc& desc) {
        query q;
        q.count = desc.count;
        return queries_.alloc(std::move(q));
    }

    void device::destroy(buffer_handle h) {
        buffer b;
        if (!buffers_.free(h, &b))
            return;
        stats_.buffer_bytes -= b.size;
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
        return b ? b->contents : nullptr;
    }

    id<MTLTexture> device::view_of(texture& t, const subresource_range& range, texture_dim dim, conduit::format format,
                                   bool stencil) {
        const subresource_range r = resolve_range(range, t.desc);
        const uint32_t layers = layer_count_of(t.desc);
        MTLPixelFormat pf =
            format == conduit::format::unknown ? to_pixel_format(t.desc.format) : to_pixel_format(format);
        if (stencil) {
            pf = to_pixel_format(t.desc.format) == MTLPixelFormatDepth32Float_Stencil8 ? MTLPixelFormatX32_Stencil8
                                                                                       : MTLPixelFormatInvalid;
#if TARGET_OS_OSX
            if (to_pixel_format(t.desc.format) == MTLPixelFormatDepth24Unorm_Stencil8)
                pf = MTLPixelFormatX24_Stencil8;
#endif
        }
        const bool whole_mips = r.base_mip == 0 && r.mip_count == (std::max)(1u, t.desc.mips);
        const bool whole_layers = r.base_layer == 0 && r.layer_count == layers;
        const bool same_type = (dim == texture_dim::tex2d && t.desc.dim == texture_dim::tex2d) || dim == t.desc.dim;
        if (format == conduit::format::unknown && !stencil && whole_mips && whole_layers && same_type)
            return t.native;
        if (t.desc.samples > 1)
            return t.native;

        view_key key;
        key.dim = static_cast<uint8_t>(dim);
        key.stencil = stencil ? 1 : 0;
        key.format = static_cast<uint16_t>(format);
        key.base_mip = r.base_mip;
        key.mip_count = r.mip_count;
        key.base_layer = r.base_layer;
        key.layer_count = r.layer_count;
        std::lock_guard lock(view_mutex_);
        if (const auto it = t.views.find(key); it != t.views.end())
            return it->second;
        MTLTextureType type = to_texture_type(dim, false, false);
        if (dim == texture_dim::tex2d && layers > 1)
            type = MTLTextureType2D;
        const NSRange levels = NSMakeRange(r.base_mip, r.mip_count);
        const NSRange slices = t.desc.dim == texture_dim::tex3d ? NSMakeRange(0, 1)
                               : dim == texture_dim::tex2d      ? NSMakeRange(r.base_layer, 1)
                                                                : NSMakeRange(r.base_layer, r.layer_count);
        id<MTLTexture> view = [t.native newTextureViewWithPixelFormat:pf textureType:type levels:levels slices:slices];
        if (view)
            t.views[key] = view;
        return view;
    }

    texture_handle device::create_surface_texture(conduit::format format, uint32_t width, uint32_t height,
                                                  bool sampled) {
        texture t;
        t.desc.dim = texture_dim::tex2d;
        t.desc.format = format;
        t.desc.width = width;
        t.desc.height = height;
        t.desc.usage = texture_usage::color_target | texture_usage::copy_src |
                       (sampled ? texture_usage::sampled : texture_usage::none);
        t.external = true;
        t.name = "swapchain image";
        return textures_.alloc(std::move(t));
    }

    void device::set_surface_texture(texture_handle handle, id<MTLTexture> texture) {
        metal::texture* t = textures_.get(handle);
        if (!t)
            return;
        std::lock_guard lock(view_mutex_);
        t->views.clear();
        t->native = texture;
    }

    void device::resize_surface_texture(texture_handle handle, uint32_t width, uint32_t height) {
        if (texture* t = textures_.get(handle)) {
            t->desc.width = width;
            t->desc.height = height;
        }
    }

    // slang numbers samplers 2*slot+1, past metal's 16 sampler slots: they are compacted in declaration order
    std::string device::prepare_stage(const shader& sh, const pipeline_layout_desc& layout,
                                      std::vector<stage_slot>& slots, std::string& error) {
        std::vector<const shader_resource*> resources;
        for (const shader_resource& r : sh.iface.resources)
            resources.push_back(&r);
        std::sort(resources.begin(), resources.end(), [](const shader_resource* a, const shader_resource* b) {
            return std::tie(a->set, a->binding) < std::tie(b->set, b->binding);
        });

        uint32_t next_sampler = 0;
        struct edit {
            size_t begin, end;
            std::string text;
        };
        std::vector<edit> edits;
        for (const shader_resource* r : resources) {
            stage_slot s;
            s.set = r->set;
            s.binding = r->binding;
            s.kind = r->kind;
            s.dim = r->dim;
            s.index = r->native_index;
            if (r->kind == resource_kind::sampler) {
                s.index = next_sampler < 16 ? next_sampler++ : 15;
                edits.push_back({r->clause_begin, r->clause_end, "[[sampler(" + std::to_string(s.index) + ")]]"});
            }
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
        }

        std::sort(edits.begin(), edits.end(), [](const edit& a, const edit& b) { return a.begin > b.begin; });
        std::string patched = sh.source;
        for (const edit& e : edits)
            patched.replace(e.begin, e.end - e.begin, e.text);

        return patched;
    }

    id<MTLFunction> device::load_function(const std::string& patched, const std::string& entry, std::string& error) {
        id<MTLLibrary> lib = nil;
        {
            std::lock_guard lock(library_mutex_);
            if (const auto it = libraries_.find(patched); it != libraries_.end())
                lib = it->second;
        }
        if (!lib) {
            NSError* err = nil;
            MTLCompileOptions* opts = [MTLCompileOptions new];
            lib = [device_ newLibraryWithSource:[NSString stringWithUTF8String:patched.c_str()]
                                        options:opts
                                          error:&err];
            if (!lib) {
                error = err ? [[err localizedDescription] UTF8String] : "newLibraryWithSource failed";
                return nil;
            }
            std::lock_guard lock(library_mutex_);
            libraries_[patched] = lib;
        }
        id<MTLFunction> function = [lib newFunctionWithName:[NSString stringWithUTF8String:entry.c_str()]];
        if (!function)
            error = "entry point " + entry + " not found";
        return function;
    }

    id<MTLFunction> device::compile_stage(const shader& sh, const pipeline_layout_desc& layout,
                                          std::vector<stage_slot>& slots, std::string& error) {
        const std::string source = prepare_stage(sh, layout, slots, error);
        return source.empty() ? nil : load_function(source, sh.entry, error);
    }

    id<MTLRenderPipelineState> device::ready_pipeline(const pipeline& p) {
        if (!p.build)
            return p.render;
        pipeline_build& b = *p.build;
        int state = b.state.load(std::memory_order_acquire);
        if (state == 0 && !b.waited.exchange(true)) {
            b.done.wait(1, 4'000'000ull);
            state = b.state.load(std::memory_order_acquire);
        }
        if (state == 1)
            return b.render;
        if (state < 0 && !b.reported.exchange(true))
            report_error(b.error);
        if (state == 0)
            stats_.pipelines_pending++;
        return nil;
    }

    void device::wait_for_compiles() {
        if (compile_group_)
            dispatch_group_wait(compile_group_, DISPATCH_TIME_FOREVER);
    }

    pipeline_handle device::create_graphics_pipeline(const graphics_pipeline_desc& desc) {
        const shader* vs = shaders_.get(desc.vertex);
        const shader* fs = shaders_.get(desc.fragment);
        const pipeline_layout* layout = pipeline_layouts_.get(desc.layout);
        if (!vs || !layout) {
            report_error("CreateGraphicsPipeline: missing shader or layout");
            return {};
        }
        pipeline p;
        std::string error;
        const std::string vs_source = prepare_stage(*vs, layout->desc, p.stages[stage_vertex], error);
        const std::string fs_source = !vs_source.empty() && fs
                                          ? prepare_stage(*fs, layout->desc, p.stages[stage_fragment], error)
                                          : std::string();
        if (vs_source.empty() || (fs && fs_source.empty())) {
            report_error(std::string("shader compile failed (") + (desc.name ? desc.name : "?") + "): " + error);
            return {};
        }

        MTLRenderPipelineDescriptor* pd = [MTLRenderPipelineDescriptor new];
        if (desc.name)
            pd.label = [NSString stringWithUTF8String:desc.name];

        MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
        for (uint32_t i = 0; i < desc.attribute_count; i++) {
            const vertex_attribute_desc& a = desc.attributes[i];
            vd.attributes[a.location].format = to_vertex_format(a.format);
            vd.attributes[a.location].offset = a.offset;
            vd.attributes[a.location].bufferIndex = vertex_buffer_base - a.stream;
        }
        for (uint32_t i = 0; i < desc.stream_count; i++) {
            const NSUInteger index = vertex_buffer_base - i;
            vd.layouts[index].stride = (std::max)(desc.streams[i].stride, 1u);
            vd.layouts[index].stepFunction = desc.streams[i].step == vertex_step::instance
                                                 ? MTLVertexStepFunctionPerInstance
                                                 : MTLVertexStepFunctionPerVertex;
            vd.layouts[index].stepRate = 1;
        }
        if (desc.attribute_count)
            pd.vertexDescriptor = vd;

        for (uint32_t i = 0; i < desc.color_count; i++) {
            const blend_target& t = desc.blend.targets[i];
            MTLRenderPipelineColorAttachmentDescriptor* ca = pd.colorAttachments[i];
            ca.pixelFormat = to_pixel_format(desc.color_formats[i]);
            ca.blendingEnabled = t.enable;
            if (t.enable) {
                ca.sourceRGBBlendFactor = to_blend_factor(t.src_color);
                ca.destinationRGBBlendFactor = to_blend_factor(t.dst_color);
                ca.rgbBlendOperation = to_blend_op(t.color_op);
                ca.sourceAlphaBlendFactor = to_blend_factor(t.src_alpha);
                ca.destinationAlphaBlendFactor = to_blend_factor(t.dst_alpha);
                ca.alphaBlendOperation = to_blend_op(t.alpha_op);
            }
            MTLColorWriteMask mask = MTLColorWriteMaskNone;
            if (has(t.write_mask, color_mask::r))
                mask |= MTLColorWriteMaskRed;
            if (has(t.write_mask, color_mask::g))
                mask |= MTLColorWriteMaskGreen;
            if (has(t.write_mask, color_mask::b))
                mask |= MTLColorWriteMaskBlue;
            if (has(t.write_mask, color_mask::a))
                mask |= MTLColorWriteMaskAlpha;
            ca.writeMask = mask;
        }
        const bool has_depth = desc.depth_format != format::unknown;
        if (has_depth) {
            pd.depthAttachmentPixelFormat = to_pixel_format(desc.depth_format);
            if (has_stencil(desc.depth_format))
                pd.stencilAttachmentPixelFormat = to_pixel_format(desc.depth_format);
        }
        pd.rasterSampleCount = (std::max)(1u, desc.samples);
        pd.alphaToCoverageEnabled = desc.blend.alpha_to_coverage;
        switch (desc.topology) {
        case topology::line_list:
            pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
            break;
        case topology::point_list:
            pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
            break;
        default:
            pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
            break;
        }

        // compiled off the render thread when allowed: a pipeline first needed mid-game skips a draw instead of hitching
        const std::string name = desc.name ? desc.name : "?";
        const std::string vs_entry = vs->entry, fs_entry = fs ? fs->entry : std::string();
        auto build = std::make_shared<pipeline_build>();
        const auto compile = [this, build, pd, vs_source, fs_source, vs_entry, fs_entry, name] {
            std::string why;
            id<MTLFunction> vs_fn = load_function(vs_source, vs_entry, why);
            id<MTLFunction> fs_fn = vs_fn && !fs_source.empty() ? load_function(fs_source, fs_entry, why) : nil;
            if (vs_fn && (fs_source.empty() || fs_fn)) {
                pd.vertexFunction = vs_fn;
                pd.fragmentFunction = fs_fn;
                NSError* err = nil;
                build->render = [device_ newRenderPipelineStateWithDescriptor:pd error:&err];
                if (!build->render)
                    why = err ? [[err localizedDescription] UTF8String] : "unknown";
            }
            if (build->render) {
                build->state.store(1, std::memory_order_release);
            } else {
                build->error = "pipeline creation failed (" + name + "): " + why;
                build->state.store(-1, std::memory_order_release);
            }
            build->done.raise(1);
        };
        if (options_.async_pipelines && compile_queue_) {
            dispatch_group_async(compile_group_, compile_queue_, ^{
              @autoreleasepool {
                  compile();
              }
            });
            p.build = build;
        } else {
            compile();
            if (!build->render) {
                report_error(build->error);
                return {};
            }
            p.render = build->render;
        }

        const depth_stencil_state& ds = desc.depth_stencil;
        MTLDepthStencilDescriptor* dd = [MTLDepthStencilDescriptor new];
        dd.depthCompareFunction = has_depth && ds.depth_test ? to_compare(ds.depth_compare) : MTLCompareFunctionAlways;
        dd.depthWriteEnabled = has_depth && ds.depth_write;
        if (has_depth && ds.stencil_test) {
            const auto face = [&](const stencil_face& f) {
                MTLStencilDescriptor* sd = [MTLStencilDescriptor new];
                sd.stencilFailureOperation = to_stencil_op(f.fail);
                sd.depthFailureOperation = to_stencil_op(f.depth_fail);
                sd.depthStencilPassOperation = to_stencil_op(f.pass);
                sd.stencilCompareFunction = to_compare(f.compare);
                sd.readMask = ds.stencil_read_mask;
                sd.writeMask = ds.stencil_write_mask;
                return sd;
            };
            dd.frontFaceStencil = face(ds.front);
            dd.backFaceStencil = face(ds.back);
        }
        p.depth_stencil = [device_ newDepthStencilStateWithDescriptor:dd];

        p.cull = desc.raster.cull == cull_mode::none   ? MTLCullModeNone
                 : desc.raster.cull == cull_mode::back ? MTLCullModeBack
                                                       : MTLCullModeFront;
        p.winding =
            desc.raster.front_face == front_face::counter_clockwise ? MTLWindingCounterClockwise : MTLWindingClockwise;
        p.fill = desc.raster.wireframe ? MTLTriangleFillModeLines : MTLTriangleFillModeFill;
        p.clip = desc.raster.depth_clip ? MTLDepthClipModeClip : MTLDepthClipModeClamp;
        p.bias_constant = desc.raster.depth_bias_constant;
        p.bias_slope = desc.raster.depth_bias_slope;
        p.bias_clamp = desc.raster.depth_bias_clamp;
        p.primitive = to_primitive(desc.topology);
        p.stream_count = desc.stream_count;
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
        id<MTLFunction> fn = compile_stage(*cs, layout->desc, p.stages[stage_compute], error);
        if (!fn) {
            report_error(std::string("shader compile failed (") + (desc.name ? desc.name : "?") + "): " + error);
            return {};
        }
        NSError* err = nil;
        p.comp = [device_ newComputePipelineStateWithFunction:fn error:&err];
        if (!p.comp) {
            report_error(std::string("compute pipeline creation failed: ") +
                         (err ? [[err localizedDescription] UTF8String] : "unknown"));
            return {};
        }
        p.threads = MTLSizeMake(cs->threads[0], cs->threads[1], cs->threads[2]);
        stats_.live_pipelines++;
        return pipelines_.alloc(std::move(p));
    }

    fence_handle device::create_fence(uint64_t initial_value) {
        fence f;
        f.value->value.store(initial_value);
        return fences_.alloc(std::move(f));
    }

    uint64_t device::fence_value(fence_handle fence) {
        metal::fence* f = fences_.get(fence);
        return f ? f->value->value.load() : 0;
    }

    void device::signal_fence(fence_handle fence, uint64_t value) {
        if (metal::fence* f = fences_.get(fence))
            f->value->raise(value);
    }

    bool device::wait_fence(fence_handle fence, uint64_t value, uint64_t timeout_ns) {
        metal::fence* f = fences_.get(fence);
        return !f || f->value->wait(value, timeout_ns);
    }

    void device::signal_when_done(id<MTLCommandBuffer> after, fence_handle fence, uint64_t value) {
        metal::fence* f = fences_.get(fence);
        if (!f)
            return;
        std::shared_ptr<metal::timeline> timeline = f->value;
        [after addCompletedHandler:^(id<MTLCommandBuffer>) { timeline->raise(value); }];
    }

    void device::begin_frame() {
        frame_++;
        const uint64_t behind = (std::max)(1u, options_.frames_in_flight);
        if (frame_ > behind)
            wait_fence(frame_fence_, frame_ - behind, 2'000'000'000ull);
    }

    void device::end_frame() {
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        signal_when_done(cb, frame_fence_, frame_);
        [cb commit];
    }

    void device::wait_idle() {
        if (!queue_)
            return;
        id<MTLCommandBuffer> cb = [queue_ commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
    }

    memory_budget device::query_memory_budget() {
        memory_budget m;
        m.device_budget = device_.recommendedMaxWorkingSetSize;
        m.device_usage = device_.currentAllocatedSize;
        m.valid = true;
        return m;
    }

    command_list* device::acquire_command_list(queue_type type) {
        std::lock_guard lock(list_mutex_);
        cmd_list* list;
        if (!free_lists_.empty()) {
            list = free_lists_.back();
            free_lists_.pop_back();
        } else {
            lists_.push_back(std::make_unique<cmd_list>(*this, type));
            list = lists_.back().get();
        }
        list->begin();
        return list;
    }

    void device::submit(queue_type, const submit_desc& desc) {
        id<MTLCommandBuffer> last = nil;
        for (conduit::command_list* base : desc.lists) {
            id<MTLCommandBuffer> cb = static_cast<cmd_list*>(base)->finish();
            if (!cb)
                continue;
            if (last)
                [last commit];
            last = cb;
        }
        if (!last && !desc.signals.empty())
            last = [queue_ commandBuffer];
        if (last) {
            for (const fence_op& op : desc.signals)
                signal_when_done(last, op.fence, op.value);
            [last commit];
        }
        {
            std::lock_guard lock(list_mutex_);
            for (conduit::command_list* base : desc.lists)
                free_lists_.push_back(static_cast<cmd_list*>(base));
        }
        stats_.submits++;
    }

    std::unique_ptr<conduit::swapchain> device::create_swapchain(const swapchain_desc& desc) {
        return make_swapchain(*this, desc);
    }
}
