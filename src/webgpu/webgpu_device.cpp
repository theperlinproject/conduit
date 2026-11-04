#include "webgpu/webgpu_internal.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <type_traits>

#if !defined(__EMSCRIPTEN__)
#include <webgpu/wgpu.h>
#endif

namespace conduit::webgpu {
    namespace {
        constexpr const char* shader_prefix = "diagnostic(off, derivative_uniformity);\n";

        struct fence_signal {
            std::shared_ptr<std::atomic<uint64_t>> counter;
            uint64_t value;
        };

        void raise(void* userdata) {
            const std::unique_ptr<fence_signal> signal(static_cast<fence_signal*>(userdata));
            uint64_t seen = signal->counter->load();
            while (seen < signal->value && !signal->counter->compare_exchange_weak(seen, signal->value)) {}
        }

        template <class Fn> constexpr auto queue_done_for() {
            if constexpr (std::is_convertible_v<void (*)(WGPUQueueWorkDoneStatus, void*, void*), Fn>)
                return +[](WGPUQueueWorkDoneStatus, void* a, void*) { raise(a); };
            else
                return +[](WGPUQueueWorkDoneStatus, WGPUStringView, void* a, void*) { raise(a); };
        }

        struct map_state {
            bool done = false;
            bool ok = false;
        };

        void on_mapped(WGPUMapAsyncStatus status, WGPUStringView, void* a, void*) {
            auto* state = static_cast<map_state*>(a);
            state->ok = status == WGPUMapAsyncStatus_Success;
            state->done = true;
        }

        uint64_t stage_bits(shader_stage s) {
            uint64_t bits = 0;
            if (s & shader_stage::vertex)
                bits |= WGPUShaderStage_Vertex;
            if (s & shader_stage::fragment)
                bits |= WGPUShaderStage_Fragment;
            if (s & shader_stage::compute)
                bits |= WGPUShaderStage_Compute;
            return bits;
        }

        bool kind_matches(binding_type type, resource_kind kind) {
            switch (type) {
            case binding_type::uniform_buffer:
            case binding_type::uniform_buffer_dynamic:
                return kind == resource_kind::uniform_buffer;
            case binding_type::storage_buffer:
                return kind == resource_kind::storage_buffer || kind == resource_kind::storage_buffer_rw;
            case binding_type::storage_buffer_rw:
                return kind == resource_kind::storage_buffer_rw;
            case binding_type::sampled_texture:
                return kind == resource_kind::sampled_texture;
            case binding_type::storage_texture_rw:
                return kind == resource_kind::storage_texture;
            case binding_type::sampler:
                return kind == resource_kind::sampler;
            }
            return false;
        }
    }

    webgpu_device::webgpu_device(owned<WGPUInstance> instance, owned<WGPUAdapter> adapter, owned<WGPUDevice> device,
                                 const device_options& options, const adapter_info& info, bool map_sync)
        : instance_(std::move(instance)), adapter_(std::move(adapter)), device_(std::move(device)), options_(options),
          map_sync_(map_sync) {
        caps_.adapter = info;
    }

    webgpu_device::~webgpu_device() {
        wait_idle();
        lists_.clear();
    }

    void webgpu_device::report_error(const std::string& message) {
        std::lock_guard lock(error_mutex_);
        if (first_error_.empty())
            first_error_ = message;
        std::fprintf(stderr, "[webgpu] %s\n", message.c_str());
    }

    native_handles webgpu_device::native() const {
        native_handles n;
        n.api = api::webgpu;
        n.instance = instance_.get();
        n.device = device_.get();
        n.adapter = adapter_.get();
        return n;
    }

    bool webgpu_device::init() {
        queue_ = owned<WGPUQueue>(wgpuDeviceGetQueue(device_.get()));
        if (!queue_)
            return false;
        fill_caps();

        WGPUSamplerDescriptor sd{};
        sd.label = view("nearest");
        sd.addressModeU = sd.addressModeV = sd.addressModeW = WGPUAddressMode_ClampToEdge;
        sd.magFilter = sd.minFilter = WGPUFilterMode_Nearest;
        sd.mipmapFilter = WGPUMipmapFilterMode_Nearest;
        sd.lodMaxClamp = 32.0f;
        sd.maxAnisotropy = 1;
        nearest_sampler_ = owned<WGPUSampler>(wgpuDeviceCreateSampler(device_.get(), &sd));

        WGPUBindGroupLayoutDescriptor ld{};
        ld.label = view("empty");
        empty_layout_ = owned<WGPUBindGroupLayout>(wgpuDeviceCreateBindGroupLayout(device_.get(), &ld));

        WGPUBindGroupDescriptor gd{};
        gd.label = view("empty group");
        gd.layout = empty_layout_.get();
        empty_group_ = owned<WGPUBindGroup>(wgpuDeviceCreateBindGroup(device_.get(), &gd));

        frame_fence_ = create_fence(0);
        return nearest_sampler_ && empty_layout_ && empty_group_;
    }

    void webgpu_device::fill_caps() {
        WGPULimits lim{};
        wgpuDeviceGetLimits(device_.get(), &lim);
        device_limits& l = caps_.limits;
        l.max_texture2d = lim.maxTextureDimension2D;
        l.max_texture_layers = lim.maxTextureArrayLayers;
        l.max_bind_groups = std::min<uint32_t>(max_bind_groups, lim.maxBindGroups);
        l.max_push_constant_bytes = 0;
        l.max_sampled_textures_per_stage = lim.maxSampledTexturesPerShaderStage;
        l.max_color_targets = std::min<uint32_t>(max_color_targets, lim.maxColorAttachments);
        l.max_msaa_samples = 4;
        l.uniform_buffer_alignment = lim.minUniformBufferOffsetAlignment;
        l.storage_buffer_alignment = lim.minStorageBufferOffsetAlignment;
        l.copy_row_pitch_alignment = 256;
        l.copy_offset_alignment = 256;
        l.max_buffer_size = lim.maxBufferSize;
        l.max_anisotropy = 16.0f;

        const auto has = [&](WGPUFeatureName f) { return wgpuDeviceHasFeature(device_.get(), f) != 0; };
        device_features& f = caps_.features;
        f.sampler_anisotropy = true;
        f.texture_compression_bc = has(WGPUFeatureName_TextureCompressionBC);
        f.timestamp_queries = false;
        f.depth_clip_control = has(WGPUFeatureName_DepthClipControl);
        f.draw_indirect_first_instance = has(WGPUFeatureName_IndirectFirstInstance);
        f.shader_float16 = has(WGPUFeatureName_ShaderF16);
        f.unified_memory = caps_.adapter.type == adapter_type::integrated || caps_.adapter.type == adapter_type::cpu;
        f.async_compute = false;
        f.dedicated_copy_queue = false;
        f.multi_draw_indirect = false;
        f.draw_indirect_count = false;

        const bool f32_filter = has(WGPUFeatureName_Float32Filterable);
        const bool f32_blend = has(WGPUFeatureName_Float32Blendable);
        const bool rg11 = has(WGPUFeatureName_RG11B10UfloatRenderable);
        const bool d32s8 = has(WGPUFeatureName_Depth32FloatStencil8);
        const format_use sampled = format_use::sampled, filterable = format_use::sampled_linear,
                         color = format_use::color_target, blend = format_use::blendable,
                         depth = format_use::depth_target, storage = format_use::storage, msaa = format_use::msaa;
        const auto set = [&](format fmt, format_use use) { caps_.formats[static_cast<size_t>(fmt)] = use; };
        for (format fmt :
             {format::r8_unorm, format::rg8_unorm, format::rgba8_unorm, format::rgba8_srgb, format::bgra8_unorm,
              format::bgra8_srgb, format::r16_float, format::rg16_float, format::rgba16_float, format::rgb10a2_unorm})
            set(fmt, sampled | filterable | color | blend | msaa);
        set(format::rgba8_unorm, sampled | filterable | color | blend | msaa | storage);
        set(format::rgba16_float, sampled | filterable | color | blend | msaa | storage);
        set(format::r8_uint, sampled | color | msaa);
        set(format::r16_uint, sampled | color | msaa);
        set(format::r32_uint, sampled | color | storage);
        set(format::rg32_uint, sampled | color | storage);
        set(format::rgba32_uint, sampled | color | storage);
        format_use f32 = sampled | color | storage;
        if (f32_filter)
            f32 = f32 | filterable;
        if (f32_blend)
            f32 = f32 | blend;
        set(format::r32_float, f32);
        set(format::rg32_float, f32);
        set(format::rgba32_float, f32);
        set(format::r11g11b10_float, rg11 ? (sampled | filterable | color | blend | msaa) : (sampled | filterable));
        set(format::d16_unorm, sampled | depth | msaa);
        set(format::d32_float, sampled | depth | msaa);
        set(format::d24_unorm_s8_uint, sampled | depth | msaa);
        if (d32s8)
            set(format::d32_float_s8_uint, sampled | depth | msaa);
        if (f.texture_compression_bc)
            for (format fmt : {format::bc1_unorm, format::bc3_unorm, format::bc7_unorm})
                set(fmt, sampled | filterable);
    }

    buffer_handle webgpu_device::create_buffer(const buffer_desc& desc) {
        buffer b;
        b.desc = desc;
        b.name = desc.name ? desc.name : "";
        b.gpu_size = align_up(std::max<uint64_t>(desc.size, 4), 4);
        WGPUBufferUsage usage = 0;
        if (desc.memory == memory_class::readback) {
            usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        } else {
            if (has(desc.usage, buffer_usage::vertex))
                usage |= WGPUBufferUsage_Vertex;
            if (has(desc.usage, buffer_usage::index))
                usage |= WGPUBufferUsage_Index;
            if (has(desc.usage, buffer_usage::uniform))
                usage |= WGPUBufferUsage_Uniform;
            if (has(desc.usage, buffer_usage::storage))
                usage |= WGPUBufferUsage_Storage;
            if (has(desc.usage, buffer_usage::indirect))
                usage |= WGPUBufferUsage_Indirect;
            if (has(desc.usage, buffer_usage::copy_src))
                usage |= WGPUBufferUsage_CopySrc;
            usage |= WGPUBufferUsage_CopyDst;
        }
        WGPUBufferDescriptor bd{};
        bd.label = view(b.name);
        bd.usage = usage;
        bd.size = b.gpu_size;
        b.native = owned<WGPUBuffer>(wgpuDeviceCreateBuffer(device_.get(), &bd));
        if (!b.native)
            return {};
        if (desc.memory != memory_class::gpu_only)
            b.shadow.assign(b.gpu_size, 0);
        stats_.buffer_bytes += b.gpu_size;
        stats_.live_buffers++;
        const buffer_handle h = buffers_.alloc(std::move(b));
        buffers_.get(h)->desc.name = buffers_.get(h)->name.c_str();
        return h;
    }

    texture_handle webgpu_device::create_texture(const texture_desc& desc) {
        const WGPUTextureFormat format = to_texture_format(desc.format);
        if (format == WGPUTextureFormat_Undefined) {
            report_error(std::string("CreateTexture: format not available on webgpu: ") +
                         std::to_string(static_cast<int>(desc.format)));
            return {};
        }
        texture t;
        t.desc = desc;
        t.desc.mips = (std::max)(1u, desc.mips);
        t.desc.samples = (std::max)(1u, desc.samples);
        t.name = desc.name ? desc.name : "";
        WGPUTextureUsage usage = 0;
        if (has(desc.usage, texture_usage::sampled))
            usage |= WGPUTextureUsage_TextureBinding;
        if (has(desc.usage, texture_usage::storage))
            usage |= WGPUTextureUsage_StorageBinding;
        if (has(desc.usage, texture_usage::color_target) || has(desc.usage, texture_usage::depth_target))
            usage |= WGPUTextureUsage_RenderAttachment;
        if (has(desc.usage, texture_usage::copy_src))
            usage |= WGPUTextureUsage_CopySrc;
        if (has(desc.usage, texture_usage::copy_dst))
            usage |= WGPUTextureUsage_CopyDst;
        WGPUTextureDescriptor td{};
        td.label = view(t.name);
        td.usage = usage;
        td.dimension = desc.dim == texture_dim::tex3d ? WGPUTextureDimension_3D : WGPUTextureDimension_2D;
        td.size = {desc.width, desc.height,
                   desc.dim == texture_dim::tex3d ? (std::max)(1u, desc.depth_or_layers) : layer_count_of(t.desc)};
        td.format = format;
        td.mipLevelCount = t.desc.mips;
        td.sampleCount = t.desc.samples;
        t.native = owned<WGPUTexture>(wgpuDeviceCreateTexture(device_.get(), &td));
        if (!t.native)
            return {};
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

    sampler_handle webgpu_device::create_sampler(const sampler_desc& desc) {
        sampler s;
        s.desc = desc;
        s.linear = desc.min == filter::linear || desc.mag == filter::linear || desc.mip == filter::linear;
        WGPUSamplerDescriptor sd{};
        sd.label = view("sampler");
        sd.addressModeU = to_address(desc.u);
        sd.addressModeV = to_address(desc.v);
        sd.addressModeW = to_address(desc.w);
        sd.magFilter = to_filter(desc.mag);
        sd.minFilter = to_filter(desc.min);
        sd.mipmapFilter = to_mip_filter(desc.mip);
        sd.lodMinClamp = desc.min_lod;
        sd.lodMaxClamp = (std::min)(desc.max_lod, 32.0f);
        sd.compare = desc.compare ? to_compare(desc.compare_op) : WGPUCompareFunction_Undefined;
        const bool all_linear = desc.min == filter::linear && desc.mag == filter::linear && desc.mip == filter::linear;
        sd.maxAnisotropy = all_linear ? static_cast<uint16_t>(std::clamp(desc.max_anisotropy, 1.0f, 16.0f)) : 1;
        s.native = owned<WGPUSampler>(wgpuDeviceCreateSampler(device_.get(), &sd));
        if (!s.native)
            return {};
        return samplers_.alloc(std::move(s));
    }

    shader_handle webgpu_device::create_shader(const shader_desc& desc) {
        if (desc.format != shader_format::wgsl) {
            report_error("CreateShader: webgpu takes wgsl");
            return {};
        }
        shader s;
        s.desc = desc;
        s.desc.code = {};
        s.desc.entry = nullptr;
        s.desc.name = nullptr;
        s.entry = desc.entry ? desc.entry : "main";
        s.source =
            std::string(shader_prefix) + std::string(reinterpret_cast<const char*>(desc.code.data()), desc.code.size());
        WGPUShaderSourceWGSL wgsl{};
        wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgsl.code = view(s.source);
        WGPUShaderModuleDescriptor md{};
        md.nextInChain = &wgsl.chain;
        md.label = view(desc.name);
        s.module = owned<WGPUShaderModule>(wgpuDeviceCreateShaderModule(device_.get(), &md));
        if (!s.module)
            return {};
        s.iface = reflect_shader(shader_format::wgsl, s.source);
        return shaders_.alloc(std::move(s));
    }

    bind_layout_handle webgpu_device::create_bind_layout(const bind_layout_desc& desc) {
        bind_layout l;
        l.entries.assign(desc.entries.begin(), desc.entries.end());
        std::sort(l.entries.begin(), l.entries.end(),
                  [](const bind_layout_entry& a, const bind_layout_entry& b) { return a.binding < b.binding; });
        return bind_layouts_.alloc(std::move(l));
    }

    bind_group_handle webgpu_device::create_bind_group(const bind_group_desc& desc) {
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

    pipeline_layout_handle webgpu_device::create_pipeline_layout(const pipeline_layout_desc& desc) {
        if (desc.push_constant_bytes)
            report_error("CreatePipelineLayout: webgpu has no push constants");
        pipeline_layout l;
        l.desc = desc;
        l.desc.name = nullptr;
        return pipeline_layouts_.alloc(l);
    }

    query_set_handle webgpu_device::create_query_set(const query_set_desc& desc) {
        query q;
        q.count = desc.count;
        return queries_.alloc(std::move(q));
    }

    void webgpu_device::destroy(buffer_handle h) {
        buffer b;
        if (!buffers_.free(h, &b))
            return;
        stats_.buffer_bytes -= b.gpu_size;
        stats_.live_buffers--;
    }

    void webgpu_device::destroy(texture_handle h) {
        texture t;
        if (!textures_.free(h, &t))
            return;
        stats_.live_textures--;
    }

    void webgpu_device::destroy(sampler_handle h) {
        samplers_.free(h);
    }
    void webgpu_device::destroy(shader_handle h) {
        shaders_.free(h);
    }
    void webgpu_device::destroy(bind_layout_handle h) {
        bind_layouts_.free(h);
    }
    void webgpu_device::destroy(bind_group_handle h) {
        if (bind_groups_.free(h))
            stats_.live_bind_groups--;
    }
    void webgpu_device::destroy(pipeline_layout_handle h) {
        pipeline_layouts_.free(h);
    }
    void webgpu_device::destroy(pipeline_handle h) {
        if (pipelines_.free(h))
            stats_.live_pipelines--;
    }
    void webgpu_device::destroy(query_set_handle h) {
        queries_.free(h);
    }
    void webgpu_device::destroy(fence_handle h) {
        fences_.free(h);
    }

    const buffer_desc* webgpu_device::get_desc(buffer_handle h) const {
        const buffer* b = buffers_.get(h);
        return b ? &b->desc : nullptr;
    }

    const texture_desc* webgpu_device::get_desc(texture_handle h) const {
        const texture* t = textures_.get(h);
        return t ? &t->desc : nullptr;
    }

    uint8_t* webgpu_device::map(buffer_handle handle) {
        buffer* b = buffers_.get(handle);
        return b && !b->shadow.empty() ? b->shadow.data() : nullptr;
    }

    void webgpu_device::flush(buffer_handle handle, uint64_t offset, uint64_t size) {
        buffer* b = buffers_.get(handle);
        if (!b || b->shadow.empty() || b->desc.memory == memory_class::readback)
            return;
        const uint64_t lo = offset & ~3ull;
        const uint64_t hi = std::min<uint64_t>(b->gpu_size, align_up(offset + size, 4));
        if (hi <= lo)
            return;
        wgpuQueueWriteBuffer(queue_.get(), b->native.get(), lo, b->shadow.data() + lo, static_cast<size_t>(hi - lo));
    }

    void webgpu_device::invalidate(buffer_handle handle, uint64_t, uint64_t) {
        buffer* b = buffers_.get(handle);
        if (!b || b->desc.memory != memory_class::readback || !map_sync_)
            return;
        map_state state;
        WGPUBufferMapCallbackInfo info{};
        info.mode = WGPUCallbackMode_AllowProcessEvents;
        info.callback = on_mapped;
        info.userdata1 = &state;
        wgpuBufferMapAsync(b->native.get(), WGPUMapMode_Read, 0, static_cast<size_t>(b->gpu_size), info);
        if (!pump([&] { return state.done; }, 10'000'000'000ull) || !state.ok) {
            report_error("Invalidate: readback map failed");
            return;
        }
        if (const void* p = wgpuBufferGetConstMappedRange(b->native.get(), 0, static_cast<size_t>(b->gpu_size)))
            std::memcpy(b->shadow.data(), p, b->shadow.size());
        wgpuBufferUnmap(b->native.get());
    }

    fence_handle webgpu_device::create_fence(uint64_t initial_value) {
        fence f;
        f.value->store(initial_value);
        return fences_.alloc(std::move(f));
    }

    uint64_t webgpu_device::fence_value(fence_handle fence) {
        webgpu::fence* f = fences_.get(fence);
        return f ? f->value->load() : 0;
    }

    void webgpu_device::signal_fence(fence_handle fence, uint64_t value) {
        if (webgpu::fence* f = fences_.get(fence)) {
            uint64_t seen = f->value->load();
            while (seen < value && !f->value->compare_exchange_weak(seen, value)) {}
        }
    }

    bool webgpu_device::wait_fence(fence_handle fence, uint64_t value, uint64_t timeout_ns) {
        webgpu::fence* f = fences_.get(fence);
        if (!f)
            return true;
        const auto counter = f->value;
        return pump([&] { return counter->load() >= value; }, timeout_ns);
    }

    bool webgpu_device::pump(const std::function<bool()>& done, uint64_t timeout_ns) {
        if (done())
            return true;
#if defined(__EMSCRIPTEN__)
        (void)timeoutNs;
        wgpuInstanceProcessEvents(instance.Get());
        return done();
#else
        const auto start = std::chrono::steady_clock::now();
        for (;;) {
            wgpuDevicePoll(device_.get(), false, nullptr);
            wgpuInstanceProcessEvents(instance_.get());
            if (done())
                return true;
            if (timeout_ns != ~0ull && static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                                 std::chrono::steady_clock::now() - start)
                                                                 .count()) >= timeout_ns)
                return false;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
#endif
    }

    void webgpu_device::begin_frame() {
        frame_++;
        const uint64_t behind = (std::max)(1u, options_.frames_in_flight);
        if (frame_ > behind)
            wait_fence(frame_fence_, frame_ - behind, 2'000'000'000ull);
        else
            pump([] { return true; }, 0);
#if !defined(__EMSCRIPTEN__)
        wgpuInstanceProcessEvents(instance_.get());
#endif
    }

    void webgpu_device::end_frame() {
        fence* f = fences_.get(frame_fence_);
        if (!f)
            return;
        using fn = decltype(WGPUQueueWorkDoneCallbackInfo::callback);
        WGPUQueueWorkDoneCallbackInfo info{};
        info.mode = WGPUCallbackMode_AllowProcessEvents;
        info.callback = queue_done_for<fn>();
        info.userdata1 = new fence_signal{f->value, frame_};
        wgpuQueueOnSubmittedWorkDone(queue_.get(), info);
    }

    void webgpu_device::wait_idle() {
        if (!device_)
            return;
#if defined(__EMSCRIPTEN__)
        wgpuInstanceProcessEvents(instance.Get());
#else
        wgpuDevicePoll(device_.get(), true, nullptr);
        wgpuInstanceProcessEvents(instance_.get());
#endif
    }

    memory_budget webgpu_device::query_memory_budget() {
        memory_budget m;
        m.device_usage = stats_.buffer_bytes + stats_.texture_bytes;
        return m;
    }

    command_list* webgpu_device::acquire_command_list(conduit::queue_type queue_type) {
        std::lock_guard lock(list_mutex_);
        webgpu_command_list* list;
        if (!free_lists_.empty()) {
            list = free_lists_.back();
            free_lists_.pop_back();
        } else {
            lists_.push_back(std::make_unique<webgpu_command_list>(*this, queue_type));
            list = lists_.back().get();
        }
        list->begin();
        return list;
    }

    void webgpu_device::submit(queue_type, const submit_desc& desc) {
        std::vector<WGPUCommandBuffer> commands;
        for (command_list* base : desc.lists) {
            auto* list = static_cast<webgpu_command_list*>(base);
            if (WGPUCommandBuffer cb = list->finish())
                commands.push_back(cb);
        }
        if (!commands.empty())
            wgpuQueueSubmit(queue_.get(), commands.size(), commands.data());
        for (WGPUCommandBuffer cb : commands)
            wgpuCommandBufferRelease(cb);
        {
            std::lock_guard lock(list_mutex_);
            for (command_list* base : desc.lists)
                free_lists_.push_back(static_cast<webgpu_command_list*>(base));
        }
        stats_.submits++;

        for (const fence_op& op : desc.signals) {
            fence* f = fences_.get(op.fence);
            if (!f)
                continue;
            using fn = decltype(WGPUQueueWorkDoneCallbackInfo::callback);
            WGPUQueueWorkDoneCallbackInfo info{};
            info.mode = WGPUCallbackMode_AllowProcessEvents;
            info.callback = queue_done_for<fn>();
            info.userdata1 = new fence_signal{f->value, op.value};
            wgpuQueueOnSubmittedWorkDone(queue_.get(), info);
        }
    }

    WGPUTextureView webgpu_device::view_of(webgpu::texture& texture, const subresource_range& range, texture_dim dim,
                                           conduit::format format, bool stencil, bool attachment) {
        const subresource_range r = resolve_range(range, texture.desc);
        const bool depth_format = is_depth_format(texture.desc.format);
        view_key key;
        key.base_mip = r.base_mip;
        key.mip_count = r.mip_count;
        key.base_layer = r.base_layer;
        key.layer_count = dim == texture_dim::tex2d ? 1 : dim == texture_dim::cube ? 6 : r.layer_count;
        key.dim = static_cast<uint16_t>(dim);
        key.format = static_cast<uint16_t>(format);
        key.aspect = attachment || !depth_format ? 0 : (stencil ? 2 : 1);
        std::lock_guard lock(texture.view_mutex);
        if (const auto it = texture.views.find(key); it != texture.views.end())
            return it->second.get();
        WGPUTextureViewDescriptor vd{};
        vd.label = view("view");
        vd.format = format == conduit::format::unknown ? WGPUTextureFormat_Undefined : to_texture_format(format);
        vd.dimension = to_view_dimension(dim);
        vd.baseMipLevel = key.base_mip;
        vd.mipLevelCount = key.mip_count;
        vd.baseArrayLayer = key.base_layer;
        vd.arrayLayerCount = key.layer_count;
        vd.aspect = key.aspect == 0 ? WGPUTextureAspect_All
                                    : (key.aspect == 1 ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_StencilOnly);
        vd.usage = 0;
        WGPUTextureView view = wgpuTextureCreateView(texture.native.get(), &vd);
        if (view)
            texture.views.emplace(key, owned<WGPUTextureView>(view));
        return view;
    }

    std::shared_ptr<native_set> webgpu_device::cached_layout(const std::string& key,
                                                             const std::function<std::shared_ptr<native_set>()>& make) {
        std::lock_guard lock(layout_mutex_);
        if (const auto it = layout_cache_.find(key); it != layout_cache_.end())
            return it->second;
        std::shared_ptr<native_set> made = make();
        if (made && made->layout)
            layout_cache_[key] = made;
        return made;
    }

    std::shared_ptr<native_set> webgpu_device::build_native_set(const pipeline_layout_desc& pl, uint32_t set,
                                                                std::span<const shader* const> stages, shader_stage) {
        const bind_layout* layout = bind_layouts_.get(pl.sets[set]);
        std::vector<native_set::entry> chosen;
        std::vector<uint64_t> visibility;
        uint32_t rhi_dynamic = 0;
        std::vector<uint32_t> pick;
        if (layout) {
            for (const bind_layout_entry& e : layout->entries) {
                uint64_t vis = 0;
                native_set::entry entry;
                entry.rhi = e;
                bool used = false;
                for (const shader* sh : stages) {
                    if (!sh)
                        continue;
                    for (const shader_resource& r : sh->iface.resources) {
                        if (r.set != set || r.binding != e.binding || !kind_matches(e.type, r.kind))
                            continue;
                        used = true;
                        vis |= stage_bits(sh->desc.stage);
                        entry.depth |= r.depth;
                        entry.comparison |= r.comparison;
                    }
                }
                const bool dynamic = e.type == binding_type::uniform_buffer_dynamic;
                if (used) {
                    if (dynamic)
                        pick.push_back(rhi_dynamic);
                    chosen.push_back(entry);
                    visibility.push_back(vis);
                }
                if (dynamic)
                    rhi_dynamic++;
            }
        }

        std::string key;
        for (size_t i = 0; i < chosen.size(); i++) {
            const native_set::entry& e = chosen[i];
            key += std::to_string(e.rhi.binding) + ':' + std::to_string(static_cast<int>(e.rhi.type)) + ':' +
                   std::to_string(visibility[i]) + ':' + std::to_string(static_cast<int>(e.rhi.view_dim)) +
                   (e.rhi.filterable ? 'f' : 'n') + (e.depth ? 'd' : '-') + (e.comparison ? 'c' : '-') + ';';
        }
        auto made = cached_layout(key, [&]() -> std::shared_ptr<native_set> {
            auto result = std::make_shared<native_set>();
            result->entries = chosen;
            std::vector<WGPUBindGroupLayoutEntry> native(chosen.size());
            for (size_t i = 0; i < chosen.size(); i++) {
                const bind_layout_entry& e = chosen[i].rhi;
                WGPUBindGroupLayoutEntry& n = native[i];
                n = {};
                n.binding = e.binding;
                n.visibility = visibility[i];
                switch (e.type) {
                case binding_type::uniform_buffer:
                case binding_type::uniform_buffer_dynamic:
                    n.buffer.type = WGPUBufferBindingType_Uniform;
                    n.buffer.hasDynamicOffset = e.type == binding_type::uniform_buffer_dynamic;
                    break;
                case binding_type::storage_buffer:
                    n.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
                    break;
                case binding_type::storage_buffer_rw:
                    n.buffer.type = WGPUBufferBindingType_Storage;
                    break;
                case binding_type::sampled_texture:
                    n.texture.sampleType = chosen[i].depth ? WGPUTextureSampleType_Depth
                                           : e.filterable  ? WGPUTextureSampleType_Float
                                                           : WGPUTextureSampleType_UnfilterableFloat;
                    n.texture.viewDimension = to_view_dimension(e.view_dim);
                    break;
                case binding_type::sampler:
                    n.sampler.type = chosen[i].comparison ? WGPUSamplerBindingType_Comparison
                                     : e.filterable       ? WGPUSamplerBindingType_Filtering
                                                          : WGPUSamplerBindingType_NonFiltering;
                    break;
                case binding_type::storage_texture_rw:
                    n.storageTexture.access = WGPUStorageTextureAccess_ReadWrite;
                    n.storageTexture.format = WGPUTextureFormat_RGBA8Unorm;
                    n.storageTexture.viewDimension = to_view_dimension(e.view_dim);
                    break;
                }
            }
            WGPUBindGroupLayoutDescriptor ld{};
            ld.label = view("pipeline set");
            ld.entryCount = native.size();
            ld.entries = native.data();
            result->layout = owned<WGPUBindGroupLayout>(wgpuDeviceCreateBindGroupLayout(device_.get(), &ld));
            return result;
        });
        if (!made || !made->layout)
            return nullptr;
        auto mine = std::make_shared<native_set>();
        mine->entries = made->entries;
        mine->dynamic_pick = pick;
        mine->rhi_dynamic_count = rhi_dynamic;
        WGPUBindGroupLayout raw = made->layout.get();
        wgpuBindGroupLayoutAddRef(raw);
        mine->layout = owned<WGPUBindGroupLayout>(raw);
        return mine;
    }

    WGPUBindGroup webgpu_device::native_group_for(bind_group_handle handle, const native_set& set) {
        bind_group* g = bind_groups_.get(handle);
        if (!g)
            return nullptr;
        if (const auto it = g->native.find(set.layout.get()); it != g->native.end())
            return it->second.get();
        std::vector<WGPUBindGroupEntry> entries(set.entries.size());
        for (size_t i = 0; i < set.entries.size(); i++) {
            const bind_layout_entry& le = set.entries[i].rhi;
            WGPUBindGroupEntry& n = entries[i];
            n = {};
            n.binding = le.binding;
            const bind_group_entry* src = nullptr;
            for (const bind_group_entry& e : g->entries)
                if (e.binding == le.binding && e.array_index == 0) {
                    src = &e;
                    break;
                }
            if (!src) {
                report_error("bind group lacks binding " + std::to_string(le.binding));
                return nullptr;
            }
            switch (le.type) {
            case binding_type::uniform_buffer:
            case binding_type::uniform_buffer_dynamic:
            case binding_type::storage_buffer:
            case binding_type::storage_buffer_rw: {
                buffer* b = buffers_.get(src->buffer.buffer);
                if (!b) {
                    report_error("bind group buffer at binding " + std::to_string(le.binding) + " is gone");
                    return nullptr;
                }
                n.buffer = b->native.get();
                n.offset = src->buffer.offset;
                n.size = src->buffer.size ? src->buffer.size : b->gpu_size - src->buffer.offset;
                break;
            }
            case binding_type::sampled_texture:
            case binding_type::storage_texture_rw: {
                texture* t = textures_.get(src->texture.texture);
                if (!t) {
                    report_error("bind group texture at binding " + std::to_string(le.binding) + " is gone");
                    return nullptr;
                }
                n.textureView =
                    view_of(*t, src->texture.range, le.view_dim, src->texture.format, src->texture.stencil, false);
                break;
            }
            case binding_type::sampler: {
                sampler* s = samplers_.get(src->sampler);
                if (!s) {
                    report_error("bind group sampler at binding " + std::to_string(le.binding) + " is gone");
                    return nullptr;
                }
                n.sampler = (!le.filterable && s->linear && !set.entries[i].comparison) ? nearest_sampler_.get()
                                                                                        : s->native.get();
                break;
            }
            }
        }
        WGPUBindGroupDescriptor bd{};
        bd.label = view("bind group");
        bd.layout = set.layout.get();
        bd.entryCount = entries.size();
        bd.entries = entries.data();
        WGPUBindGroup made = wgpuDeviceCreateBindGroup(device_.get(), &bd);
        if (!made)
            return nullptr;
        g->native.emplace(set.layout.get(), owned<WGPUBindGroup>(made));
        return made;
    }

    texture_handle webgpu_device::create_surface_texture(conduit::format format, uint32_t width, uint32_t height,
                                                         bool sampled) {
        texture t;
        t.external = true;
        t.desc.format = format;
        t.desc.width = width;
        t.desc.height = height;
        t.desc.usage = texture_usage::color_target | texture_usage::copy_src |
                       (sampled ? texture_usage::sampled : texture_usage::none);
        t.name = "swapchain image";
        return textures_.alloc(std::move(t));
    }

    void webgpu_device::set_surface_texture(texture_handle handle, WGPUTexture texture) {
        webgpu::texture* t = textures_.get(handle);
        if (!t)
            return;
        {
            std::lock_guard lock(t->view_mutex);
            t->views.clear();
        }
        t->native = owned<WGPUTexture>(texture);
    }

    void webgpu_device::resize_surface_texture(texture_handle handle, uint32_t width, uint32_t height) {
        if (texture* t = textures_.get(handle)) {
            t->desc.width = width;
            t->desc.height = height;
        }
    }
}
