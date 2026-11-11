#include "backends.hpp"
#include "d3d12/d3d12_convert.hpp"
#include "d3d12/d3d12_device.hpp"

#include <algorithm>
#include <cstdio>

namespace conduit::d3d12 {
    namespace {
        D3D12_COMMAND_LIST_TYPE queue_type_of(ID3D12CommandQueue* queue) {
#ifdef __MINGW32__
            D3D12_COMMAND_QUEUE_DESC d{};
            queue->GetDesc(&d);
            return d.Type;
#else
            return queue->GetDesc().Type;
#endif
        }

        uint64_t luid_of(ID3D12Device* device) {
#ifdef __MINGW32__
            LUID id{};
            device->GetAdapterLuid(&id);
#else
            const LUID id = device->GetAdapterLuid();
#endif
            return (static_cast<uint64_t>(id.HighPart) << 32) | id.LowPart;
        }

        std::string narrow(const wchar_t* w) {
            std::string out;
            for (; *w; w++)
                out.push_back(*w < 0x80 ? static_cast<char>(*w) : '?');
            return out;
        }

        class d3d12_instance final : public instance {
        public:
            explicit d3d12_instance(const instance_options& options) : options_(options) {}

            bool init() {
                state_ = std::make_shared<instance_state>();
                library& lib = state_->lib;
                lib.d3d12 = LoadLibraryW(L"d3d12.dll");
                lib.dxgi = LoadLibraryW(L"dxgi.dll");
                if (!lib.d3d12 || !lib.dxgi)
                    return false;
                lib.create_device =
                    function_cast<PFN_D3D12_CREATE_DEVICE>(GetProcAddress(lib.d3d12, "D3D12CreateDevice"));
                lib.get_debug_interface =
                    function_cast<PFN_D3D12_GET_DEBUG_INTERFACE>(GetProcAddress(lib.d3d12, "D3D12GetDebugInterface"));
                lib.serialize_root_signature = function_cast<PFN_D3D12_SERIALIZE_VERSIONED_ROOT_SIGNATURE>(
                    GetProcAddress(lib.d3d12, "D3D12SerializeVersionedRootSignature"));
                lib.create_factory2 =
                    function_cast<decltype(lib.create_factory2)>(GetProcAddress(lib.dxgi, "CreateDXGIFactory2"));
                if (options_.hooks.dxgi_create_factory)
                    lib.create_factory2 =
                        reinterpret_cast<decltype(lib.create_factory2)>(options_.hooks.dxgi_create_factory);
                if (options_.hooks.d3d12_create_device)
                    lib.create_device = reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(options_.hooks.d3d12_create_device);
                if (!lib.create_device || !lib.serialize_root_signature || !lib.create_factory2)
                    return false;

                UINT factory_flags = 0;
                if (options_.validation && lib.get_debug_interface) {
                    winrt::com_ptr<ID3D12Debug> debug;
                    if (SUCCEEDED(lib.get_debug_interface(winrt::guid_of<ID3D12Debug>(), debug.put_void()))) {
                        debug->EnableDebugLayer();
                        state_->validation = true;
                        factory_flags |= DXGI_CREATE_FACTORY_DEBUG;
                    }
                }
                if (FAILED(lib.create_factory2(factory_flags, winrt::guid_of<IDXGIFactory4>(),
                                               state_->factory.put_void())))
                    return false;

                const auto f5 = state_->factory.try_as<IDXGIFactory5>();
                BOOL tearing = FALSE;
                if (f5 &&
                    SUCCEEDED(f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
                    state_->tearing = tearing != FALSE;

                enumerate_adapters();
                return true;
            }

            api get_api() const override { return api::d3d12; }
            const std::vector<adapter_info>& adapters() const override { return adapters_; }

            std::unique_ptr<conduit::device> adopt_device(const adopt_desc& desc,
                                                          const conduit::device_options& device_options) override {
                auto* native = static_cast<ID3D12Device*>(desc.device);
                auto* queue = static_cast<ID3D12CommandQueue*>(desc.queue);
                if (!native || !queue || queue_type_of(queue) != D3D12_COMMAND_LIST_TYPE_DIRECT)
                    return nullptr;
                const uint64_t luid = luid_of(native);
                for (size_t i = 0; i < adapters_.size(); i++) {
                    if (adapters_[i].luid != luid)
                        continue;
                    auto device = std::make_unique<d3d12::device>(state_, native_[i], adapters_[i], device_options);
                    device->adopt(native, queue);
                    std::string error;
                    if (!device->init(error)) {
                        std::fprintf(stderr, "[d3d12] adopting the device failed: %s\n", error.c_str());
                        return nullptr;
                    }
                    return device;
                }
                return nullptr;
            }

            std::unique_ptr<conduit::device> create_device(const conduit::device_options& device_options) override {
                if (device_options.adapter_index >= adapters_.size() || !adapters_[device_options.adapter_index].usable)
                    return nullptr;
                auto device = std::make_unique<d3d12::device>(state_, native_[device_options.adapter_index],
                                                              adapters_[device_options.adapter_index], device_options);
                std::string error;
                if (!device->init(error)) {
                    std::fprintf(stderr, "[d3d12] device creation failed: %s\n", error.c_str());
                    return nullptr;
                }
                return device;
            }

        private:
            void consider(winrt::com_ptr<IDXGIAdapter1> adapter) {
                DXGI_ADAPTER_DESC1 d{};
                adapter->GetDesc1(&d);
                adapter_info a;
                a.api = api::d3d12;
                a.index = static_cast<uint32_t>(adapters_.size());
                a.name = narrow(d.Description);
                a.vendor_id = d.VendorId;
                a.device_id = d.DeviceId;
                a.dedicated_vram = d.DedicatedVideoMemory;
                a.shared_memory = d.SharedSystemMemory;
                a.luid = (static_cast<uint64_t>(d.AdapterLuid.HighPart) << 32) | d.AdapterLuid.LowPart;
                const bool software = (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
                a.type = software ? adapter_type::cpu
                                  : (d.DedicatedVideoMemory > (512ull << 20) ? adapter_type::discrete
                                                                             : adapter_type::integrated);

                LARGE_INTEGER umd{};
                if (SUCCEEDED(adapter->CheckInterfaceSupport(winrt::guid_of<IDXGIDevice>(), &umd)))
                    a.driver_version = static_cast<uint64_t>(umd.QuadPart);

                winrt::com_ptr<ID3D12Device> probe;
                if (FAILED(state_->lib.create_device(adapter.get(), D3D_FEATURE_LEVEL_12_0,
                                                     winrt::guid_of<ID3D12Device>(), probe.put_void()))) {
                    a.usable = false;
                    a.unusable_reason = "no Direct3D 12 feature level 12_0";
                } else {
                    static const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_12_2, D3D_FEATURE_LEVEL_12_1,
                                                               D3D_FEATURE_LEVEL_12_0};
                    D3D12_FEATURE_DATA_FEATURE_LEVELS fl{};
                    fl.NumFeatureLevels = static_cast<UINT>(std::size(levels));
                    fl.pFeatureLevelsRequested = levels;
                    if (SUCCEEDED(probe->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &fl, sizeof(fl))))
                        a.api_version = fl.MaxSupportedFeatureLevel;
                    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_0};
                    if (FAILED(probe->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))) ||
                        sm.HighestShaderModel < D3D_SHADER_MODEL_6_0) {
                        a.usable = false;
                        a.unusable_reason = "needs shader model 6.0";
                    }
                }
                adapters_.push_back(std::move(a));
                native_.push_back(std::move(adapter));
            }

            void enumerate_adapters() {
                winrt::com_ptr<IDXGIFactory6> f6;
                winrt::com_ptr<IDXGIAdapter1> warp;
                auto handle = [&](const winrt::com_ptr<IDXGIAdapter1>& a) {
                    DXGI_ADAPTER_DESC1 d{};
                    a->GetDesc1(&d);
                    if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                        if (!warp)
                            warp = a;
                        return;
                    }
                    consider(a);
                };
                f6 = state_->factory.try_as<IDXGIFactory6>();
                if (f6) {
                    for (UINT i = 0;; i++) {
                        winrt::com_ptr<IDXGIAdapter1> a;
                        if (FAILED(f6->EnumAdapterByGpuPreference(i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                                                                  winrt::guid_of<IDXGIAdapter1>(), a.put_void())))
                            break;
                        handle(a);
                    }
                } else {
                    for (UINT i = 0;; i++) {
                        winrt::com_ptr<IDXGIAdapter1> a;
                        if (FAILED(state_->factory->EnumAdapters1(i, a.put())))
                            break;
                        handle(a);
                    }
                }
                if (warp)
                    consider(warp);
            }

            instance_options options_;
            std::shared_ptr<instance_state> state_;
            std::vector<adapter_info> adapters_;
            std::vector<winrt::com_ptr<IDXGIAdapter1>> native_;
        };
    }

    device::device(std::shared_ptr<instance_state> instance, winrt::com_ptr<IDXGIAdapter1> adapter_in,
                   const adapter_info& info, const device_options& opts)
        : adapter_(std::move(adapter_in)), inst_(std::move(instance)), options_(opts) {
        caps_.adapter = info;
        options_.frames_in_flight = std::clamp(options_.frames_in_flight, 1u, 4u);
        use_validator_ = options_.state_validation;
    }

    device::~device() {
        stop_compile_threads();
        if (!device_)
            return;
        wait_idle();
        retire_.flush_all();
    }

    void device::adopt(ID3D12Device* device, ID3D12CommandQueue* queue) {
        device_.copy_from(device);
        queues_[0].queue.copy_from(queue);
    }

    texture_handle device::import_texture(void* native, const texture_desc& desc) {
        return native ? wrap_swapchain_image(static_cast<ID3D12Resource*>(native), desc) : texture_handle{};
    }

    bool device::init(std::string& error) {
        if (!device_ && FAILED(inst_->lib.create_device(adapter_.get(), D3D_FEATURE_LEVEL_12_0,
                                                        winrt::guid_of<ID3D12Device>(), device_.put_void()))) {
            error = "D3D12CreateDevice failed";
            return false;
        }
        device4_ = device_.try_as<ID3D12Device4>();
        wait_event_ = winrt::handle{CreateEventW(nullptr, FALSE, FALSE, nullptr)};

        if (inst_->validation)
            info_queue_ = device_.try_as<ID3D12InfoQueue>();
        if (info_queue_) {
            D3D12_MESSAGE_SEVERITY severities[] = {D3D12_MESSAGE_SEVERITY_INFO};
            D3D12_MESSAGE_ID noise[] = {D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,
                                        D3D12_MESSAGE_ID_CLEARDEPTHSTENCILVIEW_MISMATCHINGCLEARVALUE,
                                        D3D12_MESSAGE_ID_EXECUTECOMMANDLISTS_GPU_WRITTEN_READBACK_RESOURCE_MAPPED};
            D3D12_INFO_QUEUE_FILTER filter{};
            filter.DenyList.NumSeverities = 1;
            filter.DenyList.pSeverityList = severities;
            filter.DenyList.NumIDs = static_cast<UINT>(std::size(noise));
            filter.DenyList.pIDList = noise;
            info_queue_->PushStorageFilter(&filter);
        }

        const D3D12_COMMAND_LIST_TYPE types[queue_count] = {
            D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_COMPUTE, D3D12_COMMAND_LIST_TYPE_COPY};
        for (size_t i = 0; i < queue_count; i++) {
            if (i == 1 && !options_.allow_async_compute)
                continue;
            D3D12_COMMAND_QUEUE_DESC qd{};
            qd.Type = types[i];
            qd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
            if (!queues_[i].queue && FAILED(device_->CreateCommandQueue(&qd, winrt::guid_of<ID3D12CommandQueue>(),
                                                                        queues_[i].queue.put_void()))) {
                error = "CreateCommandQueue failed";
                return false;
            }
            device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, winrt::guid_of<ID3D12Fence>(), queues_[i].fence.put_void());
        }

        D3D12MA::ALLOCATOR_DESC ad{};
        ad.pDevice = device_.get();
        ad.pAdapter = adapter_.get();
        ad.Flags = D3D12MA::ALLOCATOR_FLAG_MSAA_TEXTURES_ALWAYS_COMMITTED;
        if (FAILED(D3D12MA::CreateAllocator(&ad, allocator_.put()))) {
            error = "D3D12MA allocator creation failed";
            return false;
        }

        if (!rtv_pool_.init(device_.get(), D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2048) ||
            !dsv_pool_.init(device_.get(), D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 512) ||
            !resource_heap_.init(device_.get(), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 400000) ||
            !sampler_heap_.init(device_.get(), D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2048)) {
            error = "descriptor heap creation failed";
            return false;
        }

        if (const auto device1 = device_.try_as<ID3D12Device1>()) {
            HRESULT lr =
                device1->CreatePipelineLibrary(options_.pipeline_cache_data.data(), options_.pipeline_cache_data.size(),
                                               winrt::guid_of<ID3D12PipelineLibrary>(), pipeline_library_.put_void());
            if (FAILED(lr) && !options_.pipeline_cache_data.empty())
                device1->CreatePipelineLibrary(nullptr, 0, winrt::guid_of<ID3D12PipelineLibrary>(),
                                               pipeline_library_.put_void());
        }

        frame_groups_.resize(options_.frames_in_flight);
        slot_serials_.assign(options_.frames_in_flight, queue_serials{});
        fill_caps();
        return true;
    }

    void device::fill_caps() {
        device_features& f = caps_.features;
        device_limits& l = caps_.limits;
        D3D12_FEATURE_DATA_D3D12_OPTIONS o{};
        device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &o, sizeof(o));
        D3D12_FEATURE_DATA_D3D12_OPTIONS1 o1{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &o1, sizeof(o1)))) {
            f.subgroup_ops = o1.WaveOps;
            l.min_subgroup_size = o1.WaveLaneCountMin;
            l.max_subgroup_size = o1.WaveLaneCountMax;
        }
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &o4, sizeof(o4))))
            f.shader_float16 = o4.Native16BitShaderOpsSupported;
        D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))))
            f.ray_tracing = o5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1;
        D3D12_FEATURE_DATA_D3D12_OPTIONS6 o6{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &o6, sizeof(o6))))
            f.variable_rate_shading = o6.VariableShadingRateTier != D3D12_VARIABLE_SHADING_RATE_TIER_NOT_SUPPORTED;
        D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &o7, sizeof(o7))))
            f.mesh_shaders = o7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED;
        D3D12_FEATURE_DATA_D3D12_OPTIONS12 o12{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &o12, sizeof(o12))))
            f.enhanced_barriers = o12.EnhancedBarriersSupported;
        D3D12_FEATURE_DATA_D3D12_OPTIONS16 o16{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &o16, sizeof(o16))))
            f.resizable_bar = o16.GPUUploadHeapSupported;
        D3D12_FEATURE_DATA_ARCHITECTURE1 arch{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE1, &arch, sizeof(arch))))
            uma_ = f.unified_memory = arch.UMA != FALSE;
        D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_6};
        if (FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))))
            sm.HighestShaderModel = D3D_SHADER_MODEL_6_0;
        f.cooperative_matrix = false;

        D3D12_FEATURE_DATA_D3D12_OPTIONS21 o21{};
        if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21, &o21, sizeof(o21))))
            f.work_graphs = o21.WorkGraphsTier >= D3D12_WORK_GRAPHS_TIER_1_0;
        f.multi_draw_indirect = f.draw_indirect_count = f.draw_indirect_first_instance = true;
        f.gpu_driven = true;
        f.async_compute = options_.allow_async_compute;
        f.dedicated_copy_queue = true;
        f.timestamp_queries = true;
        f.pipeline_cache_blob = static_cast<bool>(pipeline_library_);
        start_compile_threads();
        f.texture_compression_bc = true;
        f.sampler_anisotropy = true;
        f.shader_int64 = o1.Int64ShaderOps != FALSE;
        f.shader_float64 = o.DoublePrecisionFloatShaderOps != FALSE;
        f.depth_clip_control = true;
        f.buffer_device_address = true;
        f.bindless = o.ResourceBindingTier >= D3D12_RESOURCE_BINDING_TIER_3;
        f.tearing = inst_->tearing;
        f.hdr_output = true;
        f.conservative_raster = o.ConservativeRasterizationTier != D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;

        l.max_texture2d = 16384;
        l.max_texture_layers = 2048;
        l.max_sampled_textures_per_stage = 128;
        l.max_color_targets = 8;
        l.uniform_buffer_alignment = 256;
        l.storage_buffer_alignment = 16;
        l.copy_row_pitch_alignment = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
        l.copy_offset_alignment = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
        l.max_anisotropy = 16.0f;
        l.max_buffer_size = 1ull << 31;
        UINT64 freq = 0;
        if (queues_[0].queue && SUCCEEDED(queues_[0].queue->GetTimestampFrequency(&freq)) && freq)
            l.timestamp_period_ns = static_cast<float>(1.0e9 / static_cast<double>(freq));

        l.max_msaa_samples = 1;
        for (UINT samples : {2u, 4u, 8u}) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS ms{DXGI_FORMAT_R8G8B8A8_UNORM, samples};
            if (SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &ms, sizeof(ms))) &&
                ms.NumQualityLevels > 0)
                l.max_msaa_samples = samples;
        }

        for (size_t i = 1; i < static_cast<size_t>(format::count); i++) {
            const format_map fm = map_format(static_cast<format>(i));
            D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{fm.dsv != DXGI_FORMAT_UNKNOWN ? fm.dsv : fm.view};
            if (fs.Format == DXGI_FORMAT_UNKNOWN ||
                FAILED(device_->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))))
                continue;
            format_use use = format_use::none;
            const UINT s1 = fs.Support1;
            UINT sample_bits = s1;
            if (fm.dsv != DXGI_FORMAT_UNKNOWN) {
                D3D12_FEATURE_DATA_FORMAT_SUPPORT vs{fm.view};
                sample_bits = SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &vs, sizeof(vs)))
                                  ? vs.Support1
                                  : 0;
            }
            if (sample_bits & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE)
                use = use | format_use::sampled | format_use::sampled_linear;
            if (s1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET)
                use = use | format_use::color_target | format_use::msaa;
            if (s1 & D3D12_FORMAT_SUPPORT1_BLENDABLE)
                use = use | format_use::blendable;
            if (s1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)
                use = use | format_use::depth_target | format_use::msaa;
            if (fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE)
                use = use | format_use::storage;
            caps_.formats[i] = use;
        }
    }

    native_handles device::native() const {
        native_handles n;
        n.api = api::d3d12;
        n.device = device_.get();
        n.graphics_queue = queues_[0].queue.get();
        n.compute_queue = queues_[1].queue ? queues_[1].queue.get() : queues_[0].queue.get();
        n.adapter = adapter_.get();
        n.factory = inst_->factory.get();
        return n;
    }

    native_texture device::native_of(texture_handle h, conduit::state state) {
        texture_rec* t = textures_.get(h);
        if (!t || !t->resource)
            return {};
        native_texture n;
        n.resource = t->resource.get();
        n.state = static_cast<uint32_t>(to_states(state, queue_type::graphics));
        n.native_format = static_cast<uint32_t>(t->resource_format);
        n.width = t->desc.width;
        n.height = t->desc.height;
        n.mips = t->mips;
        n.layers = t->layers;
        return n;
    }

    void device::report_error(const std::string& message) {
        {
            std::lock_guard lock(inst_->message_mutex);
            inst_->error_count++;
            if (inst_->messages.size() < 256)
                inst_->messages.push_back(message);
        }
        std::fprintf(stderr, "[rhi/d3d12] %s\n", message.c_str());
    }

    void device::check_removed(HRESULT hr, const char* what) {
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET || hr == DXGI_ERROR_DEVICE_HUNG) {
            report_error(std::string(what) + ": device removed (reason " +
                         std::to_string(device_->GetDeviceRemovedReason()) + ")");
        } else if (FAILED(hr)) {
            report_error(std::string(what) + " failed (" + std::to_string(static_cast<long>(hr)) + ")");
        }
    }

    memory_budget device::query_memory_budget() {
        memory_budget out;
        const auto a3 = adapter_.try_as<IDXGIAdapter3>();
        if (!a3)
            return out;
        DXGI_QUERY_VIDEO_MEMORY_INFO local{}, non_local{};
        if (SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local)) &&
            SUCCEEDED(a3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &non_local))) {
            out.device_budget = local.Budget;
            out.device_usage = local.CurrentUsage;
            out.host_budget = non_local.Budget;
            out.host_usage = non_local.CurrentUsage;
            out.valid = true;
        }
        return out;
    }

    std::string device::debug_report() {
        std::string out;
        {
            std::lock_guard lock(inst_->message_mutex);
            for (auto& m : inst_->messages)
                out += m + "\n";
            inst_->messages.clear();
            inst_->error_count = 0;
        }
        if (info_queue_) {
            const UINT64 count = info_queue_->GetNumStoredMessagesAllowedByRetrievalFilter();
            for (UINT64 i = 0; i < count; i++) {
                SIZE_T len = 0;
                info_queue_->GetMessage(i, nullptr, &len);
                std::vector<uint8_t> raw(len);
                auto* msg = reinterpret_cast<D3D12_MESSAGE*>(raw.data());
                if (SUCCEEDED(info_queue_->GetMessage(i, msg, &len)) && msg->Severity <= D3D12_MESSAGE_SEVERITY_WARNING)
                    out += std::string("d3d12: ") + msg->pDescription + "\n";
            }
            info_queue_->ClearStoredMessages();
        }
        if (use_validator_)
            for (auto& m : validator_.take_errors())
                out += "state: " + m + "\n";
        return out;
    }

    // the Windows SDK (and DirectX-Headers) return the handle; mingw-w64 uses the older out-parameter form
    static D3D12_CPU_DESCRIPTOR_HANDLE heap_cpu_start(ID3D12DescriptorHeap* h) {
#ifdef __MINGW32__
        D3D12_CPU_DESCRIPTOR_HANDLE r{};
        h->GetCPUDescriptorHandleForHeapStart(&r);
        return r;
#else
        return h->GetCPUDescriptorHandleForHeapStart();
#endif
    }

    static D3D12_GPU_DESCRIPTOR_HANDLE heap_gpu_start(ID3D12DescriptorHeap* h) {
#ifdef __MINGW32__
        D3D12_GPU_DESCRIPTOR_HANDLE r{};
        h->GetGPUDescriptorHandleForHeapStart(&r);
        return r;
#else
        return h->GetGPUDescriptorHandleForHeapStart();
#endif
    }

    bool cpu_descriptor_pool::init(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t cap) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = type;
        hd.NumDescriptors = cap;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if (FAILED(device->CreateDescriptorHeap(&hd, winrt::guid_of<ID3D12DescriptorHeap>(), heap_.put_void())))
            return false;
        cpu_start_ = heap_cpu_start(heap_.get());
        increment_ = device->GetDescriptorHandleIncrementSize(type);
        capacity_ = cap;
        return true;
    }

    uint32_t cpu_descriptor_pool::alloc() {
        std::lock_guard lock(mutex_);
        if (!free_list_.empty()) {
            const uint32_t i = free_list_.back();
            free_list_.pop_back();
            return i;
        }
        return next_ < capacity_ ? next_++ : UINT32_MAX;
    }

    void cpu_descriptor_pool::free(uint32_t index) {
        std::lock_guard lock(mutex_);
        free_list_.push_back(index);
    }

    bool range_heap::init(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t capacity) {
        D3D12_DESCRIPTOR_HEAP_DESC hd{};
        hd.Type = type;
        hd.NumDescriptors = capacity;
        hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        if (FAILED(device->CreateDescriptorHeap(&hd, winrt::guid_of<ID3D12DescriptorHeap>(), heap_.put_void())))
            return false;
        cpu_start_ = heap_cpu_start(heap_.get());
        gpu_start_ = heap_gpu_start(heap_.get());
        increment_ = device->GetDescriptorHandleIncrementSize(type);
        free_ranges_.emplace_back(0, capacity);
        return true;
    }

    bool range_heap::alloc(uint32_t count, uint32_t& offset) {
        std::lock_guard lock(mutex_);
        for (auto it = free_ranges_.begin(); it != free_ranges_.end(); ++it) {
            if (it->second < count)
                continue;
            offset = it->first;
            it->first += count;
            it->second -= count;
            if (it->second == 0)
                free_ranges_.erase(it);
            used_ += count;
            return true;
        }
        return false;
    }

    void range_heap::free(uint32_t offset, uint32_t count) {
        if (!count)
            return;
        std::lock_guard lock(mutex_);
        used_ -= count;
        auto it = std::lower_bound(free_ranges_.begin(), free_ranges_.end(), std::make_pair(offset, 0u));
        it = free_ranges_.insert(it, {offset, count});
        auto next = it + 1;
        if (next != free_ranges_.end() && it->first + it->second == next->first) {
            it->second += next->second;
            free_ranges_.erase(next);
        }
        if (it != free_ranges_.begin()) {
            auto prev = it - 1;
            if (prev->first + prev->second == it->first) {
                prev->second += it->second;
                free_ranges_.erase(it);
            }
        }
    }
}

namespace conduit::backends {
    std::unique_ptr<conduit::instance> create_d3d12_instance(const instance_options& options) {
        auto instance = std::make_unique<d3d12::d3d12_instance>(options);
        if (!instance->init())
            return nullptr;
        return instance;
    }
}
