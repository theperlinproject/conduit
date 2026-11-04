#include <conduit/vendor.hpp>

#if defined(CONDUIT_HAS_STREAMLINE) && CONDUIT_HAS_STREAMLINE

#include <windows.h>

#include <sl.h>
#include <sl_appidentity.h>
#include <sl_consts.h>
#include <sl_device_wrappers.h>
#include <sl_dlss.h>
#include <sl_dlss_g.h>
#include <sl_pcl.h>
#include <sl_reflex.h>
#if defined(_MSC_VER)
#include <sl_security.h>
#define CONDUIT_CAN_VERIFY_SIGNATURES 1
#else
#define CONDUIT_CAN_VERIFY_SIGNATURES 0
#endif

#include "common/util.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>

namespace conduit::vendor {
    namespace {
        constexpr uint32_t viewport = 0;

        struct sl_api {
            HMODULE module = nullptr;
            PFun_slInit* init = nullptr;
            PFun_slShutdown* shutdown = nullptr;
            PFun_slIsFeatureSupported* is_feature_supported = nullptr;
            PFun_slSetD3DDevice* set_d3d_device = nullptr;
            PFun_slGetNewFrameToken* get_new_frame_token = nullptr;
            PFun_slSetConstants* set_constants = nullptr;
            PFun_slSetTagForFrame* set_tag_for_frame = nullptr;
            PFun_slEvaluateFeature* evaluate_feature = nullptr;
            PFun_slFreeResources* free_resources = nullptr;
            PFun_slGetFeatureFunction* get_feature_function = nullptr;

            template <class F> bool bind(F*& slot, const char* name) {
                slot = function_cast<F*>(GetProcAddress(module, name));
                return slot != nullptr;
            }

            bool load(const std::wstring& dir, bool verify, std::string& log) {
                std::wstring path = dir.empty() ? L"" : dir + (dir.back() == L'\\' || dir.back() == L'/' ? L"" : L"\\");
                path += L"sl.interposer.dll";
                wchar_t full[MAX_PATH * 2];
                const DWORD n = GetFullPathNameW(path.c_str(), static_cast<DWORD>(std::size(full)), full, nullptr);
                if (n == 0 || n >= std::size(full) || GetFileAttributesW(full) == INVALID_FILE_ATTRIBUTES) {
                    log += "streamline: sl.interposer.dll not found; ";
                    return false;
                }
#if CONDUIT_CAN_VERIFY_SIGNATURES
                if (verify && !sl::security::verifyEmbeddedSignature(full)) {
                    log += "streamline: sl.interposer.dll is not signed; ";
                    return false;
                }
#else
                if (verify) {
                    log += "streamline: this build cannot verify module signatures; ";
                    return false;
                }
#endif
                module = LoadLibraryW(full);
                if (!module) {
                    log += "streamline: the interposer failed to load; ";
                    return false;
                }
                const bool ok =
                    bind(init, "slInit") && bind(shutdown, "slShutdown") &&
                    bind(is_feature_supported, "slIsFeatureSupported") && bind(set_d3d_device, "slSetD3DDevice") &&
                    bind(get_new_frame_token, "slGetNewFrameToken") && bind(set_constants, "slSetConstants") &&
                    bind(set_tag_for_frame, "slSetTagForFrame") && bind(evaluate_feature, "slEvaluateFeature") &&
                    bind(free_resources, "slFreeResources") && bind(get_feature_function, "slGetFeatureFunction");
                if (!ok) {
                    log += "streamline: the interposer lacks entry points; ";
                    FreeLibrary(module);
                    module = nullptr;
                }
                return ok;
            }
        };

        void sl_log(sl::LogType type, const char* message) {
            if (type == sl::LogType::eInfo)
                return;
            std::fprintf(stderr, "[rhi/streamline] %s", message);
        }

        void to_sl(const mat4& in, sl::float4x4& out) {
            for (uint32_t r = 0; r < 4; r++)
                out.setRow(r, sl::float4(in.m[r * 4 + 0], in.m[r * 4 + 1], in.m[r * 4 + 2], in.m[r * 4 + 3]));
        }

        sl::Boolean b(bool v) {
            return v ? sl::Boolean::eTrue : sl::Boolean::eFalse;
        }

        class streamline;

        class dlss_upscaler final : public upscaler {
        public:
            dlss_upscaler(streamline& owner, const upscale_desc& desc) : owner_(owner), desc_(desc) {}
            ~dlss_upscaler() override;

            const char* name() const override { return "DLSS"; }
            upscale_desc desc() const override { return desc_; }
            bool resize(const upscale_desc& d) override {
                desc_ = d;
                options_dirty_ = true;
                return true;
            }
            bool evaluate(command_list& list, const upscale_inputs& in) override;

        private:
            streamline& owner_;
            upscale_desc desc_;
            bool options_dirty_ = true;
        };

        class streamline final : public session, public frame_generator, public low_latency {
        public:
            ~streamline() override { shutdown(); }

            bool init(api session_api, const vendor::session_request& session_request, vendor_hooks& hooks,
                      std::string& log) {
                api_ = session_api;
                request_ = session_request;
                if (api_ != api::d3d12 && api_ != api::vulkan)
                    return false;
                if (!sl_.load(request_.modules_path, request_.verify_signatures, log))
                    return false;

                std::vector<sl::Feature> features;
                if (request_.upscaler)
                    features.push_back(sl::kFeatureDLSS);
                if (request_.frame_generation) {
                    features.push_back(sl::kFeatureDLSS_G);
                    request_.low_latency = true;
                }
                if (request_.low_latency) {
                    features.push_back(sl::kFeatureReflex);
                    features.push_back(sl::kFeaturePCL);
                }

                sl::Preferences pref{};
                pref.showConsole = false;
                pref.logLevel = request_.verbose_log ? sl::LogLevel::eVerbose : sl::LogLevel::eDefault;
                pref.logMessageCallback = sl_log;
                pref.pathToLogsAndData = request_.log_path.empty() ? nullptr : request_.log_path.c_str();
                path_to_plugins_ = request_.modules_path;
                if (!path_to_plugins_.empty()) {
                    plugin_paths_[0] = path_to_plugins_.c_str();
                    pref.pathsToPlugins = plugin_paths_;
                    pref.numPathsToPlugins = 1;
                }
                pref.flags =
                    sl::PreferenceFlags::eDisableCLStateTracking | sl::PreferenceFlags::eUseFrameBasedResourceTagging;
                pref.featuresToLoad = features.data();
                pref.numFeaturesToLoad = static_cast<uint32_t>(features.size());
                pref.applicationId = request_.application_id;
                pref.engine = sl::EngineType::eCustom;
                pref.engineVersion = "1.0";
                pref.projectId = "5b6c3f0e-2d2e-4c55-9a79-7d6c2f6a1c11";
                pref.renderAPI = api_ == api::vulkan ? sl::RenderAPI::eVulkan : sl::RenderAPI::eD3D12;
                const sl::Result r = sl_.init(pref, sl::kSDKVersion);
                if (r != sl::Result::eOk) {
                    log += "streamline: slInit failed (" + std::to_string(static_cast<int>(r)) + "); ";
                    return false;
                }
                initialised_ = true;

                if (api_ == api::d3d12) {
                    hooks.d3d12_create_device =
                        reinterpret_cast<void*>(GetProcAddress(sl_.module, "D3D12CreateDevice"));
                    hooks.dxgi_create_factory =
                        reinterpret_cast<void*>(GetProcAddress(sl_.module, "CreateDXGIFactory2"));
                    if (!hooks.d3d12_create_device || !hooks.dxgi_create_factory) {
                        log += "streamline: the interposer does not export the d3d entry points; ";
                        return false;
                    }
                } else {
                    hooks.vk_get_instance_proc_addr =
                        reinterpret_cast<void*>(GetProcAddress(sl_.module, "vkGetInstanceProcAddr"));
                    if (!hooks.vk_get_instance_proc_addr) {
                        log += "streamline: the interposer does not export vkGetInstanceProcAddr; ";
                        return false;
                    }
                }
                return true;
            }

            bool attach(conduit::device& device) override {
                if (!initialised_)
                    return false;
                this->device_ = &device;
                const native_handles native = device.native();
                if (api_ == api::d3d12 && sl_.set_d3d_device(native.device) != sl::Result::eOk) {
                    caps_.log += "streamline: slSetD3DDevice failed; ";
                    return false;
                }

                sl::AdapterInfo adapter{};
                uint64_t luid = device.caps().adapter.luid;
                adapter.deviceLUID = reinterpret_cast<uint8_t*>(&luid);
                adapter.deviceLUIDSizeInBytes = sizeof(luid);
                adapter.vkPhysicalDevice = native.physical_device;
                auto supported = [&](sl::Feature f) { return sl_.is_feature_supported(f, adapter) == sl::Result::eOk; };
                caps_.dlss = request_.upscaler && supported(sl::kFeatureDLSS);
                caps_.dlss_frame_generation = request_.frame_generation && supported(sl::kFeatureDLSS_G);
                caps_.reflex = request_.low_latency && supported(sl::kFeatureReflex);

                if (caps_.dlss)
                    resolve(sl::kFeatureDLSS, "slDLSSSetOptions", set_dlss_options_);
                if (caps_.dlss_frame_generation) {
                    resolve(sl::kFeatureDLSS_G, "slDLSSGSetOptions", set_dlssg_options_);
                    resolve(sl::kFeatureDLSS_G, "slDLSSGGetState", get_dlssg_state_);
                    caps_.dlss_frame_generation = set_dlssg_options_ && get_dlssg_state_;
                }
                if (caps_.reflex) {
                    resolve(sl::kFeatureReflex, "slReflexSetOptions", reflex_set_options_);
                    resolve(sl::kFeatureReflex, "slReflexSleep", reflex_sleep_);
                    resolve(sl::kFeaturePCL, "slPCLSetMarker", pcl_set_marker_);
                    caps_.reflex = reflex_set_options_ && reflex_sleep_ && pcl_set_marker_;
                }
                attached_ = true;
                return caps_.dlss || caps_.dlss_frame_generation || caps_.reflex;
            }

            capabilities caps() const override { return caps_; }

            void begin_frame(uint64_t frame) override {
                if (!attached_)
                    return;
                const uint32_t index = static_cast<uint32_t>(frame);
                sl::FrameToken* token = nullptr;
                if (sl_.get_new_frame_token(token, &index) == sl::Result::eOk)
                    current_ = token;
            }

            std::unique_ptr<upscaler> create_upscaler(const upscale_desc& desc) override {
                if (!caps_.dlss)
                    return nullptr;
                return std::make_unique<dlss_upscaler>(*this, desc);
            }
            frame_generator* frame_gen() override { return caps_.dlss_frame_generation ? this : nullptr; }
            low_latency* latency() override { return caps_.reflex ? this : nullptr; }

            void shutdown() override {
                if (!initialised_)
                    return;
                initialised_ = false;
                attached_ = false;
                if (caps_.dlss)
                    sl_.free_resources(sl::kFeatureDLSS, sl::ViewportHandle(viewport));
                if (caps_.dlss_frame_generation)
                    sl_.free_resources(sl::kFeatureDLSS_G, sl::ViewportHandle(viewport));
                sl_.shutdown();
                caps_ = {};
            }

            bool set_multiplier(uint32_t multiplier, uint32_t back_buffers, uint32_t width, uint32_t height) override {
                if (!caps_.dlss_frame_generation)
                    return false;
                sl::DLSSGOptions o{};
                o.mode = multiplier > 1 ? sl::DLSSGMode::eOn : sl::DLSSGMode::eOff;
                o.numFramesToGenerate = multiplier > 1 ? multiplier - 1 : 1;
                o.numBackBuffers = back_buffers;
                o.colorWidth = width;
                o.colorHeight = height;
                return set_dlssg_options_(sl::ViewportHandle(viewport), o) == sl::Result::eOk;
            }

            bool tag(command_list& list, const hudless_inputs& in) override {
                if (!caps_.dlss_frame_generation || !current_ || !device_)
                    return false;
                sl::Constants c =
                    constants_for(in.camera, in.jitter_x, in.jitter_y, in.motion_scale_x, in.motion_scale_y, in.reset);
                const sl::ViewportHandle vp(viewport);
                if (sl_.set_constants(c, *current_, vp) != sl::Result::eOk)
                    return false;
                sl::Resource depth, motion, hudless;
                if (!make_resource(in.depth, in.depth_state, depth) ||
                    !make_resource(in.motion, in.motion_state, motion) ||
                    !make_resource(in.hudless, in.hudless_state, hudless))
                    return false;
                sl::ResourceTag tags[] = {
                    {&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent},
                    {&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent},
                    {&hudless, sl::kBufferTypeHUDLessColor, sl::ResourceLifecycle::eValidUntilPresent},
                };
                const bool ok = sl_.set_tag_for_frame(*current_, vp, tags, 3, list.native_handle()) == sl::Result::eOk;
                list.invalidate_bound_state();
                return ok;
            }

            frame_gen_state state() override {
                frame_gen_state s;
                if (!caps_.dlss_frame_generation)
                    return s;
                sl::DLSSGState st{};
                if (get_dlssg_state_(sl::ViewportHandle(viewport), st, nullptr) != sl::Result::eOk)
                    return s;
                s.active = st.status == sl::DLSSGStatus::eOk;
                s.max_multiplier = st.numFramesToGenerateMax + 1;
                s.presented_per_rendered_frame = (std::max)(1u, st.numFramesActuallyPresented);
                s.status = s.active ? "ok" : "status " + std::to_string(static_cast<uint32_t>(st.status));
                return s;
            }

            bool set_mode(bool enabled, bool boost, uint32_t frame_limit_us) override {
                if (!caps_.reflex)
                    return false;
                sl::ReflexOptions o{};
                o.mode = enabled ? (boost ? sl::ReflexMode::eLowLatencyWithBoost : sl::ReflexMode::eLowLatency)
                                 : sl::ReflexMode::eOff;
                o.frameLimitUs = frame_limit_us;
                return reflex_set_options_(o) == sl::Result::eOk;
            }
            void sleep() override {
                if (caps_.reflex && current_)
                    reflex_sleep_(*current_);
            }
            void mark(latency_marker marker) override {
                if (!caps_.reflex || !current_)
                    return;
                sl::PCLMarker m = sl::PCLMarker::eSimulationStart;
                switch (marker) {
                case latency_marker::simulation_start:
                    m = sl::PCLMarker::eSimulationStart;
                    break;
                case latency_marker::simulation_end:
                    m = sl::PCLMarker::eSimulationEnd;
                    break;
                case latency_marker::render_submit_start:
                    m = sl::PCLMarker::eRenderSubmitStart;
                    break;
                case latency_marker::render_submit_end:
                    m = sl::PCLMarker::eRenderSubmitEnd;
                    break;
                case latency_marker::present_start:
                    m = sl::PCLMarker::ePresentStart;
                    break;
                case latency_marker::present_end:
                    m = sl::PCLMarker::ePresentEnd;
                    break;
                }
                pcl_set_marker_(m, *current_);
            }

            bool make_resource(texture_handle h, conduit::state state, sl::Resource& out) {
                const native_texture n = device_->native_of(h, state);
                if (!n.resource)
                    return false;
                out = sl::Resource(sl::ResourceType::eTex2d, n.resource, n.memory, n.view, n.state);
                out.width = n.width;
                out.height = n.height;
                out.nativeFormat = n.native_format;
                out.mipLevels = n.mips;
                out.arrayLayers = n.layers;
                out.usage = n.usage;
                out.flags = n.flags;
                return true;
            }

            sl::Constants constants_for(const camera_data& cam, float jitter_x, float jitter_y, float mv_scale_x,
                                        float mv_scale_y, bool reset) {
                sl::Constants c{};
                to_sl(cam.view_to_clip, c.cameraViewToClip);
                to_sl(cam.clip_to_view, c.clipToCameraView);
                to_sl(cam.clip_to_prev_clip, c.clipToPrevClip);
                to_sl(cam.prev_clip_to_clip, c.prevClipToClip);
                to_sl(mat4{}, c.clipToLensClip);
                c.jitterOffset = {jitter_x, jitter_y};
                c.mvecScale = {mv_scale_x, mv_scale_y};
                c.cameraPinholeOffset = {0.0f, 0.0f};
                c.cameraPos = {cam.position[0], cam.position[1], cam.position[2]};
                c.cameraUp = {cam.up[0], cam.up[1], cam.up[2]};
                c.cameraRight = {cam.right[0], cam.right[1], cam.right[2]};
                c.cameraFwd = {cam.forward[0], cam.forward[1], cam.forward[2]};
                c.cameraNear = cam.near_plane;
                c.cameraFar = cam.far_plane;
                c.cameraFOV = cam.fov_y;
                c.cameraAspectRatio = cam.aspect;
                c.motionVectorsInvalidValue = std::numeric_limits<float>::infinity();
                c.depthInverted = b(cam.reverse_z);
                c.cameraMotionIncluded = sl::Boolean::eTrue;
                c.motionVectors3D = sl::Boolean::eFalse;
                c.reset = b(reset);
                c.orthographicProjection = sl::Boolean::eFalse;
                c.motionVectorsDilated = sl::Boolean::eFalse;
                c.motionVectorsJittered = sl::Boolean::eFalse;
                return c;
            }

        private:
            friend class dlss_upscaler;

            sl_api sl_;
            device* device_ = nullptr;
            sl::FrameToken* current_ = nullptr;
            PFun_slDLSSSetOptions* set_dlss_options_ = nullptr;
            capabilities caps_;

            template <class F> void resolve(sl::Feature feature, const char* name, F*& out) {
                void* fn = nullptr;
                if (sl_.get_feature_function(feature, name, fn) == sl::Result::eOk)
                    out = reinterpret_cast<F*>(fn);
            }

            api api_ = api::none;
            session_request request_;
            std::wstring path_to_plugins_;
            const wchar_t* plugin_paths_[1] = {};
            bool initialised_ = false;
            bool attached_ = false;
            PFun_slDLSSGSetOptions* set_dlssg_options_ = nullptr;
            PFun_slDLSSGGetState* get_dlssg_state_ = nullptr;
            PFun_slReflexSetOptions* reflex_set_options_ = nullptr;
            PFun_slReflexSleep* reflex_sleep_ = nullptr;
            PFun_slPCLSetMarker* pcl_set_marker_ = nullptr;
        };

        dlss_upscaler::~dlss_upscaler() {}

        bool dlss_upscaler::evaluate(command_list& list, const upscale_inputs& in) {
            if (!owner_.current_ || !owner_.device_ || !owner_.set_dlss_options_)
                return false;
            const sl::ViewportHandle vp(viewport);

            if (options_dirty_) {
                sl::DLSSOptions o{};
                switch (desc_.quality) {
                case quality::native:
                    o.mode = sl::DLSSMode::eDLAA;
                    break;
                case quality::ultra_quality:
                    o.mode = sl::DLSSMode::eUltraQuality;
                    break;
                case quality::quality:
                    o.mode = sl::DLSSMode::eMaxQuality;
                    break;
                case quality::balanced:
                    o.mode = sl::DLSSMode::eBalanced;
                    break;
                case quality::performance:
                    o.mode = sl::DLSSMode::eMaxPerformance;
                    break;
                case quality::ultra_performance:
                    o.mode = sl::DLSSMode::eUltraPerformance;
                    break;
                }
                o.outputWidth = desc_.output_width;
                o.outputHeight = desc_.output_height;
                o.colorBuffersHDR = b(desc_.hdr);
                o.useAutoExposure = b(desc_.auto_exposure);
                o.preExposure = in.pre_exposure;
                if (owner_.set_dlss_options_(vp, o) != sl::Result::eOk)
                    return false;
                options_dirty_ = false;
            }

            sl::Constants c = owner_.constants_for(in.camera, in.jitter_x, in.jitter_y, in.motion_scale_x,
                                                   in.motion_scale_y, in.reset);
            if (owner_.sl_.set_constants(c, *owner_.current_, vp) != sl::Result::eOk)
                return false;

            sl::Resource color, depth, motion, output, exposure;
            if (!owner_.make_resource(in.color, in.color_state, color) ||
                !owner_.make_resource(in.depth, in.depth_state, depth) ||
                !owner_.make_resource(in.motion, in.motion_state, motion) ||
                !owner_.make_resource(in.output, in.output_state, output))
                return false;
            const sl::Extent render_extent{0, 0, desc_.render_width, desc_.render_height};
            const sl::Extent output_extent{0, 0, desc_.output_width, desc_.output_height};

            sl::ResourceTag tags[5] = {
                {&color, sl::kBufferTypeScalingInputColor, sl::ResourceLifecycle::eValidUntilEvaluate, &render_extent},
                {&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilEvaluate, &render_extent},
                {&motion, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilEvaluate, &render_extent},
                {&output, sl::kBufferTypeScalingOutputColor, sl::ResourceLifecycle::eValidUntilEvaluate,
                 &output_extent},
                {&exposure, sl::kBufferTypeExposure, sl::ResourceLifecycle::eValidUntilEvaluate},
            };
            uint32_t tag_count = 4;
            if (in.exposure && owner_.make_resource(in.exposure, in.exposure_state, exposure))
                tag_count = 5;

            void* native = list.native_handle();
            bool ok = owner_.sl_.set_tag_for_frame(*owner_.current_, vp, tags, tag_count, native) == sl::Result::eOk;
            if (ok) {
                const sl::BaseStructure* inputs[] = {&vp};
                ok = owner_.sl_.evaluate_feature(sl::kFeatureDLSS, *owner_.current_, inputs, 1, native) ==
                     sl::Result::eOk;
            }
            list.invalidate_bound_state();
            return ok;
        }
    }

    std::unique_ptr<vendor::session> make_streamline_session(conduit::api api, const session_request& request,
                                                             vendor_hooks& hooks, std::string* log) {
        auto session = std::make_unique<streamline>();
        std::string text;
        if (!session->init(api, request, hooks, text)) {
            if (log)
                *log += text;
            return nullptr;
        }
        if (log)
            *log += text;
        return session;
    }
}

#endif
