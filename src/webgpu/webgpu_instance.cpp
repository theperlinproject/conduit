#include "webgpu/webgpu_internal.hpp"

#include <chrono>
#include <cstdio>
#include <thread>

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
extern "C" {
WGPUDevice emscripten_webgpu_get_device(void);
}
#else
#include <webgpu/wgpu.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace conduit::webgpu {
    namespace {
        void on_uncaptured_error(WGPUDevice const*, WGPUErrorType type, WGPUStringView message, void*, void*) {
            std::fprintf(stderr, "[webgpu] validation error (%d): %s\n", static_cast<int>(type), str(message).c_str());
        }

        void on_device_lost(WGPUDevice const*, WGPUDeviceLostReason reason, WGPUStringView message, void*, void*) {
            if (reason != WGPUDeviceLostReason_Destroyed)
                std::fprintf(stderr, "[webgpu] device lost (%d): %s\n", static_cast<int>(reason), str(message).c_str());
        }

        adapter_type to_adapter_type(WGPUAdapterType type) {
            switch (type) {
            case WGPUAdapterType_DiscreteGPU:
                return adapter_type::discrete;
            case WGPUAdapterType_IntegratedGPU:
                return adapter_type::integrated;
            case WGPUAdapterType_CPU:
                return adapter_type::cpu;
            default:
                return adapter_type::unknown;
            }
        }

        WGPUPresentMode to_present(present_mode mode) {
            switch (mode) {
            case present_mode::immediate:
                return WGPUPresentMode_Immediate;
            case present_mode::mailbox:
                return WGPUPresentMode_Mailbox;
            default:
                return WGPUPresentMode_Fifo;
            }
        }

        struct request_state {
            bool done = false;
            bool ok = false;
            WGPUDevice device = nullptr;
            std::string message;
        };

        void on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* a, void*) {
            auto* s = static_cast<request_state*>(a);
            s->ok = status == WGPURequestDeviceStatus_Success;
            s->device = device;
            s->message = str(message);
            s->done = true;
        }

        bool wait_for(WGPUInstance instance, const bool& done, uint64_t timeout_ms) {
            const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
            while (!done) {
                wgpuInstanceProcessEvents(instance);
                if (std::chrono::steady_clock::now() > end)
                    return false;
#if !defined(__EMSCRIPTEN__)
                std::this_thread::sleep_for(std::chrono::microseconds(200));
#endif
            }
            return true;
        }

        constexpr WGPUFeatureName wanted_features[] = {
            WGPUFeatureName_DepthClipControl,        WGPUFeatureName_Depth32FloatStencil8,
            WGPUFeatureName_TextureCompressionBC,    WGPUFeatureName_IndirectFirstInstance,
            WGPUFeatureName_RG11B10UfloatRenderable, WGPUFeatureName_Float32Filterable,
            WGPUFeatureName_Float32Blendable,
        };

        class webgpu_swapchain final : public swapchain {
        public:
            webgpu_swapchain(webgpu_device& device, owned<WGPUSurface> surface, const swapchain_desc& desc)
                : device_(device), surface_(std::move(surface)), width_(desc.width), height_(desc.height),
                  present_(desc.present), allow_sampling_(desc.allow_sampling) {}

            ~webgpu_swapchain() override {
                if (handle_)
                    device_.destroy(handle_);
                if (surface_)
                    wgpuSurfaceUnconfigure(surface_.get());
            }

            bool init() {
                WGPUSurfaceCapabilities caps{};
                if (wgpuSurfaceGetCapabilities(surface_.get(), device_.adapter(), &caps) != WGPUStatus_Success ||
                    !caps.formatCount) {
                    std::fprintf(stderr, "[webgpu] surface reports no capabilities\n");
                    return false;
                }
                WGPUTextureFormat chosen = caps.formats[0];
                for (WGPUTextureFormat want : {WGPUTextureFormat_BGRA8Unorm, WGPUTextureFormat_RGBA8Unorm}) {
                    bool found = false;
                    for (size_t i = 0; i < caps.formatCount; i++)
                        found |= caps.formats[i] == want;
                    if (found) {
                        chosen = want;
                        break;
                    }
                }
                usage_ = WGPUTextureUsage_RenderAttachment;
                if (caps.usages & WGPUTextureUsage_CopySrc)
                    usage_ |= WGPUTextureUsage_CopySrc;
                sampled_ = allow_sampling_ && (caps.usages & WGPUTextureUsage_TextureBinding);
                if (sampled_)
                    usage_ |= WGPUTextureUsage_TextureBinding;
                present_modes_.assign(caps.presentModes, caps.presentModes + caps.presentModeCount);
                wgpuSurfaceCapabilitiesFreeMembers(caps);

                native_ = chosen;
                format_ = from_texture_format(chosen);
                if (format_ == format::unknown) {
                    std::fprintf(stderr, "[webgpu] surface format has no rhi equivalent\n");
                    return false;
                }
                handle_ = device_.create_surface_texture(format_, width_, height_, sampled_);
                configure();
                return true;
            }

            bool acquire(image& out) override {
                if (!width_ || !height_)
                    return false;
                if (needs_configure_)
                    configure();
                WGPUSurfaceTexture st{};
                wgpuSurfaceGetCurrentTexture(surface_.get(), &st);
                switch (st.status) {
                case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
                    break;
                case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
                    needs_configure_ = true;
                    break;
                case WGPUSurfaceGetCurrentTextureStatus_Outdated:
                case WGPUSurfaceGetCurrentTextureStatus_Lost:
                    if (st.texture)
                        wgpuTextureRelease(st.texture);
                    needs_configure_ = true;
                    return false;
                default:
                    if (st.texture)
                        wgpuTextureRelease(st.texture);
                    return false;
                }
                device_.set_surface_texture(handle_, st.texture);
                out.texture = handle_;
                out.index = 0;
                acquired_ = true;
                return true;
            }

            void present() override {
                if (!acquired_)
                    return;
                acquired_ = false;
#if !defined(__EMSCRIPTEN__)
                wgpuSurfacePresent(surface_.get());
#endif
                device_.set_surface_texture(handle_, nullptr);
            }

            bool resize(uint32_t w, uint32_t h) override {
                if (w == width_ && h == height_)
                    return true;
                width_ = w;
                height_ = h;
                device_.resize_surface_texture(handle_, w, h);
                needs_configure_ = true;
                return w && h;
            }

            void set_present_mode(present_mode mode) override {
                if (mode == present_)
                    return;
                present_ = mode;
                needs_configure_ = true;
            }

            format get_format() const override { return format_; }
            uint32_t width() const override { return width_; }
            uint32_t height() const override { return height_; }
            present_mode get_present_mode() const override { return effective_present_; }
            uint32_t image_count() const override { return 2; }

        private:
            void configure() {
                needs_configure_ = false;
                if (!width_ || !height_)
                    return;
                WGPUPresentMode mode = to_present(present_);
                bool supported = false;
                for (WGPUPresentMode m : present_modes_)
                    supported |= m == mode;
                if (!supported)
                    mode = WGPUPresentMode_Fifo;
                effective_present_ = mode == WGPUPresentMode_Immediate ? present_mode::immediate
                                     : mode == WGPUPresentMode_Mailbox ? present_mode::mailbox
                                                                       : present_mode::fifo;
                WGPUSurfaceConfiguration cfg{};
                cfg.device = device_.dev();
                cfg.format = native_;
                cfg.usage = usage_;
                cfg.width = width_;
                cfg.height = height_;
                cfg.alphaMode = WGPUCompositeAlphaMode_Auto;
                cfg.presentMode = mode;
                wgpuSurfaceConfigure(surface_.get(), &cfg);
            }

            webgpu_device& device_;
            owned<WGPUSurface> surface_;
            texture_handle handle_;
            uint32_t width_, height_;
            present_mode present_;
            present_mode effective_present_ = present_mode::fifo;
            bool allow_sampling_;
            bool sampled_ = false;
            bool needs_configure_ = false;
            bool acquired_ = false;
            format format_ = format::unknown;
            WGPUTextureFormat native_ = WGPUTextureFormat_Undefined;
            WGPUTextureUsage usage_ = 0;
            std::vector<WGPUPresentMode> present_modes_;
        };

        owned<WGPUSurface> create_surface(WGPUInstance instance, const native_window& w) {
            if (!w.window)
                return {};
            WGPUSurfaceDescriptor sd{};
            sd.label = view("conduit surface");
#if defined(__EMSCRIPTEN__)
            WGPUEmscriptenSurfaceSourceCanvasHTMLSelector src{};
            src.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
            src.selector = view(static_cast<const char*>(w.window));
            sd.nextInChain = &src.chain;
            return owned<WGPUSurface>(wgpuInstanceCreateSurface(instance, &sd));
#elif defined(_WIN32)
            WGPUSurfaceSourceWindowsHWND src{};
            src.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
            src.hinstance = GetModuleHandleW(nullptr);
            src.hwnd = w.window;
            sd.nextInChain = &src.chain;
            return owned<WGPUSurface>(wgpuInstanceCreateSurface(instance, &sd));
#elif defined(__ANDROID__)
            WGPUSurfaceSourceAndroidNativeWindow src{};
            src.chain.sType = WGPUSType_SurfaceSourceAndroidNativeWindow;
            src.window = w.window;
            sd.nextInChain = &src.chain;
            return owned<WGPUSurface>(wgpuInstanceCreateSurface(instance, &sd));
#elif defined(__APPLE__)
            WGPUSurfaceSourceMetalLayer src{};
            src.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
            src.layer = w.window;
            sd.nextInChain = &src.chain;
            return owned<WGPUSurface>(wgpuInstanceCreateSurface(instance, &sd));
#else
            if (w.wayland) {
                WGPUSurfaceSourceWaylandSurface src{};
                src.chain.sType = WGPUSType_SurfaceSourceWaylandSurface;
                src.display = w.display;
                src.surface = w.window;
                sd.nextInChain = &src.chain;
                return owned<WGPUSurface>(wgpuInstanceCreateSurface(instance, &sd));
            }
            WGPUSurfaceSourceXlibWindow src{};
            src.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
            src.display = w.display;
            src.window = reinterpret_cast<uintptr_t>(w.window);
            sd.nextInChain = &src.chain;
            return owned<WGPUSurface>(wgpuInstanceCreateSurface(instance, &sd));
#endif
        }

        class webgpu_instance final : public instance {
        public:
            explicit webgpu_instance(const instance_options& options) : options_(options) {}

            bool init() {
#if defined(__EMSCRIPTEN__)
                instance_ = owned<WGPUInstance>(wgpuCreateInstance(nullptr));
                if (!instance_)
                    return false;
                adapter_info info;
                info.api = api::webgpu;
                info.name = "Browser WebGPU";
                infos_.push_back(info);
                return true;
#else
                WGPUInstanceExtras extras{};
                extras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
                extras.backends = WGPUInstanceBackend_Primary;
                extras.flags = options_.validation ? (WGPUInstanceFlag_Debug | WGPUInstanceFlag_Validation)
                                                   : WGPUInstanceFlag_Default;
                WGPUInstanceDescriptor id{};
                id.nextInChain = &extras.chain;
                instance_ = owned<WGPUInstance>(wgpuCreateInstance(&id));
                if (!instance_)
                    return false;

                WGPUInstanceEnumerateAdapterOptions eo{};
                eo.backends = WGPUInstanceBackend_Primary;
                const size_t count = wgpuInstanceEnumerateAdapters(instance_.get(), &eo, nullptr);
                std::vector<WGPUAdapter> raw(count);
                if (count)
                    wgpuInstanceEnumerateAdapters(instance_.get(), &eo, raw.data());
                for (WGPUAdapter a : raw) {
                    WGPUAdapterInfo ai{};
                    wgpuAdapterGetInfo(a, &ai);
                    adapter_info info;
                    info.api = api::webgpu;
                    info.index = static_cast<uint32_t>(infos_.size());
                    info.name = str(ai.device);
                    if (info.name.empty())
                        info.name = str(ai.description);
                    info.vendor_id = ai.vendorID;
                    info.device_id = ai.deviceID;
                    info.type = to_adapter_type(ai.adapterType);
                    info.driver_name = str(ai.description);
                    wgpuAdapterInfoFreeMembers(ai);
                    adapters_.emplace_back(a);
                    infos_.push_back(std::move(info));
                }
                return !infos_.empty();
#endif
            }

            api get_api() const override { return api::webgpu; }
            const std::vector<adapter_info>& adapters() const override { return infos_; }

            std::unique_ptr<conduit::device> create_device(const device_options& opts) override {
                if (opts.adapter_index >= infos_.size())
                    return nullptr;
#if defined(__EMSCRIPTEN__)
                WGPUDevice raw = emscripten_webgpu_get_device();
                if (!raw)
                    return nullptr;
                wgpuInstanceAddRef(instance_.get());
                auto device =
                    std::make_unique<webgpu_device>(owned<WGPUInstance>(instance_.get()), owned<WGPUAdapter>(),
                                                    owned<WGPUDevice>(raw), opts, infos_[0], false);
#else
                WGPUAdapter adapter = adapters_[opts.adapter_index].get();
                std::vector<WGPUFeatureName> features;
                for (WGPUFeatureName f : wanted_features)
                    if (wgpuAdapterHasFeature(adapter, f))
                        features.push_back(f);
                WGPULimits limits{};
                wgpuAdapterGetLimits(adapter, &limits);

                WGPUDeviceDescriptor dd{};
                dd.label = view("conduit device");
                dd.requiredFeatureCount = features.size();
                dd.requiredFeatures = features.data();
                dd.requiredLimits = &limits;
                dd.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
                dd.deviceLostCallbackInfo.callback = on_device_lost;
                dd.uncapturedErrorCallbackInfo.callback = on_uncaptured_error;

                request_state state;
                WGPURequestDeviceCallbackInfo cb{};
                cb.mode = WGPUCallbackMode_AllowProcessEvents;
                cb.callback = on_device;
                cb.userdata1 = &state;
                wgpuAdapterRequestDevice(adapter, &dd, cb);
                if (!wait_for(instance_.get(), state.done, 15000) || !state.ok || !state.device) {
                    std::fprintf(stderr, "[webgpu] device request failed: %s\n", state.message.c_str());
                    return nullptr;
                }
                wgpuAdapterAddRef(adapter);
                wgpuInstanceAddRef(instance_.get());
                auto device = std::make_unique<webgpu_device>(
                    owned<WGPUInstance>(instance_.get()), owned<WGPUAdapter>(adapter), owned<WGPUDevice>(state.device),
                    opts, infos_[opts.adapter_index], true);
#endif
                if (!device->init())
                    return nullptr;
                return device;
            }

        private:
            instance_options options_;
            owned<WGPUInstance> instance_;
            std::vector<owned<WGPUAdapter>> adapters_;
            std::vector<adapter_info> infos_;
        };
    }

    std::unique_ptr<conduit::instance> make_instance(const instance_options& options) {
        auto instance = std::make_unique<webgpu_instance>(options);
        if (!instance->init())
            return nullptr;
        return instance;
    }

    std::unique_ptr<conduit::swapchain> webgpu_device::create_swapchain(const swapchain_desc& desc) {
        owned<WGPUSurface> surface = create_surface(inst(), desc.window);
        if (!surface) {
            report_error("CreateSwapchain: could not create a surface for this window");
            return nullptr;
        }
        auto swapchain = std::make_unique<webgpu_swapchain>(*this, std::move(surface), desc);
        if (!swapchain->init())
            return nullptr;
        return swapchain;
    }
}

namespace conduit::backends {
    std::unique_ptr<instance> create_webgpu_instance(const instance_options& options) {
        return webgpu::make_instance(options);
    }
}
