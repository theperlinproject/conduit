#include "backends.hpp"
#include "d3d11/d3d11_internal.hpp"

#include <algorithm>
#include <cstdio>

namespace conduit::d3d11 {
    namespace {
        std::string narrow(const wchar_t* w) {
            std::string out;
            for (; *w; w++)
                out.push_back(*w < 0x80 ? static_cast<char>(*w) : '?');
            return out;
        }

        class swapchain_impl final : public swapchain {
        public:
            swapchain_impl(d3d11::device& device, const swapchain_desc& desc)
                : device_(device), desc_(desc), width_(desc.width), height_(desc.height), present_mode_(desc.present),
                  hdr_(desc.hdr) {}

            ~swapchain_impl() override {
                device_.wait_idle();
                if (handle_)
                    device_.release_backbuffer(handle_);
                chain_ = nullptr;
            }

            bool init() {
                hwnd_ = static_cast<HWND>(desc_.window.window);
                if (!hwnd_)
                    return false;
                if (!width_ || !height_) {
                    RECT rc{};
                    GetClientRect(hwnd_, &rc);
                    width_ = static_cast<uint32_t>(std::max<LONG>(rc.right - rc.left, 1));
                    height_ = static_cast<uint32_t>(std::max<LONG>(rc.bottom - rc.top, 1));
                }
                format_ = hdr_ ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
                const DXGI_SWAP_EFFECT effects[] = {DXGI_SWAP_EFFECT_FLIP_DISCARD, DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
                                                    DXGI_SWAP_EFFECT_DISCARD};
                for (const DXGI_SWAP_EFFECT effect : effects) {
                    const bool flip = effect != DXGI_SWAP_EFFECT_DISCARD;
                    DXGI_SWAP_CHAIN_DESC1 sd{};
                    sd.Width = width_;
                    sd.Height = height_;
                    sd.Format = format_;
                    sd.SampleDesc.Count = 1;
                    sd.BufferUsage =
                        DXGI_USAGE_RENDER_TARGET_OUTPUT | (desc_.allow_sampling ? DXGI_USAGE_SHADER_INPUT : 0);
                    sd.BufferCount = flip ? std::clamp(desc_.image_count, 2u, 4u) : 1;
                    sd.SwapEffect = effect;
                    sd.Scaling = DXGI_SCALING_STRETCH;
                    tearing_ = flip && device_.inst().tearing;
                    flags_ = tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
                    sd.Flags = flags_;
                    if (SUCCEEDED(device_.inst().factory->CreateSwapChainForHwnd(device_.dev(), hwnd_, &sd, nullptr,
                                                                                 nullptr, chain_.put())))
                        break;
                    chain_ = nullptr;
                }
                if (!chain_)
                    return false;
                device_.inst().factory->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER);
                if (hdr_)
                    if (auto c3 = chain_.try_as<IDXGISwapChain3>())
                        c3->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709);
                return wrap();
            }

            bool acquire(image& out) override {
                if (!width_ || !height_ || !handle_)
                    return false;
                out.texture = handle_;
                out.index = 0;
                return true;
            }

            void present() override {
                const UINT interval = present_mode_ == present_mode::fifo ? 1 : 0;
                const UINT present_flags = interval == 0 && tearing_ && present_mode_ == present_mode::immediate
                                               ? DXGI_PRESENT_ALLOW_TEARING
                                               : 0;
                auto lock = device_.lock_context();
                chain_->Present(interval, present_flags);
            }

            bool resize(uint32_t w, uint32_t h) override {
                if (w == 0 || h == 0) {
                    width_ = w;
                    height_ = h;
                    return false;
                }
                if (w == width_ && h == height_ && handle_)
                    return true;
                width_ = w;
                height_ = h;
                device_.wait_idle();
                {
                    auto lock = device_.lock_context();
                    device_.imm()->ClearState();
                    device_.imm()->Flush();
                }
                if (handle_) {
                    device_.release_backbuffer(handle_);
                    handle_ = {};
                }
                if (FAILED(chain_->ResizeBuffers(0, width_, height_, DXGI_FORMAT_UNKNOWN, flags_)))
                    return false;
                return wrap();
            }

            void set_present_mode(present_mode mode) override { present_mode_ = mode; }
            format get_format() const override { return hdr_ ? format::rgba16_float : format::bgra8_unorm; }
            uint32_t width() const override { return width_; }
            uint32_t height() const override { return height_; }
            present_mode get_present_mode() const override { return present_mode_; }
            uint32_t image_count() const override { return 2; }
            bool is_hdr() const override { return hdr_; }

        private:
            bool wrap() {
                winrt::com_ptr<ID3D11Texture2D> buffer;
                if (FAILED(chain_->GetBuffer(0, winrt::guid_of<ID3D11Texture2D>(), buffer.put_void())))
                    return false;
                texture_desc td;
                td.width = width_;
                td.height = height_;
                td.format = get_format();
                td.usage = texture_usage::color_target |
                           (desc_.allow_sampling ? texture_usage::sampled : texture_usage::none) |
                           texture_usage::copy_src;
                handle_ = device_.wrap_backbuffer(buffer.get(), td);
                return static_cast<bool>(handle_);
            }

            device& device_;
            swapchain_desc desc_;
            HWND hwnd_ = nullptr;
            winrt::com_ptr<IDXGISwapChain1> chain_;
            texture_handle handle_;
            DXGI_FORMAT format_ = DXGI_FORMAT_B8G8R8A8_UNORM;
            UINT flags_ = 0;
            uint32_t width_ = 0, height_ = 0;
            present_mode present_mode_ = present_mode::fifo;
            bool tearing_ = false, hdr_ = false;
        };

        class d3d11_instance final : public instance {
        public:
            explicit d3d11_instance(const instance_options& options) : options_(options) {}

            bool init() {
                state_ = std::make_shared<instance_state>();
                library& lib = state_->lib;
                lib.d3d11 = LoadLibraryW(L"d3d11.dll");
                lib.dxgi = LoadLibraryW(L"dxgi.dll");
                if (!lib.d3d11 || !lib.dxgi)
                    return false;
                lib.create_device =
                    function_cast<PFN_D3D11_CREATE_DEVICE>(GetProcAddress(lib.d3d11, "D3D11CreateDevice"));
                lib.create_factory1 =
                    function_cast<decltype(lib.create_factory1)>(GetProcAddress(lib.dxgi, "CreateDXGIFactory1"));
                if (!lib.create_device || !lib.create_factory1)
                    return false;
                lib.compiler = LoadLibraryW(L"d3dcompiler_47.dll");
                if (lib.compiler)
                    lib.compile = function_cast<compile_fn>(GetProcAddress(lib.compiler, "D3DCompile"));
                state_->validation = options_.validation;
                if (FAILED(lib.create_factory1(winrt::guid_of<IDXGIFactory2>(), state_->factory.put_void())))
                    return false;

                BOOL tearing = FALSE;
                if (auto f5 = state_->factory.try_as<IDXGIFactory5>())
                    if (SUCCEEDED(
                            f5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
                        state_->tearing = tearing != FALSE;
                enumerate_adapters();
                return true;
            }

            api get_api() const override { return api::d3d11; }
            const std::vector<adapter_info>& adapters() const override { return adapters_; }

            std::unique_ptr<conduit::device> create_device(const conduit::device_options& device_options) override {
                if (device_options.adapter_index >= adapters_.size() || !adapters_[device_options.adapter_index].usable)
                    return nullptr;
                auto device = std::make_unique<d3d11::device>(state_, native_[device_options.adapter_index],
                                                              adapters_[device_options.adapter_index], device_options);
                std::string error;
                if (!device->init(error)) {
                    std::fprintf(stderr, "[d3d11] device creation failed: %s\n", error.c_str());
                    return nullptr;
                }
                return device;
            }

        private:
            void consider(winrt::com_ptr<IDXGIAdapter1> adapter) {
                DXGI_ADAPTER_DESC1 d{};
                adapter->GetDesc1(&d);
                adapter_info a;
                a.api = api::d3d11;
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

                const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
                D3D_FEATURE_LEVEL got{};
                HRESULT hr = state_->lib.create_device(adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                                       D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels, 2, D3D11_SDK_VERSION,
                                                       nullptr, &got, nullptr);
                if (FAILED(hr))
                    hr = state_->lib.create_device(adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                                                   D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels + 1, 1, D3D11_SDK_VERSION,
                                                   nullptr, &got, nullptr);
                if (FAILED(hr)) {
                    a.usable = false;
                    a.unusable_reason = "no Direct3D 11 feature level 11_0";
                } else {
                    a.api_version = got;
                    if (!state_->lib.compile) {
                        a.usable = false;
                        a.unusable_reason = "d3dcompiler_47.dll is missing";
                    }
                }
                adapters_.push_back(std::move(a));
                native_.push_back(std::move(adapter));
            }

            void enumerate_adapters() {
                winrt::com_ptr<IDXGIAdapter1> warp;
                const auto handle = [&](const winrt::com_ptr<IDXGIAdapter1>& a) {
                    DXGI_ADAPTER_DESC1 d{};
                    a->GetDesc1(&d);
                    if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
                        if (!warp)
                            warp = a;
                        return;
                    }
                    consider(a);
                };
                if (auto f6 = state_->factory.try_as<IDXGIFactory6>()) {
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

    std::unique_ptr<conduit::swapchain> make_swapchain(d3d11::device& device, const swapchain_desc& desc) {
        auto sc = std::make_unique<swapchain_impl>(device, desc);
        if (!sc->init())
            return nullptr;
        return sc;
    }
}

namespace conduit::backends {
    std::unique_ptr<conduit::instance> create_d3d11_instance(const instance_options& options) {
        auto instance = std::make_unique<d3d11::d3d11_instance>(options);
        if (!instance->init())
            return nullptr;
        return instance;
    }
}
