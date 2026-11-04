#include "d3d12/d3d12_device.hpp"

#include "d3d12/d3d12_convert.hpp"

#include <algorithm>

namespace conduit::d3d12 {
    namespace {
        class swapchain_impl final : public conduit::swapchain {
        public:
            swapchain_impl(d3d12::device& device, const swapchain_desc& desc) : device_(device), desc_(desc) {}

            ~swapchain_impl() override {
                device_.wait_idle();
                release_buffers();
            }

            bool init() {
                hwnd_ = static_cast<HWND>(desc_.window.window);
                if (!hwnd_)
                    return false;
                width_ = desc_.width;
                height_ = desc_.height;
                present_mode_ = desc_.present;
                image_count_ = std::clamp(desc_.image_count, 2u, 8u);
                tearing_ = device_.instance_info().tearing;
                hdr_ = false;

                DXGI_SWAP_CHAIN_DESC1 sd{};
                sd.Width = (std::max)(1u, width_);
                sd.Height = (std::max)(1u, height_);
                sd.Format = desc_.hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_B8G8R8A8_UNORM;
                sd.SampleDesc.Count = 1;
                sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT | (desc_.allow_sampling ? DXGI_USAGE_SHADER_INPUT : 0);
                sd.BufferCount = image_count_;
                sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
                sd.Scaling = DXGI_SCALING_STRETCH;
                sd.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
                sd.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT |
                           (tearing_ ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0);
                flags_ = sd.Flags;

                winrt::com_ptr<IDXGISwapChain1> sc1;
                if (FAILED(device_.instance_info().factory->CreateSwapChainForHwnd(
                        device_.direct_queue().queue.get(), hwnd_, &sd, nullptr, nullptr, sc1.put()))) {
                    if (!desc_.hdr)
                        return false;
                    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
                    if (FAILED(device_.instance_info().factory->CreateSwapChainForHwnd(
                            device_.direct_queue().queue.get(), hwnd_, &sd, nullptr, nullptr, sc1.put())))
                        return false;
                }
                format_ = sd.Format;
                chain_ = sc1.try_as<IDXGISwapChain3>();
                if (!chain_)
                    return false;
                device_.instance_info().factory->MakeWindowAssociation(hwnd_, DXGI_MWA_NO_ALT_ENTER);
                chain_->SetMaximumFrameLatency(std::min(image_count_ - 1, (std::max)(1u, device_.frames_in_flight())));
                waitable_.attach(chain_->GetFrameLatencyWaitableObject());

                if (format_ == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                    UINT support = 0;
                    if (SUCCEEDED(chain_->CheckColorSpaceSupport(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709, &support)) &&
                        (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)) {
                        hdr_ = SUCCEEDED(chain_->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709));
                    }
                }
                return wrap_buffers();
            }

            bool acquire(image& out) override {
                if (width_ == 0 || height_ == 0 || IsIconic(hwnd_))
                    return false;
                if (pending_resize_ && !do_resize())
                    return false;
                if (!acquired_ && waitable_)
                    WaitForSingleObjectEx(waitable_.get(), 1000, TRUE);
                acquired_ = true;
                const UINT index = chain_->GetCurrentBackBufferIndex();
                out.index = index;
                out.texture = textures_[index];
                return true;
            }

            void present() override {
                if (!acquired_)
                    return;
                acquired_ = false;
                UINT interval = 1;
                UINT flags_out = 0;
                if (present_mode_ != present_mode::fifo) {
                    interval = 0;
                    if (present_mode_ == present_mode::immediate && tearing_)
                        flags_out = DXGI_PRESENT_ALLOW_TEARING;
                }
                HRESULT hr;
                {
                    std::lock_guard lock(device_.direct_queue().mutex);
                    hr = chain_->Present(interval, flags_out);
                }
                device_.check_removed(hr, "Present");
                if (FAILED(hr) && hr != DXGI_ERROR_DEVICE_REMOVED && hr != DXGI_ERROR_DEVICE_RESET)
                    pending_resize_ = true;
            }

            bool resize(uint32_t w, uint32_t h) override {
                if (w == 0 || h == 0) {
                    width_ = w;
                    height_ = h;
                    return false;
                }
                if (w == width_ && h == height_ && !pending_resize_)
                    return true;
                width_ = w;
                height_ = h;
                pending_resize_ = true;
                return do_resize();
            }

            void set_present_mode(present_mode mode) override { present_mode_ = mode; }
            format get_format() const override {
                return format_ == DXGI_FORMAT_R16G16B16A16_FLOAT ? format::rgba16_float : format::bgra8_unorm;
            }
            uint32_t width() const override { return width_; }
            uint32_t height() const override { return height_; }
            present_mode get_present_mode() const override { return present_mode_; }
            uint32_t image_count() const override { return image_count_; }
            bool is_hdr() const override { return hdr_; }

        private:
            static std::string hex_string(uint32_t v) {
                static const char* digits = "0123456789abcdef";
                std::string out(8, '0');
                for (int i = 7; i >= 0; i--, v >>= 4)
                    out[i] = digits[v & 15];
                return out;
            }

            bool do_resize() {
                // buffers may still be in flight and every reference must be gone before ResizeBuffers
                device_.wait_idle();
                release_buffers();
                const HRESULT hr = chain_->ResizeBuffers(0, width_, height_, DXGI_FORMAT_UNKNOWN, flags_);
                if (FAILED(hr)) {
                    device_.report_error("swapchain ResizeBuffers failed (0x" + hex_string(static_cast<uint32_t>(hr)) +
                                         "); building a new swapchain");
                    chain_ = nullptr;
                    waitable_.close();
                    desc_.width = width_;
                    desc_.height = height_;
                    acquired_ = false;
                    pending_resize_ = !init();
                    return !pending_resize_;
                }
                pending_resize_ = false;
                acquired_ = false;
                return wrap_buffers();
            }

            bool wrap_buffers() {
                textures_.resize(image_count_);
                for (uint32_t i = 0; i < image_count_; i++) {
                    winrt::com_ptr<ID3D12Resource> res;
                    if (FAILED(chain_->GetBuffer(i, winrt::guid_of<ID3D12Resource>(), res.put_void())))
                        return false;
                    texture_desc td;
                    td.width = width_;
                    td.height = height_;
                    td.format = get_format();
                    td.usage = texture_usage::color_target |
                               (desc_.allow_sampling ? texture_usage::sampled : texture_usage::none) |
                               texture_usage::copy_dst;
                    textures_[i] = device_.wrap_swapchain_image(res.get(), td);
                }
                return true;
            }

            void release_buffers() {
                for (texture_handle& t : textures_)
                    if (t)
                        device_.destroy_wrapped(t);
                textures_.clear();
            }

            device& device_;
            swapchain_desc desc_;
            HWND hwnd_ = nullptr;
            winrt::com_ptr<IDXGISwapChain3> chain_;
            winrt::handle waitable_;
            std::vector<texture_handle> textures_;
            DXGI_FORMAT format_ = DXGI_FORMAT_B8G8R8A8_UNORM;
            UINT flags_ = 0;
            uint32_t width_ = 0, height_ = 0, image_count_ = 3;
            present_mode present_mode_ = present_mode::fifo;
            bool tearing_ = false, hdr_ = false, acquired_ = false, pending_resize_ = false;
        };
    }

    std::unique_ptr<conduit::swapchain> make_swapchain(d3d12::device& device, const swapchain_desc& desc) {
        auto sc = std::make_unique<swapchain_impl>(device, desc);
        if (!sc->init())
            return nullptr;
        return sc;
    }
}
