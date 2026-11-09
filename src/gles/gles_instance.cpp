#include "backends.hpp"
#include "gles/gles_internal.hpp"
#include "gles/gles_platform.hpp"

#include <algorithm>
#include <cctype>

namespace conduit::gles {
    namespace {
        // everything is drawn into ordinary textures and one of them is blitted to the window when presenting
        class gles_swapchain final : public swapchain {
        public:
            gles_swapchain(device& dev, const swapchain_desc& desc) : device_(dev), desc_(desc), mode_(desc.present) {}

            ~gles_swapchain() override {
                destroy_images();
                device_.thread().call([this] { surface_.reset(); });
            }

            bool init() {
                bool ok = false;
                device_.thread().call([&] {
                    surface_ = device_.context().create_surface(desc_.window, desc_.width, desc_.height);
                    if (surface_) {
                        uint32_t w = 0, h = 0;
                        surface_->extent(w, h);
                        width_ = desc_.width ? desc_.width : w;
                        height_ = desc_.height ? desc_.height : h;
                        ok = true;
                    }
                });
                if (!ok) {
                    device_.report_error("could not create the OpenGL ES window surface");
                    return false;
                }
                return create_images();
            }

            bool acquire(image& out) override {
                if (images_.empty() || !width_ || !height_)
                    return false;
                current_ = (current_ + 1) % static_cast<uint32_t>(images_.size());
                out.texture = images_[current_];
                out.index = current_;
                return true;
            }

            void present() override {
                if (images_.empty())
                    return;
                device_.thread().post(
                    [this, texture = images_[current_], mode = mode_] { device_.present_texture(texture, *surface_, mode); });
            }

            bool resize(uint32_t width, uint32_t height) override {
                if (!width || !height)
                    return false;
                if (width == width_ && height == height_)
                    return true;
                destroy_images();
                width_ = width;
                height_ = height;
                device_.thread().call([&] { surface_->resize(width, height); });
                return create_images();
            }

            void set_present_mode(present_mode mode) override { mode_ = mode; }
            format get_format() const override { return format::rgba8_unorm; }
            uint32_t width() const override { return width_; }
            uint32_t height() const override { return height_; }
            present_mode get_present_mode() const override { return mode_; }
            uint32_t image_count() const override { return static_cast<uint32_t>(images_.size()); }

        private:
            bool create_images() {
                const uint32_t count = std::clamp(desc_.image_count, 2u, 4u);
                for (uint32_t i = 0; i < count; i++) {
                    texture_desc t;
                    t.format = format::rgba8_unorm;
                    t.width = width_;
                    t.height = height_;
                    t.usage = texture_usage::color_target | texture_usage::copy_src | texture_usage::copy_dst |
                              (desc_.allow_sampling ? texture_usage::sampled : texture_usage::none);
                    t.name = "swapchain image";
                    const texture_handle h = device_.create_texture(t);
                    if (!h) {
                        destroy_images();
                        return false;
                    }
                    images_.push_back(h);
                }
                current_ = static_cast<uint32_t>(images_.size()) - 1;
                return true;
            }

            void destroy_images() {
                for (const texture_handle h : images_)
                    device_.destroy(h);
                images_.clear();
            }

            device& device_;
            swapchain_desc desc_;
            std::unique_ptr<gl_surface> surface_;
            std::vector<texture_handle> images_;
            uint32_t current_ = 0;
            uint32_t width_ = 0, height_ = 0;
            present_mode mode_;
        };

        uint32_t vendor_from(const std::string& vendor, const std::string& renderer) {
            std::string text = vendor + " " + renderer;
            std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
            const auto has = [&](const char* s) { return text.find(s) != std::string::npos; };
            if (has("nvidia"))
                return vendor_id::nvidia;
            if (has("amd") || has("ati ") || has("radeon"))
                return vendor_id::amd;
            if (has("intel"))
                return vendor_id::intel;
            if (has("apple"))
                return vendor_id::apple;
            if (has("qualcomm") || has("adreno"))
                return 0x5143;
            if (has("arm") || has("mali"))
                return 0x13B5;
            if (has("imagination") || has("powervr"))
                return 0x1010;
            return 0;
        }

        class gles_instance final : public instance {
        public:
            explicit gles_instance(const instance_options& options) : options_(options) {}

            bool init() {
                if (!platform_available())
                    return false;
                gl_info info;
                std::string error;
                if (!probe_gl(info, error))
                    return false;
                adapter_info a;
                a.api = api::gles;
                a.index = 0;
                a.name = info.renderer.empty() ? "OpenGL ES" : info.renderer;
                a.vendor_id = vendor_from(info.vendor, info.renderer);
                std::string lower = info.renderer;
                std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
                const bool software = lower.find("llvmpipe") != std::string::npos ||
                                      lower.find("swiftshader") != std::string::npos ||
                                      lower.find("softpipe") != std::string::npos;
                a.type = software ? adapter_type::cpu : adapter_type::integrated;
                a.driver_name = info.version;
                a.api_version = (info.major << 22) | (info.minor << 12);
                if (info.major < 3) {
                    a.usable = false;
                    a.unusable_reason = "needs OpenGL ES 3.0 or WebGL 2";
                }
                adapters_.push_back(std::move(a));
                return true;
            }

            api get_api() const override { return api::gles; }
            const std::vector<adapter_info>& adapters() const override { return adapters_; }

            std::unique_ptr<conduit::device> create_device(const device_options& options) override {
                if (options.adapter_index >= adapters_.size() || !adapters_[options.adapter_index].usable)
                    return nullptr;
                device_options o = options;
                o.validation = o.validation || options_.validation;
                auto dev = std::make_unique<gles::device>(adapters_[options.adapter_index], o);
                std::string error;
                if (!dev->init(error))
                    return nullptr;
                return dev;
            }

        private:
            instance_options options_;
            std::vector<adapter_info> adapters_;
        };
    }

    std::unique_ptr<conduit::swapchain> make_swapchain(device& device, const swapchain_desc& desc) {
        auto sc = std::make_unique<gles_swapchain>(device, desc);
        if (!sc->init())
            return nullptr;
        return sc;
    }
}

namespace conduit::backends {
    std::unique_ptr<conduit::instance> create_gles_instance(const instance_options& options) {
        auto instance = std::make_unique<gles::gles_instance>(options);
        if (!instance->init())
            return nullptr;
        return instance;
    }
}
