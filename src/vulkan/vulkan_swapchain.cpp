#include "vulkan/vulkan_device.hpp"

#include "vulkan/vulkan_convert.hpp"

#include <algorithm>
#include <cstdint>

namespace conduit::vk {
    namespace {
        struct surface_header {
            VkStructureType s_type;
            const void* p_next;
            VkFlags flags;
        };
#if defined(_WIN32)
        struct win32_surface_info {
            surface_header header;
            void* hinstance;
            void* hwnd;
        };
        constexpr auto win32_type = static_cast<VkStructureType>(1000009000);
#elif defined(__APPLE__)
        struct metal_surface_info {
            surface_header header;
            const void* layer;
        };
        constexpr auto metal_type = static_cast<VkStructureType>(1000217000);
#elif defined(__ANDROID__)
        struct android_surface_info {
            surface_header header;
            void* window;
        };
        constexpr auto android_type = static_cast<VkStructureType>(1000008000);
#else
        struct xlib_surface_info {
            surface_header header;
            void* dpy;
            unsigned long window;
        };
        struct wayland_surface_info {
            surface_header header;
            void* display;
            void* surface;
        };
        constexpr auto xlib_type = static_cast<VkStructureType>(1000004000);
        constexpr auto wayland_type = static_cast<VkStructureType>(1000006000);
#endif

        template <class Info> VkSurfaceKHR create_with(VkInstance instance, const char* fn_name, const Info& info) {
            using fn = VkResult(VKAPI_PTR*)(VkInstance, const Info*, const VkAllocationCallbacks*, VkSurfaceKHR*);
            auto create = reinterpret_cast<fn>(vkGetInstanceProcAddr(instance, fn_name));
            VkSurfaceKHR surface = VK_NULL_HANDLE;
            if (!create || create(instance, &info, nullptr, &surface) != VK_SUCCESS)
                return VK_NULL_HANDLE;
            return surface;
        }

        VkSurfaceKHR create_surface(VkInstance instance, const native_window& w) {
#if defined(_WIN32)
            win32_surface_info info{{win32_type, nullptr, 0}, w.display, w.window};
            return create_with(instance, "vkCreateWin32SurfaceKHR", info);
#elif defined(__APPLE__)
            metal_surface_info info{{metal_type, nullptr, 0}, w.window};
            return create_with(instance, "vkCreateMetalSurfaceEXT", info);
#elif defined(__ANDROID__)
            android_surface_info info{{android_type, nullptr, 0}, w.window};
            return create_with(instance, "vkCreateAndroidSurfaceKHR", info);
#else
            if (w.wayland) {
                wayland_surface_info info{{wayland_type, nullptr, 0}, w.display, w.window};
                return create_with(instance, "vkCreateWaylandSurfaceKHR", info);
            }
            xlib_surface_info info{
                {xlib_type, nullptr, 0}, w.display, static_cast<unsigned long>(reinterpret_cast<uintptr_t>(w.window))};
            return create_with(instance, "vkCreateXlibSurfaceKHR", info);
#endif
        }

        format from_vk(VkFormat f) {
            switch (f) {
            case VK_FORMAT_B8G8R8A8_UNORM:
                return format::bgra8_unorm;
            case VK_FORMAT_B8G8R8A8_SRGB:
                return format::bgra8_srgb;
            case VK_FORMAT_R8G8B8A8_UNORM:
                return format::rgba8_unorm;
            case VK_FORMAT_R8G8B8A8_SRGB:
                return format::rgba8_srgb;
            case VK_FORMAT_R16G16B16A16_SFLOAT:
                return format::rgba16_float;
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
                return format::rgb10a2_unorm;
            default:
                return format::unknown;
            }
        }
    }

    class swapchain final : public conduit::swapchain {
    public:
        swapchain(vk::device& device, const swapchain_desc& desc) : dev_(device), desc_(desc) {}

        ~swapchain() override {
            dev_.fn_.vkDeviceWaitIdle(dev_.device_);
            destroy();
            for (VkSemaphore s : acquire_sems_)
                dev_.fn_.vkDestroySemaphore(dev_.device_, s, nullptr);
            if (surface_)
                vkDestroySurfaceKHR(dev_.inst_->instance, surface_, nullptr);
        }

        bool init() {
            if (!dev_.swapchain_ext_ || !dev_.inst_->surface)
                return false;
            surface_ = create_surface(dev_.inst_->instance, desc_.window);
            if (!surface_)
                return false;
            VkBool32 supported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev_.physical_, dev_.queues_[0].family, surface_, &supported);
            if (!supported)
                return false;
            build(desc_.width, desc_.height);
            return true;
        }

        bool acquire(image& out) override {
            if (surface_lost_) {
                surface_lost_ = false;
                dev_.fn_.vkDeviceWaitIdle(dev_.device_);
                destroy();
                if (surface_)
                    vkDestroySurfaceKHR(dev_.inst_->instance, surface_, nullptr);
                surface_ = create_surface(dev_.inst_->instance, desc_.window);
                if (!surface_)
                    return false;
                needs_rebuild_ = true;
            }
            if (swapchain_ && !needs_rebuild_) {
                // some drivers keep handing out stale-sized images without ever reporting out-of-date
                VkSurfaceCapabilitiesKHR caps{};
                if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev_.physical_, surface_, &caps) == VK_SUCCESS &&
                    caps.currentExtent.width != 0xFFFFFFFFu) {
                    if (!caps.currentExtent.width || !caps.currentExtent.height)
                        return false;
                    if (caps.currentExtent.width != extent_.width || caps.currentExtent.height != extent_.height)
                        needs_rebuild_ = true;
                }
            }
            if (!swapchain_)
                needs_rebuild_ = true;
            if (needs_rebuild_) {
                needs_rebuild_ = false;
                if (!build(desc_.width, desc_.height))
                    return false;
            }
            for (int attempt = 0; attempt < 2; attempt++) {
                VkSemaphore sem = acquire_sems_[ring_];
                // bounded: a window that is hidden or occluded must not hang the whole game loop
                const VkResult r = dev_.fn_.vkAcquireNextImageKHR(dev_.device_, swapchain_, 250'000'000ull, sem,
                                                                  VK_NULL_HANDLE, &index_);
                if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
                    if (r == VK_SUBOPTIMAL_KHR)
                        needs_rebuild_ = true;
                    ring_ = (ring_ + 1) % acquire_sems_.size();
                    dev_.set_pending_acquire(sem);
                    out.texture = handles_[index_];
                    out.index = index_;
                    return true;
                }
                if (r == VK_TIMEOUT || r == VK_NOT_READY)
                    return false;
                if (r == VK_ERROR_SURFACE_LOST_KHR) {
                    surface_lost_ = true;
                    return false;
                }
                if (r != VK_ERROR_OUT_OF_DATE_KHR) {
                    dev_.report_error("vkAcquireNextImageKHR failed (" + std::to_string(r) + ")");
                    return false;
                }
                if (!build(desc_.width, desc_.height))
                    return false;
            }
            return false;
        }

        void present() override {
            if (!swapchain_)
                return;
            VkSemaphore done = render_done_[index_];
            dev_.submit_present(done);
            VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &done;
            pi.swapchainCount = 1;
            pi.pSwapchains = &swapchain_;
            pi.pImageIndices = &index_;
            std::lock_guard lock(*dev_.present_mutex());
            const VkResult r = dev_.fn_.vkQueuePresentKHR(dev_.present_queue(), &pi);
            if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR)
                needs_rebuild_ = true;
            else if (r == VK_ERROR_SURFACE_LOST_KHR)
                surface_lost_ = true;
            else if (r != VK_SUCCESS)
                dev_.report_error("vkQueuePresentKHR failed (" + std::to_string(r) + ")");
        }

        bool resize(uint32_t width, uint32_t height) override {
            const bool same = desc_.width == width && desc_.height == height;
            desc_.width = width;
            desc_.height = height;
            if (!width || !height) {
                dev_.fn_.vkDeviceWaitIdle(dev_.device_);
                destroy();
                return false;
            }
            if (swapchain_ && same && !needs_rebuild_)
                return true;
            needs_rebuild_ = false;
            return build(width, height);
        }

        void set_present_mode(present_mode mode) override {
            if (mode == desc_.present)
                return;
            desc_.present = mode;
            needs_rebuild_ = true;
        }

        format get_format() const override { return rhi_format_; }
        uint32_t width() const override { return swapchain_ ? extent_.width : 0; }
        uint32_t height() const override { return swapchain_ ? extent_.height : 0; }
        present_mode get_present_mode() const override { return active_mode_; }
        uint32_t image_count() const override { return static_cast<uint32_t>(handles_.size()); }
        bool is_hdr() const override { return hdr_active_; }

    private:
        void destroy() {
            for (texture_handle h : handles_)
                dev_.destroy_wrapped(h);
            handles_.clear();
            images_.clear();
            for (VkSemaphore s : render_done_)
                dev_.fn_.vkDestroySemaphore(dev_.device_, s, nullptr);
            render_done_.clear();
            if (swapchain_) {
                dev_.fn_.vkDestroySwapchainKHR(dev_.device_, swapchain_, nullptr);
                swapchain_ = VK_NULL_HANDLE;
            }
        }

        bool build(uint32_t width, uint32_t height) {
            VkSurfaceCapabilitiesKHR caps{};
            if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev_.physical_, surface_, &caps) != VK_SUCCESS)
                return false;
            VkExtent2D ext = caps.currentExtent;
            if (ext.width == 0xFFFFFFFFu) {
                ext.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
                ext.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
            }
            if (!ext.width || !ext.height) {
                dev_.fn_.vkDeviceWaitIdle(dev_.device_);
                destroy();
                return false;
            }

            uint32_t count = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(dev_.physical_, surface_, &count, nullptr);
            std::vector<VkSurfaceFormatKHR> formats(count);
            vkGetPhysicalDeviceSurfaceFormatsKHR(dev_.physical_, surface_, &count, formats.data());
            VkSurfaceFormatKHR chosen{};
            bool found = false;
            hdr_active_ = false;
            if (desc_.hdr)
                for (const auto& f : formats)
                    if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                        f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
                        chosen = f;
                        found = hdr_active_ = true;
                    }
            for (VkFormat want : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM})
                for (const auto& f : formats)
                    if (!found && f.format == want && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                        chosen = f;
                        found = true;
                    }
            if (!found && !formats.empty() && from_vk(formats[0].format) != format::unknown) {
                chosen = formats[0];
                found = true;
            }
            if (!found) {
                dev_.report_error("swapchain: no usable surface format");
                return false;
            }

            vkGetPhysicalDeviceSurfacePresentModesKHR(dev_.physical_, surface_, &count, nullptr);
            std::vector<VkPresentModeKHR> modes(count);
            vkGetPhysicalDeviceSurfacePresentModesKHR(dev_.physical_, surface_, &count, modes.data());
            auto has = [&](VkPresentModeKHR m) { return std::find(modes.begin(), modes.end(), m) != modes.end(); };
            VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
            active_mode_ = present_mode::fifo;
            if (desc_.present == present_mode::mailbox && has(VK_PRESENT_MODE_MAILBOX_KHR)) {
                mode = VK_PRESENT_MODE_MAILBOX_KHR;
                active_mode_ = present_mode::mailbox;
            } else if (desc_.present == present_mode::immediate) {
                if (has(VK_PRESENT_MODE_IMMEDIATE_KHR)) {
                    mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                    active_mode_ = present_mode::immediate;
                } else if (has(VK_PRESENT_MODE_MAILBOX_KHR)) {
                    mode = VK_PRESENT_MODE_MAILBOX_KHR;
                    active_mode_ = present_mode::mailbox;
                }
            }

            uint32_t image_count = (std::max)(desc_.image_count, caps.minImageCount);
            if (caps.maxImageCount)
                image_count = (std::min)(image_count, caps.maxImageCount);

            VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            usage |= caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            usage |= caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            if (desc_.allow_sampling)
                usage |= caps.supportedUsageFlags & VK_IMAGE_USAGE_SAMPLED_BIT;

            VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
            ci.surface = surface_;
            ci.minImageCount = image_count;
            ci.imageFormat = chosen.format;
            ci.imageColorSpace = chosen.colorSpace;
            ci.imageExtent = ext;
            ci.imageArrayLayers = 1;
            ci.imageUsage = usage;
            ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            ci.preTransform = (caps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
                                  ? VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR
                                  : caps.currentTransform;
            ci.compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                                    ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR
                                    : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
            ci.presentMode = mode;
            ci.clipped = VK_TRUE;
            ci.oldSwapchain = swapchain_;

            dev_.fn_.vkDeviceWaitIdle(dev_.device_);
            VkSwapchainKHR fresh = VK_NULL_HANDLE;
            const VkResult r = dev_.fn_.vkCreateSwapchainKHR(dev_.device_, &ci, nullptr, &fresh);
            for (texture_handle h : handles_)
                dev_.destroy_wrapped(h);
            handles_.clear();
            for (VkSemaphore s : render_done_)
                dev_.fn_.vkDestroySemaphore(dev_.device_, s, nullptr);
            render_done_.clear();
            if (swapchain_)
                dev_.fn_.vkDestroySwapchainKHR(dev_.device_, swapchain_, nullptr);
            swapchain_ = VK_NULL_HANDLE;
            if (r != VK_SUCCESS) {
                dev_.report_error("vkCreateSwapchainKHR failed (" + std::to_string(r) + ")");
                return false;
            }
            swapchain_ = fresh;
            extent_ = ext;
            rhi_format_ = from_vk(chosen.format);

            vk_get_swapchain_images_khr(image_count);
            texture_desc td;
            td.format = rhi_format_;
            td.width = ext.width;
            td.height = ext.height;
            td.usage = texture_usage::color_target |
                       ((usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ? texture_usage::copy_src : texture_usage::none) |
                       ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) ? texture_usage::copy_dst : texture_usage::none) |
                       ((usage & VK_IMAGE_USAGE_SAMPLED_BIT) ? texture_usage::sampled : texture_usage::none);
            td.name = "backbuffer";
            for (VkImage image : images_)
                handles_.push_back(dev_.wrap_swapchain_image(image, td, chosen.format));

            VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            for (size_t i = 0; i < images_.size(); i++) {
                VkSemaphore s = VK_NULL_HANDLE;
                dev_.fn_.vkCreateSemaphore(dev_.device_, &si, nullptr, &s);
                render_done_.push_back(s);
            }
            // acquire semaphores outlive rebuilds; one spare so a pending wait never meets a reused semaphore
            while (acquire_sems_.size() < images_.size() + 1) {
                VkSemaphore s = VK_NULL_HANDLE;
                dev_.fn_.vkCreateSemaphore(dev_.device_, &si, nullptr, &s);
                acquire_sems_.push_back(s);
            }
            index_ = 0;
            return true;
        }

        void vk_get_swapchain_images_khr(uint32_t) {
            uint32_t n = 0;
            dev_.fn_.vkGetSwapchainImagesKHR(dev_.device_, swapchain_, &n, nullptr);
            images_.resize(n);
            dev_.fn_.vkGetSwapchainImagesKHR(dev_.device_, swapchain_, &n, images_.data());
        }

        device& dev_;
        swapchain_desc desc_;
        VkSurfaceKHR surface_ = VK_NULL_HANDLE;
        VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
        VkExtent2D extent_{};
        format rhi_format_ = format::bgra8_unorm;
        present_mode active_mode_ = present_mode::fifo;
        bool hdr_active_ = false;
        bool needs_rebuild_ = false;
        bool surface_lost_ = false;
        std::vector<VkImage> images_;
        std::vector<texture_handle> handles_;
        std::vector<VkSemaphore> render_done_;
        std::vector<VkSemaphore> acquire_sems_;
        size_t ring_ = 0;
        uint32_t index_ = 0;
    };

    std::unique_ptr<conduit::swapchain> make_swapchain(vk::device& device, const swapchain_desc& desc) {
        auto sc = std::make_unique<swapchain>(device, desc);
        if (!sc->init())
            return nullptr;
        return sc;
    }
}
