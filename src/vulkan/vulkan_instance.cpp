#include "backends.hpp"
#include "vulkan/vulkan_device.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>

namespace conduit::vk {
    namespace {
        VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                      VkDebugUtilsMessageTypeFlagsEXT,
                                                      const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
            auto* state = static_cast<instance_state*>(user);
            if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
                return VK_FALSE;
            std::lock_guard lock(state->message_mutex);
            if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                state->error_count++;
            if (state->messages.size() < 256)
                state->messages.emplace_back(data && data->pMessage ? data->pMessage : "");
            std::fprintf(stderr, "[vulkan] %s\n", data && data->pMessage ? data->pMessage : "");
            return VK_FALSE;
        }

        bool has_ext(const std::vector<VkExtensionProperties>& list, const char* name) {
            for (const auto& e : list)
                if (std::strcmp(e.extensionName, name) == 0)
                    return true;
            return false;
        }

        std::vector<std::string> device_extension_names(VkPhysicalDevice device) {
            uint32_t count = 0;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> props(count);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &count, props.data());
            std::vector<std::string> out;
            out.reserve(count);
            for (const auto& p : props)
                out.emplace_back(p.extensionName);
            return out;
        }
    }

    instance_state::~instance_state() {
        if (messenger && vkDestroyDebugUtilsMessengerEXT)
            vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
        if (instance && vkDestroyInstance)
            vkDestroyInstance(instance, nullptr);
    }

    bool physical_info::has(const char* name) const {
        for (const auto& e : extensions)
            if (e == name)
                return true;
        return false;
    }

    namespace {
        class vulkan_instance final : public instance {
        public:
            explicit vulkan_instance(const instance_options& options) : options_(options) {}

            bool init() {
                if (options_.hooks.vk_get_instance_proc_addr) {
                    volkInitializeCustom(
                        reinterpret_cast<PFN_vkGetInstanceProcAddr>(options_.hooks.vk_get_instance_proc_addr));
                } else if (volkInitialize() != VK_SUCCESS) {
                    return false;
                }

                uint32_t api_version = VK_API_VERSION_1_0;
                if (vkEnumerateInstanceVersion)
                    vkEnumerateInstanceVersion(&api_version);
                if (api_version < VK_API_VERSION_1_1)
                    return false;

                state_ = std::make_shared<instance_state>();
                state_->api_version = (std::min)(api_version, VK_MAKE_API_VERSION(0, 1, 3, 0));

                uint32_t ext_count = 0;
                vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
                std::vector<VkExtensionProperties> available(ext_count);
                vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, available.data());

                std::vector<const char*> extensions;
                auto enable = [&](const char* name) {
                    if (!has_ext(available, name))
                        return false;
                    extensions.push_back(name);
                    state_->enabled_extensions.emplace_back(name);
                    return true;
                };

                if (enable(VK_KHR_SURFACE_EXTENSION_NAME)) {
                    state_->surface = true;
#if defined(_WIN32)
                    enable("VK_KHR_win32_surface");
#elif defined(__APPLE__)
                    enable("VK_EXT_metal_surface");
#elif defined(__ANDROID__)
                    enable("VK_KHR_android_surface");
#else
                    enable("VK_KHR_xlib_surface");
                    enable("VK_KHR_xcb_surface");
                    enable("VK_KHR_wayland_surface");
#endif
                    state_->color_space_ext = enable(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
                }
#if defined(__APPLE__)
                const bool portability = enable(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
                enable(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
#else
                const bool portability = false;
#endif

                std::vector<const char*> layers;
                if (options_.validation) {
                    uint32_t layer_count = 0;
                    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
                    std::vector<VkLayerProperties> layer_props(layer_count);
                    vkEnumerateInstanceLayerProperties(&layer_count, layer_props.data());
                    for (const auto& l : layer_props)
                        if (std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                            layers.push_back("VK_LAYER_KHRONOS_validation");
                            state_->validation = true;
                        }
                }
                if (options_.validation)
                    state_->debug_utils = enable(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

                for (const char* e : options_.hooks.instance_extensions)
                    if (std::find(extensions.begin(), extensions.end(), e) == extensions.end() && has_ext(available, e))
                        extensions.push_back(e);

                VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
                app.pApplicationName = options_.app_name;
                app.pEngineName = "conduit";
                app.apiVersion = state_->api_version;

                VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
                ci.pApplicationInfo = &app;
                ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
                ci.ppEnabledExtensionNames = extensions.data();
                ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
                ci.ppEnabledLayerNames = layers.data();
                if (portability)
                    ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;

                if (vkCreateInstance(&ci, nullptr, &state_->instance) != VK_SUCCESS) {
                    state_->instance = VK_NULL_HANDLE;
                    return false;
                }
                volkLoadInstanceOnly(state_->instance);

                if (state_->debug_utils) {
                    VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
                    mi.messageSeverity =
                        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                    mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                    mi.pfnUserCallback = debug_callback;
                    mi.pUserData = state_.get();
                    vkCreateDebugUtilsMessengerEXT(state_->instance, &mi, nullptr, &state_->messenger);
                }

                enumerate_adapters();
                return true;
            }

            api get_api() const override { return api::vulkan; }
            const std::vector<adapter_info>& adapters() const override { return adapters_; }

            std::unique_ptr<conduit::device> create_device(const conduit::device_options& device_options) override {
                if (device_options.adapter_index >= adapters_.size() || !adapters_[device_options.adapter_index].usable)
                    return nullptr;
                auto device = std::make_unique<vk::device>(state_, infos_[device_options.adapter_index],
                                                           adapters_[device_options.adapter_index], device_options);
                std::string error;
                if (!device->init(error)) {
                    std::fprintf(stderr, "[vulkan] device creation failed: %s\n", error.c_str());
                    return nullptr;
                }
                return device;
            }

        private:
            void enumerate_adapters() {
                uint32_t count = 0;
                vkEnumeratePhysicalDevices(state_->instance, &count, nullptr);
                std::vector<VkPhysicalDevice> devices(count);
                vkEnumeratePhysicalDevices(state_->instance, &count, devices.data());

                for (VkPhysicalDevice dev : devices) {
                    physical_info info;
                    info.device = dev;
                    vkGetPhysicalDeviceProperties(dev, &info.props);
                    info.extensions = device_extension_names(dev);

                    adapter_info a;
                    a.api = api::vulkan;
                    a.index = static_cast<uint32_t>(adapters_.size());
                    a.name = info.props.deviceName;
                    a.vendor_id = info.props.vendorID;
                    a.device_id = info.props.deviceID;
                    a.api_version = info.props.apiVersion;
                    a.driver_version = info.props.driverVersion;
                    switch (info.props.deviceType) {
                    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                        a.type = adapter_type::discrete;
                        break;
                    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                        a.type = adapter_type::integrated;
                        break;
                    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                        a.type = adapter_type::virtual_gpu;
                        break;
                    case VK_PHYSICAL_DEVICE_TYPE_CPU:
                        a.type = adapter_type::cpu;
                        break;
                    default:
                        a.type = adapter_type::unknown;
                        break;
                    }

                    VkPhysicalDeviceMemoryProperties mem;
                    vkGetPhysicalDeviceMemoryProperties(dev, &mem);
                    for (uint32_t i = 0; i < mem.memoryHeapCount; i++) {
                        if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                            a.dedicated_vram = std::max<uint64_t>(a.dedicated_vram, mem.memoryHeaps[i].size);
                        else
                            a.shared_memory = std::max<uint64_t>(a.shared_memory, mem.memoryHeaps[i].size);
                    }
                    if (a.type == adapter_type::integrated) {
                        a.shared_memory = (std::max)(a.shared_memory, a.dedicated_vram);
                        a.dedicated_vram = 0;
                    }

                    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
                    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                    if (info.props.apiVersion >= VK_API_VERSION_1_2) {
                        props2.pNext = &driver;
                        vkGetPhysicalDeviceProperties2(dev, &props2);
                        a.driver_name = driver.driverName;
                    }

                    a.usable = true;
                    if (info.props.apiVersion < VK_API_VERSION_1_2) {
                        a.usable = false;
                        a.unusable_reason = "needs Vulkan 1.2";
                    } else {
                        const bool api13 = info.props.apiVersion >= VK_API_VERSION_1_3;
                        VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
                        VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
                        VkPhysicalDeviceDynamicRenderingFeaturesKHR dr{
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
                        VkPhysicalDeviceSynchronization2FeaturesKHR s2{
                            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
                        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
                        f2.pNext = &v12;
                        if (api13) {
                            v12.pNext = &v13;
                        } else if (info.has(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) &&
                                   info.has(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME)) {
                            v12.pNext = &dr;
                            dr.pNext = &s2;
                        }
                        vkGetPhysicalDeviceFeatures2(dev, &f2);
                        const bool dynamic_rendering = api13 ? v13.dynamicRendering : dr.dynamicRendering;
                        const bool sync2 = api13 ? v13.synchronization2 : s2.synchronization2;
                        if (!v12.timelineSemaphore) {
                            a.usable = false;
                            a.unusable_reason = "no timeline semaphores";
                        } else if (!dynamic_rendering) {
                            a.usable = false;
                            a.unusable_reason = "no dynamic rendering";
                        } else if (!sync2) {
                            a.usable = false;
                            a.unusable_reason = "no synchronization2";
                        }
                    }

                    adapters_.push_back(std::move(a));
                    physical_.push_back(dev);
                    infos_.push_back(std::move(info));
                }
            }

            instance_options options_;
            std::shared_ptr<instance_state> state_;
            std::vector<adapter_info> adapters_;
            std::vector<VkPhysicalDevice> physical_;
            std::vector<physical_info> infos_;
        };
    }
}

namespace conduit::backends {
    std::unique_ptr<conduit::instance> create_vulkan_instance(const instance_options& options) {
        auto instance = std::make_unique<vk::vulkan_instance>(options);
        if (!instance->init())
            return nullptr;
        return instance;
    }
}
