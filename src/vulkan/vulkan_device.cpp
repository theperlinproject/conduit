#include "vulkan/vulkan_device.hpp"

#include "vulkan/vulkan_convert.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>

namespace conduit::vk {
    namespace {
        constexpr uint32_t max_frames_in_flight = 4;

        template <class... Ts> void chain(void** tail, Ts*... structs) {
            ((*tail = structs, tail = reinterpret_cast<void**>(&reinterpret_cast<VkBaseOutStructure*>(structs)->pNext)),
             ...);
        }
    }

    device::device(std::shared_ptr<instance_state> instance, const physical_info& phys, const adapter_info& adapter,
                   const device_options& opts)
        : inst_(std::move(instance)), phys_(phys), options_(opts) {
        physical_ = phys.device;
        caps_.adapter = adapter;
        options_.frames_in_flight = std::clamp(options_.frames_in_flight, 1u, max_frames_in_flight);
        use_validator_ = options_.state_validation;
    }

    device::~device() {
        {
            std::lock_guard lock(compile_mutex_);
            compile_stop_ = true;
        }
        compile_wake_.notify_all();
        for (std::thread& t : compile_threads_)
            t.join();
        compile_threads_.clear();
        if (!device_)
            return;
        fn_.vkDeviceWaitIdle(device_);
        retire_.flush_all();

        buffers_.for_each([&](buffer_handle, buffer_rec& r) { vmaDestroyBuffer(allocator_, r.buffer, r.alloc); });
        textures_.for_each([&](texture_handle, texture_rec& r) {
            for (auto& [key, view] : r.views)
                fn_.vkDestroyImageView(device_, view, nullptr);
            if (!r.external)
                vmaDestroyImage(allocator_, r.image, r.alloc);
        });
        samplers_.for_each([&](sampler_handle, sampler_rec& r) { fn_.vkDestroySampler(device_, r.sampler, nullptr); });
        shaders_.for_each([&](shader_handle, shader_rec& r) { fn_.vkDestroyShaderModule(device_, r.module, nullptr); });
        bind_layouts_.for_each([&](bind_layout_handle, bind_layout_rec& r) {
            fn_.vkDestroyDescriptorSetLayout(device_, r.layout, nullptr);
        });
        pipeline_layouts_.for_each([&](pipeline_layout_handle, pipeline_layout_rec& r) {
            fn_.vkDestroyPipelineLayout(device_, r.layout, nullptr);
        });
        pipelines_.for_each([&](pipeline_handle, pipeline_rec& r) {
            fn_.vkDestroyPipeline(device_, r.build ? r.build->pipeline : r.pipeline, nullptr);
        });
        queries_.for_each([&](query_set_handle, query_rec& r) { fn_.vkDestroyQueryPool(device_, r.pool, nullptr); });
        fences_.for_each([&](fence_handle, fence_rec& r) { release_timeline(r.timeline); });

        for (VkDescriptorPool p : persistent_pools_)
            fn_.vkDestroyDescriptorPool(device_, p, nullptr);
        for (auto& f : frame_descriptors_)
            for (VkDescriptorPool p : f.pools)
                fn_.vkDestroyDescriptorPool(device_, p, nullptr);
        for (auto& [id, tp] : thread_pools_)
            for (auto& slot_pools : tp->slots)
                for (auto& q : slot_pools)
                    if (q.pool)
                        fn_.vkDestroyCommandPool(device_, q.pool, nullptr);
        for (auto& q : queues_)
            release_timeline(q.timeline);
        for (auto& [key, pass] : render_passes_)
            fn_.vkDestroyRenderPass(device_, pass, nullptr);
        for (auto& list : slot_framebuffers_)
            for (VkFramebuffer f : list)
                fn_.vkDestroyFramebuffer(device_, f, nullptr);
        for (VkFence f : all_fences_)
            fn_.vkDestroyFence(device_, f, nullptr);
        if (empty_set_layout_)
            fn_.vkDestroyDescriptorSetLayout(device_, empty_set_layout_, nullptr);
        if (pipeline_cache_)
            fn_.vkDestroyPipelineCache(device_, pipeline_cache_, nullptr);
        if (allocator_)
            vmaDestroyAllocator(allocator_);
        fn_.vkDestroyDevice(device_, nullptr);
    }

    bool device::pick_queues(std::string& error) {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, nullptr);
        std::vector<VkQueueFamilyProperties> fams(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_, &count, fams.data());

        int gfx = -1, comp = -1, copy = -1;
        for (uint32_t i = 0; i < count; i++) {
            const VkQueueFlags f = fams[i].queueFlags;
            if (gfx < 0 && (f & VK_QUEUE_GRAPHICS_BIT) && (f & VK_QUEUE_COMPUTE_BIT))
                gfx = static_cast<int>(i);
            if (comp < 0 && (f & VK_QUEUE_COMPUTE_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT))
                comp = static_cast<int>(i);
            if (copy < 0 && (f & VK_QUEUE_TRANSFER_BIT) && !(f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)))
                copy = static_cast<int>(i);
        }
        if (gfx < 0) {
            error = "no graphics+compute queue family";
            return false;
        }

        struct slot {
            uint32_t family, index;
        };
        slot slots[queue_count] = {
            {static_cast<uint32_t>(gfx), 0}, {static_cast<uint32_t>(gfx), 0}, {static_cast<uint32_t>(gfx), 0}};
        if (options_.allow_async_compute) {
            if (comp >= 0)
                slots[1] = {static_cast<uint32_t>(comp), 0};
            else if (fams[gfx].queueCount > 1)
                slots[1] = {static_cast<uint32_t>(gfx), 1};
        }
        if (copy >= 0)
            slots[2] = {static_cast<uint32_t>(copy), 0};
        else if (slots[1].family != static_cast<uint32_t>(gfx) || slots[1].index != 0)
            slots[2] = slots[1];

        queue_priorities_.fill(1.0f);
        std::vector<VkDeviceQueueCreateInfo> infos;
        std::vector<slot> unique;
        for (const slot& s : slots) {
            bool seen = false;
            for (const slot& u : unique)
                seen |= (u.family == s.family && u.index == s.index);
            if (!seen)
                unique.push_back(s);
        }
        std::set<uint32_t> families;
        for (const slot& u : unique)
            families.insert(u.family);
        for (uint32_t fam : families) {
            uint32_t need = 1;
            for (const slot& u : unique)
                if (u.family == fam)
                    need = (std::max)(need, u.index + 1);
            VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
            qi.queueFamilyIndex = fam;
            qi.queueCount = need;
            qi.pQueuePriorities = queue_priorities_.data();
            infos.push_back(qi);
        }
        queue_create_infos_ = std::move(infos);
        for (size_t i = 0; i < queue_count; i++) {
            queues_[i].family = slots[i].family;
            queue_index_[i] = slots[i].index;
        }
        return true;
    }

    bool device::init(std::string& error) {
        if (!pick_queues(error))
            return false;

        const bool api12 = phys_.props.apiVersion >= VK_API_VERSION_1_2;
        const bool api13 = phys_.props.apiVersion >= VK_API_VERSION_1_3;
        vkGetPhysicalDeviceMemoryProperties(physical_, &mem_props_);

        // Vulkan 1.1 drivers expose the 1.2 and 1.3 features this backend likes through extensions, or not at all.
        // Each one is optional: what is missing is replaced by a slower path instead of refusing the device
        const bool ext_timeline = !api12 && phys_.has(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
        const bool ext_bda = !api12 && phys_.has(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        const bool ext_reset = !api12 && phys_.has(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
        const bool ext_f16 = !api12 && phys_.has(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        const bool ext_indexing = !api12 && phys_.has(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
        const bool ext_count = !api12 && phys_.has(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
        const bool ext_render = !api13 && phys_.has(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME) &&
                                (api12 || (phys_.has(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME) &&
                                           phys_.has(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME)));
        const bool ext_sync2 = !api13 && phys_.has(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

        VkPhysicalDeviceFeatures2 sup2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDeviceVulkan11Features sup11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features sup12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features sup13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceTimelineSemaphoreFeatures sup_ts{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        VkPhysicalDeviceBufferDeviceAddressFeatures sup_bda{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        VkPhysicalDeviceHostQueryResetFeatures sup_reset{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES};
        VkPhysicalDeviceShaderFloat16Int8Features sup_f16{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceDescriptorIndexingFeatures sup_indexing{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
        VkPhysicalDeviceDynamicRenderingFeaturesKHR sup_dr{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
        VkPhysicalDeviceSynchronization2FeaturesKHR sup_s2{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
        VkPhysicalDeviceDepthClipEnableFeaturesEXT sup_clip{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};
        {
            void** tail = &sup2.pNext;
            auto link = [&tail](auto& st) {
                *tail = &st;
                tail = reinterpret_cast<void**>(&reinterpret_cast<VkBaseOutStructure*>(&st)->pNext);
            };
            link(sup11);
            if (api12)
                link(sup12);
            if (api13)
                link(sup13);
            if (ext_timeline)
                link(sup_ts);
            if (ext_bda)
                link(sup_bda);
            if (ext_reset)
                link(sup_reset);
            if (ext_f16)
                link(sup_f16);
            if (ext_indexing)
                link(sup_indexing);
            if (ext_render)
                link(sup_dr);
            if (ext_sync2)
                link(sup_s2);
            if (phys_.has(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME))
                link(sup_clip);
            vkGetPhysicalDeviceFeatures2(physical_, &sup2);
        }
        const VkPhysicalDeviceFeatures& sf = sup2.features;

        const bool f_timeline = api12 ? sup12.timelineSemaphore != 0 : (ext_timeline && sup_ts.timelineSemaphore);
        const bool f_bda = api12 ? sup12.bufferDeviceAddress != 0 : (ext_bda && sup_bda.bufferDeviceAddress);
        const bool f_count = api12 ? sup12.drawIndirectCount != 0 : ext_count;
        const bool f_reset = api12 ? sup12.hostQueryReset != 0 : (ext_reset && sup_reset.hostQueryReset);
        const bool f_f16 = api12 ? sup12.shaderFloat16 != 0 : (ext_f16 && sup_f16.shaderFloat16);
        const bool f_runtime_array =
            api12 ? sup12.runtimeDescriptorArray != 0 : (ext_indexing && sup_indexing.runtimeDescriptorArray);
        const bool f_partially_bound = api12 ? sup12.descriptorBindingPartiallyBound != 0
                                             : (ext_indexing && sup_indexing.descriptorBindingPartiallyBound);
        const bool f_nonuniform = api12 ? sup12.shaderSampledImageArrayNonUniformIndexing != 0
                                        : (ext_indexing && sup_indexing.shaderSampledImageArrayNonUniformIndexing);
        dynamic_rendering_ = api13 ? sup13.dynamicRendering != 0 : (ext_render && sup_dr.dynamicRendering);
        sync2_ = api13 ? sup13.synchronization2 != 0 : (ext_sync2 && sup_s2.synchronization2);
        timeline_native_ = f_timeline;

        VkPhysicalDeviceFeatures2 en2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        VkPhysicalDeviceVulkan11Features en11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        VkPhysicalDeviceVulkan12Features en12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceVulkan13Features en13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceTimelineSemaphoreFeatures en_ts{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
        VkPhysicalDeviceBufferDeviceAddressFeatures en_bda{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
        VkPhysicalDeviceHostQueryResetFeatures en_reset{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES};
        VkPhysicalDeviceShaderFloat16Int8Features en_f16{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
        VkPhysicalDeviceDescriptorIndexingFeatures en_indexing{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
        VkPhysicalDeviceDynamicRenderingFeaturesKHR en_dr{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR};
        VkPhysicalDeviceSynchronization2FeaturesKHR en_s2{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR};
        VkPhysicalDeviceDepthClipEnableFeaturesEXT en_clip{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};

        VkPhysicalDeviceFeatures& ef = en2.features;
        // robustBufferAccess stays off: bounds checks cost shader time and the renderer never indexes out of range
        ef.samplerAnisotropy = sf.samplerAnisotropy;
        ef.multiDrawIndirect = sf.multiDrawIndirect;
        ef.drawIndirectFirstInstance = sf.drawIndirectFirstInstance;
        ef.textureCompressionBC = sf.textureCompressionBC;
        ef.independentBlend = sf.independentBlend;
        ef.fillModeNonSolid = sf.fillModeNonSolid;
        ef.depthClamp = sf.depthClamp;
        ef.depthBiasClamp = sf.depthBiasClamp;
        ef.shaderInt64 = sf.shaderInt64;
        ef.shaderFloat64 = sf.shaderFloat64;
        ef.imageCubeArray = sf.imageCubeArray;
        ef.sampleRateShading = sf.sampleRateShading;
        ef.fragmentStoresAndAtomics = sf.fragmentStoresAndAtomics;
        ef.vertexPipelineStoresAndAtomics = sf.vertexPipelineStoresAndAtomics;
        ef.shaderStorageImageReadWithoutFormat = sf.shaderStorageImageReadWithoutFormat;
        ef.shaderStorageImageWriteWithoutFormat = sf.shaderStorageImageWriteWithoutFormat;
        ef.shaderStorageImageExtendedFormats = sf.shaderStorageImageExtendedFormats;

        en11.shaderDrawParameters = sup11.shaderDrawParameters;
        en11.storageBuffer16BitAccess = sup11.storageBuffer16BitAccess;
        en11.uniformAndStorageBuffer16BitAccess = sup11.uniformAndStorageBuffer16BitAccess;

        en12.timelineSemaphore = f_timeline;
        en12.bufferDeviceAddress = f_bda;
        en12.drawIndirectCount = f_count;
        en12.hostQueryReset = f_reset;
        en12.shaderFloat16 = f_f16;
        en12.samplerMirrorClampToEdge = api12 ? sup12.samplerMirrorClampToEdge : VK_FALSE;
        en12.runtimeDescriptorArray = f_runtime_array;
        en12.descriptorBindingPartiallyBound = f_partially_bound;
        en12.shaderSampledImageArrayNonUniformIndexing = f_nonuniform;
        en_ts.timelineSemaphore = f_timeline;
        en_bda.bufferDeviceAddress = f_bda;
        en_reset.hostQueryReset = f_reset;
        en_f16.shaderFloat16 = f_f16;
        en_indexing.runtimeDescriptorArray = f_runtime_array;
        en_indexing.descriptorBindingPartiallyBound = f_partially_bound;
        en_indexing.shaderSampledImageArrayNonUniformIndexing = f_nonuniform;

        std::vector<const char*> extensions;
        auto enable = [&](const char* name) {
            if (!phys_.has(name))
                return false;
            if (std::find_if(extensions.begin(), extensions.end(),
                             [&](const char* e) { return std::strcmp(e, name) == 0; }) == extensions.end())
                extensions.push_back(name);
            return true;
        };
        swapchain_ext_ = enable(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        if (api13) {
            en13.dynamicRendering = VK_TRUE;
            en13.synchronization2 = VK_TRUE;
            en13.maintenance4 = sup13.maintenance4;
        } else {
            if (dynamic_rendering_) {
                if (!api12) {
                    enable(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
                    enable(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
                }
                enable(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
                en_dr.dynamicRendering = VK_TRUE;
                dynamic_rendering_ext_ = true;
            }
            if (sync2_) {
                enable(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
                en_s2.synchronization2 = VK_TRUE;
            }
        }
        if (!api12) {
            if (f_timeline)
                enable(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
            if (f_bda)
                enable(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
            if (f_reset)
                enable(VK_EXT_HOST_QUERY_RESET_EXTENSION_NAME);
            if (f_f16)
                enable(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
            if (f_runtime_array || f_partially_bound || f_nonuniform)
                enable(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
            if (f_count)
                enable(VK_KHR_DRAW_INDIRECT_COUNT_EXTENSION_NAME);
        }
        const bool memory_budget = enable(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        if (sup_clip.depthClipEnable && enable(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME)) {
            en_clip.depthClipEnable = VK_TRUE;
            depth_clip_enable_ = true;
        }
        enable("VK_KHR_portability_subset");
        for (const char* e : options_.hooks.device_extensions)
            enable(e);

        {
            void** tail = &en2.pNext;
            auto link = [&tail](auto& st) {
                *tail = &st;
                tail = reinterpret_cast<void**>(&reinterpret_cast<VkBaseOutStructure*>(&st)->pNext);
            };
            link(en11);
            if (api12) {
                link(en12);
            } else {
                if (f_timeline)
                    link(en_ts);
                if (f_bda)
                    link(en_bda);
                if (f_reset)
                    link(en_reset);
                if (f_f16)
                    link(en_f16);
                if (f_runtime_array || f_partially_bound || f_nonuniform)
                    link(en_indexing);
            }
            if (api13) {
                link(en13);
            } else {
                if (dynamic_rendering_)
                    link(en_dr);
                if (sync2_)
                    link(en_s2);
            }
            if (depth_clip_enable_)
                link(en_clip);
            *tail = options_.hooks.device_feature_chain;
        }

        VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        ci.pNext = &en2;
        ci.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos_.size());
        ci.pQueueCreateInfos = queue_create_infos_.data();
        ci.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        ci.ppEnabledExtensionNames = extensions.data();
        const VkResult created = vkCreateDevice(physical_, &ci, nullptr, &device_);
        if (created != VK_SUCCESS) {
            error = "vkCreateDevice failed (" + std::to_string(created) + ")";
            device_ = VK_NULL_HANDLE;
            return false;
        }
        volkLoadDeviceTable(&fn_, device_);
        auto alias = [](auto& core, auto ext) {
            if (!core)
                core = ext;
        };
        alias(fn_.vkCmdBeginRendering, fn_.vkCmdBeginRenderingKHR);
        alias(fn_.vkCmdEndRendering, fn_.vkCmdEndRenderingKHR);
        alias(fn_.vkCmdPipelineBarrier2, fn_.vkCmdPipelineBarrier2KHR);
        alias(fn_.vkQueueSubmit2, fn_.vkQueueSubmit2KHR);
        alias(fn_.vkCmdWriteTimestamp2, fn_.vkCmdWriteTimestamp2KHR);
        alias(fn_.vkCmdDrawIndexedIndirectCount, fn_.vkCmdDrawIndexedIndirectCountKHR);
        alias(fn_.vkWaitSemaphores, fn_.vkWaitSemaphoresKHR);
        alias(fn_.vkGetSemaphoreCounterValue, fn_.vkGetSemaphoreCounterValueKHR);
        alias(fn_.vkSignalSemaphore, fn_.vkSignalSemaphoreKHR);
        alias(fn_.vkGetBufferDeviceAddress, fn_.vkGetBufferDeviceAddressKHR);
        alias(fn_.vkResetQueryPool, fn_.vkResetQueryPoolEXT);
        host_query_reset_ = f_reset && fn_.vkResetQueryPool;

        std::mutex* mutexes[queue_count] = {};
        for (size_t i = 0; i < queue_count; i++) {
            fn_.vkGetDeviceQueue(device_, queues_[i].family, queue_index_[i], &queues_[i].queue);
            for (size_t j = 0; j < i; j++)
                if (queues_[j].queue == queues_[i].queue) {
                    queues_[i].shared = queues_[j].shared = true;
                    mutexes[i] = mutexes[j];
                }
            if (!mutexes[i])
                mutexes[i] = &queue_mutexes_[i];
            queues_[i].mutex = mutexes[i];

            queues_[i].timeline = make_timeline(0);
            if (!queues_[i].timeline) {
                error = "timeline semaphore creation failed";
                return false;
            }
        }

        VmaVulkanFunctions vf{};
        vf.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
        vf.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
        VmaAllocatorCreateInfo ai{};
        ai.physicalDevice = physical_;
        ai.device = device_;
        ai.instance = inst_->instance;
        ai.vulkanApiVersion = (std::min)(phys_.props.apiVersion, inst_->api_version);
        ai.preferredLargeHeapBlockSize = 32ull << 20;
        ai.pVulkanFunctions = &vf;
        if (f_bda)
            ai.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        if (memory_budget)
            ai.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
        if (vmaCreateAllocator(&ai, &allocator_) != VK_SUCCESS) {
            error = "VMA allocator creation failed";
            return false;
        }

        VkPipelineCacheCreateInfo pc{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        pc.initialDataSize = options_.pipeline_cache_data.size();
        pc.pInitialData = options_.pipeline_cache_data.empty() ? nullptr : options_.pipeline_cache_data.data();
        if (fn_.vkCreatePipelineCache(device_, &pc, nullptr, &pipeline_cache_) != VK_SUCCESS && pc.initialDataSize) {
            pc.initialDataSize = 0;
            pc.pInitialData = nullptr;
            fn_.vkCreatePipelineCache(device_, &pc, nullptr, &pipeline_cache_);
        }

        if (options_.async_pipelines && !std::getenv("CONDUIT_SYNC_PIPELINES")) {
            const uint32_t cores = std::max(1u, std::thread::hardware_concurrency());
            const uint32_t workers = std::clamp(cores / 2, 1u, 4u);
            for (uint32_t i = 0; i < workers; i++)
                compile_threads_.emplace_back([this] { compile_worker(); });
        }

        VkDescriptorSetLayoutCreateInfo empty{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        fn_.vkCreateDescriptorSetLayout(device_, &empty, nullptr, &empty_set_layout_);

        frame_descriptors_.resize(options_.frames_in_flight);
        slot_framebuffers_.resize(options_.frames_in_flight);
        slot_serials_.assign(options_.frames_in_flight, queue_serials{});

        caps_.features.buffer_device_address = f_bda;
        fill_caps(caps_.adapter);
        caps_.features.multi_draw_indirect = ef.multiDrawIndirect && ef.drawIndirectFirstInstance;
        caps_.features.draw_indirect_first_instance = ef.drawIndirectFirstInstance;
        caps_.features.draw_indirect_count = f_count && fn_.vkCmdDrawIndexedIndirectCount;
        caps_.features.sampler_anisotropy = ef.samplerAnisotropy;
        caps_.features.texture_compression_bc = ef.textureCompressionBC;
        caps_.features.shader_float16 = f_f16;
        caps_.features.shader_int64 = ef.shaderInt64;
        caps_.features.shader_float64 = ef.shaderFloat64;
        caps_.features.depth_clip_control = depth_clip_enable_;
        caps_.features.bindless = f_runtime_array && f_partially_bound && f_nonuniform;
        caps_.features.enhanced_barriers = sync2_;
        caps_.features.timestamp_queries = phys_.props.limits.timestampComputeAndGraphics != 0;
        caps_.features.pipeline_cache_blob = true;
        caps_.features.async_compute = queues_[1].queue != queues_[0].queue;
        caps_.features.dedicated_copy_queue =
            queues_[2].queue != queues_[0].queue && queues_[2].queue != queues_[1].queue;
        caps_.features.mesh_shaders = phys_.has("VK_EXT_mesh_shader");
        caps_.features.work_graphs = phys_.has("VK_AMDX_shader_enqueue");
        caps_.features.gpu_driven = caps_.features.multi_draw_indirect && caps_.features.draw_indirect_count;
        caps_.features.ray_tracing = phys_.has("VK_KHR_ray_query") && phys_.has("VK_KHR_acceleration_structure");
        caps_.features.variable_rate_shading = phys_.has("VK_KHR_fragment_shading_rate");
        caps_.features.cooperative_vector = phys_.has("VK_NV_cooperative_vector");
        caps_.features.cooperative_matrix = phys_.has("VK_KHR_cooperative_matrix");
        caps_.features.hdr_output = inst_->color_space_ext && swapchain_ext_;
        caps_.features.tearing = swapchain_ext_;
        return true;
    }

    void device::fill_caps(const adapter_info&) {
        const VkPhysicalDeviceLimits& l = phys_.props.limits;
        caps_.limits.max_texture2d = l.maxImageDimension2D;
        caps_.limits.max_texture_layers = l.maxImageArrayLayers;
        caps_.limits.max_push_constant_bytes = std::min<uint32_t>(l.maxPushConstantsSize, max_push_constant_bytes);
        caps_.limits.max_sampled_textures_per_stage = l.maxPerStageDescriptorSampledImages;
        caps_.limits.max_color_targets = std::min<uint32_t>(l.maxColorAttachments, max_color_targets);
        const VkSampleCountFlags msaa = l.framebufferColorSampleCounts & l.framebufferDepthSampleCounts;
        caps_.limits.max_msaa_samples = (msaa & VK_SAMPLE_COUNT_8_BIT)   ? 8
                                        : (msaa & VK_SAMPLE_COUNT_4_BIT) ? 4
                                        : (msaa & VK_SAMPLE_COUNT_2_BIT) ? 2
                                                                         : 1;
        caps_.limits.uniform_buffer_alignment = static_cast<uint32_t>(l.minUniformBufferOffsetAlignment);
        caps_.limits.storage_buffer_alignment = static_cast<uint32_t>(l.minStorageBufferOffsetAlignment);
        caps_.limits.copy_row_pitch_alignment =
            static_cast<uint32_t>(std::max<VkDeviceSize>(1, l.optimalBufferCopyRowPitchAlignment));
        caps_.limits.copy_offset_alignment =
            static_cast<uint32_t>(std::max<VkDeviceSize>(4, l.optimalBufferCopyOffsetAlignment));
        caps_.limits.max_anisotropy = l.maxSamplerAnisotropy;
        caps_.limits.timestamp_period_ns = l.timestampPeriod;
        caps_.limits.max_buffer_size = 1ull << 30;

        VkPhysicalDeviceSubgroupProperties sub{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sub};
        vkGetPhysicalDeviceProperties2(physical_, &p2);
        caps_.features.subgroup_ops = (sub.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) &&
                                      (sub.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT);
        caps_.limits.min_subgroup_size = caps_.limits.max_subgroup_size = sub.subgroupSize;

        for (uint32_t i = 0; i < mem_props_.memoryTypeCount; i++) {
            const auto flags = mem_props_.memoryTypes[i].propertyFlags;
            const auto& heap = mem_props_.memoryHeaps[mem_props_.memoryTypes[i].heapIndex];
            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                if (caps_.adapter.type == adapter_type::integrated || caps_.adapter.type == adapter_type::cpu)
                    caps_.features.unified_memory = true;
                else if (heap.size > (256ull << 20) + (64ull << 20))
                    caps_.features.resizable_bar = true;
            }
        }

        VkFormatProperties d24{};
        vkGetPhysicalDeviceFormatProperties(physical_, VK_FORMAT_D24_UNORM_S8_UINT, &d24);
        d24_supported_ = (d24.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;

        for (size_t i = 1; i < static_cast<size_t>(format::count); i++) {
            const format f = static_cast<format>(i);
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(physical_, format_to_vk(f), &props);
            const VkFormatFeatureFlags o = props.optimalTilingFeatures;
            format_use use = format_use::none;
            if (o & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)
                use = use | format_use::sampled;
            if (o & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)
                use = use | format_use::sampled_linear;
            if (o & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)
                use = use | format_use::color_target | format_use::msaa;
            if (o & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT)
                use = use | format_use::blendable;
            if (o & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
                use = use | format_use::depth_target | format_use::msaa;
            if (o & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)
                use = use | format_use::storage;
            caps_.formats[i] = use;
        }
    }

    VkFormat device::format_to_vk(format f) const {
        if (f == format::d24_unorm_s8_uint && !d24_supported_)
            return VK_FORMAT_D32_SFLOAT_S8_UINT;
        return base_format(f);
    }

    native_handles device::native() const {
        native_handles n;
        n.api = api::vulkan;
        n.instance = inst_->instance;
        n.physical_device = physical_;
        n.device = device_;
        n.graphics_queue = queues_[0].queue;
        n.compute_queue = queues_[1].queue;
        n.graphics_family = queues_[0].family;
        n.compute_family = queues_[1].family;
        n.get_instance_proc_addr = reinterpret_cast<void*>(vkGetInstanceProcAddr);
        n.get_device_proc_addr = reinterpret_cast<void*>(vkGetDeviceProcAddr);
        return n;
    }

    native_texture device::native_of(texture_handle h, conduit::state state) {
        texture_rec* t = textures_.get(h);
        if (!t)
            return {};
        native_texture n;
        n.resource = reinterpret_cast<void*>(t->image);
        if (t->alloc) {
            VmaAllocationInfo info{};
            vmaGetAllocationInfo(allocator_, t->alloc, &info);
            n.memory = reinterpret_cast<void*>(info.deviceMemory);
        }
        n.view = reinterpret_cast<void*>(view(*t, {}, texture_dim::tex2d, format::unknown, false));
        n.state = use_of(state, queue_type::graphics, (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0).layout;
        n.native_format = t->format;
        n.width = t->desc.width;
        n.height = t->desc.height;
        n.mips = t->desc.mips;
        n.layers = t->layers;
        n.usage = t->usage;
        n.flags = t->create_flags;
        n.aspect = t->aspect;
        return n;
    }

    void device::name_object(VkObjectType type, uint64_t handle, const char* name) {
        if (!inst_->debug_utils || !name || !*name || !vkSetDebugUtilsObjectNameEXT)
            return;
        VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
        info.objectType = type;
        info.objectHandle = handle;
        info.pObjectName = name;
        vkSetDebugUtilsObjectNameEXT(device_, &info);
    }

    void device::report_error(const std::string& message) {
        {
            std::lock_guard lock(inst_->message_mutex);
            inst_->error_count++;
            if (inst_->messages.size() < 256)
                inst_->messages.push_back(message);
        }
        std::fprintf(stderr, "[rhi/vulkan] %s\n", message.c_str());
    }

    queue_serials device::completed() {
        queue_serials out{};
        for (size_t i = 0; i < queue_count; i++) {
            out[i] = timeline_value(*queues_[i].timeline);
        }
        return out;
    }

    void device::retire(std::function<void()> release) {
        queue_serials snapshot;
        for (size_t i = 0; i < queue_count; i++)
            snapshot[i] = queues_[i].serial.load();
        retire_.push(snapshot, std::move(release));
    }

    buffer_handle device::create_buffer(const buffer_desc& desc) {
        if (desc.size == 0) {
            report_error("CreateBuffer: zero size");
            return {};
        }
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = desc.size;
        VkBufferUsageFlags usage = 0;
        if (has(desc.usage, buffer_usage::vertex))
            usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        if (has(desc.usage, buffer_usage::index))
            usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        if (has(desc.usage, buffer_usage::uniform))
            usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        if (has(desc.usage, buffer_usage::storage))
            usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (has(desc.usage, buffer_usage::indirect))
            usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (has(desc.usage, buffer_usage::copy_src))
            usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (has(desc.usage, buffer_usage::copy_dst))
            usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (caps_.features.buffer_device_address &&
            (usage & (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)))
            usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
        bi.usage = usage;

        uint32_t family_indices[queue_count];
        uint32_t family_count = 0;
        if (desc.cross_queue) {
            for (auto& q : queues_) {
                bool seen = false;
                for (uint32_t i = 0; i < family_count; i++)
                    seen |= family_indices[i] == q.family;
                if (!seen)
                    family_indices[family_count++] = q.family;
            }
        }
        if (family_count > 1) {
            bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bi.queueFamilyIndexCount = family_count;
            bi.pQueueFamilyIndices = family_indices;
        }

        VmaAllocationCreateInfo ai{};
        auto host = [&](bool random) {
            ai.usage = VMA_MEMORY_USAGE_AUTO;
            ai.flags =
                VMA_ALLOCATION_CREATE_MAPPED_BIT | (random ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                                                           : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        };
        memory_class memory = desc.memory;
        switch (memory) {
        case memory_class::gpu_only:
            ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            break;
        case memory_class::upload:
            host(false);
            break;
        case memory_class::readback:
            host(true);
            break;
        case memory_class::gpu_mappable:
            if (caps_.features.resizable_bar || caps_.features.unified_memory) {
                host(false);
                ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                ai.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
            } else {
                host(false);
                memory = memory_class::upload;
            }
            break;
        }

        buffer_rec rec;
        VmaAllocationInfo info{};
        VkResult r = vmaCreateBuffer(allocator_, &bi, &ai, &rec.buffer, &rec.alloc, &info);
        if (r != VK_SUCCESS && desc.memory == memory_class::gpu_mappable) {
            ai.requiredFlags = 0;
            host(false);
            r = vmaCreateBuffer(allocator_, &bi, &ai, &rec.buffer, &rec.alloc, &info);
        }
        if (r != VK_SUCCESS) {
            report_error("CreateBuffer '" + std::string(desc.name ? desc.name : "") + "' failed (" + std::to_string(r) +
                         ")");
            return {};
        }
        rec.desc = desc;
        rec.name = desc.name ? desc.name : "";
        rec.mapped = static_cast<uint8_t*>(info.pMappedData);
        VkMemoryPropertyFlags flags = 0;
        vmaGetAllocationMemoryProperties(allocator_, rec.alloc, &flags);
        rec.coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        if (caps_.features.buffer_device_address && (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) {
            VkBufferDeviceAddressInfo ba{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            ba.buffer = rec.buffer;
            rec.address = fn_.vkGetBufferDeviceAddress(device_, &ba);
        }
        name_object(VK_OBJECT_TYPE_BUFFER, reinterpret_cast<uint64_t>(rec.buffer), desc.name);
        buffer_bytes_ += static_cast<int64_t>(desc.size);
        live_buffers_++;

        buffer_handle h = buffers_.alloc(std::move(rec));
        buffer_rec* stored = buffers_.get(h);
        stored->desc.name = stored->name.c_str();
        stored->desc.memory = memory;
        if (use_validator_)
            validator_.on_create(h, stored->desc);
        return h;
    }

    texture_handle device::create_texture(const texture_desc& desc) {
        if (desc.format == format::unknown || desc.width == 0 || desc.height == 0) {
            report_error("CreateTexture: invalid description");
            return {};
        }
        texture_rec rec;
        rec.desc = desc;
        rec.desc.mips = (std::max)(1u, desc.mips);
        rec.name = desc.name ? desc.name : "";
        rec.format = format_to_vk(desc.format);
        rec.aspect = aspect_of(desc.format);
        rec.layers = layer_count_of(desc);

        VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ii.imageType = desc.dim == texture_dim::tex3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        ii.format = rec.format;
        ii.extent = {desc.width, desc.height, depth_of(desc)};
        ii.mipLevels = rec.desc.mips;
        ii.arrayLayers = rec.layers;
        ii.samples = samples_of(desc.samples);
        ii.tiling = VK_IMAGE_TILING_OPTIMAL;
        ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (desc.dim == texture_dim::cube)
            ii.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        if (!get_format_info(desc.format).depth)
            ii.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        VkImageUsageFlags usage = 0;
        if (has(desc.usage, texture_usage::sampled))
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (has(desc.usage, texture_usage::storage))
            usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (has(desc.usage, texture_usage::color_target))
            usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (has(desc.usage, texture_usage::depth_target))
            usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (has(desc.usage, texture_usage::copy_src))
            usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (has(desc.usage, texture_usage::copy_dst))
            usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ii.usage = usage;
        rec.usage = usage;
        rec.create_flags = ii.flags;

        uint32_t family_indices[queue_count];
        uint32_t family_count = 0;
        if (desc.cross_queue) {
            for (auto& q : queues_) {
                bool seen = false;
                for (uint32_t i = 0; i < family_count; i++)
                    seen |= family_indices[i] == q.family;
                if (!seen)
                    family_indices[family_count++] = q.family;
            }
        }
        if (family_count > 1) {
            ii.sharingMode = VK_SHARING_MODE_CONCURRENT;
            ii.queueFamilyIndexCount = family_count;
            ii.pQueueFamilyIndices = family_indices;
        }

        VmaAllocationCreateInfo ai{};
        ai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
            ai.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        const VkResult r = vmaCreateImage(allocator_, &ii, &ai, &rec.image, &rec.alloc, nullptr);
        if (r != VK_SUCCESS) {
            report_error("CreateTexture '" + rec.name + "' failed (" + std::to_string(r) + ")");
            return {};
        }
        name_object(VK_OBJECT_TYPE_IMAGE, reinterpret_cast<uint64_t>(rec.image), desc.name);

        uint64_t bytes = 0;
        for (uint32_t m = 0; m < rec.desc.mips; m++)
            bytes += static_cast<uint64_t>(row_bytes(desc.format, mip_extent(desc.width, m))) *
                     row_blocks(desc.format, mip_extent(desc.height, m)) * depth_of(desc) * rec.layers *
                     (std::max)(1u, desc.samples);
        texture_bytes_ += static_cast<int64_t>(bytes);
        live_textures_++;

        texture_handle h = textures_.alloc(std::move(rec));
        texture_rec* stored = textures_.get(h);
        stored->desc.name = stored->name.c_str();
        if (use_validator_)
            validator_.on_create(h, stored->desc);
        return h;
    }

    texture_handle device::wrap_swapchain_image(VkImage image, const texture_desc& desc, VkFormat format) {
        texture_rec rec;
        rec.image = image;
        rec.desc = desc;
        rec.name = desc.name ? desc.name : "backbuffer";
        rec.format = format;
        rec.aspect = VK_IMAGE_ASPECT_COLOR_BIT;
        rec.layers = 1;
        rec.external = true;
        texture_handle h = textures_.alloc(std::move(rec));
        texture_rec* stored = textures_.get(h);
        stored->desc.name = stored->name.c_str();
        if (use_validator_)
            validator_.on_create(h, stored->desc);
        return h;
    }

    void device::destroy_wrapped(texture_handle h) {
        texture_rec rec;
        if (!textures_.free(h, &rec))
            return;
        if (use_validator_)
            validator_.on_destroy(h);
        for (auto& [key, view] : rec.views)
            fn_.vkDestroyImageView(device_, view, nullptr);
    }

    VkImageView device::make_view(texture_rec& tex, const subresource_range& range, VkImageViewType type,
                                  VkFormat format, VkImageAspectFlags aspect) {
        const subresource_range r = resolve_range(range, tex.desc);
        const uint64_t key = (static_cast<uint64_t>(r.base_mip) & 0xF) |
                             (static_cast<uint64_t>(r.mip_count) & 0xF) << 4 |
                             (static_cast<uint64_t>(r.base_layer) & 0xFFF) << 8 |
                             (static_cast<uint64_t>(r.layer_count) & 0xFFF) << 20 | static_cast<uint64_t>(type) << 32 |
                             static_cast<uint64_t>(format) << 36 | static_cast<uint64_t>(aspect) << 56;
        std::lock_guard lock(view_mutex_);
        auto it = tex.views.find(key);
        if (it != tex.views.end())
            return it->second;
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = tex.image;
        vi.viewType = type;
        vi.format = format;
        vi.subresourceRange = {aspect, r.base_mip, r.mip_count, r.base_layer,
                               (type == VK_IMAGE_VIEW_TYPE_2D || type == VK_IMAGE_VIEW_TYPE_3D) ? 1u : r.layer_count};
        if (type == VK_IMAGE_VIEW_TYPE_CUBE)
            vi.subresourceRange.layerCount = 6;
        VkImageView view = VK_NULL_HANDLE;
        if (fn_.vkCreateImageView(device_, &vi, nullptr, &view) != VK_SUCCESS) {
            report_error("vkCreateImageView failed");
            return VK_NULL_HANDLE;
        }
        tex.views.emplace(key, view);
        return view;
    }

    VkImageView device::view(texture_rec& tex, const subresource_range& range, texture_dim dim, conduit::format format,
                             bool stencil) {
        VkFormat vk_format = format == conduit::format::unknown ? tex.format : format_to_vk(format);
        VkImageAspectFlags aspect = tex.aspect;
        if (tex.aspect & VK_IMAGE_ASPECT_DEPTH_BIT)
            aspect = stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT;
        return make_view(tex, range, view_type_of(dim), vk_format, aspect);
    }

    VkImageView device::attachment_view(texture_handle h, uint32_t mip, uint32_t layer) {
        texture_rec* tex = textures_.get(h);
        if (!tex)
            return VK_NULL_HANDLE;
        return make_view(*tex, {mip, 1, layer, 1}, VK_IMAGE_VIEW_TYPE_2D, tex->format, tex->aspect);
    }

    sampler_handle device::create_sampler(const sampler_desc& desc) {
        std::lock_guard lock(sampler_mutex_);
        for (auto& [d, h] : sampler_cache_)
            if (d == desc)
                return h;
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter = to_vk(desc.mag);
        si.minFilter = to_vk(desc.min);
        si.mipmapMode = desc.mip == filter::nearest ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = to_vk(desc.u);
        si.addressModeV = to_vk(desc.v);
        si.addressModeW = to_vk(desc.w);
        si.mipLodBias = desc.lod_bias;
        si.anisotropyEnable = desc.max_anisotropy > 1.0f && caps_.features.sampler_anisotropy;
        si.maxAnisotropy = (std::min)(desc.max_anisotropy, caps_.limits.max_anisotropy);
        si.compareEnable = desc.compare;
        si.compareOp = to_vk(desc.compare_op);
        si.minLod = desc.min_lod;
        si.maxLod = desc.max_lod;
        const bool opaque = desc.border[3] > 0.5f;
        const bool white = desc.border[0] > 0.5f;
        si.borderColor = opaque ? (white ? VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE : VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK)
                                : VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        sampler_rec rec;
        rec.desc = desc;
        if (fn_.vkCreateSampler(device_, &si, nullptr, &rec.sampler) != VK_SUCCESS) {
            report_error("vkCreateSampler failed");
            return {};
        }
        sampler_handle h = samplers_.alloc(rec);
        sampler_cache_.emplace_back(desc, h);
        return h;
    }

    shader_handle device::create_shader(const shader_desc& desc) {
        if (desc.format != shader_format::spirv || desc.code.size() < 20 || desc.code.size() % 4 != 0) {
            report_error("CreateShader: expects SPIR-V");
            return {};
        }
        VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        si.codeSize = desc.code.size();
        std::vector<uint32_t> words(desc.code.size() / 4);
        std::memcpy(words.data(), desc.code.data(), desc.code.size());
        si.pCode = words.data();
        shader_rec rec;
        rec.stage = desc.stage;
        rec.entry = desc.entry ? desc.entry : "main";
        if (fn_.vkCreateShaderModule(device_, &si, nullptr, &rec.module) != VK_SUCCESS) {
            report_error("vkCreateShaderModule failed");
            return {};
        }
        name_object(VK_OBJECT_TYPE_SHADER_MODULE, reinterpret_cast<uint64_t>(rec.module), desc.name);
        return shaders_.alloc(std::move(rec));
    }

    bind_layout_handle device::create_bind_layout(const bind_layout_desc& desc) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bind_layout_rec rec;
        for (const bind_layout_entry& e : desc.entries) {
            VkDescriptorSetLayoutBinding b{};
            b.binding = e.binding;
            b.descriptorCount = (std::max)(1u, e.count);
            b.stageFlags = to_vk(e.stages);
            switch (e.type) {
            case binding_type::uniform_buffer:
                b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                break;
            case binding_type::uniform_buffer_dynamic:
                b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                rec.dynamic_count++;
                break;
            case binding_type::storage_buffer:
            case binding_type::storage_buffer_rw:
                b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            case binding_type::sampled_texture:
                b.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            case binding_type::storage_texture_rw:
                b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                break;
            case binding_type::sampler:
                b.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
            }
            bindings.push_back(b);
            rec.entries.push_back(e);
        }
        VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        li.bindingCount = static_cast<uint32_t>(bindings.size());
        li.pBindings = bindings.data();
        if (fn_.vkCreateDescriptorSetLayout(device_, &li, nullptr, &rec.layout) != VK_SUCCESS) {
            report_error("vkCreateDescriptorSetLayout failed");
            return {};
        }
        name_object(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, reinterpret_cast<uint64_t>(rec.layout), desc.name);
        return bind_layouts_.alloc(std::move(rec));
    }

    VkDescriptorPool device::new_descriptor_pool(bool freeable) {
        const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2048}, {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 8192},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024},  {VK_DESCRIPTOR_TYPE_SAMPLER, 4096},
        };
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.flags = freeable ? VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT : 0;
        pi.maxSets = 1024;
        pi.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
        pi.pPoolSizes = sizes;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        fn_.vkCreateDescriptorPool(device_, &pi, nullptr, &pool);
        return pool;
    }

    VkDescriptorSet device::allocate_set(VkDescriptorSetLayout layout, bool per_frame, VkDescriptorPool& pool_out) {
        std::lock_guard lock(descriptor_mutex_);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;

        if (per_frame) {
            frame_descriptors& fd = frame_descriptors_[slot_];
            while (true) {
                if (fd.used >= fd.pools.size())
                    fd.pools.push_back(new_descriptor_pool(false));
                ai.descriptorPool = fd.pools[fd.used];
                const VkResult r = fn_.vkAllocateDescriptorSets(device_, &ai, &set);
                if (r == VK_SUCCESS) {
                    pool_out = ai.descriptorPool;
                    return set;
                }
                if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL)
                    return VK_NULL_HANDLE;
                fd.used++;
            }
        }

        for (int attempt = 0; attempt < 2; attempt++) {
            if (persistent_pools_.empty() || attempt == 1)
                persistent_pools_.push_back(new_descriptor_pool(true));
            ai.descriptorPool = persistent_pools_.back();
            const VkResult r = fn_.vkAllocateDescriptorSets(device_, &ai, &set);
            if (r == VK_SUCCESS) {
                pool_out = ai.descriptorPool;
                return set;
            }
            if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL)
                break;
        }
        return VK_NULL_HANDLE;
    }

    bind_group_handle device::create_bind_group(const bind_group_desc& desc) {
        bind_layout_rec* layout = bind_layouts_.get(desc.layout);
        if (!layout) {
            report_error("CreateBindGroup: unknown layout");
            return {};
        }
        bind_group_rec rec;
        rec.layout = desc.layout;
        rec.per_frame = desc.per_frame;
        rec.dynamic_count = layout->dynamic_count;
        rec.set = allocate_set(layout->layout, desc.per_frame, rec.pool);
        if (!rec.set) {
            report_error("CreateBindGroup: descriptor allocation failed");
            return {};
        }

        std::vector<VkWriteDescriptorSet> writes;
        std::vector<VkDescriptorBufferInfo> buffer_infos;
        std::vector<VkDescriptorImageInfo> image_infos;
        writes.reserve(desc.entries.size());
        buffer_infos.reserve(desc.entries.size());
        image_infos.reserve(desc.entries.size());

        for (const bind_group_entry& e : desc.entries) {
            const bind_layout_entry* le = nullptr;
            for (const bind_layout_entry& cand : layout->entries)
                if (cand.binding == e.binding)
                    le = &cand;
            if (!le) {
                report_error("CreateBindGroup: binding " + std::to_string(e.binding) + " is not in the layout");
                continue;
            }
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = rec.set;
            w.dstBinding = e.binding;
            w.dstArrayElement = e.array_index;
            w.descriptorCount = 1;
            switch (le->type) {
            case binding_type::uniform_buffer:
            case binding_type::uniform_buffer_dynamic:
            case binding_type::storage_buffer:
            case binding_type::storage_buffer_rw: {
                buffer_rec* b = buffers_.get(e.buffer.buffer);
                if (!b) {
                    report_error("CreateBindGroup: binding " + std::to_string(e.binding) + " has no buffer");
                    continue;
                }
                if (use_validator_) {
                    state_validator::bound_resource br;
                    br.buffer = e.buffer.buffer;
                    br.want = le->type == binding_type::storage_buffer_rw ? state::shader_write
                              : le->type == binding_type::storage_buffer  ? state::shader_read
                                                                          : state::uniform_buffer;
                    rec.bound.push_back(br);
                }
                buffer_infos.push_back({b->buffer, e.buffer.offset, e.buffer.size ? e.buffer.size : VK_WHOLE_SIZE});
                w.pBufferInfo = &buffer_infos.back();
                w.descriptorType = le->type == binding_type::uniform_buffer ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                   : le->type == binding_type::uniform_buffer_dynamic
                                       ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                break;
            }
            case binding_type::sampled_texture:
            case binding_type::storage_texture_rw: {
                texture_rec* t = textures_.get(e.texture.texture);
                if (!t) {
                    report_error("CreateBindGroup: binding " + std::to_string(e.binding) + " has no texture");
                    continue;
                }
                if (use_validator_) {
                    state_validator::bound_resource br;
                    br.texture = e.texture.texture;
                    br.range = e.texture.range;
                    br.want =
                        le->type == binding_type::storage_texture_rw ? state::shader_write : state::depth_read_sampled;
                    rec.bound.push_back(br);
                }
                const texture_dim dim = e.texture.view_dim != texture_dim::tex2d ? e.texture.view_dim : le->view_dim;
                VkImageView image_view = view(*t, e.texture.range, dim, e.texture.format, e.texture.stencil);
                const bool depth = (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
                const bool storage = le->type == binding_type::storage_texture_rw;
                image_infos.push_back({VK_NULL_HANDLE, image_view,
                                       storage ? VK_IMAGE_LAYOUT_GENERAL
                                               : (depth ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                        : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)});
                w.pImageInfo = &image_infos.back();
                w.descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                break;
            }
            case binding_type::sampler: {
                sampler_rec* s = samplers_.get(e.sampler);
                if (!s) {
                    report_error("CreateBindGroup: binding " + std::to_string(e.binding) + " has no sampler");
                    continue;
                }
                image_infos.push_back({s->sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                w.pImageInfo = &image_infos.back();
                w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                break;
            }
            }
            writes.push_back(w);
        }
        fn_.vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        live_groups_++;

        bind_group_handle h = bind_groups_.alloc(std::move(rec));
        if (desc.per_frame) {
            std::lock_guard lock(descriptor_mutex_);
            frame_descriptors_[slot_].groups.push_back(h);
        }
        return h;
    }

    pipeline_layout_handle device::create_pipeline_layout(const pipeline_layout_desc& desc) {
        pipeline_layout_rec rec;
        rec.sets = desc.sets;
        std::vector<VkDescriptorSetLayout> sets;
        uint32_t used = 0;
        for (uint32_t i = 0; i < max_bind_groups; i++)
            if (desc.sets[i])
                used = i + 1;
        for (uint32_t i = 0; i < used; i++) {
            bind_layout_rec* l = bind_layouts_.get(desc.sets[i]);
            sets.push_back(l ? l->layout : empty_set_layout_);
        }
        VkPushConstantRange pcr{};
        VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        li.setLayoutCount = used;
        li.pSetLayouts = sets.data();
        if (desc.push_constant_bytes) {
            pcr.stageFlags = to_vk(desc.push_constant_stages);
            pcr.size = desc.push_constant_bytes;
            li.pushConstantRangeCount = 1;
            li.pPushConstantRanges = &pcr;
            rec.push_stages = pcr.stageFlags;
            rec.push_bytes = desc.push_constant_bytes;
        }
        if (fn_.vkCreatePipelineLayout(device_, &li, nullptr, &rec.layout) != VK_SUCCESS) {
            report_error("vkCreatePipelineLayout failed");
            return {};
        }
        name_object(VK_OBJECT_TYPE_PIPELINE_LAYOUT, reinterpret_cast<uint64_t>(rec.layout), desc.name);
        return pipeline_layouts_.alloc(rec);
    }

    struct device::pipeline_job {
        std::string name, vs_entry, fs_entry;
        VkPipelineShaderStageCreateInfo stages[2]{};
        VkVertexInputBindingDescription bindings[max_vertex_streams]{};
        VkVertexInputAttributeDescription attrs[max_vertex_attributes]{};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        VkPipelineRasterizationDepthClipStateCreateInfoEXT clip{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT};
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState blend_states[max_color_targets]{};
        VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        VkPipelineDynamicStateCreateInfo dyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        VkFormat color_formats[max_color_targets]{};
        VkPipelineRenderingCreateInfo ri{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        VkGraphicsPipelineCreateInfo pi{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        std::shared_ptr<pipeline_build> build;
    };

    bool pipeline_build::wait_for(uint64_t timeout_ns) {
        std::unique_lock lock(mutex);
        return done.wait_for(lock, std::chrono::nanoseconds(timeout_ns), [&] { return state.load() != 0; });
    }

    void device::run_pipeline_job(pipeline_job& job) {
        pipeline_build& b = *job.build;
        VkPipeline pipeline = VK_NULL_HANDLE;
        const VkResult r = fn_.vkCreateGraphicsPipelines(device_, pipeline_cache_, 1, &job.pi, nullptr, &pipeline);
        {
            std::lock_guard lock(b.mutex);
            if (r == VK_SUCCESS) {
                name_object(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(pipeline), job.name.c_str());
                b.pipeline = pipeline;
                b.state.store(1, std::memory_order_release);
            } else {
                b.error = "vkCreateGraphicsPipelines failed for '" + job.name + "' (" + std::to_string(r) + ")";
                b.state.store(-1, std::memory_order_release);
            }
        }
        b.done.notify_all();
    }

    void device::compile_worker() {
        for (;;) {
            std::shared_ptr<pipeline_job> job;
            {
                std::unique_lock lock(compile_mutex_);
                compile_wake_.wait(lock, [&] { return compile_stop_ || !compile_queue_.empty(); });
                if (compile_queue_.empty())
                    return;
                job = std::move(compile_queue_.front());
                compile_queue_.pop_front();
                compile_running_++;
            }
            run_pipeline_job(*job);
            pipelines_pending_--;
            {
                std::lock_guard lock(compile_mutex_);
                compile_running_--;
            }
            compile_idle_.notify_all();
        }
    }

    void device::drain_pipeline_jobs() {
        std::unique_lock lock(compile_mutex_);
        compile_idle_.wait(lock, [&] { return compile_queue_.empty() && compile_running_ == 0; });
    }

    VkPipeline device::ready_pipeline(const pipeline_rec& p) {
        if (!p.build)
            return p.pipeline;
        pipeline_build& b = *p.build;
        int state = b.state.load(std::memory_order_acquire);
        // first bind: most pipelines finish within a few ms, worth a short wait so nothing blinks
        if (state == 0 && !b.waited.exchange(true)) {
            b.wait_for(4'000'000ull);
            state = b.state.load(std::memory_order_acquire);
        }
        if (state == 1)
            return b.pipeline;
        if (state < 0 && !b.reported.exchange(true))
            report_error(b.error);
        return VK_NULL_HANDLE;
    }

    pipeline_handle device::create_graphics_pipeline(const graphics_pipeline_desc& desc) {
        pipeline_layout_rec* layout = pipeline_layouts_.get(desc.layout);
        shader_rec* vs = shaders_.get(desc.vertex);
        shader_rec* fs = shaders_.get(desc.fragment);
        if (!layout || !vs) {
            report_error("CreateGraphicsPipeline: missing layout or vertex shader");
            return {};
        }
        auto job = std::make_shared<pipeline_job>();
        job->name = desc.name ? desc.name : "";
        job->vs_entry = vs->entry;
        job->fs_entry = fs ? fs->entry : std::string();
        auto& stages = job->stages;
        uint32_t stage_count = 0;
        stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                 nullptr,
                                 0,
                                 VK_SHADER_STAGE_VERTEX_BIT,
                                 vs->module,
                                 job->vs_entry.c_str(),
                                 nullptr};
        if (fs)
            stages[stage_count++] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                     nullptr,
                                     0,
                                     VK_SHADER_STAGE_FRAGMENT_BIT,
                                     fs->module,
                                     job->fs_entry.c_str(),
                                     nullptr};

        auto& bindings = job->bindings;
        for (uint32_t i = 0; i < desc.stream_count; i++)
            bindings[i] = {i, desc.streams[i].stride,
                           desc.streams[i].step == vertex_step::instance ? VK_VERTEX_INPUT_RATE_INSTANCE
                                                                         : VK_VERTEX_INPUT_RATE_VERTEX};
        auto& attrs = job->attrs;
        for (uint32_t i = 0; i < desc.attribute_count; i++)
            attrs[i] = {desc.attributes[i].location, desc.attributes[i].stream, base_format(desc.attributes[i].format),
                        desc.attributes[i].offset};
        auto& vi = job->vi;
        vi.vertexBindingDescriptionCount = desc.stream_count;
        vi.pVertexBindingDescriptions = bindings;
        vi.vertexAttributeDescriptionCount = desc.attribute_count;
        vi.pVertexAttributeDescriptions = attrs;

        auto& ia = job->ia;
        switch (desc.topology) {
        case topology::triangle_list:
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            break;
        case topology::triangle_strip:
            ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
            break;
        case topology::line_list:
            ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            break;
        case topology::point_list:
            ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
            break;
        }

        auto& vp = job->vp;
        vp.viewportCount = vp.scissorCount = 1;

        auto& clip = job->clip;
        auto& rs = job->rs;
        rs.polygonMode = desc.raster.wireframe ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
        rs.cullMode = desc.raster.cull == cull_mode::none   ? VK_CULL_MODE_NONE
                      : desc.raster.cull == cull_mode::back ? VK_CULL_MODE_BACK_BIT
                                                            : VK_CULL_MODE_FRONT_BIT;
        rs.frontFace = desc.raster.front_face == front_face::counter_clockwise ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                                                               : VK_FRONT_FACE_CLOCKWISE;
        rs.lineWidth = 1.0f;
        rs.depthBiasEnable = desc.raster.depth_bias_constant != 0.0f || desc.raster.depth_bias_slope != 0.0f;
        rs.depthBiasConstantFactor = desc.raster.depth_bias_constant;
        rs.depthBiasSlopeFactor = desc.raster.depth_bias_slope;
        rs.depthBiasClamp = desc.raster.depth_bias_clamp;
        if (!desc.raster.depth_clip) {
            if (depth_clip_enable_) {
                clip.depthClipEnable = VK_FALSE;
                rs.pNext = &clip;
            } else {
                rs.depthClampEnable = VK_TRUE;
            }
        }

        auto& ms = job->ms;
        ms.rasterizationSamples = samples_of(desc.samples);
        ms.alphaToCoverageEnable = desc.blend.alpha_to_coverage;

        auto& ds = job->ds;
        ds.depthTestEnable = desc.depth_stencil.depth_test;
        ds.depthWriteEnable = desc.depth_stencil.depth_write;
        ds.depthCompareOp = to_vk(desc.depth_stencil.depth_compare);
        ds.stencilTestEnable = desc.depth_stencil.stencil_test;
        auto face = [&](const stencil_face& f) {
            VkStencilOpState s{};
            s.failOp = to_vk(f.fail);
            s.depthFailOp = to_vk(f.depth_fail);
            s.passOp = to_vk(f.pass);
            s.compareOp = to_vk(f.compare);
            s.compareMask = desc.depth_stencil.stencil_read_mask;
            s.writeMask = desc.depth_stencil.stencil_write_mask;
            return s;
        };
        ds.front = face(desc.depth_stencil.front);
        ds.back = face(desc.depth_stencil.back);

        auto& blend_states = job->blend_states;
        for (uint32_t i = 0; i < desc.color_count; i++) {
            const blend_target& t = desc.blend.targets[i];
            VkPipelineColorBlendAttachmentState& s = blend_states[i];
            s = {};
            s.blendEnable = t.enable;
            s.srcColorBlendFactor = to_vk(t.src_color);
            s.dstColorBlendFactor = to_vk(t.dst_color);
            s.colorBlendOp = to_vk(t.color_op);
            s.srcAlphaBlendFactor = to_vk(t.src_alpha);
            s.dstAlphaBlendFactor = to_vk(t.dst_alpha);
            s.alphaBlendOp = to_vk(t.alpha_op);
            s.colorWriteMask = (has(t.write_mask, color_mask::r) ? VK_COLOR_COMPONENT_R_BIT : 0) |
                               (has(t.write_mask, color_mask::g) ? VK_COLOR_COMPONENT_G_BIT : 0) |
                               (has(t.write_mask, color_mask::b) ? VK_COLOR_COMPONENT_B_BIT : 0) |
                               (has(t.write_mask, color_mask::a) ? VK_COLOR_COMPONENT_A_BIT : 0);
        }
        auto& cb = job->cb;
        cb.attachmentCount = desc.color_count;
        cb.pAttachments = blend_states;

        static constexpr VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                                            VK_DYNAMIC_STATE_STENCIL_REFERENCE};
        auto& dyn = job->dyn;
        dyn.dynamicStateCount = static_cast<uint32_t>(std::size(dynamic_states));
        dyn.pDynamicStates = dynamic_states;

        auto& color_formats = job->color_formats;
        for (uint32_t i = 0; i < desc.color_count; i++)
            color_formats[i] = format_to_vk(desc.color_formats[i]);
        auto& ri = job->ri;
        render_pass_key pass_key;
        ri.colorAttachmentCount = desc.color_count;
        ri.pColorAttachmentFormats = color_formats;
        if (desc.depth_format != format::unknown) {
            ri.depthAttachmentFormat = format_to_vk(desc.depth_format);
            if (has_stencil(desc.depth_format))
                ri.stencilAttachmentFormat = ri.depthAttachmentFormat;
        }

        auto& pi = job->pi;
        if (dynamic_rendering_) {
            pi.pNext = &ri;
        } else {
            // a render pass with the same formats and sample count stands in for the attachments
            pass_key.color_count = desc.color_count;
            for (uint32_t i = 0; i < desc.color_count; i++) {
                pass_key.color_format[i] = color_formats[i];
                pass_key.color_store[i] = VK_ATTACHMENT_STORE_OP_STORE;
            }
            pass_key.depth_format = ri.depthAttachmentFormat;
            pass_key.depth_stencil = ri.stencilAttachmentFormat != VK_FORMAT_UNDEFINED;
            pass_key.depth_store = VK_ATTACHMENT_STORE_OP_STORE;
            pass_key.samples = desc.samples;
            pi.renderPass = render_pass(pass_key);
            pi.subpass = 0;
        }
        pi.stageCount = stage_count;
        pi.pStages = stages;
        pi.pVertexInputState = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vp;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;
        pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb;
        pi.pDynamicState = &dyn;
        pi.layout = layout->layout;

        pipeline_rec rec;
        rec.bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;
        rec.layout_handle = desc.layout;
        rec.layout = layout->layout;
        rec.push_stages = layout->push_stages;
        rec.push_bytes = layout->push_bytes;
        rec.color_count = desc.color_count;
        rec.color_formats = desc.color_formats;
        rec.depth_format = desc.depth_format;
        rec.stream_count = desc.stream_count;
        job->build = std::make_shared<pipeline_build>();
        if (!compile_threads_.empty()) {
            rec.build = job->build;
            pipelines_pending_++;
            {
                std::lock_guard lock(compile_mutex_);
                compile_queue_.push_back(job);
            }
            compile_wake_.notify_one();
        } else {
            run_pipeline_job(*job);
            if (job->build->state.load() != 1) {
                report_error(job->build->error);
                return {};
            }
            rec.pipeline = job->build->pipeline;
        }
        live_pipelines_++;
        return pipelines_.alloc(rec);
    }

    pipeline_handle device::create_compute_pipeline(const compute_pipeline_desc& desc) {
        pipeline_layout_rec* layout = pipeline_layouts_.get(desc.layout);
        shader_rec* cs = shaders_.get(desc.compute);
        if (!layout || !cs) {
            report_error("CreateComputePipeline: missing layout or shader");
            return {};
        }
        VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    nullptr,
                    0,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    cs->module,
                    cs->entry.c_str(),
                    nullptr};
        pi.layout = layout->layout;
        pipeline_rec rec;
        if (fn_.vkCreateComputePipelines(device_, pipeline_cache_, 1, &pi, nullptr, &rec.pipeline) != VK_SUCCESS) {
            report_error("vkCreateComputePipelines failed");
            return {};
        }
        name_object(VK_OBJECT_TYPE_PIPELINE, reinterpret_cast<uint64_t>(rec.pipeline), desc.name);
        rec.bind_point = VK_PIPELINE_BIND_POINT_COMPUTE;
        rec.compute = true;
        rec.layout_handle = desc.layout;
        rec.layout = layout->layout;
        rec.push_stages = layout->push_stages;
        rec.push_bytes = layout->push_bytes;
        live_pipelines_++;
        return pipelines_.alloc(rec);
    }

    query_set_handle device::create_query_set(const query_set_desc& desc) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = desc.count;
        query_rec rec;
        rec.count = desc.count;
        if (fn_.vkCreateQueryPool(device_, &qi, nullptr, &rec.pool) != VK_SUCCESS) {
            report_error("vkCreateQueryPool failed");
            return {};
        }
        if (host_query_reset_)
            fn_.vkResetQueryPool(device_, rec.pool, 0, desc.count);
        return queries_.alloc(rec);
    }

    void device::destroy(buffer_handle h) {
        buffer_rec rec;
        if (!buffers_.free(h, &rec))
            return;
        buffer_bytes_ -= static_cast<int64_t>(rec.desc.size);
        live_buffers_--;
        if (use_validator_)
            validator_.on_destroy(h);
        retire([this, rec] { vmaDestroyBuffer(allocator_, rec.buffer, rec.alloc); });
    }

    void device::destroy(texture_handle h) {
        texture_rec rec;
        if (!textures_.free(h, &rec))
            return;
        live_textures_--;
        if (use_validator_)
            validator_.on_destroy(h);
        retire([this, rec = std::move(rec)] {
            for (auto& [key, view] : rec.views)
                fn_.vkDestroyImageView(device_, view, nullptr);
            if (!rec.external)
                vmaDestroyImage(allocator_, rec.image, rec.alloc);
        });
    }

    void device::destroy(sampler_handle) {}

    void device::destroy(shader_handle h) {
        shader_rec rec;
        if (!shaders_.free(h, &rec))
            return;
        if (!compile_threads_.empty())
            drain_pipeline_jobs();
        retire([this, m = rec.module] { fn_.vkDestroyShaderModule(device_, m, nullptr); });
    }

    void device::destroy(bind_layout_handle h) {
        bind_layout_rec rec;
        if (!bind_layouts_.free(h, &rec))
            return;
        retire([this, l = rec.layout] { fn_.vkDestroyDescriptorSetLayout(device_, l, nullptr); });
    }

    void device::destroy(bind_group_handle h) {
        bind_group_rec* g = bind_groups_.get(h);
        if (!g || g->per_frame)
            return;
        bind_group_rec rec;
        bind_groups_.free(h, &rec);
        live_groups_--;
        retire([this, rec] {
            std::lock_guard lock(descriptor_mutex_);
            fn_.vkFreeDescriptorSets(device_, rec.pool, 1, &rec.set);
        });
    }

    void device::destroy(pipeline_layout_handle h) {
        pipeline_layout_rec rec;
        if (!pipeline_layouts_.free(h, &rec))
            return;
        // a pipeline still compiling may reference the layout
        if (!compile_threads_.empty())
            drain_pipeline_jobs();
        retire([this, l = rec.layout] { fn_.vkDestroyPipelineLayout(device_, l, nullptr); });
    }

    void device::destroy(pipeline_handle h) {
        pipeline_rec rec;
        if (!pipelines_.free(h, &rec))
            return;
        live_pipelines_--;
        retire([this, p = rec.pipeline, build = rec.build] {
            if (build) {
                build->wait_for(~0ull >> 2);
                fn_.vkDestroyPipeline(device_, build->pipeline, nullptr);
            } else {
                fn_.vkDestroyPipeline(device_, p, nullptr);
            }
        });
    }

    void device::destroy(query_set_handle h) {
        query_rec rec;
        if (!queries_.free(h, &rec))
            return;
        retire([this, p = rec.pool] { fn_.vkDestroyQueryPool(device_, p, nullptr); });
    }

    void device::destroy(fence_handle h) {
        fence_rec rec;
        if (!fences_.free(h, &rec))
            return;
        retire([this, t = std::move(rec.timeline)]() mutable { release_timeline(t); });
    }

    const buffer_desc* device::get_desc(buffer_handle h) const {
        buffer_rec* r = buffers_.get(h);
        return r ? &r->desc : nullptr;
    }

    const texture_desc* device::get_desc(texture_handle h) const {
        texture_rec* r = textures_.get(h);
        return r ? &r->desc : nullptr;
    }

    uint8_t* device::map(buffer_handle h) {
        buffer_rec* r = buffers_.get(h);
        return r ? r->mapped : nullptr;
    }

    void device::flush(buffer_handle h, uint64_t offset, uint64_t size) {
        buffer_rec* r = buffers_.get(h);
        if (r && !r->coherent)
            vmaFlushAllocation(allocator_, r->alloc, offset, size ? size : VK_WHOLE_SIZE);
    }

    void device::invalidate(buffer_handle h, uint64_t offset, uint64_t size) {
        buffer_rec* r = buffers_.get(h);
        if (r && !r->coherent)
            vmaInvalidateAllocation(allocator_, r->alloc, offset, size ? size : VK_WHOLE_SIZE);
    }

    uint64_t device::gpu_address(buffer_handle h) const {
        buffer_rec* r = buffers_.get(h);
        return r ? r->address : 0;
    }

    fence_handle device::create_fence(uint64_t initial) {
        fence_rec rec;
        rec.timeline = make_timeline(initial);
        if (!rec.timeline)
            return {};
        return fences_.alloc(rec);
    }

    uint64_t device::fence_value(fence_handle h) {
        fence_rec* f = fences_.get(h);
        return f ? timeline_value(*f->timeline) : 0;
    }

    bool device::wait_fence(fence_handle h, uint64_t value, uint64_t timeout_ns) {
        fence_rec* f = fences_.get(h);
        return f && timeline_wait(*f->timeline, value, timeout_ns);
    }

    void device::signal_fence(fence_handle h, uint64_t value) {
        if (fence_rec* f = fences_.get(h))
            timeline_signal(*f->timeline, value);
    }

    thread_pools& device::pools_for_this_thread() {
        std::lock_guard lock(pools_mutex_);
        auto& entry = thread_pools_[std::this_thread::get_id()];
        if (!entry) {
            entry = std::make_unique<thread_pools>();
            entry->slots.resize(options_.frames_in_flight);
        }
        return *entry;
    }

    command_list* device::acquire_command_list(queue_type queue) {
        thread_pools& tp = pools_for_this_thread();
        const uint32_t s = slot_;
        auto& per = tp.slots[s][static_cast<size_t>(queue)];
        if (!per.pool) {
            VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pi.queueFamilyIndex = queues_[static_cast<size_t>(queue)].family;
            pi.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            if (fn_.vkCreateCommandPool(device_, &pi, nullptr, &per.pool) != VK_SUCCESS) {
                report_error("vkCreateCommandPool failed");
                return nullptr;
            }
        }
        if (per.used == per.lists.size()) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = per.pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            if (fn_.vkAllocateCommandBuffers(device_, &ai, &cmd) != VK_SUCCESS) {
                report_error("vkAllocateCommandBuffers failed");
                return nullptr;
            }
            per.lists.push_back(std::make_unique<cmd_list>(*this, queue, cmd));
        }
        cmd_list* list = per.lists[per.used++].get();
        list->begin();
        return list;
    }

    void device::submit(conduit::queue_type queue_type, const submit_desc& desc) {
        queue_info& q = queues_[static_cast<size_t>(queue_type)];
        std::vector<VkCommandBuffer> cmds;
        cmds.reserve(desc.lists.size());
        for (command_list* list : desc.lists) {
            auto* vk_list = static_cast<cmd_list*>(list);
            if (vk_list->open_labels() != 0)
                report_error("command list submitted with an open debug label");
            vk_list->end();
            counters_.add(vk_list->counters());
            cmds.push_back(vk_list->cmd());
        }

        std::vector<timeline_op> waits, signals;
        for (const fence_op& w : desc.waits)
            if (fence_rec* f = fences_.get(w.fence))
                waits.push_back({f->timeline.get(), w.value});
        for (const fence_op& s : desc.signals)
            if (fence_rec* f = fences_.get(s.fence))
                signals.push_back({f->timeline.get(), s.value});
        VkSemaphore acquire = VK_NULL_HANDLE;
        if (queue_type == conduit::queue_type::graphics)
            acquire = pending_acquire_.exchange(VK_NULL_HANDLE);

        const VkResult r = queue_submit(q, cmds, waits, signals, acquire, VK_NULL_HANDLE);
        if (r != VK_SUCCESS) {
            report_error("queue submit failed (" + std::to_string(r) + ")");
            lost_ = (r == VK_ERROR_DEVICE_LOST);
            return;
        }
        counters_.count_submit();
    }

    void device::submit_present(VkSemaphore render_done) {
        VkSemaphore acquire = pending_acquire_.exchange(VK_NULL_HANDLE);
        if (queue_submit(queues_[0], {}, {}, {}, acquire, render_done) != VK_SUCCESS)
            report_error("present submit failed");
    }

    // One submission, whatever the driver offers: synchronization2 plus timeline semaphores (Vulkan 1.3, the fast
    // path), timeline semaphores with classic submits, or on Vulkan 1.1 drivers without them a VkFence per submit
    // that the timelines track, with gpu waits turned into cpu waits.
    VkResult device::queue_submit(queue_info& q, std::span<const VkCommandBuffer> cmds,
                                  std::span<const timeline_op> waits, std::span<const timeline_op> signals,
                                  VkSemaphore binary_wait, VkSemaphore binary_signal) {
        if (timeline_native_ && sync2_) {
            std::vector<VkCommandBufferSubmitInfo> infos;
            infos.reserve(cmds.size());
            for (VkCommandBuffer c : cmds) {
                VkCommandBufferSubmitInfo ci{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
                ci.commandBuffer = c;
                infos.push_back(ci);
            }
            std::vector<VkSemaphoreSubmitInfo> wait_infos, signal_infos;
            auto sem_info = [](VkSemaphore s, uint64_t value) {
                VkSemaphoreSubmitInfo i{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
                i.semaphore = s;
                i.value = value;
                i.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                return i;
            };
            for (const timeline_op& w : waits)
                wait_infos.push_back(sem_info(w.target->semaphore, w.value));
            if (binary_wait)
                wait_infos.push_back(sem_info(binary_wait, 0));
            for (const timeline_op& s : signals)
                signal_infos.push_back(sem_info(s.target->semaphore, s.value));
            if (binary_signal)
                signal_infos.push_back(sem_info(binary_signal, 0));

            std::lock_guard lock(*q.mutex);
            const uint64_t serial = q.serial.load() + 1;
            signal_infos.push_back(sem_info(q.timeline->semaphore, serial));
            VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            si.waitSemaphoreInfoCount = static_cast<uint32_t>(wait_infos.size());
            si.pWaitSemaphoreInfos = wait_infos.data();
            si.commandBufferInfoCount = static_cast<uint32_t>(infos.size());
            si.pCommandBufferInfos = infos.data();
            si.signalSemaphoreInfoCount = static_cast<uint32_t>(signal_infos.size());
            si.pSignalSemaphoreInfos = signal_infos.data();
            const VkResult r = fn_.vkQueueSubmit2(q.queue, 1, &si, VK_NULL_HANDLE);
            if (r == VK_SUCCESS)
                q.serial.store(serial);
            return r;
        }

        std::vector<VkSemaphore> wait_sems, signal_sems;
        std::vector<VkPipelineStageFlags> wait_stages;
        std::vector<uint64_t> wait_values, signal_values;

        if (timeline_native_) {
            for (const timeline_op& w : waits) {
                wait_sems.push_back(w.target->semaphore);
                wait_values.push_back(w.value);
            }
            if (binary_wait) {
                wait_sems.push_back(binary_wait);
                wait_values.push_back(0);
            }
            wait_stages.assign(wait_sems.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            for (const timeline_op& s : signals) {
                signal_sems.push_back(s.target->semaphore);
                signal_values.push_back(s.value);
            }
            if (binary_signal) {
                signal_sems.push_back(binary_signal);
                signal_values.push_back(0);
            }
            std::lock_guard lock(*q.mutex);
            const uint64_t serial = q.serial.load() + 1;
            signal_sems.push_back(q.timeline->semaphore);
            signal_values.push_back(serial);
            VkTimelineSemaphoreSubmitInfo ti{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            ti.waitSemaphoreValueCount = static_cast<uint32_t>(wait_values.size());
            ti.pWaitSemaphoreValues = wait_values.data();
            ti.signalSemaphoreValueCount = static_cast<uint32_t>(signal_values.size());
            ti.pSignalSemaphoreValues = signal_values.data();
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO, &ti};
            si.waitSemaphoreCount = static_cast<uint32_t>(wait_sems.size());
            si.pWaitSemaphores = wait_sems.data();
            si.pWaitDstStageMask = wait_stages.data();
            si.commandBufferCount = static_cast<uint32_t>(cmds.size());
            si.pCommandBuffers = cmds.data();
            si.signalSemaphoreCount = static_cast<uint32_t>(signal_sems.size());
            si.pSignalSemaphores = signal_sems.data();
            const VkResult r = fn_.vkQueueSubmit(q.queue, 1, &si, VK_NULL_HANDLE);
            if (r == VK_SUCCESS)
                q.serial.store(serial);
            return r;
        }

        // emulated timelines: wait for the other work on the cpu first, outside the queue lock
        for (const timeline_op& w : waits) {
            if (timeline_value(*w.target) >= w.value)
                continue;
            if (!timeline_wait(*w.target, w.value, 10'000'000'000ull))
                report_error("a fence wait never completed (value " + std::to_string(w.value) +
                             "); submitting without it");
        }
        std::shared_ptr<pooled_fence> fence = acquire_fence();
        VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        if (binary_wait) {
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &binary_wait;
            si.pWaitDstStageMask = &stage;
        }
        si.commandBufferCount = static_cast<uint32_t>(cmds.size());
        si.pCommandBuffers = cmds.data();
        if (binary_signal) {
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &binary_signal;
        }
        std::lock_guard lock(*q.mutex);
        const uint64_t serial = q.serial.load() + 1;
        const VkResult r = fn_.vkQueueSubmit(q.queue, 1, &si, fence->fence);
        if (r != VK_SUCCESS)
            return r;
        q.serial.store(serial);
        auto track = [&](timeline& t, uint64_t value) {
            std::lock_guard tl(t.mutex);
            t.submitted = (std::max)(t.submitted, value);
            t.inflight.push_back({value, fence});
            t.changed.notify_all();
        };
        track(*q.timeline, serial);
        for (const timeline_op& s : signals)
            track(*s.target, s.value);
        return VK_SUCCESS;
    }

    std::shared_ptr<timeline> device::make_timeline(uint64_t initial) {
        auto t = std::make_shared<timeline>();
        t->done = t->submitted = initial;
        if (timeline_native_) {
            VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
            type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            type.initialValue = initial;
            VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
            if (fn_.vkCreateSemaphore(device_, &si, nullptr, &t->semaphore) != VK_SUCCESS)
                return nullptr;
        }
        return t;
    }

    void device::release_timeline(std::shared_ptr<timeline>& t) {
        if (!t)
            return;
        {
            std::lock_guard lock(t->mutex);
            t->inflight.clear();
        }
        if (t->semaphore) {
            fn_.vkDestroySemaphore(device_, t->semaphore, nullptr);
            t->semaphore = VK_NULL_HANDLE;
        }
        t.reset();
    }

    std::shared_ptr<pooled_fence> device::acquire_fence() {
        VkFence fence = VK_NULL_HANDLE;
        {
            std::lock_guard lock(fence_pool_mutex_);
            if (!free_fences_.empty()) {
                fence = free_fences_.back();
                free_fences_.pop_back();
            }
        }
        if (!fence) {
            VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fn_.vkCreateFence(device_, &fi, nullptr, &fence);
            std::lock_guard lock(fence_pool_mutex_);
            all_fences_.push_back(fence);
        }
        return std::shared_ptr<pooled_fence>(new pooled_fence{fence}, [this](pooled_fence* p) {
            if (p->fence) {
                fn_.vkResetFences(device_, 1, &p->fence);
                std::lock_guard lock(fence_pool_mutex_);
                free_fences_.push_back(p->fence);
            }
            delete p;
        });
    }

    // timeline.mutex held
    void device::poll_timeline(timeline& t) {
        for (auto it = t.inflight.begin(); it != t.inflight.end();) {
            if (fn_.vkGetFenceStatus(device_, it->fence->fence) == VK_SUCCESS) {
                t.done = (std::max)(t.done, it->value);
                it = t.inflight.erase(it);
            } else {
                ++it;
            }
        }
    }

    uint64_t device::timeline_value(timeline& t) {
        if (t.semaphore) {
            uint64_t value = 0;
            fn_.vkGetSemaphoreCounterValue(device_, t.semaphore, &value);
            return value;
        }
        std::lock_guard lock(t.mutex);
        poll_timeline(t);
        return t.done;
    }

    bool device::timeline_wait(timeline& t, uint64_t value, uint64_t timeout_ns) {
        if (t.semaphore) {
            VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wi.semaphoreCount = 1;
            wi.pSemaphores = &t.semaphore;
            wi.pValues = &value;
            return fn_.vkWaitSemaphores(device_, &wi, timeout_ns) == VK_SUCCESS;
        }
        using clock = std::chrono::steady_clock;
        const auto start = clock::now();
        auto expired = [&] {
            if (timeout_ns == ~0ull)
                return false;
            return static_cast<uint64_t>(
                       std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count()) >=
                   timeout_ns;
        };
        for (;;) {
            std::shared_ptr<pooled_fence> fence;
            {
                std::unique_lock lock(t.mutex);
                poll_timeline(t);
                if (t.done >= value)
                    return true;
                uint64_t best = ~0ull;
                for (const timeline::pending& p : t.inflight)
                    if (p.value >= value && p.value < best) {
                        best = p.value;
                        fence = p.fence;
                    }
                if (!fence) {
                    // nothing submitted yet reaches the value: it can still come from another thread
                    if (expired())
                        return false;
                    t.changed.wait_for(lock, std::chrono::milliseconds(2));
                    continue;
                }
            }
            const VkResult r = fn_.vkWaitForFences(device_, 1, &fence->fence, VK_TRUE, 20'000'000ull);
            if (r != VK_SUCCESS && r != VK_TIMEOUT)
                return false;
            if (r == VK_TIMEOUT && expired())
                return false;
        }
    }

    void device::timeline_signal(timeline& t, uint64_t value) {
        if (t.semaphore) {
            VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            si.semaphore = t.semaphore;
            si.value = value;
            fn_.vkSignalSemaphore(device_, &si);
            return;
        }
        std::lock_guard lock(t.mutex);
        t.done = (std::max)(t.done, value);
        t.submitted = (std::max)(t.submitted, value);
        t.changed.notify_all();
    }

    void device::pipeline_barrier(VkCommandBuffer cmd, const VkDependencyInfo& info) {
        if (sync2_) {
            fn_.vkCmdPipelineBarrier2(cmd, &info);
            return;
        }
        thread_local std::vector<VkMemoryBarrier> memory;
        thread_local std::vector<VkBufferMemoryBarrier> buffers;
        thread_local std::vector<VkImageMemoryBarrier> images;
        memory.clear();
        buffers.clear();
        images.clear();
        VkPipelineStageFlags src = 0, dst = 0;
        for (uint32_t i = 0; i < info.memoryBarrierCount; i++) {
            const VkMemoryBarrier2& b = info.pMemoryBarriers[i];
            src |= legacy_stages(b.srcStageMask);
            dst |= legacy_stages(b.dstStageMask);
            memory.push_back({VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, legacy_access(b.srcAccessMask),
                              legacy_access(b.dstAccessMask)});
        }
        for (uint32_t i = 0; i < info.bufferMemoryBarrierCount; i++) {
            const VkBufferMemoryBarrier2& b = info.pBufferMemoryBarriers[i];
            src |= legacy_stages(b.srcStageMask);
            dst |= legacy_stages(b.dstStageMask);
            buffers.push_back({VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, nullptr, legacy_access(b.srcAccessMask),
                               legacy_access(b.dstAccessMask), b.srcQueueFamilyIndex, b.dstQueueFamilyIndex, b.buffer,
                               b.offset, b.size});
        }
        for (uint32_t i = 0; i < info.imageMemoryBarrierCount; i++) {
            const VkImageMemoryBarrier2& b = info.pImageMemoryBarriers[i];
            src |= legacy_stages(b.srcStageMask);
            dst |= legacy_stages(b.dstStageMask);
            images.push_back({VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, nullptr, legacy_access(b.srcAccessMask),
                              legacy_access(b.dstAccessMask), b.oldLayout, b.newLayout, b.srcQueueFamilyIndex,
                              b.dstQueueFamilyIndex, b.image, b.subresourceRange});
        }
        if (!src)
            src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        if (!dst)
            dst = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        fn_.vkCmdPipelineBarrier(cmd, src, dst, 0, static_cast<uint32_t>(memory.size()), memory.data(),
                                 static_cast<uint32_t>(buffers.size()), buffers.data(),
                                 static_cast<uint32_t>(images.size()), images.data());
    }

    void device::write_timestamp(VkCommandBuffer cmd, VkQueryPool pool, uint32_t index) {
        if (sync2_)
            fn_.vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, index);
        else
            fn_.vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, index);
    }

    // render passes for drivers without dynamic rendering. One pass per distinct set of formats and load/store
    // operations, created on first use. Pipelines are built against the same cache with neutral operations, which
    // is compatible with any pass that shares the formats and sample count
    VkRenderPass device::render_pass(const render_pass_key& key) {
        std::lock_guard lock(render_pass_mutex_);
        for (auto& [k, pass] : render_passes_)
            if (k == key)
                return pass;

        VkAttachmentDescription attachments[max_color_targets * 2 + 1]{};
        VkAttachmentReference color_refs[max_color_targets]{};
        VkAttachmentReference resolve_refs[max_color_targets]{};
        VkAttachmentReference depth_ref{};
        uint32_t count = 0;
        bool any_resolve = false;
        const VkSampleCountFlagBits samples = samples_of(key.samples);
        for (uint32_t i = 0; i < key.color_count; i++) {
            VkAttachmentDescription& a = attachments[count];
            a.format = key.color_format[i];
            a.samples = samples;
            a.loadOp = static_cast<VkAttachmentLoadOp>(key.color_load[i]);
            a.storeOp = static_cast<VkAttachmentStoreOp>(key.color_store[i]);
            a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout = a.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            color_refs[i] = {count++, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        }
        for (uint32_t i = 0; i < key.color_count; i++) {
            if (!key.color_resolve[i]) {
                resolve_refs[i] = {VK_ATTACHMENT_UNUSED, VK_IMAGE_LAYOUT_UNDEFINED};
                continue;
            }
            any_resolve = true;
            VkAttachmentDescription& a = attachments[count];
            a.format = key.color_format[i];
            a.samples = VK_SAMPLE_COUNT_1_BIT;
            a.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            a.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout = a.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            resolve_refs[i] = {count++, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        }
        const bool have_depth = key.depth_format != VK_FORMAT_UNDEFINED;
        if (have_depth) {
            const VkImageLayout layout = key.depth_read_only ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                             : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentDescription& a = attachments[count];
            a.format = key.depth_format;
            a.samples = samples;
            a.loadOp = static_cast<VkAttachmentLoadOp>(key.depth_load);
            a.storeOp = static_cast<VkAttachmentStoreOp>(key.depth_store);
            a.stencilLoadOp = key.depth_stencil ? a.loadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            a.stencilStoreOp = key.depth_stencil ? a.storeOp : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            a.initialLayout = a.finalLayout = layout;
            depth_ref = {count++, layout};
        }

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = key.color_count;
        sub.pColorAttachments = color_refs;
        sub.pResolveAttachments = any_resolve ? resolve_refs : nullptr;
        sub.pDepthStencilAttachment = have_depth ? &depth_ref : nullptr;

        VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ci.attachmentCount = count;
        ci.pAttachments = attachments;
        ci.subpassCount = 1;
        ci.pSubpasses = &sub;
        VkRenderPass pass = VK_NULL_HANDLE;
        if (fn_.vkCreateRenderPass(device_, &ci, nullptr, &pass) != VK_SUCCESS) {
            report_error("vkCreateRenderPass failed");
            return VK_NULL_HANDLE;
        }
        render_passes_.emplace_back(key, pass);
        return pass;
    }

    // framebuffers live for the frame slot that recorded them
    VkFramebuffer device::make_framebuffer(VkRenderPass pass, const VkImageView* views, uint32_t count, uint32_t width,
                                           uint32_t height) {
        VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fi.renderPass = pass;
        fi.attachmentCount = count;
        fi.pAttachments = views;
        fi.width = width;
        fi.height = height;
        fi.layers = 1;
        VkFramebuffer fb = VK_NULL_HANDLE;
        if (fn_.vkCreateFramebuffer(device_, &fi, nullptr, &fb) != VK_SUCCESS) {
            report_error("vkCreateFramebuffer failed");
            return VK_NULL_HANDLE;
        }
        std::lock_guard lock(framebuffer_mutex_);
        slot_framebuffers_[slot_].push_back(fb);
        return fb;
    }

    std::unique_ptr<conduit::swapchain> device::create_swapchain(const swapchain_desc& desc) {
        return make_swapchain(*this, desc);
    }

    void device::reset_slot(uint32_t s) {
        {
            std::lock_guard lock(pools_mutex_);
            for (auto& [id, tp] : thread_pools_)
                for (auto& per : tp->slots[s]) {
                    if (per.pool)
                        fn_.vkResetCommandPool(device_, per.pool, 0);
                    per.used = 0;
                }
        }
        {
            std::lock_guard lock(framebuffer_mutex_);
            for (VkFramebuffer f : slot_framebuffers_[s])
                fn_.vkDestroyFramebuffer(device_, f, nullptr);
            slot_framebuffers_[s].clear();
        }
        std::lock_guard lock(descriptor_mutex_);
        frame_descriptors& fd = frame_descriptors_[s];
        for (VkDescriptorPool p : fd.pools)
            fn_.vkResetDescriptorPool(device_, p, 0);
        fd.used = 0;
        for (bind_group_handle g : fd.groups) {
            if (bind_groups_.free(g))
                live_groups_--;
        }
        fd.groups.clear();
    }

    void device::begin_frame() {
        frame_++;
        slot_ = static_cast<uint32_t>(frame_ % options_.frames_in_flight);

        const queue_serials& need = slot_serials_[slot_];
        for (size_t i = 0; i < queue_count; i++) {
            if (!need[i])
                continue;
            // reusing a slot early would corrupt a slow frame still in flight, so keep waiting for a long time
            bool ok = false;
            for (int attempt = 0; attempt < 12 && !ok; attempt++)
                ok = timeline_wait(*queues_[i].timeline, need[i], 10'000'000'000ull);
            if (!ok)
                report_error("frame wait did not complete");
        }
        reset_slot(slot_);
        retire_.collect(completed());
    }

    void device::end_frame() {
        for (size_t i = 0; i < queue_count; i++)
            slot_serials_[slot_][i] = queues_[i].serial.load();
        retire_.collect(completed());
    }

    void device::wait_idle() {
        fn_.vkDeviceWaitIdle(device_);
        retire_.collect(completed());
    }

    memory_budget device::query_memory_budget() {
        memory_budget out;
        VmaBudget budgets[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(allocator_, budgets);
        for (uint32_t i = 0; i < mem_props_.memoryHeapCount; i++) {
            const bool local = (mem_props_.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
            (local ? out.device_budget : out.host_budget) += budgets[i].budget;
            (local ? out.device_usage : out.host_usage) += budgets[i].usage;
        }
        out.valid = true;
        return out;
    }

    device_stats device::stats() const {
        device_stats s;
        counters_.fill(s);
        s.buffer_bytes = static_cast<uint64_t>(std::max<int64_t>(0, buffer_bytes_));
        s.texture_bytes = static_cast<uint64_t>(std::max<int64_t>(0, texture_bytes_));
        s.live_buffers = static_cast<uint32_t>(live_buffers_.load());
        s.live_textures = static_cast<uint32_t>(live_textures_.load());
        s.live_pipelines = static_cast<uint32_t>(live_pipelines_.load());
        s.live_bind_groups = static_cast<uint32_t>(live_groups_.load());
        s.pipelines_pending = std::max(s.pipelines_pending, pipelines_pending_.load());
        return s;
    }

    std::vector<uint8_t> device::serialize_pipeline_cache() {
        size_t size = 0;
        if (!pipeline_cache_ || fn_.vkGetPipelineCacheData(device_, pipeline_cache_, &size, nullptr) != VK_SUCCESS ||
            !size)
            return {};
        std::vector<uint8_t> data(size);
        if (fn_.vkGetPipelineCacheData(device_, pipeline_cache_, &size, data.data()) != VK_SUCCESS)
            return {};
        data.resize(size);
        return data;
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
        if (use_validator_)
            for (auto& m : validator_.take_errors())
                out += "state: " + m + "\n";
        return out;
    }
}
