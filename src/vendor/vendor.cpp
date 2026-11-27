#include <conduit/render_profile.hpp>
#include <conduit/vendor.hpp>

#include <algorithm>
#include <cmath>

namespace conduit::vendor {
#if defined(CONDUIT_HAS_STREAMLINE) && CONDUIT_HAS_STREAMLINE
    std::unique_ptr<session> make_streamline_session(api api, const session_request& request, vendor_hooks& hooks,
                                                     std::string* log);
#endif

    std::unique_ptr<session> session::begin(conduit::api api, const session_request& request, vendor_hooks& hooks,
                                            std::string* log) {
        // the interposer wraps every dxgi and vulkan entry point once loaded: not worth it for nothing
        if (!request.upscaler && !request.frame_generation && !request.low_latency) {
            if (log)
                *log += "no vendor feature requested; ";
            return nullptr;
        }
#if defined(CONDUIT_HAS_STREAMLINE) && CONDUIT_HAS_STREAMLINE
        return make_streamline_session(api, request, hooks, log);
#else
        (void)api;
        (void)request;
        (void)hooks;
        if (log)
            *log += "vendor sdks are not built into this binary; ";
        return nullptr;
#endif
    }

    float scale_of(vendor::quality quality) {
        switch (quality) {
        case vendor::quality::native:
            return 1.0f;
        case vendor::quality::ultra_quality:
            return 1.0f / 1.3f;
        case vendor::quality::quality:
            return 1.0f / 1.5f;
        case vendor::quality::balanced:
            return 1.0f / 1.72f;
        case vendor::quality::performance:
            return 0.5f;
        case vendor::quality::ultra_performance:
            return 1.0f / 3.0f;
        }
        return 1.0f;
    }

    quality quality_for(float render_scale) {
        for (const vendor::quality q : {quality::ultra_performance, quality::performance, quality::balanced,
                                        quality::quality, quality::ultra_quality})
            if (scale_of(q) + 0.005f >= render_scale)
                return q;
        return quality::native;
    }

    session_request request_for(const render_profile& profile) {
        session_request r;
        const bool nvidia =
            !profile.fallbacks.empty() && profile.fallbacks.front().adapter.vendor_id == vendor_id::nvidia;
        const bool served = nvidia && (profile.api == api::d3d12 || profile.api == api::vulkan);
        r.upscaler = served && std::find(profile.upscalers.begin(), profile.upscalers.end(), conduit::upscaler::dlss) !=
                                   profile.upscalers.end();
        r.frame_generation = served && profile.frame_generation;
        // reflex costs nothing when idle and is what makes generated frames bearable
        r.low_latency = served && (r.frame_generation || profile.workload == workload::low_latency || r.upscaler);
        return r;
    }

    const char* quality_name(vendor::quality quality) {
        switch (quality) {
        case vendor::quality::native:
            return "native";
        case vendor::quality::ultra_quality:
            return "ultra quality";
        case vendor::quality::quality:
            return "quality";
        case vendor::quality::balanced:
            return "balanced";
        case vendor::quality::performance:
            return "performance";
        case vendor::quality::ultra_performance:
            return "ultra performance";
        }
        return "native";
    }

    void render_size_for(vendor::quality quality, uint32_t output_width, uint32_t output_height, uint32_t& width,
                         uint32_t& height) {
        const float s = scale_of(quality);
        auto even = [](float v) { return (std::max)(2u, (static_cast<uint32_t>(std::ceil(v)) + 1u) & ~1u); };
        width = even(static_cast<float>(output_width) * s);
        height = even(static_cast<float>(output_height) * s);
        width = (std::min)(width, (std::max)(2u, (output_width + 1u) & ~1u));
        height = (std::min)(height, (std::max)(2u, (output_height + 1u) & ~1u));
    }

    uint32_t jitter_phases(uint32_t render_width, uint32_t output_width) {
        const float ratio = static_cast<float>(output_width) / static_cast<float>(std::max(1u, render_width));
        return (std::max)(8u, static_cast<uint32_t>(std::ceil(8.0f * ratio * ratio)));
    }

    namespace {
        float halton(uint32_t index, uint32_t base) {
            float f = 1.0f, r = 0.0f;
            while (index > 0) {
                f /= static_cast<float>(base);
                r += f * static_cast<float>(index % base);
                index /= base;
            }
            return r;
        }
    }

    std::array<float, 2> jitter_for(uint64_t frame, uint32_t render_width, uint32_t output_width) {
        const uint32_t phases = jitter_phases(render_width, output_width);
        const uint32_t i = static_cast<uint32_t>(frame % phases) + 1;
        return {halton(i, 2) - 0.5f, halton(i, 3) - 0.5f};
    }
}
