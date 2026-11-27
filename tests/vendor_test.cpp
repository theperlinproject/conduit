#include <conduit/render_profile.hpp>
#include <conduit/vendor.hpp>

#include <cmath>
#include <cstdio>
#include <set>

using namespace conduit;
using namespace conduit::vendor;

static int g_failed = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                       \
            g_failed++;                                                                                                \
        }                                                                                                              \
    } while (0)

int main() {
    uint32_t w = 0, h = 0;
    render_size_for(quality::native, 1920, 1080, w, h);
    CHECK(w == 1920 && h == 1080);
    render_size_for(quality::quality, 2560, 1440, w, h);
    CHECK(w == 1708 && h == 960);
    render_size_for(quality::performance, 3840, 2160, w, h);
    CHECK(w == 1920 && h == 1080);
    render_size_for(quality::ultra_performance, 1921, 1081, w, h);
    CHECK(w % 2 == 0 && h % 2 == 0 && w >= 641 && w <= 646);
    render_size_for(quality::quality, 1, 1, w, h);
    CHECK(w == 2 && h == 2);
    for (quality q : {quality::native, quality::ultra_quality, quality::quality, quality::balanced,
                      quality::performance, quality::ultra_performance}) {
        render_size_for(q, 1280, 720, w, h);
        CHECK(w <= 1280 && h <= 720 && w >= 2 && h >= 2);
        CHECK(scale_of(q) > 0.0f && scale_of(q) <= 1.0f);
    }

    CHECK(jitter_phases(1920, 1920) == 8);
    CHECK(jitter_phases(960, 1920) == 32);
    std::set<std::pair<int, int>> seen;
    const uint32_t phases = jitter_phases(1280, 1920);
    for (uint32_t i = 0; i < phases; i++) {
        const auto j = jitter_for(i, 1280, 1920);
        CHECK(j[0] >= -0.5f && j[0] < 0.5f && j[1] >= -0.5f && j[1] < 0.5f);
        seen.insert({static_cast<int>(std::floor((j[0] + 0.5f) * 8)), static_cast<int>(std::floor((j[1] + 0.5f) * 8))});
        const auto again = jitter_for(i + phases, 1280, 1920);
        CHECK(again[0] == j[0] && again[1] == j[1]);
    }
    CHECK(seen.size() >= phases / 2);

    vendor_hooks hooks;
    std::string log;
    auto session = vendor::session::begin(api::vulkan, {}, hooks, &log);
    CHECK(session == nullptr);
    CHECK(!log.empty());

    CHECK(quality_for(1.0f) == quality::native);
    CHECK(quality_for(0.75f) == quality::ultra_quality);
    CHECK(quality_for(0.667f) == quality::quality);
    CHECK(quality_for(0.55f) == quality::balanced);
    CHECK(quality_for(0.5f) == quality::performance);
    CHECK(quality_for(0.2f) == quality::ultra_performance);
    for (float s = 0.3f; s <= 1.0f; s += 0.05f)
        CHECK(scale_of(quality_for(s)) + 0.005f >= s);

    // a session is only worth starting where the sdk serves the gpu
    render_profile p;
    p.api = api::d3d12;
    p.fallbacks.emplace_back();
    p.fallbacks[0].adapter.vendor_id = vendor_id::nvidia;
    p.upscalers = {conduit::upscaler::dlss, conduit::upscaler::native};
    p.frame_generation = true;
    session_request r = request_for(p);
    CHECK(r.upscaler && r.frame_generation && r.low_latency);
    p.frame_generation = false;
    p.workload = workload::low_latency;
    r = request_for(p);
    CHECK(r.upscaler && !r.frame_generation && r.low_latency);
    p.fallbacks[0].adapter.vendor_id = vendor_id::amd;
    r = request_for(p);
    CHECK(!r.upscaler && !r.frame_generation && !r.low_latency);
    log.clear();
    CHECK(vendor::session::begin(api::d3d12, r, hooks, &log) == nullptr);
    CHECK(log.contains("no vendor feature"));
    p.fallbacks[0].adapter.vendor_id = vendor_id::nvidia;
    p.api = api::d3d11;
    r = request_for(p);
    CHECK(!r.upscaler && !r.low_latency);

    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::puts("vendor_test ok");
    return 0;
}
