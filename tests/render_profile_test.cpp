#include <conduit/render_profile.hpp>

#include <cstdio>

using namespace conduit;

static int g_failed = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                       \
            g_failed++;                                                                                                \
        }                                                                                                              \
    } while (0)

static candidate make(conduit::api api, uint32_t index, const char* name, uint32_t vendor, adapter_type type,
                      uint64_t vram_gib, uint64_t shared_gib = 0) {
    candidate c;
    c.adapter.api = api;
    c.adapter.index = index;
    c.adapter.name = name;
    c.adapter.vendor_id = vendor;
    c.adapter.device_id = 0x1234 + index;
    c.adapter.type = type;
    c.adapter.dedicated_vram = vram_gib << 30;
    c.adapter.shared_memory = shared_gib << 30;
    c.adapter.driver_version = 555;
    return c;
}

static void with_caps(candidate& c) {
    c.have_caps = true;
    c.caps.adapter = c.adapter;
    c.caps.features.multi_draw_indirect = true;
    c.caps.features.draw_indirect_count = true;
    c.caps.features.async_compute = true;
    c.caps.features.tearing = true;
    c.caps.limits.max_anisotropy = 16;
}

static probe machine(std::initializer_list<candidate> list) {
    probe p;
    p.candidates = list;
    return p;
}

static void test_desktop_nvidia() {
    candidate vk = make(api::vulkan, 0, "RTX 4080", vendor_id::nvidia, adapter_type::discrete, 16);
    candidate dx = make(api::d3d12, 0, "RTX 4080", vendor_id::nvidia, adapter_type::discrete, 16);
    candidate d11 = make(api::d3d11, 0, "RTX 4080", vendor_id::nvidia, adapter_type::discrete, 16);
    candidate igpu = make(api::d3d12, 1, "UHD", vendor_id::intel, adapter_type::integrated, 0, 16);
    with_caps(vk);
    with_caps(dx);
    const render_profile p = choose_profile(machine({d11, igpu, vk, dx}), {}, {}, true, false);
    CHECK(p.api == api::d3d12);
    CHECK(p.adapter_index == 0);
    CHECK(p.tier == tier::ultra);
    CHECK(p.gpu_culling && p.async_compute && p.frame_generation && p.tearing);
    CHECK(p.tier == tier::ultra && p.max_anisotropy == 16.0f);
    CHECK(!p.upscalers.empty() && p.upscalers.front() == upscaler::dlss);
    CHECK(p.upscalers.back() == upscaler::native);
    CHECK(p.fallbacks.size() == 4);
    CHECK(p.fallbacks[1].adapter.api == api::vulkan);
    CHECK(p.fallbacks[2].adapter.api == api::d3d11);
    CHECK(p.fallbacks.back().adapter.type == adapter_type::integrated);
}

static void test_linux_prefers_vulkan() {
    candidate vk = make(api::vulkan, 0, "RX 7800", vendor_id::amd, adapter_type::discrete, 16);
    with_caps(vk);
    const render_profile p = choose_profile(machine({vk}), {}, {}, false, false);
    CHECK(p.api == api::vulkan);
    CHECK(p.upscalers.front() == upscaler::fsr);
}

static void test_integrated_is_capped() {
    candidate igpu = make(api::d3d12, 0, "Radeon 780M", vendor_id::amd, adapter_type::integrated, 0, 32);
    with_caps(igpu);
    igpu.caps.features.async_compute = true;
    const render_profile p = choose_profile(machine({igpu}), {}, {}, true, false);
    CHECK(p.tier == tier::medium);
    CHECK(!p.async_compute);
    CHECK(!p.frame_generation);
}

static void test_software_last() {
    candidate cpu = make(api::vulkan, 0, "llvmpipe", 0x10005, adapter_type::cpu, 0, 8);
    candidate old = make(api::d3d11, 0, "GT 710", vendor_id::nvidia, adapter_type::discrete, 1);
    const render_profile p = choose_profile(machine({cpu, old}), {}, {}, true, false);
    CHECK(p.api == api::d3d11);
    CHECK(p.tier == tier::minimal);
    CHECK(p.fallbacks.back().adapter.type == adapter_type::cpu);
    const render_profile only = choose_profile(machine({cpu}), {}, {}, false, false);
    CHECK(only.api == api::vulkan && only.tier == tier::minimal);
}

static void test_user_override() {
    candidate vk = make(api::vulkan, 0, "RTX 4080", vendor_id::nvidia, adapter_type::discrete, 16);
    candidate dx = make(api::d3d12, 0, "RTX 4080", vendor_id::nvidia, adapter_type::discrete, 16);
    user_choice u;
    u.api = api::vulkan;
    render_profile p = choose_profile(machine({vk, dx}), u, {}, true, false);
    CHECK(p.api == api::vulkan);
    u.api = api::metal;
    p = choose_profile(machine({vk, dx}), u, {}, true, false);
    CHECK(p.api == api::d3d12);
    u = {};
    u.auto_tier = false;
    u.tier = tier::low;
    u.allow_frame_generation = false;
    p = choose_profile(machine({dx}), u, {}, true, false);
    CHECK(p.tier == tier::low && !p.frame_generation);
}

constexpr uint32_t vk12 = (1u << 22) | (2u << 12);
constexpr uint32_t vk13 = (1u << 22) | (3u << 12);

static void test_amd_prefers_vulkan_on_windows() {
    candidate vk = make(api::vulkan, 0, "RX 7800 XT", vendor_id::amd, adapter_type::discrete, 16);
    candidate dx = make(api::d3d12, 0, "RX 7800 XT", vendor_id::amd, adapter_type::discrete, 16);
    vk.adapter.api_version = vk13;
    vk.adapter.driver_name = "AMD proprietary driver";
    with_caps(vk);
    with_caps(dx);
    render_profile p = choose_profile(machine({dx, vk}), {}, {}, true, false);
    CHECK(p.api == api::vulkan);
    CHECK(p.fallbacks.size() == 2 && p.fallbacks[1].adapter.api == api::d3d12);

    vk.adapter.api_version = vk12;
    p = choose_profile(machine({dx, vk}), {}, {}, true, false);
    CHECK(p.api == api::d3d12);

    vk.adapter.api_version = vk13;
    std::vector<known_failure> failures;
    record_failure(failures, vk.adapter);
    record_failure(failures, vk.adapter);
    CHECK(choose_profile(machine({dx, vk}), {}, failures, true, false).api == api::d3d12);

    user_choice u;
    u.api = api::d3d12;
    CHECK(choose_profile(machine({dx, vk}), u, {}, true, false).api == api::d3d12);
}

static void test_nvidia_keeps_d3d12() {
    candidate vk = make(api::vulkan, 0, "RTX 4070", vendor_id::nvidia, adapter_type::discrete, 12);
    candidate dx = make(api::d3d12, 0, "RTX 4070", vendor_id::nvidia, adapter_type::discrete, 12);
    vk.adapter.api_version = vk13;
    with_caps(vk);
    with_caps(dx);
    CHECK(choose_profile(machine({vk, dx}), {}, {}, true, false).api == api::d3d12);
}

static void test_translation_layer_loses() {
    candidate vk =
        make(api::vulkan, 0, "Microsoft Direct3D12 (AMD Radeon RX 580)", vendor_id::amd, adapter_type::discrete, 8);
    candidate dx = make(api::d3d12, 0, "AMD Radeon RX 580", vendor_id::amd, adapter_type::discrete, 8);
    vk.adapter.api_version = vk13;
    vk.adapter.driver_name = "Dozen";
    with_caps(vk);
    with_caps(dx);
    CHECK(choose_profile(machine({vk, dx}), {}, {}, true, false).api == api::d3d12);
    CHECK(choose_profile(machine({vk}), {}, {}, true, false).api == api::vulkan);
}

static void test_amd_on_linux_stays_vulkan() {
    candidate vk = make(api::vulkan, 0, "RX 7800 XT", vendor_id::amd, adapter_type::discrete, 16);
    vk.adapter.api_version = vk13;
    vk.adapter.driver_name = "radv";
    with_caps(vk);
    CHECK(choose_profile(machine({vk}), {}, {}, false, false).api == api::vulkan);
}

static void test_failure_memory() {
    candidate vk = make(api::vulkan, 0, "GPU", vendor_id::nvidia, adapter_type::discrete, 8);
    candidate dx = make(api::d3d12, 0, "GPU", vendor_id::nvidia, adapter_type::discrete, 8);
    std::vector<known_failure> failures;
    record_failure(failures, dx.adapter);
    CHECK(choose_profile(machine({vk, dx}), {}, failures, true, false).api == api::d3d12);
    record_failure(failures, dx.adapter);
    CHECK(choose_profile(machine({vk, dx}), {}, failures, true, false).api == api::vulkan);
    dx.adapter.driver_version = 556;
    CHECK(choose_profile(machine({vk, dx}), {}, failures, true, false).api == api::d3d12);

    const std::vector<known_failure> round = parse_failures(serialize_failures(failures));
    CHECK(round.size() == 1);
    CHECK(round[0].api == api::d3d12 && round[0].count == 2 && round[0].driver_version == 555);
}

static void test_nothing_usable() {
    candidate bad = make(api::vulkan, 0, "ancient", 0x1, adapter_type::discrete, 1);
    bad.adapter.usable = false;
    const render_profile p = choose_profile(machine({bad}), {}, {}, false, false);
    CHECK(p.api == api::none);
    CHECK(p.fallbacks.empty());
}

static void test_real_probe() {
    probe_options o;
    o.read_caps = true;
    const conduit::probe probe = probe_system(o);
    for (const candidate& c : probe.candidates)
        std::printf("candidate %s %s score %.0f caps=%d\n", api_name(c.adapter.api), c.adapter.name.c_str(), c.score,
                    c.have_caps);
    const render_profile p = choose_profile(probe, {}, {}, false, false);
    std::printf("profile: %s tier %s\n", api_name(p.api), tier_name(p.tier));
    for (const std::string& r : p.reasons)
        std::printf("  %s\n", r.c_str());
    for (const candidate& c : probe.candidates)
        CHECK(c.adapter.api != api::none);
}

int main() {
    test_desktop_nvidia();
    test_amd_prefers_vulkan_on_windows();
    test_nvidia_keeps_d3d12();
    test_translation_layer_loses();
    test_amd_on_linux_stays_vulkan();
    test_linux_prefers_vulkan();
    test_integrated_is_capped();
    test_software_last();
    test_user_override();
    test_failure_memory();
    test_nothing_usable();
    test_real_probe();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::puts("render_profile_test ok");
    return 0;
}
