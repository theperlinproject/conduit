#include <conduit/render_profile.hpp>

#include <algorithm>
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
    const system_info system = query_system();
    CHECK(system.logical_cores > 0);
    CHECK(system.performance_cores > 0 && system.performance_cores <= system.logical_cores);
    CHECK(system.os != os_kind::unknown);
    const render_profile p = choose_profile(probe, {}, {}, system);
    std::printf("%s", p.explain().c_str());
    for (const candidate& c : probe.candidates)
        CHECK(c.adapter.api != api::none);
    const bool any_usable = std::any_of(probe.candidates.begin(), probe.candidates.end(),
                                        [](const candidate& c) { return c.adapter.usable; });
    if (!any_usable) {
        CHECK(p.api == api::none && p.fallbacks.empty());
    } else if (p.api != api::none) {
        CHECK(p.device().adapter_index == p.adapter_index);
        CHECK(!p.fallbacks.empty() && !hardware_key(p.fallbacks.front().adapter, system).empty());
    }
}

static system_info desktop(uint32_t cores, uint32_t fast) {
    system_info s;
    s.os = os_kind::windows;
    s.logical_cores = cores;
    s.performance_cores = fast;
    s.memory_bytes = 32ull << 30;
    return s;
}

static void test_workloads_differ() {
    candidate dx = make(api::d3d12, 0, "RTX 4080", vendor_id::nvidia, adapter_type::discrete, 16);
    with_caps(dx);
    dx.caps.features.gpu_driven = true;
    dx.caps.features.bindless = true;
    const system_info host = desktop(24, 16);

    user_choice u;
    u.workload = workload::draw_heavy;
    u.expected_draws = 20000;
    const render_profile heavy = choose_profile(machine({dx}), u, {}, host);
    CHECK(heavy.gpu_culling && heavy.gpu_driven && heavy.multi_draw_indirect && heavy.bindless);
    CHECK(heavy.record_threads == 8);
    CHECK(heavy.frames_in_flight == 3);
    CHECK(heavy.async_pipelines && heavy.pipeline_threads == 4);

    // a few hundred draws: a culling pass and a second list would both cost more than they save
    u.expected_draws = 300;
    const render_profile light = choose_profile(machine({dx}), u, {}, host);
    CHECK(!light.gpu_culling && !light.gpu_driven);
    CHECK(light.record_threads == 1);
    CHECK(light.find("gpu culling") && light.find("gpu culling")->state == render_profile::verdict::unused);

    u = {};
    u.workload = workload::low_latency;
    const render_profile fast = choose_profile(machine({dx}), u, {}, host);
    CHECK(fast.frames_in_flight == 2 && !fast.frame_generation);
    CHECK(fast.find("frame generation")->state == render_profile::verdict::unused);

    u.workload = workload::efficiency;
    const render_profile quiet = choose_profile(machine({dx}), u, {}, host);
    CHECK(!quiet.tearing && !quiet.async_compute && quiet.record_threads <= 2 && quiet.frames_in_flight == 2);

    // 4k on a mid-range card: render lower and upscale; the same card at 1080p stays native
    candidate mid = make(api::d3d12, 0, "RTX 3060", vendor_id::nvidia, adapter_type::discrete, 8);
    with_caps(mid);
    u = {};
    u.output_width = 3840;
    u.output_height = 2160;
    const render_profile uhd = choose_profile(machine({mid}), u, {}, host);
    CHECK(uhd.render_scale < 1.0f && uhd.render_scale >= 0.5f);
    CHECK(uhd.find("upscaling")->state == render_profile::verdict::enabled);
    u.output_width = 1920;
    u.output_height = 1080;
    CHECK(choose_profile(machine({mid}), u, {}, host).render_scale == 1.0f);
}

static void test_host_shapes_threads() {
    candidate vk = make(api::vulkan, 0, "RX 7800", vendor_id::amd, adapter_type::discrete, 16);
    with_caps(vk);
    user_choice u;
    CHECK(choose_profile(machine({vk}), u, {}, desktop(4, 4)).record_threads == 3);
    CHECK(choose_profile(machine({vk}), u, {}, desktop(2, 2)).record_threads == 1);
    // unknown host: no guess
    CHECK(choose_profile(machine({vk}), u, {}, false, false).record_threads == 1);

    // a phone: few lists, shallow queue, no second compute queue
    candidate phone = make(api::vulkan, 0, "Adreno 740", vendor_id::qualcomm, adapter_type::integrated, 0, 8);
    with_caps(phone);
    phone.caps.features.unified_memory = true;
    system_info android;
    android.os = os_kind::android;
    android.logical_cores = 8;
    android.performance_cores = 1;
    const render_profile mobile = choose_profile(machine({phone}), u, {}, android);
    CHECK(mobile.record_threads == 1 && mobile.frames_in_flight == 2 && !mobile.async_compute);
    CHECK(mobile.upload_ring_bytes <= (16ull << 20));
    CHECK(mobile.memory_budget_bytes == (4ull << 30));

    // a browser encodes on one thread whatever the machine has
    candidate web = make(api::webgpu, 0, "WebGPU", vendor_id::nvidia, adapter_type::discrete, 8);
    system_info browser = desktop(16, 16);
    browser.os = os_kind::web;
    const render_profile page = choose_profile(machine({web}), u, {}, browser);
    CHECK(page.record_threads == 1 && !page.async_pipelines && page.pipeline_threads == 1);
    CHECK(page.find("parallel recording")->state == render_profile::verdict::unavailable);
}

static void test_explained() {
    candidate vk = make(api::vulkan, 0, "RTX 4070", vendor_id::nvidia, adapter_type::discrete, 12);
    candidate dx = make(api::d3d12, 0, "RTX 4070", vendor_id::nvidia, adapter_type::discrete, 12);
    with_caps(vk);
    with_caps(dx);
    dx.caps.features.mesh_shaders = true;
    render_profile p = choose_profile(machine({vk, dx}), {}, {}, desktop(16, 8));
    CHECK(p.api == api::d3d12);
    // the terms of a score add up to it
    for (const candidate& c : p.fallbacks) {
        double sum = 0;
        for (const score_term& t : c.terms)
            sum += t.points;
        CHECK(sum > c.score - 1e-6 && sum < c.score + 1e-6);
        CHECK(!c.terms.empty());
    }
    CHECK(p.find("mesh shaders") && p.find("mesh shaders")->state == render_profile::verdict::unused);
    for (const render_profile::decision& d : p.decisions)
        CHECK(!d.reason.empty());
    const std::string text = p.explain();
    CHECK(text.contains("Direct3D 12") && text.contains("decisions:") && text.contains("discrete gpu"));

    // the application can overrule the heuristics, and the override shows in the score
    user_choice u;
    u.bias.push_back({api::vulkan, 50.0});
    p = choose_profile(machine({vk, dx}), u, {}, desktop(16, 8));
    CHECK(p.api == api::vulkan);
    CHECK(p.explain().contains("bias set by the application"));

    // same machine, same answer
    const render_profile again = choose_profile(machine({vk, dx}), u, {}, desktop(16, 8));
    CHECK(again.explain() == p.explain());

    system_info a = desktop(16, 8), b = desktop(8, 8);
    CHECK(hardware_key(vk.adapter, a) != hardware_key(vk.adapter, b));
    CHECK(hardware_key(vk.adapter, a) != hardware_key(dx.adapter, a));
    candidate newer = vk;
    newer.adapter.driver_version = 556;
    CHECK(hardware_key(vk.adapter, a) != hardware_key(newer.adapter, a));
}

static void test_apple_silicon() {
    system_info mac;
    mac.os = os_kind::macos;
    mac.logical_cores = 10;
    mac.performance_cores = 4;
    // metal has no multi draw indirect call; that is not a sign of a weak gpu
    candidate m4 = make(api::metal, 0, "Apple M4", vendor_id::apple, adapter_type::integrated, 0, 12);
    with_caps(m4);
    m4.caps.features.multi_draw_indirect = m4.caps.features.draw_indirect_count = false;
    m4.caps.features.async_compute = false;
    m4.caps.features.unified_memory = true;
    const render_profile base = choose_profile(machine({m4}), {}, {}, mac);
    CHECK(base.api == api::metal && base.tier == tier::medium);
    CHECK(base.render_scale == 1.0f);
    CHECK(base.record_threads == 3 && base.frames_in_flight == 2);
    CHECK(!base.gpu_culling && !base.async_compute && !base.frame_generation);

    candidate max = make(api::metal, 0, "Apple M4 Max", vendor_id::apple, adapter_type::integrated, 0, 48);
    with_caps(max);
    max.caps.features.multi_draw_indirect = false;
    CHECK(choose_profile(machine({max}), {}, {}, mac).tier == tier::high);

    // vulkan on a mac is moltenvk: metal underneath, so metal directly
    candidate mvk = make(api::vulkan, 0, "Apple M4", vendor_id::apple, adapter_type::integrated, 0, 12);
    mvk.adapter.driver_name = "MoltenVK";
    with_caps(mvk);
    CHECK(choose_profile(machine({mvk, m4}), {}, {}, mac).api == api::metal);
}

static void test_tuner() {
    render_profile p;
    p.record_threads = 2;
    tuner_options o;
    o.target_ms = 16.0f;
    o.settle_frames = 30;
    o.cooldown_frames = 100;
    o.max_record_threads = 6;
    adaptive_tuner t(p, o);

    // comfortably inside the budget: nothing moves
    int changes = 0;
    for (int i = 0; i < 2000; i++)
        changes += t.update({6.0f, 9.0f});
    CHECK(changes == 0 && t.render_scale() == 1.0f && t.record_threads() == 2);
    CHECK(t.state() == bottleneck::balanced);

    // one hitch is not a trend
    changes += t.update({60.0f, 80.0f});
    for (int i = 0; i < 200; i++)
        changes += t.update({6.0f, 9.0f});
    CHECK(changes == 0);

    // gpu bound: the scale steps down, and never twice inside a cooldown
    int last_change = -1000, closest = 1000;
    for (int i = 0; i < 1000; i++)
        if (t.update({6.0f, 24.0f})) {
            closest = (std::min)(closest, i - last_change);
            last_change = i;
            changes++;
        }
    CHECK(t.state() == bottleneck::gpu);
    CHECK(t.render_scale() < 1.0f && t.render_scale() >= o.min_render_scale);
    CHECK(changes >= 2 && closest > 100);
    CHECK(t.record_threads() == 2);

    // a scene sitting on the edge of the margin does not flip settings back and forth
    for (int i = 0; i < 1000; i++)
        t.update({6.0f, i % 2 ? 13.4f : 13.8f});
    const float held = t.render_scale();
    changes = 0;
    for (int i = 0; i < 3000; i++)
        changes += t.update({6.0f, i % 2 ? 13.4f : 13.8f});
    CHECK(changes == 0 && t.render_scale() == held);

    // cpu bound: another recording thread, up to the limit
    for (int i = 0; i < 4000; i++)
        t.update({22.0f, 8.0f});
    CHECK(t.state() == bottleneck::cpu);
    CHECK(t.record_threads() > 2 && t.record_threads() <= 6);

    // the scale stays where the application pinned it
    tuner_options fixed = o;
    fixed.allow_render_scale = false;
    adaptive_tuner pinned(p, fixed);
    for (int i = 0; i < 2000; i++)
        pinned.update({6.0f, 30.0f});
    CHECK(pinned.render_scale() == 1.0f && pinned.state() == bottleneck::gpu);
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
    test_workloads_differ();
    test_host_shapes_threads();
    test_explained();
    test_apple_silicon();
    test_tuner();
    test_real_probe();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::puts("render_profile_test ok");
    return 0;
}
