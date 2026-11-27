#include <conduit/render_profile.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <bit>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#elif !defined(__EMSCRIPTEN__)
#include <unistd.h>
#endif

namespace conduit {
    const char* tier_name(conduit::tier tier) {
        switch (tier) {
        case conduit::tier::minimal:
            return "minimal";
        case conduit::tier::low:
            return "low";
        case conduit::tier::medium:
            return "medium";
        case conduit::tier::high:
            return "high";
        case conduit::tier::ultra:
            return "ultra";
        }
        return "medium";
    }

    const char* upscaler_name(upscaler u) {
        switch (u) {
        case upscaler::native:
            return "native";
        case upscaler::spatial:
            return "spatial";
        case upscaler::fsr:
            return "fsr";
        case upscaler::xess:
            return "xess";
        case upscaler::dlss:
            return "dlss";
        }
        return "native";
    }

    namespace {
        constexpr uint64_t bytes_per_gib = 1ull << 30;

        uint64_t usable_memory(const adapter_info& a) {
            return a.type == adapter_type::discrete ? a.dedicated_vram
                                                    : (std::max)(a.dedicated_vram, a.shared_memory / 2);
        }

        bool is_software(const adapter_info& a) {
            return a.type == adapter_type::cpu || a.type == adapter_type::virtual_gpu;
        }

        constexpr uint32_t vulkan13 = (1u << 22) | (3u << 12);

        bool contains(const std::string& haystack, const char* needle) {
            std::string h = haystack;
            std::transform(h.begin(), h.end(), h.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            return h.contains(needle);
        }

        bool is_translation_layer(const adapter_info& a) {
            return a.api == api::vulkan && (contains(a.driver_name, "dozen") || contains(a.driver_name, "moltenvk") ||
                                            contains(a.driver_name, "gfxstream") || contains(a.driver_name, "venus") ||
                                            contains(a.name, "d3d12") || contains(a.name, "microsoft direct3d"));
        }

        double driver_adjust(const adapter_info& a, bool windows, bool apple, std::string* note) {
            auto say = [&](const char* text) {
                if (note)
                    *note = text;
            };
            if (is_translation_layer(a)) {
                say("vulkan here is a translation layer over another api: native api preferred");
                return -300.0;
            }
            if (!windows || apple || a.type == adapter_type::cpu || a.type == adapter_type::virtual_gpu)
                return 0.0;
            const bool recent_vulkan = a.api == api::vulkan && a.api_version >= vulkan13;
            const bool old_vulkan = a.api == api::vulkan && a.api_version != 0 && a.api_version < vulkan13;
            switch (a.vendor_id) {
            case vendor_id::amd:
                if (recent_vulkan) {
                    say("amd gpu with a current vulkan driver: vulkan preferred");
                    return 35.0;
                }
                if (old_vulkan) {
                    say("amd vulkan driver is old (below 1.3): direct3d 12 preferred");
                    return -40.0;
                }
                return 0.0;
            case vendor_id::nvidia:
                if (old_vulkan) {
                    say("nvidia vulkan driver is old (below 1.3): direct3d 12 preferred");
                    return -20.0;
                }
                return 0.0;
            case vendor_id::intel:
                if (old_vulkan) {
                    say("intel vulkan driver is old (below 1.3): direct3d 12 preferred");
                    return -40.0;
                }
                return 0.0;
            default:
                return 0.0;
            }
        }

        double api_base(conduit::api api, bool windows, bool apple) {
            switch (api) {
            case conduit::api::vulkan:
                return windows ? 100.0 : 110.0;
            case conduit::api::d3d12:
                return windows ? 110.0 : 0.0;
            case conduit::api::metal:
                return apple ? 120.0 : 0.0;
            case conduit::api::d3d11:
                return 40.0;
            case conduit::api::webgpu:
                return 30.0;
            case conduit::api::gles:
                return 5.0;
            default:
                return 0.0;
            }
        }
    }

    namespace {
        // the score and, when asked, each term of it. kept in one place so the number and its explanation
        // cannot drift apart
        double score_terms(const candidate& c, bool windows, bool apple, std::vector<score_term>* terms) {
            const adapter_info& a = c.adapter;
            if (!a.usable) {
                if (terms)
                    terms->push_back({-1e9, a.unusable_reason.empty() ? "adapter is not usable" : a.unusable_reason});
                return -1e9;
            }
            double s = 0.0;
            const auto add = [&](double points, const char* reason) {
                if (points == 0.0)
                    return;
                s += points;
                if (terms)
                    terms->push_back({points, reason});
            };
            const double base = api_base(a.api, windows, apple);
            if (base <= 0) {
                if (terms)
                    terms->push_back({-1e9, "this api does not run on this platform"});
                return -1e9;
            }
            add(base, "api preference on this platform");
            // OpenGL ES only wins when nothing else works at all, whatever the adapter bonuses add up to
            if (a.api == api::gles)
                add(-2000.0, "opengl es is the last resort");
            switch (a.type) {
            case adapter_type::discrete:
                add(1000, "discrete gpu");
                break;
            case adapter_type::integrated:
                add(300, "integrated gpu");
                break;
            case adapter_type::unknown:
                add(200, "gpu of unknown type");
                break;
            case adapter_type::virtual_gpu:
                add(-400, "virtual gpu");
                break;
            case adapter_type::cpu:
                add(-800, "software rasteriser");
                break;
            }
            const double gib = static_cast<double>(usable_memory(a)) / static_cast<double>(bytes_per_gib);
            add(40.0 * std::log2(1.0 + gib), "usable gpu memory");
            if (c.have_caps) {
                const device_features& f = c.caps.features;
                add(f.draw_indirect_count ? 25 : 0, "gpu-side draw count");
                add(f.async_compute ? 10 : 0, "async compute queue");
                add(f.timestamp_queries ? 3 : 0, "timestamp queries");
                add(f.mesh_shaders ? 8 : 0, "mesh shaders");
                add(f.ray_tracing ? 8 : 0, "ray tracing");
                add(f.bindless ? 6 : 0, "bindless resources");
                add(f.enhanced_barriers ? 4 : 0, "enhanced barriers");
                // metal has no such call and does not need one: its indirect draws are cheap to issue in a loop
                if (!f.multi_draw_indirect && a.api != api::metal)
                    add(-60, "no multi draw indirect");
            }
            if (windows && a.vendor_id == vendor_id::intel && a.type == adapter_type::integrated &&
                a.api == api::vulkan)
                add(-30, "intel integrated gpu on windows: direct3d preferred over vulkan");
            std::string note;
            const double driver = driver_adjust(a, windows, apple, &note);
            add(driver, note.c_str());
            return s;
        }
    }

    double score_candidate(const candidate& c, bool windows, bool apple) {
        return score_terms(c, windows, apple, nullptr);
    }

    tier tier_for(const candidate& c) {
        const adapter_info& a = c.adapter;
        if (is_software(a))
            return tier::minimal;
        const double gib = static_cast<double>(usable_memory(a)) / static_cast<double>(bytes_per_gib);
        int level;
        if (gib < 1.5)
            level = 0;
        else if (gib < 3.0)
            level = 1;
        else if (gib < 6.0)
            level = 2;
        else if (gib < 10.0)
            level = 3;
        else
            level = 4;
        // an integrated gpu shares its memory and its power budget; the larger apple chips are the exception,
        // and the memory they are sold with is the only reliable sign of one
        const bool large_apple = a.api == api::metal && a.vendor_id == vendor_id::apple && gib >= 10.0;
        if (a.type == adapter_type::integrated)
            level = (std::min)(level, large_apple ? 3 : 2);
        if (a.api == api::gles)
            level = (std::min)(level, 1);
        if (c.have_caps) {
            const device_features& f = c.caps.features;
            // elsewhere a driver without multi draw indirect is an old one
            if (!f.multi_draw_indirect && a.api != api::metal)
                level = (std::min)(level, 1);
            if (a.type == adapter_type::discrete && level >= 3 && (f.mesh_shaders || f.cooperative_vector))
                level = 4;
        }
        return static_cast<tier>(level);
    }

    void probe::sort() {
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const candidate& a, const candidate& b) { return a.score > b.score; });
    }

    conduit::probe probe_system(const probe_options& options) {
        conduit::probe probe;
#if defined(_WIN32)
        constexpr bool windows = true;
#else
        constexpr bool windows = false;
#endif
#if defined(__APPLE__)
        constexpr bool apple = true;
#else
        constexpr bool apple = false;
#endif
        std::vector<conduit::api> apis = options.apis;
        if (apis.empty())
            apis = compiled_apis();
        for (conduit::api api : apis) {
            if (api == conduit::api::none)
                continue;
            instance_options io;
            io.validation = options.validation;
            std::unique_ptr<conduit::instance> instance = create_instance(api, io);
            if (!instance)
                continue;
            const size_t first = probe.candidates.size();
            for (const adapter_info& adapter : instance->adapters()) {
                if (!options.allow_software && is_software(adapter))
                    continue;
                candidate c;
                c.adapter = adapter;
                c.score = score_candidate(c, windows, apple);
                probe.candidates.push_back(std::move(c));
            }
            if (!options.read_caps || probe.candidates.size() == first)
                continue;
            size_t best = first;
            for (size_t i = first; i < probe.candidates.size(); i++)
                if (probe.candidates[i].score > probe.candidates[best].score)
                    best = i;
            if (probe.candidates[best].score < -1e8)
                continue;
            device_options dopts;
            dopts.adapter_index = probe.candidates[best].adapter.index;
            dopts.validation = false;
            if (std::unique_ptr<conduit::device> device = instance->create_device(dopts)) {
                probe.candidates[best].caps = device->caps();
                probe.candidates[best].have_caps = true;
                probe.candidates[best].score = score_candidate(probe.candidates[best], windows, apple);
            } else {
                probe.candidates[best].notes.emplace_back("device creation failed during probe");
                probe.candidates[best].adapter.usable = false;
                probe.candidates[best].score = -1e9;
            }
        }
        probe.sort();
        return probe;
    }

    namespace {
        bool failed(const std::vector<known_failure>& failures, const adapter_info& a) {
            for (const known_failure& f : failures)
                if (f.api == a.api && f.vendor_id == a.vendor_id && f.device_id == a.device_id &&
                    f.driver_version == a.driver_version && f.count >= 2)
                    return true;
            return false;
        }
    }

    const char* os_name(os_kind os) {
        switch (os) {
        case os_kind::windows:
            return "windows";
        case os_kind::linux_desktop:
            return "linux";
        case os_kind::macos:
            return "macos";
        case os_kind::ios:
            return "ios";
        case os_kind::android:
            return "android";
        case os_kind::web:
            return "web";
        default:
            return "unknown";
        }
    }

    const char* workload_name(workload w) {
        switch (w) {
        case workload::balanced:
            return "balanced";
        case workload::draw_heavy:
            return "draw heavy";
        case workload::gpu_heavy:
            return "gpu heavy";
        case workload::texture_heavy:
            return "texture heavy";
        case workload::compute_heavy:
            return "compute heavy";
        case workload::low_latency:
            return "low latency";
        case workload::efficiency:
            return "efficiency";
        }
        return "balanced";
    }

    const char* bottleneck_name(bottleneck b) {
        switch (b) {
        case bottleneck::cpu:
            return "cpu";
        case bottleneck::gpu:
            return "gpu";
        case bottleneck::balanced:
            return "balanced";
        default:
            return "unknown";
        }
    }

    const render_profile::decision* render_profile::find(std::string_view feature) const {
        for (const decision& d : decisions)
            if (d.feature == feature)
                return &d;
        return nullptr;
    }

    device_options render_profile::device() const {
        device_options o;
        o.adapter_index = adapter_index;
        o.frames_in_flight = frames_in_flight;
        o.allow_async_compute = async_compute;
        o.async_pipelines = async_pipelines;
        return o;
    }

    namespace {
        // whether the backend can record several lists at once and gain from it. d3d11 and opengl es replay
        // on one context, and a browser encodes on its one thread
        bool records_in_parallel(api api, const system_info& system) {
            if (system.browser())
                return false;
            return api == api::vulkan || api == api::d3d12 || api == api::metal;
        }

        bool compiles_pipelines_off_thread(api api) {
            return api == api::vulkan || api == api::d3d12 || api == api::metal;
        }

        // pixels a tier shades comfortably at native resolution: 720p, 1080p, 1080p, 1440p, 4k
        constexpr std::array<double, 5> tier_pixels{0.92e6, 2.07e6, 2.07e6, 3.69e6, 8.29e6};
    }

    render_profile choose_profile(const conduit::probe& probe, const user_choice& user,
                                  const std::vector<known_failure>& failures, bool windows, bool apple) {
        system_info system;
        system.os = windows ? os_kind::windows : apple ? os_kind::macos : os_kind::unknown;
        return choose_profile(probe, user, failures, system);
    }

    render_profile choose_profile(const conduit::probe& probe, const user_choice& user,
                                  const std::vector<known_failure>& failures, const system_info& system) {
        const bool windows = system.os == os_kind::windows;
        const bool apple = system.os == os_kind::macos || system.os == os_kind::ios;
        render_profile p;
        p.workload = user.workload;
        p.system = system;
        std::vector<candidate> pool;
        for (candidate c : probe.candidates) {
            c.terms.clear();
            c.score = score_terms(c, windows, apple, &c.terms);
            if (c.score < -1e8)
                continue;
            std::string driver_note;
            if (driver_adjust(c.adapter, windows, apple, &driver_note) != 0.0 && !driver_note.empty())
                c.notes.push_back(driver_note);
            for (const user_choice::api_bias& b : user.bias)
                if (b.api == c.adapter.api && b.points != 0.0) {
                    c.score += b.points;
                    c.terms.push_back({b.points, "bias set by the application"});
                }
            if (failed(failures, c.adapter)) {
                c.score -= 5000;
                c.terms.push_back({-5000, "failed on a previous launch with this driver"});
                c.notes.emplace_back("failed on a previous launch with this driver");
            }
            pool.push_back(std::move(c));
        }
        std::stable_sort(pool.begin(), pool.end(),
                         [](const candidate& a, const candidate& b) { return a.score > b.score; });

        if (user.api != api::none || user.adapter_index >= 0) {
            auto matches = [&](const candidate& c) {
                return (user.api == api::none || c.adapter.api == user.api) &&
                       (user.adapter_index < 0 || static_cast<int>(c.adapter.index) == user.adapter_index);
            };
            auto it = std::stable_partition(pool.begin(), pool.end(), matches);
            if (it != pool.begin())
                p.reasons.emplace_back("renderer chosen by the user");
            else
                p.reasons.emplace_back("the requested renderer is not available here; picked automatically");
        }
        if (pool.empty()) {
            p.reasons.emplace_back("no usable graphics adapter found");
            return p;
        }

        const candidate& best = pool.front();
        const adapter_info& adapter = best.adapter;
        p.api = adapter.api;
        p.adapter_index = adapter.index;
        p.adapter_name = adapter.name;
        p.fallbacks = pool;
        p.reasons.push_back(std::string("using ") + api_name(p.api) + " on " + adapter.name);
        for (const std::string& n : best.notes)
            p.reasons.push_back(n);

        p.tier = user.auto_tier ? tier_for(best) : user.tier;
        if (is_software(adapter))
            p.reasons.emplace_back("software rasteriser: expect low frame rates");

        const bool have_caps = best.have_caps;
        const device_features f = have_caps ? best.caps.features : device_features{};
        const bool discrete = adapter.type == adapter_type::discrete;
        const bool unified = have_caps ? f.unified_memory : adapter.type == adapter_type::integrated;
        const workload load = user.workload;
        const bool constrained = system.mobile() || system.browser() || load == workload::efficiency;

        const auto decide = [&](const char* feature, bool available, bool wanted, std::string reason) {
            const render_profile::verdict v = !available ? render_profile::verdict::unavailable
                                              : wanted   ? render_profile::verdict::enabled
                                                         : render_profile::verdict::unused;
            p.decisions.push_back({feature, v, std::move(reason)});
            return available && wanted;
        };

        static constexpr std::array<float, 5> tier_anisotropy{1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
        const float anisotropy = tier_anisotropy[static_cast<size_t>(p.tier)];
        p.max_anisotropy =
            have_caps ? (std::min)(anisotropy, (std::max)(1.0f, best.caps.limits.max_anisotropy)) : anisotropy;
        p.msaa = 1;

        // frames in flight: a third frame keeps a fast gpu fed, but every queued frame is a frame of latency
        // and a set of per-frame memory
        if (load == workload::low_latency) {
            p.frames_in_flight = 2;
            p.decisions.push_back({"frames in flight", render_profile::verdict::enabled,
                                   "2: low latency asked for, a third queued frame would add a frame of lag"});
        } else if (p.tier >= tier::medium && discrete && !constrained) {
            p.frames_in_flight = 3;
            p.decisions.push_back({"frames in flight", render_profile::verdict::enabled,
                                   "3: a discrete gpu is kept busy while the cpu prepares the next frame"});
        } else {
            p.frames_in_flight = 2;
            p.decisions.push_back({"frames in flight", render_profile::verdict::enabled,
                                   "2: shared memory or a modest gpu gains nothing from a deeper queue"});
        }

        // gpu-driven rendering: culling in compute pays for its pass once there are enough draws to cull
        const bool can_gpu_drive = have_caps && (f.gpu_driven || (f.multi_draw_indirect && f.draw_indirect_count));
        p.gpu_culling_min_draws = constrained ? 4096 : 1024;
        if (!have_caps) {
            p.gpu_culling = p.tier >= tier::medium;
            p.decisions.push_back({"gpu culling",
                                   p.gpu_culling ? render_profile::verdict::enabled : render_profile::verdict::unused,
                                   "capabilities were not read: decided from the tier alone"});
        } else {
            const bool few_draws = user.expected_draws != 0 && user.expected_draws < p.gpu_culling_min_draws;
            p.gpu_culling = decide("gpu culling", can_gpu_drive, !few_draws,
                                   !can_gpu_drive ? "needs multi draw indirect and a gpu-side draw count"
                                   : few_draws ? "too few draws for a culling pass to pay for itself: cull on the cpu"
                                               : "compute culls and writes the indirect arguments");
        }
        p.gpu_driven = p.gpu_culling && can_gpu_drive;
        p.multi_draw_indirect =
            decide("multi draw indirect", have_caps && f.multi_draw_indirect, true,
                   have_caps && f.multi_draw_indirect ? "one call issues a whole batch of draws"
                                                      : "not supported: indirect draws are issued one by one");

        // async compute shares memory bandwidth with rendering on unified memory, so it only helps where the
        // gpu has its own
        p.async_compute =
            decide("async compute", have_caps && f.async_compute, discrete && !constrained,
                   !(have_caps && f.async_compute) ? "no separate compute queue"
                   : !discrete   ? "the gpu shares memory with the cpu: a second queue only competes for bandwidth"
                   : constrained ? "not worth a second queue for this workload"
                                 : "compute runs beside rendering on its own queue");

        p.bindless = decide("bindless", have_caps && f.bindless,
                            p.tier >= tier::medium && (load == workload::draw_heavy ||
                                                       load == workload::texture_heavy || load == workload::balanced),
                            !(have_caps && f.bindless) ? "not supported: resources are bound per group"
                            : p.tier < tier::medium ? "modest gpu: conventional binding keeps descriptor memory small"
                                                    : "materials index one table instead of rebinding groups");

        const bool tear_wanted = load != workload::efficiency;
        p.tearing = decide("tearing", have_caps && f.tearing, tear_wanted,
                           !(have_caps && f.tearing) ? "the display path cannot present unsynchronised"
                           : tear_wanted             ? "available for uncapped presentation"
                                                     : "left off: presenting faster than the display wastes power");
        p.hdr = decide("hdr", have_caps && f.hdr_output, user.allow_hdr,
                       !(have_caps && f.hdr_output) ? "no hdr output"
                       : user.allow_hdr             ? "allowed by the user"
                                                    : "not asked for");

        // recording threads: the fastest cores minus the one that runs the game, and never more lists than
        // the frame has draws to fill
        const bool parallel = records_in_parallel(p.api, system);
        uint32_t threads = 1;
        const uint32_t fast_cores = system.performance_cores ? system.performance_cores : system.logical_cores;
        if (parallel && fast_cores > 2) {
            threads = (std::min)(fast_cores - 1, constrained ? 2u : 8u);
            if (load == workload::gpu_heavy || load == workload::texture_heavy)
                threads = (std::min)(threads, 2u);
            p.min_draws_per_list = 512;
            if (user.expected_draws)
                threads = std::clamp(user.expected_draws / p.min_draws_per_list, 1u, threads);
        }
        p.record_threads = threads;
        p.decisions.push_back({"parallel recording",
                               threads > 1 ? render_profile::verdict::enabled
                               : parallel  ? render_profile::verdict::unused
                                           : render_profile::verdict::unavailable,
                               !parallel         ? "this api records on one thread"
                               : fast_cores == 0 ? "core count unknown: one recording thread"
                               : threads > 1     ? std::to_string(threads) + " lists at once, at least " +
                                                   std::to_string(p.min_draws_per_list) + " draws each"
                                             : "too few cores or draws for a second list to pay for itself"});

        const bool off_thread = compiles_pipelines_off_thread(p.api);
        p.async_pipelines =
            decide("async pipelines", off_thread, true,
                   off_thread ? "pipelines compile off the render thread; a draw is skipped rather than stalled"
                              : "this api compiles pipelines where it is asked to");
        p.pipeline_threads = !off_thread || system.logical_cores < 4
                                 ? 1
                                 : std::clamp(system.logical_cores / 2, 1u, constrained ? 2u : 4u);

        // memory: stay under what the gpu can hold without paging, and size the upload ring to the tier
        const uint64_t usable = usable_memory(adapter);
        if (usable)
            p.memory_budget_bytes = unified ? usable : usable / 5 * 4;
        static constexpr std::array<uint64_t, 5> tier_ring{4ull << 20, 8ull << 20, 16ull << 20, 32ull << 20,
                                                           64ull << 20};
        p.upload_ring_bytes = tier_ring[static_cast<size_t>(p.tier)];
        if (load == workload::texture_heavy)
            p.upload_ring_bytes *= 2;
        if (constrained)
            p.upload_ring_bytes = (std::min)(p.upload_ring_bytes, uint64_t{16} << 20);
        if (p.memory_budget_bytes)
            p.upload_ring_bytes =
                (std::min)(p.upload_ring_bytes, (std::max)(p.memory_budget_bytes / 32, uint64_t{4} << 20));

        if (user.allow_upscaling && p.tier >= tier::low) {
            if (adapter.vendor_id == vendor_id::nvidia && p.api != api::d3d11 && p.api != api::gles)
                p.upscalers.push_back(upscaler::dlss);
            if (adapter.vendor_id == vendor_id::intel && discrete)
                p.upscalers.push_back(upscaler::xess);
            if (p.api == api::vulkan || p.api == api::d3d12)
                p.upscalers.push_back(upscaler::fsr);
            p.upscalers.push_back(upscaler::spatial);
        }
        p.upscalers.push_back(upscaler::native);
        const bool can_upscale = p.upscalers.front() != upscaler::native;

        // frame generation raises the presented rate, not the simulated one, and adds latency: it suits a
        // strong gpu that is held back by the cpu and never a latency-sensitive title
        const bool can_generate = p.tier >= tier::high && discrete && (p.api == api::d3d12 || p.api == api::vulkan);
        p.frame_generation =
            decide("frame generation", can_generate,
                   user.allow_frame_generation && load != workload::low_latency && load != workload::efficiency,
                   !can_generate                   ? "needs a strong discrete gpu on direct3d 12 or vulkan"
                   : !user.allow_frame_generation  ? "turned off by the user"
                   : load == workload::low_latency ? "left off: generated frames add latency"
                   : load == workload::efficiency  ? "left off: it spends power on frames nobody simulated"
                                                   : "available on this gpu");
        p.neural_features = have_caps && (f.cooperative_vector || f.cooperative_matrix);

        p.render_scale = 1.0f;
        const double pixels = static_cast<double>(user.output_width) * user.output_height;
        if (pixels > 0) {
            double budget = tier_pixels[static_cast<size_t>(p.tier)];
            if (load == workload::gpu_heavy || constrained)
                budget *= 0.75;
            if (pixels > budget && can_upscale) {
                p.render_scale = std::clamp(static_cast<float>(std::sqrt(budget / pixels)), 0.5f, 1.0f);
                p.render_scale = std::round(p.render_scale * 20.0f) / 20.0f;
            }
            p.decisions.push_back(
                {"upscaling",
                 !can_upscale            ? render_profile::verdict::unavailable
                 : p.render_scale < 1.0f ? render_profile::verdict::enabled
                                         : render_profile::verdict::unused,
                 !can_upscale ? "no upscaler for this gpu and api"
                 : p.render_scale < 1.0f
                     ? std::string("the output has more pixels than this gpu shades comfortably: render at ") +
                           std::to_string(static_cast<int>(p.render_scale * 100.0f + 0.5f)) + "% and upscale with " +
                           upscaler_name(p.upscalers.front())
                     : "the gpu shades this resolution natively"});
        } else if (p.tier <= tier::low && can_upscale) {
            p.render_scale = 0.75f;
            p.reasons.emplace_back("modest gpu: an upscaler at 75% would help (off until turned on in the options)");
        }

        // what the hardware has that nothing here uses, so it is visible rather than silently ignored
        if (have_caps) {
            const auto unused = [&](bool has, const char* feature, const char* reason) {
                if (has)
                    p.decisions.push_back({feature, render_profile::verdict::unused, reason});
            };
            unused(f.mesh_shaders, "mesh shaders", "supported; conduit has no mesh pipeline yet");
            unused(f.ray_tracing, "ray tracing", "supported; left to the application");
            unused(f.variable_rate_shading, "variable rate shading", "supported; left to the application");
            unused(f.work_graphs && !p.gpu_driven, "work graphs", "supported; only used with gpu-driven rendering");
        }
        p.reasons.push_back(std::string("tier ") + tier_name(p.tier));
        if (load != workload::balanced)
            p.reasons.push_back(std::string("tuned for a ") + workload_name(load) + " workload");
        return p;
    }

    std::string render_profile::explain() const {
        std::string out;
        const auto put = [&](const char* format, auto... args) {
            const int n = std::snprintf(nullptr, 0, format, args...);
            if (n <= 0)
                return;
            const size_t at = out.size();
            out.resize(at + static_cast<size_t>(n) + 1);
            std::snprintf(out.data() + at, static_cast<size_t>(n) + 1, format, args...);
            out.pop_back();
        };
        if (api == conduit::api::none) {
            out = "no usable graphics adapter\n";
            for (const std::string& r : reasons)
                out += "  " + r + "\n";
            return out;
        }
        put("%s on %s (adapter %u), tier %s, %s workload\n", api_name(api), adapter_name.c_str(), adapter_index,
            tier_name(tier), workload_name(workload));
        put("host: %s, %u cores (%u fast), %.1f GiB\n", os_name(system.os), system.logical_cores,
            system.performance_cores, static_cast<double>(system.memory_bytes) / static_cast<double>(1ull << 30));
        for (const std::string& r : reasons)
            out += "  " + r + "\n";
        out += "settings:\n";
        put("  frames in flight %u, record threads %u (min %u draws a list), pipeline threads %u\n", frames_in_flight,
            record_threads, min_draws_per_list, pipeline_threads);
        put("  upload ring %llu MiB, memory budget %llu MiB, render scale %.2f, anisotropy %.0f\n",
            static_cast<unsigned long long>(upload_ring_bytes >> 20),
            static_cast<unsigned long long>(memory_budget_bytes >> 20), static_cast<double>(render_scale),
            static_cast<double>(max_anisotropy));
        out += "decisions:\n";
        for (const decision& d : decisions)
            put("  %-22s %-11s %s\n", d.feature.c_str(),
                d.state == verdict::enabled  ? "enabled"
                : d.state == verdict::unused ? "unused"
                                             : "unavailable",
                d.reason.c_str());
        out += "candidates:\n";
        for (const candidate& c : fallbacks) {
            put("  %8.0f  %s on %s\n", c.score, api_name(c.adapter.api), c.adapter.name.c_str());
            for (const score_term& t : c.terms)
                put("    %+8.0f  %s\n", t.points, t.reason.c_str());
        }
        return out;
    }

    system_info query_system() {
        system_info info;
        info.logical_cores = std::thread::hardware_concurrency();
        info.performance_cores = info.logical_cores;
#if defined(__EMSCRIPTEN__)
        info.os = os_kind::web;
#elif defined(_WIN32)
        info.os = os_kind::windows;
        MEMORYSTATUSEX memory{};
        memory.dwLength = sizeof(memory);
        if (GlobalMemoryStatusEx(&memory))
            info.memory_bytes = memory.ullTotalPhys;
        // hybrid cpus report an efficiency class per core: the highest class is the fast one
        DWORD bytes = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);
        std::vector<uint8_t> buffer(bytes);
        if (bytes && GetLogicalProcessorInformationEx(
                         RelationProcessorCore,
                         reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()), &bytes)) {
            uint8_t top = 0;
            uint32_t fast = 0;
            for (DWORD at = 0; at < bytes;) {
                const auto* e = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + at);
                uint32_t threads = 0;
                for (WORD g = 0; g < e->Processor.GroupCount; g++)
                    threads +=
                        static_cast<uint32_t>(std::popcount(static_cast<uint64_t>(e->Processor.GroupMask[g].Mask)));
                if (e->Processor.EfficiencyClass > top) {
                    top = e->Processor.EfficiencyClass;
                    fast = 0;
                }
                if (e->Processor.EfficiencyClass == top)
                    fast += threads;
                at += e->Size;
            }
            if (fast)
                info.performance_cores = fast;
        }
#elif defined(__APPLE__)
#if TARGET_OS_IPHONE
        info.os = os_kind::ios;
#else
        info.os = os_kind::macos;
#endif
        uint64_t memory = 0;
        size_t size = sizeof(memory);
        if (sysctlbyname("hw.memsize", &memory, &size, nullptr, 0) == 0)
            info.memory_bytes = memory;
        // perflevel0 is the fastest class on apple silicon; intel macs do not have the key
        int32_t fast = 0;
        size = sizeof(fast);
        if (sysctlbyname("hw.perflevel0.logicalcpu", &fast, &size, nullptr, 0) == 0 && fast > 0)
            info.performance_cores = static_cast<uint32_t>(fast);
#else
#if defined(__ANDROID__)
        info.os = os_kind::android;
#else
        info.os = os_kind::linux_desktop;
#endif
        const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page > 0)
            info.memory_bytes = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page);
        // big.little: the cores that reach the highest frequency are the fast ones
        uint64_t top = 0;
        uint32_t fast = 0;
        for (uint32_t cpu = 0; cpu < info.logical_cores; cpu++) {
            const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq";
            uint64_t khz = 0;
            if (std::FILE* file = std::fopen(path.c_str(), "r")) {
                unsigned long long value = 0;
                if (std::fscanf(file, "%llu", &value) == 1)
                    khz = value;
                std::fclose(file);
            }
            if (!khz)
                continue;
            if (khz > top) {
                top = khz;
                fast = 0;
            }
            if (khz == top)
                fast++;
        }
        if (fast)
            info.performance_cores = fast;
#endif
        return info;
    }

    std::string hardware_key(const adapter_info& adapter, const system_info& system) {
        char key[160];
        std::snprintf(key, sizeof(key), "%s-%04x-%04x-%llx-%s-%uc%u", api_id(adapter.api), adapter.vendor_id,
                      adapter.device_id, static_cast<unsigned long long>(adapter.driver_version), os_name(system.os),
                      system.logical_cores, system.performance_cores);
        return key;
    }

    adaptive_tuner::adaptive_tuner(const render_profile& profile, const tuner_options& options)
        : options_(options), render_scale_(profile.render_scale), record_threads_(profile.record_threads) {
        options_.max_record_threads = (std::max)(options_.max_record_threads, profile.record_threads);
        options_.min_render_scale = std::clamp(options_.min_render_scale, 0.25f, 1.0f);
    }

    bottleneck adaptive_tuner::classify() const {
        // without a gpu time only the cpu side can be judged
        const float lead = options_.margin * options_.target_ms;
        if (gpu_ <= 0.0f)
            return cpu_ > options_.target_ms ? bottleneck::cpu : bottleneck::balanced;
        if (cpu_ > gpu_ + lead && cpu_ > options_.target_ms * (1.0f - options_.margin))
            return bottleneck::cpu;
        if (gpu_ > cpu_ + lead && gpu_ > options_.target_ms * (1.0f - options_.margin))
            return bottleneck::gpu;
        return bottleneck::balanced;
    }

    bool adaptive_tuner::update(const frame_sample& sample) {
        // a slow moving average: one hitch must not look like a trend
        const float blend = frames_ == 0 ? 1.0f : 0.05f;
        cpu_ += (sample.cpu_ms - cpu_) * blend;
        gpu_ += (sample.gpu_ms - gpu_) * blend;
        frames_++;
        if (cooldown_) {
            cooldown_--;
            pending_frames_ = 0;
            return false;
        }
        const bottleneck now = classify();
        if (now != pending_) {
            pending_ = now;
            pending_frames_ = 0;
        }
        if (++pending_frames_ < options_.settle_frames)
            return false;
        pending_frames_ = 0;
        state_ = now;

        const float scale = render_scale_;
        const uint32_t threads = record_threads_;
        const float headroom = options_.target_ms * (1.0f - 2.0f * options_.margin);
        if (now == bottleneck::gpu && options_.allow_render_scale)
            render_scale_ = (std::max)(options_.min_render_scale, render_scale_ - options_.scale_step);
        else if (now != bottleneck::gpu && options_.allow_render_scale && gpu_ > 0.0f && gpu_ < headroom)
            render_scale_ = (std::min)(1.0f, render_scale_ + options_.scale_step);
        if (now == bottleneck::cpu)
            record_threads_ = (std::min)(options_.max_record_threads, record_threads_ + 1);
        else if (cpu_ < headroom * 0.5f && record_threads_ > 1)
            record_threads_--;
        if (scale == render_scale_ && threads == record_threads_)
            return false;
        cooldown_ = options_.cooldown_frames;
        return true;
    }

    std::string serialize_failures(const std::vector<known_failure>& failures) {
        std::ostringstream out;
        for (const known_failure& f : failures)
            out << api_id(f.api) << ' ' << f.vendor_id << ' ' << f.device_id << ' ' << f.driver_version << ' '
                << f.count << '\n';
        return out.str();
    }

    std::vector<known_failure> parse_failures(const std::string& text) {
        std::vector<known_failure> out;
        std::istringstream in(text);
        std::string id;
        known_failure f;
        while (in >> id >> f.vendor_id >> f.device_id >> f.driver_version >> f.count) {
            f.api = api_from_id(id);
            if (f.api != api::none)
                out.push_back(f);
        }
        return out;
    }

    void record_failure(std::vector<known_failure>& failures, const adapter_info& a) {
        for (known_failure& f : failures)
            if (f.api == a.api && f.vendor_id == a.vendor_id && f.device_id == a.device_id &&
                f.driver_version == a.driver_version) {
                f.count++;
                return;
            }
        failures.push_back({a.api, a.vendor_id, a.device_id, a.driver_version, 1});
    }
}
