#include <conduit/render_profile.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>

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

    double score_candidate(const candidate& c, bool windows, bool apple) {
        const adapter_info& a = c.adapter;
        if (!a.usable)
            return -1e9;
        double s = api_base(a.api, windows, apple);
        if (s <= 0)
            return -1e9;
        // OpenGL ES only wins when nothing else works at all, whatever the adapter bonuses add up to
        if (a.api == api::gles)
            s -= 2000.0;
        switch (a.type) {
        case adapter_type::discrete:
            s += 1000;
            break;
        case adapter_type::integrated:
            s += 300;
            break;
        case adapter_type::unknown:
            s += 200;
            break;
        case adapter_type::virtual_gpu:
            s -= 400;
            break;
        case adapter_type::cpu:
            s -= 800;
            break;
        }
        const double gib = static_cast<double>(usable_memory(a)) / static_cast<double>(bytes_per_gib);
        s += 40.0 * std::log2(1.0 + gib);
        if (c.have_caps) {
            const device_features& f = c.caps.features;
            s += f.draw_indirect_count ? 25 : 0;
            s += f.async_compute ? 10 : 0;
            s += f.timestamp_queries ? 3 : 0;
            s += f.mesh_shaders ? 8 : 0;
            s += f.ray_tracing ? 8 : 0;
            s += f.bindless ? 6 : 0;
            s += f.enhanced_barriers ? 4 : 0;
            if (!f.multi_draw_indirect)
                s -= 60;
        }
        if (windows && a.vendor_id == vendor_id::intel && a.type == adapter_type::integrated && a.api == api::vulkan)
            s -= 30;
        s += driver_adjust(a, windows, apple, nullptr);
        return s;
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
        if (a.type == adapter_type::integrated)
            level = (std::min)(level, 2);
        if (a.api == api::gles)
            level = (std::min)(level, 1);
        if (c.have_caps) {
            const device_features& f = c.caps.features;
            if (!f.multi_draw_indirect)
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

    render_profile choose_profile(const conduit::probe& probe, const user_choice& user,
                                  const std::vector<known_failure>& failures, bool windows, bool apple) {
        render_profile p;
        std::vector<candidate> pool;
        for (candidate c : probe.candidates) {
            c.score = score_candidate(c, windows, apple);
            if (c.score < -1e8)
                continue;
            std::string driver_note;
            if (driver_adjust(c.adapter, windows, apple, &driver_note) != 0.0 && !driver_note.empty())
                c.notes.push_back(driver_note);
            if (failed(failures, c.adapter)) {
                c.score -= 5000;
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
        p.api = best.adapter.api;
        p.adapter_index = best.adapter.index;
        p.adapter_name = best.adapter.name;
        p.fallbacks = pool;
        p.reasons.push_back(std::string("using ") + api_name(p.api) + " on " + best.adapter.name);
        for (const std::string& n : best.notes)
            p.reasons.push_back(n);

        p.tier = user.auto_tier ? tier_for(best) : user.tier;
        if (is_software(best.adapter))
            p.reasons.emplace_back("software rasteriser: expect low frame rates");

        const bool have_caps = best.have_caps;
        const device_features f = have_caps ? best.caps.features : device_features{};

        static constexpr std::array<float, 5> tier_anisotropy{1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
        const float anisotropy = tier_anisotropy[static_cast<size_t>(p.tier)];
        p.max_anisotropy =
            have_caps ? (std::min)(anisotropy, (std::max)(1.0f, best.caps.limits.max_anisotropy)) : anisotropy;
        p.msaa = 1;

        p.frames_in_flight = (p.tier >= tier::medium && best.adapter.type == adapter_type::discrete) ? 3 : 2;
        p.gpu_culling =
            !have_caps ? p.tier >= tier::medium : (f.gpu_driven || (f.multi_draw_indirect && f.draw_indirect_count));
        p.async_compute = have_caps && f.async_compute && best.adapter.type == adapter_type::discrete;
        p.tearing = have_caps && f.tearing;
        p.hdr = user.allow_hdr && have_caps && f.hdr_output;

        if (user.allow_upscaling && p.tier >= tier::low) {
            if (best.adapter.vendor_id == vendor_id::nvidia && p.api != api::d3d11 && p.api != api::gles)
                p.upscalers.push_back(upscaler::dlss);
            if (best.adapter.vendor_id == vendor_id::intel && best.adapter.type == adapter_type::discrete)
                p.upscalers.push_back(upscaler::xess);
            if (p.api == api::vulkan || p.api == api::d3d12)
                p.upscalers.push_back(upscaler::fsr);
            p.upscalers.push_back(upscaler::spatial);
        }
        p.upscalers.push_back(upscaler::native);

        p.frame_generation = user.allow_frame_generation && p.tier >= tier::high &&
                             best.adapter.type == adapter_type::discrete &&
                             (p.api == api::d3d12 || p.api == api::vulkan);
        p.neural_features = have_caps && (f.cooperative_vector || f.cooperative_matrix);
        p.render_scale = 1.0f;
        if (p.tier <= tier::low && !p.upscalers.empty() && p.upscalers.front() != upscaler::native) {
            p.render_scale = 0.75f;
            p.reasons.emplace_back("modest gpu: an upscaler at 75% would help (off until turned on in the options)");
        }
        p.reasons.push_back(std::string("tier ") + tier_name(p.tier));
        return p;
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
