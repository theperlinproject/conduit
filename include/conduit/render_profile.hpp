#pragma once
#include <conduit/device.hpp>

#include <string>
#include <vector>

namespace conduit {
    enum class tier : uint8_t { minimal, low, medium, high, ultra };
    const char* tier_name(tier tier);

    enum class upscaler : uint8_t { native, spatial, fsr, xess, dlss };
    const char* upscaler_name(upscaler u);

    struct candidate {
        adapter_info adapter;
        bool have_caps = false;
        device_caps caps;
        double score = 0.0;
        std::vector<std::string> notes;
    };

    struct probe_options {
        std::vector<api> apis;
        bool read_caps = true;
        bool allow_software = true;
        bool validation = false;
    };

    struct probe {
        std::vector<candidate> candidates;
        void sort();
    };

    probe probe_system(const probe_options& options);

    struct user_choice {
        conduit::api api = conduit::api::none;
        int adapter_index = -1;
        conduit::tier tier = conduit::tier::medium;
        bool auto_tier = true;
        bool allow_frame_generation = true;
        bool allow_upscaling = true;
        bool allow_hdr = false;
    };

    struct known_failure {
        conduit::api api = conduit::api::none;
        uint32_t vendor_id = 0;
        uint32_t device_id = 0;
        uint64_t driver_version = 0;
        uint32_t count = 0;
    };

    struct render_profile {
        conduit::api api = conduit::api::none;
        uint32_t adapter_index = 0;
        std::string adapter_name;
        conduit::tier tier = conduit::tier::medium;

        uint32_t frames_in_flight = 2;
        bool gpu_culling = false;
        bool async_compute = false;
        bool tearing = false;
        bool hdr = false;
        uint32_t msaa = 1;
        float render_scale = 1.0f;
        float max_anisotropy = 1.0f;

        std::vector<upscaler> upscalers;
        bool frame_generation = false;
        bool neural_features = false;

        std::vector<candidate> fallbacks;
        std::vector<std::string> reasons;
    };

    // deterministic for a given probe, so it can be tested without a gpu
    render_profile choose_profile(const probe& probe, const user_choice& user,
                                  const std::vector<known_failure>& failures, bool windows, bool apple);

    tier tier_for(const candidate& c);
    double score_candidate(const candidate& c, bool windows, bool apple);

    std::string serialize_failures(const std::vector<known_failure>& failures);
    std::vector<known_failure> parse_failures(const std::string& text);
    void record_failure(std::vector<known_failure>& failures, const adapter_info& adapter);
}
