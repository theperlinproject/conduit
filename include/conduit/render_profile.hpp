#pragma once
#include <conduit/device.hpp>

#include <string>
#include <string_view>
#include <vector>

namespace conduit {
    enum class tier : uint8_t { minimal, low, medium, high, ultra };
    const char* tier_name(tier tier);

    enum class upscaler : uint8_t { native, spatial, fsr, xess, dlss };
    const char* upscaler_name(upscaler u);

    // one line of a candidate's score, so a ranking can be read instead of guessed at
    struct score_term {
        double points = 0.0;
        std::string reason;
    };

    struct candidate {
        adapter_info adapter;
        bool have_caps = false;
        device_caps caps;
        double score = 0.0;
        std::vector<std::string> notes;
        std::vector<score_term> terms;
    };

    enum class os_kind : uint8_t { unknown, windows, linux_desktop, macos, ios, android, web };
    const char* os_name(os_kind os);

    // what the host is, apart from its gpus. zero means unknown: every consumer has a default for that
    struct system_info {
        os_kind os = os_kind::unknown;
        uint32_t logical_cores = 0;
        // cores of the fastest class; equal to logical_cores on a cpu with a single class
        uint32_t performance_cores = 0;
        uint64_t memory_bytes = 0;
        bool mobile() const { return os == os_kind::ios || os == os_kind::android; }
        bool browser() const { return os == os_kind::web; }
    };
    system_info query_system();

    // what the application mostly spends its frame on. the same machine gets a different configuration for
    // each: a feature that pays for itself in one costs more than it saves in another
    enum class workload : uint8_t {
        balanced,
        draw_heavy,    // cpu bound: many draw calls, many small objects
        gpu_heavy,     // gpu bound: high resolution, heavy shading, few draws
        texture_heavy, // memory bound: large textures and streaming
        compute_heavy, // simulation or culling on the gpu competing with rendering
        low_latency,   // input to photon time matters more than throughput
        efficiency     // battery, heat and browsers: stay well inside the budget
    };
    const char* workload_name(workload w);

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

        conduit::workload workload = conduit::workload::balanced;
        // rough numbers for a typical frame; zero means unknown and the profile assumes a middling load
        uint32_t expected_draws = 0;
        uint32_t output_width = 0;
        uint32_t output_height = 0;
        // added to the score of every candidate on an api, for applications that know better than the heuristics
        struct api_bias {
            conduit::api api = conduit::api::none;
            double points = 0.0;
        };
        std::vector<api_bias> bias;
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

        conduit::workload workload = conduit::workload::balanced;
        system_info system;

        // how to drive the device
        bool gpu_driven = false;          // compute writes the indirect arguments and a gpu-side draw count
        bool multi_draw_indirect = false; // one call issues many draws
        bool bindless = false;
        bool async_pipelines = false;
        uint32_t record_threads = 1;       // lists recorded at once, the calling thread included
        uint32_t min_draws_per_list = 512; // below this a second list costs more than it saves
        uint32_t pipeline_threads = 1;
        uint32_t gpu_culling_min_draws = 0; // cull on the cpu below this many draws
        uint64_t upload_ring_bytes = 16ull << 20;
        uint64_t memory_budget_bytes = 0; // what textures and buffers should stay under; zero when unknown

        // every choice above with its reason, and what the hardware offers that this profile leaves alone
        enum class verdict : uint8_t { enabled, unused, unavailable };
        struct decision {
            std::string feature;
            verdict state = verdict::unavailable;
            std::string reason;
        };
        std::vector<decision> decisions;
        const decision* find(std::string_view feature) const;

        // options for instance::create_device that match this profile
        device_options device() const;
        // the whole profile as text: selection, score, decisions and limits
        std::string explain() const;
    };

    // deterministic for a given probe, so it can be tested without a gpu
    render_profile choose_profile(const probe& probe, const user_choice& user,
                                  const std::vector<known_failure>& failures, const system_info& system);
    // the same with a host known only by its family: thread counts fall back to one
    render_profile choose_profile(const probe& probe, const user_choice& user,
                                  const std::vector<known_failure>& failures, bool windows, bool apple);

    tier tier_for(const candidate& c);
    double score_candidate(const candidate& c, bool windows, bool apple);

    std::string serialize_failures(const std::vector<known_failure>& failures);
    std::vector<known_failure> parse_failures(const std::string& text);
    void record_failure(std::vector<known_failure>& failures, const adapter_info& adapter);

    // identifies what a measured setting was measured on: a stored value is only reused while the api, gpu,
    // driver and core count are all the same
    std::string hardware_key(const adapter_info& adapter, const system_info& system);

    enum class bottleneck : uint8_t { unknown, cpu, gpu, balanced };
    const char* bottleneck_name(bottleneck b);

    struct frame_sample {
        float cpu_ms = 0; // time the application and conduit spent producing the frame
        float gpu_ms = 0; // time the gpu spent on it, zero when it cannot be measured
    };

    struct tuner_options {
        float target_ms = 16.6f;
        float min_render_scale = 0.5f;
        float scale_step = 0.05f;
        // a side must lead by this share of the frame before it is called the bottleneck
        float margin = 0.15f;
        // frames a reading must persist before anything changes, and frames to sit still after a change
        uint32_t settle_frames = 90;
        uint32_t cooldown_frames = 600;
        uint32_t max_record_threads = 1;
        bool allow_render_scale = true;
    };

    // watches frame times and nudges the two settings that are safe to change while running. it costs a few
    // arithmetic operations a frame, allocates nothing, and waits out a cooldown after every change so a scene
    // on the edge does not flip back and forth
    class adaptive_tuner {
    public:
        adaptive_tuner() = default;
        adaptive_tuner(const render_profile& profile, const tuner_options& options);

        // true on the frame a recommendation changes
        bool update(const frame_sample& sample);

        bottleneck state() const { return state_; }
        float render_scale() const { return render_scale_; }
        uint32_t record_threads() const { return record_threads_; }
        float cpu_ms() const { return cpu_; }
        float gpu_ms() const { return gpu_; }

    private:
        bottleneck classify() const;

        tuner_options options_;
        bottleneck state_ = bottleneck::unknown;
        bottleneck pending_ = bottleneck::unknown;
        float render_scale_ = 1.0f;
        uint32_t record_threads_ = 1;
        float cpu_ = 0, gpu_ = 0;
        uint32_t pending_frames_ = 0;
        uint32_t cooldown_ = 0;
        uint64_t frames_ = 0;
    };
}
