#pragma once
#include <conduit/device.hpp>

#include <array>
#include <memory>
#include <string>

namespace conduit::vendor {
    enum class quality : uint8_t { native, ultra_quality, quality, balanced, performance, ultra_performance };
    float scale_of(quality quality);
    const char* quality_name(quality quality);
    void render_size_for(quality quality, uint32_t output_width, uint32_t output_height, uint32_t& width,
                         uint32_t& height);

    std::array<float, 2> jitter_for(uint64_t frame, uint32_t render_width, uint32_t output_width);
    uint32_t jitter_phases(uint32_t render_width, uint32_t output_width);

    struct mat4 {
        float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    };

    struct camera_data {
        mat4 view_to_clip;
        mat4 clip_to_view;
        mat4 clip_to_prev_clip;
        mat4 prev_clip_to_clip;
        float position[3] = {};
        float up[3] = {0, 1, 0};
        float right[3] = {1, 0, 0};
        float forward[3] = {0, 0, 1};
        float near_plane = 0.05f;
        float far_plane = 1000.0f;
        float fov_y = 1.2f;
        float aspect = 16.0f / 9.0f;
        bool reverse_z = true;
    };

    struct upscale_desc {
        uint32_t render_width = 0, render_height = 0;
        uint32_t output_width = 0, output_height = 0;
        vendor::quality quality = vendor::quality::quality;
        bool hdr = true;
        bool auto_exposure = true;
    };

    struct upscale_inputs {
        texture_handle color;
        state color_state = state::shader_read_other;
        texture_handle depth;
        state depth_state = state::shader_read_other;
        texture_handle motion;
        state motion_state = state::shader_read_other;
        texture_handle exposure;
        state exposure_state = state::shader_read_other;
        texture_handle output;
        state output_state = state::shader_write;

        float jitter_x = 0, jitter_y = 0;
        float motion_scale_x = 1, motion_scale_y = 1;
        float pre_exposure = 1.0f;
        bool reset = false;
        camera_data camera;
    };

    class upscaler {
    public:
        virtual ~upscaler() = default;
        virtual const char* name() const = 0;
        virtual bool evaluate(command_list& list, const upscale_inputs& inputs) = 0;
        virtual upscale_desc desc() const = 0;
        virtual bool resize(const upscale_desc& desc) = 0;
    };

    struct hudless_inputs {
        texture_handle depth;
        state depth_state = state::shader_read_other;
        texture_handle motion;
        state motion_state = state::shader_read_other;
        texture_handle hudless;
        state hudless_state = state::shader_read_other;
        camera_data camera;
        float jitter_x = 0, jitter_y = 0;
        float motion_scale_x = 1, motion_scale_y = 1;
        bool reset = false;
    };

    struct frame_gen_state {
        bool active = false;
        uint32_t max_multiplier = 1;
        uint32_t presented_per_rendered_frame = 1;
        std::string status;
    };

    class frame_generator {
    public:
        virtual ~frame_generator() = default;
        virtual bool set_multiplier(uint32_t multiplier, uint32_t back_buffers, uint32_t width, uint32_t height) = 0;
        virtual bool tag(command_list& list, const hudless_inputs& inputs) = 0;
        virtual frame_gen_state state() = 0;
    };

    enum class latency_marker : uint8_t {
        simulation_start,
        simulation_end,
        render_submit_start,
        render_submit_end,
        present_start,
        present_end
    };

    class low_latency {
    public:
        virtual ~low_latency() = default;
        virtual bool set_mode(bool enabled, bool boost, uint32_t frame_limit_us) = 0;
        virtual void sleep() = 0;
        virtual void mark(latency_marker marker) = 0;
    };

    struct session_request {
        bool upscaler = true;
        bool frame_generation = false;
        bool low_latency = true;
        std::wstring modules_path;
        std::wstring log_path;
        bool verify_signatures = true;
        bool verbose_log = false;
        uint32_t application_id = 0;
    };

    struct capabilities {
        bool dlss = false;
        bool dlss_frame_generation = false;
        bool reflex = false;
        std::string log;
    };

    class session {
    public:
        virtual ~session() = default;
        static std::unique_ptr<session> begin(api api, const session_request& request, vendor_hooks& hooks,
                                              std::string* log);

        virtual bool attach(device& device) = 0;
        virtual capabilities caps() const = 0;

        virtual void begin_frame(uint64_t frame) = 0;
        virtual std::unique_ptr<upscaler> create_upscaler(const upscale_desc& desc) = 0;
        virtual frame_generator* frame_gen() = 0;
        virtual low_latency* latency() = 0;
        virtual void shutdown() = 0;
    };
}
