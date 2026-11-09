#pragma once
#include <conduit/types.hpp>

#include <cstdint>
#include <memory>
#include <string>

// where an OpenGL ES context comes from: EGL on Android, Linux and Windows (ANGLE), WebGL 2 through Emscripten
namespace conduit::gles {
    struct gl_info {
        std::string vendor;
        std::string renderer;
        std::string version;
        uint32_t major = 0;
        uint32_t minor = 0;
    };

    // the default framebuffer of a window. everything conduit draws goes to an offscreen texture first, so this only
    // ever receives one blit per frame
    class gl_surface {
    public:
        virtual ~gl_surface() = default;
        // makes the surface current on the context
        virtual bool bind() = 0;
        virtual void swap() = 0;
        virtual void set_interval(int interval) = 0;
        virtual void extent(uint32_t& width, uint32_t& height) = 0;
        virtual void resize(uint32_t width, uint32_t height) = 0;
    };

    // used from a single thread: the one that created it
    class gl_context {
    public:
        virtual ~gl_context() = default;
        virtual bool make_current() = 0;
        virtual void release() = 0;
        virtual void* get_proc(const char* name) = 0;
        virtual std::unique_ptr<gl_surface> create_surface(const native_window& window, uint32_t width,
                                                           uint32_t height) = 0;
        // whether a swap interval of zero can be honoured
        virtual bool can_disable_vsync() const = 0;
    };

    // cheap: only checks that the libraries are there
    bool platform_available();
    // creates the highest OpenGL ES context (3.2 down to 3.0) that can be had without a window
    std::unique_ptr<gl_context> create_gl_context(bool debug, std::string& error);
    // reads the driver strings without leaving a context behind
    bool probe_gl(gl_info& info, std::string& error);
    // splits "OpenGL ES 3.2 Mesa ..." into 3 and 2
    bool parse_gl_version(const std::string& version, uint32_t& major, uint32_t& minor);
    // true when the extension list names it with or without its GL_ prefix
    bool has_extension_name(const std::string& list, const char* name);
}
