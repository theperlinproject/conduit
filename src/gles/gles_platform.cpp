#include "gles/gles_platform.hpp"
#include "gles/gles_gl.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__EMSCRIPTEN__)
#include <emscripten.h>
#include <emscripten/html5.h>
#else
#if !defined(_WIN32) && !defined(__ANDROID__)
#ifndef EGL_NO_X11
#define EGL_NO_X11 1
#endif
#endif
#include <EGL/egl.h>
#include <EGL/eglext.h>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace conduit::gles {
    bool parse_gl_version(const std::string& version, uint32_t& major, uint32_t& minor) {
        const size_t at = version.find("OpenGL ES ");
        if (at == std::string::npos)
            return false;
        unsigned maj = 0, min = 0;
        if (std::sscanf(version.c_str() + at + 10, "%u.%u", &maj, &min) < 1)
            return false;
        major = maj;
        minor = min;
        return true;
    }

    bool has_extension_name(const std::string& list, const char* name) {
        const std::string wanted = name;
        const std::string alt = wanted.rfind("GL_", 0) == 0 ? wanted.substr(3) : "GL_" + wanted;
        size_t pos = 0;
        while (pos < list.size()) {
            size_t end = list.find(' ', pos);
            if (end == std::string::npos)
                end = list.size();
            const std::string_view word(list.data() + pos, end - pos);
            if (word == wanted || word == alt)
                return true;
            pos = end + 1;
        }
        return false;
    }

#if defined(__EMSCRIPTEN__)
    namespace {
        const char* canvas_selector() {
#ifdef CONDUIT_GLES_CANVAS
            return CONDUIT_GLES_CANVAS;
#else
            return "#canvas";
#endif
        }

        EM_JS(int, conduit_gles_probe, (char* vendor, char* renderer, int capacity), {
            try {
                var canvas = typeof document !== 'undefined' ? document.createElement('canvas') : new OffscreenCanvas(1, 1);
                var gl = canvas.getContext('webgl2');
                if (!gl)
                    return 0;
                var info = gl.getExtension('WEBGL_debug_renderer_info');
                var v = info ? gl.getParameter(info.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR);
                var r = info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
                stringToUTF8(String(v), vendor, capacity);
                stringToUTF8(String(r), renderer, capacity);
                var lose = gl.getExtension('WEBGL_lose_context');
                if (lose)
                    lose.loseContext();
                return 1;
            } catch (e) {
                return 0;
            }
        });

        class web_surface final : public gl_surface {
        public:
            explicit web_surface(const char* selector) : selector_(selector ? selector : canvas_selector()) {}
            bool bind() override { return true; }
            // the browser presents the canvas when control returns to its event loop
            void swap() override {}
            void set_interval(int) override {}
            void extent(uint32_t& width, uint32_t& height) override {
                int w = 0, h = 0;
                emscripten_get_canvas_element_size(selector_.c_str(), &w, &h);
                width = static_cast<uint32_t>(std::max(w, 1));
                height = static_cast<uint32_t>(std::max(h, 1));
            }
            void resize(uint32_t width, uint32_t height) override {
                emscripten_set_canvas_element_size(selector_.c_str(), static_cast<int>(width),
                                                   static_cast<int>(height));
            }

        private:
            std::string selector_;
        };

        class web_context final : public gl_context {
        public:
            ~web_context() override {
                if (context_)
                    emscripten_webgl_destroy_context(context_);
            }

            bool init(std::string& error) {
                EmscriptenWebGLContextAttributes attributes;
                emscripten_webgl_init_context_attributes(&attributes);
                attributes.majorVersion = 2;
                attributes.minorVersion = 0;
                attributes.alpha = false;
                attributes.depth = false;
                attributes.stencil = false;
                attributes.antialias = false;
                attributes.premultipliedAlpha = false;
                attributes.preserveDrawingBuffer = false;
                attributes.powerPreference = EM_WEBGL_POWER_PREFERENCE_HIGH_PERFORMANCE;
                attributes.failIfMajorPerformanceCaveat = false;
                attributes.enableExtensionsByDefault = true;
                context_ = emscripten_webgl_create_context(canvas_selector(), &attributes);
                if (context_ <= 0) {
                    error = "could not create a WebGL 2 context on " + std::string(canvas_selector());
                    context_ = 0;
                    return false;
                }
                if (!make_current()) {
                    error = "could not make the WebGL context current";
                    return false;
                }
                static const char* const extensions[] = {"EXT_color_buffer_float",
                                                         "EXT_color_buffer_half_float",
                                                         "OES_texture_float_linear",
                                                         "EXT_texture_filter_anisotropic",
                                                         "WEBGL_compressed_texture_s3tc",
                                                         "WEBGL_compressed_texture_s3tc_srgb",
                                                         "EXT_texture_compression_bptc",
                                                         "EXT_texture_norm16",
                                                         "EXT_float_blend"};
                for (const char* name : extensions)
                    emscripten_webgl_enable_extension(context_, name);
                return true;
            }

            bool make_current() override { return emscripten_webgl_make_current(context_) == EMSCRIPTEN_RESULT_SUCCESS; }
            void release() override {}
            void* get_proc(const char*) override { return nullptr; }
            std::unique_ptr<gl_surface> create_surface(const native_window& window, uint32_t, uint32_t) override {
                return std::make_unique<web_surface>(static_cast<const char*>(window.window));
            }
            bool can_disable_vsync() const override { return false; }

        private:
            EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context_ = 0;
        };
    }

    bool platform_available() {
        return true;
    }

    std::unique_ptr<gl_context> create_gl_context(bool, std::string& error) {
        auto context = std::make_unique<web_context>();
        if (!context->init(error))
            return nullptr;
        return context;
    }

    bool probe_gl(gl_info& info, std::string& error) {
        char vendor[256] = {}, renderer[256] = {};
        if (!conduit_gles_probe(vendor, renderer, static_cast<int>(sizeof(vendor)))) {
            error = "this browser has no WebGL 2";
            return false;
        }
        info.vendor = vendor;
        info.renderer = renderer;
        info.version = "OpenGL ES 3.0 (WebGL 2.0)";
        info.major = 3;
        info.minor = 0;
        return true;
    }
#else
    namespace {
#define CONDUIT_EGL_FUNCTIONS(X)                                                                                     \
    X(eglGetProcAddress)                                                                                             \
    X(eglGetDisplay)                                                                                                 \
    X(eglInitialize)                                                                                                 \
    X(eglTerminate)                                                                                                  \
    X(eglBindAPI)                                                                                                    \
    X(eglChooseConfig)                                                                                               \
    X(eglGetConfigAttrib)                                                                                            \
    X(eglCreateContext)                                                                                              \
    X(eglDestroyContext)                                                                                             \
    X(eglCreateWindowSurface)                                                                                        \
    X(eglCreatePbufferSurface)                                                                                       \
    X(eglDestroySurface)                                                                                             \
    X(eglMakeCurrent)                                                                                                \
    X(eglSwapBuffers)                                                                                                \
    X(eglSwapInterval)                                                                                               \
    X(eglQuerySurface)                                                                                               \
    X(eglQueryString)                                                                                                \
    X(eglGetError)

        struct egl_api {
#define CONDUIT_EGL_DECLARE(name) decltype(&::name) name = nullptr;
            CONDUIT_EGL_FUNCTIONS(CONDUIT_EGL_DECLARE)
#undef CONDUIT_EGL_DECLARE
        };

#if defined(_WIN32)
        using lib_handle = HMODULE;
        lib_handle open_library(const char* name) {
            return LoadLibraryA(name);
        }
        void* find_symbol(lib_handle lib, const char* name) {
            return reinterpret_cast<void*>(GetProcAddress(lib, name));
        }
        constexpr const char* egl_names[] = {"libEGL.dll", "EGL.dll"};
        constexpr const char* gles_names[] = {"libGLESv2.dll", "GLESv2.dll"};
#else
        using lib_handle = void*;
        lib_handle open_library(const char* name) {
            return dlopen(name, RTLD_NOW | RTLD_LOCAL);
        }
        void* find_symbol(lib_handle lib, const char* name) {
            return dlsym(lib, name);
        }
#if defined(__ANDROID__)
        constexpr const char* egl_names[] = {"libEGL.so"};
        constexpr const char* gles_names[] = {"libGLESv2.so"};
#elif defined(__APPLE__)
        constexpr const char* egl_names[] = {"libEGL.dylib"};
        constexpr const char* gles_names[] = {"libGLESv2.dylib"};
#else
        constexpr const char* egl_names[] = {"libEGL.so.1", "libEGL.so"};
        constexpr const char* gles_names[] = {"libGLESv2.so.2", "libGLESv2.so"};
#endif
#endif

        struct libraries {
            lib_handle egl = nullptr;
            lib_handle gles = nullptr;
            egl_api api;
            bool loaded = false;
        };

        // loaded once and kept for the life of the process: unloading a GL driver while anything still points into it
        // is never worth the risk
        libraries& libs() {
            static libraries l = [] {
                libraries out;
                for (const char* name : egl_names)
                    if ((out.egl = open_library(name)))
                        break;
                for (const char* name : gles_names)
                    if ((out.gles = open_library(name)))
                        break;
                if (!out.egl || !out.gles)
                    return out;
                bool ok = true;
#define CONDUIT_EGL_LOAD(fn)                                                                                         \
    out.api.fn = reinterpret_cast<decltype(&::fn)>(find_symbol(out.egl, #fn));                                       \
    ok = ok && out.api.fn != nullptr;
                CONDUIT_EGL_FUNCTIONS(CONDUIT_EGL_LOAD)
#undef CONDUIT_EGL_LOAD
                out.loaded = ok;
                return out;
            }();
            return l;
        }

        bool has_egl_extension(const char* list, const char* name) {
            return list && has_extension_name(list, name);
        }

        class egl_context;

        class egl_surface final : public gl_surface {
        public:
            egl_surface(egl_context& context, EGLSurface surface, void (*destroy_native)(void*) = nullptr,
                        void* native = nullptr)
                : context_(context), surface_(surface), destroy_native_(destroy_native), native_(native) {}
            ~egl_surface() override;
            bool bind() override;
            void swap() override;
            void set_interval(int interval) override;
            void extent(uint32_t& width, uint32_t& height) override;
            void resize(uint32_t width, uint32_t height) override;

        private:
            egl_context& context_;
            EGLSurface surface_;
            void (*destroy_native_)(void*);
            void* native_;
            int interval_ = -1;
        };

        class egl_context final : public gl_context {
        public:
            ~egl_context() override {
                const egl_api& egl = libs().api;
                if (display_ != EGL_NO_DISPLAY) {
                    egl.eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
                    if (pbuffer_ != EGL_NO_SURFACE)
                        egl.eglDestroySurface(display_, pbuffer_);
                    if (context_ != EGL_NO_CONTEXT)
                        egl.eglDestroyContext(display_, context_);
                    egl.eglTerminate(display_);
                }
            }

            bool init(bool debug, std::string& error) {
                libraries& l = libs();
                if (!l.loaded) {
                    error = "no EGL and OpenGL ES libraries on this system";
                    return false;
                }
                const egl_api& egl = l.api;
                display_ = egl.eglGetDisplay(EGL_DEFAULT_DISPLAY);
                if (display_ == EGL_NO_DISPLAY) {
                    error = "eglGetDisplay failed";
                    return false;
                }
                EGLint major = 0, minor = 0;
                if (!egl.eglInitialize(display_, &major, &minor)) {
                    error = "eglInitialize failed";
                    display_ = EGL_NO_DISPLAY;
                    return false;
                }
                extensions_ = egl.eglQueryString(display_, EGL_EXTENSIONS) ? egl.eglQueryString(display_, EGL_EXTENSIONS) : "";
                if (!egl.eglBindAPI(EGL_OPENGL_ES_API)) {
                    error = "eglBindAPI(OpenGL ES) failed";
                    return false;
                }
                surfaceless_ = has_egl_extension(extensions_.c_str(), "EGL_KHR_surfaceless_context");

                if (!choose_config()) {
                    error = "no EGL config for OpenGL ES 3";
                    return false;
                }
                struct version {
                    EGLint major, minor;
                };
                static constexpr version versions[] = {{3, 2}, {3, 1}, {3, 0}};
                for (const version& v : versions) {
                    std::vector<EGLint> attributes = {EGL_CONTEXT_MAJOR_VERSION, v.major};
                    if (v.minor > 0 || major > 1 || minor >= 5 ||
                        has_egl_extension(extensions_.c_str(), "EGL_KHR_create_context")) {
                        attributes.push_back(EGL_CONTEXT_MINOR_VERSION);
                        attributes.push_back(v.minor);
                    }
                    if (debug) {
                        attributes.push_back(EGL_CONTEXT_OPENGL_DEBUG);
                        attributes.push_back(EGL_TRUE);
                    }
                    attributes.push_back(EGL_NONE);
                    context_ = egl.eglCreateContext(display_, config_, EGL_NO_CONTEXT, attributes.data());
                    if (context_ != EGL_NO_CONTEXT)
                        break;
                }
                if (context_ == EGL_NO_CONTEXT) {
                    error = "eglCreateContext failed for OpenGL ES 3.0 to 3.2";
                    return false;
                }
                if (!surfaceless_) {
                    const EGLint attributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
                    pbuffer_ = egl.eglCreatePbufferSurface(display_, config_, attributes);
                    if (pbuffer_ == EGL_NO_SURFACE) {
                        error = "this EGL has neither surfaceless contexts nor pbuffers";
                        return false;
                    }
                }
                if (!make_current()) {
                    error = "eglMakeCurrent failed";
                    return false;
                }
                // swapping to zero is legal in EGL, but a compositor may still pace the window: say so only when the
                // call is accepted
                vsync_off_ = egl.eglSwapInterval(display_, 0) == EGL_TRUE;
                egl.eglSwapInterval(display_, 1);
                return true;
            }

            bool make_current() override {
                bound_ = nullptr;
                return libs().api.eglMakeCurrent(display_, pbuffer_, pbuffer_, context_) == EGL_TRUE;
            }
            void release() override {
                bound_ = nullptr;
                libs().api.eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            }
            void* get_proc(const char* name) override {
                libraries& l = libs();
                if (void* p = find_symbol(l.gles, name))
                    return p;
                return reinterpret_cast<void*>(l.api.eglGetProcAddress(name));
            }
            bool can_disable_vsync() const override { return vsync_off_; }

            std::unique_ptr<gl_surface> create_surface(const native_window& window, uint32_t width,
                                                       uint32_t height) override;

            EGLDisplay display() const { return display_; }
            EGLContext context() const { return context_; }
            EGLSurface fallback() const { return pbuffer_; }
            egl_surface* bound() const { return bound_; }
            void set_bound(egl_surface* s) { bound_ = s; }

        private:
            bool choose_config() {
                const egl_api& egl = libs().api;
#if defined(__ANDROID__)
                constexpr EGLint alpha = 8;
#else
                constexpr EGLint alpha = 0;
#endif
                for (const EGLint renderable : {EGL_OPENGL_ES3_BIT, EGL_OPENGL_ES2_BIT}) {
                    for (const EGLint surface_types : {EGL_WINDOW_BIT | EGL_PBUFFER_BIT, EGL_WINDOW_BIT}) {
                        const EGLint attributes[] = {EGL_SURFACE_TYPE, surface_types, EGL_RENDERABLE_TYPE, renderable,
                                                     EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                                                     EGL_ALPHA_SIZE, alpha, EGL_DEPTH_SIZE, 0, EGL_STENCIL_SIZE, 0,
                                                     EGL_NONE};
                        EGLint count = 0;
                        if (egl.eglChooseConfig(display_, attributes, &config_, 1, &count) && count > 0)
                            return true;
                    }
                }
                return false;
            }

            EGLDisplay display_ = EGL_NO_DISPLAY;
            EGLContext context_ = EGL_NO_CONTEXT;
            EGLConfig config_ = nullptr;
            EGLSurface pbuffer_ = EGL_NO_SURFACE;
            std::string extensions_;
            bool surfaceless_ = false;
            bool vsync_off_ = false;
            egl_surface* bound_ = nullptr;
        };

        egl_surface::~egl_surface() {
            const egl_api& egl = libs().api;
            if (context_.bound() == this)
                context_.make_current();
            egl.eglDestroySurface(context_.display(), surface_);
            if (destroy_native_)
                destroy_native_(native_);
        }

        bool egl_surface::bind() {
            if (context_.bound() == this)
                return true;
            if (libs().api.eglMakeCurrent(context_.display(), surface_, surface_, context_.context()) != EGL_TRUE)
                return false;
            context_.set_bound(this);
            interval_ = -1;
            return true;
        }

        void egl_surface::swap() {
            libs().api.eglSwapBuffers(context_.display(), surface_);
        }

        void egl_surface::set_interval(int interval) {
            if (interval == interval_)
                return;
            interval_ = interval;
            libs().api.eglSwapInterval(context_.display(), interval);
        }

        void egl_surface::extent(uint32_t& width, uint32_t& height) {
            EGLint w = 0, h = 0;
            const egl_api& egl = libs().api;
            egl.eglQuerySurface(context_.display(), surface_, EGL_WIDTH, &w);
            egl.eglQuerySurface(context_.display(), surface_, EGL_HEIGHT, &h);
            width = static_cast<uint32_t>(std::max(w, 1));
            height = static_cast<uint32_t>(std::max(h, 1));
        }

        using wl_egl_window_create_fn = void* (*)(void*, int, int);
        using wl_egl_window_resize_fn = void (*)(void*, int, int, int, int);
        using wl_egl_window_destroy_fn = void (*)(void*);

        struct wayland_native {
            lib_handle lib = nullptr;
            void* window = nullptr;
            wl_egl_window_resize_fn resize = nullptr;
            wl_egl_window_destroy_fn destroy = nullptr;
        };

        void destroy_wayland(void* p) {
            auto* w = static_cast<wayland_native*>(p);
            if (w->window && w->destroy)
                w->destroy(w->window);
            delete w;
        }

        void egl_surface::resize(uint32_t width, uint32_t height) {
            if (destroy_native_ == &destroy_wayland) {
                auto* w = static_cast<wayland_native*>(native_);
                if (w->resize)
                    w->resize(w->window, static_cast<int>(width), static_cast<int>(height), 0, 0);
            }
        }

        std::unique_ptr<gl_surface> egl_context::create_surface(const native_window& window, uint32_t width,
                                                                uint32_t height) {
            const egl_api& egl = libs().api;
            EGLNativeWindowType native = reinterpret_cast<EGLNativeWindowType>(window.window);
            void (*destroy_native)(void*) = nullptr;
            void* native_state = nullptr;
#if !defined(_WIN32) && !defined(__ANDROID__) && !defined(__APPLE__)
            if (window.wayland) {
                // the host hands over a wl_surface, EGL wants a wl_egl_window around it
                auto* w = new wayland_native;
                w->lib = open_library("libwayland-egl.so.1");
                if (!w->lib) {
                    delete w;
                    return nullptr;
                }
                const auto create = reinterpret_cast<wl_egl_window_create_fn>(find_symbol(w->lib, "wl_egl_window_create"));
                w->resize = reinterpret_cast<wl_egl_window_resize_fn>(find_symbol(w->lib, "wl_egl_window_resize"));
                w->destroy = reinterpret_cast<wl_egl_window_destroy_fn>(find_symbol(w->lib, "wl_egl_window_destroy"));
                if (!create || !(w->window = create(window.window, static_cast<int>(std::max(width, 1u)),
                                                    static_cast<int>(std::max(height, 1u))))) {
                    delete w;
                    return nullptr;
                }
                native = reinterpret_cast<EGLNativeWindowType>(w->window);
                destroy_native = &destroy_wayland;
                native_state = w;
            }
#else
            (void)width;
            (void)height;
#endif
            // an X11 window only takes configs with its own visual: when the context's config is refused, try the
            // others of the same shape. contexts accept any config with the same color format
            EGLSurface surface = egl.eglCreateWindowSurface(display_, config_, native, nullptr);
            if (surface == EGL_NO_SURFACE) {
                EGLConfig configs[64];
                EGLint count = 0;
                const EGLint attributes[] = {EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                                             EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_DEPTH_SIZE,
                                             0, EGL_STENCIL_SIZE, 0, EGL_NONE};
                if (egl.eglChooseConfig(display_, attributes, configs, 64, &count))
                    for (EGLint i = 0; i < count && surface == EGL_NO_SURFACE; i++)
                        surface = egl.eglCreateWindowSurface(display_, configs[i], native, nullptr);
            }
            if (surface == EGL_NO_SURFACE) {
                if (destroy_native)
                    destroy_native(native_state);
                return nullptr;
            }
            return std::make_unique<egl_surface>(*this, surface, destroy_native, native_state);
        }
    }

    bool platform_available() {
        return libs().loaded;
    }

    std::unique_ptr<gl_context> create_gl_context(bool debug, std::string& error) {
        auto context = std::make_unique<egl_context>();
        if (!context->init(debug, error))
            return nullptr;
        return context;
    }

    bool probe_gl(gl_info& info, std::string& error) {
        auto context = create_gl_context(false, error);
        if (!context)
            return false;
        using get_string_fn = const GLubyte*(GL_APIENTRY*)(GLenum);
        const auto get_string = reinterpret_cast<get_string_fn>(context->get_proc("glGetString"));
        if (!get_string) {
            error = "glGetString is missing";
            return false;
        }
        const auto text = [&](GLenum name) {
            const GLubyte* s = get_string(name);
            return s ? std::string(reinterpret_cast<const char*>(s)) : std::string();
        };
        info.vendor = text(GL_VENDOR);
        info.renderer = text(GL_RENDERER);
        info.version = text(GL_VERSION);
        if (!parse_gl_version(info.version, info.major, info.minor)) {
            error = "not an OpenGL ES context: " + info.version;
            return false;
        }
        return true;
    }
#endif
}
