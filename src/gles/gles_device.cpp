#include "gles/gles_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>

#ifndef GL_TEXTURE_BORDER_COLOR
#define GL_TEXTURE_BORDER_COLOR 0x1004
#endif
#ifndef GL_CLAMP_TO_BORDER
#define GL_CLAMP_TO_BORDER 0x812D
#endif

namespace conduit::gles {
    const format_gl& format_of(format f) {
        static const format_gl table[] = {
            /* unknown */ {},
            /* r8_unorm */ {GL_R8, GL_RED, GL_UNSIGNED_BYTE, 1},
            /* rg8_unorm */ {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, 2},
            /* rgba8_unorm */ {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4},
            /* rgba8_srgb */ {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4},
            /* bgra8_unorm */ {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, false, false, true},
            /* bgra8_srgb */ {GL_SRGB8_ALPHA8, GL_RGBA, GL_UNSIGNED_BYTE, 4, false, false, true},
            /* r8_uint */ {GL_R8UI, GL_RED_INTEGER, GL_UNSIGNED_BYTE, 1, false, true},
            /* r16_uint */ {GL_R16UI, GL_RED_INTEGER, GL_UNSIGNED_SHORT, 2, false, true},
            /* r32_uint */ {GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, 4, false, true},
            /* rg32_uint */ {GL_RG32UI, GL_RG_INTEGER, GL_UNSIGNED_INT, 8, false, true},
            /* rgba32_uint */ {GL_RGBA32UI, GL_RGBA_INTEGER, GL_UNSIGNED_INT, 16, false, true},
            /* r16_float */ {GL_R16F, GL_RED, GL_HALF_FLOAT, 2},
            /* rg16_float */ {GL_RG16F, GL_RG, GL_HALF_FLOAT, 4},
            /* rgba16_float */ {GL_RGBA16F, GL_RGBA, GL_HALF_FLOAT, 8},
            /* r32_float */ {GL_R32F, GL_RED, GL_FLOAT, 4},
            /* rg32_float */ {GL_RG32F, GL_RG, GL_FLOAT, 8},
            /* rgb32_float */ {GL_RGB32F, GL_RGB, GL_FLOAT, 12},
            /* rgba32_float */ {GL_RGBA32F, GL_RGBA, GL_FLOAT, 16},
            /* rgb10a2_unorm */ {GL_RGB10_A2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, 4},
            /* r11g11b10_float */ {GL_R11F_G11F_B10F, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, 4},
            /* rg16_unorm */ {GL_RG16_EXT, GL_RG, GL_UNSIGNED_SHORT, 4, false, false, false, false, false, true},
            /* rgba16_unorm */ {GL_RGBA16_EXT, GL_RGBA, GL_UNSIGNED_SHORT, 8, false, false, false, false, false, true},
            /* d16_unorm */ {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT, 2},
            /* d32_float */ {GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, 4},
            /* d24_unorm_s8_uint */ {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, 4},
            /* d32_float_s8_uint */
            {GL_DEPTH32F_STENCIL8, GL_DEPTH_STENCIL, GL_FLOAT_32_UNSIGNED_INT_24_8_REV, 8},
            /* bc1_unorm */ {GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 0, 0, 8, true, false, false, true},
            /* bc3_unorm */ {GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, 0, 0, 16, true, false, false, true},
            /* bc7_unorm */ {GL_COMPRESSED_RGBA_BPTC_UNORM_EXT, 0, 0, 16, true, false, false, false, true},
        };
        static_assert(std::size(table) == static_cast<size_t>(format::count));
        return table[static_cast<size_t>(f)];
    }

    namespace {
        GLenum wrap_of(address_mode m, bool border) {
            switch (m) {
            case address_mode::repeat:
                return GL_REPEAT;
            case address_mode::mirror:
                return GL_MIRRORED_REPEAT;
            case address_mode::border:
                return border ? GL_CLAMP_TO_BORDER : GL_CLAMP_TO_EDGE;
            default:
                return GL_CLAMP_TO_EDGE;
            }
        }

        GLenum compare_of(compare_op op) {
            switch (op) {
            case compare_op::never:
                return GL_NEVER;
            case compare_op::less:
                return GL_LESS;
            case compare_op::equal:
                return GL_EQUAL;
            case compare_op::less_equal:
                return GL_LEQUAL;
            case compare_op::greater:
                return GL_GREATER;
            case compare_op::not_equal:
                return GL_NOTEQUAL;
            case compare_op::greater_equal:
                return GL_GEQUAL;
            default:
                return GL_ALWAYS;
            }
        }

        GLenum target_of(texture_dim dim) {
            switch (dim) {
            case texture_dim::tex2d_array:
                return GL_TEXTURE_2D_ARRAY;
            case texture_dim::cube:
                return GL_TEXTURE_CUBE_MAP;
            case texture_dim::tex3d:
                return GL_TEXTURE_3D;
            default:
                return GL_TEXTURE_2D;
            }
        }
    }

    device::device(const adapter_info& info, const device_options& options) : options_(options) {
        caps_.adapter = info;
        options_.frames_in_flight = std::clamp(options_.frames_in_flight, 1u, 4u);
    }

    device::~device() {
        closing_ = true;
        thread_.call([this] {
            if (context_) {
                gl_.glFinish();
                release_all();
                context_->release();
            }
            return 0;
        });
        thread_.stop();
        context_.reset();
    }

    void device::report_error(const std::string& message) {
        std::lock_guard lock(error_mutex_);
        if (errors_.size() < 64)
            errors_.push_back(message);
        std::fprintf(stderr, "[gles] %s\n", message.c_str());
    }

    void device::report_once(const std::string& message) {
        {
            std::lock_guard lock(error_mutex_);
            if (!reported_.emplace(message, true).second)
                return;
        }
        report_error(message);
    }

    std::string device::debug_report() {
        std::lock_guard lock(error_mutex_);
        std::string out;
        for (const std::string& e : errors_)
            out += e + "\n";
        return out;
    }

    native_handles device::native() const {
        native_handles n;
        n.api = api::gles;
        return n;
    }

    void device::drain_errors() {
        for (int i = 0; i < 16 && gl_.glGetError() != GL_NO_ERROR; i++) {
        }
    }

    bool device::query_features(std::string& error) {
        const auto text = [&](GLenum name) {
            const GLubyte* s = gl_.glGetString(name);
            return s ? std::string(reinterpret_cast<const char*>(s)) : std::string();
        };
        const std::string version = text(GL_VERSION);
        if (!parse_gl_version(version, features_.major, features_.minor) || features_.major < 3) {
            error = "OpenGL ES 3.0 is the lowest this backend runs on, the driver reports " + version;
            return false;
        }
        GLint count = 0;
        gl_.glGetIntegerv(GL_NUM_EXTENSIONS, &count);
        for (GLint i = 0; i < count; i++) {
            const GLubyte* e = gl_.glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i));
            if (e) {
                extensions_ += reinterpret_cast<const char*>(e);
                extensions_ += ' ';
            }
        }
        const uint32_t level = features_.major * 10 + features_.minor;
        const auto ext = [&](const char* name) { return has_extension_name(extensions_, name); };
        features_.glsl_version = std::clamp(level * 10, 300u, 320u);
        features_.compute = level >= 31 && gl_.glDispatchCompute && gl_.glMemoryBarrier;
        features_.indirect = level >= 31 && gl_.glDrawElementsIndirect;
        features_.storage_images = level >= 31 && gl_.glBindImageTexture && gl_.glMemoryBarrier;
        features_.indexed_blend = (level >= 32 || ext("GL_EXT_draw_buffers_indexed") || ext("GL_OES_draw_buffers_indexed")) &&
                                  gl_.glBlendFuncSeparatei && gl_.glBlendEquationSeparatei && gl_.glColorMaski &&
                                  gl_.glEnablei && gl_.glDisablei;
        features_.copy_image = (level >= 32 || ext("GL_EXT_copy_image") || ext("GL_OES_copy_image")) && gl_.glCopyImageSubData;
        features_.debug_groups = (level >= 32 || ext("GL_KHR_debug")) && gl_.glPushDebugGroup && gl_.glPopDebugGroup;
        features_.color_float = ext("GL_EXT_color_buffer_float");
        features_.half_float_color = features_.color_float || ext("GL_EXT_color_buffer_half_float");
        features_.float_blend = ext("GL_EXT_float_blend");
        features_.float_linear = ext("GL_OES_texture_float_linear");
        features_.anisotropy = ext("GL_EXT_texture_filter_anisotropic");
        features_.s3tc = ext("GL_EXT_texture_compression_s3tc");
        features_.bptc = ext("GL_EXT_texture_compression_bptc");
        features_.norm16 = ext("GL_EXT_texture_norm16");
        features_.border_clamp = level >= 32 || ext("GL_EXT_texture_border_clamp") || ext("GL_OES_texture_border_clamp");
        features_.map_range = gl_.glMapBufferRange && gl_.glUnmapBuffer;
        features_.vsync_off = context_->can_disable_vsync();
        return true;
    }

    bool device::format_supported(format f) const {
        const format_gl& g = format_of(f);
        if (!g.internal)
            return false;
        if (g.needs_s3tc && !features_.s3tc)
            return false;
        if (g.needs_bptc && !features_.bptc)
            return false;
        if (g.needs_norm16 && !features_.norm16)
            return false;
        return true;
    }

    void device::fill_caps() {
        const auto integer = [&](GLenum name, GLint fallback) {
            GLint v = fallback;
            gl_.glGetIntegerv(name, &v);
            return v;
        };
        device_limits& l = caps_.limits;
        l.max_texture2d = static_cast<uint32_t>(integer(GL_MAX_TEXTURE_SIZE, 2048));
        l.max_texture_layers = static_cast<uint32_t>(integer(GL_MAX_ARRAY_TEXTURE_LAYERS, 256));
        l.max_bind_groups = max_bind_groups;
        l.max_push_constant_bytes = max_push_constant_bytes;
        max_texture_units_ = static_cast<uint32_t>(std::clamp(integer(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, 32), 8, 48));
        l.max_sampled_textures_per_stage = static_cast<uint32_t>(integer(GL_MAX_TEXTURE_IMAGE_UNITS, 16));
        l.max_color_targets = static_cast<uint32_t>(
            std::clamp(std::min(integer(GL_MAX_COLOR_ATTACHMENTS, 4), integer(GL_MAX_DRAW_BUFFERS, 4)), 1, 8));
        l.max_msaa_samples = static_cast<uint32_t>(std::max(integer(GL_MAX_SAMPLES, 1), 1));
        l.uniform_buffer_alignment = static_cast<uint32_t>(std::max(integer(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, 256), 1));
        l.storage_buffer_alignment =
            features_.compute ? static_cast<uint32_t>(std::max(integer(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, 256), 1))
                              : 256u;
        l.copy_row_pitch_alignment = 4;
        l.copy_offset_alignment = 4;
        l.max_buffer_size = 1ull << 30;
        max_uniform_bindings_ = static_cast<uint32_t>(std::max(integer(GL_MAX_UNIFORM_BUFFER_BINDINGS, 24), 8));
        if (features_.anisotropy) {
            GLfloat a = 1.0f;
            gl_.glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &a);
            l.max_anisotropy = std::max(a, 1.0f);
        }

        device_features& f = caps_.features;
        f.multi_draw_indirect = features_.indirect;
        f.sampler_anisotropy = features_.anisotropy;
        f.texture_compression_bc = features_.s3tc && features_.bptc;
        f.unified_memory = true;
        f.tearing = features_.vsync_off;
        f.depth_clip_control = false;

        for (size_t i = 1; i < static_cast<size_t>(format::count); i++) {
            const format fmt = static_cast<format>(i);
            if (!format_supported(fmt))
                continue;
            const format_info& info = get_format_info(fmt);
            const format_gl& g = format_of(fmt);
            format_use use = format_use::sampled;
            const bool wide_float = info.floating && info.bytes_per_block / std::max<uint8_t>(info.channels, 1) == 4 && !info.depth;
            if (!g.integer && !info.depth && !info.compressed && (!wide_float || features_.float_linear))
                use = use | format_use::sampled_linear;
            if (info.compressed)
                use = use | format_use::sampled_linear;
            if (info.depth) {
                use = use | format_use::depth_target | format_use::msaa;
            } else if (!info.compressed) {
                bool renderable = true;
                bool blendable = !g.integer;
                if (info.floating) {
                    renderable = wide_float ? features_.color_float
                                            : (fmt == format::r11g11b10_float ? features_.half_float_color : features_.half_float_color);
                    if (fmt == format::rgb32_float)
                        renderable = false;
                    if (wide_float)
                        blendable = features_.float_blend;
                }
                if (g.needs_norm16)
                    renderable = false;
                if (renderable) {
                    use = use | format_use::color_target | format_use::msaa;
                    if (blendable)
                        use = use | format_use::blendable;
                }
                if (features_.storage_images &&
                    (fmt == format::rgba8_unorm || fmt == format::r32_float || fmt == format::rgba16_float ||
                     fmt == format::rgba32_float || fmt == format::r32_uint || fmt == format::rgba32_uint))
                    use = use | format_use::storage;
            }
            caps_.formats[i] = use;
        }
    }

    bool device::init(std::string& error) {
        thread_.start();
        const bool ok = thread_.call([&] {
            context_ = create_gl_context(options_.validation, error);
            if (!context_)
                return false;
            if (!gl_.load([this](const char* name) { return context_->get_proc(name); })) {
                error = "the OpenGL ES driver lacks functions every OpenGL ES 3.0 implementation has";
                return false;
            }
            if (!query_features(error))
                return false;
            fill_caps();

            gl_.glGenVertexArrays(1, &vao_);
            gl_.glBindVertexArray(vao_);
            gl_.glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            gl_.glPixelStorei(GL_PACK_ALIGNMENT, 1);
            cache_.unpack_row = cache_.pack_row = 0;
            gl_.glGenSamplers(1, &default_sampler_);
            gl_.glSamplerParameteri(default_sampler_, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            gl_.glSamplerParameteri(default_sampler_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            gl_.glSamplerParameteri(default_sampler_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            gl_.glSamplerParameteri(default_sampler_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            gl_.glSamplerParameteri(default_sampler_, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
            GLuint fbos[2] = {};
            gl_.glGenFramebuffers(2, fbos);
            scratch_read_ = fbos[0];
            scratch_draw_ = fbos[1];

            // push constants go through a ring of uniform buffer slots, one per change
            push_ring_size_ = 1u << 20;
            gl_.glGenBuffers(1, &push_ring_);
            gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, push_ring_);
            gl_.glBufferData(GL_COPY_WRITE_BUFFER, static_cast<GLsizeiptr>(push_ring_size_), nullptr, GL_STREAM_DRAW);
            drain_errors();
            return true;
        });
        if (!ok)
            return false;

        std::string renderer = caps_.adapter.name;
        frame_fence_ = create_fence(0);
        return true;
    }

    // ---------------------------------------------------------------- state helpers (GL thread)

    void device::use_texture(uint32_t unit, GLenum target, GLuint texture) {
        if (cache_.active_unit != unit) {
            gl_.glActiveTexture(GL_TEXTURE0 + unit);
            cache_.active_unit = unit;
        }
        if (unit < cache_.unit_texture.size()) {
            if (cache_.unit_texture[unit] == texture && cache_.unit_target[unit] == target)
                return;
            cache_.unit_texture[unit] = texture;
            cache_.unit_target[unit] = target;
        }
        gl_.glBindTexture(target, texture);
    }

    void device::use_program(GLuint program) {
        if (cache_.program == program)
            return;
        cache_.program = program;
        gl_.glUseProgram(program);
    }

    void device::use_scissor_test(bool enable) {
        if (cache_.scissor_test == enable)
            return;
        cache_.scissor_test = enable;
        if (enable)
            gl_.glEnable(GL_SCISSOR_TEST);
        else
            gl_.glDisable(GL_SCISSOR_TEST);
    }

    void device::bind_framebuffer(GLenum target, GLuint fbo) {
        if (target == GL_FRAMEBUFFER) {
            if (cache_.draw_framebuffer == fbo && cache_.read_framebuffer == fbo)
                return;
            cache_.draw_framebuffer = cache_.read_framebuffer = fbo;
        } else if (target == GL_DRAW_FRAMEBUFFER) {
            if (cache_.draw_framebuffer == fbo)
                return;
            cache_.draw_framebuffer = fbo;
        } else {
            if (cache_.read_framebuffer == fbo)
                return;
            cache_.read_framebuffer = fbo;
        }
        gl_.glBindFramebuffer(target, fbo);
    }

    // ---------------------------------------------------------------- resources

    buffer_handle device::create_buffer(const buffer_desc& desc) {
        buffer b;
        b.desc = desc;
        b.name = desc.name ? desc.name : "";
        uint64_t size = align_up(std::max<uint64_t>(desc.size, 16), 16);
        if (has(desc.usage, buffer_usage::uniform))
            size = align_up(size, 16);
        if (desc.stride && has(desc.usage, buffer_usage::storage))
            size = align_up(size, desc.stride);
        if (size > caps_.limits.max_buffer_size) {
            report_error("CreateBuffer: " + std::to_string(size) + " bytes is more than the supported buffer size");
            return {};
        }
        b.size = size;
        const bool cpu_side = desc.memory != memory_class::gpu_only;
        if (cpu_side)
            b.shadow.assign(size, 0);
        if (desc.memory != memory_class::readback) {
            const bool stream = desc.memory == memory_class::upload || desc.memory == memory_class::gpu_mappable;
            const bool ok = thread_.call([&] {
                gl_.glGenBuffers(1, &b.gl);
                gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, b.gl);
                gl_.glBufferData(GL_COPY_WRITE_BUFFER, static_cast<GLsizeiptr>(size), nullptr,
                                 stream ? GL_STREAM_DRAW : GL_DYNAMIC_DRAW);
                return gl_.glGetError() == GL_NO_ERROR;
            });
            if (!ok) {
                thread_.post([this, id = b.gl] { gl_.glDeleteBuffers(1, &id); });
                report_error("CreateBuffer: the driver could not allocate " + std::to_string(size) + " bytes");
                return {};
            }
        }
        buffer_bytes_ += size;
        live_buffers_++;
        const buffer_handle h = buffers_.alloc(std::move(b));
        buffers_.get(h)->desc.name = buffers_.get(h)->name.c_str();
        return h;
    }

    texture_handle device::create_texture(const texture_desc& desc) {
        if (!format_supported(desc.format)) {
            report_error("CreateTexture: format " + std::to_string(static_cast<int>(desc.format)) +
                         " is not available in this OpenGL ES context");
            return {};
        }
        texture t;
        t.desc = desc;
        t.desc.width = std::max(desc.width, 1u);
        t.desc.height = std::max(desc.height, 1u);
        t.desc.mips = std::max(desc.mips, 1u);
        t.name = desc.name ? desc.name : "";
        t.fmt = format_of(desc.format);
        t.target = target_of(desc.dim);
        const format_info& info = get_format_info(desc.format);
        const uint32_t layers = layer_count_of(t.desc);
        const uint32_t depth = depth_of(t.desc);
        if (t.desc.width > caps_.limits.max_texture2d || t.desc.height > caps_.limits.max_texture2d) {
            report_error("CreateTexture: " + std::to_string(t.desc.width) + "x" + std::to_string(t.desc.height) +
                         " is larger than this context's textures");
            return {};
        }
        const bool multisampled = desc.samples > 1;
        if (multisampled && (desc.dim != texture_dim::tex2d || t.desc.mips > 1)) {
            report_error("CreateTexture: multisampled textures are single level 2D textures here");
            return {};
        }
        t.renderbuffer = multisampled;
        if (multisampled && has(desc.usage, texture_usage::sampled))
            report_once("multisampled textures cannot be sampled in OpenGL ES 3.0: resolve them first");

        const bool ok = thread_.call([&] {
            drain_errors();
            if (t.renderbuffer) {
                gl_.glGenRenderbuffers(1, &t.gl);
                gl_.glBindRenderbuffer(GL_RENDERBUFFER, t.gl);
                gl_.glRenderbufferStorageMultisample(
                    GL_RENDERBUFFER, static_cast<GLsizei>(std::min(desc.samples, caps_.limits.max_msaa_samples)),
                    t.fmt.internal, static_cast<GLsizei>(t.desc.width), static_cast<GLsizei>(t.desc.height));
            } else {
                gl_.glGenTextures(1, &t.gl);
                use_texture(scratch_unit(), t.target, t.gl);
                const GLsizei levels = static_cast<GLsizei>(std::min(t.desc.mips, mip_levels_for(t.desc.width, t.desc.height)));
                if (t.target == GL_TEXTURE_3D || t.target == GL_TEXTURE_2D_ARRAY)
                    gl_.glTexStorage3D(t.target, levels, t.fmt.internal, static_cast<GLsizei>(t.desc.width),
                                       static_cast<GLsizei>(t.desc.height),
                                       static_cast<GLsizei>(t.target == GL_TEXTURE_3D ? depth : layers));
                else
                    gl_.glTexStorage2D(t.target, levels, t.fmt.internal, static_cast<GLsizei>(t.desc.width),
                                       static_cast<GLsizei>(t.desc.height));
                gl_.glTexParameteri(t.target, GL_TEXTURE_BASE_LEVEL, 0);
                gl_.glTexParameteri(t.target, GL_TEXTURE_MAX_LEVEL, levels - 1);
                t.base_level = 0;
                t.max_level = levels - 1;
                t.levels = levels;
            }
            return gl_.glGetError() == GL_NO_ERROR;
        });
        if (!ok) {
            thread_.post([this, id = t.gl, rb = t.renderbuffer] {
                if (rb)
                    gl_.glDeleteRenderbuffers(1, &id);
                else
                    gl_.glDeleteTextures(1, &id);
            });
            report_error("CreateTexture: the driver rejected a " + std::to_string(t.desc.width) + "x" +
                         std::to_string(t.desc.height) + " texture (format " +
                         std::to_string(static_cast<int>(desc.format)) + ")");
            return {};
        }
        uint64_t bytes = 0;
        for (uint32_t m = 0; m < t.desc.mips; m++)
            bytes += static_cast<uint64_t>(row_bytes(desc.format, mip_extent(t.desc.width, m))) *
                     row_blocks(desc.format, mip_extent(t.desc.height, m)) * layers * (info.compressed ? 1 : depth);
        texture_bytes_ += bytes * std::max(desc.samples, 1u);
        live_textures_++;
        const texture_handle h = textures_.alloc(std::move(t));
        textures_.get(h)->desc.name = textures_.get(h)->name.c_str();
        return h;
    }

    sampler_handle device::create_sampler(const sampler_desc& desc) {
        sampler s;
        s.desc = desc;
        thread_.call([&] {
            const auto setup = [&](GLuint id, bool nearest) {
                const auto mag = [&](filter f) { return nearest || f == filter::nearest ? GL_NEAREST : GL_LINEAR; };
                GLenum min_filter;
                const bool mip_nearest = nearest || desc.mip == filter::nearest;
                if (desc.min == filter::nearest || nearest)
                    min_filter = mip_nearest ? GL_NEAREST_MIPMAP_NEAREST : GL_NEAREST_MIPMAP_LINEAR;
                else
                    min_filter = mip_nearest ? GL_LINEAR_MIPMAP_NEAREST : GL_LINEAR_MIPMAP_LINEAR;
                gl_.glSamplerParameteri(id, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(min_filter));
                gl_.glSamplerParameteri(id, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(mag(desc.mag)));
                gl_.glSamplerParameteri(id, GL_TEXTURE_WRAP_S, static_cast<GLint>(wrap_of(desc.u, features_.border_clamp)));
                gl_.glSamplerParameteri(id, GL_TEXTURE_WRAP_T, static_cast<GLint>(wrap_of(desc.v, features_.border_clamp)));
                gl_.glSamplerParameteri(id, GL_TEXTURE_WRAP_R, static_cast<GLint>(wrap_of(desc.w, features_.border_clamp)));
                gl_.glSamplerParameterf(id, GL_TEXTURE_MIN_LOD, desc.min_lod);
                gl_.glSamplerParameterf(id, GL_TEXTURE_MAX_LOD, std::min(desc.max_lod, 1000.0f));
                if (desc.compare) {
                    gl_.glSamplerParameteri(id, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
                    gl_.glSamplerParameteri(id, GL_TEXTURE_COMPARE_FUNC, static_cast<GLint>(compare_of(desc.compare_op)));
                }
                if (features_.anisotropy && !nearest && desc.max_anisotropy > 1.0f)
                    gl_.glSamplerParameterf(id, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                            std::min(desc.max_anisotropy, caps_.limits.max_anisotropy));
                if (features_.border_clamp && gl_.glSamplerParameterfv)
                    gl_.glSamplerParameterfv(id, GL_TEXTURE_BORDER_COLOR, desc.border.data());
            };
            gl_.glGenSamplers(1, &s.gl);
            setup(s.gl, false);
            gl_.glGenSamplers(1, &s.gl_nearest);
            setup(s.gl_nearest, true);
            return 0;
        });
        return samplers_.alloc(std::move(s));
    }

    shader_handle device::create_shader(const shader_desc& desc) {
        if (desc.format != shader_format::spirv) {
            report_error("CreateShader: OpenGL ES takes SPIR-V and translates it");
            return {};
        }
        if (desc.code.size() < 20 || desc.code.size() % 4 != 0) {
            report_error("CreateShader: not a SPIR-V module");
            return {};
        }
        shader s;
        s.stage = desc.stage;
        s.words.resize(desc.code.size() / 4);
        std::memcpy(s.words.data(), desc.code.data(), desc.code.size());
        if (s.words[0] != 0x07230203u) {
            report_error("CreateShader: bad SPIR-V magic number");
            return {};
        }
        s.entry = desc.entry ? desc.entry : "main";
        s.name = desc.name ? desc.name : "";
        return shaders_.alloc(std::move(s));
    }

    query_set_handle device::create_query_set(const query_set_desc& desc) {
        query q;
        q.count = desc.count;
        return queries_.alloc(q);
    }

    // everything is destroyed on the GL thread, in order after the work already submitted
    void device::destroy(buffer_handle h) {
        thread_.post([this, h] {
            buffer b;
            if (!buffers_.free(h, &b))
                return;
            if (b.gl) {
                forget_buffer(b.gl);
                gl_.glDeleteBuffers(1, &b.gl);
            }
            buffer_bytes_ -= b.size;
            live_buffers_--;
        });
    }

    void device::delete_framebuffers_using(GLuint gl, bool renderbuffer) {
        const uint64_t tag = static_cast<uint64_t>(gl) | (renderbuffer ? 1ull << 32 : 0);
        for (auto it = framebuffers_.begin(); it != framebuffers_.end();) {
            bool uses = false;
            for (uint64_t k : it->second.key)
                uses = uses || (k != 0 && (k & 0x1FFFFFFFFull) == tag);
            if (uses) {
                if (cache_.draw_framebuffer == it->second.id)
                    cache_.draw_framebuffer = ~0u;
                if (cache_.read_framebuffer == it->second.id)
                    cache_.read_framebuffer = ~0u;
                gl_.glDeleteFramebuffers(1, &it->second.id);
                it = framebuffers_.erase(it);
            } else {
                ++it;
            }
        }
    }

    void device::destroy(texture_handle h) {
        thread_.post([this, h] {
            texture t;
            if (!textures_.free(h, &t))
                return;
            delete_framebuffers_using(t.gl, t.renderbuffer);
            if (t.renderbuffer)
                gl_.glDeleteRenderbuffers(1, &t.gl);
            else {
                for (uint32_t u = 0; u < cache_.unit_texture.size(); u++)
                    if (cache_.unit_texture[u] == t.gl && cache_.unit_target[u] == t.target)
                        cache_.unit_texture[u] = 0;
                gl_.glDeleteTextures(1, &t.gl);
            }
            live_textures_--;
        });
    }

    void device::destroy(sampler_handle h) {
        thread_.post([this, h] {
            sampler s;
            if (!samplers_.free(h, &s))
                return;
            for (GLuint& bound : cache_.unit_sampler)
                if (bound == s.gl || bound == s.gl_nearest)
                    bound = ~0u;
            gl_.glDeleteSamplers(1, &s.gl);
            gl_.glDeleteSamplers(1, &s.gl_nearest);
        });
    }

    void device::destroy(shader_handle h) {
        thread_.post([this, h] { shaders_.free(h); });
    }
    void device::destroy(bind_layout_handle h) {
        thread_.post([this, h] { bind_layouts_.free(h); });
    }
    void device::destroy(bind_group_handle h) {
        thread_.post([this, h] { bind_groups_.free(h); });
    }
    void device::destroy(pipeline_layout_handle h) {
        thread_.post([this, h] { pipeline_layouts_.free(h); });
    }
    void device::destroy(pipeline_handle h) {
        thread_.post([this, h] {
            pipeline p;
            if (!pipelines_.free(h, &p))
                return;
            if (cache_.program == p.program)
                cache_.program = ~0u;
            if (p.program)
                gl_.glDeleteProgram(p.program);
        });
    }
    void device::destroy(query_set_handle h) {
        thread_.post([this, h] { queries_.free(h); });
    }
    void device::destroy(fence_handle h) {
        std::lock_guard lock(fence_mutex_);
        fences_.free(h);
    }

    void device::release_all() {
        const auto delete_buffer = [&](buffer_handle, buffer& b) {
            if (b.gl)
                gl_.glDeleteBuffers(1, &b.gl);
        };
        buffers_.for_each(delete_buffer);
        textures_.for_each([&](texture_handle, texture& t) {
            if (t.renderbuffer)
                gl_.glDeleteRenderbuffers(1, &t.gl);
            else
                gl_.glDeleteTextures(1, &t.gl);
        });
        samplers_.for_each([&](sampler_handle, sampler& s) {
            gl_.glDeleteSamplers(1, &s.gl);
            gl_.glDeleteSamplers(1, &s.gl_nearest);
        });
        pipelines_.for_each([&](pipeline_handle, pipeline& p) {
            if (p.program)
                gl_.glDeleteProgram(p.program);
        });
        for (auto& [key, fb] : framebuffers_)
            gl_.glDeleteFramebuffers(1, &fb.id);
        framebuffers_.clear();
        gl_.glDeleteFramebuffers(1, &scratch_read_);
        gl_.glDeleteFramebuffers(1, &scratch_draw_);
        gl_.glDeleteSamplers(1, &default_sampler_);
        gl_.glDeleteBuffers(1, &push_ring_);
        gl_.glDeleteVertexArrays(1, &vao_);
    }

    const buffer_desc* device::get_desc(buffer_handle h) const {
        const buffer* b = buffers_.get(h);
        return b ? &b->desc : nullptr;
    }

    const texture_desc* device::get_desc(texture_handle h) const {
        const texture* t = textures_.get(h);
        return t ? &t->desc : nullptr;
    }

    uint8_t* device::map(buffer_handle handle) {
        buffer* b = buffers_.get(handle);
        return b && !b->shadow.empty() ? b->shadow.data() : nullptr;
    }

    // upload memory is plain memory until flushed: the range is copied now, so the application may write on, and
    // reaches the buffer in order with the work submitted before and after
    void device::flush(buffer_handle handle, uint64_t offset, uint64_t size) {
        buffer* b = buffers_.get(handle);
        if (!b || b->shadow.empty() || b->desc.memory == memory_class::readback || !b->gl)
            return;
        const uint64_t lo = std::min<uint64_t>(offset & ~uint64_t{3}, b->size);
        const uint64_t hi = std::min(align_up(offset + (size ? size : b->size - lo), 4), b->size);
        if (hi <= lo)
            return;
        auto copy = std::make_shared<std::vector<uint8_t>>(b->shadow.begin() + static_cast<ptrdiff_t>(lo),
                                                           b->shadow.begin() + static_cast<ptrdiff_t>(hi));
        thread_.post([this, handle, lo, copy] {
            const buffer* target = buffers_.get(handle);
            if (!target)
                return;
            gl_.glBindBuffer(GL_COPY_WRITE_BUFFER, target->gl);
            gl_.glBufferSubData(GL_COPY_WRITE_BUFFER, static_cast<GLintptr>(lo), static_cast<GLsizeiptr>(copy->size()),
                                copy->data());
        });
    }

    // readbacks land in the buffer when their work is replayed, which the fence the application waits on covers
    void device::invalidate(buffer_handle, uint64_t, uint64_t) {}

    // ---------------------------------------------------------------- fences and submission

    fence_handle device::create_fence(uint64_t initial_value) {
        return fences_.alloc(fence_state{initial_value});
    }

    uint64_t device::fence_value(fence_handle fence) {
        std::lock_guard lock(fence_mutex_);
        const fence_state* f = fences_.get(fence);
        return f ? f->value : 0;
    }

    void device::signal_now(fence_handle fence, uint64_t value) {
        {
            std::lock_guard lock(fence_mutex_);
            fence_state* f = fences_.get(fence);
            if (!f)
                return;
            f->value = std::max(f->value, value);
        }
        fence_cv_.notify_all();
    }

    bool device::wait_now(fence_handle fence, uint64_t value, uint64_t timeout_ns) {
        std::unique_lock lock(fence_mutex_);
        const auto ready = [&] {
            const fence_state* f = fences_.get(fence);
            return !f || f->value >= value;
        };
        if (ready())
            return true;
#ifdef __EMSCRIPTEN__
        (void)timeout_ns;
        return false;
#else
        if (timeout_ns == ~0ull) {
            fence_cv_.wait(lock, ready);
            return true;
        }
        return fence_cv_.wait_for(lock, std::chrono::nanoseconds(timeout_ns), ready);
#endif
    }

    bool device::wait_fence(fence_handle fence, uint64_t value, uint64_t timeout_ns) {
        return wait_now(fence, value, timeout_ns);
    }

    void device::signal_fence(fence_handle fence, uint64_t value) {
        // in order with the work already submitted
        thread_.post([this, fence, value] { signal_now(fence, value); });
    }

    command_list* device::acquire_command_list(queue_type queue) {
        std::lock_guard lock(list_mutex_);
        cmd_list* list;
        if (!free_lists_.empty()) {
            list = free_lists_.back();
            free_lists_.pop_back();
        } else {
            lists_.push_back(std::make_unique<cmd_list>(*this, queue));
            list = lists_.back().get();
        }
        list->begin();
        return list;
    }

    void device::submit(queue_type, const submit_desc& desc) {
        std::vector<std::vector<uint8_t>> streams;
        streams.reserve(desc.lists.size());
        {
            std::lock_guard lock(list_mutex_);
            for (command_list* base : desc.lists) {
                auto* list = static_cast<cmd_list*>(base);
                counters_.add(list->counters());
                streams.push_back(list->take());
                free_lists_.push_back(list);
            }
        }
        counters_.count_submit();
        std::vector<fence_op> waits(desc.waits.begin(), desc.waits.end());
        std::vector<fence_op> signals(desc.signals.begin(), desc.signals.end());
        thread_.post([this, streams = std::move(streams), waits = std::move(waits), signals = std::move(signals)]() mutable {
            execute(std::move(streams), std::move(waits), std::move(signals));
        });
    }

    void device::execute(std::vector<std::vector<uint8_t>> streams, std::vector<fence_op> waits,
                         std::vector<fence_op> signals) {
        for (const fence_op& w : waits)
            if (!wait_now(w.fence, w.value, 10'000'000'000ull))
                report_error("a submit waited 10 seconds on a fence that was never signalled");
        for (const std::vector<uint8_t>& s : streams)
            replay(s);
        gl_.glFlush();
        for (const fence_op& s : signals)
            signal_now(s.fence, s.value);
    }

    void device::begin_frame() {
        frame_++;
        const uint64_t behind = std::max(1u, options_.frames_in_flight);
        if (frame_ > behind && !wait_now(frame_fence_, frame_ - behind, 10'000'000'000ull))
            report_error("begin_frame waited 10 seconds for the GL thread to finish an earlier frame");
    }

    void device::end_frame() {
        thread_.post([this, fence = frame_fence_, value = frame_] { signal_now(fence, value); });
    }

    void device::wait_idle() {
        thread_.call([this] {
            gl_.glFinish();
            return 0;
        });
    }

    memory_budget device::query_memory_budget() {
        memory_budget m;
        m.device_usage = buffer_bytes_ + texture_bytes_;
        return m;
    }

    device_stats device::stats() const {
        device_stats s;
        counters_.fill(s);
        s.buffer_bytes = buffer_bytes_;
        s.texture_bytes = texture_bytes_;
        s.live_buffers = live_buffers_;
        s.live_textures = live_textures_;
        s.live_pipelines = pipelines_.live();
        s.live_bind_groups = bind_groups_.live();
        return s;
    }

    std::unique_ptr<conduit::swapchain> device::create_swapchain(const swapchain_desc& desc) {
        return make_swapchain(*this, desc);
    }
}
