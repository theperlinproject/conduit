#include "gles/gles_internal.hpp"

#include <algorithm>
#include <cstdio>

namespace conduit::gles {
    namespace {
        GLenum compare_func(compare_op op) {
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

        GLenum stencil_operation(stencil_op op) {
            switch (op) {
            case stencil_op::zero:
                return GL_ZERO;
            case stencil_op::replace:
                return GL_REPLACE;
            case stencil_op::increment_clamp:
                return GL_INCR;
            case stencil_op::decrement_clamp:
                return GL_DECR;
            case stencil_op::invert:
                return GL_INVERT;
            case stencil_op::increment_wrap:
                return GL_INCR_WRAP;
            case stencil_op::decrement_wrap:
                return GL_DECR_WRAP;
            default:
                return GL_KEEP;
            }
        }

        GLenum blend_factor_of(blend_factor f) {
            switch (f) {
            case blend_factor::zero:
                return GL_ZERO;
            case blend_factor::src_color:
                return GL_SRC_COLOR;
            case blend_factor::inv_src_color:
                return GL_ONE_MINUS_SRC_COLOR;
            case blend_factor::src_alpha:
                return GL_SRC_ALPHA;
            case blend_factor::inv_src_alpha:
                return GL_ONE_MINUS_SRC_ALPHA;
            case blend_factor::dst_color:
                return GL_DST_COLOR;
            case blend_factor::inv_dst_color:
                return GL_ONE_MINUS_DST_COLOR;
            case blend_factor::dst_alpha:
                return GL_DST_ALPHA;
            case blend_factor::inv_dst_alpha:
                return GL_ONE_MINUS_DST_ALPHA;
            case blend_factor::src_alpha_saturate:
                return GL_SRC_ALPHA_SATURATE;
            default:
                return GL_ONE;
            }
        }

        GLenum blend_equation_of(blend_op op) {
            switch (op) {
            case blend_op::subtract:
                return GL_FUNC_SUBTRACT;
            case blend_op::reverse_subtract:
                return GL_FUNC_REVERSE_SUBTRACT;
            case blend_op::minimum:
                return GL_MIN;
            case blend_op::maximum:
                return GL_MAX;
            default:
                return GL_FUNC_ADD;
            }
        }

        bool vertex_format(format f, vertex_attribute_gl& a) {
            a.normalized = false;
            a.integer = false;
            switch (f) {
            case format::r8_unorm:
                a.size = 1, a.type = GL_UNSIGNED_BYTE, a.normalized = true;
                return true;
            case format::rg8_unorm:
                a.size = 2, a.type = GL_UNSIGNED_BYTE, a.normalized = true;
                return true;
            case format::rgba8_unorm:
                a.size = 4, a.type = GL_UNSIGNED_BYTE, a.normalized = true;
                return true;
            case format::r8_uint:
                a.size = 1, a.type = GL_UNSIGNED_BYTE, a.integer = true;
                return true;
            case format::r16_uint:
                a.size = 1, a.type = GL_UNSIGNED_SHORT, a.integer = true;
                return true;
            case format::r32_uint:
                a.size = 1, a.type = GL_UNSIGNED_INT, a.integer = true;
                return true;
            case format::rg32_uint:
                a.size = 2, a.type = GL_UNSIGNED_INT, a.integer = true;
                return true;
            case format::rgba32_uint:
                a.size = 4, a.type = GL_UNSIGNED_INT, a.integer = true;
                return true;
            case format::r16_float:
                a.size = 1, a.type = GL_HALF_FLOAT;
                return true;
            case format::rg16_float:
                a.size = 2, a.type = GL_HALF_FLOAT;
                return true;
            case format::rgba16_float:
                a.size = 4, a.type = GL_HALF_FLOAT;
                return true;
            case format::r32_float:
                a.size = 1, a.type = GL_FLOAT;
                return true;
            case format::rg32_float:
                a.size = 2, a.type = GL_FLOAT;
                return true;
            case format::rgb32_float:
                a.size = 3, a.type = GL_FLOAT;
                return true;
            case format::rgba32_float:
                a.size = 4, a.type = GL_FLOAT;
                return true;
            case format::rgb10a2_unorm:
                a.size = 4, a.type = GL_UNSIGNED_INT_2_10_10_10_REV, a.normalized = true;
                return true;
            case format::rg16_unorm:
                a.size = 2, a.type = GL_UNSIGNED_SHORT, a.normalized = true;
                return true;
            case format::rgba16_unorm:
                a.size = 4, a.type = GL_UNSIGNED_SHORT, a.normalized = true;
                return true;
            default:
                return false;
            }
        }

        // ES 3.0 requires a fragment shader in every program
        const char* empty_fragment(uint32_t version) {
            switch (version) {
            case 320:
                return "#version 320 es\nprecision mediump float;\nvoid main() {}\n";
            case 310:
                return "#version 310 es\nprecision mediump float;\nvoid main() {}\n";
            default:
                return "#version 300 es\nprecision mediump float;\nvoid main() {}\n";
            }
        }
    }

    bind_layout_handle device::create_bind_layout(const bind_layout_desc& desc) {
        bind_layout l;
        l.entries.assign(desc.entries.begin(), desc.entries.end());
        return bind_layouts_.alloc(std::move(l));
    }

    bind_group_handle device::create_bind_group(const bind_group_desc& desc) {
        if (!bind_layouts_.get(desc.layout)) {
            report_error("CreateBindGroup: unknown layout");
            return {};
        }
        bind_group g;
        g.layout = desc.layout;
        g.entries.assign(desc.entries.begin(), desc.entries.end());
        uint32_t highest = 0;
        for (const bind_group_entry& e : g.entries)
            highest = std::max(highest, e.binding + 1);
        g.by_binding.assign(highest, -1);
        for (size_t i = 0; i < g.entries.size(); i++)
            g.by_binding[g.entries[i].binding] = static_cast<int32_t>(i);
        return bind_groups_.alloc(std::move(g));
    }

    pipeline_layout_handle device::create_pipeline_layout(const pipeline_layout_desc& desc) {
        pipeline_layout l;
        l.desc = desc;
        for (uint32_t s = 0; s < max_bind_groups; s++) {
            if (!desc.sets[s])
                continue;
            const bind_layout* bl = bind_layouts_.get(desc.sets[s]);
            if (!bl) {
                report_error("CreatePipelineLayout: unknown bind layout");
                return {};
            }
            uint32_t highest = 0;
            for (const bind_layout_entry& e : bl->entries)
                highest = std::max(highest, e.binding + 1);
            l.sets[s].assign(highest, layout_slot{});
            uint32_t dynamic = 0;
            for (const bind_layout_entry& e : bl->entries) {
                if (e.count > 1) {
                    report_error("CreatePipelineLayout: arrays of descriptors are not available in OpenGL ES");
                    return {};
                }
                layout_slot& slot = l.sets[s][e.binding];
                slot.present = true;
                slot.type = e.type;
                slot.dim = e.view_dim;
                switch (e.type) {
                case binding_type::uniform_buffer:
                    slot.index = l.uniform_blocks++;
                    break;
                case binding_type::uniform_buffer_dynamic:
                    slot.index = l.uniform_blocks++;
                    slot.dynamic_index = dynamic++;
                    break;
                case binding_type::storage_buffer:
                case binding_type::storage_buffer_rw:
                    slot.index = l.storage_blocks++;
                    break;
                case binding_type::storage_texture_rw:
                    slot.index = l.images++;
                    break;
                default:
                    break;
                }
            }
        }
        if (desc.push_constant_bytes) {
            l.push_slot = l.uniform_blocks++;
            l.push_bytes = std::min(desc.push_constant_bytes, max_push_constant_bytes);
        }
        if (l.uniform_blocks > max_uniform_bindings_) {
            report_error("CreatePipelineLayout: " + std::to_string(l.uniform_blocks) +
                         " uniform buffers are more than this OpenGL ES context binds");
            return {};
        }
        return pipeline_layouts_.alloc(std::move(l));
    }

    pipeline_handle device::create_graphics_pipeline(const graphics_pipeline_desc& desc) {
        return build_pipeline(&desc, nullptr);
    }

    pipeline_handle device::create_compute_pipeline(const compute_pipeline_desc& desc) {
        return build_pipeline(nullptr, &desc);
    }

    pipeline_handle device::build_pipeline(const graphics_pipeline_desc* graphics,
                                           const compute_pipeline_desc* compute) {
        const char* name = graphics ? graphics->name : compute->name;
        const std::string label = name ? name : "?";
        const pipeline_layout_handle layout_handle = graphics ? graphics->layout : compute->layout;
        const pipeline_layout* layout = pipeline_layouts_.get(layout_handle);
        if (!layout) {
            report_error("CreatePipeline (" + label + "): missing pipeline layout");
            return {};
        }
        if (compute && !features_.compute) {
            report_error("CreateComputePipeline (" + label + "): compute shaders need OpenGL ES 3.1");
            return {};
        }

        struct stage_source {
            const gles::shader* source = nullptr;
            translated_stage translated;
            GLenum type = 0;
        };
        std::vector<stage_source> stages;
        const auto add = [&](shader_handle h, GLenum type) {
            const shader* sh = shaders_.get(h);
            if (!sh)
                return false;
            stages.push_back({sh, {}, type});
            return true;
        };
        if (graphics) {
            if (!add(graphics->vertex, GL_VERTEX_SHADER)) {
                report_error("CreateGraphicsPipeline (" + label + "): missing vertex shader");
                return {};
            }
            if (graphics->fragment && !add(graphics->fragment, GL_FRAGMENT_SHADER)) {
                report_error("CreateGraphicsPipeline (" + label + "): unknown fragment shader");
                return {};
            }
        } else if (!add(compute->compute, GL_COMPUTE_SHADER)) {
            report_error("CreateComputePipeline (" + label + "): missing compute shader");
            return {};
        }

        // the slow part of building a pipeline runs here, on the caller's thread
        for (stage_source& s : stages) {
            std::string error;
            if (!translate_stage(s.source->words, s.source->stage, s.source->entry, features_.glsl_version, *layout,
                                 s.translated, error)) {
                report_error("shader translation failed (" + label + "): " + error);
                return {};
            }
        }
        if (std::getenv("CONDUIT_GLES_DUMP_SHADERS"))
            for (const stage_source& s : stages)
                std::fprintf(stderr, "[gles] %s:\n%s\n", label.c_str(), s.translated.glsl.c_str());

        pipeline p;
        p.compute = compute != nullptr;
        p.layout = layout_handle;
        if (graphics) {
            p.topology = graphics->topology == topology::triangle_strip ? GL_TRIANGLE_STRIP
                         : graphics->topology == topology::line_list    ? GL_LINES
                         : graphics->topology == topology::point_list   ? GL_POINTS
                                                                        : GL_TRIANGLES;
            pipeline_state& st = p.state;
            const raster_state& r = graphics->raster;
            st.cull_face = r.cull == cull_mode::back ? GL_BACK : r.cull == cull_mode::front ? GL_FRONT : 0;
            // the geometry is flipped in the vertex shader to match the top-down images the other backends make, which
            // mirrors the winding GL sees
            st.front_face = r.front_face == front_face::counter_clockwise ? GL_CW : GL_CCW;
            st.offset = r.depth_bias_constant != 0.0f || r.depth_bias_slope != 0.0f;
            st.offset_slope = r.depth_bias_slope;
            st.offset_units = r.depth_bias_constant;
            const depth_stencil_state& ds = graphics->depth_stencil;
            st.depth_test = ds.depth_test;
            st.depth_write = ds.depth_write;
            st.depth_func = ds.depth_test ? compare_func(ds.depth_compare) : GL_ALWAYS;
            st.stencil_test = ds.stencil_test;
            st.stencil_read = ds.stencil_read_mask;
            st.stencil_write = ds.stencil_write_mask;
            const auto face = [](const stencil_face& f) {
                return stencil_gl{stencil_operation(f.fail), stencil_operation(f.depth_fail), stencil_operation(f.pass),
                                  compare_func(f.compare)};
            };
            st.front = face(ds.front);
            st.back = face(ds.back);
            st.alpha_to_coverage = graphics->blend.alpha_to_coverage;
            p.color_count = std::min(graphics->color_count, max_color_targets);
            st.blend_count = p.color_count;
            for (uint32_t i = 0; i < p.color_count; i++) {
                const blend_target& t = graphics->blend.targets[i];
                blend_gl& b = st.blend[i];
                b.enable = t.enable;
                b.src_rgb = blend_factor_of(t.src_color);
                b.dst_rgb = blend_factor_of(t.dst_color);
                b.src_alpha = blend_factor_of(t.src_alpha);
                b.dst_alpha = blend_factor_of(t.dst_alpha);
                b.op_rgb = blend_equation_of(t.color_op);
                b.op_alpha = blend_equation_of(t.alpha_op);
                b.mask = static_cast<uint8_t>(t.write_mask);
            }
            if (!features_.indexed_blend)
                for (uint32_t i = 1; i < p.color_count; i++)
                    if (!(st.blend[i] == st.blend[0]))
                        report_once("OpenGL ES " + std::to_string(features_.major) + "." +
                                    std::to_string(features_.minor) +
                                    " blends every color target the same way: using the state of target 0 (" +
                                    label + ")");
            p.stream_count = std::min(graphics->stream_count, max_vertex_streams);
            p.streams = graphics->streams;
            p.attribute_count = std::min(graphics->attribute_count, max_vertex_attributes);
            for (uint32_t i = 0; i < p.attribute_count; i++) {
                const vertex_attribute_desc& a = graphics->attributes[i];
                vertex_attribute_gl& g = p.attributes[i];
                g.location = a.location;
                g.stream = a.stream;
                g.offset = a.offset;
                if (!vertex_format(a.format, g) || a.stream >= max_vertex_streams) {
                    report_error("CreateGraphicsPipeline (" + label + "): unsupported vertex format");
                    return {};
                }
            }
        }

        bool linked = thread_.call([&] {
            std::vector<GLuint> shaders;
            const auto fail = [&](const std::string& message) {
                report_error(message);
                for (GLuint s : shaders)
                    gl_.glDeleteShader(s);
                return false;
            };
            const auto compile = [&](GLenum type, const std::string& source) -> GLuint {
                const GLuint s = gl_.glCreateShader(type);
                const char* text = source.c_str();
                gl_.glShaderSource(s, 1, &text, nullptr);
                gl_.glCompileShader(s);
                GLint ok = 0;
                gl_.glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
                if (!ok) {
                    GLint length = 0;
                    gl_.glGetShaderiv(s, GL_INFO_LOG_LENGTH, &length);
                    std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
                    gl_.glGetShaderInfoLog(s, length, nullptr, log.data());
                    report_error("GLSL compile failed (" + label + "):\n" + log + "\n--- source ---\n" + source);
                    gl_.glDeleteShader(s);
                    return 0;
                }
                return s;
            };
            for (const stage_source& s : stages) {
                const GLuint id = compile(s.type, s.translated.glsl);
                if (!id)
                    return fail("CreatePipeline (" + label + "): a shader did not compile");
                shaders.push_back(id);
            }
            if (graphics && !graphics->fragment) {
                const GLuint id = compile(GL_FRAGMENT_SHADER, empty_fragment(features_.glsl_version));
                if (!id)
                    return fail("CreatePipeline (" + label + "): the empty fragment shader did not compile");
                shaders.push_back(id);
            }
            const GLuint program = gl_.glCreateProgram();
            for (GLuint s : shaders)
                gl_.glAttachShader(program, s);
            gl_.glLinkProgram(program);
            for (GLuint s : shaders)
                gl_.glDeleteShader(s);
            GLint ok = 0;
            gl_.glGetProgramiv(program, GL_LINK_STATUS, &ok);
            if (!ok) {
                GLint length = 0;
                gl_.glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
                std::string log(static_cast<size_t>(std::max(length, 1)), '\0');
                gl_.glGetProgramInfoLog(program, length, nullptr, log.data());
                gl_.glDeleteProgram(program);
                shaders.clear();
                report_error("GLSL link failed (" + label + "): " + log);
                return false;
            }
            p.program = program;

            gl_.glUseProgram(program);
            cache_.program = program;
            // block bindings by name; every stage that declares one agrees on its slot
            for (const stage_source& s : stages)
                for (const translated_block& b : s.translated.uniform_blocks) {
                    const GLuint index = gl_.glGetUniformBlockIndex(program, b.name.c_str());
                    if (index != GL_INVALID_INDEX)
                        gl_.glUniformBlockBinding(program, index, b.index);
                }
            // one unit per texture and sampler pair
            std::map<std::string, uint32_t> units;
            for (const stage_source& s : stages)
                for (const translated_sampler& t : s.translated.samplers) {
                    auto [it, added] = units.try_emplace(t.name, static_cast<uint32_t>(units.size()));
                    if (!added)
                        continue;
                    combined_sampler c;
                    c.unit = it->second;
                    c.set = t.set;
                    c.binding = t.binding;
                    c.sampler_set = t.sampler_set;
                    c.sampler_binding = t.sampler_binding;
                    c.dummy_sampler = t.dummy_sampler;
                    c.target = t.target;
                    p.samplers.push_back(c);
                    const GLint location = gl_.glGetUniformLocation(program, t.name.c_str());
                    if (location >= 0)
                        gl_.glUniform1i(location, static_cast<GLint>(c.unit));
                }
            return true;
        });
        if (!linked)
            return {};
        if (p.samplers.size() > max_texture_units_) {
            thread_.post([this, program = p.program] { gl_.glDeleteProgram(program); });
            report_error("CreatePipeline (" + label + "): " + std::to_string(p.samplers.size()) +
                         " texture and sampler pairs are more than this OpenGL ES context has units for");
            return {};
        }
        return pipelines_.alloc(std::move(p));
    }
}
