#include "gles/gles_internal.hpp"

#include <spirv_glsl.hpp>

// conduit builds its own SPIRV-Cross in a private namespace (SPIRV_CROSS_NAMESPACE_OVERRIDE) so it can link next to an
// application that ships a different SPIRV-Cross build
namespace sc = SPIRV_CROSS_NAMESPACE;

#include <set>

// Conduit shaders are SPIR-V written for Vulkan. OpenGL ES wants GLSL ES, with its bindings assigned by the program
// rather than by the shader, so each stage is rewritten with SPIRV-Cross after the pipeline layout has been applied:
//   - every (set, binding) becomes a uniform block binding, storage block binding or image unit of the layout
//   - texture and sampler pairs become combined samplers named after the pair, so the pipeline can find them
//   - push constants become a uniform block on a slot the layout reserves
//   - clip space is converted from the D3D convention (z in 0..1, y up into a top-down image) to GL's
namespace conduit::gles {
    namespace {
        spv::ExecutionModel model_of(shader_stage stage) {
            switch (stage) {
            case shader_stage::fragment:
                return spv::ExecutionModelFragment;
            case shader_stage::compute:
                return spv::ExecutionModelGLCompute;
            default:
                return spv::ExecutionModelVertex;
            }
        }

        char suffix_of(shader_stage stage) {
            return stage == shader_stage::fragment ? 'f' : stage == shader_stage::compute ? 'c' : 'v';
        }

        bool is_uniform(binding_type t) {
            return t == binding_type::uniform_buffer || t == binding_type::uniform_buffer_dynamic;
        }
    }

    bool translate_stage(const std::vector<uint32_t>& words, shader_stage stage, const std::string& entry,
                         uint32_t glsl_version, const pipeline_layout& layout, translated_stage& out,
                         std::string& error) {
        try {
            sc::CompilerGLSL compiler(words);
            const spv::ExecutionModel model = model_of(stage);

            std::string entry_name = entry;
            bool found = false;
            std::string first_of_model;
            for (const auto& ep : compiler.get_entry_points_and_stages()) {
                if (ep.execution_model != model)
                    continue;
                if (first_of_model.empty())
                    first_of_model = ep.name;
                if (ep.name == entry_name)
                    found = true;
            }
            if (!found) {
                if (first_of_model.empty()) {
                    error = "the shader has no entry point for this stage";
                    return false;
                }
                entry_name = first_of_model;
            }
            compiler.set_entry_point(entry_name, model);

            sc::CompilerGLSL::Options options;
            options.es = true;
            options.version = glsl_version;
            options.enable_420pack_extension = false;
            options.separate_shader_objects = false;
            options.emit_push_constant_as_uniform_buffer = true;
            options.force_flattened_io_blocks = true;
            options.vertex.fixup_clipspace = true;
            options.vertex.flip_vert_y = true;
            options.vertex.support_nonzero_base_instance = false;
            options.fragment.default_float_precision = sc::CompilerGLSL::Options::Highp;
            options.fragment.default_int_precision = sc::CompilerGLSL::Options::Highp;
            compiler.set_common_options(options);

            const sc::ShaderResources resources = compiler.get_shader_resources();
            const char suffix = suffix_of(stage);

            const auto slot_of = [&](const sc::Resource& r, uint32_t& set, uint32_t& binding) {
                set = compiler.get_decoration(r.id, spv::DecorationDescriptorSet);
                binding = compiler.get_decoration(r.id, spv::DecorationBinding);
                return layout.find(set, binding);
            };
            const auto missing = [&](const char* what, uint32_t set, uint32_t binding) {
                error = std::string("the shader uses ") + what + " at set " + std::to_string(set) + " binding " +
                        std::to_string(binding) + " that the pipeline layout does not have, or has as another type";
                return false;
            };

            for (const sc::Resource& ub : resources.uniform_buffers) {
                uint32_t set = 0, binding = 0;
                const layout_slot* slot = slot_of(ub, set, binding);
                if (!slot || !is_uniform(slot->type))
                    return missing("a uniform buffer", set, binding);
                const std::string name = "cdt_ub" + std::to_string(slot->index) + suffix;
                compiler.set_decoration(ub.id, spv::DecorationBinding, slot->index);
                compiler.unset_decoration(ub.id, spv::DecorationDescriptorSet);
                compiler.set_name(ub.base_type_id, name);
                compiler.set_name(ub.id, name + "_i");
                out.uniform_blocks.push_back({name, slot->index});
            }

            for (const sc::Resource& pc : resources.push_constant_buffers) {
                if (layout.push_slot == ~0u) {
                    error = "the shader uses push constants but the pipeline layout reserves none";
                    return false;
                }
                const std::string name = std::string("cdt_push_") + suffix;
                compiler.set_decoration(pc.id, spv::DecorationBinding, layout.push_slot);
                compiler.set_name(pc.base_type_id, name);
                compiler.set_name(pc.id, name + "_i");
                out.uniform_blocks.push_back({name, layout.push_slot});
                out.uses_push = true;
                out.push_bytes = std::max<uint32_t>(
                    out.push_bytes, static_cast<uint32_t>(compiler.get_declared_struct_size(compiler.get_type(pc.base_type_id))));
            }

            for (const sc::Resource& sb : resources.storage_buffers) {
                if (glsl_version < 310) {
                    error = "storage buffers need OpenGL ES 3.1";
                    return false;
                }
                uint32_t set = 0, binding = 0;
                const layout_slot* slot = slot_of(sb, set, binding);
                if (!slot || (slot->type != binding_type::storage_buffer && slot->type != binding_type::storage_buffer_rw))
                    return missing("a storage buffer", set, binding);
                compiler.set_decoration(sb.id, spv::DecorationBinding, slot->index);
                compiler.unset_decoration(sb.id, spv::DecorationDescriptorSet);
                const std::string name = "cdt_sb" + std::to_string(slot->index) + suffix;
                compiler.set_name(sb.base_type_id, name);
                compiler.set_name(sb.id, name + "_i");
                out.needs_storage = true;
            }

            for (const sc::Resource& img : resources.storage_images) {
                if (glsl_version < 310) {
                    error = "storage images need OpenGL ES 3.1";
                    return false;
                }
                uint32_t set = 0, binding = 0;
                const layout_slot* slot = slot_of(img, set, binding);
                if (!slot || slot->type != binding_type::storage_texture_rw)
                    return missing("a storage texture", set, binding);
                compiler.set_decoration(img.id, spv::DecorationBinding, slot->index);
                compiler.unset_decoration(img.id, spv::DecorationDescriptorSet);
                out.needs_storage = true;
            }

            if (!resources.subpass_inputs.empty()) {
                error = "subpass inputs are not available in OpenGL ES";
                return false;
            }
            const auto target_of = [&](const sc::SPIRType& type, GLenum& target) {
                if (type.image.ms) {
                    error = "multisampled textures cannot be sampled in OpenGL ES here";
                    return false;
                }
                switch (type.image.dim) {
                case spv::Dim2D:
                    target = type.image.arrayed ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
                    return true;
                case spv::DimCube:
                    if (type.image.arrayed) {
                        error = "cube map arrays are not available in OpenGL ES 3.0 and 3.1";
                        return false;
                    }
                    target = GL_TEXTURE_CUBE_MAP;
                    return true;
                case spv::Dim3D:
                    target = GL_TEXTURE_3D;
                    return true;
                case spv::DimBuffer:
                    error = "texel buffers are not available in OpenGL ES 3.0 and 3.1";
                    return false;
                default:
                    error = "unsupported texture dimension";
                    return false;
                }
            };

            std::set<std::string> seen;
            // a texture that is already combined with its sampler in the shader: it follows conduit's numbering, the
            // sampler sits at the binding after the texture
            for (const sc::Resource& si : resources.sampled_images) {
                translated_sampler s;
                s.set = compiler.get_decoration(si.id, spv::DecorationDescriptorSet);
                s.binding = compiler.get_decoration(si.id, spv::DecorationBinding);
                s.sampler_set = s.set;
                s.sampler_binding = s.binding + 1;
                if (!target_of(compiler.get_type(si.type_id), s.target))
                    return false;
                const layout_slot* slot = layout.find(s.set, s.binding);
                if (!slot || slot->type != binding_type::sampled_texture)
                    return missing("a sampled texture", s.set, s.binding);
                const layout_slot* sampler_slot = layout.find(s.sampler_set, s.sampler_binding);
                if (!sampler_slot || sampler_slot->type != binding_type::sampler)
                    return missing("a sampler", s.sampler_set, s.sampler_binding);
                s.name = "cdt_s" + std::to_string(s.set) + "_" + std::to_string(s.binding) + "_" +
                         std::to_string(s.sampler_set) + "_" + std::to_string(s.sampler_binding);
                compiler.set_name(si.id, s.name);
                compiler.unset_decoration(si.id, spv::DecorationBinding);
                compiler.unset_decoration(si.id, spv::DecorationDescriptorSet);
                if (seen.insert(s.name).second)
                    out.samplers.push_back(std::move(s));
            }

            // the texture of each combined sampler, so its GL texture target is known
            std::map<uint32_t, GLenum> targets;
            for (const sc::Resource& img : resources.separate_images) {
                GLenum target = GL_TEXTURE_2D;
                if (!target_of(compiler.get_type(img.type_id), target))
                    return false;
                targets[img.id] = target;
            }

            const sc::VariableID dummy = compiler.build_dummy_sampler_for_combined_images();
            compiler.build_combined_image_samplers();
            for (const sc::CombinedImageSampler& cs : compiler.get_combined_image_samplers()) {
                translated_sampler s;
                s.set = compiler.get_decoration(cs.image_id, spv::DecorationDescriptorSet);
                s.binding = compiler.get_decoration(cs.image_id, spv::DecorationBinding);
                s.dummy_sampler = dummy != 0 && cs.sampler_id == dummy;
                if (!s.dummy_sampler) {
                    s.sampler_set = compiler.get_decoration(cs.sampler_id, spv::DecorationDescriptorSet);
                    s.sampler_binding = compiler.get_decoration(cs.sampler_id, spv::DecorationBinding);
                }
                const auto target = targets.find(cs.image_id);
                s.target = target != targets.end() ? target->second : GL_TEXTURE_2D;
                s.name = "cdt_s" + std::to_string(s.set) + "_" + std::to_string(s.binding) + "_" +
                         (s.dummy_sampler ? std::string("d")
                                          : std::to_string(s.sampler_set) + "_" + std::to_string(s.sampler_binding));
                const layout_slot* slot = layout.find(s.set, s.binding);
                if (!slot || slot->type != binding_type::sampled_texture)
                    return missing("a sampled texture", s.set, s.binding);
                if (!s.dummy_sampler) {
                    const layout_slot* sampler_slot = layout.find(s.sampler_set, s.sampler_binding);
                    if (!sampler_slot || sampler_slot->type != binding_type::sampler)
                        return missing("a sampler", s.sampler_set, s.sampler_binding);
                }
                compiler.set_name(cs.combined_id, s.name);
                compiler.unset_decoration(cs.combined_id, spv::DecorationBinding);
                compiler.unset_decoration(cs.combined_id, spv::DecorationDescriptorSet);
                if (seen.insert(s.name).second)
                    out.samplers.push_back(std::move(s));
            }

            // before ES 3.1 the stages meet by name rather than by location, so a varying is named after its location
            if (stage == shader_stage::vertex)
                for (const sc::Resource& r : resources.stage_outputs)
                    compiler.set_name(r.id, "cdt_v" + std::to_string(compiler.get_decoration(r.id, spv::DecorationLocation)));
            if (stage == shader_stage::fragment)
                for (const sc::Resource& r : resources.stage_inputs)
                    compiler.set_name(r.id, "cdt_v" + std::to_string(compiler.get_decoration(r.id, spv::DecorationLocation)));

            out.glsl = compiler.compile();
            return true;
        } catch (const sc::CompilerError& e) {
            error = std::string("SPIRV-Cross: ") + e.what();
        } catch (const std::exception& e) {
            error = std::string("shader translation failed: ") + e.what();
        }
        return false;
    }
}
