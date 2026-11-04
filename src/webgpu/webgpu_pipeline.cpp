#include <cmath>
#include "webgpu/webgpu_internal.hpp"

namespace conduit::webgpu {
    namespace {
        WGPUPrimitiveTopology to_topology(topology t) {
            switch (t) {
            case topology::triangle_list:
                return WGPUPrimitiveTopology_TriangleList;
            case topology::triangle_strip:
                return WGPUPrimitiveTopology_TriangleStrip;
            case topology::line_list:
                return WGPUPrimitiveTopology_LineList;
            case topology::point_list:
                return WGPUPrimitiveTopology_PointList;
            }
            return WGPUPrimitiveTopology_TriangleList;
        }

        WGPUBlendComponent component(blend_op op, blend_factor src, blend_factor dst) {
            WGPUBlendComponent c;
            c.operation = to_blend_op(op);
            c.srcFactor = to_blend_factor(src);
            c.dstFactor = to_blend_factor(dst);
            if (op == blend_op::minimum || op == blend_op::maximum)
                c.srcFactor = c.dstFactor = WGPUBlendFactor_One;
            return c;
        }

        WGPUStencilFaceState face(const stencil_face& f) {
            return {to_compare(f.compare), to_stencil_op(f.fail), to_stencil_op(f.depth_fail), to_stencil_op(f.pass)};
        }
    }

    pipeline_handle webgpu_device::create_graphics_pipeline(const graphics_pipeline_desc& desc) {
        const pipeline_layout* pl = pipeline_layouts_.get(desc.layout);
        const shader* vs = shaders_.get(desc.vertex);
        const shader* fs = shaders_.get(desc.fragment);
        if (!pl || !vs) {
            report_error("CreateGraphicsPipeline: missing layout or vertex shader");
            return {};
        }
        pipeline p;
        p.layout_handle = desc.layout;
        std::array<WGPUBindGroupLayout, max_bind_groups> layouts{};
        const shader* stages[2] = {vs, fs};
        uint32_t set_count = 0;
        for (uint32_t s = 0; s < max_bind_groups; s++) {
            if (!pl->desc.sets[s])
                continue;
            p.sets[s] = build_native_set(pl->desc, s, stages, shader_stage::all_graphics);
            if (!p.sets[s])
                return {};
            set_count = s + 1;
        }
        for (uint32_t s = 0; s < set_count; s++)
            layouts[s] = p.sets[s] ? p.sets[s]->layout.get() : empty_layout_.get();
        WGPUPipelineLayoutDescriptor pld{};
        pld.label = view(desc.name);
        pld.bindGroupLayoutCount = set_count;
        pld.bindGroupLayouts = layouts.data();
        p.layout = owned<WGPUPipelineLayout>(wgpuDeviceCreatePipelineLayout(device_.get(), &pld));
        if (!p.layout)
            return {};

        std::array<std::vector<WGPUVertexAttribute>, max_vertex_streams> attributes;
        for (uint32_t i = 0; i < desc.attribute_count; i++) {
            const vertex_attribute_desc& a = desc.attributes[i];
            const WGPUVertexFormat vf = to_vertex_format(a.format);
            if (vf == static_cast<WGPUVertexFormat>(0) || a.stream >= desc.stream_count) {
                report_error("CreateGraphicsPipeline: unusable vertex attribute");
                return {};
            }
            WGPUVertexAttribute attr{};
            attr.format = vf;
            attr.offset = a.offset;
            attr.shaderLocation = a.location;
            attributes[a.stream].push_back(attr);
        }
        std::array<WGPUVertexBufferLayout, max_vertex_streams> streams{};
        for (uint32_t i = 0; i < desc.stream_count; i++) {
            streams[i].stepMode =
                desc.streams[i].step == vertex_step::instance ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
            streams[i].arrayStride = desc.streams[i].stride;
            streams[i].attributeCount = attributes[i].size();
            streams[i].attributes = attributes[i].data();
        }
        p.stream_count = desc.stream_count;
        p.set_count = set_count;

        WGPURenderPipelineDescriptor rd{};
        rd.label = view(desc.name);
        rd.layout = p.layout.get();
        rd.vertex.module = vs->module.get();
        rd.vertex.entryPoint = view(vs->entry);
        rd.vertex.bufferCount = desc.stream_count;
        rd.vertex.buffers = streams.data();

        rd.primitive.topology = to_topology(desc.topology);
        rd.primitive.frontFace = desc.raster.front_face == front_face::clockwise ? WGPUFrontFace_CW : WGPUFrontFace_CCW;
        rd.primitive.cullMode = desc.raster.cull == cull_mode::none   ? WGPUCullMode_None
                                : desc.raster.cull == cull_mode::back ? WGPUCullMode_Back
                                                                      : WGPUCullMode_Front;
        rd.primitive.unclippedDepth = !desc.raster.depth_clip && caps_.features.depth_clip_control;

        WGPUDepthStencilState ds{};
        if (desc.depth_format != format::unknown) {
            ds.format = to_texture_format(desc.depth_format);
            ds.depthWriteEnabled = desc.depth_stencil.depth_write ? WGPUOptionalBool_True : WGPUOptionalBool_False;
            ds.depthCompare = desc.depth_stencil.depth_test ? to_compare(desc.depth_stencil.depth_compare)
                                                            : WGPUCompareFunction_Always;
            if (desc.depth_stencil.stencil_test) {
                ds.stencilFront = face(desc.depth_stencil.front);
                ds.stencilBack = face(desc.depth_stencil.back);
            } else {
                ds.stencilFront = ds.stencilBack = {WGPUCompareFunction_Always, WGPUStencilOperation_Keep,
                                                    WGPUStencilOperation_Keep, WGPUStencilOperation_Keep};
            }
            ds.stencilReadMask = desc.depth_stencil.stencil_read_mask;
            ds.stencilWriteMask = desc.depth_stencil.stencil_test ? desc.depth_stencil.stencil_write_mask : 0;
            ds.depthBias = static_cast<int32_t>(std::lround(desc.raster.depth_bias_constant));
            ds.depthBiasSlopeScale = desc.raster.depth_bias_slope;
            ds.depthBiasClamp = desc.raster.depth_bias_clamp;
            rd.depthStencil = &ds;
        }

        rd.multisample.count = (std::max)(1u, desc.samples);
        rd.multisample.mask = 0xFFFFFFFFu;
        rd.multisample.alphaToCoverageEnabled = desc.blend.alpha_to_coverage && rd.multisample.count > 1;

        std::array<WGPUColorTargetState, max_color_targets> targets{};
        std::array<WGPUBlendState, max_color_targets> blends{};
        WGPUFragmentState fragment{};
        if (fs) {
            for (uint32_t i = 0; i < desc.color_count; i++) {
                const blend_target& bt = desc.blend.targets[i];
                targets[i].format = to_texture_format(desc.color_formats[i]);
                targets[i].writeMask = static_cast<WGPUColorWriteMask>(bt.write_mask);
                if (bt.enable) {
                    blends[i].color = component(bt.color_op, bt.src_color, bt.dst_color);
                    blends[i].alpha = component(bt.alpha_op, bt.src_alpha, bt.dst_alpha);
                    targets[i].blend = &blends[i];
                }
            }
            fragment.module = fs->module.get();
            fragment.entryPoint = view(fs->entry);
            fragment.targetCount = desc.color_count;
            fragment.targets = targets.data();
            rd.fragment = &fragment;
        }

        p.render = owned<WGPURenderPipeline>(wgpuDeviceCreateRenderPipeline(device_.get(), &rd));
        if (!p.render)
            return {};
        stats_.live_pipelines++;
        return pipelines_.alloc(std::move(p));
    }

    pipeline_handle webgpu_device::create_compute_pipeline(const compute_pipeline_desc& desc) {
        const pipeline_layout* pl = pipeline_layouts_.get(desc.layout);
        const shader* cs = shaders_.get(desc.compute);
        if (!pl || !cs) {
            report_error("CreateComputePipeline: missing layout or shader");
            return {};
        }
        pipeline p;
        p.compute = true;
        p.layout_handle = desc.layout;
        std::array<WGPUBindGroupLayout, max_bind_groups> layouts{};
        const shader* stages[1] = {cs};
        uint32_t set_count = 0;
        for (uint32_t s = 0; s < max_bind_groups; s++) {
            if (!pl->desc.sets[s])
                continue;
            p.sets[s] = build_native_set(pl->desc, s, stages, shader_stage::compute);
            if (!p.sets[s])
                return {};
            set_count = s + 1;
        }
        for (uint32_t s = 0; s < set_count; s++)
            layouts[s] = p.sets[s] ? p.sets[s]->layout.get() : empty_layout_.get();
        WGPUPipelineLayoutDescriptor pld{};
        pld.label = view(desc.name);
        pld.bindGroupLayoutCount = set_count;
        pld.bindGroupLayouts = layouts.data();
        p.layout = owned<WGPUPipelineLayout>(wgpuDeviceCreatePipelineLayout(device_.get(), &pld));
        if (!p.layout)
            return {};
        WGPUComputePipelineDescriptor cd{};
        cd.label = view(desc.name);
        cd.layout = p.layout.get();
        cd.compute.module = cs->module.get();
        cd.compute.entryPoint = view(cs->entry);
        p.set_count = set_count;
        p.comp = owned<WGPUComputePipeline>(wgpuDeviceCreateComputePipeline(device_.get(), &cd));
        if (!p.comp)
            return {};
        stats_.live_pipelines++;
        return pipelines_.alloc(std::move(p));
    }
}
