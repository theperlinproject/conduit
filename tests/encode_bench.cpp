// cpu cost of recording and submitting a draw-heavy frame, per api: the numbers conduit's own overhead moves
#include <conduit/device.hpp>
#include <conduit/shader_library.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace conduit;
using clock_type = std::chrono::steady_clock;

namespace {
    struct scene {
        std::unique_ptr<instance> inst;
        std::unique_ptr<device> dev;
        texture_handle target;
        std::vector<texture_handle> textures;
        sampler_handle sampler;
        buffer_handle uniforms, vertices;
        bind_layout_handle uniform_layout, texture_layout;
        pipeline_layout_handle layout;
        std::vector<pipeline_handle> pipelines;
        bind_group_handle uniform_group;
        std::vector<bind_group_handle> texture_groups;
        std::vector<bind_group_entry> texture_entries_template;
    };

    double median(std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    }

    bool build(scene& s, api which, const shader_library& lib) {
        instance_options io;
        s.inst = create_instance(which, io);
        if (!s.inst || s.inst->adapters().empty())
            return false;
        device_options dopts;
        for (const auto& a : s.inst->adapters())
            if (a.usable) {
                dopts.adapter_index = a.index;
                break;
            }
        s.dev = s.inst->create_device(dopts);
        if (!s.dev)
            return false;
        device& d = *s.dev;

        texture_desc td;
        td.format = format::rgba8_unorm;
        td.width = td.height = 256;
        td.usage = texture_usage::color_target;
        td.name = "target";
        s.target = d.create_texture(td);
        for (int i = 0; i < 8; i++) {
            texture_desc t;
            t.format = format::rgba8_unorm;
            t.width = t.height = 16;
            t.usage = texture_usage::sampled;
            t.dim = i % 2 ? texture_dim::tex2d : texture_dim::tex2d_array;
            s.textures.push_back(d.create_texture(t));
        }
        s.sampler = d.create_sampler({});

        buffer_desc ub;
        ub.size = 256ull * 8192;
        ub.usage = buffer_usage::uniform;
        ub.memory = memory_class::upload;
        s.uniforms = d.create_buffer(ub);
        buffer_desc vb;
        vb.size = 32ull * 3 * 8192;
        vb.usage = buffer_usage::vertex;
        vb.memory = memory_class::upload;
        s.vertices = d.create_buffer(vb);

        const bind_layout_entry uniform_entries[] = {
            {0, binding_type::uniform_buffer_dynamic, shader_stage::all, 1, texture_dim::tex2d}};
        std::vector<bind_layout_entry> texture_entries;
        for (const texture_slot& t : lib.texture_slots()) {
            texture_entries.push_back({t.slot * 2, binding_type::sampled_texture,
                                       shader_stage::fragment | shader_stage::vertex, 1,
                                       t.array ? texture_dim::tex2d_array : texture_dim::tex2d});
            texture_entries.push_back({t.slot * 2 + 1, binding_type::sampler,
                                       shader_stage::fragment | shader_stage::vertex, 1, texture_dim::tex2d});
        }
        bind_layout_desc uld;
        uld.entries = uniform_entries;
        s.uniform_layout = d.create_bind_layout(uld);
        bind_layout_desc tld;
        tld.entries = texture_entries;
        s.texture_layout = d.create_bind_layout(tld);
        pipeline_layout_desc pld;
        pld.sets[0] = s.uniform_layout;
        pld.sets[1] = s.texture_layout;
        s.layout = d.create_pipeline_layout(pld);

        const shader_target target = shader_library::target_for(which);
        const program_info* blit = lib.find("blit");
        if (which != api::none) {
            if (!blit)
                return false;
            const auto vs = lib.code(*blit, shader_stage::vertex, target);
            const auto fs = lib.code(*blit, shader_stage::fragment, target);
            if (vs.empty() || fs.empty())
                return false;
            shader_desc sd;
            sd.format = shader_library::format_for(target);
            sd.stage = shader_stage::vertex;
            sd.code = vs;
            sd.entry = shader_library::entry_for(shader_stage::vertex, target);
            const shader_handle v = d.create_shader(sd);
            sd.stage = shader_stage::fragment;
            sd.code = fs;
            sd.entry = shader_library::entry_for(shader_stage::fragment, target);
            const shader_handle f = d.create_shader(sd);
            for (int i = 0; i < 2; i++) {
                graphics_pipeline_desc gd;
                gd.vertex = v;
                gd.fragment = f;
                gd.layout = s.layout;
                gd.streams[0] = {32, vertex_step::vertex};
                gd.stream_count = 1;
                for (uint32_t a = 0; a < 2; a++)
                    gd.attributes[a] = {a, 0, format::rgba32_float, a * 16};
                gd.attribute_count = 2;
                gd.color_formats[0] = format::rgba8_unorm;
                gd.color_count = 1;
                gd.blend.targets[0].enable = i == 1;
                gd.name = i ? "blit-blend" : "blit";
                s.pipelines.push_back(d.create_graphics_pipeline(gd));
                if (!s.pipelines.back())
                    return false;
            }
        } else {
            for (int i = 0; i < 2; i++) {
                graphics_pipeline_desc gd;
                gd.layout = s.layout;
                gd.stream_count = 1;
                gd.streams[0] = {32, vertex_step::vertex};
                gd.color_formats[0] = format::rgba8_unorm;
                gd.color_count = 1;
                s.pipelines.push_back(d.create_graphics_pipeline(gd));
            }
        }

        bind_group_entry ue;
        ue.binding = 0;
        ue.buffer = {s.uniforms, 0, 256};
        bind_group_desc ug;
        ug.layout = s.uniform_layout;
        ug.entries = std::span<const bind_group_entry>(&ue, 1);
        s.uniform_group = d.create_bind_group(ug);
        for (int g = 0; g < 8; g++) {
            std::vector<bind_group_entry> entries;
            for (const texture_slot& t : lib.texture_slots()) {
                bind_group_entry te;
                te.binding = t.slot * 2;
                te.texture.texture = s.textures[t.array ? (g & ~1) : (g | 1)];
                te.texture.view_dim = t.array ? texture_dim::tex2d_array : texture_dim::tex2d;
                entries.push_back(te);
                bind_group_entry se;
                se.binding = t.slot * 2 + 1;
                se.sampler = s.sampler;
                entries.push_back(se);
            }
            s.texture_entries_template = entries;
            bind_group_desc tg;
            tg.layout = s.texture_layout;
            tg.entries = entries;
            s.texture_groups.push_back(d.create_bind_group(tg));
        }
        // transition the sampled textures once, as a renderer would after uploading them
        command_list* list = d.acquire_command_list(queue_type::graphics);
        std::vector<texture_barrier> barriers;
        for (texture_handle t : s.textures)
            barriers.push_back({t, state::undefined, state::shader_read_fragment, {}});
        barriers.push_back({s.target, state::undefined, state::color_target, {}});
        list->barrier(barriers, {});
        d.submit(queue_type::graphics, list);
        d.wait_idle();
        return true;
    }

    // one pass of draws the way a world renderer issues them: a dynamic uniform offset and a vertex range per
    // draw, a material change every 8 draws and a pipeline change every 64
    void record(scene& s, command_list& list, uint32_t draws) {
        rendering_desc rd;
        rd.colors[0].texture = s.target;
        rd.colors[0].load = load_op::clear;
        rd.color_count = 1;
        list.begin_rendering(rd);
        for (uint32_t i = 0; i < draws; i++) {
            if (i % 64 == 0)
                list.set_pipeline(s.pipelines[(i / 64) & 1]);
            if (i % 8 == 0)
                list.set_bind_group(1, s.texture_groups[(i / 8) % s.texture_groups.size()]);
            const uint32_t offset = (i % 8192) * 256;
            list.set_bind_group(0, s.uniform_group, std::span<const uint32_t>(&offset, 1));
            list.set_vertex_buffer(0, s.vertices, static_cast<uint64_t>(i % 8192) * 96);
            list.draw(3, 1, 0, 0);
        }
        list.end_rendering();
    }

    struct result {
        double record_ns_per_draw = 0;
        double submit_us = 0;
        double frame_us = 0;
        double bind_group_create_ns = 0;
    };

    result run(scene& s, uint32_t draws, int frames) {
        device& d = *s.dev;
        std::vector<double> rec, sub, frame;
        for (int f = 0; f < frames + 20; f++) {
            const auto t0 = clock_type::now();
            d.begin_frame();
            command_list* list = d.acquire_command_list(queue_type::graphics);
            const auto t1 = clock_type::now();
            record(s, *list, draws);
            const auto t2 = clock_type::now();
            d.submit(queue_type::graphics, list);
            d.end_frame();
            const auto t3 = clock_type::now();
            if (f >= 20) {
                rec.push_back(std::chrono::duration<double, std::nano>(t2 - t1).count() / draws);
                sub.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());
                frame.push_back(std::chrono::duration<double, std::micro>(t3 - t0).count());
            }
        }
        d.wait_idle();

        // per-frame bind groups, the way transient materials and ui batches make them
        std::vector<double> create;
        for (int f = 0; f < 40; f++) {
            d.begin_frame();
            std::vector<bind_group_handle> made;
            const auto t0 = clock_type::now();
            for (int i = 0; i < 256; i++) {
                bind_group_desc tg;
                tg.layout = s.texture_layout;
                tg.entries = s.texture_entries_template;
                tg.per_frame = true;
                made.push_back(d.create_bind_group(tg));
            }
            const auto t1 = clock_type::now();
            for (bind_group_handle h : made)
                d.destroy(h);
            d.end_frame();
            create.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count() / 256);
        }
        d.wait_idle();
        return {median(rec), median(sub), median(frame), median(create)};
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: encode_bench <pack> [api ...] [--draws N] [--frames N]\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    shader_library lib;
    std::string error;
    if (!lib.load(bytes, &error)) {
        std::fprintf(stderr, "pack: %s\n", error.c_str());
        return 1;
    }
    uint32_t draws = 8000;
    int frames = 200;
    std::vector<api> apis;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--draws") && i + 1 < argc)
            draws = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc)
            frames = std::atoi(argv[++i]);
        else
            apis.push_back(api_from_id(argv[i]));
    }
    if (apis.empty()) {
        apis = compiled_apis();
        apis.push_back(api::none);
    }
    for (api which : apis) {
        scene s;
        if (!build(s, which, lib)) {
            std::printf("%-8s unavailable\n", api_id(which));
            continue;
        }
        const result r = run(s, draws, frames);
        const std::string report = s.dev->debug_report();
        std::printf("%-8s %-28s record %7.1f ns/draw  submit %8.1f us  frame %9.1f us  bind group %7.1f ns%s\n",
                    api_id(which), s.dev->caps().adapter.name.c_str(), r.record_ns_per_draw, r.submit_us, r.frame_us,
                    r.bind_group_create_ns, report.empty() ? "" : "  (errors reported)");
        if (!report.empty())
            std::fprintf(stderr, "%s\n", report.substr(0, 2000).c_str());
    }
    return 0;
}
