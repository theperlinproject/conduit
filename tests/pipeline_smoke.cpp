#include <conduit/device.hpp>
#include <conduit/shader_library.hpp>

#include <cstdio>
#include <fstream>
#include <iterator>

using namespace conduit;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: pipeline_smoke <pack> [api]\n");
        return 2;
    }
    const conduit::api api = argc > 2 ? api_from_id(argv[2]) : conduit::api::vulkan;
    std::ifstream in(argv[1], std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    shader_library lib;
    std::string error;
    if (!lib.load(bytes, &error)) {
        std::fprintf(stderr, "pack: %s\n", error.c_str());
        return 1;
    }

    instance_options io;
    io.validation = true;
    auto instance = create_instance(api, io);
    if (!instance || instance->adapters().empty()) {
        std::puts("no driver: skipped");
        return 77;
    }
    device_options dopts;
    for (const auto& a : instance->adapters())
        if (a.usable)
            dopts.adapter_index = a.index;
    dopts.validation = true;
    auto dev = instance->create_device(dopts);
    if (!dev) {
        std::fprintf(stderr, "device creation failed\n");
        return 1;
    }

    const shader_target target = shader_library::target_for(api);
    std::vector<bind_layout_entry> uniform_entries{
        {0, binding_type::uniform_buffer_dynamic, shader_stage::all, 1, texture_dim::tex2d}};
    std::vector<bind_layout_entry> texture_entries;
    for (const texture_slot& s : lib.texture_slots()) {
        texture_entries.push_back({s.slot * 2, binding_type::sampled_texture,
                                   shader_stage::fragment | shader_stage::vertex, 1,
                                   s.array ? texture_dim::tex2d_array : texture_dim::tex2d});
        texture_entries.push_back({s.slot * 2 + 1, binding_type::sampler, shader_stage::fragment | shader_stage::vertex,
                                   1, texture_dim::tex2d});
    }
    bind_layout_desc uld;
    uld.entries = uniform_entries;
    bind_layout_desc tld;
    tld.entries = texture_entries;
    pipeline_layout_desc pld;
    pld.sets[0] = dev->create_bind_layout(uld);
    pld.sets[1] = dev->create_bind_layout(tld);
    pipeline_layout_handle layout = dev->create_pipeline_layout(pld);

    int failed = 0;
    for (const program_info& p : lib.programs()) {
        auto vs = lib.code(p, shader_stage::vertex, target);
        auto fs = lib.code(p, shader_stage::fragment, target);
        if (vs.empty() || fs.empty())
            continue;
        shader_desc sd;
        sd.format = shader_library::format_for(target);
        sd.stage = shader_stage::vertex;
        sd.code = vs;
        sd.entry = shader_library::entry_for(shader_stage::vertex, target);
        sd.name = p.name.c_str();
        shader_handle v = dev->create_shader(sd);
        sd.stage = shader_stage::fragment;
        sd.code = fs;
        sd.entry = shader_library::entry_for(shader_stage::fragment, target);
        shader_handle f = dev->create_shader(sd);

        graphics_pipeline_desc gd;
        gd.vertex = v;
        gd.fragment = f;
        gd.layout = layout;
        gd.streams[0] = {32, vertex_step::vertex};
        gd.stream_count = 1;
        for (uint32_t i = 0; i < 2; i++)
            gd.attributes[i] = {i, 0, format::rgba32_float, i * 16};
        gd.attribute_count = 2;
        gd.color_formats[0] = format::rgba8_unorm;
        gd.color_count = 1;
        pipeline_handle pipe = dev->create_graphics_pipeline(gd);
        const std::string report = dev->debug_report();
        if (!pipe || !report.empty()) {
            std::fprintf(stderr, "FAIL %s %s\n%s", p.name.c_str(), pipe ? "validation output" : "no pipeline",
                         report.c_str());
            failed++;
        }
        if (pipe)
            dev->destroy(pipe);
        dev->destroy(v);
        dev->destroy(f);
    }
    dev->wait_idle();
    if (failed) {
        std::fprintf(stderr, "%d program(s) failed\n", failed);
        return 1;
    }
    std::printf("pipeline_smoke ok (%zu programs)\n", lib.programs().size());
    return 0;
}
