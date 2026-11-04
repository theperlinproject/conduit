#include <conduit/device.hpp>
#include "common/util.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <thread>

using namespace conduit;

static int g_failed = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                       \
            g_failed++;                                                                                                \
        }                                                                                                              \
    } while (0)

static std::string g_shader_dir;
static api g_api = api::vulkan;

static std::vector<uint8_t> load_file(const std::string& name) {
    std::ifstream in(g_shader_dir + "/" + name, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

struct ctx {
    std::unique_ptr<conduit::instance> instance;
    std::unique_ptr<device> dev;
};

static void expect_no_errors(device& dev, const char* where) {
    const std::string report = dev.debug_report();
    if (!report.empty()) {
        std::fprintf(stderr, "FAIL %s produced validation output:\n%s", where, report.c_str());
        g_failed++;
    }
}

static buffer_handle make_buffer(device& dev, uint64_t size, buffer_usage usage, memory_class mem, const char* name) {
    buffer_desc d;
    d.size = size;
    d.usage = usage;
    d.memory = mem;
    d.name = name;
    if (has(usage, buffer_usage::storage))
        d.stride = 4;
    return dev.create_buffer(d);
}

static shader_handle make_shader(device& dev, const char* file, shader_stage stage) {
    std::string path = file;
    path.resize(path.size() - 3);
    path += g_api == api::d3d12 ? "dxil" : "spv";
    auto code = load_file(path);
    shader_desc sd;
    sd.format = g_api == api::d3d12 ? shader_format::dxil : shader_format::spirv;
    sd.stage = stage;
    sd.code = code;
    sd.name = file;
    return dev.create_shader(sd);
}

static void test_buffer_round_trip(device& dev) {
    buffer_handle up = make_buffer(dev, 1024, buffer_usage::copy_src, memory_class::upload, "up");
    buffer_handle mid =
        make_buffer(dev, 1024, buffer_usage::copy_src | buffer_usage::copy_dst, memory_class::gpu_only, "mid");
    buffer_handle rb = make_buffer(dev, 1024, buffer_usage::copy_dst, memory_class::readback, "rb");
    uint8_t* p = dev.map(up);
    CHECK(p != nullptr);
    CHECK(dev.map(mid) == nullptr);
    for (int i = 0; i < 1024; i++)
        p[i] = static_cast<uint8_t>(i * 13);
    dev.flush(up, 0, 1024);

    command_list* cl = dev.acquire_command_list(queue_type::graphics);
    cl->barrier(buffer_barrier{mid, state::undefined, state::copy_dst});
    cl->copy_buffer(mid, 0, up, 0, 1024);
    cl->barrier(buffer_barrier{mid, state::copy_dst, state::copy_src});
    cl->copy_buffer(rb, 0, mid, 0, 1024);
    dev.submit(queue_type::graphics, cl);
    dev.wait_idle();
    dev.invalidate(rb, 0, 1024);
    CHECK(std::memcmp(dev.map(rb), p, 1024) == 0);
    expect_no_errors(dev, "buffer round trip");
    dev.destroy(up);
    dev.destroy(mid);
    dev.destroy(rb);
}

static void test_fill_and_fences(device& dev) {
    buffer_handle gpu =
        make_buffer(dev, 256, buffer_usage::copy_src | buffer_usage::copy_dst, memory_class::gpu_only, "fill");
    buffer_handle rb = make_buffer(dev, 256, buffer_usage::copy_dst, memory_class::readback, "fill-rb");
    fence_handle fence = dev.create_fence(0);

    command_list* cl = dev.acquire_command_list(queue_type::graphics);
    cl->barrier(buffer_barrier{gpu, state::undefined, state::copy_dst});
    cl->fill_buffer(gpu, 0, 256, 0xA5A5A5A5u);
    cl->barrier(buffer_barrier{gpu, state::copy_dst, state::copy_src});
    cl->copy_buffer(rb, 0, gpu, 0, 256);
    fence_op signal{fence, 7};
    submit_desc sd;
    sd.lists = std::span<command_list* const>(&cl, 1);
    sd.signals = std::span(&signal, 1);
    dev.submit(queue_type::graphics, sd);
    CHECK(dev.wait_fence(fence, 7, 5'000'000'000ull));
    CHECK(dev.fence_value(fence) >= 7);
    dev.invalidate(rb, 0, 256);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(dev.map(rb));
    CHECK(words[0] == 0xA5A5A5A5u && words[63] == 0xA5A5A5A5u);
    expect_no_errors(dev, "fill and fences");
    dev.destroy(fence);
    dev.destroy(gpu);
    dev.destroy(rb);
}

struct scene {
    shader_handle vs, fs;
    sampler_handle sampler;
    bind_layout_handle layout;
    pipeline_layout_handle pipe_layout;
    pipeline_handle pipe_front, pipe_back;
    buffer_handle vertices, instances, indices, ubo;
    texture_handle tex, target;
    bind_group_handle group;
};

static scene build_scene(device& dev) {
    scene s;
    s.vs = make_shader(dev, "raster.vs.spv", shader_stage::vertex);
    s.fs = make_shader(dev, "raster.fs.spv", shader_stage::fragment);
    s.sampler = dev.create_sampler({});

    const bind_layout_entry entries[] = {
        {0, binding_type::uniform_buffer_dynamic, shader_stage::vertex},
        {1, binding_type::sampled_texture, shader_stage::fragment},
        {2, binding_type::sampler, shader_stage::fragment},
    };
    s.layout = dev.create_bind_layout({entries});
    pipeline_layout_desc pl;
    pl.sets[0] = s.layout;
    pl.push_constant_bytes = 16;
    pl.push_constant_stages = shader_stage::fragment;
    s.pipe_layout = dev.create_pipeline_layout(pl);

    graphics_pipeline_desc gp;
    gp.vertex = s.vs;
    gp.fragment = s.fs;
    gp.layout = s.pipe_layout;
    gp.streams[0] = {16, vertex_step::vertex};
    gp.streams[1] = {8, vertex_step::instance};
    gp.stream_count = 2;
    gp.attributes[0] = {0, 0, format::rg32_float, 0};
    gp.attributes[1] = {1, 0, format::rg32_float, 8};
    gp.attributes[2] = {2, 1, format::rg32_float, 0};
    gp.attribute_count = 3;
    gp.raster.cull = cull_mode::back;
    gp.color_formats[0] = format::rgba8_unorm;
    gp.color_count = 1;
    gp.name = "front-ccw";
    s.pipe_front = dev.create_graphics_pipeline(gp);
    gp.raster.front_face = front_face::clockwise;
    gp.name = "front-cw";
    s.pipe_back = dev.create_graphics_pipeline(gp);

    const float verts[] = {-0.2f, -0.2f, 0, 0, 0.2f, -0.2f, 1, 0, 0.0f, 0.2f, 0.5f, 1};
    const float inst[8][2] = {{-0.75f, -0.5f}, {-0.25f, -0.5f}, {0.25f, -0.5f}, {0.75f, -0.5f},
                              {-0.75f, 0.5f},  {-0.25f, 0.5f},  {0.25f, 0.5f},  {0.75f, 0.5f}};
    const uint16_t idx[] = {0, 1, 2};
    const float frame[] = {1.0f, 1.0f, 0, 0};

    buffer_handle staging = make_buffer(dev, 4096, buffer_usage::copy_src, memory_class::upload, "staging");
    uint8_t* st = dev.map(staging);
    std::memcpy(st + 0, verts, sizeof(verts));
    std::memcpy(st + 256, inst, sizeof(inst));
    std::memcpy(st + 512, idx, sizeof(idx));
    std::memset(st + 1024, 0, 256);
    std::memcpy(st + 1024, frame, sizeof(frame));
    const float frame2[] = {2.0f, 2.0f, 0, 0};
    std::memcpy(st + 1280, frame2, sizeof(frame2));
    const uint8_t white[16] = {255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255};
    std::memset(st + 2048, 0, 512);
    std::memcpy(st + 2048, white, 8);
    std::memcpy(st + 2048 + 256, white, 8);
    dev.flush(staging, 0, 4096);

    s.vertices = make_buffer(dev, sizeof(verts), buffer_usage::vertex | buffer_usage::copy_dst, memory_class::gpu_only,
                             "vertices");
    s.instances = make_buffer(dev, sizeof(inst), buffer_usage::vertex | buffer_usage::copy_dst, memory_class::gpu_only,
                              "instances");
    s.indices = make_buffer(dev, 16, buffer_usage::index | buffer_usage::copy_dst, memory_class::gpu_only, "indices");
    s.ubo = make_buffer(dev, 512, buffer_usage::uniform | buffer_usage::copy_dst, memory_class::gpu_only, "ubo");

    texture_desc td;
    td.format = format::rgba8_unorm;
    td.width = td.height = 2;
    td.usage = texture_usage::sampled | texture_usage::copy_dst;
    td.name = "white";
    s.tex = dev.create_texture(td);
    td.width = td.height = 64;
    td.usage = texture_usage::color_target | texture_usage::copy_src;
    td.name = "target";
    s.target = dev.create_texture(td);

    const bind_group_entry ge[] = {
        {0, 0, {s.ubo, 0, 16}, {}, {}},
        {1, 0, {}, {s.tex}, {}},
        {2, 0, {}, {}, s.sampler},
    };
    bind_group_desc gd;
    gd.layout = s.layout;
    gd.entries = ge;
    s.group = dev.create_bind_group(gd);

    command_list* cl = dev.acquire_command_list(queue_type::graphics);
    const buffer_barrier to_copy[] = {{s.vertices, state::undefined, state::copy_dst},
                                      {s.instances, state::undefined, state::copy_dst},
                                      {s.indices, state::undefined, state::copy_dst},
                                      {s.ubo, state::undefined, state::copy_dst}};
    cl->barrier({}, to_copy);
    cl->barrier(texture_barrier{s.tex, state::undefined, state::copy_dst});
    cl->copy_buffer(s.vertices, 0, staging, 0, sizeof(verts));
    cl->copy_buffer(s.instances, 0, staging, 256, sizeof(inst));
    cl->copy_buffer(s.indices, 0, staging, 512, 16);
    cl->copy_buffer(s.ubo, 0, staging, 1024, 512);
    buffer_texture_copy region;
    region.buffer_offset = 2048;
    region.buffer_row_pitch = 256;
    region.width = region.height = 2;
    cl->copy_buffer_to_texture(s.tex, staging, std::span(&region, 1));
    const buffer_barrier to_use[] = {{s.vertices, state::copy_dst, state::vertex_buffer},
                                     {s.instances, state::copy_dst, state::vertex_buffer},
                                     {s.indices, state::copy_dst, state::index_buffer},
                                     {s.ubo, state::copy_dst, state::uniform_buffer}};
    cl->barrier({}, to_use);
    cl->barrier(texture_barrier{s.tex, state::copy_dst, state::shader_read_fragment});
    dev.submit(queue_type::graphics, cl);
    dev.wait_idle();
    dev.destroy(staging);
    return s;
}

static void read_target(device& dev, scene& s, std::vector<uint8_t>& out) {
    buffer_handle rb = make_buffer(dev, 64 * 256, buffer_usage::copy_dst, memory_class::readback, "target-rb");
    command_list* cl = dev.acquire_command_list(queue_type::graphics);
    cl->barrier(texture_barrier{s.target, state::color_target, state::copy_src});
    buffer_texture_copy r;
    r.buffer_row_pitch = 256;
    r.width = r.height = 64;
    cl->copy_texture_to_buffer(rb, s.target, std::span(&r, 1));
    cl->barrier(texture_barrier{s.target, state::copy_src, state::color_target});
    dev.submit(queue_type::graphics, cl);
    dev.wait_idle();
    dev.invalidate(rb, 0, 64 * 256);
    out.assign(dev.map(rb), dev.map(rb) + 64 * 256);
    dev.destroy(rb);
}

static const uint8_t* pixel(const std::vector<uint8_t>& img, int x, int y) {
    return &img[static_cast<size_t>(y) * 256 + x * 4];
}

static const uint8_t* instance_pixel(const std::vector<uint8_t>& img, int i) {
    const float cx = -0.75f + 0.5f * (i % 4);
    const float cy = i < 4 ? -0.5f : 0.5f;
    return pixel(img, static_cast<int>((cx + 1.0f) * 32.0f), static_cast<int>((1.0f - cy) * 32.0f));
}

static void begin_pass(command_list* cl, scene& s) {
    rendering_desc rd;
    rd.color_count = 1;
    rd.colors[0].texture = s.target;
    rd.colors[0].load = load_op::clear;
    cl->begin_rendering(rd);
}

static void test_raster(device& dev) {
    scene s = build_scene(dev);
    CHECK(s.pipe_front && s.pipe_back && s.group);
    const float tint[4] = {1.0f, 0.5f, 0.25f, 1.0f};
    const uint32_t offsets[] = {0};

    for (int variant = 0; variant < 2; variant++) {
        command_list* cl = dev.acquire_command_list(queue_type::graphics);
        cl->barrier(
            texture_barrier{s.target, variant == 0 ? state::undefined : state::color_target, state::color_target});
        begin_pass(cl, s);
        cl->set_pipeline(variant == 0 ? s.pipe_front : s.pipe_back);
        cl->set_bind_group(0, s.group, offsets);
        cl->push_constants(tint, sizeof(tint));
        cl->set_vertex_buffer(0, s.vertices);
        cl->set_vertex_buffer(1, s.instances);
        cl->set_index_buffer(s.indices, index_type::u16);
        cl->draw_indexed(3, 8, 0, 0, 0);
        cl->end_rendering();
        dev.submit(queue_type::graphics, cl);

        std::vector<uint8_t> img;
        read_target(dev, s, img);
        if (variant == 0) {
            for (int i = 0; i < 8; i++) {
                const uint8_t* p = instance_pixel(img, i);
                CHECK(p[0] == 255 && p[1] >= 127 && p[1] <= 129 && p[2] >= 63 && p[2] <= 65 && p[3] == 255);
            }
            CHECK(pixel(img, 1, 1)[3] == 0);
            CHECK(pixel(img, 8, 48 - 4)[0] == 255);
            CHECK(pixel(img, 8, 48 + 7)[0] == 0 || pixel(img, 8, 48 + 7)[0] == 255);
        } else {
            for (int i = 0; i < 8; i++)
                CHECK(instance_pixel(img, i)[3] == 0);
        }
    }

    {
        command_list* cl = dev.acquire_command_list(queue_type::graphics);
        cl->barrier(texture_barrier{s.target, state::color_target, state::color_target});
        begin_pass(cl, s);
        cl->set_pipeline(s.pipe_front);
        const uint32_t second[] = {256};
        cl->set_bind_group(0, s.group, second);
        cl->push_constants(tint, sizeof(tint));
        cl->set_vertex_buffer(0, s.vertices);
        cl->set_vertex_buffer(1, s.instances);
        cl->set_index_buffer(s.indices, index_type::u16);
        cl->draw_indexed(3, 1, 0, 0, 0);
        cl->end_rendering();
        dev.submit(queue_type::graphics, cl);
        std::vector<uint8_t> img;
        read_target(dev, s, img);
        CHECK(pixel(img, 8 - 4, 48 + 2)[0] == 255);
    }

    {
        const device_stats before = dev.stats();
        command_list* cl = dev.acquire_command_list(queue_type::graphics);
        cl->barrier(texture_barrier{s.target, state::color_target, state::color_target});
        begin_pass(cl, s);
        const uint32_t second[] = {256};
        for (int i = 0; i < 2; i++) {
            cl->set_pipeline(s.pipe_front);
            cl->set_bind_group(0, s.group, second);
            cl->push_constants(tint, sizeof(tint));
            cl->set_vertex_buffer(0, s.vertices);
            cl->set_vertex_buffer(1, s.instances);
            cl->set_index_buffer(s.indices, index_type::u16);
            cl->draw_indexed(3, 1, 0, 0, 0);
        }
        cl->end_rendering();
        dev.submit(queue_type::graphics, cl);
        const device_stats after = dev.stats();
        CHECK(after.draws - before.draws == 2);
        CHECK(after.pipeline_binds - before.pipeline_binds == 1);
        CHECK(after.bind_group_binds - before.bind_group_binds == 1);
        CHECK(after.submits - before.submits == 1);
        std::vector<uint8_t> img;
        read_target(dev, s, img);
        CHECK(pixel(img, 8 - 4, 48 + 2)[0] == 255);
    }

    expect_no_errors(dev, "raster");
    for (auto h : {s.vertices, s.instances, s.indices, s.ubo})
        dev.destroy(h);
    dev.destroy(s.tex);
    dev.destroy(s.target);
    dev.destroy(s.group);
    dev.destroy(s.pipe_front);
    dev.destroy(s.pipe_back);
    dev.destroy(s.pipe_layout);
    dev.destroy(s.layout);
    dev.destroy(s.vs);
    dev.destroy(s.fs);
    dev.wait_idle();
    CHECK(dev.stats().live_buffers == 0);
    CHECK(dev.stats().live_textures == 0);
}

static void test_compute_indirect(device& dev) {
    scene s = build_scene(dev);
    shader_handle cs = make_shader(dev, "cull.cs.spv", shader_stage::compute);
    const bind_layout_entry entries[] = {{0, binding_type::storage_buffer_rw, shader_stage::compute},
                                         {1, binding_type::storage_buffer_rw, shader_stage::compute}};
    bind_layout_handle cl_layout = dev.create_bind_layout({entries});
    pipeline_layout_desc pl;
    pl.sets[0] = cl_layout;
    pipeline_layout_handle cpl = dev.create_pipeline_layout(pl);
    compute_pipeline_desc cd;
    cd.compute = cs;
    cd.layout = cpl;
    pipeline_handle cpipe = dev.create_compute_pipeline(cd);
    CHECK(cpipe);

    const uint64_t arg_bytes = sizeof(draw_indexed_indirect_command) * 8;
    buffer_handle args =
        make_buffer(dev, arg_bytes, buffer_usage::storage | buffer_usage::indirect | buffer_usage::copy_dst,
                    memory_class::gpu_only, "args");
    buffer_handle counter =
        make_buffer(dev, 16, buffer_usage::storage | buffer_usage::indirect | buffer_usage::copy_dst,
                    memory_class::gpu_only, "counter");
    const bind_group_entry ge[] = {{0, 0, {args, 0, arg_bytes}, {}, {}}, {1, 0, {}, {}, {}}};
    bind_group_entry group[2] = {ge[0], ge[1]};
    group[1].buffer = {counter, 0, 16};
    bind_group_desc gd;
    gd.layout = cl_layout;
    gd.entries = group;
    bind_group_handle cg = dev.create_bind_group(gd);

    const float tint[4] = {1.0f, 0.5f, 0.25f, 1.0f};
    const uint32_t offsets[] = {0};
    command_list* cl = dev.acquire_command_list(queue_type::graphics);
    cl->barrier(texture_barrier{s.target, state::undefined, state::color_target});
    const buffer_barrier zero[] = {{args, state::undefined, state::copy_dst},
                                   {counter, state::undefined, state::copy_dst}};
    cl->barrier({}, zero);
    cl->fill_buffer(args, 0, arg_bytes, 0);
    cl->fill_buffer(counter, 0, 16, 0);
    const buffer_barrier to_shader[] = {{args, state::copy_dst, state::shader_write},
                                        {counter, state::copy_dst, state::shader_write}};
    cl->barrier({}, to_shader);
    cl->set_pipeline(cpipe);
    cl->set_bind_group(0, cg, {});
    cl->dispatch(1, 1, 1);
    const buffer_barrier to_draw[] = {{args, state::shader_write, state::indirect_argument},
                                      {counter, state::shader_write, state::indirect_argument}};
    cl->barrier({}, to_draw);
    begin_pass(cl, s);
    cl->set_pipeline(s.pipe_front);
    cl->set_bind_group(0, s.group, offsets);
    cl->push_constants(tint, sizeof(tint));
    cl->set_vertex_buffer(0, s.vertices);
    cl->set_vertex_buffer(1, s.instances);
    cl->set_index_buffer(s.indices, index_type::u16);
    if (dev.caps().features.draw_indirect_count)
        cl->draw_indexed_indirect_count(args, 0, counter, 0, 8, sizeof(draw_indexed_indirect_command));
    else
        cl->draw_indexed_indirect(args, 0, 8, sizeof(draw_indexed_indirect_command));
    cl->end_rendering();
    dev.submit(queue_type::graphics, cl);

    std::vector<uint8_t> img;
    read_target(dev, s, img);
    for (int i = 0; i < 8; i++) {
        const bool drawn = i == 0 || i == 2 || i == 5;
        CHECK((instance_pixel(img, i)[3] == 255) == drawn);
    }
    expect_no_errors(dev, "compute indirect");
    std::printf("  indirect path: %s\n",
                dev.caps().features.draw_indirect_count ? "draw-indirect-count" : "multi-draw-indirect");

    dev.destroy(cg);
    dev.destroy(args);
    dev.destroy(counter);
    dev.destroy(cpipe);
    dev.destroy(cpl);
    dev.destroy(cl_layout);
    dev.destroy(cs);
    for (auto h : {s.vertices, s.instances, s.indices, s.ubo})
        dev.destroy(h);
    dev.destroy(s.tex);
    dev.destroy(s.target);
    dev.destroy(s.group);
    dev.destroy(s.pipe_front);
    dev.destroy(s.pipe_back);
    dev.destroy(s.pipe_layout);
    dev.destroy(s.layout);
    dev.destroy(s.vs);
    dev.destroy(s.fs);
    dev.wait_idle();
}

static void test_native_texture(device& dev) {
    texture_desc td;
    td.width = 64;
    td.height = 32;
    td.format = format::rgba8_unorm;
    td.usage = texture_usage::sampled | texture_usage::color_target;
    td.name = "native";
    texture_handle t = dev.create_texture(td);
    const native_texture n = dev.native_of(t, state::shader_read_fragment);
    CHECK(n.resource != nullptr);
    CHECK(n.width == 64 && n.height == 32 && n.mips == 1);
    CHECK(n.native_format != 0);
    if (g_api == api::vulkan) {
        CHECK(n.view != nullptr && n.memory != nullptr);
        CHECK(n.state == 5);
    }
    dev.destroy(t);
    CHECK(dev.native_of(t, state::shader_read_fragment).resource == nullptr);
    expect_no_errors(dev, "native texture");
}

static void test_frames_and_retire(device& dev) {
    for (int f = 0; f < 9; f++) {
        dev.begin_frame();
        buffer_handle b = make_buffer(dev, 4096, buffer_usage::copy_dst, memory_class::gpu_only, "frame-temp");
        buffer_handle up = make_buffer(dev, 4096, buffer_usage::copy_src, memory_class::upload, "frame-up");
        command_list* cl = dev.acquire_command_list(queue_type::graphics);
        cl->barrier(buffer_barrier{b, state::undefined, state::copy_dst});
        cl->copy_buffer(b, 0, up, 0, 4096);
        dev.submit(queue_type::graphics, cl);
        dev.destroy(b);
        dev.destroy(up);
        dev.end_frame();
    }
    dev.wait_idle();
    CHECK(dev.stats().live_buffers == 0);
    expect_no_errors(dev, "frames");
}

static void test_threaded_recording(device& dev) {
    constexpr int thread_count = 4;
    buffer_handle src = make_buffer(dev, 256, buffer_usage::copy_src, memory_class::upload, "t-src");
    std::memset(dev.map(src), 0x5A, 256);
    buffer_handle dst[thread_count];
    command_list* lists[thread_count];
    for (auto& i : dst)
        i = make_buffer(dev, 256, buffer_usage::copy_dst, memory_class::readback, "t-dst");
    dev.begin_frame();
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (int i = 0; i < thread_count; i++)
        threads.emplace_back([&, i] {
            lists[i] = dev.acquire_command_list(queue_type::graphics);
            lists[i]->barrier(buffer_barrier{dst[i], state::undefined, state::copy_dst});
            lists[i]->copy_buffer(dst[i], 0, src, 0, 256);
        });
    for (auto& t : threads)
        t.join();
    submit_desc sd;
    sd.lists = std::span<command_list* const>(lists, thread_count);
    dev.submit(queue_type::graphics, sd);
    dev.end_frame();
    dev.wait_idle();
    for (auto i : dst) {
        dev.invalidate(i, 0, 256);
        CHECK(dev.map(i)[0] == 0x5A && dev.map(i)[255] == 0x5A);
        dev.destroy(i);
    }
    dev.destroy(src);
    expect_no_errors(dev, "threaded recording");
}

static void test_validator_catches_mistakes(device& dev) {
    buffer_handle b =
        make_buffer(dev, 64, buffer_usage::copy_dst | buffer_usage::copy_src, memory_class::gpu_only, "v");
    command_list* cl = dev.acquire_command_list(queue_type::graphics);
    cl->barrier(buffer_barrier{b, state::undefined, state::copy_dst});
    cl->barrier(buffer_barrier{b, state::vertex_buffer, state::copy_src});
    dev.submit(queue_type::graphics, cl);
    dev.wait_idle();
    CHECK(!dev.debug_report().empty());
    dev.destroy(b);
}

static void test_queries_and_cache(device& dev) {
    if (dev.caps().features.timestamp_queries) {
        query_set_handle q = dev.create_query_set({4, "ts"});
        buffer_handle rb = make_buffer(dev, 64, buffer_usage::copy_dst, memory_class::readback, "ts-rb");
        command_list* cl = dev.acquire_command_list(queue_type::graphics);
        cl->write_timestamp(q, 0);
        cl->write_timestamp(q, 1);
        cl->barrier(buffer_barrier{rb, state::undefined, state::copy_dst});
        cl->resolve_queries(q, 0, 2, rb, 0);
        dev.submit(queue_type::graphics, cl);
        dev.wait_idle();
        dev.invalidate(rb, 0, 64);
        const uint64_t* t = reinterpret_cast<const uint64_t*>(dev.map(rb));
        CHECK(t[1] >= t[0]);
        dev.destroy(q);
        dev.destroy(rb);
    }
    const auto blob = dev.serialize_pipeline_cache();
    std::printf("  pipeline cache blob: %zu bytes\n", blob.size());
    expect_no_errors(dev, "queries");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: gpu_selftest <shader dir> [vulkan|d3d12]\n");
        return 2;
    }
    g_shader_dir = argv[1];
    if (argc > 2)
        g_api = api_from_id(argv[2]);

    instance_options io;
    io.validation = true;
    auto instance = create_instance(g_api, io);
    if (!instance || instance->adapters().empty()) {
        std::printf("no %s driver on this machine: skipped\n", api_name(g_api));
        return 77;
    }
    uint32_t pick = 0;
    for (const auto& a : instance->adapters()) {
        std::printf("adapter %u: %s (%s) usable=%d %s\n", a.index, a.name.c_str(), vendor_name(a.vendor_id), a.usable,
                    a.unusable_reason.c_str());
        if (a.usable)
            pick = a.index;
    }
    device_options dopts;
    dopts.adapter_index = pick;
    dopts.validation = true;
    dopts.state_validation = true;
    auto dev = instance->create_device(dopts);
    if (!dev) {
        std::fprintf(stderr, "device creation failed\n");
        return 1;
    }

    test_buffer_round_trip(*dev);
    test_fill_and_fences(*dev);
    test_raster(*dev);
    test_compute_indirect(*dev);
    test_native_texture(*dev);
    test_frames_and_retire(*dev);
    test_threaded_recording(*dev);
    test_validator_catches_mistakes(*dev);
    test_queries_and_cache(*dev);

    dev->wait_idle();
    expect_no_errors(*dev, "shutdown");
    dev.reset();
    instance.reset();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::puts("gpu_selftest ok");
    return 0;
}
