#include "backends.hpp"
#include "common/handle_pool.hpp"
#include "common/retire_queue.hpp"

#include <cstdio>
#include <cstring>

namespace conduit {
    std::vector<std::string> null_device_take_errors(device& device);
}

static int g_failed = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                       \
            g_failed++;                                                                                                \
        }                                                                                                              \
    } while (0)

using namespace conduit;

static std::unique_ptr<device> make_device() {
    auto instance = create_instance(api::none);
    device_options options;
    options.state_validation = true;
    return instance->create_device(options);
}

static void test_handle_generations() {
    handle_pool<buffer_tag, int> pool;
    buffer_handle a = pool.alloc(7);
    CHECK(pool.get(a) && *pool.get(a) == 7);
    pool.free(a);
    CHECK(pool.get(a) == nullptr);
    buffer_handle b = pool.alloc(9);
    CHECK(b.index() == a.index());
    CHECK(b.generation() != a.generation());
    CHECK(pool.get(a) == nullptr);
    CHECK(pool.get(b) && *pool.get(b) == 9);
    CHECK(pool.get(buffer_handle{}) == nullptr);
}

static void test_retire() {
    retire_queue q;
    queue_serials submitted{};
    int freed = 0;
    submitted[0] = 1;
    q.push(submitted, [&] { freed++; });
    queue_serials done{};
    q.collect(done);
    CHECK(freed == 0);
    done[0] = 1;
    q.collect(done);
    CHECK(freed == 1);
}

static void test_copies() {
    auto dev = make_device();
    buffer_desc up;
    up.size = 256;
    up.usage = buffer_usage::copy_src;
    up.memory = memory_class::upload;
    up.name = "up";
    buffer_handle src = dev->create_buffer(up);
    buffer_desc rb = up;
    rb.usage = buffer_usage::copy_dst;
    rb.memory = memory_class::readback;
    rb.name = "rb";
    buffer_handle dst = dev->create_buffer(rb);
    buffer_desc gpu;
    gpu.size = 256;
    gpu.usage = buffer_usage::copy_src | buffer_usage::copy_dst;
    gpu.name = "gpu";
    buffer_handle mid = dev->create_buffer(gpu);

    uint8_t* p = dev->map(src);
    CHECK(p != nullptr);
    CHECK(dev->map(mid) == nullptr);
    for (int i = 0; i < 256; i++)
        p[i] = static_cast<uint8_t>(i);

    command_list* cl = dev->acquire_command_list(queue_type::graphics);
    cl->barrier(buffer_barrier{mid, state::undefined, state::copy_dst});
    cl->copy_buffer(mid, 0, src, 0, 256);
    cl->barrier(buffer_barrier{mid, state::copy_dst, state::copy_src});
    cl->copy_buffer(dst, 0, mid, 0, 256);
    dev->submit(queue_type::graphics, cl);
    CHECK(std::memcmp(dev->map(dst), p, 256) == 0);
    CHECK(null_device_take_errors(*dev).empty());

    cl = dev->acquire_command_list(queue_type::graphics);
    cl->barrier(buffer_barrier{mid, state::shader_write, state::copy_src});
    dev->submit(queue_type::graphics, cl);
    CHECK(!null_device_take_errors(*dev).empty());

    cl = dev->acquire_command_list(queue_type::graphics);
    cl->barrier(buffer_barrier{mid, state::copy_src, state::shader_read});
    cl->copy_buffer(mid, 0, src, 0, 16);
    dev->submit(queue_type::graphics, cl);
    CHECK(!null_device_take_errors(*dev).empty());
}

static void test_texture_round_trip() {
    auto dev = make_device();
    texture_desc td;
    td.format = format::rgba8_unorm;
    td.width = 8;
    td.height = 4;
    td.usage = texture_usage::sampled | texture_usage::copy_dst | texture_usage::copy_src;
    td.name = "tex";
    texture_handle tex = dev->create_texture(td);

    buffer_desc up;
    up.size = 8 * 4 * 4;
    up.usage = buffer_usage::copy_src | buffer_usage::copy_dst;
    up.memory = memory_class::upload;
    buffer_handle in = dev->create_buffer(up);
    buffer_handle out = dev->create_buffer(up);
    uint8_t* p = dev->map(in);
    for (uint32_t i = 0; i < up.size; i++)
        p[i] = static_cast<uint8_t>(i * 7);

    buffer_texture_copy region;
    region.width = 8;
    region.height = 4;
    command_list* cl = dev->acquire_command_list(queue_type::graphics);
    cl->barrier(texture_barrier{tex, state::undefined, state::copy_dst});
    cl->copy_buffer_to_texture(tex, in, std::span(&region, 1));
    cl->barrier(texture_barrier{tex, state::copy_dst, state::copy_src});
    cl->copy_texture_to_buffer(out, tex, std::span(&region, 1));
    dev->submit(queue_type::graphics, cl);
    CHECK(std::memcmp(dev->map(out), p, up.size) == 0);
    CHECK(null_device_take_errors(*dev).empty());
}

static void test_draw_validation() {
    auto dev = make_device();
    const uint8_t blob[4] = {};
    shader_desc sd;
    sd.code = blob;
    shader_handle vs = dev->create_shader(sd);
    pipeline_layout_handle layout = dev->create_pipeline_layout({});
    graphics_pipeline_desc gp;
    gp.vertex = vs;
    gp.layout = layout;
    gp.color_formats[0] = format::bgra8_unorm;
    gp.color_count = 1;
    pipeline_handle pipe = dev->create_graphics_pipeline(gp);

    texture_desc td;
    td.format = format::bgra8_unorm;
    td.width = td.height = 16;
    td.usage = texture_usage::color_target;
    texture_handle rt = dev->create_texture(td);

    command_list* cl = dev->acquire_command_list(queue_type::graphics);
    cl->barrier(texture_barrier{rt, state::undefined, state::color_target});
    rendering_desc rd;
    rd.color_count = 1;
    rd.colors[0].texture = rt;
    cl->begin_rendering(rd);
    cl->set_pipeline(pipe);
    cl->draw(3, 1, 0, 0);
    cl->end_rendering();
    dev->submit(queue_type::graphics, cl);
    CHECK(null_device_take_errors(*dev).empty());
    CHECK(dev->stats().draws == 1);

    cl = dev->acquire_command_list(queue_type::graphics);
    rendering_desc empty;
    cl->begin_rendering(empty);
    cl->set_pipeline(pipe);
    cl->draw(3, 1, 0, 0);
    cl->end_rendering();
    dev->submit(queue_type::graphics, cl);
    CHECK(!null_device_take_errors(*dev).empty());
}

static void test_deferred_destroy() {
    auto dev = make_device();
    buffer_desc d;
    d.size = 16;
    d.usage = buffer_usage::copy_dst;
    buffer_handle b = dev->create_buffer(d);
    dev->destroy(b);
    dev->wait_idle();
    CHECK(dev->get_desc(b) == nullptr);
    CHECK(dev->stats().live_buffers == 0);
}

static void test_swapchain() {
    auto dev = make_device();
    swapchain_desc sd;
    sd.width = 64;
    sd.height = 32;
    auto sc = dev->create_swapchain(sd);
    swapchain::image img;
    CHECK(sc->acquire(img));
    CHECK(!sc->resize(0, 0));
    CHECK(!sc->acquire(img));
    CHECK(sc->resize(128, 64));
    CHECK(sc->acquire(img));
}

int main() {
    test_handle_generations();
    test_retire();
    test_copies();
    test_texture_round_trip();
    test_draw_validation();
    test_deferred_destroy();
    test_swapchain();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::puts("null_selftest ok");
    return 0;
}
