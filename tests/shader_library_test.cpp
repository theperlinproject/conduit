#include <conduit/shader_library.hpp>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string_view>

static int g_failed = 0;
#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);                                       \
            g_failed++;                                                                                                \
        }                                                                                                              \
    } while (0)

using namespace conduit;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: shader_library_test <pack>\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(!bytes.empty());

    shader_library lib;
    std::string error;
    CHECK(lib.load(bytes, &error));
    CHECK(lib.programs().size() == 2);

    const program_info* blit = lib.find("blit");
    CHECK(blit != nullptr);
    if (blit) {
        const uniform_member* tint = blit->find_uniform("u_tint");
        CHECK(tint && tint->offset == 0 && tint->bytes == 16);
        const uniform_member* transform = blit->find_uniform("u_transform");
        CHECK(transform && transform->bytes == 64);
        const uniform_member* lights = blit->find_uniform("u_lights");
        CHECK(lights && lights->elements == 4 && lights->bytes == 64);
        CHECK(blit->uniform_bytes == 160);
        for (shader_target t : {shader_target::spirv, shader_target::hlsl, shader_target::msl, shader_target::wgsl}) {
            CHECK(!lib.code(*blit, shader_stage::vertex, t).empty());
            CHECK(!lib.code(*blit, shader_stage::fragment, t).empty());
        }
        CHECK(lib.code(*blit, shader_stage::compute, shader_target::spirv).empty());

        const auto spirv = lib.code(*blit, shader_stage::fragment, shader_target::spirv);
        uint32_t magic = 0;
        if (spirv.size() >= 4)
            std::memcpy(&magic, spirv.data(), 4);
        CHECK(magic == 0x07230203u);
        const auto dxil = lib.code(*blit, shader_stage::fragment, shader_target::dxil);
        CHECK(dxil.empty() || (dxil.size() > 4 && std::memcmp(dxil.data(), "DXBC", 4) == 0));
    }

    const program_info* fill = lib.find("fill");
    CHECK(fill != nullptr);
    if (fill) {
        CHECK(!lib.code(*fill, shader_stage::compute, shader_target::spirv).empty());
        CHECK(lib.code(*fill, shader_stage::vertex, shader_target::spirv).empty());
        const auto msl = lib.code(*fill, shader_stage::compute, shader_target::msl);
        const std::string_view text(reinterpret_cast<const char*>(msl.data()), msl.size());
        CHECK(text.starts_with("// conduit:threads 64 1 1"));
    }

    for (const program_info& p : lib.programs()) {
        for (const uniform_member& u : p.uniforms) {
            CHECK(u.offset % 16 == 0);
            CHECK(u.offset + u.bytes <= p.uniform_bytes);
        }
    }

    const texture_slot* layers = lib.find_texture("s_layers");
    CHECK(layers && layers->slot == 0 && layers->array && !layers->unfilterable);
    const texture_slot* source = lib.find_texture("s_source");
    CHECK(source && source->slot == 1 && !source->array);
    const texture_slot* depth = lib.find_texture("s_depth");
    CHECK(depth && depth->unfilterable);
    CHECK(lib.find_texture("s_missing") == nullptr);

    CHECK(shader_library::target_for(api::d3d12) == shader_target::dxil);
    CHECK(shader_library::format_for(shader_target::hlsl) == shader_format::hlsl);

    std::vector<uint8_t> broken(bytes.begin(), bytes.begin() + 40);
    shader_library bad;
    CHECK(!bad.load(broken, &error));

    if (g_failed) {
        std::fprintf(stderr, "%d failure(s)\n", g_failed);
        return 1;
    }
    std::puts("shader_library_test: ok");
    return 0;
}
