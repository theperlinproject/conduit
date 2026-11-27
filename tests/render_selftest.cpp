// renders 256 scissored tiles whose colours encode the bind group, texture, uniform offset and vertex range each
// draw used, reads them back and checks every one: what a redundant-state filter or a group cache gets wrong
// shows up as a wrong tile
#include <conduit/device.hpp>
#include <conduit/shader_library.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace conduit;

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
            t.usage = texture_usage::sampled | texture_usage::copy_dst;
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

    struct rgba {
        float r, g, b, a;
    };
    struct test {
        scene& s;
        device& d;
        texture_handle target;
        buffer_handle readback, staging;
        std::vector<texture_handle> tex;
        std::vector<rgba> color;
        std::vector<bind_group_handle> groups;
        int failures = 0;

        uint8_t q(float v) { return static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); }
        void fill(texture_handle t, rgba c, bool now = true) {
            uint8_t* p = d.map(staging);
            for (int y = 0; y < 16; y++)
                for (int x = 0; x < 16; x++) {
                    uint8_t* px = p + y * 256 + x * 4;
                    px[0] = q(c.r), px[1] = q(c.g), px[2] = q(c.b), px[3] = q(c.a);
                }
            d.flush(staging, 0, 4096);
            command_list* l = d.acquire_command_list(queue_type::graphics);
            l->barrier(texture_barrier{t, state::undefined, state::copy_dst, {}});
            buffer_texture_copy r;
            r.buffer_row_pitch = 256;
            r.width = r.height = 16;
            l->copy_buffer_to_texture(t, staging, std::span<const buffer_texture_copy>(&r, 1));
            l->barrier(texture_barrier{t, state::copy_dst, state::shader_read_fragment, {}});
            d.submit(queue_type::graphics, l);
            if (now)
                d.wait_idle();
        }
        texture_handle make_tex(int k, rgba c) {
            texture_desc t;
            t.format = format::rgba8_unorm;
            t.width = t.height = 16;
            t.usage = texture_usage::sampled | texture_usage::copy_dst;
            t.dim = k % 2 ? texture_dim::tex2d : texture_dim::tex2d_array;
            const texture_handle h = d.create_texture(t);
            fill(h, c);
            return h;
        }
        // group made of layered texture a (array) and source texture b (2d)
        bind_group_handle make_group(texture_handle a, texture_handle b, bool per_frame) {
            std::vector<bind_group_entry> e = s.texture_entries_template;
            for (auto& x : e)
                if (x.texture.texture)
                    x.texture.texture = x.texture.view_dim == texture_dim::tex2d_array ? a : b;
            bind_group_desc tg;
            tg.layout = s.texture_layout;
            tg.entries = e;
            tg.per_frame = per_frame;
            return d.create_bind_group(tg);
        }
        void uniforms(uint32_t i, rgba tint) {
            float* u = reinterpret_cast<float*>(d.map(s.uniforms) + i * 256);
            const int c = i % 16, r = i / 16;
            const float t[4] = {tint.r, tint.g, tint.b, tint.a};
            const float rc[4] = {-1.0f + c * 0.125f, 1.0f - (r + 1) * 0.125f, 0.125f, 0.125f};
            std::memcpy(u, t, 16);
            std::memcpy(u + 4, rc, 16);
            for (int k = 0; k < 16; k++)
                u[8 + k] = k % 5 == 0 ? 1.0f : 0.0f;
            for (int k = 0; k < 16; k++)
                u[24 + k] = k == 0 || k == 5 || k == 10 ? 1.0f : 0.0f; // lights sum to white
            u[24 + 3] = u[24 + 7] = 0;
        }
        void begin_pass(command_list& l) {
            rendering_desc rd;
            rd.colors[0].texture = target;
            rd.colors[0].load = load_op::clear;
            rd.color_count = 1;
            l.begin_rendering(rd);
        }
        void check(command_list* l, const std::vector<rgba>& want, const char* what) {
            buffer_texture_copy r;
            r.buffer_row_pitch = 1024;
            r.width = r.height = 256;
            l->barrier(texture_barrier{target, state::color_target, state::copy_src, {}});
            l->copy_texture_to_buffer(readback, target, std::span<const buffer_texture_copy>(&r, 1));
            l->barrier(texture_barrier{target, state::copy_src, state::color_target, {}});
            d.submit(queue_type::graphics, l);
            d.end_frame();
            d.wait_idle();
            d.invalidate(readback, 0, 1024 * 256);
            const uint8_t* p = d.map(readback);
            int bad = 0;
            for (int i = 0; i < 256; i++) {
                const int px = (i % 16) * 16 + 8, py = (i / 16) * 16 + 8;
                const uint8_t* got = p + py * 1024 + px * 4;
                const uint8_t exp[4] = {q(want[i].r), q(want[i].g), q(want[i].b), q(want[i].a)};
                for (int c = 0; c < 4; c++)
                    if (std::abs(int(got[c]) - int(exp[c])) > 2) {
                        if (bad++ < 5)
                            std::printf("  %s tile %d: got %d %d %d %d want %d %d %d %d\n", what, i, got[0], got[1],
                                        got[2], got[3], exp[0], exp[1], exp[2], exp[3]);
                        break;
                    }
            }
            std::printf("%-28s %s (%d bad tiles)\n", what, bad ? "FAIL" : "ok", bad);
            failures += bad != 0;
        }
        rgba mul(rgba a, rgba b) { return {a.r * b.r, a.g * b.g, a.b * b.b, a.a * b.a}; }
        rgba tint_of(uint32_t i) { return {(i % 4 + 1) / 4.0f, ((i / 4) % 4 + 1) / 4.0f, 1.0f, 1.0f}; }
        void draw_tile(command_list& l, uint32_t i, uint32_t vb_offset) {
            const uint32_t off = i * 256;
            l.set_bind_group(0, s.uniform_group, std::span<const uint32_t>(&off, 1));
            l.set_vertex_buffer(0, s.vertices, vb_offset);
            l.set_scissor({int32_t(i % 16) * 16, int32_t(i / 16) * 16, 16, 16});
            l.draw(3, 1, 0, 0);
        }
    };
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: render_selftest <pack> <api>\n");
        return 2;
    }
    const api which = api_from_id(argv[2]);
    std::ifstream in(argv[1], std::ios::binary);
    const std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    shader_library lib;
    std::string error;
    if (!lib.load(bytes, &error))
        return 1;
    scene s;
    if (which == api::none || !build(s, which, lib)) {
        std::printf("%s is not available here: skipped\n", argv[2]);
        return 77;
    }
    device& d = *s.dev;
    test t{s, d};
    texture_desc td;
    td.format = format::rgba8_unorm;
    td.width = td.height = 256;
    td.usage = texture_usage::color_target | texture_usage::copy_src;
    t.target = d.create_texture(td);
    {
        command_list* l = d.acquire_command_list(queue_type::graphics);
        l->barrier(texture_barrier{t.target, state::undefined, state::color_target, {}});
        d.submit(queue_type::graphics, l);
    }
    buffer_desc rb;
    rb.size = 1024 * 256;
    rb.usage = buffer_usage::copy_dst;
    rb.memory = memory_class::readback;
    t.readback = d.create_buffer(rb);
    buffer_desc sb;
    sb.size = 4096;
    sb.usage = buffer_usage::copy_src;
    sb.memory = memory_class::upload;
    t.staging = d.create_buffer(sb);
    const rgba palette[8] = {{1, 1, 1, 1},     {1, 0.5f, 0.25f, 1}, {0.5f, 1, 1, 1}, {1, 1, 0.5f, 1},
                             {0.25f, 1, 1, 1}, {1, 0.25f, 1, 1},    {1, 1, 1, 0.5f}, {0.5f, 0.5f, 1, 1}};
    for (int k = 0; k < 8; k++) {
        t.tex.push_back(t.make_tex(k, palette[k]));
        t.color.push_back(palette[k]);
    }
    for (int g = 0; g < 8; g++)
        t.groups.push_back(t.make_group(t.tex[g & ~1], t.tex[g | 1], false));
    // vertex data: 4 copies of a tile-covering triangle, and a buffer of degenerate ones
    float* v = reinterpret_cast<float*>(d.map(s.vertices));
    for (int c = 0; c < 4; c++) {
        const float tri[3][8] = {
            {0, 0, 0, 1, 0.5f, 0.5f, 0, 0}, {2, 0, 0, 1, 0.5f, 0.5f, 0, 0}, {0, 2, 0, 1, 0.5f, 0.5f, 0, 0}};
        std::memcpy(v + c * 24, tri, sizeof(tri));
    }
    d.flush(s.vertices, 0, 4 * 96);
    for (uint32_t i = 0; i < 256; i++)
        t.uniforms(i, t.tint_of(i));
    d.flush(s.uniforms, 0, 256 * 256);
    const rgba clear{0, 0, 0, 0};

    for (int round = 0; round < 3; round++) {
        // static groups with every kind of redundant and alternating state the filter must see through
        d.begin_frame();
        command_list* l = d.acquire_command_list(queue_type::graphics);
        t.begin_pass(*l);
        std::vector<rgba> want(256);
        const viewport full{0, 0, 256, 256, 0, 1};
        for (uint32_t i = 0; i < 256; i++) {
            l->set_pipeline(s.pipelines[(i / 5) & 1]);
            l->set_viewport(full);
            const uint32_t g = (i * 7 / 3 + round) % 8;
            if (i % 3 == 0)
                l->set_bind_group(1, t.groups[(g + 1) % 8]);
            l->set_bind_group(1, t.groups[g]);
            if (i % 11 == 0) {
                // a degenerate vertex range first: the next tile must still use the good one
                const uint32_t off = i * 256;
                l->set_bind_group(0, s.uniform_group, std::span<const uint32_t>(&off, 1));
                l->set_vertex_buffer(0, s.vertices, 4 * 96);
                l->draw(3, 1, 0, 0);
            }
            t.draw_tile(*l, i, (i % 4) * 96);
            if (i % 13 == 0)
                t.draw_tile(*l, i, (i % 4) * 96);
            want[i] = t.mul(t.mul(t.color[g & ~1], t.color[g | 1]), t.tint_of(i));
        }
        l->end_rendering();
        t.check(l, want, "static groups");

        // per-frame groups, many with identical contents, destroyed after the frame
        d.begin_frame();
        l = d.acquire_command_list(queue_type::graphics);
        t.begin_pass(*l);
        std::vector<bind_group_handle> made;
        for (uint32_t i = 0; i < 256; i++) {
            const uint32_t a = ((i / 3) % 4) * 2, b = ((i / 7 + round) % 4) * 2 + 1;
            made.push_back(t.make_group(t.tex[a], t.tex[b], true));
            l->set_pipeline(s.pipelines[0]);
            l->set_bind_group(1, made.back());
            t.draw_tile(*l, i, 0);
            want[i] = t.mul(t.mul(t.color[a], t.color[b]), t.tint_of(i));
        }
        l->end_rendering();
        t.check(l, want, "per-frame groups");
        for (bind_group_handle h : made)
            d.destroy(h);

        // replace a texture: per-frame groups naming the new one must not hit the old one's cached group
        d.destroy(t.tex[3]);
        t.color[3] = {0.25f * (round + 1), 1, 0.5f, 1};
        t.tex[3] = t.make_tex(3, t.color[3]);
        d.destroy(t.groups[2]);
        d.destroy(t.groups[3]);
        t.groups[2] = t.make_group(t.tex[2], t.tex[3], false);
        t.groups[3] = t.make_group(t.tex[2], t.tex[3], false);
        d.begin_frame();
        l = d.acquire_command_list(queue_type::graphics);
        t.begin_pass(*l);
        made.clear();
        for (uint32_t i = 0; i < 256; i++) {
            const uint32_t a = (i % 4) * 2, b = i % 2 ? 3 : 1;
            made.push_back(t.make_group(t.tex[a], t.tex[b], true));
            l->set_bind_group(1, made.back());
            l->set_pipeline(s.pipelines[i & 1]);
            t.draw_tile(*l, i, 96);
            want[i] = t.mul(t.mul(t.color[a], t.color[b]), t.tint_of(i));
        }
        l->end_rendering();
        t.check(l, want, "after texture replacement");
        for (bind_group_handle h : made)
            d.destroy(h);
    }
    // indexed indirect: one call for 256 commands, some with zero instances, via the plain and the count entry
    for (int count_path = 0; count_path < 2; count_path++) {
        buffer_desc bd;
        bd.size = 256 * 96;
        bd.usage = buffer_usage::vertex;
        bd.memory = memory_class::upload;
        const buffer_handle vb = d.create_buffer(bd);
        bd.size = 8;
        bd.usage = buffer_usage::index;
        const buffer_handle ib = d.create_buffer(bd);
        bd.size = 256 * 20 + 20;
        bd.usage = buffer_usage::indirect;
        const buffer_handle args = d.create_buffer(bd);
        bd.size = 4;
        const buffer_handle count = d.create_buffer(bd);
        float* vv = reinterpret_cast<float*>(d.map(vb));
        for (int i = 0; i < 256; i++) {
            const float x0 = -1.0f + (i % 16) * 0.125f, y0 = 1.0f - (i / 16 + 1) * 0.125f, w = 0.125f * 1.9f;
            const float tri[3][8] = {{x0, y0, 0, 1, 0.5f, 0.5f, 0, 0},
                                     {x0 + w, y0, 0, 1, 0.5f, 0.5f, 0, 0},
                                     {x0, y0 + w, 0, 1, 0.5f, 0.5f, 0, 0}};
            std::memcpy(vv + i * 24, tri, sizeof(tri));
        }
        d.flush(vb, 0, 256 * 96);
        const uint16_t idx[4] = {0, 1, 2, 0};
        std::memcpy(d.map(ib), idx, 8);
        d.flush(ib, 0, 8);
        std::vector<rgba> want(256);
        draw_indexed_indirect_command* cmd = reinterpret_cast<draw_indexed_indirect_command*>(d.map(args) + 20);
        for (uint32_t i = 0; i < 256; i++) {
            cmd[i] = {3, i % 7 == 3 ? 0u : 1u, 0, int32_t(i * 3), 0};
            want[i] = i % 7 == 3 ? rgba{0, 0, 0, 0} : t.mul(t.color[0], t.color[1]);
        }
        d.flush(args, 0, 256 * 20 + 20);
        const uint32_t n = 256;
        std::memcpy(d.map(count), &n, 4);
        d.flush(count, 0, 4);
        float* u = reinterpret_cast<float*>(d.map(s.uniforms) + 300 * 256);
        const float white[8] = {1, 1, 1, 1, 0, 0, 1, 1};
        std::memcpy(u, white, 32);
        for (int k = 0; k < 16; k++)
            u[8 + k] = k % 5 == 0 ? 1.0f : 0.0f;
        for (int k = 0; k < 16; k++)
            u[24 + k] = k == 0 || k == 5 || k == 10 ? 1.0f : 0.0f;
        d.flush(s.uniforms, 300 * 256, 256);
        d.begin_frame();
        command_list* l = d.acquire_command_list(queue_type::graphics);
        t.begin_pass(*l);
        l->set_pipeline(s.pipelines[0]);
        l->set_bind_group(1, t.groups[0]);
        const uint32_t off = 300 * 256;
        l->set_bind_group(0, s.uniform_group, std::span<const uint32_t>(&off, 1));
        l->set_vertex_buffer(0, vb, 0);
        l->set_index_buffer(ib, index_type::u16, 0);
        if (count_path)
            l->draw_indexed_indirect_count(args, 20, count, 0, 256, 20);
        else
            l->draw_indexed_indirect(args, 20, 256, 20);
        l->end_rendering();
        t.check(l, want, count_path ? "indexed indirect count" : "indexed indirect");
        for (buffer_handle h : {vb, ib, args, count})
            d.destroy(h);
    }
    // partial readback: only the asked-for range has to be fresh
    {
        d.begin_frame();
        command_list* l = d.acquire_command_list(queue_type::graphics);
        t.begin_pass(*l);
        l->end_rendering();
        buffer_texture_copy r;
        r.buffer_row_pitch = 1024;
        r.width = r.height = 256;
        l->barrier(texture_barrier{t.target, state::color_target, state::copy_src, {}});
        l->copy_texture_to_buffer(t.readback, t.target, std::span<const buffer_texture_copy>(&r, 1));
        l->barrier(texture_barrier{t.target, state::copy_src, state::color_target, {}});
        d.submit(queue_type::graphics, l);
        d.end_frame();
        d.wait_idle();
        d.invalidate(t.readback, 1024 * 100 + 12, 40);
        const uint8_t* p = d.map(t.readback);
        bool ok = true;
        for (int k = 0; k < 40; k++)
            ok &= p[1024 * 100 + 12 + k] == 0;
        std::printf("%-28s %s\n", "partial readback", ok ? "ok" : "FAIL");
        t.failures += !ok;
    }
    const device_stats st = d.stats();
    std::printf("stats: draws %llu binds %llu pipelines %llu submits %llu\n", (unsigned long long)st.draws,
                (unsigned long long)st.bind_group_binds, (unsigned long long)st.pipeline_binds,
                (unsigned long long)st.submits);
    const std::string report = d.debug_report();
    if (!report.empty()) {
        std::printf("errors: %s\n", report.c_str());
        t.failures++;
    }
    std::printf("render_selftest %s %s\n", argv[2], t.failures ? "FAILED" : "ok");
    return t.failures ? 1 : 0;
}
