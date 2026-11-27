// what conduit itself costs, per api and per scenario: recording, submission, creation, uploads, threads and
// allocations. every row is a distribution, because a frame time is only as good as its worst percentile
#include <conduit/device.hpp>
#include <conduit/render_profile.hpp>
#include <conduit/shader_library.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iterator>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <libproc.h>
#include <unistd.h>
#endif

using namespace conduit;
using clock_type = std::chrono::steady_clock;

namespace {
    std::atomic<uint64_t> g_allocs{0}, g_alloc_bytes{0};
}

void* operator new(size_t size) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    g_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
    if (void* p = std::malloc(size ? size : 1))
        return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {
    std::free(p);
}
void operator delete(void* p, size_t) noexcept {
    std::free(p);
}

namespace {
    double ns_since(clock_type::time_point t) {
        return std::chrono::duration<double, std::nano>(clock_type::now() - t).count();
    }

    // instructions retired by this process: unlike a timer it does not care how busy the machine is, so it is
    // the number to compare two builds by. zero where the platform does not report it
    uint64_t instructions() {
#if defined(__APPLE__)
        rusage_info_v4 info{};
        if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&info)) == 0)
            return info.ri_instructions;
#endif
        return 0;
    }

    struct summary {
        double avg = 0, median = 0, p95 = 0, p99 = 0, min = 0, max = 0;
    };

    summary summarize(std::vector<double> v) {
        summary s;
        if (v.empty())
            return s;
        std::sort(v.begin(), v.end());
        for (double x : v)
            s.avg += x;
        s.avg /= static_cast<double>(v.size());
        const auto at = [&](double q) { return v[(std::min)(v.size() - 1, static_cast<size_t>(q * v.size()))]; };
        s.median = at(0.5);
        s.p95 = at(0.95);
        s.p99 = at(0.99);
        s.min = v.front();
        s.max = v.back();
        return s;
    }

    struct row {
        std::string api, name, unit;
        summary s;
    };
    std::vector<row> g_rows;
    const char* g_api = "";

    void emit(const char* name, const char* unit, const std::vector<double>& samples) {
        const summary s = summarize(samples);
        g_rows.push_back({g_api, name, unit, s});
        std::printf("%-7s %-28s %-9s median %10.1f  p95 %10.1f  p99 %10.1f  min %10.1f  max %10.1f  avg %10.1f\n",
                    g_api, name, unit, s.median, s.p95, s.p99, s.min, s.max, s.avg);
    }
    void emit(const char* name, const char* unit, double value) {
        emit(name, unit, std::vector<double>{value});
    }

    struct options {
        uint32_t draws = 8000;
        int frames = 200;
        bool quick = false;
        bool threads = true;
    };

    struct scene {
        std::unique_ptr<instance> inst;
        std::unique_ptr<device> dev;
        texture_handle target;
        std::vector<texture_handle> textures;
        sampler_handle sampler;
        buffer_handle uniforms, vertices, indices, indirect;
        bind_layout_handle uniform_layout, texture_layout;
        pipeline_layout_handle layout;
        shader_handle vs, fs;
        std::vector<pipeline_handle> pipelines;
        bind_group_handle uniform_group;
        std::vector<bind_group_handle> texture_groups;
        std::vector<bind_group_entry> texture_entries;
    };

    graphics_pipeline_desc pipeline_desc(const scene& s, api which, uint32_t variant) {
        graphics_pipeline_desc gd;
        gd.vertex = s.vs;
        gd.fragment = s.fs;
        gd.layout = s.layout;
        gd.streams[0] = {32, vertex_step::vertex};
        gd.stream_count = 1;
        if (which != api::none) {
            for (uint32_t a = 0; a < 2; a++)
                gd.attributes[a] = {a, 0, format::rgba32_float, a * 16};
            gd.attribute_count = 2;
        }
        gd.color_formats[0] = format::rgba8_unorm;
        gd.color_count = 1;
        // every variant is a distinct pipeline for the driver: the blend factors walk through their combinations
        blend_target& b = gd.blend.targets[0];
        b.enable = variant != 0;
        b.src_color = static_cast<blend_factor>(variant % 10);
        b.dst_color = static_cast<blend_factor>((variant / 10) % 10);
        b.src_alpha = blend_factor::one;
        b.dst_alpha = static_cast<blend_factor>((variant / 100) % 2);
        return gd;
    }

    bool build(scene& s, api which, const shader_library& lib) {
        s.inst = create_instance(which, {});
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

        const auto buffer = [&](uint64_t size, buffer_usage usage, memory_class memory) {
            buffer_desc bd;
            bd.size = size;
            bd.usage = usage;
            bd.memory = memory;
            return d.create_buffer(bd);
        };
        s.uniforms = buffer(256ull * 8192, buffer_usage::uniform, memory_class::upload);
        s.vertices = buffer(32ull * 3 * 8192, buffer_usage::vertex, memory_class::upload);
        s.indices = buffer(4ull * 3 * 8192, buffer_usage::index, memory_class::upload);
        s.indirect = buffer(20ull * 64, buffer_usage::indirect, memory_class::upload);

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

        if (which != api::none) {
            const shader_target target = shader_library::target_for(which);
            const program_info* blit = lib.find("blit");
            if (!blit)
                return false;
            shader_desc sd;
            sd.format = shader_library::format_for(target);
            sd.stage = shader_stage::vertex;
            sd.code = lib.code(*blit, shader_stage::vertex, target);
            sd.entry = shader_library::entry_for(shader_stage::vertex, target);
            if (sd.code.empty())
                return false;
            s.vs = d.create_shader(sd);
            sd.stage = shader_stage::fragment;
            sd.code = lib.code(*blit, shader_stage::fragment, target);
            sd.entry = shader_library::entry_for(shader_stage::fragment, target);
            if (sd.code.empty())
                return false;
            s.fs = d.create_shader(sd);
        } else {
            s.vs = d.create_shader({});
        }
        for (uint32_t i = 0; i < 2; i++) {
            s.pipelines.push_back(d.create_graphics_pipeline(pipeline_desc(s, which, i)));
            if (!s.pipelines.back())
                return false;
        }

        bind_group_entry ue;
        ue.binding = 0;
        ue.buffer = {s.uniforms, 0, 256};
        bind_group_desc ug;
        ug.layout = s.uniform_layout;
        ug.entries = std::span<const bind_group_entry>(&ue, 1);
        s.uniform_group = d.create_bind_group(ug);
        for (int g = 0; g < 8; g++) {
            s.texture_entries.clear();
            for (const texture_slot& t : lib.texture_slots()) {
                bind_group_entry te;
                te.binding = t.slot * 2;
                te.texture.texture = s.textures[t.array ? (g & ~1) : (g | 1)];
                te.texture.view_dim = t.array ? texture_dim::tex2d_array : texture_dim::tex2d;
                s.texture_entries.push_back(te);
                bind_group_entry se;
                se.binding = t.slot * 2 + 1;
                se.sampler = s.sampler;
                s.texture_entries.push_back(se);
            }
            bind_group_desc tg;
            tg.layout = s.texture_layout;
            tg.entries = s.texture_entries;
            s.texture_groups.push_back(d.create_bind_group(tg));
        }
        // an engine flushes what it wrote before submitting; backends without real mapping rely on it
        d.flush(s.uniforms, 0, 0);
        d.flush(s.vertices, 0, 0);
        d.flush(s.indices, 0, 0);
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

    enum class pattern { world, indexed, material_per_draw, pipeline_per_draw };

    // world: a dynamic uniform offset and a vertex range per draw, a material every 8 draws, a pipeline every 64
    void record(const scene& s, command_list& list, uint32_t first, uint32_t count, pattern p, bool clear) {
        rendering_desc rd;
        rd.colors[0].texture = s.target;
        rd.colors[0].load = clear ? load_op::clear : load_op::load;
        rd.color_count = 1;
        list.begin_rendering(rd);
        if (p == pattern::indexed)
            list.set_index_buffer(s.indices, index_type::u32);
        const uint32_t material_every = p == pattern::material_per_draw ? 1 : 8;
        const uint32_t pipeline_every = p == pattern::pipeline_per_draw ? 1 : 64;
        for (uint32_t i = first; i < first + count; i++) {
            if (i == first || i % pipeline_every == 0)
                list.set_pipeline(s.pipelines[(i / pipeline_every) & 1]);
            if (i == first || i % material_every == 0)
                list.set_bind_group(1, s.texture_groups[(i / material_every) % s.texture_groups.size()]);
            const uint32_t offset = (i % 8192) * 256;
            list.set_bind_group(0, s.uniform_group, std::span<const uint32_t>(&offset, 1));
            list.set_vertex_buffer(0, s.vertices, static_cast<uint64_t>(i % 8192) * 96);
            if (p == pattern::indexed)
                list.draw_indexed(3, 1, (i % 8192) * 3, 0, 0);
            else
                list.draw(3, 1, 0, 0);
        }
        list.end_rendering();
    }

    void bench_frames(scene& s, const options& o) {
        device& d = *s.dev;
        const int warmup = 20;
        std::vector<double> rec, instr, sub, frame, allocs, bytes;
        for (int f = 0; f < o.frames + warmup; f++) {
            const uint64_t a0 = g_allocs.load(), b0 = g_alloc_bytes.load();
            const auto t0 = clock_type::now();
            d.begin_frame();
            command_list* list = d.acquire_command_list(queue_type::graphics);
            const auto t1 = clock_type::now();
            const uint64_t i0 = instructions();
            record(s, *list, 0, o.draws, pattern::world, true);
            const uint64_t i1 = instructions();
            const auto t2 = clock_type::now();
            d.submit(queue_type::graphics, list);
            d.end_frame();
            const auto t3 = clock_type::now();
            if (f < warmup)
                continue;
            rec.push_back(std::chrono::duration<double, std::nano>(t2 - t1).count() / o.draws);
            if (i1 > i0)
                instr.push_back(static_cast<double>(i1 - i0) / o.draws);
            sub.push_back(std::chrono::duration<double, std::micro>(t3 - t2).count());
            frame.push_back(std::chrono::duration<double, std::micro>(t3 - t0).count());
            allocs.push_back(static_cast<double>(g_allocs.load() - a0));
            bytes.push_back(static_cast<double>(g_alloc_bytes.load() - b0));
        }
        d.wait_idle();
        emit("record world", "ns/draw", rec);
        if (!instr.empty())
            emit("record world", "instr/draw", instr);
        emit("submit + end_frame", "us", sub);
        emit("cpu frame", "us", frame);
        emit("allocations", "/frame", allocs);
        emit("allocated", "B/frame", bytes);

        const struct {
            pattern p;
            const char* name;
        } patterns[] = {{pattern::indexed, "record indexed"},
                        {pattern::material_per_draw, "record material/draw"},
                        {pattern::pipeline_per_draw, "record pipeline/draw"}};
        for (const auto& p : patterns) {
            std::vector<double> v, vi;
            for (int f = 0; f < o.frames / 4 + warmup; f++) {
                d.begin_frame();
                command_list* list = d.acquire_command_list(queue_type::graphics);
                const auto t = clock_type::now();
                const uint64_t i0 = instructions();
                record(s, *list, 0, o.draws, p.p, true);
                const uint64_t i1 = instructions();
                const double ns = ns_since(t);
                d.submit(queue_type::graphics, list);
                d.end_frame();
                if (f < warmup)
                    continue;
                v.push_back(ns / o.draws);
                if (i1 > i0)
                    vi.push_back(static_cast<double>(i1 - i0) / o.draws);
            }
            d.wait_idle();
            emit(p.name, "ns/draw", v);
            if (!vi.empty())
                emit(p.name, "instr/draw", vi);
        }
    }

    void bench_gpu_time(scene& s, const options& o) {
        device& d = *s.dev;
        if (!d.caps().features.timestamp_queries)
            return;
        query_set_desc qd;
        qd.count = 2;
        const query_set_handle queries = d.create_query_set(qd);
        buffer_desc bd;
        bd.size = 16;
        bd.usage = buffer_usage::copy_dst;
        bd.memory = memory_class::readback;
        const buffer_handle out = d.create_buffer(bd);
        const uint8_t* mapped = d.map(out);
        std::vector<double> gpu;
        for (int f = 0; f < 24 && queries && mapped; f++) {
            d.begin_frame();
            command_list* list = d.acquire_command_list(queue_type::graphics);
            list->write_timestamp(queries, 0);
            record(s, *list, 0, o.draws, pattern::world, true);
            list->write_timestamp(queries, 1);
            list->resolve_queries(queries, 0, 2, out, 0);
            d.submit(queue_type::graphics, list);
            d.end_frame();
            d.wait_idle();
            d.invalidate(out, 0, 16);
            uint64_t t[2];
            std::memcpy(t, mapped, sizeof(t));
            if (f >= 4 && t[1] > t[0])
                gpu.push_back(static_cast<double>(t[1] - t[0]) * d.timestamp_period_ns() / 1000.0);
        }
        if (!gpu.empty())
            emit("gpu frame", "us", gpu);
        d.destroy(out);
        d.destroy(queries);
    }

    void bench_indirect_and_barriers(scene& s, const options& o) {
        device& d = *s.dev;
        const int frames = o.quick ? 12 : 60;
        std::vector<double> indirect, barrier;
        if (uint8_t* args = d.map(s.indirect)) {
            std::memset(args, 0, 20 * 64);
            d.flush(s.indirect, 0, 20 * 64);
        }
        for (int f = 0; f < frames; f++) {
            d.begin_frame();
            command_list* list = d.acquire_command_list(queue_type::graphics);
            std::array<texture_barrier, 8> to_copy, to_read;
            for (size_t i = 0; i < 8; i++) {
                to_copy[i] = {s.textures[i], state::shader_read_fragment, state::copy_dst, {}};
                to_read[i] = {s.textures[i], state::copy_dst, state::shader_read_fragment, {}};
            }
            auto t = clock_type::now();
            for (int i = 0; i < 64; i++) {
                list->barrier(to_copy, {});
                list->barrier(to_read, {});
            }
            const double barrier_ns = ns_since(t) / (64.0 * 16.0);

            rendering_desc rd;
            rd.colors[0].texture = s.target;
            rd.color_count = 1;
            list->begin_rendering(rd);
            list->set_pipeline(s.pipelines[0]);
            list->set_bind_group(1, s.texture_groups[0]);
            const uint32_t offset = 0;
            list->set_bind_group(0, s.uniform_group, std::span<const uint32_t>(&offset, 1));
            list->set_vertex_buffer(0, s.vertices, 0);
            list->set_index_buffer(s.indices, index_type::u32);
            t = clock_type::now();
            for (int i = 0; i < 256; i++)
                list->draw_indexed_indirect(s.indirect, 0, 64, 20);
            const double indirect_ns = ns_since(t) / 256.0;
            list->end_rendering();
            d.submit(queue_type::graphics, list);
            d.end_frame();
            if (f >= 4) {
                barrier.push_back(barrier_ns);
                indirect.push_back(indirect_ns);
            }
        }
        d.wait_idle();
        emit("barrier", "ns/each", barrier);
        emit("indirect x64 draws", "ns/call", indirect);
    }

    void bench_creation(scene& s, api which, const options& o) {
        device& d = *s.dev;
        const int frames = o.quick ? 8 : 40;
        std::vector<double> group, transient, buffers, textures, destroy;
        for (int f = 0; f < frames; f++) {
            d.begin_frame();
            std::vector<bind_group_handle> made;
            made.reserve(512);
            bind_group_desc tg;
            tg.layout = s.texture_layout;
            tg.entries = s.texture_entries;
            auto t = clock_type::now();
            for (int i = 0; i < 256; i++)
                made.push_back(d.create_bind_group(tg));
            group.push_back(ns_since(t) / 256);
            // transient groups are bound and drawn with: some backends only build the native object at the
            // first bind, and timing the create call alone would hide it
            tg.per_frame = true;
            command_list* list = d.acquire_command_list(queue_type::graphics);
            rendering_desc rd;
            rd.colors[0].texture = s.target;
            rd.color_count = 1;
            list->begin_rendering(rd);
            list->set_pipeline(s.pipelines[0]);
            const uint32_t offset = 0;
            list->set_bind_group(0, s.uniform_group, std::span<const uint32_t>(&offset, 1));
            list->set_vertex_buffer(0, s.vertices, 0);
            t = clock_type::now();
            for (int i = 0; i < 256; i++) {
                made.push_back(d.create_bind_group(tg));
                list->set_bind_group(1, made.back());
                list->draw(3, 1, 0, 0);
            }
            transient.push_back(ns_since(t) / 256);
            list->end_rendering();
            d.submit(queue_type::graphics, list);
            for (bind_group_handle h : made)
                d.destroy(h);

            std::vector<buffer_handle> bufs;
            bufs.reserve(64);
            buffer_desc bd;
            bd.size = 64 << 10;
            bd.usage = buffer_usage::vertex | buffer_usage::copy_dst;
            t = clock_type::now();
            for (int i = 0; i < 64; i++)
                bufs.push_back(d.create_buffer(bd));
            buffers.push_back(ns_since(t) / 64);
            std::vector<texture_handle> texs;
            texs.reserve(32);
            texture_desc td;
            td.format = format::rgba8_unorm;
            td.width = td.height = 64;
            td.usage = texture_usage::sampled | texture_usage::copy_dst;
            t = clock_type::now();
            for (int i = 0; i < 32; i++)
                texs.push_back(d.create_texture(td));
            textures.push_back(ns_since(t) / 32);
            t = clock_type::now();
            for (buffer_handle h : bufs)
                d.destroy(h);
            for (texture_handle h : texs)
                d.destroy(h);
            destroy.push_back(ns_since(t) / 96);
            d.end_frame();
        }
        d.wait_idle();
        emit("create bind group", "ns", group);
        emit("transient group + draw", "ns", transient);
        emit("create buffer 64K", "ns", buffers);
        emit("create texture 64x64", "ns", textures);
        emit("destroy resource", "ns", destroy);

        // pipelines nothing has seen before: what a material first met in the middle of a level costs
        std::vector<double> pipelines;
        std::vector<pipeline_handle> made;
        const uint32_t count = o.quick ? 8 : 48;
        for (uint32_t i = 0; i < count; i++) {
            const graphics_pipeline_desc gd = pipeline_desc(s, which, 2 + i);
            const auto t = clock_type::now();
            made.push_back(d.create_graphics_pipeline(gd));
            pipelines.push_back(ns_since(t) / 1000.0);
        }
        emit("create pipeline", "us", pipelines);
        for (pipeline_handle h : made)
            d.destroy(h);
        d.wait_idle();
    }

    void bench_uploads(scene& s, const options& o) {
        device& d = *s.dev;
        const uint64_t size = 1 << 20;
        buffer_desc sd;
        sd.size = size;
        sd.usage = buffer_usage::copy_src;
        sd.memory = memory_class::upload;
        const buffer_handle staging = d.create_buffer(sd);
        buffer_desc gd;
        gd.size = size;
        gd.usage = buffer_usage::vertex | buffer_usage::copy_dst;
        const buffer_handle gpu = d.create_buffer(gd);
        texture_desc td;
        td.format = format::rgba8_unorm;
        td.width = td.height = 256;
        td.usage = texture_usage::sampled | texture_usage::copy_dst;
        const texture_handle tex = d.create_texture(td);
        std::vector<uint8_t> source(size, 0x5A);
        uint8_t* mapped = d.map(staging);
        if (!mapped)
            return;
        std::vector<double> buffer_us, texture_us;
        state before = state::undefined;
        for (int f = 0; f < (o.quick ? 8 : 60); f++) {
            d.begin_frame();
            auto t = clock_type::now();
            std::memcpy(mapped, source.data(), size);
            d.flush(staging, 0, size);
            command_list* list = d.acquire_command_list(queue_type::graphics);
            list->barrier(buffer_barrier{gpu, before == state::undefined ? state::undefined : state::vertex_buffer,
                                         state::copy_dst});
            list->copy_buffer(gpu, 0, staging, 0, size);
            list->barrier(buffer_barrier{gpu, state::copy_dst, state::vertex_buffer});
            buffer_us.push_back(ns_since(t) / 1000.0);

            t = clock_type::now();
            list->barrier(texture_barrier{tex, before, state::copy_dst, {}});
            buffer_texture_copy region;
            region.width = region.height = 256;
            region.buffer_row_pitch = 256 * 4;
            list->copy_buffer_to_texture(tex, staging, std::span<const buffer_texture_copy>(&region, 1));
            list->barrier(texture_barrier{tex, state::copy_dst, state::shader_read_fragment, {}});
            texture_us.push_back(ns_since(t) / 1000.0);
            before = state::shader_read_fragment;
            d.submit(queue_type::graphics, list);
            d.end_frame();
        }
        d.wait_idle();
        emit("upload 1 MiB buffer", "us", buffer_us);
        emit("upload 256x256 texture", "us", texture_us);
        d.destroy(staging);
        d.destroy(gpu);
        d.destroy(tex);
    }

    // sixteen full-screen passes that each sample the previous one: almost no cpu work and a lot of memory
    // traffic, which is where framebuffer compression and the like show up. timed from submit until the gpu is done
    void bench_fill(scene& s, const options& o, bool reinterpretable) {
        device& d = *s.dev;
        if (!s.fs)
            return;
        const uint32_t size = 2048;
        texture_handle targets[2];
        bind_group_handle groups[2];
        for (int i = 0; i < 2; i++) {
            texture_desc td;
            td.format = format::rgba8_unorm;
            td.width = td.height = size;
            td.usage = texture_usage::color_target | texture_usage::sampled;
            if (reinterpretable)
                td.view_format = format::rgba8_srgb;
            targets[i] = d.create_texture(td);
            std::vector<bind_group_entry> entries = s.texture_entries;
            for (bind_group_entry& e : entries)
                if (e.texture.texture && e.texture.view_dim == texture_dim::tex2d)
                    e.texture.texture = targets[i];
            bind_group_desc gd;
            gd.layout = s.texture_layout;
            gd.entries = entries;
            groups[i] = d.create_bind_group(gd);
        }
        const uint32_t slot = 8191;
        if (uint8_t* u = d.map(s.uniforms)) {
            float data[40] = {};
            for (int i = 0; i < 4; i++)
                data[i] = 1.0f;
            data[4] = data[5] = -1.0f;
            data[6] = data[7] = 1.0f;
            data[8] = data[13] = data[18] = data[23] = 1.0f;
            data[24] = data[29] = data[34] = 1.0f;
            std::memcpy(u + slot * 256, data, sizeof(data));
            d.flush(s.uniforms, slot * 256, 256);
        }
        if (uint8_t* v = d.map(s.vertices)) {
            const float tri[24] = {0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0};
            std::memcpy(v, tri, sizeof(tri));
            d.flush(s.vertices, 0, sizeof(tri));
        }
        std::vector<double> wall, gpu;
        query_set_handle queries;
        buffer_handle resolved;
        const uint8_t* stamps = nullptr;
        if (d.caps().features.timestamp_queries) {
            query_set_desc qd;
            qd.count = 2;
            queries = d.create_query_set(qd);
            buffer_desc bd;
            bd.size = 16;
            bd.usage = buffer_usage::copy_dst;
            bd.memory = memory_class::readback;
            resolved = d.create_buffer(bd);
            stamps = d.map(resolved);
        }
        state states[2] = {state::undefined, state::undefined};
        for (int f = 0; f < (o.quick ? 6 : 40); f++) {
            d.begin_frame();
            command_list* list = d.acquire_command_list(queue_type::graphics);
            if (stamps)
                list->write_timestamp(queries, 0);
            for (int pass = 0; pass < 16; pass++) {
                const int dst = pass & 1, src = dst ^ 1;
                const texture_barrier barriers[2] = {{targets[dst], states[dst], state::color_target, {}},
                                                     {targets[src], states[src], state::shader_read_fragment, {}}};
                list->barrier(barriers, {});
                states[dst] = state::color_target;
                states[src] = state::shader_read_fragment;
                rendering_desc rd;
                rd.colors[0].texture = targets[dst];
                rd.colors[0].load = load_op::dont_care;
                rd.color_count = 1;
                list->begin_rendering(rd);
                list->set_pipeline(s.pipelines[0]);
                list->set_bind_group(1, groups[src]);
                const uint32_t offset = slot * 256;
                list->set_bind_group(0, s.uniform_group, std::span<const uint32_t>(&offset, 1));
                list->set_vertex_buffer(0, s.vertices, 0);
                list->draw(3, 1, 0, 0);
                list->end_rendering();
            }
            if (stamps) {
                list->write_timestamp(queries, 1);
                list->resolve_queries(queries, 0, 2, resolved, 0);
            }
            const auto t = clock_type::now();
            d.submit(queue_type::graphics, list);
            d.end_frame();
            d.wait_idle();
            if (f < 4)
                continue;
            wall.push_back(ns_since(t) / 1000.0);
            if (stamps) {
                d.invalidate(resolved, 0, 16);
                uint64_t at[2];
                std::memcpy(at, stamps, sizeof(at));
                if (at[1] > at[0])
                    gpu.push_back(static_cast<double>(at[1] - at[0]) * d.timestamp_period_ns() / 1000.0);
            }
        }
        const std::string name = std::string("16 passes of 2048x2048") + (reinterpretable ? " +view" : "");
        emit(name.c_str(), "us", wall);
        if (!gpu.empty())
            emit(name.c_str(), "gpu us", gpu);
        if (stamps) {
            d.destroy(resolved);
            d.destroy(queries);
        }
        for (int i = 0; i < 2; i++) {
            d.destroy(groups[i]);
            d.destroy(targets[i]);
        }
        d.wait_idle();
    }

    // workers park between frames, so a frame pays for recording and not for thread start-up
    class crew {
    public:
        explicit crew(uint32_t count) {
            for (uint32_t i = 0; i < count; i++)
                threads_.emplace_back([this, i] { run(i); });
        }
        ~crew() {
            {
                std::lock_guard lock(mutex_);
                quit_ = true;
                generation_++;
            }
            wake_.notify_all();
            for (std::thread& t : threads_)
                t.join();
        }
        template <class Fn> void each(Fn&& fn) {
            job_ = [&](uint32_t i) { fn(i); };
            {
                std::lock_guard lock(mutex_);
                pending_ = static_cast<uint32_t>(threads_.size());
                generation_++;
            }
            wake_.notify_all();
            std::unique_lock lock(mutex_);
            done_.wait(lock, [&] { return pending_ == 0; });
        }

    private:
        void run(uint32_t index) {
            uint64_t seen = 0;
            for (;;) {
                {
                    std::unique_lock lock(mutex_);
                    wake_.wait(lock, [&] { return generation_ != seen; });
                    seen = generation_;
                    if (quit_)
                        return;
                }
                job_(index);
                std::lock_guard lock(mutex_);
                if (--pending_ == 0)
                    done_.notify_one();
            }
        }
        std::vector<std::thread> threads_;
        std::mutex mutex_;
        std::condition_variable wake_, done_;
        std::function<void(uint32_t)> job_;
        uint64_t generation_ = 0;
        uint32_t pending_ = 0;
        bool quit_ = false;
    };

    void bench_threads(scene& s, const options& o) {
        device& d = *s.dev;
        const uint32_t cores = (std::max)(1u, std::thread::hardware_concurrency());
        for (uint32_t threads : {1u, 2u, 4u, 8u}) {
            if (threads > cores)
                break;
            crew workers(threads);
            std::vector<command_list*> lists(threads);
            std::vector<double> wall;
            const uint32_t each = o.draws / threads;
            for (int f = 0; f < (o.quick ? 12 : 80); f++) {
                d.begin_frame();
                for (command_list*& l : lists)
                    l = d.acquire_command_list(queue_type::graphics);
                const auto t = clock_type::now();
                workers.each([&](uint32_t i) { record(s, *lists[i], i * each, each, pattern::world, i == 0); });
                const double ns = ns_since(t);
                submit_desc sd;
                sd.lists = lists;
                d.submit(queue_type::graphics, sd);
                d.end_frame();
                if (f >= 8)
                    wall.push_back(ns / (static_cast<double>(each) * threads));
            }
            d.wait_idle();
            const std::string name = "record on " + std::to_string(threads) + " thread" + (threads > 1 ? "s" : "");
            emit(name.c_str(), "ns/draw", wall);
        }
    }

    void bench_startup(api which, const std::vector<uint8_t>& pack) {
        std::vector<double> init, probe, load;
        for (int i = 0; i < 5; i++) {
            auto t = clock_type::now();
            {
                std::unique_ptr<instance> inst = create_instance(which, {});
                std::unique_ptr<device> dev = inst ? inst->create_device({}) : nullptr;
                init.push_back(ns_since(t) / 1e6);
            }
            if (which != api::none) {
                probe_options po;
                po.apis = {which};
                t = clock_type::now();
                const conduit::probe p = probe_system(po);
                probe.push_back(ns_since(t) / 1e6);
            }
            t = clock_type::now();
            shader_library lib;
            lib.load(pack);
            load.push_back(ns_since(t) / 1000.0);
        }
        emit("instance + device", "ms", init);
        if (!probe.empty())
            emit("probe with caps", "ms", probe);
        emit("load shader pack", "us", load);
    }

    void write_json(const char* path) {
        std::FILE* f = std::fopen(path, "w");
        if (!f)
            return;
        std::fprintf(f, "[\n");
        for (size_t i = 0; i < g_rows.size(); i++) {
            const row& r = g_rows[i];
            std::fprintf(f,
                         "  {\"api\": \"%s\", \"name\": \"%s\", \"unit\": \"%s\", \"median\": %.3f, \"p95\": %.3f, "
                         "\"p99\": %.3f, \"min\": %.3f, \"max\": %.3f, \"avg\": %.3f}%s\n",
                         r.api.c_str(), r.name.c_str(), r.unit.c_str(), r.s.median, r.s.p95, r.s.p99, r.s.min, r.s.max,
                         r.s.avg, i + 1 < g_rows.size() ? "," : "");
        }
        std::fprintf(f, "]\n");
        std::fclose(f);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(
            stderr,
            "usage: conduit_bench <pack> [api ...] [--draws N] [--frames N] [--quick] [--one-thread] [--json file]\n");
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
    options o;
    const char* json = nullptr;
    std::vector<api> apis;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--draws") && i + 1 < argc)
            o.draws = static_cast<uint32_t>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--frames") && i + 1 < argc)
            o.frames = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--json") && i + 1 < argc)
            json = argv[++i];
        else if (!std::strcmp(argv[i], "--quick"))
            o.quick = true;
        else if (!std::strcmp(argv[i], "--one-thread"))
            o.threads = false;
        else
            apis.push_back(api_from_id(argv[i]));
    }
    if (o.quick) {
        o.draws = (std::min)(o.draws, 2000u);
        o.frames = (std::min)(o.frames, 24);
    }
    if (apis.empty()) {
        apis = compiled_apis();
        apis.push_back(api::none);
    }
    for (api which : apis) {
        g_api = api_id(which);
        scene s;
        if (!build(s, which, lib)) {
            std::printf("%-7s unavailable\n", g_api);
            continue;
        }
        std::printf("%-7s %s, %u draws, %d frames\n", g_api, s.dev->caps().adapter.name.c_str(), o.draws, o.frames);
        bench_frames(s, o);
        bench_gpu_time(s, o);
        bench_indirect_and_barriers(s, o);
        bench_creation(s, which, o);
        bench_uploads(s, o);
        bench_fill(s, o, false);
        bench_fill(s, o, true);
        if (o.threads)
            bench_threads(s, o);
        const std::string report = s.dev->debug_report();
        if (!report.empty())
            std::fprintf(stderr, "%s errors reported:\n%s\n", g_api, report.substr(0, 2000).c_str());
        s.dev->wait_idle();
        s = {};
        bench_startup(which, bytes);
    }
    if (json)
        write_json(json);
    return 0;
}
