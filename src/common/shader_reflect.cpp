#include "common/shader_reflect.hpp"

#include <charconv>
#include <cstdio>
#include <cstdlib>

namespace conduit {
    namespace {
        bool is_ident(char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
        }
        bool is_space(char c) {
            return c == ' ' || c == '\t' || c == '\r' || c == '\n';
        }

        struct cursor {
            std::string_view text;
            size_t at = 0;

            bool done() const { return at >= text.size(); }
            void skip() {
                while (at < text.size() && is_space(text[at]))
                    at++;
            }
            bool eat(std::string_view token) {
                skip();
                if (text.substr(at, token.size()) != token)
                    return false;
                at += token.size();
                return true;
            }
            bool number(uint32_t& out) {
                skip();
                const char* begin = text.data() + at;
                const char* end = text.data() + text.size();
                const auto [p, ec] = std::from_chars(begin, end, out);
                if (ec != std::errc())
                    return false;
                at += static_cast<size_t>(p - begin);
                return true;
            }
            std::string_view ident() {
                skip();
                const size_t start = at;
                while (at < text.size() && is_ident(text[at]))
                    at++;
                return text.substr(start, at - start);
            }
        };

        std::string_view line_at(std::string_view text, size_t pos, size_t& line_begin) {
            size_t b = pos ? text.rfind('\n', pos - 1) : std::string_view::npos;
            b = b == std::string_view::npos ? 0 : b + 1;
            size_t e = text.find('\n', pos);
            if (e == std::string_view::npos)
                e = text.size();
            line_begin = b;
            return text.substr(b, e - b);
        }

        void reflect_hlsl(std::string_view src, shader_interface& out) {
            size_t pos = 0;
            while ((pos = src.find("register(", pos)) != std::string_view::npos) {
                const size_t clause_begin = pos;
                size_t line_begin;
                const std::string_view line = line_at(src, pos, line_begin);
                cursor c{src, pos + 9};
                c.skip();
                if (c.at >= src.size()) {
                    pos += 9;
                    continue;
                }
                shader_resource r;
                r.register_class = src[c.at++];
                uint32_t index = 0;
                if (!c.number(index)) {
                    pos += 9;
                    continue;
                }
                r.binding = index;
                if (c.eat(",")) {
                    c.skip();
                    if (c.eat("space")) {
                        uint32_t space = 0;
                        c.number(space);
                        r.set = space;
                    }
                }
                if (!c.eat(")")) {
                    pos += 9;
                    continue;
                }
                r.clause_begin = clause_begin;
                r.clause_end = c.at;
                pos = c.at;

                cursor decl{line, 0};
                std::string_view type = decl.ident();
                while (type == "static" || type == "globallycoherent" || type == "extern")
                    type = decl.ident();
                if (type == "cbuffer" || type == "ConstantBuffer") {
                    r.kind = resource_kind::uniform_buffer;
                } else if (type == "SamplerState") {
                    r.kind = resource_kind::sampler;
                } else if (type == "SamplerComparisonState") {
                    r.kind = resource_kind::sampler;
                    r.comparison = true;
                } else if (type == "Texture2D") {
                    r.kind = resource_kind::sampled_texture;
                } else if (type == "Texture2DArray") {
                    r.kind = resource_kind::sampled_texture;
                    r.dim = texture_dim::tex2d_array;
                } else if (type == "Texture3D") {
                    r.kind = resource_kind::sampled_texture;
                    r.dim = texture_dim::tex3d;
                } else if (type == "TextureCube") {
                    r.kind = resource_kind::sampled_texture;
                    r.dim = texture_dim::cube;
                } else if (type == "RWTexture2D" || type == "RWTexture2DArray" || type == "RWTexture3D") {
                    r.kind = resource_kind::storage_texture;
                    r.dim = type == "RWTexture2DArray" ? texture_dim::tex2d_array
                            : type == "RWTexture3D"    ? texture_dim::tex3d
                                                       : texture_dim::tex2d;
                } else if (type == "StructuredBuffer" || type == "ByteAddressBuffer") {
                    r.kind = resource_kind::storage_buffer;
                } else if (type == "RWStructuredBuffer" || type == "RWByteAddressBuffer") {
                    r.kind = resource_kind::storage_buffer_rw;
                } else {
                    continue;
                }
                std::string_view rest = line.substr(decl.at);
                if (const size_t lt = rest.find('<'); lt < 3) {
                    int depth = 0;
                    size_t i = lt;
                    for (; i < rest.size(); i++) {
                        if (rest[i] == '<')
                            depth++;
                        else if (rest[i] == '>' && --depth == 0)
                            break;
                    }
                    rest = rest.substr(std::min(i + 1, rest.size()));
                }
                cursor n{rest, 0};
                r.name = std::string(n.ident());
                out.resources.push_back(std::move(r));
            }
        }

        void reflect_wgsl(std::string_view src, shader_interface& out) {
            size_t pos = 0;
            while (pos < src.size()) {
                size_t line_begin;
                const std::string_view line = line_at(src, pos, line_begin);
                pos = line_begin + line.size() + 1;
                if (line.contains("var<push_constant>"))
                    out.uses_push_constants = true;
                if (line.empty() || line[0] != '@')
                    continue;
                if (line.find("@binding(") == std::string_view::npos || line.find("@group(") == std::string_view::npos)
                    continue;
                shader_resource r;
                uint32_t binding = 0, group = 0;
                size_t first = std::string_view::npos, last = 0;
                {
                    const size_t b = line.find("@binding(");
                    cursor c{line, b + 9};
                    if (!c.number(binding) || !c.eat(")"))
                        continue;
                    first = (std::min)(first, b);
                    last = (std::max)(last, c.at);
                }
                {
                    const size_t g = line.find("@group(");
                    cursor c{line, g + 7};
                    if (!c.number(group) || !c.eat(")"))
                        continue;
                    first = (std::min)(first, g);
                    last = (std::max)(last, c.at);
                }
                r.binding = binding;
                r.set = group;
                r.clause_begin = line_begin + first;
                r.clause_end = line_begin + last;
                const size_t var_at = line.find("var", last);
                if (var_at == std::string_view::npos)
                    continue;
                cursor c{line, var_at + 3};
                std::string_view space;
                if (line[c.at] == '<') {
                    const size_t close = line.find('>', c.at);
                    if (close == std::string_view::npos)
                        continue;
                    space = line.substr(c.at + 1, close - c.at - 1);
                    c.at = close + 1;
                }
                r.name = std::string(c.ident());
                c.skip();
                if (c.at < line.size() && line[c.at] == ':')
                    c.at++;
                c.skip();
                const std::string_view type = line.substr(c.at);
                const auto starts = [&](std::string_view p) { return type.substr(0, p.size()) == p; };
                if (space.substr(0, 7) == "uniform") {
                    r.kind = resource_kind::uniform_buffer;
                } else if (space.substr(0, 7) == "storage") {
                    r.kind =
                        space.contains("read_write") ? resource_kind::storage_buffer_rw : resource_kind::storage_buffer;
                } else if (starts("sampler_comparison")) {
                    r.kind = resource_kind::sampler;
                    r.comparison = true;
                } else if (starts("sampler")) {
                    r.kind = resource_kind::sampler;
                } else if (starts("texture_storage_")) {
                    r.kind = resource_kind::storage_texture;
                    r.dim = starts("texture_storage_2d_array") ? texture_dim::tex2d_array
                            : starts("texture_storage_3d")     ? texture_dim::tex3d
                                                               : texture_dim::tex2d;
                } else if (starts("texture_depth_")) {
                    r.kind = resource_kind::sampled_texture;
                    r.depth = true;
                    r.dim = starts("texture_depth_2d_array") ? texture_dim::tex2d_array
                            : starts("texture_depth_cube")   ? texture_dim::cube
                                                             : texture_dim::tex2d;
                } else if (starts("texture_")) {
                    r.kind = resource_kind::sampled_texture;
                    r.dim = starts("texture_2d_array") ? texture_dim::tex2d_array
                            : starts("texture_3d")     ? texture_dim::tex3d
                            : starts("texture_cube")   ? texture_dim::cube
                                                       : texture_dim::tex2d;
                } else {
                    continue;
                }
                out.resources.push_back(std::move(r));
            }
        }

        void reflect_msl(std::string_view src, shader_interface& out) {
            uint32_t texture_base = ~0u, texture_set = 1, buffer_base = ~0u, buffer_set = 0;
            if (const size_t m = src.find("// conduit:msl-sets "); m != std::string_view::npos) {
                const std::string line(src.substr(m, src.find('\n', m) - m));
                unsigned tb = 0, ts = 0, bb = 0, bs = 0;
                if (std::sscanf(line.c_str(), "// conduit:msl-sets texture %u %u buffer %u %u", &tb, &ts, &bb, &bs) ==
                    4) {
                    texture_base = tb;
                    texture_set = ts;
                    buffer_base = bb;
                    buffer_set = bs;
                }
            }
            size_t pos = 0;
            while ((pos = src.find("[[", pos)) != std::string_view::npos) {
                const size_t clause_begin = pos;
                cursor c{src, pos + 2};
                const std::string_view kind = c.ident();
                pos += 2;
                if (kind != "buffer" && kind != "texture" && kind != "sampler")
                    continue;
                if (!c.eat("("))
                    continue;
                uint32_t index = 0;
                if (!c.number(index) || !c.eat(")") || !c.eat("]]"))
                    continue;
                shader_resource r;
                r.binding = index;
                r.native_index = index;
                r.clause_begin = clause_begin;
                r.clause_end = c.at;
                size_t b = clause_begin;
                int angle = 0;
                while (b > 0) {
                    const char ch = src[b - 1];
                    if (ch == '>')
                        angle++;
                    else if (ch == '<')
                        angle--;
                    else if ((ch == ',' || ch == '(') && angle <= 0)
                        break;
                    b--;
                }
                std::string_view decl = src.substr(b, clause_begin - b);
                while (!decl.empty() && is_space(decl.back()))
                    decl.remove_suffix(1);
                size_t name_begin = decl.size();
                while (name_begin > 0 && is_ident(decl[name_begin - 1]))
                    name_begin--;
                r.name = std::string(decl.substr(name_begin));
                while (!decl.empty() && is_space(decl.front()))
                    decl.remove_prefix(1);
                const auto starts = [&](std::string_view p) { return decl.substr(0, p.size()) == p; };
                if (kind == "sampler") {
                    r.kind = resource_kind::sampler;
                    r.set = 1;
                } else if (kind == "texture") {
                    r.set = 1;
                    if (starts("depth")) {
                        r.depth = true;
                        r.kind = resource_kind::sampled_texture;
                        r.dim = starts("depth2d_array") ? texture_dim::tex2d_array
                                : starts("depthcube")   ? texture_dim::cube
                                                        : texture_dim::tex2d;
                    } else {
                        const bool write = decl.contains("access::write") || decl.contains("access::read_write");
                        r.kind = write ? resource_kind::storage_texture : resource_kind::sampled_texture;
                        r.dim = starts("texture2d_array") ? texture_dim::tex2d_array
                                : starts("texture3d")     ? texture_dim::tex3d
                                : starts("texturecube")   ? texture_dim::cube
                                                          : texture_dim::tex2d;
                    }
                    if (index >= texture_base) {
                        r.set = texture_set;
                        r.binding = index - texture_base;
                    }
                } else {
                    r.set = 0;
                    if (index >= buffer_base) {
                        r.set = buffer_set;
                        r.binding = index - buffer_base;
                    }
                    if (starts("constant"))
                        r.kind = resource_kind::uniform_buffer;
                    else if (starts("const device"))
                        r.kind = resource_kind::storage_buffer;
                    else
                        r.kind = resource_kind::storage_buffer_rw;
                }
                out.resources.push_back(std::move(r));
                pos = c.at;
            }
        }
    }

    const shader_resource* shader_interface::find(uint32_t set, uint32_t binding, resource_kind kind) const {
        for (const shader_resource& r : resources)
            if (r.set == set && r.binding == binding && r.kind == kind)
                return &r;
        return nullptr;
    }

    shader_interface reflect_shader(shader_format format, std::string_view source) {
        shader_interface out;
        switch (format) {
        case shader_format::wgsl:
            reflect_wgsl(source, out);
            break;
        case shader_format::hlsl:
            reflect_hlsl(source, out);
            break;
        case shader_format::msl:
            reflect_msl(source, out);
            break;
        default:
            break;
        }
        return out;
    }
}
