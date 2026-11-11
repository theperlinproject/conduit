#pragma once
// OpenGL ES entry points, loaded at run time so conduit never links a GL library. Headers come from the Khronos
// registry (or the NDK / Emscripten); the prototypes they declare are only used to spell the function pointer types.
#ifndef GL_GLES_PROTOTYPES
#define GL_GLES_PROTOTYPES 1 // NOLINT(readability-identifier-naming): the Khronos header's switch
#endif
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES 1 // NOLINT(readability-identifier-naming): the Khronos header's switch
#endif
#if defined(__has_include) && __has_include(<GLES3/gl32.h>)
#include <GLES3/gl32.h>
#elif defined(__has_include) && __has_include(<GLES3/gl31.h>)
#include <GLES3/gl31.h>
#else
#include <GLES3/gl3.h>
#endif
#include <GLES2/gl2ext.h>

#include <cstdint>
#include <functional>
#include <string>

// every function used unconditionally: OpenGL ES 3.0, which is also what WebGL 2 exposes
#define CONDUIT_GLES_CORE(X)                                                                                           \
    X(glGetString)                                                                                                     \
    X(glGetStringi)                                                                                                    \
    X(glGetIntegerv)                                                                                                   \
    X(glGetFloatv)                                                                                                     \
    X(glGetError)                                                                                                      \
    X(glEnable)                                                                                                        \
    X(glDisable)                                                                                                       \
    X(glFlush)                                                                                                         \
    X(glFinish)                                                                                                        \
    X(glPixelStorei)                                                                                                   \
    X(glViewport)                                                                                                      \
    X(glScissor)                                                                                                       \
    X(glDepthRangef)                                                                                                   \
    X(glClearBufferfv)                                                                                                 \
    X(glClearBufferiv)                                                                                                 \
    X(glClearBufferuiv)                                                                                                \
    X(glClearBufferfi)                                                                                                 \
    X(glColorMask)                                                                                                     \
    X(glDepthMask)                                                                                                     \
    X(glStencilMaskSeparate)                                                                                           \
    X(glDepthFunc)                                                                                                     \
    X(glStencilFuncSeparate)                                                                                           \
    X(glStencilOpSeparate)                                                                                             \
    X(glBlendFuncSeparate)                                                                                             \
    X(glBlendEquationSeparate)                                                                                         \
    X(glBlendColor)                                                                                                    \
    X(glCullFace)                                                                                                      \
    X(glFrontFace)                                                                                                     \
    X(glPolygonOffset)                                                                                                 \
    X(glGenBuffers)                                                                                                    \
    X(glDeleteBuffers)                                                                                                 \
    X(glBindBuffer)                                                                                                    \
    X(glBufferData)                                                                                                    \
    X(glBufferSubData)                                                                                                 \
    X(glBindBufferRange)                                                                                               \
    X(glCopyBufferSubData)                                                                                             \
    X(glGenTextures)                                                                                                   \
    X(glDeleteTextures)                                                                                                \
    X(glBindTexture)                                                                                                   \
    X(glActiveTexture)                                                                                                 \
    X(glTexStorage2D)                                                                                                  \
    X(glTexStorage3D)                                                                                                  \
    X(glTexSubImage2D)                                                                                                 \
    X(glTexSubImage3D)                                                                                                 \
    X(glCompressedTexSubImage2D)                                                                                       \
    X(glCompressedTexSubImage3D)                                                                                       \
    X(glTexParameteri)                                                                                                 \
    X(glGenSamplers)                                                                                                   \
    X(glDeleteSamplers)                                                                                                \
    X(glBindSampler)                                                                                                   \
    X(glSamplerParameteri)                                                                                             \
    X(glSamplerParameterf)                                                                                             \
    X(glGenFramebuffers)                                                                                               \
    X(glDeleteFramebuffers)                                                                                            \
    X(glBindFramebuffer)                                                                                               \
    X(glFramebufferTexture2D)                                                                                          \
    X(glFramebufferTextureLayer)                                                                                       \
    X(glFramebufferRenderbuffer)                                                                                       \
    X(glCheckFramebufferStatus)                                                                                        \
    X(glDrawBuffers)                                                                                                   \
    X(glReadBuffer)                                                                                                    \
    X(glBlitFramebuffer)                                                                                               \
    X(glInvalidateFramebuffer)                                                                                         \
    X(glReadPixels)                                                                                                    \
    X(glGenRenderbuffers)                                                                                              \
    X(glDeleteRenderbuffers)                                                                                           \
    X(glBindRenderbuffer)                                                                                              \
    X(glRenderbufferStorageMultisample)                                                                                \
    X(glCreateShader)                                                                                                  \
    X(glShaderSource)                                                                                                  \
    X(glCompileShader)                                                                                                 \
    X(glGetShaderiv)                                                                                                   \
    X(glGetShaderInfoLog)                                                                                              \
    X(glDeleteShader)                                                                                                  \
    X(glCreateProgram)                                                                                                 \
    X(glAttachShader)                                                                                                  \
    X(glLinkProgram)                                                                                                   \
    X(glGetProgramiv)                                                                                                  \
    X(glGetProgramInfoLog)                                                                                             \
    X(glDeleteProgram)                                                                                                 \
    X(glUseProgram)                                                                                                    \
    X(glGetUniformLocation)                                                                                            \
    X(glGetUniformBlockIndex)                                                                                          \
    X(glUniformBlockBinding)                                                                                           \
    X(glUniform1i)                                                                                                     \
    X(glGenVertexArrays)                                                                                               \
    X(glBindVertexArray)                                                                                               \
    X(glDeleteVertexArrays)                                                                                            \
    X(glEnableVertexAttribArray)                                                                                       \
    X(glDisableVertexAttribArray)                                                                                      \
    X(glVertexAttribPointer)                                                                                           \
    X(glVertexAttribIPointer)                                                                                          \
    X(glVertexAttribDivisor)                                                                                           \
    X(glDrawArrays)                                                                                                    \
    X(glDrawArraysInstanced)                                                                                           \
    X(glDrawElements)                                                                                                  \
    X(glDrawElementsInstanced)                                                                                         \
    X(glFenceSync)                                                                                                     \
    X(glDeleteSync)                                                                                                    \
    X(glClientWaitSync)

// OpenGL ES 3.1, 3.2 and extension entry points. Any of these may be null. The types are spelled out so the table
// builds against headers that stop at ES 3.0 (Emscripten) and so one name can come from either core or an extension.
#define CONDUIT_GLES_OPTIONAL(X)                                                                                       \
    X(glMapBufferRange, void*, (GLenum, GLintptr, GLsizeiptr, GLbitfield))                                             \
    X(glUnmapBuffer, GLboolean, (GLenum))                                                                              \
    X(glDispatchCompute, void, (GLuint, GLuint, GLuint))                                                               \
    X(glDispatchComputeIndirect, void, (GLintptr))                                                                     \
    X(glDrawElementsIndirect, void, (GLenum, GLenum, const void*))                                                     \
    X(glDrawArraysIndirect, void, (GLenum, const void*))                                                               \
    X(glMemoryBarrier, void, (GLbitfield))                                                                             \
    X(glBindImageTexture, void, (GLuint, GLuint, GLint, GLboolean, GLint, GLenum, GLenum))                             \
    X(glCopyImageSubData, void,                                                                                        \
      (GLuint, GLenum, GLint, GLint, GLint, GLint, GLuint, GLenum, GLint, GLint, GLint, GLint, GLsizei, GLsizei,       \
       GLsizei))                                                                                                       \
    X(glPushDebugGroup, void, (GLenum, GLuint, GLsizei, const GLchar*))                                                \
    X(glPopDebugGroup, void, ())                                                                                       \
    X(glBlendFuncSeparatei, void, (GLuint, GLenum, GLenum, GLenum, GLenum))                                            \
    X(glBlendEquationSeparatei, void, (GLuint, GLenum, GLenum))                                                        \
    X(glColorMaski, void, (GLuint, GLboolean, GLboolean, GLboolean, GLboolean))                                        \
    X(glEnablei, void, (GLenum, GLuint))                                                                               \
    X(glDisablei, void, (GLenum, GLuint))                                                                              \
    X(glSamplerParameterfv, void, (GLuint, GLenum, const GLfloat*))                                                    \
    X(glUniform1iv, void, (GLint, GLsizei, const GLint*))

// values that only exist in newer headers, or only in extension headers
#ifndef GL_SHADER_STORAGE_BUFFER
#define GL_SHADER_STORAGE_BUFFER 0x90D2
#endif
#ifndef GL_DRAW_INDIRECT_BUFFER
#define GL_DRAW_INDIRECT_BUFFER 0x8F3F
#endif
#ifndef GL_DISPATCH_INDIRECT_BUFFER
#define GL_DISPATCH_INDIRECT_BUFFER 0x90EE
#endif
#ifndef GL_COMPUTE_SHADER
#define GL_COMPUTE_SHADER 0x91B9
#endif
#ifndef GL_READ_WRITE
#define GL_READ_WRITE 0x88BA
#endif
#ifndef GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS
#define GL_MAX_SHADER_STORAGE_BUFFER_BINDINGS 0x90DD
#endif
#ifndef GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT
#define GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT 0x90DF
#endif
#ifndef GL_MAX_IMAGE_UNITS
#define GL_MAX_IMAGE_UNITS 0x8F38
#endif
#ifndef GL_ALL_BARRIER_BITS
#define GL_ALL_BARRIER_BITS 0xFFFFFFFFu
#endif
#ifndef GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT
#define GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT 0x00000001
#define GL_ELEMENT_ARRAY_BARRIER_BIT 0x00000002
#define GL_UNIFORM_BARRIER_BIT 0x00000004
#define GL_TEXTURE_FETCH_BARRIER_BIT 0x00000008
#define GL_SHADER_IMAGE_ACCESS_BARRIER_BIT 0x00000020
#define GL_COMMAND_BARRIER_BIT 0x00000040
#define GL_PIXEL_BUFFER_BARRIER_BIT 0x00000080
#define GL_TEXTURE_UPDATE_BARRIER_BIT 0x00000100
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x00000200
#define GL_FRAMEBUFFER_BARRIER_BIT 0x00000400
#define GL_SHADER_STORAGE_BARRIER_BIT 0x00002000
#endif
#ifndef GL_MAP_READ_BIT
#define GL_MAP_READ_BIT 0x0001
#endif
#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif
#ifndef GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
#define GL_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define GL_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3
#endif
#ifndef GL_COMPRESSED_RGBA_BPTC_UNORM_EXT
#define GL_COMPRESSED_RGBA_BPTC_UNORM_EXT 0x8E8C
#endif
#ifndef GL_R16_EXT
#define GL_R16_EXT 0x822A
#define GL_RG16_EXT 0x822C
#define GL_RGBA16_EXT 0x805B
#endif
#ifndef GL_COLOR_BUFFER_BIT
#define GL_COLOR_BUFFER_BIT 0x00004000
#endif

namespace conduit::gles {
    struct gl_api {
// NOLINTNEXTLINE(bugprone-macro-parentheses): name is a declarator
#define CONDUIT_GLES_DECLARE_CORE(name) decltype(&::name) name = nullptr;
        CONDUIT_GLES_CORE(CONDUIT_GLES_DECLARE_CORE)
#undef CONDUIT_GLES_DECLARE_CORE
// NOLINTNEXTLINE(bugprone-macro-parentheses): ret, name and args are pieces of a declarator
#define CONDUIT_GLES_DECLARE_OPTIONAL(name, ret, args) ret(GL_APIENTRY* name) args = nullptr;
        CONDUIT_GLES_OPTIONAL(CONDUIT_GLES_DECLARE_OPTIONAL)
#undef CONDUIT_GLES_DECLARE_OPTIONAL

#ifndef __EMSCRIPTEN__
        // false when a function every OpenGL ES 3.0 implementation has is missing. optional entry points fall back to
        // their EXT and OES names; whether the extension is actually there is for the caller to check
        bool load(const std::function<void*(const char*)>& get_proc) {
            bool ok = true;
#define CONDUIT_GLES_LOAD_CORE(fn)                                                                                     \
    fn = reinterpret_cast<decltype(&::fn)>(get_proc(#fn));                                                             \
    ok = ok && (fn) != nullptr;
            CONDUIT_GLES_CORE(CONDUIT_GLES_LOAD_CORE)
#undef CONDUIT_GLES_LOAD_CORE
            const auto find = [&](const char* base) -> void* {
                if (void* p = get_proc(base))
                    return p;
                std::string name = base;
                if (void* p = get_proc((name + "EXT").c_str()))
                    return p;
                return get_proc((name + "OES").c_str());
            };
#define CONDUIT_GLES_LOAD_OPTIONAL(fn, ret, args) fn = reinterpret_cast<decltype(fn)>(find(#fn));
            CONDUIT_GLES_OPTIONAL(CONDUIT_GLES_LOAD_OPTIONAL)
#undef CONDUIT_GLES_LOAD_OPTIONAL
            return ok;
        }
#else
        // WebGL 2 through Emscripten: the functions are linked in, and the optional ones do not exist
        bool load(const std::function<void*(const char*)>&) {
#define CONDUIT_GLES_LOAD_STATIC(fn) fn = &::fn;
            CONDUIT_GLES_CORE(CONDUIT_GLES_LOAD_STATIC)
#undef CONDUIT_GLES_LOAD_STATIC
            return true;
        }
#endif
    };
}
