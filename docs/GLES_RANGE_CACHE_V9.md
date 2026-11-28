# GLES storage-buffer range binding follow-up (v9)

The GLES command replay engine now caches `glBindBufferRange(GL_SHADER_STORAGE_BUFFER, ...)` state for 32 indexed binding points. Subsequent draws using an identical buffer, offset and range size skip the redundant driver call.

The cache is reset on context resynchronization and invalidated when a referenced buffer is destroyed. Uniform-buffer caching remains separate. Indices outside the fixed cache are bound normally. No shader semantics or VSync settings were changed by this patch.

This is a targeted CPU driver-call optimization for repeated GLES bindings, not a guarantee of higher Android FPS. It has not been executed on an Android device. Full CMake configuration was blocked by the unavailable SPIRV-Cross fetch, and standalone syntax compilation was blocked by missing GLES development headers.
