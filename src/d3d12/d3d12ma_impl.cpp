#include "d3d12/d3d12_device.hpp"

#if !defined(_MSC_VER) && !__has_include(<winrt/base.h>)
#undef __uuidof
#define __uuidof(type) ::winrt::guid_of<__typeof(type)>()
#endif

#if defined(__clang__)
#pragma clang diagnostic ignored "-Weverything"
#elif defined(_MSC_VER)
#pragma warning(push, 0)
#elif defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#endif
#include <D3D12MemAlloc.cpp> // NOLINT(bugprone-suspicious-include)
