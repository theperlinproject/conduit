#include "d3d12/d3d12_device.hpp"

#if !defined(_MSC_VER) && !__has_include(<winrt/base.h>)
#undef __uuidof
#define __uuidof(type) ::winrt::guid_of<__typeof(type)>()
#endif

#include <D3D12MemAlloc.cpp> // NOLINT(bugprone-suspicious-include)
