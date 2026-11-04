#pragma once
#include <windows.h>
#include <unknwn.h>

#include <utility>

#if !defined(_MSC_VER) && __has_include(<dxguids/dxguids.h>)
#include <dxguids/dxguids.h>
#define CONDUIT_HAS_DXGUIDS 1
#endif

namespace winrt {
    namespace impl {
        template <class T> GUID iid_of() {
#ifndef _MSC_VER
#ifdef CONDUIT_HAS_DXGUIDS
            if constexpr (requires { uuidof<T>(); })
                return uuidof<T>();
            else
#endif
                return __mingw_uuidof<T>();
#else
            return __uuidof(T);
#endif
        }
    }

    template <class T> GUID guid_of() {
        return impl::iid_of<T>();
    }

    template <class T> class com_ptr {
    public:
        com_ptr() noexcept = default;
        com_ptr(std::nullptr_t) noexcept {}
        com_ptr(const com_ptr& o) noexcept : p_(o.p_) {
            if (p_)
                p_->AddRef();
        }
        com_ptr(com_ptr&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
        com_ptr& operator=(const com_ptr& o) noexcept {
            if (this != &o) {
                com_ptr tmp(o);
                std::swap(p_, tmp.p_);
            }
            return *this;
        }
        com_ptr& operator=(com_ptr&& o) noexcept {
            if (this != &o) {
                release();
                p_ = o.p_;
                o.p_ = nullptr;
            }
            return *this;
        }
        com_ptr& operator=(std::nullptr_t) noexcept {
            release();
            return *this;
        }
        ~com_ptr() noexcept { release(); }

        T* get() const noexcept { return p_; }
        T* operator->() const noexcept { return p_; }
        explicit operator bool() const noexcept { return p_ != nullptr; }
        T** put() noexcept {
            release();
            return &p_;
        }
        void** put_void() noexcept { return reinterpret_cast<void**>(put()); }
        void copy_from(T* other) noexcept {
            if (other)
                other->AddRef();
            release();
            p_ = other;
        }
        void attach(T* other) noexcept {
            release();
            p_ = other;
        }
        T* detach() noexcept {
            T* out = p_;
            p_ = nullptr;
            return out;
        }
        template <class U> com_ptr<U> try_as() const noexcept {
            com_ptr<U> out;
            if (p_)
                p_->QueryInterface(guid_of<U>(), out.put_void());
            return out;
        }

    private:
        void release() noexcept {
            if (p_) {
                p_->Release();
                p_ = nullptr;
            }
        }
        T* p_ = nullptr;
    };

    class handle {
    public:
        handle() noexcept = default;
        explicit handle(void* h) noexcept : h_(h) {}
        handle(const handle&) = delete;
        handle& operator=(const handle&) = delete;
        handle(handle&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
        handle& operator=(handle&& o) noexcept {
            if (this != &o) {
                close();
                h_ = o.h_;
                o.h_ = nullptr;
            }
            return *this;
        }
        ~handle() noexcept { close(); }
        void* get() const noexcept { return h_; }
        explicit operator bool() const noexcept { return h_ != nullptr; }
        void attach(void* other) noexcept {
            close();
            h_ = other;
        }
        void close() noexcept {
            if (h_) {
                CloseHandle(h_);
                h_ = nullptr;
            }
        }

    private:
        void* h_ = nullptr;
    };
}
