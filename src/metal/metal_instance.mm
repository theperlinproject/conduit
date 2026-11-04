#include "backends.hpp"
#include "metal/metal_internal.hpp"

#import <CoreGraphics/CoreGraphics.h>
#include <cstdio>

namespace conduit::metal {
    namespace {
        class swapchain_impl final : public swapchain {
        public:
            swapchain_impl(metal::device& device, const swapchain_desc& desc)
                : device_(device), desc_(desc), width_(desc.width), height_(desc.height), present_mode_(desc.present),
                  hdr_(desc.hdr) {}

            ~swapchain_impl() override {
                device_.wait_idle();
                drawable_ = nil;
                if (handle_)
                    device_.destroy(handle_);
            }

            bool init() {
                id object = (__bridge id)desc_.window.window;
                if ([object isKindOfClass:[CAMetalLayer class]]) {
                    layer_ = object;
                } else if ([object respondsToSelector:NSSelectorFromString([NSString stringWithUTF8String:"layer"])]) {
                    id backing = [object valueForKey:[NSString stringWithUTF8String:"layer"]];
                    if ([backing isKindOfClass:[CAMetalLayer class]])
                        layer_ = backing;
                }
                if (!layer_)
                    return false;
                format_ = hdr_ ? format::rgba16_float : format::bgra8_unorm;
                layer_.device = device_.dev();
                layer_.pixelFormat = hdr_ ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
                layer_.framebufferOnly = !desc_.allow_sampling;
                // always three drawables: with two, nextDrawable stalls a whole refresh whenever a frame runs long
                layer_.maximumDrawableCount = 3;
#if TARGET_OS_OSX
                if (hdr_) {
                    layer_.wantsExtendedDynamicRangeContent = YES;
                    CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearDisplayP3);
                    layer_.colorspace = space;
                    CGColorSpaceRelease(space);
                }
#endif
                if (!width_ || !height_) {
                    const CGSize size = layer_.drawableSize;
                    width_ = (std::max)(1u, static_cast<uint32_t>(size.width));
                    height_ = (std::max)(1u, static_cast<uint32_t>(size.height));
                }
                layer_.drawableSize = CGSizeMake(width_, height_);
                set_present_mode(present_mode_);
                handle_ = device_.create_surface_texture(format_, width_, height_, desc_.allow_sampling);
                return handle_.value != 0;
            }

            bool acquire(image& out) override {
                if (!width_ || !height_ || !handle_)
                    return false;
                drawable_ = [layer_ nextDrawable];
                if (!drawable_)
                    return false;
                device_.set_surface_texture(handle_, drawable_.texture);
                out.texture = handle_;
                out.index = counter_++ % image_count();
                return true;
            }

            void present() override {
                if (!drawable_)
                    return;
                id<MTLCommandBuffer> cb = [device_.queue() commandBuffer];
                [cb presentDrawable:drawable_];
                [cb commit];
                drawable_ = nil;
            }

            bool resize(uint32_t w, uint32_t h) override {
                width_ = w;
                height_ = h;
                if (!w || !h)
                    return true;
                drawable_ = nil;
                layer_.drawableSize = CGSizeMake(w, h);
                device_.resize_surface_texture(handle_, w, h);
                return true;
            }

            void set_present_mode(present_mode mode) override {
                present_mode_ = mode;
#if TARGET_OS_OSX
                layer_.displaySyncEnabled = mode != present_mode::immediate;
#endif
            }

            format get_format() const override { return format_; }
            uint32_t width() const override { return width_; }
            uint32_t height() const override { return height_; }
            present_mode get_present_mode() const override { return present_mode_; }
            uint32_t image_count() const override { return static_cast<uint32_t>(layer_.maximumDrawableCount); }
            bool is_hdr() const override { return hdr_; }

        private:
            device& device_;
            swapchain_desc desc_;
            CAMetalLayer* layer_ = nil;
            id<CAMetalDrawable> drawable_ = nil;
            texture_handle handle_;
            uint32_t width_, height_;
            present_mode present_mode_;
            bool hdr_;
            format format_ = format::bgra8_unorm;
            uint32_t counter_ = 0;
        };

        uint32_t vendor_of(NSString* name) {
            NSString* n = [name lowercaseString];
            if ([n containsString:@"apple"])
                return vendor_id::apple;
            if ([n containsString:@"amd"] || [n containsString:@"radeon"])
                return vendor_id::amd;
            if ([n containsString:@"nvidia"] || [n containsString:@"geforce"])
                return vendor_id::nvidia;
            if ([n containsString:@"intel"])
                return vendor_id::intel;
            return 0;
        }

        class metal_instance final : public instance {
        public:
            explicit metal_instance(const instance_options& options) : options_(options) {}

            bool init() {
                NSArray<id<MTLDevice>>* all = nil;
#if TARGET_OS_OSX
                all = MTLCopyAllDevices();
#else
                id<MTLDevice> d = MTLCreateSystemDefaultDevice();
                all = d ? [NSArray arrayWithObjects:&d count:1] : [NSArray array];
#endif
                id<MTLDevice> preferred = MTLCreateSystemDefaultDevice();
                NSMutableArray<id<MTLDevice>>* ordered = [NSMutableArray array];
                if (preferred)
                    [ordered addObject:preferred];
                for (id<MTLDevice> d in all)
                    if (d != preferred)
                        [ordered addObject:d];
                for (id<MTLDevice> d in ordered) {
                    adapter_info a;
                    a.api = api::metal;
                    a.index = static_cast<uint32_t>(adapters_.size());
                    a.name = [d.name UTF8String];
                    a.vendor_id = vendor_of(d.name);
                    a.dedicated_vram = d.hasUnifiedMemory ? 0 : d.recommendedMaxWorkingSetSize;
                    a.shared_memory = d.hasUnifiedMemory ? d.recommendedMaxWorkingSetSize : 0;
#if TARGET_OS_OSX
                    a.type = d.hasUnifiedMemory || d.lowPower ? adapter_type::integrated : adapter_type::discrete;
                    if (d.isRemovable)
                        a.type = adapter_type::discrete;
#else
                    a.type = adapter_type::integrated;
#endif
                    a.api_version = 0;
                    for (const MTLGPUFamily fam :
                         {MTLGPUFamilyApple9, MTLGPUFamilyApple8, MTLGPUFamilyApple7, MTLGPUFamilyApple6,
                          MTLGPUFamilyApple5, MTLGPUFamilyApple4, MTLGPUFamilyApple3})
                        if ([d supportsFamily:fam]) {
                            a.api_version = static_cast<uint32_t>(fam);
                            break;
                        }
                    if (![d supportsFamily:MTLGPUFamilyApple3] && ![d supportsFamily:MTLGPUFamilyMac2]) {
                        a.usable = false;
                        a.unusable_reason = "needs an Apple3 or Mac2 class gpu";
                    }
                    adapters_.push_back(std::move(a));
                    native_.push_back(d);
                }
                return !adapters_.empty();
            }

            api get_api() const override { return api::metal; }
            const std::vector<adapter_info>& adapters() const override { return adapters_; }

            std::unique_ptr<conduit::device> create_device(const conduit::device_options& device_options) override {
                if (device_options.adapter_index >= adapters_.size() || !adapters_[device_options.adapter_index].usable)
                    return nullptr;
                auto device = std::make_unique<metal::device>(native_[device_options.adapter_index], device_options,
                                                              adapters_[device_options.adapter_index]);
                if (!device->init()) {
                    std::fprintf(stderr, "[metal] device creation failed\n");
                    return nullptr;
                }
                return device;
            }

        private:
            instance_options options_;
            std::vector<adapter_info> adapters_;
            std::vector<id<MTLDevice>> native_;
        };
    }

    std::unique_ptr<conduit::swapchain> make_swapchain(metal::device& device, const swapchain_desc& desc) {
        auto sc = std::make_unique<swapchain_impl>(device, desc);
        if (!sc->init())
            return nullptr;
        return sc;
    }
}

namespace conduit::backends {
    std::unique_ptr<conduit::instance> create_metal_instance(const instance_options& options) {
        auto instance = std::make_unique<metal::metal_instance>(options);
        if (!instance->init())
            return nullptr;
        return instance;
    }
}
