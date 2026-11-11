#include "vulkan/vulkan_device.hpp"

#include "vulkan/vulkan_convert.hpp"

#include <algorithm>
#include <cstring>

namespace conduit::vk {
    void cmd_list::fail(const std::string& message) {
        device_.report_error(message);
    }

    void cmd_list::begin() {
        pipeline_ = {};
        pipeline_handle_ = {};
        have_pipeline_ = false;
        pipeline_pending_ = false;
        in_rendering_ = false;
        groups_ = {};
        dynamic_offsets_ = {};
        dynamic_counts_ = {};
        dirty_sets_ = 0;
        push_.fill(0);
        push_dirty_ = false;
        push_used_ = 0;
        color_formats_.fill(format::unknown);
        color_count_ = 0;
        depth_format_ = format::unknown;
        have_index_ = false;
        vertex_mask_ = 0;
        invalid_ = false;
        have_viewport_ = have_scissor_ = false;
        vbuffers_.fill(VK_NULL_HANDLE);
        voffsets_.fill(0);
        ibuffer_ = VK_NULL_HANDLE;
        stencil_ref_ = 0;
        open_labels_ = 0;
        counters_ = {};

        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        device_.fn_.vkBeginCommandBuffer(cmd_, &bi);
    }

    void cmd_list::end() {
        if (in_rendering_) {
            fail("command list ended inside a rendering pass");
            if (device_.dynamic_rendering_)
                device_.fn_.vkCmdEndRendering(cmd_);
            else
                device_.fn_.vkCmdEndRenderPass(cmd_);
            in_rendering_ = false;
        }
        device_.fn_.vkEndCommandBuffer(cmd_);
    }

    void cmd_list::barrier(std::span<const texture_barrier> textures, std::span<const buffer_barrier> buffers) {
        if (in_rendering_)
            fail("barrier recorded inside a rendering pass");
        std::vector<VkImageMemoryBarrier2>& images = barrier_images_;
        std::vector<VkBufferMemoryBarrier2>& bufs = barrier_buffers_;
        images.clear();
        bufs.clear();

        for (const texture_barrier& tb : textures) {
            texture_rec* tex = device_.textures_.get(tb.texture);
            if (!tex) {
                fail("barrier on an unknown texture");
                continue;
            }
            if (device_.use_validator_)
                device_.validator_.barrier(tb);
            const bool depth = (tex->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
            const state_use src = use_of(tb.before, queue_, depth);
            const state_use dst = use_of(tb.after, queue_, depth);

            VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            b.srcStageMask = src.stages;
            b.srcAccessMask = src.access;
            b.dstStageMask = dst.stages;
            b.dstAccessMask = dst.access;
            b.oldLayout = src.layout;
            b.newLayout = dst.layout;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            const uint32_t src_family = device_.queues_[static_cast<size_t>(tb.src_queue)].family;
            const uint32_t dst_family = device_.queues_[static_cast<size_t>(tb.dst_queue)].family;
            if (src_family != dst_family && !tex->desc.cross_queue) {
                b.srcQueueFamilyIndex = src_family;
                b.dstQueueFamilyIndex = dst_family;
                if (queue_ == tb.src_queue) {
                    b.dstStageMask = 0;
                    b.dstAccessMask = 0;
                } else {
                    b.srcStageMask = 0;
                    b.srcAccessMask = 0;
                }
            }
            b.image = tex->image;
            const subresource_range r = resolve_range(tb.range, tex->desc);
            b.subresourceRange = {tex->aspect, r.base_mip, r.mip_count, r.base_layer, r.layer_count};
            images.push_back(b);
        }

        for (const buffer_barrier& bb : buffers) {
            buffer_rec* buf = device_.buffers_.get(bb.buffer);
            if (!buf) {
                fail("barrier on an unknown buffer");
                continue;
            }
            if (device_.use_validator_)
                device_.validator_.barrier(bb);
            if (buf->desc.memory == memory_class::upload || buf->desc.memory == memory_class::readback)
                continue;
            const state_use src = use_of(bb.before, queue_, false);
            const state_use dst = use_of(bb.after, queue_, false);
            VkBufferMemoryBarrier2 b{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
            b.srcStageMask = src.stages;
            b.srcAccessMask = src.access;
            b.dstStageMask = dst.stages;
            b.dstAccessMask = dst.access;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            const uint32_t src_family = device_.queues_[static_cast<size_t>(bb.src_queue)].family;
            const uint32_t dst_family = device_.queues_[static_cast<size_t>(bb.dst_queue)].family;
            if (src_family != dst_family && !buf->desc.cross_queue) {
                b.srcQueueFamilyIndex = src_family;
                b.dstQueueFamilyIndex = dst_family;
                if (queue_ == bb.src_queue) {
                    b.dstStageMask = 0;
                    b.dstAccessMask = 0;
                } else {
                    b.srcStageMask = 0;
                    b.srcAccessMask = 0;
                }
            }
            b.buffer = buf->buffer;
            b.offset = 0;
            b.size = VK_WHOLE_SIZE;
            bufs.push_back(b);
        }

        if (images.empty() && bufs.empty())
            return;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = static_cast<uint32_t>(images.size());
        di.pImageMemoryBarriers = images.data();
        di.bufferMemoryBarrierCount = static_cast<uint32_t>(bufs.size());
        di.pBufferMemoryBarriers = bufs.data();
        device_.pipeline_barrier(cmd_, di);
        counters_.barriers += images.size() + bufs.size();
    }

    void cmd_list::uav_barrier(buffer_handle) {
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        constexpr auto shaders = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.srcStageMask = queue_ == queue_type::graphics ? shaders : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstStageMask = mb.srcStageMask | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        mb.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        mb.dstAccessMask =
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        device_.pipeline_barrier(cmd_, di);
        counters_.barriers++;
    }

    void cmd_list::uav_barrier(texture_handle h) {
        texture_rec* tex = device_.textures_.get(h);
        if (!tex)
            return;
        VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        constexpr auto shaders = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcStageMask = b.dstStageMask =
            queue_ == queue_type::graphics ? shaders : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        b.oldLayout = b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = tex->image;
        b.subresourceRange = {tex->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers = &b;
        device_.pipeline_barrier(cmd_, di);
        counters_.barriers++;
    }

    void cmd_list::copy_buffer(buffer_handle dst, uint64_t dst_offset, buffer_handle src, uint64_t src_offset,
                               uint64_t size) {
        buffer_rec* d = device_.buffers_.get(dst);
        buffer_rec* s = device_.buffers_.get(src);
        if (!d || !s) {
            fail("CopyBuffer: unknown buffer");
            return;
        }
        if (device_.use_validator_) {
            device_.validator_.expect_buffer(src, state::copy_src, "CopyBuffer src");
            device_.validator_.expect_buffer(dst, state::copy_dst, "CopyBuffer dst");
        }
        VkBufferCopy region{src_offset, dst_offset, size};
        device_.fn_.vkCmdCopyBuffer(cmd_, s->buffer, d->buffer, 1, &region);
    }

    void cmd_list::copy_buffer_to_texture(texture_handle dst, buffer_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        texture_rec* t = device_.textures_.get(dst);
        buffer_rec* b = device_.buffers_.get(src);
        if (!t || !b) {
            fail("CopyBufferToTexture: unknown resource");
            return;
        }
        const format_info& fi = get_format_info(t->desc.format);
        std::vector<VkBufferImageCopy> copies;
        copies.reserve(regions.size());
        for (const buffer_texture_copy& r : regions) {
            if (device_.use_validator_) {
                device_.validator_.expect_buffer(src, state::copy_src, "CopyBufferToTexture src");
                device_.validator_.expect_texture(dst, {r.mip, 1, r.layer, 1}, state::copy_dst,
                                                  "CopyBufferToTexture dst");
            }
            VkBufferImageCopy c{};
            c.bufferOffset = r.buffer_offset;
            c.bufferRowLength = r.buffer_row_pitch ? r.buffer_row_pitch / fi.bytes_per_block * fi.block_width : 0;
            c.imageSubresource = {
                static_cast<VkImageAspectFlags>(fi.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                r.mip, r.layer, 1};
            c.imageOffset = {static_cast<int32_t>(r.x), static_cast<int32_t>(r.y), static_cast<int32_t>(r.z)};
            c.imageExtent = {r.width, r.height, r.depth};
            copies.push_back(c);
        }
        device_.fn_.vkCmdCopyBufferToImage(cmd_, b->buffer, t->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                           static_cast<uint32_t>(copies.size()), copies.data());
    }

    void cmd_list::copy_texture_to_buffer(buffer_handle dst, texture_handle src,
                                          std::span<const buffer_texture_copy> regions) {
        texture_rec* t = device_.textures_.get(src);
        buffer_rec* b = device_.buffers_.get(dst);
        if (!t || !b) {
            fail("CopyTextureToBuffer: unknown resource");
            return;
        }
        const format_info& fi = get_format_info(t->desc.format);
        std::vector<VkBufferImageCopy> copies;
        copies.reserve(regions.size());
        for (const buffer_texture_copy& r : regions) {
            if (device_.use_validator_) {
                device_.validator_.expect_buffer(dst, state::copy_dst, "CopyTextureToBuffer dst");
                device_.validator_.expect_texture(src, {r.mip, 1, r.layer, 1}, state::copy_src,
                                                  "CopyTextureToBuffer src");
            }
            VkBufferImageCopy c{};
            c.bufferOffset = r.buffer_offset;
            c.bufferRowLength = r.buffer_row_pitch ? r.buffer_row_pitch / fi.bytes_per_block * fi.block_width : 0;
            c.imageSubresource = {
                static_cast<VkImageAspectFlags>(fi.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                r.mip, r.layer, 1};
            c.imageOffset = {static_cast<int32_t>(r.x), static_cast<int32_t>(r.y), static_cast<int32_t>(r.z)};
            c.imageExtent = {r.width, r.height, r.depth};
            copies.push_back(c);
        }
        device_.fn_.vkCmdCopyImageToBuffer(cmd_, t->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, b->buffer,
                                           static_cast<uint32_t>(copies.size()), copies.data());
    }

    void cmd_list::copy_texture(texture_handle dst, texture_handle src, std::span<const texture_copy> regions) {
        texture_rec* d = device_.textures_.get(dst);
        texture_rec* s = device_.textures_.get(src);
        if (!d || !s) {
            fail("CopyTexture: unknown texture");
            return;
        }
        std::vector<VkImageCopy> copies;
        copies.reserve(regions.size());
        for (const texture_copy& r : regions) {
            if (device_.use_validator_) {
                device_.validator_.expect_texture(src, {r.src_mip, 1, r.src_layer, 1}, state::copy_src,
                                                  "CopyTexture src");
                device_.validator_.expect_texture(dst, {r.dst_mip, 1, r.dst_layer, 1}, state::copy_dst,
                                                  "CopyTexture dst");
            }
            VkImageCopy c{};
            c.srcSubresource = {s->aspect, r.src_mip, r.src_layer, 1};
            c.dstSubresource = {d->aspect, r.dst_mip, r.dst_layer, 1};
            c.srcOffset = {static_cast<int32_t>(r.src_x), static_cast<int32_t>(r.src_y), 0};
            c.dstOffset = {static_cast<int32_t>(r.dst_x), static_cast<int32_t>(r.dst_y), 0};
            c.extent = {r.width, r.height, 1};
            copies.push_back(c);
        }
        device_.fn_.vkCmdCopyImage(cmd_, s->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d->image,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, static_cast<uint32_t>(copies.size()),
                                   copies.data());
    }

    void cmd_list::fill_buffer(buffer_handle dst, uint64_t offset, uint64_t size, uint32_t value) {
        buffer_rec* b = device_.buffers_.get(dst);
        if (!b) {
            fail("FillBuffer: unknown buffer");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(dst, state::copy_dst, "FillBuffer dst");
        device_.fn_.vkCmdFillBuffer(cmd_, b->buffer, offset, size, value);
    }

    void cmd_list::resolve_texture(texture_handle dst, texture_handle src) {
        texture_rec* d = device_.textures_.get(dst);
        texture_rec* s = device_.textures_.get(src);
        if (!d || !s) {
            fail("ResolveTexture: unknown texture");
            return;
        }
        if (device_.use_validator_) {
            device_.validator_.expect_texture(src, {}, state::resolve_src, "ResolveTexture src");
            device_.validator_.expect_texture(dst, {}, state::resolve_dst, "ResolveTexture dst");
        }
        VkImageResolve region{};
        region.srcSubresource = {s->aspect, 0, 0, 1};
        region.dstSubresource = {d->aspect, 0, 0, 1};
        region.extent = {s->desc.width, s->desc.height, 1};
        device_.fn_.vkCmdResolveImage(cmd_, s->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, d->image,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    }

    void cmd_list::begin_rendering(const rendering_desc& desc) {
        if (in_rendering_)
            fail("BeginRendering inside a rendering pass");
        VkRenderingAttachmentInfo colors[max_color_targets]{};
        VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        VkRenderingAttachmentInfo stencil{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        bool have_depth = false, have_stencil = false;
        VkFormat pass_color_format[max_color_targets]{};
        VkFormat pass_depth_format = VK_FORMAT_UNDEFINED;
        uint32_t pass_samples = 1;
        target_width_ = target_height_ = 0;
        color_count_ = desc.color_count;
        color_formats_.fill(format::unknown);
        depth_format_ = format::unknown;

        auto to_load_op = [](load_op op) {
            return op == load_op::load    ? VK_ATTACHMENT_LOAD_OP_LOAD
                   : op == load_op::clear ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                          : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        };
        auto to_store_op = [](store_op op) {
            return op == store_op::store ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
        };

        for (uint32_t i = 0; i < desc.color_count; i++) {
            const color_attachment& a = desc.colors[i];
            texture_rec* t = device_.textures_.get(a.texture);
            if (!t) {
                fail("BeginRendering: unknown colour attachment");
                continue;
            }
            if (!has(t->desc.usage, texture_usage::color_target))
                fail("BeginRendering: '" + t->name + "' lacks ColorTarget usage");
            if (device_.use_validator_)
                device_.validator_.expect_texture(a.texture, {a.mip, 1, a.layer, 1}, state::color_target,
                                                  "BeginRendering colour");
            VkRenderingAttachmentInfo& c = colors[i];
            c.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            c.imageView = device_.attachment_view(a.texture, a.mip, a.layer);
            c.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            c.loadOp = to_load_op(a.load);
            c.storeOp = to_store_op(a.store);
            std::memcpy(c.clearValue.color.float32, a.clear.data(), sizeof(float) * 4);
            if (a.resolve) {
                c.resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
                c.resolveImageView = device_.attachment_view(a.resolve, 0, 0);
                c.resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            }
            color_formats_[i] = t->desc.format;
            pass_color_format[i] = t->format;
            pass_samples = t->desc.samples;
            if (!target_width_) {
                target_width_ = mip_extent(t->desc.width, a.mip);
                target_height_ = mip_extent(t->desc.height, a.mip);
            }
        }

        if (desc.depth.texture) {
            const depth_attachment& a = desc.depth;
            texture_rec* t = device_.textures_.get(a.texture);
            if (!t) {
                fail("BeginRendering: unknown depth attachment");
            } else {
                if (!has(t->desc.usage, texture_usage::depth_target))
                    fail("BeginRendering: '" + t->name + "' lacks DepthTarget usage");
                if (device_.use_validator_)
                    device_.validator_.expect_texture(a.texture, {a.mip, 1, a.layer, 1},
                                                      a.read_only ? state::depth_read : state::depth_write,
                                                      "BeginRendering depth");
                VkImageView view = device_.attachment_view(a.texture, a.mip, a.layer);
                depth.imageView = view;
                depth.imageLayout = a.read_only ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                                                : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
                depth.loadOp = a.read_only ? VK_ATTACHMENT_LOAD_OP_LOAD : to_load_op(a.load);
                depth.storeOp = a.read_only ? VK_ATTACHMENT_STORE_OP_STORE : to_store_op(a.store);
                depth.clearValue.depthStencil = {a.clear_depth, a.clear_stencil};
                have_depth = true;
                if (has_stencil(t->desc.format)) {
                    stencil = depth;
                    stencil.clearValue.depthStencil = {a.clear_depth, a.clear_stencil};
                    have_stencil = true;
                }
                depth_format_ = t->desc.format;
                pass_depth_format = t->format;
                pass_samples = t->desc.samples;
                if (!target_width_) {
                    target_width_ = mip_extent(t->desc.width, a.mip);
                    target_height_ = mip_extent(t->desc.height, a.mip);
                }
            }
        }

        if (device_.dynamic_rendering_) {
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, {target_width_, target_height_}};
            ri.layerCount = 1;
            ri.colorAttachmentCount = desc.color_count;
            ri.pColorAttachments = colors;
            ri.pDepthAttachment = have_depth ? &depth : nullptr;
            ri.pStencilAttachment = have_stencil ? &stencil : nullptr;
            device_.fn_.vkCmdBeginRendering(cmd_, &ri);
        } else {
            // Vulkan 1.1 without VK_KHR_dynamic_rendering: the same pass as a render pass and a framebuffer
            render_pass_key key;
            key.color_count = desc.color_count;
            key.samples = pass_samples;
            VkImageView views[max_color_targets * 2 + 1];
            VkClearValue clears[max_color_targets * 2 + 1]{};
            uint32_t count = 0;
            for (uint32_t i = 0; i < desc.color_count; i++) {
                key.color_format[i] = pass_color_format[i];
                key.color_load[i] = static_cast<uint8_t>(colors[i].loadOp);
                key.color_store[i] = static_cast<uint8_t>(colors[i].storeOp);
                key.color_resolve[i] = colors[i].resolveImageView != VK_NULL_HANDLE;
                clears[count] = colors[i].clearValue;
                views[count++] = colors[i].imageView;
            }
            for (uint32_t i = 0; i < desc.color_count; i++)
                if (key.color_resolve[i])
                    views[count++] = colors[i].resolveImageView;
            if (have_depth) {
                key.depth_format = pass_depth_format;
                key.depth_stencil = have_stencil;
                key.depth_load = static_cast<uint8_t>(depth.loadOp);
                key.depth_store = static_cast<uint8_t>(depth.storeOp);
                key.depth_read_only = desc.depth.read_only;
                clears[count] = depth.clearValue;
                views[count++] = depth.imageView;
            }
            VkRenderPass pass = device_.render_pass(key);
            VkFramebuffer fb = device_.make_framebuffer(pass, views, count, target_width_, target_height_);
            if (!pass || !fb) {
                fail("BeginRendering: render pass or framebuffer creation failed");
                return;
            }
            VkRenderPassBeginInfo bi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            bi.renderPass = pass;
            bi.framebuffer = fb;
            bi.renderArea = {{0, 0}, {target_width_, target_height_}};
            bi.clearValueCount = count;
            bi.pClearValues = clears;
            device_.fn_.vkCmdBeginRenderPass(cmd_, &bi, VK_SUBPASS_CONTENTS_INLINE);
        }
        in_rendering_ = true;

        set_viewport({0, 0, static_cast<float>(target_width_), static_cast<float>(target_height_), 0.0f, 1.0f});
        set_scissor({0, 0, target_width_, target_height_});
        // pipelines are only valid inside the pass they were bound in on some drivers: force a rebind
        have_pipeline_ = false;
        dirty_sets_ = 0;
        for (uint32_t i = 0; i < max_bind_groups; i++)
            if (groups_[i])
                dirty_sets_ |= 1u << i;
        push_dirty_ = push_used_ != 0;
    }

    void cmd_list::end_rendering() {
        if (!in_rendering_) {
            fail("EndRendering without BeginRendering");
            return;
        }
        if (device_.dynamic_rendering_)
            device_.fn_.vkCmdEndRendering(cmd_);
        else
            device_.fn_.vkCmdEndRenderPass(cmd_);
        in_rendering_ = false;
    }

    void cmd_list::set_pipeline(pipeline_handle handle) {
        pipeline_rec* p = device_.pipelines_.get(handle);
        if (!p) {
            fail("SetPipeline: unknown pipeline");
            return;
        }
        if (p->compute == in_rendering_)
            fail(p->compute ? "compute pipeline set inside a rendering pass"
                            : "graphics pipeline set outside a rendering pass");
        if (have_pipeline_ && !invalid_ && pipeline_handle_ == handle)
            return;
        const VkPipeline native = device_.ready_pipeline(*p);
        if (!native) {
            // still compiling (or failed): drop draws until the next pipeline instead of stalling the frame
            have_pipeline_ = false;
            pipeline_pending_ = true;
            pipeline_handle_ = {};
            return;
        }
        pipeline_pending_ = false;
        if (!have_pipeline_ || pipeline_.layout != p->layout) {
            for (uint32_t i = 0; i < max_bind_groups; i++)
                if (groups_[i])
                    dirty_sets_ |= 1u << i;
            push_dirty_ = push_used_ != 0;
        }
        pipeline_ = *p;
        pipeline_.pipeline = native;
        pipeline_.build.reset();
        pipeline_handle_ = handle;
        have_pipeline_ = true;
        device_.fn_.vkCmdBindPipeline(cmd_, p->bind_point, native);
        counters_.pipeline_binds++;
    }

    void cmd_list::set_bind_group(uint32_t set, bind_group_handle group, std::span<const uint32_t> offsets) {
        bind_group_rec* g = device_.bind_groups_.get(group);
        if (!g || set >= max_bind_groups) {
            fail("SetBindGroup: unknown group or set index");
            return;
        }
        if (offsets.size() != g->dynamic_count)
            fail("SetBindGroup: expected " + std::to_string(g->dynamic_count) + " dynamic offsets, got " +
                 std::to_string(offsets.size()));
        if (device_.use_validator_)
            device_.validator_.expect_all(g->bound, "SetBindGroup");
        const uint32_t count = std::min<uint32_t>(static_cast<uint32_t>(offsets.size()), max_dynamic_offsets);
        if (groups_[set] == group && dynamic_counts_[set] == count &&
            std::equal(offsets.begin(), offsets.begin() + count, dynamic_offsets_[set].begin()))
            return;
        groups_[set] = group;
        dynamic_counts_[set] = count;
        std::copy_n(offsets.begin(), count, dynamic_offsets_[set].begin());
        dirty_sets_ |= 1u << set;
        counters_.group_binds++;
    }

    void cmd_list::push_constants(const void* data, uint32_t bytes, uint32_t offset) {
        if (offset + bytes > max_push_constant_bytes) {
            fail("PushConstants exceed the limit");
            return;
        }
        std::memcpy(push_.data() + offset, data, bytes);
        push_used_ = (std::max)(push_used_, offset + bytes);
        push_dirty_ = true;
    }

    void cmd_list::set_viewport(const viewport& v) {
        // negative height keeps clip space y-up like d3d, with no change to shaders or winding
        viewport_ = {v.x, v.y + v.height, v.width, -v.height, v.min_depth, v.max_depth};
        have_viewport_ = true;
        device_.fn_.vkCmdSetViewport(cmd_, 0, 1, &viewport_);
    }

    void cmd_list::set_scissor(const rect& r) {
        scissor_ = {{r.x, r.y}, {r.width, r.height}};
        have_scissor_ = true;
        device_.fn_.vkCmdSetScissor(cmd_, 0, 1, &scissor_);
    }

    void cmd_list::set_vertex_buffer(uint32_t slot, buffer_handle buffer, uint64_t offset) {
        buffer_rec* b = device_.buffers_.get(buffer);
        if (!b || slot >= max_vertex_streams) {
            fail("SetVertexBuffer: unknown buffer or slot");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(buffer, state::vertex_buffer, "SetVertexBuffer");
        if (!invalid_ && vbuffers_[slot] == b->buffer && voffsets_[slot] == offset)
            return;
        vbuffers_[slot] = b->buffer;
        voffsets_[slot] = offset;
        vertex_mask_ |= 1u << slot;
        device_.fn_.vkCmdBindVertexBuffers(cmd_, slot, 1, &b->buffer, &voffsets_[slot]);
    }

    void cmd_list::set_index_buffer(buffer_handle buffer, index_type type, uint64_t offset) {
        buffer_rec* b = device_.buffers_.get(buffer);
        if (!b) {
            fail("SetIndexBuffer: unknown buffer");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(buffer, state::index_buffer, "SetIndexBuffer");
        ibuffer_ = b->buffer;
        ioffset_ = offset;
        itype_ = type == index_type::u16 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        have_index_ = true;
        device_.fn_.vkCmdBindIndexBuffer(cmd_, ibuffer_, offset, itype_);
    }

    void cmd_list::set_stencil_reference(uint32_t value) {
        stencil_ref_ = value;
        device_.fn_.vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, value);
    }

    void cmd_list::invalidate_bound_state() {
        invalid_ = true;
        for (uint32_t i = 0; i < max_bind_groups; i++)
            if (groups_[i])
                dirty_sets_ |= 1u << i;
        push_dirty_ = push_used_ != 0;
    }

    void cmd_list::flush_bindings() {
        if (invalid_) {
            invalid_ = false;
            if (have_pipeline_)
                device_.fn_.vkCmdBindPipeline(cmd_, pipeline_.bind_point, pipeline_.pipeline);
            if (in_rendering_) {
                if (have_viewport_)
                    device_.fn_.vkCmdSetViewport(cmd_, 0, 1, &viewport_);
                if (have_scissor_)
                    device_.fn_.vkCmdSetScissor(cmd_, 0, 1, &scissor_);
                device_.fn_.vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, stencil_ref_);
                for (uint32_t i = 0; i < max_vertex_streams; i++)
                    if (vbuffers_[i])
                        device_.fn_.vkCmdBindVertexBuffers(cmd_, i, 1, &vbuffers_[i], &voffsets_[i]);
                if (ibuffer_)
                    device_.fn_.vkCmdBindIndexBuffer(cmd_, ibuffer_, ioffset_, itype_);
            }
        }
        if (!have_pipeline_)
            return;
        if (dirty_sets_) {
            pipeline_layout_rec* layout = device_.pipeline_layouts_.get(pipeline_.layout_handle);
            std::array<VkDescriptorSet, max_bind_groups> sets{};
            std::array<uint32_t, max_bind_groups * max_dynamic_offsets> offsets{};
            uint32_t first = 0, set_count = 0, offset_count = 0;
            const auto flush_run = [&] {
                if (set_count)
                    device_.fn_.vkCmdBindDescriptorSets(cmd_, pipeline_.bind_point, pipeline_.layout, first, set_count,
                                                        sets.data(), offset_count, offsets.data());
                set_count = offset_count = 0;
            };
            for (uint32_t i = 0; i < max_bind_groups; i++) {
                bind_group_rec* g = nullptr;
                if ((dirty_sets_ & (1u << i)) && (!layout || layout->sets[i])) {
                    g = device_.bind_groups_.get(groups_[i]);
                    if (!g)
                        fail("draw with a destroyed or missing bind group at set " + std::to_string(i));
                }
                if (!g) {
                    flush_run();
                    continue;
                }
                if (!set_count)
                    first = i;
                sets[set_count++] = g->set;
                std::copy_n(dynamic_offsets_[i].begin(), dynamic_counts_[i], offsets.begin() + offset_count);
                offset_count += dynamic_counts_[i];
            }
            flush_run();
            dirty_sets_ = 0;
        }
        if (push_dirty_ && pipeline_.push_bytes) {
            device_.fn_.vkCmdPushConstants(cmd_, pipeline_.layout, pipeline_.push_stages, 0, pipeline_.push_bytes,
                                           push_.data());
            push_dirty_ = false;
        }
    }

    void cmd_list::draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
                        uint32_t first_instance) {
        if (pipeline_pending_)
            return;
        if (!have_pipeline_ || !in_rendering_) {
            fail("Draw without a pipeline or outside a rendering pass");
            return;
        }
        flush_bindings();
        device_.fn_.vkCmdDraw(cmd_, vertex_count, instance_count, first_vertex, first_instance);
        counters_.draws++;
    }

    void cmd_list::draw_indexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                                int32_t vertex_offset, uint32_t first_instance) {
        if (pipeline_pending_)
            return;
        if (!have_pipeline_ || !in_rendering_ || !have_index_) {
            fail("DrawIndexed without a pipeline, index buffer or rendering pass");
            return;
        }
        flush_bindings();
        device_.fn_.vkCmdDrawIndexed(cmd_, index_count, instance_count, first_index, vertex_offset, first_instance);
        counters_.draws++;
    }

    void cmd_list::draw_indexed_indirect(buffer_handle args, uint64_t offset, uint32_t max_draws, uint32_t stride) {
        if (pipeline_pending_)
            return;
        buffer_rec* a = device_.buffers_.get(args);
        if (!a || !have_pipeline_ || !in_rendering_ || !have_index_) {
            fail("DrawIndexedIndirect: missing buffer, pipeline, index buffer or rendering pass");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(args, state::indirect_argument, "DrawIndexedIndirect args");
        flush_bindings();
        if (device_.caps_.features.multi_draw_indirect || max_draws <= 1) {
            device_.fn_.vkCmdDrawIndexedIndirect(cmd_, a->buffer, offset, max_draws, stride);
        } else {
            for (uint32_t i = 0; i < max_draws; i++)
                device_.fn_.vkCmdDrawIndexedIndirect(cmd_, a->buffer, offset + static_cast<uint64_t>(i) * stride, 1,
                                                     stride);
        }
        counters_.draws += max_draws;
        counters_.indirect++;
    }

    void cmd_list::draw_indexed_indirect_count(buffer_handle args, uint64_t offset, buffer_handle count_buffer,
                                               uint64_t count_offset, uint32_t max_draws, uint32_t stride) {
        if (pipeline_pending_)
            return;
        buffer_rec* a = device_.buffers_.get(args);
        buffer_rec* c = device_.buffers_.get(count_buffer);
        if (!a || !c || !have_pipeline_ || !in_rendering_ || !have_index_) {
            fail("DrawIndexedIndirectCount: missing buffer, pipeline, index buffer or rendering pass");
            return;
        }
        if (!device_.fn_.vkCmdDrawIndexedIndirectCount) {
            // no count support: run every slot, as the other backends do; the culling pass zeroes the unused ones
            draw_indexed_indirect(args, offset, max_draws, stride);
            return;
        }
        if (device_.use_validator_) {
            device_.validator_.expect_buffer(args, state::indirect_argument, "DrawIndexedIndirectCount args");
            device_.validator_.expect_buffer(count_buffer, state::indirect_argument, "DrawIndexedIndirectCount count");
        }
        flush_bindings();
        device_.fn_.vkCmdDrawIndexedIndirectCount(cmd_, a->buffer, offset, c->buffer, count_offset, max_draws, stride);
        counters_.indirect++;
    }

    void cmd_list::dispatch(uint32_t x, uint32_t y, uint32_t z) {
        if (!have_pipeline_ || in_rendering_) {
            fail("Dispatch without a compute pipeline or inside a rendering pass");
            return;
        }
        flush_bindings();
        device_.fn_.vkCmdDispatch(cmd_, x, y, z);
        counters_.dispatches++;
    }

    void cmd_list::dispatch_indirect(buffer_handle args, uint64_t offset) {
        buffer_rec* a = device_.buffers_.get(args);
        if (!a || !have_pipeline_ || in_rendering_) {
            fail("DispatchIndirect: missing buffer or compute pipeline");
            return;
        }
        if (device_.use_validator_)
            device_.validator_.expect_buffer(args, state::indirect_argument, "DispatchIndirect args");
        flush_bindings();
        device_.fn_.vkCmdDispatchIndirect(cmd_, a->buffer, offset);
        counters_.dispatches++;
    }

    void cmd_list::write_timestamp(query_set_handle set, uint32_t index) {
        query_rec* q = device_.queries_.get(set);
        if (!q || index >= q->count) {
            fail("WriteTimestamp: bad query set or index");
            return;
        }
        device_.write_timestamp(cmd_, q->pool, index);
    }

    void cmd_list::resolve_queries(query_set_handle set, uint32_t first, uint32_t count, buffer_handle dst,
                                   uint64_t dst_offset) {
        query_rec* q = device_.queries_.get(set);
        buffer_rec* b = device_.buffers_.get(dst);
        if (!q || !b || first + count > q->count) {
            fail("ResolveQueries: bad query set or buffer");
            return;
        }
        device_.fn_.vkCmdCopyQueryPoolResults(cmd_, q->pool, first, count, b->buffer, dst_offset, sizeof(uint64_t),
                                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        mb.srcStageMask = mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        di.memoryBarrierCount = 1;
        di.pMemoryBarriers = &mb;
        device_.pipeline_barrier(cmd_, di);
        device_.fn_.vkCmdResetQueryPool(cmd_, q->pool, first, count);
    }

    void cmd_list::begin_debug_label(const char* name, uint32_t rgba) {
        open_labels_++;
        if (!device_.inst_->debug_utils || !vkCmdBeginDebugUtilsLabelEXT)
            return;
        VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        label.pLabelName = name;
        label.color[0] = ((rgba >> 24) & 0xFF) / 255.0f;
        label.color[1] = ((rgba >> 16) & 0xFF) / 255.0f;
        label.color[2] = ((rgba >> 8) & 0xFF) / 255.0f;
        label.color[3] = (rgba & 0xFF) / 255.0f;
        vkCmdBeginDebugUtilsLabelEXT(cmd_, &label);
    }

    void cmd_list::end_debug_label() {
        if (--open_labels_ < 0) {
            fail("EndDebugLabel without BeginDebugLabel");
            open_labels_ = 0;
            return;
        }
        if (device_.inst_->debug_utils && vkCmdEndDebugUtilsLabelEXT)
            vkCmdEndDebugUtilsLabelEXT(cmd_);
    }
}
