#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/cuda_draft.hpp>
#include <covoxslam/storage/completion.hpp>

namespace covoxslam::storage
{
template<schema s, layout l> class transfer_session
{
    using traits = storage_traits<s, l>;
    struct session_state
    {
        cuda_draft<s, l> device;
        pinned_payload staging;
        transfer_endpoint host;
        bool published{};
    };
    std::shared_ptr<session_state> state_;
    operation_status available() const
    {
        if (!state_) return std::unexpected(operation_error{"transfer session is uninitialized"});
        if (state_.use_count() != 1) return std::unexpected(operation_error{"transfer session has an active ticket"});
        auto descriptor = state_->device.descriptor();
        if (!descriptor) return std::unexpected(descriptor.error());
        return {};
    }
    operation_status submit(const transfer_plan& plan, cudaMemcpyKind kind)
    {
        for (const auto& region : plan.regions) {
            const auto error = region.height == 1 ? cudaMemcpyAsync(region.destination, region.source, region.width, kind, state_->device.stream()) :
                cudaMemcpy2DAsync(region.destination, region.destination_pitch, region.source, region.source_pitch,
                    region.width, region.height, kind, state_->device.stream());
            if (error != cudaSuccess) {
                cudaStreamSynchronize(state_->device.stream());
                return cuda_storage_status(error);
            }
        }
        return {};
    }
public:
    class ticket
    {
        std::shared_ptr<session_state> state_;
        completion_ticket completion_;
        std::vector<completion_ticket> readers_;
        bool download_{};
        ticket(std::shared_ptr<session_state> state, completion_ticket completion, bool download) :
            state_(std::move(state)), completion_(std::move(completion)), download_(download) {}
        friend class transfer_session;
    public:
        ticket(const ticket&) = delete;
        ticket& operator=(const ticket&) = delete;
        ticket(ticket&&) noexcept = default;
        ticket& operator=(ticket&&) noexcept = default;
        operation_result<bool> query() const { return completion_.query(); }
        operation_status wait()
        {
            auto checked = completion_.wait();
            if (!checked) { if (state_) state_->published = false; return checked; }

            for (const auto& reader : readers_) {
                checked = reader.wait();
                if (!checked) break;
            }
            if (state_) state_->published = bool(checked);
            return checked;
        }
        template<class function> operation_status read_on(cudaStream_t stream, function&& consumer)
        {
            auto checked = completion_.depend_on(stream);
            if (!checked) return checked;
            auto descriptor = state_->device.descriptor();
            if (!descriptor) return std::unexpected(descriptor.error());
            stream_drain guard{stream, descriptor->device};
            const device_map_view<s, true> view{device_accessor<s, true>{*descriptor}};
            checked = consumer(view, stream);
            if (!checked) return checked;
            auto completed = completion_ticket::record(stream, state_);
            if (!completed) return std::unexpected(completed.error());
            guard.armed = false;
            readers_.push_back(std::move(*completed));
            return {};
        }
        operation_result<device_map_view<s, true>> view()
        {
            auto ready = query();
            if (!ready) return std::unexpected(ready.error());
            if (!*ready) return std::unexpected(operation_error{"transfer is not complete"});
            state_->published = true;
            return std::as_const(state_->device).view();
        }
        template<layout destination_layout> operation_status copy_to(draft<s, destination_layout>& destination)
        {
            if (!state_ || !download_ || destination.size() != state_->host.tiles)
                return std::unexpected(operation_error{"download destination does not match the ticket"});
            auto checked = wait();
            if (!checked) return checked;

            for (std::size_t tile = 0; tile < destination.size(); ++tile)
                for (std::size_t voxel = 0; voxel < 512; ++voxel) {
                    [&]<std::size_t... field>(std::index_sequence<field...>) {
                        (std::memcpy(&destination.template scalar<field>(tile, voxel),
                            transfer_address(state_->host, field, tile * 512 + voxel), 4), ...);
                    }(std::make_index_sequence<field_count<s>>{});
                }
            return {};
        }
    };
    operation_status initialize(std::size_t tiles, std::size_t pinned_budget = 256U * 1024U * 1024U)
    {
        if (state_) return std::unexpected(operation_error{"transfer session is already initialized"});
        constexpr auto arrays = l == layout::pool_soa ? field_count<s> : 1;
        constexpr auto stride = traits::tile_stride;
        if (tiles > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / stride / arrays)
            return std::unexpected(operation_error{"transfer staging extent overflows"});
        auto replacement = std::make_shared<session_state>();
        auto checked = replacement->staging.initialize(tiles * stride * arrays, pinned_budget);
        if (!checked) return checked;
        checked = replacement->device.initialize(tiles);
        if (!checked) return checked;
        auto descriptor = replacement->device.descriptor();
        if (!descriptor) return std::unexpected(descriptor.error());
        auto& host = replacement->host;
        host = transfer_storage(*descriptor);
        host.device = -1;

        for (std::size_t field = 0; field < field_count<s>; ++field)
            host.fields[field] = tiles ? replacement->staging.data() +
                (l == layout::pool_soa ? field * tiles * stride : traits::offsets[field]) : nullptr;

        for (std::size_t tile = 0; tile < tiles; ++tile)
            for (std::size_t voxel = 0; voxel < 512; ++voxel) {
                [&]<std::size_t... field>(std::index_sequence<field...>) {
                    (std::construct_at(reinterpret_cast<field_type<s, field>*>(transfer_address(host, field, tile * 512 + voxel))), ...);
                }(std::make_index_sequence<field_count<s>>{});
            }
        state_ = std::move(replacement);
        return {};
    }
    std::size_t pinned_bytes() const noexcept { return state_ ? state_->staging.size() : 0; }
    operation_status resize(std::size_t tiles, std::size_t pinned_budget = 256U * 1024U * 1024U)
    {
        auto checked = available();
        if (!checked) return checked;
        if (tiles == state_->host.tiles) return {};
        if (pinned_bytes() > pinned_budget) return std::unexpected(operation_error{"existing staging exceeds growth budget"});
        transfer_session replacement;
        checked = replacement.initialize(tiles, pinned_budget - pinned_bytes());
        if (!checked) return checked;
        if (state_->published) {
            transfer_measurement measured;
            checked = replacement.state_->device.copy_from(state_->device, 0, 0, std::min(tiles, state_->host.tiles) * 512,
                (1U << field_count<s>) - 1, transfer_method::bulk_same_layout, measured);
            if (!checked) return checked;
            replacement.state_->published = true;
        }
        state_.swap(replacement.state_);
        return {};
    }
    template<layout source_layout> operation_result<ticket> upload(const draft<s, source_layout>& source)
    {
        auto checked = available();
        if (!checked) return std::unexpected(checked.error());
        if (source.size() != state_->host.tiles) return std::unexpected(operation_error{"staged upload size mismatch"});
        auto descriptor = state_->device.descriptor();
        if (!descriptor) return std::unexpected(descriptor.error());
        transfer_measurement measurement;
        auto plan = plan_cuda_transfer(state_->host, 0, transfer_storage(*descriptor), 0, source.size() * 512,
            (1U << field_count<s>) - 1, transfer_method::bulk_same_layout, measurement);
        if (!plan) return std::unexpected(plan.error());

        for (std::size_t tile = 0; tile < source.size(); ++tile)
            for (std::size_t voxel = 0; voxel < 512; ++voxel) {
                [&]<std::size_t... field>(std::index_sequence<field...>) {
                    (std::memcpy(transfer_address(state_->host, field, tile * 512 + voxel), &source.template scalar<field>(tile, voxel), 4), ...);
                }(std::make_index_sequence<field_count<s>>{});
            }
        state_->published = false;
        stream_drain guard{state_->device.stream(), descriptor->device};
        checked = submit(*plan, cudaMemcpyHostToDevice);
        if (!checked) return std::unexpected(checked.error());
        auto completion = completion_ticket::record(state_->device.stream(), state_);
        if (!completion) return std::unexpected(completion.error());
        guard.armed = false;
        return ticket(state_, std::move(*completion), false);
    }
    operation_result<ticket> download()
    {
        auto checked = available();
        if (!checked) return std::unexpected(checked.error());
        if (!state_->published) return std::unexpected(operation_error{"transfer payload is not published"});
        auto descriptor = state_->device.descriptor();
        if (!descriptor) return std::unexpected(descriptor.error());
        transfer_measurement measurement;
        auto plan = plan_cuda_transfer(transfer_storage(*descriptor), 0, state_->host, 0, state_->host.tiles * 512,
            (1U << field_count<s>) - 1, transfer_method::bulk_same_layout, measurement);
        if (!plan) return std::unexpected(plan.error());
        stream_drain guard{state_->device.stream(), descriptor->device};
        checked = submit(*plan, cudaMemcpyDeviceToHost);
        if (!checked) return std::unexpected(checked.error());
        auto completion = completion_ticket::record(state_->device.stream(), state_);
        if (!completion) return std::unexpected(completion.error());
        guard.armed = false;
        return ticket(state_, std::move(*completion), true);
    }
    template<layout source_layout> operation_status upload_blocking(const draft<s, source_layout>& source)
    {
        auto submitted = upload(source);
        if (!submitted) return std::unexpected(submitted.error());
        return submitted->wait();
    }
    template<layout destination_layout> operation_status download_blocking(draft<s, destination_layout>& destination)
    {
        auto submitted = download();
        if (!submitted) return std::unexpected(submitted.error());
        return submitted->copy_to(destination);
    }
};
}
