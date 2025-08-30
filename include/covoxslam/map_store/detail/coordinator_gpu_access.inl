    template<storage::schema s, class function> operation_status edit_gpu(std::uint64_t id, function&& consumer,
        std::uint64_t esdf_source = 0)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"GPU writer submap is absent"});
        auto writer = found->second->authority.acquire(writer_kind::gpu, s == storage::schema::tsdf ? 1U : 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(esdf_source, [&](const auto& versions) {
            return gpu_.template edit<s>(id, versions[s == storage::schema::tsdf ? 0 : 1], esdf_source,
                std::forward<function>(consumer));
        });
    }
private:
    template<storage::schema s> operation_status synchronize_cpu_locked(std::uint64_t id)
    {
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"CPU mirror submap is absent"});
        auto& map = *found->second;
        auto ticket = map.authority.transfer(writer_kind::gpu, writer_kind::cpu, s == storage::schema::tsdf ? 1U : 2U);
        if (!ticket) return std::unexpected(ticket.error());
        const auto source = map.authority.snapshot().esdf_source;
        auto resident = gpu_.template field<s>(id);
        if (!resident) return std::unexpected(resident.error());
        auto copied = gpu_.template download<s>(id);
        if (!copied) return std::unexpected(copied.error());
        return ticket->acknowledge([&]() -> operation_status {
            auto checked = map.cpu.template install_mirror<s>(id, *copied, source);
            if (checked) (*resident)->source_identity = *copied;
            return checked;
        });
    }
public:
    template<storage::schema s> operation_status synchronize_cpu(std::uint64_t id)
    {
        std::lock_guard lock(mutex_);
        return synchronize_cpu_locked<s>(id);
    }
    operation_result<immutable_snapshot<l, es>> lease_readback(std::span<const std::uint64_t> ids,
        std::uint32_t fields, std::size_t payload_budget, std::size_t readback_budget)
    {
        std::lock_guard lock(mutex_);
        if (ids.empty() || !fields || (fields & ~3U)) return std::unexpected(operation_error{"invalid geometry selection"});
        std::unordered_set<std::uint64_t> unique;
        std::size_t download_bytes{}, payload_bytes{};

        for (auto id : ids) {
            auto found = maps_.find(id);
            if (found == maps_.end() || !unique.insert(id).second)
                return std::unexpected(operation_error{"geometry submap is absent or duplicated"});
            const auto state = found->second->authority.snapshot();
            auto checked = require_current_esdf(state, fields);
            if (!checked) return std::unexpected(checked.error());
            if (state.state == lifecycle::retired || state.writer != writer_kind::none)
                return std::unexpected(operation_error{"geometry submap is unavailable"});
            const auto* cpu = found->second->cpu.find(id);
            auto admit = [&]<storage::schema s>(unsigned field) -> operation_status {
                if (!(fields & (1U << field))) return {};
                if (!state.content[field]) return std::unexpected(operation_error{"geometry field is absent"});
                constexpr auto stride = storage::payload_tile_bytes<s, l>;
                const auto& host = select_field<s, es>(*cpu);
                std::size_t capacity = host ? host->payload.capacity() : 0;
                if (state.cpu[field] != state.content[field]) {
                    if (state.gpu[field] != state.content[field]) return std::unexpected(operation_error{"geometry has no current replica"});
                    auto resident = gpu_.template field<s>(id);
                    if (!resident) return std::unexpected(resident.error());
                    capacity = (*resident)->payload.size();
                    constexpr auto record_bytes = storage::canonical_tile_bytes<s>;
                    if (capacity > (readback_budget - download_bytes) / record_bytes)
                        return std::unexpected(operation_error{"geometry readback limit exceeded"});
                    download_bytes += capacity * record_bytes;
                }
                if (capacity > (payload_budget - payload_bytes) / stride)
                    return std::unexpected(operation_error{"geometry snapshot limit exceeded"});
                payload_bytes += capacity * stride;
                return {};
            };
            checked = admit.template operator()<storage::schema::tsdf>(0);
            if (!checked) return std::unexpected(checked.error());
            checked = admit.template operator()<es>(1);
            if (!checked) return std::unexpected(checked.error());
        }

        for (auto id : ids) {
            const auto state = maps_.find(id)->second->authority.snapshot();
            if ((fields & 1U) && state.cpu[0] != state.content[0]) {
                auto checked = synchronize_cpu_locked<storage::schema::tsdf>(id);
                if (!checked) return std::unexpected(checked.error());
            }
            if ((fields & 2U) && state.cpu[1] != state.content[1]) {
                auto checked = synchronize_cpu_locked<es>(id);
                if (!checked) return std::unexpected(checked.error());
            }
        }
        return lease_locked(ids, fields, payload_budget);
    }
    operation_status evict_gpu(std::uint64_t id)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        const auto state = map.authority.snapshot();
        return map.authority.evict(writer_kind::gpu, [&] { return gpu_.evict(id, state); });
    }
    operation_status synchronize_gpu(std::uint64_t id, std::uint32_t fields, std::size_t budget)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        auto& map = *found->second;
        auto ticket = map.authority.transfer(writer_kind::cpu, writer_kind::gpu, fields);
        if (!ticket) return std::unexpected(ticket.error());
        const std::array<std::uint64_t, 1> ids{id};
        auto source = map.cpu.snapshot(ids, fields, budget);
        if (!source) return std::unexpected(source.error());
        source->front().record.versions.esdf_source = map.authority.snapshot().esdf_source;
        return ticket->acknowledge([&] { return gpu_.publish(source->front(), fields); });
    }
    template<storage::schema s, class function> operation_result<storage::completion_ticket> read_gpu(
        std::uint64_t id, cudaStream_t stream, function&& consumer) const
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"submap identity is absent"});
        const auto state = found->second->authority.snapshot();
        const unsigned field = s == storage::schema::tsdf ? 0 : 1;
        auto dependency = require_current_esdf(state, 1U << field);
        if (!dependency) return std::unexpected(dependency.error());
        if (!state.content[field] || state.gpu[field] != state.content[field])
            return std::unexpected(operation_error{"GPU mirror is not current"});
        return gpu_.template read_on<s>(id, stream, std::forward<function>(consumer));
    }
