    operation_result<std::vector<map_snapshot<l, es>>> snapshot(std::span<const std::uint64_t> ids,
        std::uint32_t fields, std::size_t maximum_payload_bytes) const
    {
        if (!fields || (fields & ~3U)) return std::unexpected(operation_error{"snapshot field mask is invalid"});
        std::unordered_set<std::uint64_t> unique;
        std::vector<map_snapshot<l, es>> result;
        std::size_t remaining = maximum_payload_bytes;

        for (auto id : ids) {
            const auto* entry = find(id);
            if (!entry || entry->record.state == lifecycle::retired || !unique.insert(id).second)
                return std::unexpected(operation_error{"snapshot submap is absent retired or duplicated"});
            map_snapshot<l, es> copy;
            copy.record = entry->record;
            const auto bind = [&]<storage::schema s>(auto& target) -> operation_status {
                auto selected = field<s>(id);
                if (!selected) return std::unexpected(selected.error());
                const auto capacity = (*selected)->payload.capacity();
                constexpr auto bytes_per_tile = storage::payload_tile_bytes<s, l>;
                if (capacity > remaining / bytes_per_tile)
                    return std::unexpected(operation_error{"snapshot payload budget is exhausted"});
                remaining -= capacity * bytes_per_tile;
                target = *selected;
                return {};
            };
            if (fields & 1U) {
                auto checked = bind.template operator()<storage::schema::tsdf>(copy.tsdf);
                if (!checked) return std::unexpected(checked.error());
            }
            if (fields & 2U) {
                auto checked = bind.template operator()<es>(copy.esdf);
                if (!checked) return std::unexpected(checked.error());
            }
            result.push_back(std::move(copy));
        }
        return result;
    }
    operation_status create(std::uint64_t id, pose_record pose = {})
    {
        if (maps_.contains(id)) return std::unexpected(operation_error{"submap identity already exists"});
        if (!finite_pose(pose)) return std::unexpected(operation_error{"submap pose is not finite"});
        map_entry<l, es> entry;
        entry.record.id = id;
        entry.record.initial_pose = entry.record.optimized_pose = pose;
        maps_.emplace(id, std::move(entry));
        return {};
    }
    operation_status update_pose(std::uint64_t id, std::uint64_t expected, pose_record pose)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state == lifecycle::retired || !finite_pose(pose))
            return std::unexpected(operation_error{"pose publication has invalid identity or value"});
        auto& record = found->second.record;
        if (record.versions.pose != expected)
            return std::unexpected(operation_error{"pose publication is stale"});
        auto next = next_epoch(expected);
        if (!next) return std::unexpected(next.error());
        record.optimized_pose = pose;
        record.versions.pose = *next;
        return {};
    }
    operation_status retire(std::uint64_t id)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state == lifecycle::retired)
            return std::unexpected(operation_error{"submap is already absent or retired"});
        auto& entry = found->second;
        if (entry.record.writer != writer_kind::none)
            return std::unexpected(operation_error{"cannot retire a submap with a writer"});
        auto next = next_epoch(entry.record.versions.descriptor);
        if (!next) return std::unexpected(next.error());
        entry.record.state = lifecycle::retired;
        entry.record.versions.descriptor = *next;
        if (entry.tsdf && entry.tsdf->memory) entry.tsdf->memory->retire();
        if (entry.esdf && entry.esdf->memory) entry.esdf->memory->retire();
        entry.tsdf.reset();
        entry.esdf.reset();
        entry.record.cpu = entry.record.gpu = residency::absent;
        return {};
    }
    operation_status seal(std::uint64_t id, std::uint32_t fields, std::uint64_t esdf_source)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || !fields || (fields & ~3U) || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"invalid submap seal"});
        auto& entry = found->second;
        if (((fields & 1U) && !entry.tsdf) || ((fields & 2U) && (!entry.esdf || !entry.tsdf ||
            esdf_source != entry.record.versions.tsdf)) || entry.record.writer != writer_kind::none)
            return std::unexpected(operation_error{"submap seal requires current fields"});
        entry.record.versions.esdf_source = esdf_source;
        entry.record.state = lifecycle::finalized;
        return {};
    }
    template<storage::schema s> operation_result<tile_handle> lookup(block_key key) const
    {
        auto selected = field<s>(key.submap);
        if (!selected) return std::unexpected(selected.error());
        return (*selected)->index.find(key);
    }
    template<storage::schema s> operation_result<std::shared_ptr<const field_pool<s, l>>> field(std::uint64_t id) const
    {
        const auto* entry = find(id);
        if (!entry || entry->record.state == lifecycle::retired)
            return std::unexpected(operation_error{"submap identity is unavailable"});
        auto result = select_field<s, es>(*entry);
        if (!result) return std::unexpected(operation_error{"submap field is absent"});
        return result;
    }
    operation_result<block_key> resolve(tile_handle handle) const
    {
        const auto resolve_field = [&]<storage::schema s>() -> operation_result<block_key> {
            auto selected = field<s>(handle.submap);
            if (!selected) return std::unexpected(selected.error());
            return (*selected)->index.resolve(handle);
        };
        switch (handle.field) {
        case field_kind::tsdf: return resolve_field.template operator()<storage::schema::tsdf>();
        case field_kind::esdf: return resolve_field.template operator()<es>();
        default: return std::unexpected(operation_error{"tile field is invalid"});
        }
    }
