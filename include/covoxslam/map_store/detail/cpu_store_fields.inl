    template<storage::schema s> operation_result<tile_handle> insert(block_key key)
    {
        auto found = maps_.find(key.submap);
        if (found == maps_.end() || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"insertion requires an active submap"});
        auto& entry = found->second;
        auto& source = select_field<s, es>(entry);
        if (source) {
            auto existing = source->index.find(key);
            if (existing) return existing;
        }
        auto index = source ? source->index : cpu_block_index{};
        if (!source) {
            auto initialized = index.initialize(key.submap, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, {});
            if (!initialized) return std::unexpected(initialized.error());
        }
        auto handle = index.insert(key, maximum_tiles_);
        if (!handle) return std::unexpected(handle.error());
        const auto epoch = next_epoch(source ? source->content_epoch : 0);
        const auto allocation = next_epoch(entry.record.versions.allocation), descriptor = next_epoch(entry.record.versions.descriptor);
        if (!epoch || !allocation || !descriptor) return std::unexpected(operation_error{"store epoch exhausted"});
        auto replacement = clone_field<s>(key.submap, source, index.slot_count());
        if (!replacement) return std::unexpected(replacement.error());
        (*replacement)->index = std::move(index);
        auto view = (*replacement)->payload.view();

        for (std::size_t voxel = 0; voxel < 512; ++voxel)
            storage::scatter_record(view[handle->slot].voxels[voxel], storage::value<s>{});
        (*replacement)->content_epoch = *epoch;
        if (source && source->memory) source->memory->retire();
        source = std::move(*replacement);
        entry.record.versions.allocation = *allocation;
        entry.record.versions.descriptor = *descriptor;
        (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = *epoch;
        entry.record.cpu = residency::resident;
        return handle;
    }
    operation_status erase(tile_handle handle)
    {
        auto key = resolve(handle);
        if (!key) return std::unexpected(key.error());
        auto& entry = maps_.at(handle.submap);
        if (entry.record.state != lifecycle::active) return std::unexpected(operation_error{"removal requires an active submap"});
        auto remove = [&]<storage::schema s>() -> operation_status {
            auto& source = select_field<s, es>(entry);
            auto epoch = next_epoch(source->content_epoch), allocation = next_epoch(entry.record.versions.allocation);
            auto descriptor = next_epoch(entry.record.versions.descriptor);
            if (!epoch || !allocation || !descriptor) return std::unexpected(operation_error{"store epoch exhausted"});
            auto replacement = clone_field<s>(handle.submap, source, source->payload.size());
            if (!replacement) return std::unexpected(replacement.error());
            auto checked = (*replacement)->index.erase(handle);
            if (!checked) return checked;
            auto view = (*replacement)->payload.view();

            for (std::size_t voxel = 0; voxel < 512; ++voxel)
                storage::scatter_record(view[handle.slot].voxels[voxel], storage::value<s>{});
            (*replacement)->content_epoch = *epoch;
            if (source->memory) source->memory->retire();
            source = std::move(*replacement);
            entry.record.versions.allocation = *allocation;
            entry.record.versions.descriptor = *descriptor;
            (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = *epoch;
            return {};
        };
        return handle.field == field_kind::tsdf ? remove.template operator()<storage::schema::tsdf>() : remove.template operator()<es>();
    }
    template<storage::schema s> operation_status import_field(std::uint64_t id,
        std::span<const block_key> keys, std::span<const storage::value<s>> records)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"field import requires an active submap"});
        auto& entry = found->second;
        auto& field = select_field<s, es>(entry);
        const auto allocation = next_epoch(entry.record.versions.allocation);
        const auto descriptor = next_epoch(entry.record.versions.descriptor);
        if (!allocation || !descriptor) return std::unexpected(operation_error{"store epoch exhausted"});

        if (field) return std::unexpected(operation_error{"field import cannot replace published storage"});
        if (keys.size() > maximum_tiles_ || keys.size() > std::numeric_limits<std::uint32_t>::max() ||
            records.size() / 512 != keys.size() || records.size() % 512)
            return std::unexpected(operation_error{"field payload size is invalid"});
        std::unordered_set<block_key, key_hash> unique;

        for (const auto& key : keys)
            if (key.submap != id || !unique.insert(key).second)
                return std::unexpected(operation_error{"field keys conflict"});
        auto allocated = allocate_field<s>(keys.size());
        if (!allocated) return std::unexpected(allocated.error());
        auto replacement = std::move(*allocated);
        auto checked = replacement->index.initialize(id, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, keys);
        if (!checked) return checked;
        auto view = replacement->payload.view();

        for (std::size_t i = 0; i < records.size(); ++i)
            storage::scatter_record<s>(view[i / 512].voxels[i % 512], records[i]);
        replacement->content_epoch = 1;
        field = std::move(replacement);
        entry.record.versions.allocation = *allocation;
        entry.record.versions.descriptor = *descriptor;
        entry.record.cpu = residency::resident;
        if constexpr (s == storage::schema::tsdf) entry.record.versions.tsdf = 1;
        else entry.record.versions.esdf = 1;
        return {};
    }
