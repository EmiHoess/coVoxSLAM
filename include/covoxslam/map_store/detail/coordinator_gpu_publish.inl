    operation_status import_device_tsdf(std::uint64_t id, pose_record pose,
        std::span<const block_key> keys, const storage::device_buffer<storage::schema::tsdf, l>& source)
    {
        std::lock_guard lock(mutex_);

        if (maps_.contains(id) || gpu_.maps_.contains(id) || keys.size() > policy_.maximum_tiles ||
            keys.size() > source.size() || !finite_pose(pose))
            return std::unexpected(operation_error{"resident submap admission is invalid"});
        auto descriptor = source.descriptor();
        if (!descriptor) return std::unexpected(descriptor.error());
        auto replacement = std::make_shared<entry>(policy_.maximum_tiles, budget_);
        auto checked = replacement->cpu.create(id, pose);
        if (!checked) return checked;
        checked = replacement->cpu.template initialize_topology<storage::schema::tsdf>(id, keys, 1);
        if (!checked) return checked;
        checked = replacement->cpu.template initialize_topology<es>(id, keys, 1, 1);
        if (!checked) return checked;
        const auto* metadata = replacement->cpu.find(id);
        auto tsdf = gpu_.template upload<storage::schema::tsdf>(metadata->tsdf,
            metadata->record.versions.allocation, false, &source);
        if (!tsdf) return std::unexpected(tsdf.error());
        auto esdf = gpu_.template upload<es>(metadata->esdf, metadata->record.versions.allocation, false);

        if (!esdf) return std::unexpected(esdf.error());
        auto writer = replacement->authority.acquire(writer_kind::gpu, 3);
        if (!writer) return std::unexpected(writer.error());
        checked = writer->publish(1, [](const auto&) -> operation_status { return {}; });
        if (!checked) return checked;
        resident_entry<l, es> resident{metadata->record, std::move(*tsdf), std::move(*esdf)};
         resident.record.cpu = residency::absent;
        resident.record.gpu = residency::resident;
        maps_.emplace(id, std::move(replacement));
        gpu_.maps_.emplace(id, std::move(resident));
        return {};
    }
    operation_result<std::vector<resident_entry<l, es>>> lease_resident(
        std::span<const std::uint64_t> ids, std::uint32_t fields, std::size_t budget) const
    {
        std::lock_guard lock(mutex_);
        std::vector<resident_entry<l, es>> result;
        std::unordered_set<std::uint64_t> unique;
        if (!fields || (fields & ~3U)) return std::unexpected(operation_error{"invalid resident field selection"});

        for (auto id : ids) {
            auto found = maps_.find(id);
            if (found == maps_.end() || !unique.insert(id).second)
                return std::unexpected(operation_error{"resident snapshot identity is invalid"});
            const auto state = found->second->authority.snapshot();
            auto valid = require_current_esdf(state, fields);
            if (!valid) return std::unexpected(valid.error());
            auto resident = gpu_.maps_.find(id);
            if (state.state != lifecycle::finalized || state.writer != writer_kind::none || resident == gpu_.maps_.end())
                return std::unexpected(operation_error{"resident PGO requires finalized maps"});
            auto selected = resident->second;
            selected.record = found->second->cpu.find(id)->record;
            selected.record.versions.tsdf = state.content[0];
            selected.record.versions.esdf = state.content[1];
            selected.record.versions.esdf_source = state.esdf_source;
            const auto admit = [&]<storage::schema s>(const std::shared_ptr<const gpu_field<s, l>>& field) -> operation_status {
                constexpr auto ordinal = s == storage::schema::tsdf ? 0 : 1;
                if (!(fields & (1U << ordinal))) return {};
                if (!field || !state.content[ordinal] || state.gpu[ordinal] != state.content[ordinal] ||
                    field->content_epoch != state.content[ordinal])
                    return std::unexpected(operation_error{"resident snapshot requires current GPU fields"});
                auto bytes = gpu_.retained_field(field);
                if (!bytes) return std::unexpected(bytes.error());
                if (bytes->bytes > budget) return std::unexpected(operation_error{"resident snapshot budget exceeded"});
                budget -= bytes->bytes;
                return {};
            };
            valid = admit(selected.tsdf);
            if (!valid) return std::unexpected(valid.error());
            valid = admit(selected.esdf);
            if (!valid) return std::unexpected(valid.error());
            if (!(fields & 1U)) selected.tsdf.reset();
            if (!(fields & 2U)) selected.esdf.reset();
            result.push_back(std::move(selected));
        }
        return result;
    }
    template<class function> operation_status derive_gpu(std::uint64_t id, function&& consumer, bool preserve_values = true)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"ESDF derivation submap is absent"});
        auto& map = *found->second;
        const auto state = map.authority.snapshot();
        if (!state.content[0] || state.gpu[0] != state.content[0])
            return std::unexpected(operation_error{"ESDF derivation requires the current GPU TSDF"});
        auto source = gpu_.template field<storage::schema::tsdf>(id);
        if (!source) return std::unexpected(source.error());
        auto read_view = (*source)->payload.view();
        auto read_index = (*source)->index.view();
        if (!read_view) return std::unexpected(read_view.error());
        if (!read_index) return std::unexpected(read_index.error());
        auto metadata = (*source)->source_identity.lock();
        if (!metadata) return std::unexpected(operation_error{"ESDF source topology is unavailable"});
        std::vector<block_key> keys;
        keys.reserve(metadata->index.size());

        for (const auto& slot : metadata->index.records())
            if (slot.occupied) keys.push_back(slot.key);
        std::sort(keys.begin(), keys.end(), [](auto a, auto b) {
            return a.z < b.z || (a.z == b.z && (a.y < b.y || (a.y == b.y && a.x < b.x)));
        });
        auto writer = map.authority.acquire(writer_kind::gpu, 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(state.content[0], [&](const auto& versions) {
            return gpu_.template edit<es>(id, versions[1], state.content[0], [&](auto target, auto index, auto stream) {
                return consumer(*read_view, *read_index, target, index, stream, std::span<const block_key>(keys), *budget_);
            }, preserve_values);
        });
    }
