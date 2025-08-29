    std::unordered_map<std::uint64_t, map_entry<l, es>> maps_;
    std::size_t maximum_tiles_{1000000};
    std::shared_ptr<memory_budget> budget_;
    template<storage::schema s> operation_result<std::shared_ptr<field_pool<s, l>>> allocate_field(std::size_t count)
    {
        constexpr auto stride = storage::payload_tile_bytes<s, l>;
        if (count > maximum_tiles_ || count > std::numeric_limits<std::size_t>::max() / stride)
            return std::unexpected(operation_error{"field allocation exceeds capacity"});
        auto claim = budget_->reserve({count * stride, 0, 0});
        if (!claim) return std::unexpected(claim.error());
        auto field = std::make_shared<field_pool<s, l>>();
        field->memory = std::move(*claim);
        auto checked = field->payload.initialize(count);
        if (!checked) return std::unexpected(checked.error());
        return field;
    }
    template<storage::schema s> operation_result<std::shared_ptr<field_pool<s, l>>> clone_field(
        std::uint64_t id, const std::shared_ptr<const field_pool<s, l>>& source, std::size_t count)
    {
        auto allocated = allocate_field<s>(count);
        if (!allocated) return std::unexpected(allocated.error());
        auto target = std::move(*allocated);
        if (source) {
            target->index = source->index;
            auto in = source->payload.view();
            auto out = target->payload.view();

            for (std::size_t tile = 0; tile < std::min(count, in.size()); ++tile)
                for (std::size_t voxel = 0; voxel < 512; ++voxel)
                    storage::scatter_record(out[tile].voxels[voxel], storage::gather_record(in[tile].voxels[voxel]));
        }
        else {
            auto checked = target->index.initialize(id, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, {});
            if (!checked) return std::unexpected(checked.error());
        }
        return target;
    }
public:
    explicit map_store(std::size_t maximum_tiles = 1000000, std::shared_ptr<memory_budget> budget = {})
        : maximum_tiles_(maximum_tiles), budget_(budget ? std::move(budget) : std::make_shared<memory_budget>(residency_policy{}.limits)) {}
    budget_snapshot memory() const { return budget_->snapshot(); }
    template<storage::schema s> operation_status initialize_topology(std::uint64_t id,
        std::span<const block_key> keys, std::uint64_t epoch, std::uint64_t esdf_source = 0)
    {
        auto found = maps_.find(id);

        if (found == maps_.end() || found->second.record.state != lifecycle::active || !epoch ||
            select_field<s, es>(found->second))
            return std::unexpected(operation_error{"topology requires an empty active field"});
        cpu_block_index index;
        auto checked = index.initialize(id, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, keys);
        if (!checked) return checked;
        auto field = allocate_field<s>(keys.size());

        if (!field) return std::unexpected(field.error());
        (*field)->index = std::move(index);
         (*field)->content_epoch = epoch;
        return install_mirror<s>(id, std::move(*field), esdf_source);
    }
    template<storage::schema s> operation_status install_mirror(std::uint64_t id,
        std::shared_ptr<const field_pool<s, l>> replacement, std::uint64_t esdf_source)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state == lifecycle::retired || !replacement)
            return std::unexpected(operation_error{"mirror destination is unavailable"});
        auto& entry = found->second;
        auto& field = select_field<s, es>(entry);
        const auto kind = s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf;
        auto generation = next_epoch(entry.record.versions.allocation), descriptor = next_epoch(entry.record.versions.descriptor);
        if (!generation || !descriptor || !budget_->owns(replacement->memory) || replacement->index.submap() != id ||
            replacement->index.field() != kind || replacement->payload.size() > maximum_tiles_ ||
            replacement->payload.size() != replacement->index.slot_count() || !replacement->content_epoch ||
            (field && replacement->content_epoch < field->content_epoch))
            return std::unexpected(operation_error{"mirror metadata or admission is invalid"});
        if (field && field->memory) field->memory->retire();
        field = std::move(replacement);
        entry.record.versions.allocation = *generation; entry.record.versions.descriptor = *descriptor;
        (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = field->content_epoch;
        if constexpr (s != storage::schema::tsdf) entry.record.versions.esdf_source = esdf_source;
        entry.record.cpu = residency::resident;
        return {};
    }
    template<storage::schema s, class function> operation_status edit_field(std::uint64_t id, function&& edit)
    {
        auto found = maps_.find(id);
        if (found == maps_.end() || found->second.record.state != lifecycle::active)
            return std::unexpected(operation_error{"field editing requires an active submap"});
        auto& entry = found->second;
        auto& source = select_field<s, es>(entry);
        auto epoch = next_epoch(source ? source->content_epoch : 0);
        auto allocation = next_epoch(entry.record.versions.allocation), descriptor = next_epoch(entry.record.versions.descriptor);
        if (!epoch || !allocation || !descriptor) return std::unexpected(operation_error{"field editing epoch exhausted"});
        auto replacement = clone_field<s>(id, source, source ? source->payload.size() : 0);
        if (!replacement) return std::unexpected(replacement.error());
        field_editor<s, l> editor(**replacement, *budget_, maximum_tiles_);
        auto checked = edit(editor);
        if (!checked) return checked;
        checked = editor.status();
        if (!checked) return checked;
        (*replacement)->content_epoch = *epoch;
        if (source && source->memory) source->memory->retire();
        source = std::move(*replacement);
        entry.record.versions.allocation = *allocation; entry.record.versions.descriptor = *descriptor;
        (s == storage::schema::tsdf ? entry.record.versions.tsdf : entry.record.versions.esdf) = *epoch;
        entry.record.cpu = residency::resident;
        return {};
    }
    operation_status restore(map_snapshot<l, es> snapshot)
    {
        auto& record = snapshot.record;
        if (maps_.contains(record.id) || record.state != lifecycle::finalized || record.writer != writer_kind::none ||
            !finite_pose(record.initial_pose) || !finite_pose(record.optimized_pose) || !record.versions.pose ||
            (!snapshot.tsdf && !snapshot.esdf) || bool(record.versions.tsdf) != bool(snapshot.tsdf) ||
            bool(record.versions.esdf) != bool(snapshot.esdf) ||
            (snapshot.esdf && (!snapshot.tsdf || record.versions.esdf_source != record.versions.tsdf)))
            return std::unexpected(operation_error{"restored submap identity or metadata conflicts"});
        auto admit = [&]<storage::schema s>(const std::shared_ptr<const field_pool<s, l>>& field) -> operation_status {
            if (!field) return {};
            const auto version = s == storage::schema::tsdf ? record.versions.tsdf : record.versions.esdf;
            const auto kind = s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf;
            if (!budget_->owns(field->memory) || !field->index.initialized() || field->index.submap() != record.id ||
                field->index.field() != kind || field->payload.size() != field->index.slot_count() ||
                field->payload.size() > maximum_tiles_ || field->content_epoch != version)
                return std::unexpected(operation_error{"restored field identity or admission differs"});
            return {};
        };
        auto checked = admit(snapshot.tsdf);
        if (!checked) return checked;
        checked = admit(snapshot.esdf);
        if (!checked) return checked;
        auto allocation = next_epoch(record.versions.allocation), descriptor = next_epoch(record.versions.descriptor);
        if (!allocation || !descriptor) return std::unexpected(operation_error{"restored allocation epochs are exhausted"});
        record.versions.allocation = *allocation; record.versions.descriptor = *descriptor;
        record.cpu = residency::resident; record.gpu = residency::absent;
        maps_.emplace(record.id, map_entry<l, es>{record, std::move(snapshot.tsdf), std::move(snapshot.esdf)});
        return {};
    }
    std::size_t size() const noexcept { return maps_.size(); }
    const map_entry<l, es>* find(std::uint64_t id) const noexcept
    {
        const auto found = maps_.find(id);
        return found == maps_.end() ? nullptr : &found->second;
    }
    operation_result<store_statistics> statistics() const
    {
        store_statistics result;
        result.submaps = maps_.size();

        for (const auto& [id, map] : maps_) {
            result.retired += map.record.state == lifecycle::retired;
            auto count = [&](const auto& field) -> operation_status {
                if (!field) return {};
                using pool = std::remove_cvref_t<decltype(*field)>;
                constexpr auto selected = std::is_same_v<pool, field_pool<storage::schema::tsdf, l>> ? storage::schema::tsdf : es;
                constexpr auto bytes_per_tile = storage::payload_tile_bytes<selected, l>;
                const auto descriptor = field->payload.view();
                const auto bytes = field->payload.capacity() * bytes_per_tile;
                if (bytes > std::numeric_limits<std::size_t>::max() - result.payload_capacity_bytes)
                    return std::unexpected(operation_error{"store statistics overflow"});
                result.payload_capacity_bytes += bytes;
                result.live_tiles += field->index.size();
                result.allocated_slots += descriptor.size();
                result.index_entries += field->index.size();
                return {};
            };
            auto checked = count(map.tsdf);
            if (!checked) return std::unexpected(checked.error());
            checked = count(map.esdf);
            if (!checked) return std::unexpected(checked.error());
        }
        return result;
    }
