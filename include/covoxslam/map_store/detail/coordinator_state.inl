    struct entry
    {
        map_store<l, es> cpu;
        map_authority authority;
        entry(std::size_t slots, std::shared_ptr<memory_budget> budget) : cpu(slots, std::move(budget)) {}
    };
    mutable std::mutex mutex_;
    residency_policy policy_;
    std::shared_ptr<memory_budget> budget_;
    std::unordered_map<std::uint64_t, std::shared_ptr<entry>> maps_;
    retirement_queue retired_;
#if defined(covoxslam_has_cuda)
    gpu_map_store<l, es> gpu_;
#endif
public:
    operation_status publish_poses(std::span<const pose_publication> poses)
    {
        std::lock_guard lock(mutex_);
        std::unordered_set<std::uint64_t> identities;

        for (const auto& update : poses) {
            auto found = maps_.find(update.submap);
            if (found == maps_.end() || !identities.insert(update.submap).second || !finite_pose(update.optimized))
                return std::unexpected(operation_error{"pose batch has invalid identities or values"});
            const auto& map = *found->second;
            const auto state = map.authority.snapshot();
            const auto* current = map.cpu.find(update.submap);
            if (!current || state.state == lifecycle::retired || state.writer != writer_kind::none ||
                current->record.versions.pose != update.expected.pose || !next_epoch(update.expected.pose) ||
                state.content[0] != update.expected.tsdf || state.content[1] != update.expected.esdf ||
                state.esdf_source != update.expected.esdf_source)
                return std::unexpected(operation_error{"pose batch source snapshot is stale"});
        }

        for (const auto& update : poses) {
            auto& map = *maps_.find(update.submap)->second;
            auto checked = map.cpu.update_pose(update.submap, update.expected.pose, update.optimized);
            if (!checked) return checked;
#if defined(covoxslam_has_cuda)
            auto resident = gpu_.maps_.find(update.submap);
            if (resident != gpu_.maps_.end()) {
                resident->second.record.optimized_pose = update.optimized;
                resident->second.record.versions.pose = update.expected.pose + 1;
            }
#endif
        }
        return {};
    }
    explicit coordinated_store(residency_policy policy = {}) : policy_(policy), budget_(std::make_shared<memory_budget>(policy.limits))
#if defined(covoxslam_has_cuda)
        , gpu_(budget_)
#endif
    {}
    operation_status configure_budget(memory_bytes limits)
    {
        std::lock_guard lock(mutex_);
        auto checked = budget_->configure(limits);
        if (checked) policy_.limits = limits;
        return checked;
    }
    budget_snapshot memory() const { return budget_->snapshot(); }
    template<storage::schema s, class function> operation_status edit(std::uint64_t id, function&& function_body,
        std::uint64_t esdf_source = 0)
    {
        std::lock_guard lock(mutex_);
        auto found = maps_.find(id);
        if (found == maps_.end()) return std::unexpected(operation_error{"mapping submap is absent"});
        auto& map = *found->second;
        auto writer = map.authority.acquire(writer_kind::cpu, s == storage::schema::tsdf ? 1U : 2U);
        if (!writer) return std::unexpected(writer.error());
        return writer->publish(esdf_source, [&](const auto&) {
            return map.cpu.template edit_field<s>(id, std::forward<function>(function_body));
        });
    }
    operation_status save(std::uint64_t id, const std::filesystem::path& destination,
        checkpoint_parameters parameters, std::size_t maximum_bytes) const
    {
        map_snapshot<l, es> selected;
        {
            std::lock_guard lock(mutex_);
            auto found = maps_.find(id);
            if (found == maps_.end()) return std::unexpected(operation_error{"checkpoint submap is absent"});
            const auto state = found->second->authority.snapshot();
            const auto fields = std::uint32_t(bool(state.content[0])) | (std::uint32_t(bool(state.content[1])) << 1);
            if (state.state != lifecycle::finalized || state.cpu != state.content)
                return std::unexpected(operation_error{"checkpoint requires complete finalized CPU storage"});
            const std::array<std::uint64_t, 1> ids{id};
            auto snapshot = found->second->cpu.snapshot(ids, fields, maximum_bytes);
            if (!snapshot) return std::unexpected(snapshot.error());
            selected = std::move(snapshot->front());
            selected.record.versions.esdf_source = state.esdf_source;
        }
        auto bytes = encode_checkpoint(selected, parameters, maximum_bytes);
        if (!bytes) return std::unexpected(bytes.error());
        return save_checkpoint_bytes(destination, *bytes);
    }
    operation_result<std::uint64_t> restore(const std::filesystem::path& source,
        checkpoint_parameters parameters, std::size_t maximum_bytes)
    {
        auto bytes = read_checkpoint_bytes(source, maximum_bytes);
        if (!bytes) return std::unexpected(bytes.error());
        std::lock_guard lock(mutex_);
        auto decoded = decode_checkpoint<l, es>(*bytes, parameters, policy_.maximum_tiles, *budget_);
        if (!decoded) return std::unexpected(decoded.error());
        const auto id = decoded->record.id;
        if (maps_.contains(id)) return std::unexpected(operation_error{"checkpoint identity already exists"});
        auto replacement = std::make_shared<entry>(policy_.maximum_tiles, budget_);
        auto checked = replacement->cpu.restore(std::move(*decoded));
        if (!checked) return std::unexpected(checked.error());
        checked = replacement->authority.restore(replacement->cpu.find(id)->record);
        if (!checked) return std::unexpected(checked.error());
        maps_.emplace(id, std::move(replacement));
        return id;
    }
    operation_result<residency_diagnostics> diagnostics() const
    {
        std::lock_guard lock(mutex_);
        residency_diagnostics result;
        result.pending_reclamation_bytes = retired_.pending_bytes();
#if defined(covoxslam_has_cuda)
        if (gpu_.pending_bytes() > std::numeric_limits<std::size_t>::max() - result.pending_reclamation_bytes)
            return std::unexpected(operation_error{"pending retirement count overflows"});
        result.pending_reclamation_bytes += gpu_.pending_bytes();
#endif

        for (const auto& [id, map] : maps_) {
            const auto state = map->authority.snapshot();
            result.active += state.state == lifecycle::active;
            result.finalized += state.state == lifecycle::finalized;
            result.retired += state.state == lifecycle::retired;
            if (state.state == lifecycle::retired) continue;

            for (unsigned field = 0; field < 2; ++field) {
                result.cpu_stale_fields += state.content[field] && state.cpu[field] != state.content[field];
                result.gpu_stale_fields += state.content[field] && state.gpu[field] != state.content[field];
            }
            auto current = map->cpu.statistics();
            if (!current) return std::unexpected(current.error());
            if (current->payload_capacity_bytes > std::numeric_limits<std::size_t>::max() - result.current_cpu_payload_bytes)
                return std::unexpected(operation_error{"current payload count overflows"});
            result.current_cpu_payload_bytes += current->payload_capacity_bytes;
        }
        return result;
    }
private:
    operation_result<immutable_snapshot<l, es>> lease_locked(std::span<const std::uint64_t> ids, std::uint32_t fields, std::size_t budget) const
    {
        std::vector<map_snapshot<l, es>> selected;

        for (auto id : ids) {
            auto found = maps_.find(id);
            if (found == maps_.end()) return std::unexpected(operation_error{"snapshot submap is absent"});
            const auto state = found->second->authority.snapshot();
            auto dependency = require_current_esdf(state, fields);
            if (!dependency) return std::unexpected(dependency.error());
            for (unsigned field = 0; field < 2; ++field)
                if ((fields & (1U << field)) && state.cpu[field] != state.content[field])
                    return std::unexpected(operation_error{"CPU snapshot requires mirror synchronization"});
            const std::array<std::uint64_t, 1> single{id};
            auto copied = found->second->cpu.snapshot(single, fields, budget);
            if (!copied) return std::unexpected(copied.error());
            copied->front().record.versions.esdf_source = found->second->authority.snapshot().esdf_source;
            selected.push_back(std::move(copied->front()));
        }
        return immutable_snapshot<l, es>::create(std::move(selected), budget);
    }
