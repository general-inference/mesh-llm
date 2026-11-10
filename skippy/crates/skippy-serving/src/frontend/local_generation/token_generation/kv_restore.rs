use super::*;

/// Prefill rate assumed for bounding a publication wait before the service
/// estimator has measured one. Deliberately fast, so the wait stays short.
const PUBLICATION_WAIT_FALLBACK_TOKENS_PER_SEC: f64 = 2_000.0;
/// Longest a lookup waits for an in-flight record, whatever it would save.
const PUBLICATION_WAIT_MAX: Duration = Duration::from_secs(10);

impl StageOpenAiBackend {
    /// An agent loop sends its next request as soon as the previous one
    /// finishes decoding, often while the recorder is still publishing that
    /// request's message-history boundary. Wait for such a record when it
    /// would restore more than the cache holds now, for no longer than
    /// re-prefilling the extra tokens is expected to take.
    pub(super) fn wait_for_in_flight_exact_state(
        &self,
        ids: &InferenceGenerationIds,
        identities: &[PrefillKvIdentity],
    ) {
        let Some(kv) = self.kv.as_ref() else {
            return;
        };
        let estimator = &self.generation_service_estimator;
        let wait = kv.wait_for_exact_state_publication(identities, |gain_tokens| {
            let prefill_secs = estimator.estimated_prefill_ms(gain_tokens).map_or_else(
                || gain_tokens as f64 / PUBLICATION_WAIT_FALLBACK_TOKENS_PER_SEC,
                |prefill_ms| prefill_ms / 1_000.0,
            );
            Duration::try_from_secs_f64(prefill_secs)
                .unwrap_or(PUBLICATION_WAIT_MAX)
                .min(PUBLICATION_WAIT_MAX)
        });
        let (outcome, pending_tokens, waited) = match wait {
            crate::kv_integration::PublicationWait::NotNeeded => return,
            crate::kv_integration::PublicationWait::Published {
                pending_tokens,
                waited,
            } => ("published", pending_tokens, waited),
            crate::kv_integration::PublicationWait::TimedOut {
                pending_tokens,
                waited,
            } => ("timed_out", pending_tokens, waited),
        };
        let mut attrs = self.openai_attrs(ids);
        attrs.insert("skippy.kv.publication_wait".to_string(), json!(outcome));
        attrs.insert(
            "skippy.kv.publication_pending_tokens".to_string(),
            json!(pending_tokens),
        );
        attrs.insert(
            "skippy.kv.publication_wait_ms".to_string(),
            json!(waited.as_secs_f64() * 1_000.0),
        );
        self.telemetry
            .emit("stage.openai_kv_publication_wait", attrs);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn lookup_and_restore_kv(
        &self,
        kv: &KvStageIntegration,
        runtime: &mut RuntimeState,
        session_id: &str,
        ids: &InferenceGenerationIds,
        prefill_tokens: &[i32],
        identities: &[PrefillKvIdentity],
        kv_identity_ms: f64,
        cache_stats: &mut GenerationCacheStats,
    ) -> (bool, usize, Option<i32>) {
        let mut restored_prefill = false;
        let mut restored_prefill_tokens = 0usize;
        let mut protected_resident_seq_id = None;
        let kv_restore_started = Instant::now();
        let kv_restore_timer = self.telemetry.is_debug_enabled().then(PhaseTimer::start);
        let estimated_cold_prefill_ms = self
            .generation_service_estimator
            .estimated_prefill_ms(prefill_tokens.len());
        match kv.restore_exact_state_with_cold_cost(
            runtime,
            session_id,
            identities,
            estimated_cold_prefill_ms,
        ) {
            Ok(Some(restored)) => {
                restored_prefill = true;
                cache_stats.status = "hit";
                cache_stats.hit_kind = Some("exact_prefix");
                let mut attrs = self.openai_attrs(ids);
                attrs.insert("skippy.kv.decision".to_string(), json!("exact_hit"));
                attrs.insert(
                    "skippy.exact_cache.hit_page_id".to_string(),
                    json!(restored.page_id),
                );
                attrs.insert(
                    "skippy.exact_cache.payload_kind".to_string(),
                    json!(restored.payload_kind.to_string()),
                );
                attrs.insert(
                    "skippy.exact_cache.restored_tokens".to_string(),
                    json!(restored.token_count),
                );
                attrs.insert(
                    "skippy.exact_cache.source".to_string(),
                    json!(restored.source),
                );
                if restored.source != "radix" {
                    attrs.insert(
                        "skippy.exact_cache.fill_ms".to_string(),
                        json!(restored.fill_ms),
                    );
                    attrs.insert(
                        "skippy.exact_cache.rewarm_enqueued".to_string(),
                        json!(restored.rewarm_enqueued),
                    );
                }
                attrs.insert(
                    "skippy.kv.matched_prefix_tokens".to_string(),
                    json!(restored.token_count),
                );
                attrs.insert(
                    "skippy.kv.suffix_prefill_tokens".to_string(),
                    json!(prefill_tokens.len().saturating_sub(restored.token_count)),
                );
                restored_prefill_tokens = restored.token_count;
                cache_stats.cached_prompt_tokens = saturating_u32(restored_prefill_tokens);
                attrs.insert(
                    "skippy.exact_cache.logical_bytes".to_string(),
                    json!(restored.logical_bytes),
                );
                attrs.insert(
                    "skippy.exact_cache.entries".to_string(),
                    json!(restored.entries),
                );
                attrs.insert(
                    "skippy.exact_cache.reconstruct_ms".to_string(),
                    json!(restored.reconstruct_ms),
                );
                attrs.insert(
                    "skippy.exact_cache.reconstruct_bytes".to_string(),
                    json!(restored.reconstruct_bytes),
                );
                attrs.insert(
                    "skippy.exact_cache.reconstruct_blocks".to_string(),
                    json!(restored.reconstruct_blocks),
                );
                attrs.insert(
                    "skippy.exact_cache.lookup_ms".to_string(),
                    json!(restored.lookup_ms),
                );
                attrs.insert(
                    "skippy.exact_cache.kv_import_ms".to_string(),
                    json!(restored.kv_import_ms),
                );
                attrs.insert(
                    "skippy.exact_cache.recurrent_import_ms".to_string(),
                    json!(restored.recurrent_import_ms),
                );
                self.telemetry
                    .emit("stage.openai_kv_lookup_decision", attrs);
            }
            Ok(None) => {
                match kv.restore_resident_prefix(runtime, session_id, identities, prefill_tokens) {
                    Ok(Some(restored)) => {
                        restored_prefill = true;
                        cache_stats.status = "hit";
                        cache_stats.hit_kind = Some("resident_prefix");
                        let mut attrs = self.openai_attrs(ids);
                        attrs.insert("skippy.kv.decision".to_string(), json!("resident_hit"));
                        attrs.insert("skippy.kv.hit_page_id".to_string(), json!(restored.page_id));
                        attrs.insert(
                            "skippy.kv.restored_tokens".to_string(),
                            json!(restored.token_count),
                        );
                        attrs.insert(
                            "skippy.kv.matched_prefix_tokens".to_string(),
                            json!(restored.token_count),
                        );
                        attrs.insert(
                            "skippy.kv.suffix_prefill_tokens".to_string(),
                            json!(prefill_tokens.len().saturating_sub(restored.token_count)),
                        );
                        restored_prefill_tokens = restored.token_count;
                        protected_resident_seq_id = Some(restored.seq_id);
                        cache_stats.cached_prompt_tokens = saturating_u32(restored_prefill_tokens);
                        attrs.insert(
                            "skippy.kv.resident_seq_id".to_string(),
                            json!(restored.seq_id),
                        );
                        self.telemetry
                            .emit("stage.openai_kv_lookup_decision", attrs);
                    }
                    Ok(None) => {
                        let mut attrs = self.openai_attrs(ids);
                        attrs.insert("skippy.kv.decision".to_string(), json!("miss"));
                        self.telemetry
                            .emit("stage.openai_kv_lookup_decision", attrs);
                    }
                    Err(error) => {
                        let mut attrs = self.openai_attrs(ids);
                        attrs.insert("skippy.kv.decision".to_string(), json!("resident_error"));
                        attrs.insert(
                            "skippy.kv.error_class".to_string(),
                            json!(crate::kv_integration::telemetry_error_class(&error)),
                        );
                        self.telemetry
                            .emit("stage.openai_kv_lookup_decision", attrs);
                    }
                }
            }
            Err(error) => {
                let mut attrs = self.openai_attrs(ids);
                attrs.insert("skippy.kv.decision".to_string(), json!("exact_error"));
                attrs.insert(
                    "skippy.kv.error_class".to_string(),
                    json!(crate::kv_integration::telemetry_error_class(&error)),
                );
                self.telemetry
                    .emit("stage.openai_kv_lookup_decision", attrs);
            }
        }
        if cache_stats.cached_prompt_tokens > 0 {
            cache_stats.restore_ms = kv_restore_started.elapsed().as_secs_f64() * 1_000.0;
        }
        if let Some(kv_restore_timer) = kv_restore_timer {
            let mut attrs = self.openai_attrs(ids);
            attrs.insert("skippy.kv.identity_ms".to_string(), json!(kv_identity_ms));
            attrs.insert(
                "skippy.kv.restore_ms".to_string(),
                json!(kv_restore_timer.elapsed_ms()),
            );
            attrs.insert(
                "skippy.kv.identity_count".to_string(),
                json!(identities.len()),
            );
            self.telemetry.emit_debug("stage.openai_kv_timing", attrs);
        }
        (
            restored_prefill,
            restored_prefill_tokens,
            protected_resident_seq_id,
        )
    }

    pub(super) fn record_and_evict_kv(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        ids: &InferenceGenerationIds,
        prefill_tokens: &[i32],
        restored_prefill: bool,
        decoded_prefill_suffix: bool,
    ) -> KvRecordResult {
        let mut resident_recorded_pages = 0usize;
        if let (true, Some(kv)) = (
            !restored_prefill || decoded_prefill_suffix,
            self.kv.as_ref(),
        ) {
            let base = self.local_kv_message_base(session_id, ids);
            let exact_identity = kv.prefill_identity(&self.config, &base, 0, prefill_tokens);
            let cold_prefill_cost = self
                .generation_service_estimator
                .estimated_prefill_ms(prefill_tokens.len());
            let l3_cost = kv.l3_benefit_cost(cold_prefill_cost);
            if let Ok(Ok(record)) = kv.record_exact_state_with_cost_and_durability(
                runtime,
                session_id,
                &exact_identity,
                crate::kv_integration::CaptureAdmission::BestEffort,
                l3_cost,
                kv.full_exact_state_writes_through_l3(prefill_tokens.len()),
            ) {
                resident_recorded_pages = resident_recorded_pages.saturating_add(1);
                let mut attrs = self.openai_attrs(ids);
                attrs.insert(
                    "skippy.exact_cache.recorded_page_id".to_string(),
                    json!(record.page_id),
                );
                attrs.insert(
                    "skippy.exact_cache.payload_kind".to_string(),
                    json!(record.payload_kind.to_string()),
                );
                attrs.insert(
                    "skippy.exact_cache.recorded_tokens".to_string(),
                    json!(record.token_count),
                );
                attrs.insert(
                    "skippy.exact_cache.stored".to_string(),
                    json!(record.stored),
                );
                attrs.insert(
                    "skippy.exact_cache.logical_bytes".to_string(),
                    json!(record.logical_bytes),
                );
                attrs.insert(
                    "skippy.exact_cache.physical_bytes".to_string(),
                    json!(record.physical_bytes),
                );
                attrs.insert(
                    "skippy.exact_cache.entries".to_string(),
                    json!(record.entries),
                );
                attrs.insert(
                    "skippy.exact_cache.evicted_entries".to_string(),
                    json!(record.evicted_entries),
                );
                attrs.insert(
                    "skippy.exact_cache.evicted_logical_bytes".to_string(),
                    json!(record.evicted_logical_bytes),
                );
                attrs.insert(
                    "skippy.exact_cache.dedupe_hash_ms".to_string(),
                    json!(record.dedupe.hash_ms),
                );
                attrs.insert(
                    "skippy.exact_cache.dedupe_block_count".to_string(),
                    json!(record.dedupe.block_count),
                );
                attrs.insert(
                    "skippy.exact_cache.dedupe_new_block_count".to_string(),
                    json!(record.dedupe.new_block_count),
                );
                attrs.insert(
                    "skippy.exact_cache.dedupe_reused_block_count".to_string(),
                    json!(record.dedupe.reused_block_count),
                );
                self.telemetry
                    .emit("stage.openai_kv_record_decision", attrs);
            }
            for identity in kv.record_identities(&self.config, &base, 0, prefill_tokens) {
                if let Ok(Some(record)) =
                    kv.record_resident_prefix(runtime, session_id, &identity, prefill_tokens)
                {
                    resident_recorded_pages = resident_recorded_pages.saturating_add(1);
                    let mut attrs = self.openai_attrs(ids);
                    attrs.insert(
                        "skippy.kv.recorded_page_id".to_string(),
                        json!(record.page_id),
                    );
                    attrs.insert(
                        "skippy.kv.recorded_tokens".to_string(),
                        json!(record.token_count),
                    );
                    attrs.insert(
                        "skippy.kv.resident_seq_id".to_string(),
                        json!(record.seq_id),
                    );
                    attrs.insert(
                        "skippy.kv.resident_entries".to_string(),
                        json!(record.entries),
                    );
                    attrs.insert(
                        "skippy.kv.evicted_entries".to_string(),
                        json!(record.evicted_entries),
                    );
                    self.telemetry
                        .emit("stage.openai_kv_record_decision", attrs);
                }
            }
        }
        // Proactive eviction: after prefill recording, evict enough
        // LRU resident-prefix entries to free one native decode batch
        // for grammar-triggered retries during the decode loop.
        let mut proactive_eviction_status = "disabled";
        let mut proactive_eviction_error_kind_attr = None;
        let mut proactive_eviction_target_tokens = 0_u64;
        let mut proactive_evicted_entries = 0_usize;
        let mut proactive_evicted_tokens = 0_u64;
        let mut proactive_eviction_error = None;
        if let Some(kv) = self.kv.as_ref() {
            match kv.evict_resident_prefix_for_decode_batch(runtime, session_id) {
                Ok(eviction) => {
                    proactive_eviction_status = if eviction.evicted_entries > 0 {
                        "evicted"
                    } else {
                        "noop"
                    };
                    proactive_eviction_target_tokens = eviction.target_tokens;
                    proactive_evicted_entries = eviction.evicted_entries;
                    proactive_evicted_tokens = eviction.evicted_tokens;
                }
                Err(error) => {
                    proactive_eviction_status = "error";
                    proactive_eviction_error_kind_attr =
                        Some(proactive_eviction_error_kind(&error));
                    proactive_eviction_error =
                        Some(error.context("evict resident-prefix KV before local OpenAI decode"));
                }
            }
        }
        KvRecordResult {
            resident_recorded_pages,
            resident_enqueued_checkpoints: 0,
            proactive_eviction_status,
            proactive_eviction_error_kind: proactive_eviction_error_kind_attr,
            proactive_eviction_target_tokens,
            proactive_evicted_entries,
            proactive_evicted_tokens,
            proactive_eviction_error,
        }
    }
}
