//! Collects ready draft lanes before submitting their noise blocks.
use super::*;
use crate::runtime_state::draft_batch::RuntimeDraftRequest;
use skippy_runtime::DFlashProposal;

#[cfg(test)]
mod tests;

type DraftResult = InferenceResult<SchedulerRuntimeOutcome<Option<DFlashProposal>>>;

pub(super) struct DraftOperation {
    request: RuntimeDraftRequest,
    cancellation: Option<skippy_inference_api::CancellationToken>,
    enqueued_at: Instant,
    reply: std_mpsc::SyncSender<DraftResult>,
}

impl DraftOperation {
    fn complete(
        self,
        result: InferenceResult<Option<DFlashProposal>>,
        lock_started: Instant,
        hold_started: Instant,
    ) {
        let outcome = result.map(|value| SchedulerRuntimeOutcome {
            value,
            queue_wait_ms: lock_started.duration_since(self.enqueued_at).as_secs_f64() * 1000.0,
            runtime_lock_wait_ms: hold_started.duration_since(lock_started).as_secs_f64() * 1000.0,
            runtime_lock_hold_ms: hold_started.elapsed().as_secs_f64() * 1000.0,
        });
        let _ = self.reply.send(outcome);
    }
}

pub(crate) fn enabled() -> bool {
    enabled_by(env::var("SKIPPY_SHARED_DRAFT").ok().as_deref())
}

/// Sharing is capability checked at submission; zero retains independent work.
fn enabled_by(setting: Option<&str>) -> bool {
    setting != Some("0")
}

impl IterationScheduler {
    pub(crate) fn execute_draft_timed(
        &self,
        request: RuntimeDraftRequest,
        cancellation: Option<&skippy_inference_api::CancellationToken>,
    ) -> DraftResult {
        let (reply, result) = std_mpsc::sync_channel(1);
        self.enqueue_command(SchedulerCommand::Draft(DraftOperation {
            request,
            cancellation: cancellation.cloned(),
            enqueued_at: Instant::now(),
            reply,
        }))?;
        result.recv().map_err(|error| {
            InferenceError::backend(format!("iteration scheduler stopped: {error}"))
        })?
    }
}

fn take_drafts(queue: &mut VecDeque<DraftOperation>, lanes: usize) -> Vec<DraftOperation> {
    let mut claimed = BTreeSet::new();
    let mut batch = Vec::new();
    let mut deferred = VecDeque::new();
    for _ in 0..queue.len() {
        if batch.len() >= lanes.max(1) {
            break;
        }
        let Some(operation) = queue.pop_front() else {
            break;
        };
        if claimed.insert(operation.request.session_id.clone()) {
            batch.push(operation);
        } else {
            deferred.push_back(operation);
        }
    }
    deferred.append(queue);
    *queue = deferred;
    batch
}

impl SchedulerWorker {
    pub(super) fn coalesce_drafts(&mut self) -> bool {
        if self.drafts.is_empty() {
            return true;
        }
        let target = self.active_runtime_sessions.min(self.max_direct_batch_size);
        let mut idle_budget = Duration::from_micros(300);
        for _ in 0..self.max_commands_per_turn {
            if self.drafts.is_empty() || self.drafts.len() >= target || idle_budget.is_zero() {
                break;
            }
            let started = Instant::now();
            let received = self.commands.recv_timeout(idle_budget);
            idle_budget = idle_budget.saturating_sub(started.elapsed());
            match received {
                Ok(SchedulerCommand::Shutdown) => {
                    self.fail_all(InferenceError::backend("iteration scheduler stopped"));
                    return false;
                }
                Ok(command) => self.handle_command(command),
                Err(std_mpsc::RecvTimeoutError::Timeout) => break,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => return false,
            }
        }
        true
    }

    pub(super) fn run_draft_batch(&mut self) {
        let batch = take_drafts(&mut self.drafts, self.max_direct_batch_size);
        if batch.is_empty() {
            return;
        }
        let lock_started = Instant::now();
        let mut runtime = lock_runtime(&self.runtime);
        let hold_started = Instant::now();
        let mut valid = Vec::new();
        for operation in batch {
            let status = ensure_direct_iteration_active(None, operation.cancellation.as_ref());
            if let Err(error) = status {
                operation.complete(Err(error), lock_started, hold_started);
            } else if runtime.session_token_count(&operation.request.session_id)
                != Some(operation.request.base_position)
            {
                operation.complete(Ok(None), lock_started, hold_started);
            } else {
                valid.push(operation);
            }
        }
        let mut shared_lanes = 0;
        if !valid.is_empty() {
            let requests: Vec<_> = valid.iter().map(|operation| &operation.request).collect();
            match runtime.dflash_propose_batch(&requests) {
                Ok(output) => {
                    shared_lanes = output.shared_lanes;
                    for (operation, proposal) in valid.into_iter().zip(output.proposals) {
                        operation.complete(Ok(proposal), lock_started, hold_started);
                    }
                }
                Err(error) => {
                    let error = openai_backend_error(error);
                    for operation in valid {
                        operation.complete(Err(error.clone()), lock_started, hold_started);
                    }
                }
            }
        }
        self.active_runtime_sessions = runtime.active_session_count();
        drop(runtime);
        self.record_compute(hold_started.elapsed());
        if shared_lanes > 1
            && let Some(telemetry) = &self.telemetry
        {
            telemetry.emit(
                "stage.shared_draft",
                BTreeMap::from([("skippy.draft.batch_size".to_string(), json!(shared_lanes))]),
            );
        }
    }

    pub(super) fn has_draft_for(&self, session_id: &str) -> bool {
        self.drafts
            .iter()
            .any(|operation| operation.request.session_id == session_id)
    }

    pub(super) fn fail_drafts(&mut self, error: &InferenceError) {
        let now = Instant::now();
        for operation in self.drafts.drain(..) {
            operation.complete(Err(error.clone()), now, now);
        }
    }
}
