//! Coalesces compatible speculative windows without sharing sampling state.
use super::*;
use crate::runtime_state::RuntimeVerificationRequest;

#[cfg(test)]
pub(super) mod tests;

struct VerifiedWindow {
    predictions: Vec<i32>,
    batch_size: usize,
}
type Verified = InferenceResult<Option<VerifiedWindow>>;
type Completion = Box<dyn FnOnce(&mut RuntimeState, Verified, Instant, Instant) + Send>;

pub(super) struct VerificationOperation {
    request: RuntimeVerificationRequest,
    cancellation: Option<skippy_inference_api::CancellationToken>,
    complete: Completion,
}

pub(crate) fn enabled() -> bool {
    enabled_by(env::var("SKIPPY_SHARED_VERIFY").ok().as_deref())
}

/// Complete causal targets share compatible windows unless explicitly disabled.
fn enabled_by(setting: Option<&str>) -> bool {
    setting != Some("0")
}

impl IterationScheduler {
    pub(crate) fn execute_verification_timed<T: Send + 'static>(
        &self,
        request: RuntimeVerificationRequest,
        cancellation: Option<&skippy_inference_api::CancellationToken>,
        finish: impl FnOnce(&mut RuntimeState, Vec<i32>, usize) -> InferenceResult<T> + Send + 'static,
    ) -> InferenceResult<SchedulerRuntimeOutcome<Option<T>>> {
        let (reply, result) = std_mpsc::sync_channel(1);
        let enqueued_at = Instant::now();
        let operation = VerificationOperation {
            request,
            cancellation: cancellation.cloned(),
            complete: Box::new(move |runtime, predictions, lock_started, hold_started| {
                let value = predictions.and_then(|predictions| {
                    predictions
                        .map(|window| finish(runtime, window.predictions, window.batch_size))
                        .transpose()
                });
                let outcome = value.map(|value| SchedulerRuntimeOutcome {
                    value,
                    queue_wait_ms: lock_started.duration_since(enqueued_at).as_secs_f64() * 1000.0,
                    runtime_lock_wait_ms: hold_started.duration_since(lock_started).as_secs_f64()
                        * 1000.0,
                    runtime_lock_hold_ms: hold_started.elapsed().as_secs_f64() * 1000.0,
                });
                let _ = reply.send(outcome);
            }),
        };
        self.enqueue_command(SchedulerCommand::Verify(operation))?;
        result.recv().map_err(|error| {
            InferenceError::backend(format!("iteration scheduler stopped: {error}"))
        })?
    }
}

fn take_batch(
    queue: &mut VecDeque<VerificationOperation>,
    lanes: usize,
    budget: usize,
) -> Vec<VerificationOperation> {
    let Some(first) = queue.pop_front() else {
        return Vec::new();
    };
    let width = first.request.token_ids.len();
    let cap = lanes.min(budget.checked_div(width).unwrap_or(1)).max(1);
    let mut claimed = BTreeSet::from([first.request.session_id.clone()]);
    let mut batch = vec![first];
    let mut deferred = VecDeque::new();
    for _ in 0..queue.len() {
        if batch.len() >= cap {
            break;
        }
        let Some(operation) = queue.pop_front() else {
            break;
        };
        let unique = claimed.insert(operation.request.session_id.clone());
        if unique && operation.request.token_ids.len() == width {
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
    pub(super) fn coalesce_verifications(&mut self) -> bool {
        if self.verifications.is_empty() {
            return true;
        }
        let target = self.active_runtime_sessions.min(self.max_direct_batch_size);
        let mut idle_budget = Duration::from_micros(300);
        for _ in 0..self.max_commands_per_turn {
            if self.verifications.len() >= target || idle_budget.is_zero() {
                break;
            }
            let started = Instant::now();
            let received = self.commands.recv_timeout(idle_budget);
            idle_budget = idle_budget.saturating_sub(started.elapsed());
            // Useful draft/repair work may execute here. Charge only idle
            // waiting to this budget, not model work on another lane.
            match received {
                Ok(SchedulerCommand::Shutdown) => {
                    self.fail_all(InferenceError::backend("iteration scheduler stopped"));
                    return false;
                }
                Ok(command) => {
                    self.handle_command(command);
                    // A peer waiting on its draft cannot enqueue verification
                    // until the draft completes. Let it join this verify wave.
                    self.run_draft_batch();
                }
                Err(std_mpsc::RecvTimeoutError::Timeout) => break,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => return false,
            }
        }
        true
    }

    pub(super) fn run_verification_batch(&mut self) {
        let batch = take_batch(
            &mut self.verifications,
            self.max_direct_batch_size,
            self.max_verify_iteration_tokens,
        );
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
                (operation.complete)(&mut runtime, Err(error), lock_started, hold_started);
            } else if runtime.session_token_count(&operation.request.session_id)
                != Some(operation.request.base_position)
            {
                (operation.complete)(&mut runtime, Ok(None), lock_started, hold_started);
            } else {
                valid.push(operation);
            }
        }
        let shared = valid.len() > 1
            && runtime.shared_verification_fits(
                &valid
                    .iter()
                    .map(|operation| &operation.request)
                    .collect::<Vec<_>>(),
            );
        let batch_size = valid.len();
        if shared {
            let requests: Vec<_> = valid.iter().map(|operation| &operation.request).collect();
            match runtime.verify_batch_sampled(&requests) {
                Ok(outputs) => {
                    for (operation, output) in valid.into_iter().zip(outputs) {
                        (operation.complete)(
                            &mut runtime,
                            Ok(Some(VerifiedWindow {
                                predictions: output,
                                batch_size,
                            })),
                            lock_started,
                            hold_started,
                        );
                    }
                }
                Err(error) => {
                    let error = openai_backend_error(error);
                    for operation in valid {
                        (operation.complete)(
                            &mut runtime,
                            Err(error.clone()),
                            lock_started,
                            hold_started,
                        );
                    }
                }
            }
        } else {
            for operation in valid {
                let request = &operation.request;
                let output = runtime
                    .verify_tokens_sampled(
                        &request.session_id,
                        &request.token_ids,
                        request.sampling.as_ref(),
                    )
                    .map(|predictions| {
                        Some(VerifiedWindow {
                            predictions,
                            batch_size: 1,
                        })
                    })
                    .map_err(openai_backend_error);
                (operation.complete)(&mut runtime, output, lock_started, hold_started);
            }
        }
        let active = runtime.active_session_count();
        drop(runtime);
        self.active_runtime_sessions = active;
        self.record_compute(hold_started.elapsed());
        if shared && let Some(telemetry) = &self.telemetry {
            telemetry.emit(
                "stage.shared_verification",
                BTreeMap::from([
                    ("skippy.verify.shared".to_string(), json!(true)),
                    ("skippy.verify.batch_size".to_string(), json!(batch_size)),
                ]),
            );
        }
    }

    pub(super) fn has_verification_for(&self, session_id: &str) -> bool {
        self.verifications
            .iter()
            .any(|operation| operation.request.session_id == session_id)
    }

    pub(super) fn fail_verifications(&mut self, error: &InferenceError) {
        let mut runtime = lock_runtime(&self.runtime);
        let now = Instant::now();
        for operation in self.verifications.drain(..) {
            (operation.complete)(&mut runtime, Err(error.clone()), now, now);
        }
    }
}
