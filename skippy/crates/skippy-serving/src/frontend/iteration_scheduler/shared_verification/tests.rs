use super::*;

#[test]
fn sharing_is_default_and_explicit_zero_retains_independent_verification() {
    for setting in [None, Some(""), Some("1")] {
        assert!(enabled_by(setting));
    }
    assert!(!enabled_by(Some("0")));
}

fn operation(id: &str, width: usize) -> VerificationOperation {
    VerificationOperation {
        request: RuntimeVerificationRequest {
            session_id: id.to_string(),
            token_ids: vec![1; width],
            sampling: None,
            base_position: 0,
        },
        cancellation: None,
        complete: Box::new(|_, _, _, _| {}),
    }
}

pub(in crate::frontend::iteration_scheduler) fn worker() -> SchedulerWorker {
    let (_commands, receiver) = std_mpsc::channel();
    SchedulerWorker {
        runtime: Arc::new(Mutex::new(RuntimeState::new_modelless_for_test(2))),
        compute_meter: Arc::default(),
        scheduler: Scheduler::new(build_scheduler_config(2, 64, 0, Some(8), Some(8), 8, false)),
        requests: BTreeMap::new(),
        direct_iterations: VecDeque::new(),
        verifications: VecDeque::new(),
        drafts: VecDeque::new(),
        max_verify_iteration_tokens: 512,
        cache_runtime_queue: CacheRuntimeQueue::new(CACHE_AGING_COST_PER_TURN, true),
        commands: receiver,
        kv_capacity_tokens: 64,
        max_direct_batch_size: 2,
        direct_group_batch_size: 2,
        batch_policy: crate::frontend::iteration_scheduler::batch_policy::default_policy(),
        max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
        max_commands_per_turn: 8,
        iteration_interval: Duration::ZERO,
        active_runtime_sessions: 0,
        cache_wave_sessions: 0,
        direct_wave_full: false,
        telemetry: None,
        last_served_direct: false,
        last_served_cache_runtime: false,
        last_emitted_lifecycle_counters: (0, 0, 0, 0),
    }
}

#[test]
fn incompatible_windows_keep_session_fifo_while_other_lanes_join() {
    let mut queue = VecDeque::from([
        operation("a", 4),
        operation("b", 2),
        operation("b", 4),
        operation("c", 4),
    ]);
    let batch = take_batch(&mut queue, 4, 32);
    assert_eq!(
        batch
            .iter()
            .map(|op| op.request.session_id.as_str())
            .collect::<Vec<_>>(),
        ["a", "c"]
    );
    assert_eq!(
        queue
            .iter()
            .map(|op| (op.request.session_id.as_str(), op.request.token_ids.len()))
            .collect::<Vec<_>>(),
        [("b", 2), ("b", 4)]
    );
}

#[test]
fn combined_windows_obey_the_native_microbatch_limit() {
    let mut queue = VecDeque::from([operation("a", 7), operation("b", 7), operation("c", 7)]);
    assert_eq!(take_batch(&mut queue, 4, 14).len(), 2);
    assert_eq!(queue.front().unwrap().request.session_id, "c");
    let mut queue = VecDeque::from([operation("empty", 0), operation("a", 7)]);
    assert_eq!(take_batch(&mut queue, 4, 14).len(), 1);
}

#[test]
fn cancelled_and_stale_windows_never_activate_a_runtime_lane() {
    let mut worker = worker();
    let (send, receive) = std_mpsc::channel();
    let mut cancelled = operation("cancelled", 4);
    let cancellation = skippy_inference_api::CancellationToken::new();
    cancellation.cancel();
    cancelled.cancellation = Some(cancellation);
    let sender = send.clone();
    cancelled.complete = Box::new(move |_, result, _, _| sender.send(result.is_err()).unwrap());
    let mut stale = operation("stale", 4);
    stale.complete =
        Box::new(move |_, result, _, _| send.send(matches!(result, Ok(None))).unwrap());
    worker.verifications.extend([cancelled, stale]);
    worker.run_verification_batch();
    assert!(receive.recv().unwrap());
    assert!(receive.recv().unwrap());
    assert_eq!(lock_runtime(&worker.runtime).active_session_count(), 0);
}

#[test]
fn runtime_mutation_is_a_barrier_after_queued_verification() {
    let mut worker = worker();
    let (send, receive) = std_mpsc::channel();
    let mut stale = operation("stale", 4);
    let sender = send.clone();
    stale.complete = Box::new(move |_, _, _, _| sender.send("verify").unwrap());
    worker.verifications.push_back(stale);
    worker.handle_command(SchedulerCommand::ExecuteRuntime(RuntimeOperation {
        label: "repair",
        control: None,
        run: Box::new(move |_| {
            send.send("repair").unwrap();
            Duration::ZERO
        }),
    }));
    assert_eq!(
        [receive.recv().unwrap(), receive.recv().unwrap()],
        ["verify", "repair"]
    );
}

#[test]
fn shutdown_completes_waiting_verification_with_an_error() {
    let mut worker = worker();
    let (send, receive) = std_mpsc::channel();
    let mut operation = operation("waiting", 4);
    operation.complete = Box::new(move |_, result, _, _| send.send(result.is_err()).unwrap());
    worker.verifications.push_back(operation);
    worker.fail_all(InferenceError::backend("stopped"));
    assert!(receive.recv().unwrap());
    assert!(worker.verifications.is_empty());
}

#[test]
fn repairs_on_other_lanes_do_not_flush_waiting_verification() {
    let mut worker = worker();
    let (send, receive) = std_mpsc::channel();
    let mut waiting = operation("a", 4);
    let sender = send.clone();
    waiting.complete = Box::new(move |_, _, _, _| sender.send("verify-a").unwrap());
    worker.verifications.push_back(waiting);
    let sender = send.clone();
    worker.handle_command(SchedulerCommand::ExecuteLaneRuntime(
        "b".to_string(),
        RuntimeOperation {
            label: "repair-b",
            control: None,
            run: Box::new(move |_| {
                sender.send("repair-b").unwrap();
                Duration::ZERO
            }),
        },
    ));
    assert_eq!(receive.recv().unwrap(), "repair-b");
    assert_eq!(worker.verifications.len(), 1);
    worker.handle_command(SchedulerCommand::ExecuteLaneRuntime(
        "a".to_string(),
        RuntimeOperation {
            label: "repair-a",
            control: None,
            run: Box::new(move |_| {
                send.send("repair-a").unwrap();
                Duration::ZERO
            }),
        },
    ));
    assert_eq!(
        [receive.recv().unwrap(), receive.recv().unwrap()],
        ["verify-a", "repair-a"]
    );
}
