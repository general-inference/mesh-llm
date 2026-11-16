use super::super::shared_verification::tests::worker;
use super::*;

#[test]
fn sharing_is_default_and_explicit_zero_retains_independent_drafts() {
    for setting in [None, Some(""), Some("1")] {
        assert!(enabled_by(setting));
    }
    assert!(!enabled_by(Some("0")));
}

fn operation(id: &str, width: usize) -> (DraftOperation, std_mpsc::Receiver<DraftResult>) {
    let (reply, result) = std_mpsc::sync_channel(1);
    (
        DraftOperation {
            request: RuntimeDraftRequest {
                session_id: id.to_string(),
                anchor: 1,
                max_tokens: width,
                base_position: 0,
            },
            cancellation: None,
            enqueued_at: Instant::now(),
            reply,
        },
        result,
    )
}

#[test]
fn ragged_drafts_join_without_reordering_one_sessions_work() {
    let (a, _) = operation("a", 6);
    let (b, _) = operation("b", 3);
    let (again, _) = operation("a", 2);
    let (c, _) = operation("c", 4);
    let mut queue = VecDeque::from([a, again, b, c]);
    let batch = take_drafts(&mut queue, 3);
    assert_eq!(
        batch
            .iter()
            .map(|op| (op.request.session_id.as_str(), op.request.max_tokens))
            .collect::<Vec<_>>(),
        [("a", 6), ("b", 3), ("c", 4)]
    );
    assert_eq!(queue.front().unwrap().request.max_tokens, 2);
}

#[test]
fn cancelled_and_stale_drafts_do_not_activate_a_native_lane() {
    let mut worker = worker();
    let (mut cancelled, error) = operation("cancelled", 6);
    let token = skippy_inference_api::CancellationToken::new();
    token.cancel();
    cancelled.cancellation = Some(token);
    let (stale, empty) = operation("stale", 3);
    worker.drafts.extend([cancelled, stale]);
    worker.run_draft_batch();
    assert!(error.recv().unwrap().is_err());
    assert!(empty.recv().unwrap().unwrap().value.is_none());
    assert_eq!(lock_runtime(&worker.runtime).active_session_count(), 0);
}

#[test]
fn mutations_respect_draft_session_barriers_and_shutdown_returns_errors() {
    let mut worker = worker();
    let (waiting, result) = operation("a", 6);
    worker.drafts.push_back(waiting);
    let (send, receive) = std_mpsc::channel();
    let sender = send.clone();
    worker.handle_command(SchedulerCommand::ExecuteLaneRuntime(
        "b".to_string(),
        RuntimeOperation {
            label: "repair-b",
            control: None,
            run: Box::new(move |_| {
                sender.send("b").unwrap();
                Duration::ZERO
            }),
        },
    ));
    assert_eq!(receive.recv().unwrap(), "b");
    assert_eq!(worker.drafts.len(), 1);
    worker.handle_command(SchedulerCommand::ExecuteRuntime(RuntimeOperation {
        label: "mutation",
        control: None,
        run: Box::new(move |_| {
            send.send("global").unwrap();
            Duration::ZERO
        }),
    }));
    assert_eq!(receive.recv().unwrap(), "global");
    assert!(result.recv().unwrap().unwrap().value.is_none());
    assert!(worker.drafts.is_empty());
    let (waiting, result) = operation("waiting", 4);
    worker.drafts.push_back(waiting);
    worker.fail_all(InferenceError::backend("stopped"));
    assert!(result.recv().unwrap().is_err());
}
