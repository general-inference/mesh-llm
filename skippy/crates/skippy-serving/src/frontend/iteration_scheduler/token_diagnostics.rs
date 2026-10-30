//! Test-only, request-scoped native signal capture at the scheduler boundary.
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use skippy_runtime::TokenSignal;

use crate::runtime_state::RuntimeState;

type CapturedSignals = BTreeMap<String, Vec<(i32, Option<TokenSignal>)>>;

fn captures() -> &'static Mutex<CapturedSignals> {
    static CAPTURES: OnceLock<Mutex<CapturedSignals>> = OnceLock::new();
    CAPTURES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(super) fn record_last_work(runtime: &mut RuntimeState, session_id: &str, token: i32) {
    let mut captures = captures().lock().unwrap();
    let Some(events) = captures.get_mut(session_id) else {
        return;
    };
    events.push((token, runtime.last_token_signal(session_id).ok()));
}

pub(in crate::frontend) fn enable(session_id: &str) {
    captures()
        .lock()
        .unwrap()
        .insert(session_id.to_string(), Vec::new());
}

pub(in crate::frontend) fn take(session_id: &str) -> Vec<(i32, Option<TokenSignal>)> {
    captures()
        .lock()
        .unwrap()
        .remove(session_id)
        .unwrap_or_default()
}
