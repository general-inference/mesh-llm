//! Test-only, request-scoped native signal capture at the scheduler boundary.
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use skippy_runtime::TokenSignal;

use crate::runtime_state::RuntimeState;

#[derive(Debug)]
pub(in crate::frontend) struct CapturedSignal {
    pub(in crate::frontend) token: i32,
    pub(in crate::frontend) signal: Option<TokenSignal>,
    pub(in crate::frontend) batch_size: usize,
}

type CapturedSignals = BTreeMap<String, Vec<CapturedSignal>>;

fn captures() -> &'static Mutex<CapturedSignals> {
    static CAPTURES: OnceLock<Mutex<CapturedSignals>> = OnceLock::new();
    CAPTURES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub(super) fn record_last_work(
    runtime: &mut RuntimeState,
    session_id: &str,
    token: i32,
    batch_size: usize,
) {
    let mut captures = captures().lock().unwrap();
    let Some(events) = captures.get_mut(session_id) else {
        return;
    };
    events.push(CapturedSignal {
        token,
        signal: runtime.last_token_signal(session_id).ok(),
        batch_size,
    });
}

pub(in crate::frontend) fn enable(session_id: &str) {
    captures()
        .lock()
        .unwrap()
        .insert(session_id.to_string(), Vec::new());
}

pub(in crate::frontend) fn take(session_id: &str) -> Vec<CapturedSignal> {
    captures()
        .lock()
        .unwrap()
        .remove(session_id)
        .unwrap_or_default()
}
