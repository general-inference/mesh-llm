//! Exact-state records between export and radix publication.
//!
//! A capture exports its state on the runtime thread, then the recorder
//! worker deduplicates and publishes it into the radix. In an agent loop the
//! next request arrives as soon as the previous one finishes decoding, often
//! before the recorder has published the previous request's message-history
//! boundary. Without a way to see that in-flight record, the lookup falls
//! back to an older boundary and every step re-prefills the last tool result
//! twice. Lookups use this index to wait, briefly and boundedly, for an
//! in-flight record that would serve a longer prefix than the radix holds.

use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use super::super::{KvStageIntegration, PrefillKvIdentity};

#[derive(Debug, Default)]
pub(crate) struct PendingPublications {
    state: Mutex<PendingState>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct PendingState {
    next_id: u64,
    entries: BTreeMap<u64, PendingEntry>,
}

#[derive(Debug)]
struct PendingEntry {
    namespace: String,
    token_ids: Vec<i32>,
}

/// Registration of one record awaiting publication. Dropping it (after the
/// radix insert, or when the record is dropped unpublished) removes the
/// entry and wakes waiting lookups.
#[derive(Debug)]
pub(crate) struct PendingPublication {
    id: u64,
    owner: Arc<PendingPublications>,
}

impl Drop for PendingPublication {
    fn drop(&mut self) {
        let mut state = self
            .owner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.entries.remove(&self.id);
        drop(state);
        self.owner.changed.notify_all();
    }
}

/// What a lookup did about in-flight records before reading the radix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PublicationWait {
    /// No in-flight record would extend the restorable prefix.
    NotNeeded,
    /// The longer in-flight record left the queue (normally published).
    Published {
        pending_tokens: usize,
        waited: Duration,
    },
    /// The bound expired first; the lookup proceeds with what is published.
    TimedOut {
        pending_tokens: usize,
        waited: Duration,
    },
}

impl PendingPublications {
    pub(crate) fn register(
        self: &Arc<Self>,
        namespace: &str,
        token_ids: &[i32],
    ) -> PendingPublication {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = state.next_id;
        state.next_id = state.next_id.wrapping_add(1);
        state.entries.insert(
            id,
            PendingEntry {
                namespace: namespace.to_string(),
                token_ids: token_ids.to_vec(),
            },
        );
        PendingPublication {
            id,
            owner: Arc::clone(self),
        }
    }

    /// Longest in-flight record in `namespace` that is a prefix of `tokens`
    /// and longer than `published_tokens`.
    fn longest_extending(
        state: &PendingState,
        namespace: &str,
        tokens: &[i32],
        published_tokens: usize,
    ) -> Option<usize> {
        state
            .entries
            .values()
            .filter(|entry| {
                entry.namespace == namespace
                    && entry.token_ids.len() > published_tokens
                    && tokens.starts_with(&entry.token_ids)
            })
            .map(|entry| entry.token_ids.len())
            .max()
    }

    /// Waits until no in-flight record would serve more of `tokens` than the
    /// `published_tokens` the radix already holds, or until `max_wait(gain)`
    /// passes, where `gain` is the extra prefix the longest such record
    /// would restore.
    pub(crate) fn wait_for_extending(
        &self,
        namespace: &str,
        tokens: &[i32],
        published_tokens: usize,
        max_wait: impl FnOnce(usize) -> Duration,
    ) -> PublicationWait {
        let started = Instant::now();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(pending_tokens) =
            Self::longest_extending(&state, namespace, tokens, published_tokens)
        else {
            return PublicationWait::NotNeeded;
        };
        let deadline = started + max_wait(pending_tokens - published_tokens);
        loop {
            if Self::longest_extending(&state, namespace, tokens, published_tokens).is_none() {
                return PublicationWait::Published {
                    pending_tokens,
                    waited: started.elapsed(),
                };
            }
            let now = Instant::now();
            if now >= deadline {
                return PublicationWait::TimedOut {
                    pending_tokens,
                    waited: started.elapsed(),
                };
            }
            state = self
                .changed
                .wait_timeout(state, deadline - now)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
    }
}

impl KvStageIntegration {
    /// Before a lookup, waits for an exported record that would restore more
    /// of a candidate prefix than the radix holds now. `max_wait` receives
    /// the extra tokens that record would restore and bounds the wait.
    pub(crate) fn wait_for_exact_state_publication(
        &self,
        identities: &[PrefillKvIdentity],
        max_wait: impl Fn(usize) -> Duration,
    ) -> PublicationWait {
        if !self.should_lookup() || !self.payload_is_exact_state() {
            return PublicationWait::NotNeeded;
        }
        for identity in identities {
            let published_tokens = self
                .radix
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .peek_recurrent(&identity.namespace, &identity.token_ids)
                .map_or(0, |matched| matched.matched_tokens);
            let wait = self.pending_publications.wait_for_extending(
                &identity.namespace,
                &identity.token_ids,
                published_tokens,
                &max_wait,
            );
            if wait != PublicationWait::NotNeeded {
                return wait;
            }
        }
        PublicationWait::NotNeeded
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use super::{PendingPublications, PublicationWait};

    #[test]
    fn lookup_does_not_wait_without_a_longer_in_flight_prefix() {
        let pending = Arc::new(PendingPublications::default());
        let _shorter = pending.register("ns", &[1, 2]);
        let _other_namespace = pending.register("other", &[1, 2, 3, 4]);
        let _diverging = pending.register("ns", &[1, 2, 9, 9]);
        let wait = pending.wait_for_extending("ns", &[1, 2, 3, 4, 5], 2, |_| {
            panic!("no wait bound is needed")
        });
        assert_eq!(wait, PublicationWait::NotNeeded);
    }

    #[test]
    fn lookup_waits_for_a_longer_in_flight_prefix_to_publish() {
        let pending = Arc::new(PendingPublications::default());
        let record = pending.register("ns", &[1, 2, 3]);
        let publisher = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            drop(record);
        });
        let wait = pending.wait_for_extending("ns", &[1, 2, 3, 4], 1, |gain| {
            assert_eq!(gain, 2);
            Duration::from_secs(30)
        });
        publisher.join().unwrap();
        assert!(
            matches!(
                wait,
                PublicationWait::Published {
                    pending_tokens: 3,
                    ..
                }
            ),
            "{wait:?}"
        );
    }

    #[test]
    fn lookup_wait_is_bounded() {
        let pending = Arc::new(PendingPublications::default());
        let _stuck = pending.register("ns", &[1, 2, 3]);
        let wait =
            pending.wait_for_extending("ns", &[1, 2, 3, 4], 0, |_| Duration::from_millis(10));
        assert!(
            matches!(
                wait,
                PublicationWait::TimedOut {
                    pending_tokens: 3,
                    ..
                }
            ),
            "{wait:?}"
        );
    }
}
