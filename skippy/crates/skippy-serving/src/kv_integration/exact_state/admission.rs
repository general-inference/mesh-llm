//! Admission budget for in-flight exact-state payloads: how many exported
//! snapshots may be exporting, queued for the recorder, or held by it, and
//! which capture classes may take the last slots.

use std::sync::Arc;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{AcqRel, Acquire},
};

use super::publication::PendingPublication;
use super::snapshot_headroom;

/// Admission class for an exact-state capture, from most to least likely to
/// be on a follow-up request's path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureAdmission {
    /// The rendered message history without the assistant marker: the
    /// prefix every append-only follow-up in the conversation (the next chat
    /// turn, the next tool-call step) shares token for token. Owns a slot no
    /// other class can take, so it is never declined because earlier
    /// post-decode or prefill-ladder records are still being stored.
    MessageBoundary,
    /// Same-session continuation boundary (post-decode). Re-derivable only by
    /// replaying generation; competes for the reserved recorder slot.
    Continuation,
    /// Prefill-ladder boundary (shared checkpoint / final prefill state).
    /// Re-derivable by re-prefill; best-effort admission.
    BestEffort,
}

impl CaptureAdmission {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MessageBoundary => "message_boundary",
            Self::Continuation => "continuation",
            Self::BestEffort => "best_effort",
        }
    }
}

/// In-flight exact-state payloads per stage for continuation and
/// best-effort captures — exporting, queued in the recorder channel, or held
/// by the recorder worker. Sized as channel capacity (1) plus one
/// worker-held record.
pub const EXACT_STATE_ADMISSION_TOTAL: usize = 2;
/// Slots reserved for [`CaptureAdmission::Continuation`]: BestEffort captures
/// can never occupy more than `TOTAL - RESERVED` slots, so a continuation
/// boundary always finds a slot unless the whole budget is exhausted.
pub const EXACT_STATE_ADMISSION_RESERVED: usize = 1;
/// Slots per serving lane above [`EXACT_STATE_ADMISSION_TOTAL`] that only
/// [`CaptureAdmission::MessageBoundary`] may take, so concurrent
/// conversations on different lanes each keep their boundary. Host memory
/// stays guarded by the per-snapshot headroom reservation, which counts
/// every in-flight snapshot including these.
pub const EXACT_STATE_ADMISSION_MESSAGE_BOUNDARY_RESERVED: usize = 1;

/// Move-only credit for one in-flight exact-state payload. Construction
/// reserves a slot atomically (CAS, so competing producers cannot
/// oversubscribe the budget); Drop releases it exactly once on every terminal
/// path — payload stored, skipped, errored, command dropped unexecuted, or
/// channel receiver dropped. The credit moves into
/// [`super::super::PendingExactStateRecord`] so the worker's completion
/// releases it.
#[derive(Debug)]
pub struct ExactStateAdmissionCredit {
    class: CaptureAdmission,
    total: Arc<AtomicUsize>,
    best_effort: Arc<AtomicUsize>,
    pub(super) snapshot_reservation: Option<snapshot_headroom::SnapshotReservation>,
    /// Lets lookups wait for this record's radix publication; dropped with
    /// the credit, after the worker's radix insert.
    pub(super) publication: Option<PendingPublication>,
}

/// Takes one `total` slot while fewer than `cap` are held.
fn reserve_slot(counter: &AtomicUsize, cap: usize) -> bool {
    let mut observed = counter.load(Acquire);
    loop {
        if observed >= cap {
            return false;
        }
        match counter.compare_exchange_weak(observed, observed + 1, AcqRel, Acquire) {
            Ok(_) => return true,
            Err(actual) => observed = actual,
        }
    }
}

impl ExactStateAdmissionCredit {
    /// Acquires on a single-lane stage.
    pub(crate) fn acquire(
        total: &Arc<AtomicUsize>,
        best_effort: &Arc<AtomicUsize>,
        class: CaptureAdmission,
    ) -> Option<Self> {
        Self::acquire_for_lanes(total, best_effort, class, 1)
    }

    /// Acquires on a stage serving `lanes` concurrent sequences; only the
    /// message-boundary reserve scales with lanes.
    pub(crate) fn acquire_for_lanes(
        total: &Arc<AtomicUsize>,
        best_effort: &Arc<AtomicUsize>,
        class: CaptureAdmission,
        lanes: usize,
    ) -> Option<Self> {
        let admitted = match class {
            CaptureAdmission::MessageBoundary => reserve_slot(
                total,
                EXACT_STATE_ADMISSION_TOTAL
                    + EXACT_STATE_ADMISSION_MESSAGE_BOUNDARY_RESERVED * lanes.max(1),
            ),
            CaptureAdmission::Continuation => reserve_slot(total, EXACT_STATE_ADMISSION_TOTAL),
            CaptureAdmission::BestEffort => {
                let reserved_cap = EXACT_STATE_ADMISSION_TOTAL - EXACT_STATE_ADMISSION_RESERVED;
                if !reserve_slot(best_effort, reserved_cap) {
                    return None;
                }
                // Reserve the total slot; roll the BestEffort reservation back
                // if the total budget was exhausted concurrently.
                let admitted = reserve_slot(total, EXACT_STATE_ADMISSION_TOTAL);
                if !admitted {
                    best_effort.fetch_sub(1, AcqRel);
                }
                admitted
            }
        };
        admitted.then(|| Self {
            class,
            total: Arc::clone(total),
            best_effort: Arc::clone(best_effort),
            snapshot_reservation: None,
            publication: None,
        })
    }
}

impl Drop for ExactStateAdmissionCredit {
    fn drop(&mut self) {
        let _ = self.total.fetch_sub(1, AcqRel);
        if self.class == CaptureAdmission::BestEffort {
            let _ = self.best_effort.fetch_sub(1, AcqRel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CaptureAdmission, EXACT_STATE_ADMISSION_MESSAGE_BOUNDARY_RESERVED,
        EXACT_STATE_ADMISSION_RESERVED, EXACT_STATE_ADMISSION_TOTAL, ExactStateAdmissionCredit,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn counters() -> (Arc<AtomicUsize>, Arc<AtomicUsize>) {
        (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)))
    }

    #[test]
    fn construction_and_drop_account_exactly_once() {
        let (total, best_effort) = counters();
        let guard = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(guard.is_some(), "fresh budget admits Continuation");
        assert_eq!(total.load(Ordering::Acquire), 1);
        drop(guard);
        assert_eq!(total.load(Ordering::Acquire), 0);
    }

    #[test]
    fn continuation_declines_when_total_budget_exhausted() {
        let (total, best_effort) = counters();
        let first = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        let second = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(first.is_some() && second.is_some(), "two slots for TOTAL=2");
        let third = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(
            third.is_none(),
            "third Continuation must decline at TOTAL=2"
        );
        assert_eq!(total.load(Ordering::Acquire), 2);
    }

    #[test]
    fn best_effort_declines_at_reserved_cap_even_with_budget_headroom() {
        let (total, best_effort) = counters();
        let first =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(first.is_some());
        let second =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(
            second.is_none(),
            "BestEffort cap is TOTAL - RESERVED = 1; second must decline"
        );
        assert_eq!(best_effort.load(Ordering::Acquire), 1);
    }

    #[test]
    fn continuation_admits_after_best_effort_under_reserved_share() {
        let (total, best_effort) = counters();
        let best =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(best.is_some(), "BestEffort takes its single slot");
        let cont = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        assert!(
            cont.is_some(),
            "Continuation must admit into the reserved share"
        );
    }

    #[test]
    fn concurrent_producers_never_oversubscribe_the_budget() {
        use std::sync::Barrier;
        const THREADS: usize = 4;
        const ATTEMPTS: usize = 25;
        // ONE shared budget for every producer: racing acquisitions hit the
        // SAME counters, classes are mixed per attempt (both CAS paths,
        // including the BestEffort rollback, race concurrently), credits are
        // RETAINED across a barrier, and rollback/drain are asserted exactly.
        let (total, best_effort) = counters();
        let start = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|thread| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    let mut held = Vec::new();
                    start.wait();
                    for attempt in 0..ATTEMPTS {
                        let class = if (thread + attempt) % 2 == 0 {
                            CaptureAdmission::Continuation
                        } else {
                            CaptureAdmission::BestEffort
                        };
                        if let Some(credit) =
                            ExactStateAdmissionCredit::acquire(&total, &best_effort, class)
                        {
                            held.push((class, credit));
                        }
                    }
                    held
                })
            })
            .collect();
        let mut held: Vec<_> = handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect();
        let held_best_effort = held
            .iter()
            .filter(|(class, _)| *class == CaptureAdmission::BestEffort)
            .count();
        // Retention bounds with mixed classes: Continuation fills at most
        // TOTAL slots, BestEffort at most TOTAL - RESERVED, and together they
        // cannot exceed the total budget.
        assert!(held.len() <= EXACT_STATE_ADMISSION_TOTAL);
        assert!(
            held_best_effort <= EXACT_STATE_ADMISSION_TOTAL - EXACT_STATE_ADMISSION_RESERVED,
            "BestEffort holds must respect the reserved share"
        );
        assert_eq!(
            total.load(Ordering::Acquire),
            held.len(),
            "retained credits must account for every outstanding slot"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            held_best_effort,
            "retained credits must account for the BestEffort counter"
        );

        // Top the budget up to exactly TOTAL held credits so the hammer below
        // runs against a deterministically exhausted budget.
        while held.len() < EXACT_STATE_ADMISSION_TOTAL {
            let credit = ExactStateAdmissionCredit::acquire(
                &total,
                &best_effort,
                CaptureAdmission::Continuation,
            )
            .expect("topping up an under-subscribed budget must succeed");
            held.push((CaptureAdmission::Continuation, credit));
        }

        // Rollback hammer: the total budget is exhausted by the held credits,
        // so every concurrent BestEffort attempt must decline, and the
        // rollback path must not leak a BestEffort slot (a missing rollback
        // would drift the counter above the held count).
        let hammer = Arc::new(Barrier::new(THREADS));
        let hammers: Vec<_> = (0..THREADS)
            .map(|_| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let hammer = Arc::clone(&hammer);
                std::thread::spawn(move || {
                    hammer.wait();
                    for _ in 0..ATTEMPTS {
                        assert!(
                            ExactStateAdmissionCredit::acquire(
                                &total,
                                &best_effort,
                                CaptureAdmission::BestEffort
                            )
                            .is_none(),
                            "an exhausted budget must decline every BestEffort attempt"
                        );
                    }
                })
            })
            .collect();
        for handle in hammers {
            handle.join().unwrap();
        }
        assert_eq!(
            total.load(Ordering::Acquire),
            held.len(),
            "the rollback hammer must not change the total"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            held_best_effort,
            "the rollback hammer must not leak a BestEffort slot"
        );

        // Drain: dropping every held credit releases each slot exactly once,
        // and the budget recovers for new admissions.
        drop(held);
        assert_eq!(total.load(Ordering::Acquire), 0);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
        let recovery_continuation = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        );
        let recovery_best_effort =
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort);
        assert!(
            recovery_continuation.is_some() && recovery_best_effort.is_some(),
            "budget must recover after the drain"
        );
    }

    #[test]
    fn rollback_is_exercised_when_the_budget_is_held_by_continuations() {
        use std::sync::Barrier;
        const THREADS: usize = 4;
        const ATTEMPTS: usize = 50;
        // Deterministic rollback exercise: the budget STARTS held by exactly
        // TWO Continuation credits (total=2, best_effort=0), so the first
        // BestEffort attempts must pass the class cap and fail the total CAS
        // — the rollback path. Overlapping contenders can still decline at
        // the class cap while a rollback is in flight; the exact 2/0
        // assertions afterwards are what prove no rollback leaked. The
        // mixed-contention test above can retain a BestEffort credit, in
        // which case its hammer exits at the class cap throughout.
        let (total, best_effort) = counters();
        let first = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("first Continuation admitted");
        let second = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::Continuation,
        )
        .expect("second Continuation admitted");
        assert_eq!(total.load(Ordering::Acquire), 2);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);

        let barrier = Arc::new(Barrier::new(THREADS));
        let hammers: Vec<_> = (0..THREADS)
            .map(|_| {
                let total = Arc::clone(&total);
                let best_effort = Arc::clone(&best_effort);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..ATTEMPTS {
                        assert!(
                            ExactStateAdmissionCredit::acquire(
                                &total,
                                &best_effort,
                                CaptureAdmission::BestEffort
                            )
                            .is_none(),
                            "a continuation-exhausted budget must decline every BestEffort via rollback"
                        );
                    }
                })
            })
            .collect();
        for handle in hammers {
            handle.join().unwrap();
        }
        assert_eq!(
            total.load(Ordering::Acquire),
            2,
            "rollback leaves the total at the held credits"
        );
        assert_eq!(
            best_effort.load(Ordering::Acquire),
            0,
            "rollback must not leak a BestEffort slot"
        );

        // Drain and recovery.
        drop((first, second));
        assert_eq!(total.load(Ordering::Acquire), 0);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
        assert!(
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort)
                .is_some(),
            "budget recovers after the drain"
        );
    }

    #[test]
    fn message_boundary_admits_while_continuations_fill_the_shared_budget() {
        let (total, best_effort) = counters();
        let held: Vec<_> = (0..EXACT_STATE_ADMISSION_TOTAL)
            .map(|_| {
                ExactStateAdmissionCredit::acquire(
                    &total,
                    &best_effort,
                    CaptureAdmission::Continuation,
                )
                .expect("continuations fill the shared budget")
            })
            .collect();
        assert!(
            ExactStateAdmissionCredit::acquire(&total, &best_effort, CaptureAdmission::BestEffort)
                .is_none()
        );
        let boundary = ExactStateAdmissionCredit::acquire(
            &total,
            &best_effort,
            CaptureAdmission::MessageBoundary,
        );
        assert!(
            boundary.is_some(),
            "a message boundary must not be declined by earlier post-decode records"
        );
        assert!(
            ExactStateAdmissionCredit::acquire(
                &total,
                &best_effort,
                CaptureAdmission::MessageBoundary
            )
            .is_none(),
            "the boundary reserve is bounded"
        );
        assert_eq!(
            total.load(Ordering::Acquire),
            EXACT_STATE_ADMISSION_TOTAL + EXACT_STATE_ADMISSION_MESSAGE_BOUNDARY_RESERVED
        );
        drop(boundary);
        assert!(
            ExactStateAdmissionCredit::acquire(
                &total,
                &best_effort,
                CaptureAdmission::Continuation
            )
            .is_none(),
            "a released boundary slot is not taken by a continuation"
        );
        drop(held);
        assert_eq!(total.load(Ordering::Acquire), 0);
        assert_eq!(best_effort.load(Ordering::Acquire), 0);
    }

    #[test]
    fn message_boundary_reserve_scales_with_lanes() {
        let (total, best_effort) = counters();
        let _continuations: Vec<_> = (0..EXACT_STATE_ADMISSION_TOTAL)
            .map(|_| {
                ExactStateAdmissionCredit::acquire(
                    &total,
                    &best_effort,
                    CaptureAdmission::Continuation,
                )
                .expect("continuations fill the shared budget")
            })
            .collect();
        let boundary = |lanes| {
            ExactStateAdmissionCredit::acquire_for_lanes(
                &total,
                &best_effort,
                CaptureAdmission::MessageBoundary,
                lanes,
            )
        };
        let first = boundary(2);
        let second = boundary(2);
        assert!(
            first.is_some() && second.is_some(),
            "each of two lanes keeps a boundary slot"
        );
        assert!(boundary(2).is_none());
    }
}
