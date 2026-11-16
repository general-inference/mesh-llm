//! Shared draft submissions with independent target-feature contexts.
use super::*;

/// One lane's next draft. Every session must be exclusively borrowed while
/// its proposal runs, just as for [`StageSession::dflash_propose`].
pub struct DFlashBatchRequest<'a> {
    pub session: &'a mut StageSession,
    pub anchor: i32,
    pub max_tokens: usize,
}

fn independent(requests: &mut [DFlashBatchRequest<'_>]) -> Result<Vec<Option<DFlashProposal>>> {
    requests
        .iter_mut()
        .map(|request| {
            request
                .session
                .dflash_propose(request.anchor, request.max_tokens)
        })
        .collect()
}

impl StageSession {
    /// Runs compatible noise blocks in one draft forward. Unattached drafts,
    /// different draft models, unsupported heads and excess native capacity
    /// retain independent submissions. A draft failure clears its affected
    /// draft lanes and returns empty proposals; target state is untouched.
    pub fn dflash_propose_batch(
        requests: &mut [DFlashBatchRequest<'_>],
    ) -> Result<Vec<Option<DFlashProposal>>> {
        if requests.len() < 2 {
            return independent(requests);
        }
        let drafts = requests
            .iter()
            .map(|request| {
                request
                    .session
                    .dflash
                    .as_ref()
                    .and_then(|link| link.draft.upgrade())
            })
            .collect::<Option<Vec<_>>>();
        let Some(drafts) = drafts else {
            return independent(requests);
        };
        if drafts.iter().any(|draft| !Arc::ptr_eq(draft, &drafts[0])) {
            return independent(requests);
        }
        // Validate positions before ingesting features or changing any lane.
        let positions = requests
            .iter()
            .map(|request| {
                i32::try_from(request.session.native_position()?)
                    .context("anchor position exceeds i32")
            })
            .collect::<Result<Vec<_>>>()?;
        let started = Instant::now();
        let mut blocks = Vec::new();
        let mut order = Vec::new();
        for (index, request) in requests.iter_mut().enumerate() {
            let link = request
                .session
                .dflash
                .as_mut()
                .expect("checked attached draft");
            link.flush_pending();
            if link.context.can_propose(positions[index] as u64) && request.max_tokens > 0 {
                order.push(index);
                blocks.push(NoiseBlock {
                    seq_id: link.seq_id,
                    position: positions[index],
                    anchor: request.anchor,
                    max_tokens: request.max_tokens,
                });
            } else if request.max_tokens == 0 && link.context.can_propose(positions[index] as u64) {
                link.context.truncated(positions[index] as u64);
            }
        }
        let mut draft = lock_draft(&drafts[0]);
        if !draft.blocks_fit(&blocks) {
            drop(draft);
            return independent(requests);
        }
        let proposed = draft.propose_blocks(&blocks);
        let shared = blocks.len() > 1;
        if proposed.is_err() {
            for block in &blocks {
                draft.clear(block.seq_id);
            }
        }
        drop(draft);
        let elapsed = elapsed_us(started);
        let compute_share = elapsed / blocks.len().max(1) as u64;
        let mut outputs = vec![
            Some(DFlashProposal {
                tokens: Vec::new(),
                proposal_compute_us: elapsed
            });
            requests.len()
        ];
        for request in requests.iter_mut() {
            let link = request
                .session
                .dflash
                .as_mut()
                .expect("checked attached draft");
            link.stats.proposals += 1;
        }
        if shared {
            for &index in &order {
                requests[index]
                    .session
                    .dflash
                    .as_mut()
                    .expect("checked attached draft")
                    .stats
                    .shared_proposals += 1;
            }
        }
        match proposed {
            Ok(tokens) => {
                for (index, tokens) in order.into_iter().zip(tokens) {
                    let link = requests[index]
                        .session
                        .dflash
                        .as_mut()
                        .expect("checked attached draft");
                    link.context.truncated(positions[index] as u64);
                    link.stats.propose_us = link.stats.propose_us.saturating_add(elapsed);
                    outputs[index] = Some(DFlashProposal {
                        tokens,
                        proposal_compute_us: compute_share,
                    });
                }
            }
            Err(error) => {
                write_native_log_note(format!(
                    "dflash shared proposal failed, draft contexts cleared: {error:#}"
                ));
                for index in order {
                    let link = requests[index]
                        .session
                        .dflash
                        .as_mut()
                        .expect("checked attached draft");
                    link.context.cleared(positions[index] as u64);
                    link.stats.propose_us = link.stats.propose_us.saturating_add(elapsed);
                }
            }
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn without_draft() -> StageSession {
        StageSession {
            raw: ptr::null_mut(),
            token_count: 11,
            terminal_stage: true,
            batched_activation_exports: true,
            dflash: None,
            dflash_split: None,
        }
    }

    #[test]
    fn requests_without_drafts_do_not_call_native_or_change_target_positions() {
        let mut a = without_draft();
        let mut b = without_draft();
        let outputs = StageSession::dflash_propose_batch(&mut [
            DFlashBatchRequest {
                session: &mut a,
                anchor: 1,
                max_tokens: 6,
            },
            DFlashBatchRequest {
                session: &mut b,
                anchor: 2,
                max_tokens: 3,
            },
        ])
        .unwrap();
        assert_eq!(outputs, [None, None]);
        assert_eq!((a.token_count(), b.token_count()), (11, 11));
        assert!(
            StageSession::dflash_propose_batch(&mut [])
                .unwrap()
                .is_empty()
        );
    }
}
