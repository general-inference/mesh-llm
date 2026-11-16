//! Independent lanes sharing one advisory draft-model forward.
use super::*;
use skippy_runtime::dflash::proposal_batch::DFlashBatchRequest;

pub(crate) struct RuntimeDraftRequest {
    pub(crate) session_id: String,
    pub(crate) anchor: i32,
    pub(crate) max_tokens: usize,
    pub(crate) base_position: u64,
}

pub(crate) struct RuntimeDraftBatch {
    pub(crate) proposals: Vec<Option<skippy_runtime::DFlashProposal>>,
    pub(crate) shared_lanes: usize,
}

impl RuntimeState {
    pub(crate) fn dflash_propose_batch(
        &mut self,
        requests: &[&RuntimeDraftRequest],
    ) -> Result<RuntimeDraftBatch> {
        self.complete_parked_session_ends();
        let indexes: BTreeMap<_, _> = requests
            .iter()
            .enumerate()
            .map(|(index, request)| (request.session_id.as_str(), index))
            .collect();
        anyhow::ensure!(indexes.len() == requests.len(), "duplicate draft session");
        let mut order = Vec::new();
        let mut before = Vec::new();
        let mut native: Vec<_> = self
            .sessions
            .iter_mut()
            .filter_map(|(id, lane)| {
                let &index = indexes.get(id.as_str())?;
                let request = requests[index];
                order.push(index);
                before.push(
                    lane.session
                        .dflash_draft_stats()
                        .map_or(0, |stats| stats.shared_proposals),
                );
                Some(DFlashBatchRequest {
                    session: &mut lane.session,
                    anchor: request.anchor,
                    max_tokens: request.max_tokens,
                })
            })
            .collect();
        anyhow::ensure!(
            native.len() == requests.len(),
            "draft session is not active"
        );
        let outputs = StageSession::dflash_propose_batch(&mut native)?;
        let shared_lanes = native
            .iter()
            .zip(before)
            .filter(|(request, before)| {
                request
                    .session
                    .dflash_draft_stats()
                    .is_some_and(|stats| stats.shared_proposals > *before)
            })
            .count();
        let mut proposals = vec![None; requests.len()];
        for (index, output) in order.into_iter().zip(outputs) {
            proposals[index] = output;
        }
        Ok(RuntimeDraftBatch {
            proposals,
            shared_lanes,
        })
    }
}
