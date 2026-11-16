//! Shared target forwards for already-active speculative lanes.
use super::*;
use skippy_runtime::verification_batch::VerifyBatchRequest;

pub(crate) struct RuntimeVerificationRequest {
    pub(crate) session_id: String,
    pub(crate) token_ids: Vec<i32>,
    pub(crate) sampling: Option<SamplingConfig>,
    pub(crate) base_position: u64,
}

impl RuntimeState {
    pub(crate) fn supports_shared_verification(&self) -> bool {
        self.model.supports_shared_verification()
    }

    pub(crate) fn shared_verification_fits(
        &self,
        requests: &[&RuntimeVerificationRequest],
    ) -> bool {
        if !self.supports_shared_verification() {
            return false;
        }
        let total = requests.iter().try_fold(0usize, |sum, request| {
            sum.checked_add(request.token_ids.len())
        });
        total.is_some_and(|total| {
            requests.iter().all(|request| {
                self.sessions
                    .get(&request.session_id)
                    .and_then(|lane| lane.session.verification_batch_capacity())
                    .is_some_and(|capacity| total <= capacity)
            })
        })
    }

    pub(crate) fn verify_batch_sampled(
        &mut self,
        requests: &[&RuntimeVerificationRequest],
    ) -> Result<Vec<Vec<i32>>> {
        self.complete_parked_session_ends();
        anyhow::ensure!(
            self.supports_shared_verification(),
            "target cannot share verification"
        );
        let indexes: BTreeMap<_, _> = requests
            .iter()
            .enumerate()
            .map(|(index, request)| (request.session_id.as_str(), index))
            .collect();
        anyhow::ensure!(
            indexes.len() == requests.len(),
            "duplicate verification session"
        );
        let mut order = Vec::new();
        let mut native: Vec<_> = self
            .sessions
            .iter_mut()
            .filter_map(|(id, lane)| {
                let &index = indexes.get(id.as_str())?;
                let request = requests[index];
                order.push(index);
                Some(VerifyBatchRequest {
                    session: &mut lane.session,
                    token_ids: &request.token_ids,
                    sampling: request.sampling.as_ref(),
                })
            })
            .collect();
        anyhow::ensure!(
            native.len() == requests.len(),
            "verification session is not active"
        );
        let outputs = StageSession::verify_batch_sampled(&mut native)?;
        let mut predictions = vec![Vec::new(); requests.len()];
        for (index, output) in order.into_iter().zip(outputs) {
            predictions[index] = output;
            self.add_session_tokens(
                &requests[index].session_id,
                requests[index].token_ids.len() as u64,
            );
        }
        Ok(predictions)
    }
}
