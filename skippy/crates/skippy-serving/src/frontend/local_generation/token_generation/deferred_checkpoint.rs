//! Where a deferred prefill stops to record a reusable exact state before
//! prefilling the rest of the prompt: the message-history boundary when the
//! request has one, otherwise the shared prefill-ladder grid.

use super::*;

/// Decision prefix for the message-history boundary checkpoint. It shares
/// the reserved [`crate::kv_integration::CaptureAdmission::MessageBoundary`]
/// admission with the prefill-only chat-prefix path.
pub(in crate::frontend) const MESSAGE_BOUNDARY_CHECKPOINT: &str = "message_boundary_checkpoint";

/// The checkpoint a deferred prefill stops at before its suffix.
pub(super) enum DeferredCheckpoint {
    /// The rendered message history, shared by append-only follow-ups.
    MessageBoundary(PrefillKvIdentity),
    /// The shared prefill-ladder grid.
    SharedGrid(PrefillKvIdentity),
}

impl DeferredCheckpoint {
    pub(super) fn identity(&self) -> &PrefillKvIdentity {
        match self {
            Self::MessageBoundary(identity) | Self::SharedGrid(identity) => identity,
        }
    }
}

pub(super) fn deferred_suffix_split(
    restored_prefill_tokens: usize,
    prompt_token_count: usize,
    checkpoint_token_count: Option<usize>,
) -> (usize, Option<usize>) {
    let suffix_start = restored_prefill_tokens.min(prompt_token_count);
    let checkpoint = checkpoint_token_count
        .filter(|&boundary| boundary > suffix_start && boundary < prompt_token_count);
    (suffix_start, checkpoint)
}

/// Pick the deferred-prefill checkpoint boundary. A valid EXPLICIT chat
/// boundary wins: it is the reusable message-history prefix a follow-up
/// request shares token-for-token, so it is the checkpoint most likely to be
/// on a future request's path. The shared-grid checkpoint is the fallback for
/// absent, invalid, or already-restored chat boundaries.
fn select_deferred_checkpoint_boundary(
    shared_checkpoint: Option<usize>,
    chat_checkpoint: Option<usize>,
    restored_prefill_tokens: usize,
    prompt_token_count: usize,
) -> Option<usize> {
    let valid =
        |boundary: &usize| *boundary > restored_prefill_tokens && *boundary < prompt_token_count;
    chat_checkpoint
        .filter(valid)
        .or_else(|| shared_checkpoint.filter(valid))
}

impl StageOpenAiBackend {
    pub(super) fn deferred_exact_checkpoint_identity(
        &self,
        request: &LocalGeneration<'_>,
        session_id: &str,
        prefill_tokens: &[i32],
        restored_prefill_tokens: usize,
    ) -> Option<DeferredCheckpoint> {
        let kv = self.kv.as_ref().filter(|kv| kv.payload_is_exact_state())?;
        let base = self.local_kv_message_base(session_id, request.ids);
        let shared_checkpoint =
            kv.exact_shared_checkpoint_identity(&self.config, &base, 0, prefill_tokens);
        let chat_checkpoint = request
            .recurrent_cache_prefix_token_ids
            .filter(|checkpoint_tokens| {
                !checkpoint_tokens.is_empty()
                    && checkpoint_tokens.len() <= prefill_tokens.len()
                    && prefill_tokens.starts_with(checkpoint_tokens)
            })
            .map(|checkpoint_tokens| {
                kv.prefill_identity(&self.config, &base, 0, checkpoint_tokens)
            });
        let shared_boundary = shared_checkpoint
            .as_ref()
            .and_then(|identity| usize::try_from(identity.identity.token_count).ok());
        let chat_boundary = chat_checkpoint
            .as_ref()
            .and_then(|identity| usize::try_from(identity.identity.token_count).ok());
        let selected_boundary = select_deferred_checkpoint_boundary(
            shared_boundary,
            chat_boundary,
            restored_prefill_tokens,
            prefill_tokens.len().saturating_add(1),
        );
        if selected_boundary == shared_boundary {
            shared_checkpoint.map(DeferredCheckpoint::SharedGrid)
        } else {
            chat_checkpoint.map(DeferredCheckpoint::MessageBoundary)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{deferred_suffix_split, select_deferred_checkpoint_boundary};

    #[test]
    fn deferred_suffix_split_respects_restore_and_prompt_boundaries() {
        assert_eq!(deferred_suffix_split(0, 10, Some(4)), (0, Some(4)));
        assert_eq!(deferred_suffix_split(4, 10, Some(4)), (4, None));
        assert_eq!(deferred_suffix_split(8, 10, Some(4)), (8, None));
        assert_eq!(deferred_suffix_split(4, 10, Some(10)), (4, None));
        assert_eq!(deferred_suffix_split(99, 10, Some(4)), (10, None));
    }

    #[test]
    fn deferred_checkpoint_prefers_chat_boundary_with_shared_grid_fallback() {
        let prompt = 10;
        // Both valid: the explicit chat boundary wins over the shared grid.
        assert_eq!(
            select_deferred_checkpoint_boundary(Some(8), Some(6), 0, prompt),
            Some(6)
        );
        // Chat absent or None: shared grid is the fallback.
        assert_eq!(
            select_deferred_checkpoint_boundary(Some(8), None, 0, prompt),
            Some(8)
        );
        // Chat already restored (equal to the restored prefix): shared fallback.
        assert_eq!(
            select_deferred_checkpoint_boundary(Some(8), Some(4), 4, prompt),
            Some(8)
        );
        // Chat at the prompt end: invalid, shared fallback.
        assert_eq!(
            select_deferred_checkpoint_boundary(Some(8), Some(10), 0, prompt),
            Some(8)
        );
        // Chat beyond the prompt: invalid, shared fallback.
        assert_eq!(
            select_deferred_checkpoint_boundary(Some(8), Some(99), 0, prompt),
            Some(8)
        );
        // Shared already restored but chat valid: chat wins.
        assert_eq!(
            select_deferred_checkpoint_boundary(Some(4), Some(6), 4, prompt),
            Some(6)
        );
        // Both invalid: nothing selected.
        assert_eq!(
            select_deferred_checkpoint_boundary(Some(4), Some(10), 4, prompt),
            None
        );
        // Neither candidate exists: nothing selected.
        assert_eq!(
            select_deferred_checkpoint_boundary(None, None, 0, prompt),
            None
        );
    }
}
