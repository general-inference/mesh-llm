//! Opt-in proof against real selector/convolution weights and target features.
use super::*;
use crate::{ModelInfo, MtpSource};
use proposal_batch::DFlashBatchRequest;

struct ReferenceLanes {
    // Native sessions borrow the model: drop them before the owning target.
    sessions: Vec<StageSession>,
    anchors: Vec<i32>,
    _target: StageModel,
}

fn reference_lanes() -> Result<ReferenceLanes> {
    crate::test_support::require_native_runtime();
    let target_path = std::env::var_os("SKIPPY_DRAFT_LAYOUT_TARGET_MODEL")
        .context("set SKIPPY_DRAFT_LAYOUT_TARGET_MODEL")?;
    let draft_path = std::env::var_os("SKIPPY_DRAFT_LAYOUT_DRAFT_MODEL")
        .context("set SKIPPY_DRAFT_LAYOUT_DRAFT_MODEL")?;
    let target_path = Path::new(&target_path);
    let layers = ModelInfo::open(target_path)?
        .tensors()?
        .iter()
        .filter_map(|tensor| tensor.layer_index)
        .max()
        .context("target has no layers")?
        + 1;
    let config = RuntimeConfig {
        layer_end: layers,
        ctx_size: 8192,
        lane_count: 3,
        n_gpu_layers: 99,
        selected_backend_device: Some("MTL0".to_owned()),
        n_batch: Some(512),
        n_ubatch: Some(512),
        mtp_source: MtpSource::Disabled,
        ..RuntimeConfig::default()
    };
    let mut target = StageModel::open(target_path, &config)?;
    let info = target.attach_dflash_draft(
        Path::new(&draft_path),
        &DFlashDraftOptions {
            max_draft_tokens: Some(6),
            ..DFlashDraftOptions::default()
        },
        &config,
    )?;
    assert_eq!(info.variant, DFlashVariant::DFlash2);
    let mut sessions = Vec::new();
    let mut anchors = Vec::new();
    for prompt in [
        "Complete this sequence: one, two, three,",
        "Write a Python function that finds the longest increasing subsequence in a list of integers.",
        "Explain why a database transaction needs isolation when several clients update the same account balance, with a concrete example.",
    ] {
        let tokens = target.tokenize(prompt, true)?;
        let mut session = target.create_session()?;
        session.prefill_chunked(&tokens[..tokens.len() - 1])?;
        anchors.push(*tokens.last().context("empty prompt")?);
        sessions.push(session);
    }
    Ok(ReferenceLanes {
        _target: target,
        sessions,
        anchors,
    })
}

fn serial(lanes: &mut ReferenceLanes, widths: &[usize]) -> Result<Vec<Vec<i32>>> {
    lanes
        .sessions
        .iter_mut()
        .zip(&lanes.anchors)
        .zip(widths)
        .map(|((session, &anchor), &width)| {
            Ok(session
                .dflash_propose(anchor, width)?
                .context("draft is not attached")?
                .tokens)
        })
        .collect()
}

fn shared(lanes: &mut ReferenceLanes, widths: &[usize], reverse: bool) -> Result<Vec<Vec<i32>>> {
    let mut requests: Vec<_> = lanes
        .sessions
        .iter_mut()
        .zip(&lanes.anchors)
        .zip(widths)
        .map(|((session, &anchor), &max_tokens)| DFlashBatchRequest {
            session,
            anchor,
            max_tokens,
        })
        .collect();
    if reverse {
        requests.reverse();
    }
    let mut tokens = StageSession::dflash_propose_batch(&mut requests)?
        .into_iter()
        .map(|proposal| Ok(proposal.context("draft is not attached")?.tokens))
        .collect::<Result<Vec<_>>>()?;
    if reverse {
        tokens.reverse();
    }
    Ok(tokens)
}

#[test]
#[ignore = "requires local selector draft/target GGUF paths and a Metal runtime"]
fn selector_noise_blocks_match_independent_lanes_and_remove_scratch() -> Result<()> {
    let mut lanes = reference_lanes()?;
    for width in [1, 3, 6] {
        let widths = [width; 3];
        let expected = serial(&mut lanes, &widths)?;
        assert_eq!(shared(&mut lanes, &widths, false)?, expected);
        assert_eq!(shared(&mut lanes, &widths, true)?, expected);
        assert_eq!(serial(&mut lanes, &widths)?, expected);
    }
    let before: Vec<_> = lanes
        .sessions
        .iter()
        .map(|session| session.dflash_draft_stats().unwrap().shared_proposals)
        .collect();
    assert!(before.iter().all(|&count| count > 0));
    let widths = [1, 3, 6];
    let expected = serial(&mut lanes, &widths)?;
    assert_eq!(shared(&mut lanes, &widths, false)?, expected);
    let after: Vec<_> = lanes
        .sessions
        .iter()
        .map(|session| session.dflash_draft_stats().unwrap().shared_proposals)
        .collect();
    assert_eq!(
        after, before,
        "ragged selector blocks must run independently"
    );
    Ok(())
}
