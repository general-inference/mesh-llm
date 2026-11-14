//! Matched concurrent decode loops with independent or shared verification.
use std::{collections::BTreeMap, time::Instant};

use anyhow::{Context, Result, bail};
use serde::Serialize;
use skippy_runtime::dflash::proposal_batch::DFlashBatchRequest;
use skippy_runtime::{StageModel, StageSession, verification_batch::VerifyBatchRequest};

use crate::{Args, Baseline, PromptCase, elapsed_ms, first_mismatch, tokens_per_second};

struct Lane {
    session: StageSession,
    current: i32,
    tokens: Vec<i32>,
    accepted: usize,
    finished: bool,
}

#[derive(Serialize)]
struct Report {
    mode: &'static str,
    shared_verify: bool,
    shared_draft: bool,
    shared_draft_lane_proposals: u64,
    lane_count: usize,
    ctx_size: u32,
    window: usize,
    decode_wall_ms: f64,
    output_tokens: usize,
    output_tokens_per_second: f64,
    shared_forwards: usize,
    single_forwards: usize,
    device_feature_rows: u64,
    lanes: Vec<LaneReport>,
}

#[derive(Serialize)]
struct LaneReport {
    id: String,
    correct: bool,
    mismatch_index: Option<usize>,
    accepted_draft_tokens: usize,
    output_tokens: Vec<i32>,
    baseline_output_tokens: Vec<i32>,
    verify_windows_in_place: u64,
    verify_windows_checkpointed: u64,
    shared_draft_proposals: u64,
}

pub(super) fn run(
    args: &Args,
    target: &StageModel,
    prompts: &[PromptCase],
    baselines: &[Baseline],
) -> Result<()> {
    let mut lanes = Vec::new();
    for baseline in baselines {
        let mut session = target.create_session()?;
        let prefix = &baseline.target_tokens[..baseline.target_tokens.len() - 1];
        session.prefill_chunked(prefix)?;
        lanes.push(Lane {
            session,
            current: *baseline.target_tokens.last().expect("nonempty prompt"),
            tokens: Vec::new(),
            accepted: 0,
            finished: false,
        });
    }
    let mut shared_forwards = 0;
    let mut single_forwards = 0;
    let started = Instant::now();
    while lanes.iter().any(|lane| !lane.finished) {
        let inputs = propose(args, &mut lanes)?;
        let (predictions, shared, single) = verify(&mut lanes, &inputs, args.shared_verify)?;
        shared_forwards += shared;
        single_forwards += single;
        for ((lane, input), predicted) in lanes.iter_mut().zip(&inputs).zip(&predictions) {
            if !lane.finished {
                commit(target, args.max_new_tokens, lane, input, predicted)?;
            }
        }
    }
    let decode_wall_ms = elapsed_ms(started);
    let mut reports = Vec::new();
    for ((lane, prompt), baseline) in lanes.into_iter().zip(prompts).zip(baselines) {
        let mismatch_index = first_mismatch(&baseline.generation.tokens, &lane.tokens);
        let rollback = lane.session.verify_rollback_counts()?;
        reports.push(LaneReport {
            id: prompt.id.clone(),
            correct: mismatch_index.is_none(),
            mismatch_index,
            accepted_draft_tokens: lane.accepted,
            output_tokens: lane.tokens,
            baseline_output_tokens: baseline.generation.tokens.clone(),
            verify_windows_in_place: rollback.in_place,
            verify_windows_checkpointed: rollback.checkpointed,
            shared_draft_proposals: lane
                .session
                .dflash_draft_stats()
                .map_or(0, |stats| stats.shared_proposals),
        });
    }
    let output_tokens = reports
        .iter()
        .map(|report| report.output_tokens.len())
        .sum();
    let report = Report {
        mode: "concurrent DFlash diagnostic; complete post-prefill loop",
        shared_verify: args.shared_verify,
        shared_draft: args.shared_draft,
        shared_draft_lane_proposals: reports.iter().map(|lane| lane.shared_draft_proposals).sum(),
        lane_count: args.dflash_lanes,
        ctx_size: args.ctx_size,
        window: args.speculative_window,
        decode_wall_ms,
        output_tokens,
        output_tokens_per_second: tokens_per_second(output_tokens, decode_wall_ms),
        shared_forwards,
        single_forwards,
        device_feature_rows: target.dflash_device_feature_rows(),
        lanes: reports,
    };
    eprintln!(
        "concurrent decode: {:.2} tok/s shared={} single={}",
        report.output_tokens_per_second, shared_forwards, single_forwards
    );
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.json_out {
        std::fs::write(path, format!("{json}\n"))?;
    } else {
        println!("{json}");
    }
    if !args.allow_mismatch && report.lanes.iter().any(|lane| !lane.correct) {
        bail!("concurrent verification diverged from serial target output");
    }
    Ok(())
}

fn propose(args: &Args, lanes: &mut [Lane]) -> Result<Vec<Vec<i32>>> {
    if args.shared_draft {
        let mut requests: Vec<_> = lanes
            .iter_mut()
            .filter(|lane| !lane.finished)
            .map(|lane| DFlashBatchRequest {
                session: &mut lane.session,
                anchor: lane.current,
                max_tokens: args
                    .speculative_window
                    .min((args.max_new_tokens - lane.tokens.len()).saturating_sub(1)),
            })
            .collect();
        let outputs = StageSession::dflash_propose_batch(&mut requests)?;
        let mut outputs = outputs.into_iter();
        return lanes
            .iter()
            .map(|lane| {
                if lane.finished {
                    return Ok(Vec::new());
                }
                let proposal = outputs
                    .next()
                    .flatten()
                    .context("concurrent target has no DFlash draft")?;
                let mut input = vec![lane.current];
                input.extend(proposal.tokens);
                Ok(input)
            })
            .collect();
    }
    lanes
        .iter_mut()
        .map(|lane| {
            if lane.finished {
                return Ok(Vec::new());
            }
            let remaining = args.max_new_tokens - lane.tokens.len();
            let proposal = lane
                .session
                .dflash_propose(
                    lane.current,
                    args.speculative_window.min(remaining.saturating_sub(1)),
                )?
                .context("concurrent target has no DFlash draft")?;
            let mut input = vec![lane.current];
            input.extend(proposal.tokens);
            Ok(input)
        })
        .collect()
}

fn verify(
    lanes: &mut [Lane],
    inputs: &[Vec<i32>],
    shared: bool,
) -> Result<(Vec<Vec<i32>>, usize, usize)> {
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, input) in inputs
        .iter()
        .enumerate()
        .filter(|(_, input)| !input.is_empty())
    {
        groups.entry(input.len()).or_default().push(index);
    }
    let mut predictions = vec![Vec::new(); lanes.len()];
    let mut shared_forwards = 0;
    let mut single_forwards = 0;
    for (width, indexes) in groups {
        if shared && indexes.len() > 1 {
            let mut requests: Vec<_> = lanes
                .iter_mut()
                .zip(inputs)
                .filter(|(_, input)| input.len() == width)
                .map(|(lane, input)| VerifyBatchRequest {
                    session: &mut lane.session,
                    token_ids: input,
                    sampling: None,
                })
                .collect();
            let outputs = StageSession::verify_batch_sampled(&mut requests)?;
            for (index, output) in indexes.into_iter().zip(outputs) {
                predictions[index] = output;
            }
            shared_forwards += 1;
        } else {
            for index in indexes {
                predictions[index] = lanes[index]
                    .session
                    .verify_tokens_sampled(&inputs[index], None)?;
                single_forwards += 1;
            }
        }
    }
    Ok((predictions, shared_forwards, single_forwards))
}

fn commit(
    target: &StageModel,
    limit: usize,
    lane: &mut Lane,
    inputs: &[i32],
    predicted: &[i32],
) -> Result<()> {
    let accepted = inputs[1..]
        .iter()
        .zip(predicted)
        .take_while(|(draft, target)| draft == target)
        .count();
    let base = lane.session.token_count() - inputs.len() as u64;
    let mut committed = 0;
    for &token in predicted.iter().take(accepted + 1) {
        lane.tokens.push(token);
        lane.current = token;
        committed += 1;
        if target.token_is_eog(token)? || lane.tokens.len() == limit {
            lane.finished = true;
            break;
        }
    }
    anyhow::ensure!(committed > 0, "verification produced no committed token");
    lane.accepted += accepted.min(committed);
    if committed == inputs.len() {
        lane.session
            .retire_verify_checkpoint(base, inputs.len() as u64)?;
    } else {
        lane.session.trim_session(base + committed as u64)?;
    }
    Ok(())
}
