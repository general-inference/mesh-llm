//! Teacher-forced target verification isolates batch numerics from rollback.
use anyhow::{Context, Result, bail, ensure};
use serde::Serialize;
use skippy_runtime::{SamplingConfig, StageModel, StageSession, TokenLogprobs};

use crate::{Args, Baseline};

#[derive(Serialize)]
struct Report {
    mode: &'static str,
    window: usize,
    lane_count: usize,
    shared_verify: bool,
    shared_forwards: usize,
    draft_attached: bool,
    prompts: Vec<Prompt>,
}

#[derive(Serialize)]
struct Prompt {
    baseline_tokens: Vec<i32>,
    serial_tokens: Vec<i32>,
    batch_tokens: Vec<i32>,
    differences: Vec<Difference>,
}

#[derive(Serialize)]
struct Difference {
    index: usize,
    serial: Row,
    batch: Row,
}

#[derive(Serialize)]
struct Row {
    token: i32,
    logprob: f32,
    top: Vec<(i32, f32)>,
}

impl From<&TokenLogprobs> for Row {
    fn from(row: &TokenLogprobs) -> Self {
        Self {
            token: row.token,
            logprob: row.logprob,
            top: row.top.clone(),
        }
    }
}

pub(super) fn run(args: &Args, target: &StageModel, baselines: &[Baseline]) -> Result<()> {
    let width = args
        .speculative_window
        .checked_add(1)
        .context("verify window overflow")?;
    let mut prompts = Vec::new();
    let (mut shared_trace, shared_forwards) = if args.shared_verify {
        let (rows, forwards) = trace_shared(target, baselines, width)?;
        (Some(rows), forwards)
    } else {
        (None, 0)
    };
    for (index, baseline) in baselines.iter().enumerate() {
        let mut inputs = vec![*baseline.target_tokens.last().expect("nonempty prompt")];
        inputs.extend_from_slice(&baseline.generation.tokens);
        inputs.truncate(baseline.generation.tokens.len());
        let serial = trace(target, baseline, &inputs, 1)?;
        let batch = if let Some(traces) = &mut shared_trace {
            std::mem::take(&mut traces[index])
        } else {
            trace(target, baseline, &inputs, width)?
        };
        let differences: Vec<_> = serial
            .iter()
            .zip(&batch)
            .enumerate()
            .filter(|(_, (serial, batch))| serial.token != batch.token)
            .map(|(index, (serial, batch))| Difference {
                index,
                serial: serial.into(),
                batch: batch.into(),
            })
            .collect();
        eprintln!("teacher-forced differences={}", differences.len());
        prompts.push(Prompt {
            baseline_tokens: baseline.generation.tokens.clone(),
            serial_tokens: serial.iter().map(|row| row.token).collect(),
            batch_tokens: batch.iter().map(|row| row.token).collect(),
            differences,
        });
    }
    let report = Report {
        mode: "teacher-forced target verification; no draft proposals or rollback",
        window: width,
        lane_count: args.dflash_lanes,
        shared_verify: args.shared_verify,
        shared_forwards,
        draft_attached: args.dflash,
        prompts,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.json_out {
        std::fs::write(path, format!("{json}\n"))?;
    } else {
        println!("{json}");
    }
    if !args.allow_mismatch
        && report.prompts.iter().any(|prompt| {
            !prompt.differences.is_empty() || prompt.serial_tokens != prompt.baseline_tokens
        })
    {
        bail!("teacher-forced verification diverged from the serial baseline");
    }
    Ok(())
}

/// Forces identical continuations through a shared target without attaching
/// any draft, so cross-sequence kernel numerics can be inspected directly.
fn trace_shared(
    target: &StageModel,
    baselines: &[Baseline],
    width: usize,
) -> Result<(Vec<Vec<TokenLogprobs>>, usize)> {
    use skippy_runtime::{StageSession, verification_batch::VerifyBatchRequest};
    let length = baselines
        .first()
        .context("no reference lanes")?
        .generation
        .tokens
        .len();
    ensure!(
        baselines
            .iter()
            .all(|baseline| baseline.generation.tokens.len() == length),
        "shared reference tracing requires equal continuation lengths"
    );
    let mut sessions = Vec::new();
    let mut inputs = Vec::new();
    for baseline in baselines {
        let mut session = target.create_session()?;
        session.prefill_chunked(&baseline.target_tokens[..baseline.target_tokens.len() - 1])?;
        sessions.push(session);
        let mut tokens = vec![
            *baseline
                .target_tokens
                .last()
                .context("empty reference prompt")?,
        ];
        tokens.extend_from_slice(&baseline.generation.tokens);
        tokens.truncate(length);
        inputs.push(tokens);
    }
    let sampling = SamplingConfig {
        enabled: true,
        temperature: 0.0,
        top_logprobs: Some(5),
        ..SamplingConfig::default()
    };
    let mut rows = vec![Vec::new(); baselines.len()];
    let mut forwards = 0;
    for offset in (0..length).step_by(width) {
        let end = (offset + width).min(length);
        let bases: Vec<_> = sessions.iter().map(StageSession::token_count).collect();
        let mut requests: Vec<_> = sessions
            .iter_mut()
            .zip(&inputs)
            .map(|(session, inputs)| VerifyBatchRequest {
                session,
                token_ids: &inputs[offset..end],
                sampling: Some(&sampling),
            })
            .collect();
        StageSession::verify_batch_sampled(&mut requests)?;
        forwards += 1;
        for (index, request) in requests.into_iter().enumerate() {
            let chunk_rows = request.session.take_token_logprobs()?;
            ensure!(
                !chunk_rows.is_empty() && chunk_rows.len() <= end - offset,
                "missing shared trace logprobs"
            );
            rows[index].extend(chunk_rows);
            request
                .session
                .retire_verify_checkpoint(bases[index], (end - offset) as u64)?;
        }
        if rows.iter().zip(baselines).any(|(rows, baseline)| {
            rows.iter()
                .zip(&baseline.generation.tokens)
                .any(|(row, &token)| row.token != token)
        }) {
            break;
        }
    }
    Ok((rows, forwards))
}

fn trace(
    target: &StageModel,
    baseline: &Baseline,
    inputs: &[i32],
    width: usize,
) -> Result<Vec<TokenLogprobs>> {
    let mut session = target.create_session()?;
    ensure!(
        width <= session.batch_size()?,
        "verify window exceeds native batch size"
    );
    session.prefill_chunked(&baseline.target_tokens[..baseline.target_tokens.len() - 1])?;
    let sampling = SamplingConfig {
        enabled: true,
        temperature: 0.0,
        top_logprobs: Some(5),
        ..SamplingConfig::default()
    };
    let mut rows = Vec::new();
    for chunk in inputs.chunks(width) {
        trace_chunk(&mut session, chunk, &sampling, width == 1)?;
        let chunk_rows = session.take_token_logprobs()?;
        ensure!(
            !chunk_rows.is_empty() && chunk_rows.len() <= chunk.len(),
            "missing trace logprobs"
        );
        rows.extend(chunk_rows);
        // Stateful sampling stops at the first rejected proposal. These are
        // deliberately forced inputs, but stop the trace rather than sample
        // or fabricate probabilities for the unreported suffix.
        if rows
            .iter()
            .zip(&baseline.generation.tokens)
            .any(|(row, &token)| row.token != token)
        {
            break;
        }
    }
    Ok(rows)
}

fn trace_chunk(
    session: &mut StageSession,
    chunk: &[i32],
    sampling: &SamplingConfig,
    serial: bool,
) -> Result<()> {
    if serial {
        session.decode_step_sampled(chunk[0], Some(sampling))?;
    } else {
        let start = session.token_count();
        session.verify_tokens_sampled(chunk, Some(sampling))?;
        // Every forced input is committed, irrespective of the prediction.
        session.retire_verify_checkpoint(start, chunk.len() as u64)?;
    }
    Ok(())
}
