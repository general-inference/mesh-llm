//! Isolate the four-lane hybrid prefill shape from the serving scheduler.
use super::native_decode_shapes::{iterate, prefill_without_intermediate_sampling};
use super::*;
use crate::runtime_state::RuntimeIterationBatchRequest;
use anyhow::Context;
use skippy_runtime::IterationBatchPhase;

#[derive(Debug, Clone, Copy)]
enum PrefillOrder {
    Serial,
    Interleaved,
    InterleavedFull,
    InterleavedSmall,
    Batched,
    BatchedWide,
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_four_lane_serial_prefill_matches_serial_decode() -> Result<()> {
    compare_four_lane_to_serial(PrefillOrder::Serial)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_four_lane_interleaved_prefill_matches_serial_decode() -> Result<()> {
    compare_four_lane_to_serial(PrefillOrder::Interleaved)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_four_lane_production_width_prefill_matches_serial_decode() -> Result<()> {
    compare_four_lane_to_serial(PrefillOrder::InterleavedFull)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_four_lane_interleaved_small_prefill_matches_serial_decode() -> Result<()> {
    compare_four_lane_to_serial(PrefillOrder::InterleavedSmall)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_four_lane_batched_prefill_matches_serial_decode() -> Result<()> {
    compare_four_lane_to_serial(PrefillOrder::Batched)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_four_lane_batched_wide_prefill_matches_serial_decode() -> Result<()> {
    compare_four_lane_to_serial(PrefillOrder::BatchedWide)
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_four_lane_batched_prefill_host_sampler_matches_serial_decode() -> Result<()> {
    let (parallel, _) = recurrent_test_backend_with_lanes(
        "native-host-four",
        "native-host-four",
        256,
        (32, 32, 4),
        8,
        1024,
        None,
    )?;
    let (serial, _) =
        recurrent_test_backend("native-host-one", "native-host-one", 256, (32, 32), 8, 256)?;
    // Requesting native logprobs retires the in-graph backend sampler while
    // leaving the same greedy token policy and four-request model batch.
    let sampling = SamplingConfig {
        top_logprobs: Some(0),
        ..SamplingConfig::default()
    };
    let prompts: [_; 4] = std::array::from_fn(|lane| {
        let mut prompt: Vec<_> = (1..=176).map(|token| token + 257 * lane as i32).collect();
        prompt.extend_from_slice(&[501, 502, 503, 504]);
        prompt
    });
    let sessions = [
        "native-host-0",
        "native-host-1",
        "native-host-2",
        "native-host-3",
    ];
    let mut parallel_runtime = parallel.runtime.lock().unwrap();
    let mut starts = Vec::new();
    for chunk in 0..prompts[0].len().div_ceil(8) {
        let start = chunk * 8;
        let end = (start + 8).min(prompts[0].len());
        let inputs: Vec<_> = prompts.iter().map(|prompt| &prompt[start..end]).collect();
        let sample_last = end == prompts[0].len();
        let next = iterate_configured(
            &mut parallel_runtime,
            &sessions,
            &inputs,
            &sampling,
            IterationBatchPhase::Prefill,
            sample_last,
        )?;
        if sample_last {
            starts = next;
        }
    }
    let mut serial_runtime = serial.runtime.lock().unwrap();
    let mut serial_token = 0;
    for chunk in prompts[3].chunks(32) {
        let next = iterate_configured(
            &mut serial_runtime,
            &[sessions[3]],
            &[chunk],
            &sampling,
            IterationBatchPhase::Prefill,
            chunk.len() < 32,
        )?;
        if let Some(&token) = next.first() {
            serial_token = token;
        }
    }
    anyhow::ensure!(starts.len() == 4 && starts[3] == serial_token);
    for step in 1..8 {
        let inputs: Vec<_> = starts.iter().map(std::slice::from_ref).collect();
        starts = iterate_configured(
            &mut parallel_runtime,
            &sessions,
            &inputs,
            &sampling,
            IterationBatchPhase::Decode,
            true,
        )?;
        let parallel_signal = parallel_runtime.last_token_signal(sessions[3])?;
        serial_token = iterate_configured(
            &mut serial_runtime,
            &[sessions[3]],
            &[&[serial_token]],
            &sampling,
            IterationBatchPhase::Decode,
            true,
        )?[0];
        let serial_signal = serial_runtime.last_token_signal(sessions[3])?;
        assert_eq!(
            starts[3], serial_token,
            "host sampler c4 diverged at token {step}; parallel={parallel_signal:?}, serial={serial_signal:?}"
        );
    }
    Ok(())
}

fn iterate_configured(
    runtime: &mut RuntimeState,
    sessions: &[&str],
    tokens: &[&[i32]],
    sampling: &SamplingConfig,
    phase: IterationBatchPhase,
    sample_last: bool,
) -> Result<Vec<i32>> {
    let requests: Vec<_> = sessions
        .iter()
        .zip(tokens)
        .map(|(session_id, token_ids)| RuntimeIterationBatchRequest {
            session_id,
            token_ids,
            positions: &[],
            sampling: sample_last.then_some(sampling),
            input: None,
            sample_last,
            phase,
        })
        .collect();
    let output = runtime.iteration_batch_sampled(&requests)?;
    if !sample_last {
        return Ok(Vec::new());
    }
    let mut predicted = vec![None; sessions.len()];
    for sample in output.samples {
        predicted[sample.request_index] = Some(sample.predicted_token);
    }
    predicted
        .into_iter()
        .map(|token| token.context("native request did not sample"))
        .collect()
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn native_single_lane_prefill_chunk_width_preserves_decode() -> Result<()> {
    let wide = decode_single_lane(32)?;
    let narrow = decode_single_lane(8)?;
    assert_eq!(
        narrow.0, wide.0,
        "single-lane 8-token prefill changed decode; narrow={:?}, wide={:?}",
        narrow.1, wide.1
    );
    Ok(())
}

fn decode_single_lane(chunk_size: usize) -> Result<(Vec<i32>, skippy_runtime::TokenSignal)> {
    let (backend, _) =
        recurrent_test_backend("native-one-tail", "native-one-tail", 256, (32, 32), 8, 256)?;
    let mut prompt: Vec<_> = (1..=176).map(|token| token + 257 * 3).collect();
    prompt.extend_from_slice(&[501, 502, 503, 504]);
    let session = "native-one-tail";
    let mut runtime = backend.runtime.lock().unwrap();
    let mut first = None;
    for (index, chunk) in prompt.chunks(chunk_size).enumerate() {
        first = prefill_without_intermediate_sampling(
            &mut runtime,
            session,
            chunk,
            index + 1 == prompt.len().div_ceil(chunk_size),
        )?
        .or(first);
    }
    let mut output = vec![first.context("single-lane prefill did not sample")?];
    while output.len() < 8 {
        let previous = *output.last().unwrap();
        output.extend(iterate(
            &mut runtime,
            &[session],
            &[&[previous]],
            IterationBatchPhase::Decode,
        )?);
    }
    Ok((output, runtime.last_token_signal(session)?))
}

fn compare_four_lane_to_serial(order: PrefillOrder) -> Result<()> {
    let (batch, microbatch) = if matches!(order, PrefillOrder::InterleavedFull) {
        (2048, 512)
    } else if matches!(order, PrefillOrder::BatchedWide) {
        (128, 128)
    } else {
        (32, 32)
    };
    let (parallel, _) = recurrent_test_backend_with_lanes(
        "native-four-tail",
        "native-four-tail",
        256,
        (batch, microbatch, 4),
        8,
        1024,
        None,
    )?;
    let (serial, _) = recurrent_test_backend(
        "native-serial-tail",
        "native-serial-tail",
        256,
        (batch, microbatch),
        8,
        256,
    )?;
    let prompts: [_; 4] = std::array::from_fn(|lane| {
        let mut prompt: Vec<_> = (1..=176).map(|token| token + 257 * lane as i32).collect();
        prompt.extend_from_slice(&[501, 502, 503, 504]);
        prompt
    });
    let sessions = [
        "native-tail-0",
        "native-tail-1",
        "native-tail-2",
        "native-tail-3",
    ];
    let mut parallel_runtime = parallel.runtime.lock().unwrap();
    let mut starts = prefill_parallel(&mut parallel_runtime, &sessions, &prompts, order)?;
    let mut serial_runtime = serial.runtime.lock().unwrap();
    let mut serial_token = if matches!(order, PrefillOrder::InterleavedFull) {
        prefill_without_intermediate_sampling(&mut serial_runtime, sessions[3], &prompts[3], true)?
            .context("production-width serial prefill did not sample")?
    } else {
        prefill_one(&mut serial_runtime, sessions[3], &prompts[3])?
    };
    anyhow::ensure!(
        starts[3] == serial_token,
        "{order:?} first token differs: parallel={}, serial={serial_token}",
        starts[3]
    );
    for step in 1..8 {
        let parallel_inputs = starts.map(|token| [token]);
        let parallel_tokens: Vec<_> = parallel_inputs
            .iter()
            .map(|input| input.as_slice())
            .collect();
        let next = iterate(
            &mut parallel_runtime,
            &sessions,
            &parallel_tokens,
            IterationBatchPhase::Decode,
        )?;
        starts.copy_from_slice(&next);
        // Lane 3 is the last native output row in this batched iteration.
        let parallel_signal = parallel_runtime.last_token_signal(sessions[3])?;
        serial_token = iterate(
            &mut serial_runtime,
            &[sessions[3]],
            &[&[serial_token]],
            IterationBatchPhase::Decode,
        )?[0];
        let serial_signal = serial_runtime.last_token_signal(sessions[3])?;
        assert_eq!(
            starts[3], serial_token,
            "{order:?} c4 lane 3 diverged at token {step}; parallel={parallel_signal:?}, serial={serial_signal:?}"
        );
    }
    Ok(())
}

fn prefill_parallel(
    runtime: &mut RuntimeState,
    sessions: &[&str; 4],
    prompts: &[Vec<i32>; 4],
    order: PrefillOrder,
) -> Result<[i32; 4]> {
    let mut starts = [0; 4];
    let batched_chunk_size = if matches!(order, PrefillOrder::BatchedWide) {
        32
    } else {
        8
    };
    let chunks = prompts[0].len().div_ceil(32);
    match order {
        PrefillOrder::Serial => {
            for lane in 0..4 {
                for chunk in 0..chunks {
                    prefill_chunk(runtime, sessions, prompts, &mut starts, lane, chunk, 32)?;
                }
            }
        }
        PrefillOrder::Interleaved => {
            for chunk in 0..chunks {
                for lane in 0..4 {
                    prefill_chunk(runtime, sessions, prompts, &mut starts, lane, chunk, 32)?;
                }
            }
        }
        PrefillOrder::InterleavedFull => {
            for lane in 0..4 {
                starts[lane] = prefill_without_intermediate_sampling(
                    runtime,
                    sessions[lane],
                    &prompts[lane],
                    true,
                )?
                .context("production-width parallel prefill did not sample")?;
            }
        }
        PrefillOrder::InterleavedSmall => {
            for chunk in 0..prompts[0].len().div_ceil(batched_chunk_size) {
                for lane in 0..4 {
                    prefill_chunk(
                        runtime,
                        sessions,
                        prompts,
                        &mut starts,
                        lane,
                        chunk,
                        batched_chunk_size,
                    )?;
                }
            }
        }
        PrefillOrder::Batched | PrefillOrder::BatchedWide => {
            let batched_chunks = prompts[0].len().div_ceil(batched_chunk_size);
            for chunk in 0..batched_chunks {
                let start = chunk * batched_chunk_size;
                let end = (start + batched_chunk_size).min(prompts[0].len());
                let sample_last = chunk + 1 == batched_chunks;
                let sampling = SamplingConfig::default();
                let requests: Vec<_> = sessions
                    .iter()
                    .zip(prompts)
                    .map(|(session_id, prompt)| RuntimeIterationBatchRequest {
                        session_id,
                        token_ids: &prompt[start..end],
                        positions: &[],
                        sampling: sample_last.then_some(&sampling),
                        input: None,
                        sample_last,
                        phase: IterationBatchPhase::Prefill,
                    })
                    .collect();
                let output = runtime.iteration_batch_sampled(&requests)?;
                for sample in output.samples {
                    starts[sample.request_index] = sample.predicted_token;
                }
            }
        }
    }
    Ok(starts)
}

fn prefill_chunk(
    runtime: &mut RuntimeState,
    sessions: &[&str; 4],
    prompts: &[Vec<i32>; 4],
    starts: &mut [i32; 4],
    lane: usize,
    chunk: usize,
    chunk_size: usize,
) -> Result<()> {
    let start = chunk * chunk_size;
    let end = (start + chunk_size).min(prompts[lane].len());
    if let Some(token) = prefill_without_intermediate_sampling(
        runtime,
        sessions[lane],
        &prompts[lane][start..end],
        end == prompts[lane].len(),
    )? {
        starts[lane] = token;
    }
    Ok(())
}

fn prefill_one(runtime: &mut RuntimeState, session: &str, prompt: &[i32]) -> Result<i32> {
    let mut sampled = None;
    for chunk in prompt.chunks(32) {
        sampled = prefill_without_intermediate_sampling(runtime, session, chunk, chunk.len() < 32)?
            .or(sampled);
    }
    sampled.context("serial prefill did not sample")
}
