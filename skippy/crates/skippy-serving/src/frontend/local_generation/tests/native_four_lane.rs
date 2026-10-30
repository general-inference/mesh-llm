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
    Batched,
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
fn native_four_lane_batched_prefill_matches_serial_decode() -> Result<()> {
    compare_four_lane_to_serial(PrefillOrder::Batched)
}

fn compare_four_lane_to_serial(order: PrefillOrder) -> Result<()> {
    let (parallel, _) = recurrent_test_backend_with_lanes(
        "native-four-tail",
        "native-four-tail",
        256,
        (32, 32, 4),
        8,
        1024,
        None,
    )?;
    let (serial, _) = recurrent_test_backend(
        "native-serial-tail",
        "native-serial-tail",
        256,
        (32, 32),
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
    let mut serial_token = prefill_one(&mut serial_runtime, sessions[3], &prompts[3])?;
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
    let batched_chunk_size = 8;
    let chunks = prompts[0].len().div_ceil(32);
    match order {
        PrefillOrder::Serial => {
            for lane in 0..4 {
                for chunk in 0..chunks {
                    prefill_chunk(runtime, sessions, prompts, &mut starts, lane, chunk)?;
                }
            }
        }
        PrefillOrder::Interleaved => {
            for chunk in 0..chunks {
                for lane in 0..4 {
                    prefill_chunk(runtime, sessions, prompts, &mut starts, lane, chunk)?;
                }
            }
        }
        PrefillOrder::Batched => {
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
) -> Result<()> {
    let start = chunk * 32;
    let end = (start + 32).min(prompts[lane].len());
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
