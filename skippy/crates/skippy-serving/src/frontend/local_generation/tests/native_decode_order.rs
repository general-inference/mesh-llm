//! Verify that changing native batch row order does not change a live hybrid continuation.
use super::native_decode_shapes::{iterate, prefill_without_intermediate_sampling};
use super::*;
use crate::frontend::iteration_scheduler::ScheduledResumeRequest;
use skippy_runtime::IterationBatchPhase;

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn native_four_lane_decode_preserves_state_across_batch_reordering() -> Result<()> {
    let fixed = decode_with_order_changes(false, HandoffFixture::large())?;
    let reordered = decode_with_order_changes(true, HandoffFixture::large())?;
    for lane in 0..4 {
        assert_eq!(
            reordered[lane], fixed[lane],
            "native c=4 continuation changed after reordering the batch: lane={lane}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL, _LAYERS and optional _GPU_LAYERS"]
fn four_lane_planned_handoff_matches_direct_decode() -> Result<()> {
    compare_handoff(HandoffFixture::large())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS"]
fn four_lane_small_context_handoff_matches_direct_decode() -> Result<()> {
    compare_handoff(HandoffFixture::small())
}

#[derive(Clone, Copy)]
struct HandoffFixture {
    context: usize,
    batch: u32,
    pool: usize,
    prompt_len: usize,
    changed_tail: bool,
}

impl HandoffFixture {
    fn large() -> Self {
        Self {
            context: 1024,
            batch: 128,
            pool: 4096,
            prompt_len: 256,
            changed_tail: false,
        }
    }

    fn small() -> Self {
        Self {
            context: 256,
            batch: 32,
            pool: 1024,
            prompt_len: 180,
            changed_tail: true,
        }
    }

    fn prompt(self, lane: usize) -> Vec<i32> {
        let count = if self.changed_tail {
            176
        } else {
            self.prompt_len
        };
        let mut prompt: Vec<_> = (1..=count)
            .map(|token| token as i32 + lane as i32 * 257)
            .collect();
        if self.changed_tail {
            prompt.extend_from_slice(&[501, 502, 503, 504]);
        }
        prompt
    }
}

fn compare_handoff(fixture: HandoffFixture) -> Result<()> {
    let direct = decode_with_order_changes(false, fixture)?;
    let planned = decode_with_scheduler_handoff(fixture)?;
    for lane in 0..4 {
        assert_eq!(
            planned[lane], direct[lane],
            "planned c=4 continuation changed after direct-to-planned handoff: lane={lane}"
        );
    }
    Ok(())
}

fn decode_with_order_changes(reorder: bool, fixture: HandoffFixture) -> Result<[Vec<i32>; 4]> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "batch-order",
        "batch-order",
        fixture.context,
        (fixture.batch, 32, 4),
        8,
        fixture.pool,
        None,
    )?;
    let mut runtime = backend.runtime.lock().unwrap();
    let sessions = [
        "batch-order-0",
        "batch-order-1",
        "batch-order-2",
        "batch-order-3",
    ];
    let mut output: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::new());
    for lane in 0..4 {
        let prompt = fixture.prompt(lane);
        for (index, chunk) in prompt.chunks(32).enumerate() {
            if let Some(token) = prefill_without_intermediate_sampling(
                &mut runtime,
                sessions[lane],
                chunk,
                index + 1 == prompt.len().div_ceil(32),
            )? {
                output[lane].push(token);
            }
        }
        assert_eq!(output[lane].len(), 1);
    }

    for step in 1..8 {
        let order = if reorder && step % 2 == 0 {
            [3, 1, 0, 2]
        } else {
            [0, 1, 2, 3]
        };
        let names: Vec<_> = order.iter().map(|&lane| sessions[lane]).collect();
        let inputs: Vec<_> = order
            .iter()
            .map(|&lane| [*output[lane].last().unwrap()])
            .collect();
        let tokens: Vec<_> = inputs.iter().map(|input| input.as_slice()).collect();
        let next = iterate(&mut runtime, &names, &tokens, IterationBatchPhase::Decode)?;
        for (lane, token) in order.into_iter().zip(next) {
            output[lane].push(token);
        }
    }
    Ok(output)
}

fn decode_with_scheduler_handoff(fixture: HandoffFixture) -> Result<[Vec<i32>; 4]> {
    let (backend, _) = recurrent_test_backend_with_lanes(
        "batch-handoff",
        "batch-handoff",
        fixture.context,
        (fixture.batch, 32, 4),
        8,
        fixture.pool,
        None,
    )?;
    let sessions = [
        "batch-handoff-0",
        "batch-handoff-1",
        "batch-handoff-2",
        "batch-handoff-3",
    ];
    let prompts: [Vec<i32>; 4] = std::array::from_fn(|lane| fixture.prompt(lane));
    let mut output: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::new());
    {
        let mut runtime = backend.runtime.lock().unwrap();
        for lane in 0..4 {
            for (index, chunk) in prompts[lane].chunks(32).enumerate() {
                if let Some(token) = prefill_without_intermediate_sampling(
                    &mut runtime,
                    sessions[lane],
                    chunk,
                    index + 1 == prompts[lane].len().div_ceil(32),
                )? {
                    output[lane].push(token);
                }
            }
        }
        let inputs: Vec<_> = output.iter().map(|tokens| [tokens[0]]).collect();
        let tokens: Vec<_> = inputs.iter().map(|input| input.as_slice()).collect();
        let next = iterate(
            &mut runtime,
            &sessions,
            &tokens,
            IterationBatchPhase::Decode,
        )?;
        for (lane, token) in next.into_iter().enumerate() {
            output[lane].push(token);
        }
    }
    let resumed = std::thread::scope(|scope| {
        let handles = (0..4)
            .map(|lane| {
                let mut generated = output[lane].clone();
                let sampling = SamplingConfig::default();
                let scheduler = &backend.iteration_scheduler;
                let prompt = &prompts[lane];
                scope.spawn(move || -> Result<Vec<i32>> {
                    scheduler.resume_generation(
                        ScheduledResumeRequest {
                            id: sessions[lane],
                            prompt_tokens: prompt,
                            generated_tokens: &generated.clone(),
                            max_tokens: 8,
                            sampling: Some(&sampling),
                            cancellation: None,
                        },
                        |token| {
                            generated.push(token);
                            Ok(TokenControl::Continue)
                        },
                    )?;
                    Ok(generated)
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("planned decoder thread panicked"))
            .collect::<Result<Vec<_>>>()
    })?;
    output = resumed.try_into().expect("four resumed lanes");
    Ok(output)
}
