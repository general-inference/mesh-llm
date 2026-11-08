//! Real-model coverage for append-only conversations: every request in a
//! tool-call loop or a plain chat must restore at least the previous
//! request's message history (its prompt before the assistant marker), even
//! when the requests arrive back to back.

use std::fmt::Write as _;

use super::tests::recurrent_test_backend;
use super::token_generation::capture_trace;
use crate::frontend::generation::StageOpenAiBackend;
use anyhow::Result;
use skippy_inference_api::{ChatCompletionRequest, InferenceBackend};

const LOOP_STEPS: usize = 6;

#[derive(Clone, Copy)]
enum LoopShape {
    /// system, user, then an assistant tool call and its tool result per step.
    ToolCalls,
    /// system, user, then an assistant reply and a new user message per step.
    PlainChat,
}

/// One step of the loop as the client saw it.
struct LoopStep {
    prompt_tokens: usize,
    boundary_tokens: usize,
    cached_tokens: usize,
    stored: Vec<usize>,
}

fn filler(step: usize) -> String {
    // Deterministic, step-specific text large enough that re-prefilling it
    // is visible in the token counts.
    (0..48)
        .map(|line| format!("file_{step}.rs:{line}: let value_{line} = compute({step}, {line});"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn loop_messages(shape: LoopShape, steps: usize) -> serde_json::Value {
    let mut messages = vec![
        serde_json::json!({"role": "system", "content": "You are a coding agent."}),
        serde_json::json!({"role": "user", "content": "Read the files one at a time and summarize them."}),
    ];
    for step in 0..steps {
        let call_id = format!("call_{step}");
        match shape {
            LoopShape::ToolCalls => {
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": format!("{{\"path\":\"file_{step}.rs\"}}")
                        }
                    }]
                }));
                messages.push(serde_json::json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": filler(step)
                }));
            }
            LoopShape::PlainChat => {
                messages.push(serde_json::json!({
                    "role": "assistant",
                    "content": format!("Summary {step}: the file computes values.")
                }));
                messages.push(serde_json::json!({
                    "role": "user",
                    "content": format!("Here is the next file:\n{}", filler(step))
                }));
            }
        }
    }
    serde_json::Value::Array(messages)
}

fn loop_request(shape: LoopShape, steps: usize) -> Result<ChatCompletionRequest> {
    let mut request = serde_json::json!({
        "model": "recurrent-agent-loop-test",
        "messages": loop_messages(shape, steps),
        "max_tokens": 8,
        "temperature": 0,
        "reasoning_effort": "none",
        "prompt_cache_key": "recurrent-agent-loop"
    });
    if matches!(shape, LoopShape::ToolCalls) {
        request["tools"] = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]
                }
            }
        }]);
    }
    Ok(serde_json::from_value(request)?)
}

async fn run_loop_step(
    backend: &StageOpenAiBackend,
    shape: LoopShape,
    step: usize,
) -> Result<LoopStep> {
    let request = loop_request(shape, step)?;
    let prompt = backend.prepare_chat_prompt(
        &request,
        crate::frontend::request::chat_template_options(
            &request,
            &backend.request_defaults.snapshot(),
        )?,
    )?;
    let prompt_tokens = backend.tokenize(&prompt.text)?;
    let boundary_tokens = backend.tokenize(
        prompt
            .recurrent_cache_prefix_text
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("step {step} had no message-history boundary"))?,
    )?;
    anyhow::ensure!(
        prompt_tokens.starts_with(&boundary_tokens),
        "step {step} boundary is not a token prefix of its prompt"
    );
    let response = backend.chat_completion(request).await?;
    let cached_tokens = response
        .usage
        .prompt_tokens_details
        .as_ref()
        .map_or(0, |details| details.cached_tokens as usize);
    Ok(LoopStep {
        prompt_tokens: prompt_tokens.len(),
        boundary_tokens: boundary_tokens.len(),
        cached_tokens,
        stored: capture_trace::drain_stored_identities()
            .iter()
            .map(|diag| diag.tokens.len())
            .collect(),
    })
}

fn render_steps(steps: &[LoopStep]) -> String {
    let mut table = String::from("step prompt boundary cached prefilled stored_during_step\n");
    for (index, step) in steps.iter().enumerate() {
        let _ = writeln!(
            table,
            "{index} {} {} {} {} {:?}",
            step.prompt_tokens,
            step.boundary_tokens,
            step.cached_tokens,
            step.prompt_tokens.saturating_sub(step.cached_tokens),
            step.stored
        );
    }
    table
}

/// Issues the loop back to back, the way an agent does: no waiting for the
/// recorder between steps. Step N+1 must restore at least step N's
/// message-history boundary, every step.
fn assert_loop_restores_previous_step(shape: LoopShape, label: &str) -> Result<()> {
    let (backend, _speculative) = recurrent_test_backend(
        "recurrent-agent-loop-test",
        "recurrent-agent-loop-test",
        8192,
        256,
        8,
        8192,
    )?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    capture_trace::reset();
    capture_trace::drain_stored_identities();
    capture_trace::drain_capture_decisions();
    let steps = runtime.block_on(async {
        let mut steps = Vec::with_capacity(LOOP_STEPS);
        for step in 0..LOOP_STEPS {
            steps.push(run_loop_step(&backend, shape, step).await?);
        }
        Ok::<_, anyhow::Error>(steps)
    })?;
    let table = render_steps(&steps);
    // Optional measurement output for before/after comparisons.
    if let Some(directory) = std::env::var_os("SKIPPY_AGENT_LOOP_REPORT_DIR") {
        std::fs::write(std::path::Path::new(&directory).join(label), &table)?;
    }
    let decisions: Vec<String> = capture_trace::drain_capture_decisions()
        .iter()
        .map(|decision| {
            format!(
                "{}:{}@{}",
                decision.decision_prefix, decision.outcome, decision.checkpoint_token_count
            )
        })
        .collect();
    for (index, pair) in steps.windows(2).enumerate() {
        let (previous, current) = (&pair[0], &pair[1]);
        assert!(
            current.cached_tokens >= previous.boundary_tokens,
            "{label}: step {} restored {} tokens, below step {index}'s message-history \
             boundary {}\n{table}capture decisions: {decisions:?}",
            index + 1,
            current.cached_tokens,
            previous.boundary_tokens,
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn recurrent_tool_loop_restores_the_previous_step_every_step() -> Result<()> {
    assert_loop_restores_previous_step(LoopShape::ToolCalls, "tool_loop")
}

#[test]
#[ignore = "requires SKIPPY_RECURRENT_CACHE_TEST_MODEL and _LAYERS; run explicitly with --ignored"]
fn recurrent_chat_loop_restores_the_previous_step_every_step() -> Result<()> {
    assert_loop_restores_previous_step(LoopShape::PlainChat, "chat_loop")
}
