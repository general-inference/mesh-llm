//! Swap probe: how a DFlash draft drafts right after it is swapped in.
//!
//! For each prompt and each restart depth R, the target prefills all but the
//! last R prompt tokens, the draft is swapped for the other draft (or
//! reloaded), the session's draft restarts there, and the target prefills the
//! last R tokens, so the new draft has seen only those rows. The reply is then
//! drafted as usual. A run without a swap is the reference.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use skippy_runtime::{DFlashDraftOptions, RuntimeConfig, StageModel};

use crate::{PromptCase, elapsed_ms};

/// Windows counted as "early": the blocks drafted right after the swap.
const EARLY_WINDOWS: usize = 8;

pub(crate) struct ProbeSetup<'a> {
    pub target: &'a StageModel,
    pub config: &'a RuntimeConfig,
    pub options: &'a DFlashDraftOptions,
    /// The attached draft first, then the one swapped in, alternating.
    pub drafts: [PathBuf; 2],
    pub depths: &'a [usize],
    pub max_new_tokens: usize,
    pub window: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct ProbeRow {
    id: String,
    category: Option<String>,
    prompt_tokens: usize,
    /// Rows the draft saw before drafting; `None` is the run without a swap.
    restart_rows: Option<usize>,
    swap_ms: Option<f64>,
    draft_path: String,
    generated: usize,
    windows: usize,
    draft_tokens: usize,
    accepted_tokens: usize,
    mean_accepted_per_window: f64,
    early_mean_accepted_per_window: f64,
    decode_ms: f64,
}

pub(crate) fn run(
    setup: &ProbeSetup<'_>,
    prompt: &PromptCase,
    tokens: &[i32],
) -> Result<Vec<ProbeRow>> {
    let mut rows = Vec::new();
    let mut attached = 0usize;
    let mut depths: Vec<Option<usize>> = vec![None];
    depths.extend(setup.depths.iter().copied().map(Some));
    for depth in depths {
        let mut session = setup.target.create_session()?;
        let body = &tokens[..tokens.len() - 1];
        let split = depth.map_or(body.len(), |rows| body.len().saturating_sub(rows));
        if split > 0 {
            session.prefill_chunk(&body[..split])?;
        }
        let swap_ms = match depth {
            Some(_) => {
                let next = 1 - attached;
                let started = Instant::now();
                setup
                    .target
                    .swap_dflash_draft(&setup.drafts[next], setup.options, setup.config)
                    .with_context(|| format!("swap in {}", setup.drafts[next].display()))?;
                let ms = elapsed_ms(started);
                attached = next;
                session.dflash_restart();
                Some(ms)
            }
            None => None,
        };
        if split < body.len() {
            session.prefill_chunk(&body[split..])?;
        }
        let mut current = *tokens.last().expect("checked non-empty prompt");
        let mut generated = 0usize;
        let mut accepted_per_window = Vec::new();
        let mut draft_tokens = 0usize;
        let started = Instant::now();
        while generated < setup.max_new_tokens {
            let window = setup
                .window
                .min((setup.max_new_tokens - generated).saturating_sub(1));
            let proposal = session
                .dflash_propose(current, window)?
                .context("target model has no DFlash draft attached")?;
            let mut inputs = Vec::with_capacity(proposal.tokens.len() + 1);
            inputs.push(current);
            inputs.extend_from_slice(&proposal.tokens);
            let base = session.token_count();
            let predictions = session.verify_tokens(&inputs)?;
            let accepted = proposal
                .tokens
                .iter()
                .zip(&predictions)
                .take_while(|(draft, target)| draft == target)
                .count();
            if !proposal.tokens.is_empty() {
                accepted_per_window.push(accepted);
                draft_tokens += proposal.tokens.len();
            }
            let mut committed = 0usize;
            let mut stopped = false;
            for &token in predictions.iter().take(accepted + 1) {
                generated += 1;
                committed += 1;
                current = token;
                if setup.target.token_is_eog(token)? || generated >= setup.max_new_tokens {
                    stopped = true;
                    break;
                }
            }
            if committed == inputs.len() {
                session.retire_verify_checkpoint(base, inputs.len() as u64)?;
            } else {
                session.trim_session(base + committed as u64)?;
            }
            if stopped {
                break;
            }
        }
        let accepted_tokens: usize = accepted_per_window.iter().sum();
        let early = &accepted_per_window[..accepted_per_window.len().min(EARLY_WINDOWS)];
        rows.push(ProbeRow {
            id: prompt.id.clone(),
            category: prompt.category.clone(),
            prompt_tokens: tokens.len(),
            restart_rows: depth.map(|rows| rows.min(body.len())),
            swap_ms,
            draft_path: display(&setup.drafts[attached]),
            generated,
            windows: accepted_per_window.len(),
            draft_tokens,
            accepted_tokens,
            mean_accepted_per_window: mean(&accepted_per_window),
            early_mean_accepted_per_window: mean(early),
            decode_ms: elapsed_ms(started),
        });
    }
    Ok(rows)
}

fn mean(values: &[usize]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<usize>() as f64 / values.len() as f64
    }
}

fn display(path: &Path) -> String {
    path.display().to_string()
}
