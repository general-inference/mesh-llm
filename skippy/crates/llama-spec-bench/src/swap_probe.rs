//! Swap probe: how a DFlash draft drafts right after it is swapped in.
//!
//! For each prompt and each restart depth R, the target prefills all but the
//! last R prompt tokens, the draft is swapped for the other draft (or
//! reloaded), the session's draft restarts there, and the target prefills the
//! last R tokens, so the new draft has seen only those rows. The reply is then
//! drafted as usual. A run without a swap is the reference.
//!
//! A resident swap exchanges the attached draft with one opened in advance; a
//! loading swap opens the incoming draft from its file each time.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use serde::Serialize;
use skippy_runtime::{
    DFlashDraftOptions, ParkedDFlashDraft, RuntimeConfig, StageModel, StageSession,
};

use crate::{PromptCase, elapsed_ms};

/// Prompt tokens per prefill call; the bench's contexts batch 2048.
const PREFILL_CHUNK: usize = 1024;

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
    /// Keep both drafts open and swap by exchanging them.
    pub resident: bool,
}

/// Which draft is attached, and the other one when it is kept open.
#[derive(Default)]
pub(crate) struct ProbeState {
    attached: usize,
    parked: Option<ParkedDFlashDraft>,
}

impl ProbeSetup<'_> {
    fn swap(&self, state: &mut ProbeState, session: &StageSession) -> Result<()> {
        let next = 1 - state.attached;
        if self.resident {
            let incoming = match state.parked.take() {
                Some(parked) => parked,
                None => self
                    .target
                    .park_dflash_draft(&self.drafts[next], self.options, self.config)
                    .with_context(|| format!("open {}", self.drafts[next].display()))?,
            };
            state.parked = Some(self.target.swap_parked_dflash_draft(session, incoming)?);
        } else {
            self.target
                .swap_dflash_draft(session, &self.drafts[next], self.options, self.config)
                .with_context(|| format!("swap in {}", self.drafts[next].display()))?;
        }
        state.attached = next;
        Ok(())
    }

    /// Opens the second draft ahead of the first resident swap, so no swap
    /// pays for loading it.
    pub(crate) fn prepare(&self, state: &mut ProbeState) -> Result<()> {
        if self.resident && state.parked.is_none() {
            state.parked = Some(self.target.park_dflash_draft(
                &self.drafts[1 - state.attached],
                self.options,
                self.config,
            )?);
        }
        Ok(())
    }
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
    state: &mut ProbeState,
    prompt: &PromptCase,
    tokens: &[i32],
) -> Result<Vec<ProbeRow>> {
    let mut rows = Vec::new();
    let mut depths: Vec<Option<usize>> = vec![None];
    depths.extend(setup.depths.iter().copied().map(Some));
    for depth in depths {
        let mut session = setup.target.create_session()?;
        let body = &tokens[..tokens.len() - 1];
        let split = depth.map_or(body.len(), |rows| body.len().saturating_sub(rows));
        prefill(&mut session, &body[..split])?;
        let swap_ms = match depth {
            Some(_) => {
                let started = Instant::now();
                setup.swap(state, &session)?;
                let ms = elapsed_ms(started);
                session.dflash_restart();
                Some(ms)
            }
            None => None,
        };
        prefill(&mut session, &body[split..])?;
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
            draft_path: display(&setup.drafts[state.attached]),
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

/// Prefills in chunks no larger than the context's batch.
fn prefill(session: &mut StageSession, tokens: &[i32]) -> Result<()> {
    for chunk in tokens.chunks(PREFILL_CHUNK) {
        session.prefill_chunk(chunk)?;
    }
    Ok(())
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
