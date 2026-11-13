//! Feature dump: the target-layer features a DFlash draft reads, captured from
//! the served (quantized) target, for training a draft on what it will see.
//!
//! Each line of the token corpus is `{"index": n, "input_ids": [...]}`, already
//! laid out the way the server lays out a conversation. The target prefills the
//! tokens with the draft attached and keeps every feature row; row `p` is the
//! draft's target layers at token `p`, concatenated in `target_layer_ids`
//! order. Rows are written as little-endian bf16 to `<dir>/<index:06>.bf16`,
//! and `<dir>/index.jsonl` records each file's rows and row width.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use skippy_runtime::{StageModel, StageSession};

use crate::elapsed_ms;

/// Tokens per prefill call; the bench's contexts batch 2048.
const PREFILL_CHUNK: usize = 1024;

#[derive(Deserialize)]
struct TokenRow {
    index: u64,
    input_ids: Vec<i32>,
}

pub(crate) fn run(target: &StageModel, corpus: &Path, dir: &Path) -> Result<()> {
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let mut index_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("index.jsonl"))?;
    let reader = BufReader::new(
        fs::File::open(corpus).with_context(|| format!("open {}", corpus.display()))?,
    );
    let started = Instant::now();
    let mut tokens_done = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: TokenRow = serde_json::from_str(&line).context("parse token corpus row")?;
        let path = dir.join(format!("{:06}.bf16", row.index));
        if path.exists() || row.input_ids.is_empty() {
            continue;
        }
        let mut session = target.create_session()?;
        session.dflash_keep_history(row.input_ids.len());
        prefill(&mut session, &row.input_ids)?;
        let (features, positions) = session
            .dflash_history_rows()?
            .context("target model has no DFlash draft attached")?;
        if positions.len() != row.input_ids.len() || positions.first() != Some(&0) {
            bail!(
                "sample {} kept {} of {} feature rows",
                row.index,
                positions.len(),
                row.input_ids.len()
            );
        }
        let width = features.len() / positions.len();
        let mut bytes = Vec::with_capacity(features.len() * 2);
        for value in features {
            bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
        }
        let partial = path.with_extension("bf16.partial");
        fs::write(&partial, &bytes)?;
        fs::rename(&partial, &path)?;
        writeln!(
            index_file,
            "{}",
            serde_json::json!({"index": row.index, "rows": positions.len(), "width": width})
        )?;
        tokens_done += positions.len();
        eprintln!(
            "dumped {} ({} rows), {:.0} tok/s",
            row.index,
            positions.len(),
            tokens_done as f64 / (elapsed_ms(started) / 1000.0)
        );
    }
    Ok(())
}

fn prefill(session: &mut StageSession, tokens: &[i32]) -> Result<()> {
    for chunk in tokens.chunks(PREFILL_CHUNK) {
        session.prefill_chunk(chunk)?;
    }
    Ok(())
}

/// The bf16 nearest to `value`, ties to even; NaN stays NaN.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let round = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(round) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::bf16_bits;

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits(-2.5), 0xc020);
        // 1 + 2^-8 is halfway between two bf16 values and rounds to the even one.
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);
        assert_eq!(bf16_bits(1.0 + 3.0 / 256.0), 0x3f82);
        assert!(f32::from_bits(u32::from(bf16_bits(f32::NAN)) << 16).is_nan());
    }
}
