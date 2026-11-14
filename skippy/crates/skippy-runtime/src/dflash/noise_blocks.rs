//! Decodes independent lanes' noise blocks together, then removes scratch KV.
use super::*;

pub(super) struct NoiseBlock {
    pub seq_id: i32,
    pub position: i32,
    pub anchor: i32,
    pub max_tokens: usize,
}

#[derive(Debug, PartialEq, Eq)]
struct BlockSpan {
    offset: usize,
    drafts: usize,
    rows: usize,
}

fn plan_blocks(
    blocks: &[NoiseBlock],
    max_drafts: usize,
    layout: BlockLayout,
    lanes: i32,
    capacity: usize,
) -> Option<Vec<BlockSpan>> {
    let mut sequences = std::collections::BTreeSet::new();
    let mut spans = Vec::with_capacity(blocks.len());
    let mut offset = 0usize;
    for block in blocks {
        if !(0..lanes).contains(&block.seq_id)
            || !sequences.insert(block.seq_id)
            || block.position < 0
        {
            return None;
        }
        let drafts = block.max_tokens.min(max_drafts);
        let rows = if drafts == 0 {
            0
        } else {
            drafts.checked_add(1)?;
            layout.block_tokens(drafts)
        };
        block
            .position
            .checked_add(i32::try_from(rows.saturating_sub(1)).ok()?)?;
        spans.push(BlockSpan {
            offset,
            drafts,
            rows,
        });
        offset = offset
            .checked_add(rows)
            .filter(|total| *total <= capacity)?;
    }
    Some(spans)
}

impl DFlashDraft {
    pub(super) fn propose(
        &mut self,
        seq_id: i32,
        position: i32,
        anchor: i32,
        max_tokens: usize,
    ) -> Result<Vec<i32>> {
        let mut outputs = self.propose_blocks(&[NoiseBlock {
            seq_id,
            position,
            anchor,
            max_tokens,
        }])?;
        Ok(outputs.remove(0))
    }

    pub(super) fn blocks_fit(&self, blocks: &[NoiseBlock]) -> bool {
        let capacity = self
            .n_ubatch
            .min(unsafe { (self.api.n_batch)(self.ctx) } as usize);
        plan_blocks(
            blocks,
            self.info.max_draft_tokens,
            self.layout,
            self.lanes,
            capacity,
        )
        .is_some_and(|spans| {
            self.noise_batch_layout
                .permits(spans.iter().map(|span| span.rows))
        })
    }

    pub(super) fn propose_blocks(&mut self, blocks: &[NoiseBlock]) -> Result<Vec<Vec<i32>>> {
        anyhow::ensure!(
            self.blocks_fit(blocks),
            "draft blocks exceed native capacity or repeat a sequence"
        );
        let spans = plan_blocks(
            blocks,
            self.info.max_draft_tokens,
            self.layout,
            self.lanes,
            self.n_ubatch,
        )
        .context("invalid draft block layout")?;
        self.batch.clear();
        for (block, span) in blocks.iter().zip(&spans) {
            if span.drafts == 0 {
                continue;
            }
            self.truncate(block.seq_id, block.position);
            for index in 0..span.rows {
                let token = if index == 0 {
                    block.anchor
                } else {
                    self.mask_token
                };
                self.batch.push_token(
                    token,
                    block.position + index as i32,
                    block.seq_id,
                    self.info.variant != DFlashVariant::DFlash2,
                );
            }
        }
        let outputs = if spans.iter().all(|span| span.rows == 0) {
            Ok(vec![Vec::new(); blocks.len()])
        } else {
            let rc = unsafe { (self.api.decode)(self.ctx, self.batch.as_raw(self.mrope)) };
            if rc == 0 {
                spans
                    .into_iter()
                    .map(|span| self.read_block_at(span.offset, span.drafts))
                    .collect()
            } else {
                Err(anyhow!("draft block decode failed with status {rc}"))
            }
        };
        // Remove noise even when decoding or readout failed. Only captured
        // target features may survive a proposal, independently in each lane.
        for block in blocks.iter().filter(|block| block.max_tokens > 0) {
            self.truncate(block.seq_id, block.position);
        }
        outputs
    }

    fn read_block_at(&self, offset: usize, drafts: usize) -> Result<Vec<i32>> {
        if drafts == 0 {
            return Ok(Vec::new());
        }
        if self.info.variant == DFlashVariant::DFlash2 {
            let lattice = unsafe { (self.api.get_embeddings_nextn)(self.ctx) };
            anyhow::ensure!(!lattice.is_null(), "DFlash2 selector produced no lattice");
            let lattice = unsafe {
                std::slice::from_raw_parts(
                    lattice.add(offset * self.n_embd_draft),
                    (drafts + 1) * self.n_embd_draft,
                )
            };
            return Ok(walk_dflash2_lattice(
                lattice.chunks_exact(self.n_embd_draft).skip(1),
                self.info.selector_top_k,
                self.n_vocab,
            ));
        }
        let mut tokens = Vec::with_capacity(drafts);
        let first = offset + self.layout.first_draft_row();
        for row in first..first + drafts {
            let row = i32::try_from(row).context("draft row exceeds i32")?;
            if let Some(readout) = &self.backend_readout
                && let Some(token) = readout.token(row, self.n_vocab)?
            {
                tokens.push(token);
                continue;
            }
            let logits = unsafe { (self.api.get_logits_ith)(self.ctx, row) };
            anyhow::ensure!(
                !logits.is_null(),
                "draft produced no logits for block row {row}"
            );
            let logits = unsafe { std::slice::from_raw_parts(logits.cast_const(), self.n_vocab) };
            tokens.push(argmax_token(logits).context("draft logits row is empty")?);
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(seq_id: i32, max_tokens: usize) -> NoiseBlock {
        NoiseBlock {
            seq_id,
            max_tokens,
            position: 10,
            anchor: 1,
        }
    }

    #[test]
    fn ragged_blocks_have_disjoint_output_spans_and_empty_blocks_have_no_rows() {
        let spans = plan_blocks(
            &[block(0, 6), block(1, 0), block(2, 3)],
            6,
            BlockLayout::BonusAnchor,
            3,
            11,
        )
        .unwrap();
        assert_eq!(
            spans,
            vec![
                BlockSpan {
                    offset: 0,
                    drafts: 6,
                    rows: 7
                },
                BlockSpan {
                    offset: 7,
                    drafts: 0,
                    rows: 0
                },
                BlockSpan {
                    offset: 7,
                    drafts: 3,
                    rows: 4
                }
            ]
        );
        assert!(
            plan_blocks(
                &[block(0, 6), block(1, 3)],
                6,
                BlockLayout::BonusAnchor,
                2,
                10
            )
            .is_none()
        );
    }

    #[test]
    fn duplicate_or_foreign_lanes_and_overflow_are_rejected_before_mutation() {
        for blocks in [
            vec![block(0, 2), block(0, 2)],
            vec![block(-1, 2)],
            vec![block(2, 2)],
        ] {
            assert!(plan_blocks(&blocks, 6, BlockLayout::BonusAnchor, 2, 32).is_none());
        }
        let mut overflow = block(0, 2);
        overflow.position = i32::MAX;
        assert!(plan_blocks(&[overflow], 6, BlockLayout::BonusAnchor, 2, 32).is_none());
        assert!(
            plan_blocks(
                &[block(0, usize::MAX)],
                usize::MAX,
                BlockLayout::BonusAnchor,
                2,
                usize::MAX
            )
            .is_none()
        );
    }

    #[test]
    fn anchor_first_blocks_keep_their_trained_width() {
        let spans = plan_blocks(
            &[block(0, 9), block(1, 3)],
            6,
            BlockLayout::AnchorFirst,
            2,
            9,
        )
        .unwrap();
        assert_eq!(
            spans[0],
            BlockSpan {
                offset: 0,
                drafts: 6,
                rows: 6
            }
        );
        assert_eq!(
            spans[1],
            BlockSpan {
                offset: 6,
                drafts: 3,
                rows: 3
            }
        );
    }
}
