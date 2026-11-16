//! Draft head selection and the sequence layout its noise graph accepts.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NoiseBatchLayout {
    Independent,
    Uniform,
    Single,
}

impl NoiseBatchLayout {
    pub(super) fn from_head(serial_head: bool, block_local: bool) -> Self {
        if serial_head {
            Self::Single
        } else if block_local {
            Self::Uniform
        } else {
            Self::Independent
        }
    }

    pub(super) fn permits(self, rows: impl IntoIterator<Item = usize>) -> bool {
        let mut active = rows.into_iter().filter(|&count| count > 0);
        let Some(first) = active.next() else {
            return true;
        };
        match self {
            Self::Independent => true,
            Self::Uniform => active.all(|count| count == first),
            Self::Single => active.next().is_none(),
        }
    }
}

impl DFlashDraft {
    /// Picks the draft family and its block layout: DSpark when the draft has
    /// a Markov head, DFlash2 when it has a selector lattice, else DFlash.
    pub(super) fn select_variant(&mut self, tensors: &DraftTensors) -> Result<()> {
        let markov = tensors.markov;
        let selector_top_k = unsafe { (self.api.model_dflash_selector_top_k)(self.model) };
        // Convolution taps and selector transitions reshape rows into one
        // equally sized block per sequence. Heads without those operations
        // may retain independent, differently sized noise blocks.
        let convolution = meta_usize(self.model, "dflash.conv_kernel_size")?.unwrap_or(0) > 0;
        // Other lattice heads may publish a top-k count too. Without the
        // proven projection set, retain independent submissions.
        self.noise_batch_layout = NoiseBatchLayout::from_head(
            markov || (selector_top_k > 0 && !tensors.selector),
            tensors.selector || convolution,
        );
        if markov {
            if selector_top_k > 0 {
                bail!(
                    "a draft with both a DSpark Markov head and a DFlash2 selector is not supported"
                );
            }
            self.info.variant = DFlashVariant::DSpark;
            let anchor_first = unsafe {
                skippy_ffi::llama_model_meta_val_str(self.model, DSPARK_SAMPLE_FROM_ANCHOR_KEY)
            }
            .is_none_or(|value| value == "true");
            if anchor_first {
                self.layout = BlockLayout::AnchorFirst;
            }
        } else if selector_top_k > 0 {
            self.info.variant = DFlashVariant::DFlash2;
            self.info.selector_top_k = positive_usize(selector_top_k, "selector top-k")?;
            let row_used = self.info.selector_top_k * (self.info.selector_top_k + 1);
            if row_used > self.n_embd_draft {
                bail!("draft hidden size is too small for the DFlash2 selector lattice");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::NoiseBatchLayout;

    #[test]
    fn selector_rows_keep_sequence_boundaries_and_ignore_absent_blocks() {
        let selector = NoiseBatchLayout::from_head(false, true);
        assert!(selector.permits([0, 7, 7, 0]));
        assert!(selector.permits([2, 2, 2]));
        assert!(selector.permits([0, 0]));
        assert!(!selector.permits([7, 4]));
        assert!(!selector.permits([0, 7, 6]));
    }

    #[test]
    fn independent_heads_accept_ragged_blocks_but_markov_heads_stay_serial() {
        assert!(NoiseBatchLayout::from_head(false, false).permits([7, 0, 4]));
        let markov = NoiseBatchLayout::from_head(true, false);
        assert!(markov.permits([0, 7, 0]));
        assert!(!markov.permits([7, 7]));
        // A rejected mixed head must never gain the selector's batch admission.
        assert!(!NoiseBatchLayout::from_head(true, true).permits([7, 7]));
    }
}
