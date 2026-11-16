//! Tensor-backed draft head and borrowed-embedding contracts.
use std::path::Path;

use anyhow::{Context, Result};

use super::{DFLASH_BORROWED_TARGET_TENSOR, DSPARK_MARKOV_TENSOR};

/// What a draft's tensors say about how it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DraftTensors {
    /// The draft carries its own token embeddings.
    pub(super) embeds: bool,
    /// The draft is a DSpark draft with a Markov head.
    pub(super) markov: bool,
    /// The complete predecessor/successor/hidden selector projection set.
    pub(super) selector: bool,
}

pub(super) fn inspect_draft_tensors(path: &Path) -> Result<DraftTensors> {
    let catalog = skippy_model::gguf_catalog::read_gguf_tensor_catalog(path)
        .with_context(|| format!("read draft GGUF {}", path.display()))?;
    let has = |name: &str| catalog.tensors.iter().any(|tensor| tensor.name == name);
    Ok(DraftTensors {
        embeds: has(DFLASH_BORROWED_TARGET_TENSOR),
        markov: has(DSPARK_MARKOV_TENSOR),
        selector: [
            "selector_predecessor.weight",
            "selector_successor.weight",
            "selector_hidden.weight",
        ]
        .into_iter()
        .all(has),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GGUF holding one-float tensors named `names`.
    fn gguf_with_tensors(names: &[&str]) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(names.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        for (index, name) in names.iter().enumerate() {
            bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
            bytes.extend_from_slice(name.as_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&1u64.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes()); // f32
            bytes.extend_from_slice(&(32 * index as u64).to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.resize(bytes.len() + 32 * names.len(), 0);
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file
    }

    #[test]
    fn a_markov_head_marks_a_dspark_draft() {
        let dflash = gguf_with_tensors(&["blk.0.attn_q.weight", "fc.weight"]);
        assert_eq!(
            inspect_draft_tensors(dflash.path()).unwrap(),
            DraftTensors {
                embeds: false,
                markov: false,
                selector: false
            }
        );
        let dspark = gguf_with_tensors(&[
            "blk.0.attn_q.weight",
            "token_embd.weight",
            "markov_w1.weight",
            "markov_w2.weight",
        ]);
        assert_eq!(
            inspect_draft_tensors(dspark.path()).unwrap(),
            DraftTensors {
                embeds: true,
                markov: true,
                selector: false
            }
        );
    }

    #[test]
    fn selector_admission_requires_the_complete_projection_set() {
        for names in [
            vec!["selector_hidden.weight"],
            vec!["selector_hidden.weight", "selector_predecessor.weight"],
        ] {
            let partial = gguf_with_tensors(&names);
            assert!(!inspect_draft_tensors(partial.path()).unwrap().selector);
        }
        let complete = gguf_with_tensors(&[
            "selector_hidden.weight",
            "selector_predecessor.weight",
            "selector_successor.weight",
        ]);
        assert!(inspect_draft_tensors(complete.path()).unwrap().selector);
    }
}
