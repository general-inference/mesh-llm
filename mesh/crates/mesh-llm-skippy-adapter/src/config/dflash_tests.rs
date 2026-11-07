use super::test_support::{parse_config, target_meta, temp_model_file_with_tensor_names};
use super::{SkippyConfigResolveRequest, resolution::resolve_skippy_config};
use std::path::Path;

fn resolve(
    config: &str,
    model_path: &Path,
    meta: &skippy_model_artifact::gguf::GgufCompactMeta,
) -> anyhow::Result<super::ResolvedSkippyConfig> {
    let config = parse_config(config);
    resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &config,
        model_id: "test/target",
        model_path,
        model_bytes: 20 << 30,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: Some(meta),
    })
}

#[test]
fn explicit_dflash_uses_the_pinned_catalog_even_when_native_mtp_is_available() {
    let _catalog =
        super::dflash_catalog::with_test_pairings(super::dflash_catalog::bootstrap_pairings());
    let directory = tempfile::tempdir().unwrap();
    let model = directory.path().join("target.gguf");
    let fixture = temp_model_file_with_tensor_names(&["blk.40.nextn.weight"], Some(1));
    std::fs::copy(fixture.path(), &model).unwrap();
    // A normal sibling drafter must not replace the requested DFlash pairing.
    std::fs::copy(fixture.path(), directory.path().join("ordinary-draft.gguf")).unwrap();
    let mut meta = target_meta("qwen35moe", 41, 2048, 256);
    meta.nextn_predict_layers = 1;
    let resolved = resolve(
        "[defaults.speculative]\nstrategy = \"dflash\"\nmode = \"auto\"",
        &model,
        &meta,
    )
    .unwrap();
    let spec = &resolved.speculative;
    let chosen = spec.default_dflash.as_ref().unwrap();
    let pairing = &chosen.pairing;
    let expected =
        super::dflash_catalog::pairing_for_target(&meta, cfg!(target_os = "macos")).unwrap();
    assert_eq!(pairing, &expected.pairing);
    assert_eq!(spec.decode.requested_strategy, "dflash");
    assert_eq!(spec.decode.effective_strategy, "dflash");
    assert!(!spec.native_mtp_enabled);
    assert!(!spec.decode.native_mtp.enabled);
    assert_eq!(
        spec.decode.dflash.as_ref().unwrap().draft_model_path,
        super::dflash_draft_cache_path(&pairing.draft)
    );
    assert_eq!(
        spec.draft_max_tokens,
        super::dflash::dflash_max_draft_tokens(0, Some(chosen))
    );
    assert!(spec.draft_model_path.is_none());

    let mut unavailable = resolved;
    unavailable.speculative.drop_default_dflash();
    let args = unavailable.to_embedded_openai_args(0, false).unwrap();
    assert!(args.speculative.dflash.is_none());
    assert!(!args.speculative.native_mtp.enabled);
    assert_eq!(args.speculative.effective_strategy, "disabled");
}

#[test]
fn catalog_selection_refuses_disabled_mode_and_unknown_targets() {
    let _catalog =
        super::dflash_catalog::with_test_pairings(super::dflash_catalog::bootstrap_pairings());
    let model = temp_model_file_with_tensor_names(&[], None);
    for (config, meta, expected) in [
        (
            "[defaults.speculative]\nstrategy = \"dflash\"\nmode = \"disabled\"",
            target_meta("qwen35moe", 40, 2048, 256),
            "conflicts",
        ),
        (
            "[defaults.speculative]\nstrategy = \"dflash\"",
            target_meta("qwen35moe", 40, 2048, 128),
            "without a catalog pairing",
        ),
    ] {
        let error = resolve(config, model.path(), &meta)
            .unwrap_err()
            .to_string();
        assert!(error.contains(expected), "{error}");
    }
}
