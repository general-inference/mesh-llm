//! Projector, media, and speech-voice settings under `multimodal`.

use crate::diagnostic::DiagnosticResult;
use crate::model::{HardwareConfig, MultimodalConfig};
use crate::validation_support::{
    validate_bool_or_auto, validate_optional_http_url, validate_optional_path,
    validate_optional_u32_range, validation_diagnostic,
};

pub(super) fn validate_multimodal_pair(
    hardware: Option<&HardwareConfig>,
    multimodal: Option<&MultimodalConfig>,
    hardware_path: &str,
    multimodal_path: &str,
) -> DiagnosticResult {
    if let (Some(hardware), Some(multimodal)) = (hardware, multimodal) {
        if let (Some(hardware_mmproj), Some(multimodal_mmproj)) =
            (hardware.mmproj.as_deref(), multimodal.mmproj.as_deref())
            && hardware_mmproj != multimodal_mmproj
        {
            return Err(validation_diagnostic(
                &format!("{multimodal_path}.mmproj"),
                format!(
                    "{multimodal_path}.mmproj must match {hardware_path}.mmproj when both are set"
                ),
            ));
        }
        if let (Some(hardware_offload), Some(multimodal_offload)) = (
            hardware.mmproj_offload.as_ref(),
            multimodal.mmproj_offload.as_ref(),
        ) && hardware_offload != multimodal_offload
        {
            return Err(validation_diagnostic(
                &format!("{multimodal_path}.mmproj_offload"),
                format!(
                    "{multimodal_path}.mmproj_offload must match {hardware_path}.mmproj_offload when both are set"
                ),
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_multimodal(config: &MultimodalConfig, base_path: &str) -> DiagnosticResult {
    validate_optional_path(config.mmproj.as_deref(), &format!("{base_path}.mmproj"))?;
    validate_optional_http_url(
        config.mmproj_url.as_deref(),
        &format!("{base_path}.mmproj_url"),
    )?;
    validate_bool_or_auto(
        config.mmproj_offload.as_ref(),
        &format!("{base_path}.mmproj_offload"),
    )?;
    validate_optional_u32_range(
        config.image_min_tokens,
        &format!("{base_path}.image_min_tokens"),
        1,
        10_000_000,
    )?;
    validate_optional_u32_range(
        config.image_max_tokens,
        &format!("{base_path}.image_max_tokens"),
        1,
        10_000_000,
    )?;
    if let (Some(min), Some(max)) = (config.image_min_tokens, config.image_max_tokens)
        && min > max
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.image_min_tokens"),
            format!(
                "{base_path}.image_min_tokens must be less than or equal to {base_path}.image_max_tokens"
            ),
        ));
    }
    if config.image_marker.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.image_marker"),
            format!(
                "{base_path}.image_marker is not supported because mtmd removed custom image markers; use {base_path}.media_marker"
            ),
        ));
    }
    if config.media_marker.as_deref().is_some_and(str::is_empty) {
        return Err(validation_diagnostic(
            &format!("{base_path}.media_marker"),
            format!("{base_path}.media_marker must not be empty"),
        ));
    }
    validate_optional_u32_range(
        config.batch_max_tokens,
        &format!("{base_path}.batch_max_tokens"),
        1,
        i32::MAX as u32,
    )?;
    if let Some(policy) = config.glm_dsa_policy.as_deref()
        && !matches!(policy, "auto" | "v1")
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.glm_dsa_policy"),
            format!("{base_path}.glm_dsa_policy must be \"auto\" or \"v1\""),
        ));
    }
    validate_optional_u32_range(
        config.generation_signal_window,
        &format!("{base_path}.generation_signal_window"),
        1,
        4096,
    )?;
    validate_speech_voices(config, base_path)?;
    if config.embeddings.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.embeddings"),
            format!("{base_path}.embeddings is documented-rejected and must not be set"),
        ));
    }
    if config.reranking.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.reranking"),
            format!("{base_path}.reranking is documented-rejected and must not be set"),
        ));
    }
    if config.pooling.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.pooling"),
            format!("{base_path}.pooling is documented-rejected and must not be set"),
        ));
    }
    if config.vocoder.is_some() {
        return Err(validation_diagnostic(
            &format!("{base_path}.vocoder"),
            format!("{base_path}.vocoder is documented-rejected and must not be set"),
        ));
    }
    Ok(())
}

/// Voice names are request identifiers; clips are absolute paths or URLs.
fn validate_speech_voices(config: &MultimodalConfig, base_path: &str) -> DiagnosticResult {
    for (name, source) in config.voices.iter().flatten() {
        let path = format!("{base_path}.voices.{name}");
        let valid_name = !name.is_empty()
            && name
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'));
        if !valid_name || name == "default" {
            return Err(validation_diagnostic(
                &path,
                format!(
                    "{base_path}.voices names must be ASCII letters, digits, '-', '_' or '.', and not 'default' (set {base_path}.default_voice instead)"
                ),
            ));
        }
        let source = source.trim();
        if source.starts_with("http://") || source.starts_with("https://") {
            validate_optional_http_url(Some(source), &path)?;
        } else if !std::path::Path::new(source).is_absolute() {
            return Err(validation_diagnostic(
                &path,
                format!("{path} must be an absolute path or an http(s) URL to an audio clip"),
            ));
        } else {
            validate_optional_path(Some(source), &path)?;
        }
    }
    if let (Some(default_voice), Some(voices)) = (config.default_voice.as_deref(), &config.voices)
        && !voices.contains_key(default_voice)
    {
        return Err(validation_diagnostic(
            &format!("{base_path}.default_voice"),
            format!("{base_path}.default_voice '{default_voice}' is not in {base_path}.voices"),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::diagnostic::legacy_validation_error_text;
    use crate::{MeshConfig, validate_config_diagnostics};

    fn errors(voices: &str) -> String {
        let source = format!(
            "[[models]]\nmodel = \"Qwen/Qwen3-0.6B:Q4_K_M\"\n\n[models.multimodal]\n{voices}\n"
        );
        let config: MeshConfig = toml::from_str(&source).expect("config should parse");
        legacy_validation_error_text(&validate_config_diagnostics(&config))
    }

    #[test]
    fn speech_voices_accept_absolute_paths_urls_and_a_listed_default() {
        let text = errors(
            "default_voice = \"max\"\nvoices = { max = \"/voices/max.wav\", narrator = \"https://example.com/n.wav\" }",
        );
        assert!(text.is_empty(), "{text}");
    }

    #[test]
    fn speech_voices_reject_relative_clips_reserved_names_and_unknown_defaults() {
        let relative = errors("voices = { max = \"voices/max.wav\" }");
        assert!(relative.contains("multimodal.voices.max"), "{relative}");
        let reserved = errors("voices = { default = \"/voices/max.wav\" }");
        assert!(reserved.contains("not 'default'"), "{reserved}");
        let spaced = errors("voices = { \"two words\" = \"/voices/max.wav\" }");
        assert!(spaced.contains("multimodal.voices"), "{spaced}");
        let unknown = errors("default_voice = \"bob\"\nvoices = { max = \"/voices/max.wav\" }");
        assert!(unknown.contains("default_voice 'bob'"), "{unknown}");
    }
}
