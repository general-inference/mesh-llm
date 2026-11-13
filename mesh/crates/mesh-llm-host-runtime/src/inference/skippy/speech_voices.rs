//! Reads the configured speech reference clips once per model load.

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::time::Duration;

/// Reference clips are a few seconds of speech; refuse anything far larger.
const MAX_VOICE_CLIP_BYTES: usize = 16 * 1024 * 1024;

/// Read every configured clip so a missing or unreadable voice fails the
/// load instead of the first speech request that names it.
pub(crate) async fn load_speech_voices(
    multimodal: &super::resolver::ResolvedMultimodalConfig,
) -> Result<skippy_serving::SpeechVoices> {
    let mut clips = BTreeMap::new();
    for (name, source) in &multimodal.voices {
        let audio = read_voice_clip(source.trim())
            .await
            .with_context(|| format!("read multimodal.voices.{name} from {source}"))?;
        clips.insert(name.clone(), audio);
    }
    skippy_serving::SpeechVoices::new(clips, multimodal.default_voice.clone())
        .context("configure multimodal.voices")
}

async fn read_voice_clip(source: &str) -> Result<Vec<u8>> {
    let audio = if source.starts_with("http://") || source.starts_with("https://") {
        fetch_voice_clip(source).await?
    } else {
        let metadata = tokio::fs::metadata(source).await?;
        if metadata.len() > MAX_VOICE_CLIP_BYTES as u64 {
            bail!("clip is larger than {MAX_VOICE_CLIP_BYTES} bytes");
        }
        tokio::fs::read(source).await?
    };
    Ok(audio)
}

async fn fetch_voice_clip(url: &str) -> Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;
    let mut response = client.get(url).send().await?.error_for_status()?;
    let mut audio = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if audio.len() + chunk.len() > MAX_VOICE_CLIP_BYTES {
            bail!("clip is larger than {MAX_VOICE_CLIP_BYTES} bytes");
        }
        audio.extend_from_slice(&chunk);
    }
    Ok(audio)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multimodal(
        voices: &[(&str, &str)],
        default_voice: Option<&str>,
    ) -> super::super::resolver::ResolvedMultimodalConfig {
        super::super::resolver::ResolvedMultimodalConfig {
            projector_url: None,
            projector_use_gpu: None,
            media_marker: None,
            image_min_tokens: None,
            image_max_tokens: None,
            batch_max_tokens: None,
            glm_dsa_policy: skippy_protocol::GlmDsaPolicy::Auto,
            generation_signal_window: None,
            voices: voices
                .iter()
                .map(|(name, source)| ((*name).to_string(), (*source).to_string()))
                .collect(),
            default_voice: default_voice.map(str::to_string),
        }
    }

    #[tokio::test]
    async fn configured_clips_load_once_and_missing_clips_fail_the_load() {
        let dir = tempfile::tempdir().unwrap();
        let clip = dir.path().join("max.wav");
        let mut wav = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        wav.resize(64, 0);
        std::fs::write(&clip, &wav).unwrap();
        let clip = clip.to_str().unwrap();

        let voices = load_speech_voices(&multimodal(&[("max", clip)], Some("max")))
            .await
            .unwrap();
        assert_eq!(voices.names().collect::<Vec<_>>(), ["max"]);
        assert_eq!(voices.default_voice(), Some("max"));

        let missing = dir.path().join("missing.wav");
        let error = load_speech_voices(&multimodal(&[("max", missing.to_str().unwrap())], None))
            .await
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("multimodal.voices.max"),
            "{error:#}"
        );

        assert!(
            load_speech_voices(&multimodal(&[], None))
                .await
                .unwrap()
                .is_empty()
        );
    }
}
