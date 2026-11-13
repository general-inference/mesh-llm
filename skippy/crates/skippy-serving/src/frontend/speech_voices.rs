//! Named reference voices that a speech request's `voice` selects.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use serde::ser::SerializeStruct;
use skippy_inference_api::{InferenceError, InferenceResult};
use skippy_runtime::SpeechSpeaker;

/// The request voice that keeps the model's own speaker choice unless a
/// default voice is configured.
pub const DEFAULT_SPEECH_VOICE: &str = "default";

/// Reference clips for one speech model, keyed by the name requests use.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpeechVoices {
    default_voice: Option<String>,
    voices: BTreeMap<String, SpeechSpeaker>,
}

/// What a request's `voice` resolves to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::frontend) struct ResolvedSpeechVoice {
    pub(in crate::frontend) speaker: Option<SpeechSpeaker>,
    /// Sampling seed. A reference voice uses a fixed seed derived from its
    /// name so the same text always renders the same way in that voice.
    pub(in crate::frontend) seed: u32,
}

impl SpeechVoices {
    /// Build the catalog from encoded clips (WAV, MP3 or FLAC). `default_voice`
    /// names the clip `"default"` requests use.
    pub fn new(clips: BTreeMap<String, Vec<u8>>, default_voice: Option<String>) -> Result<Self> {
        let mut voices = BTreeMap::new();
        for (name, audio) in clips {
            validate_speech_voice_name(&name)?;
            if !is_supported_audio_container(&audio) {
                bail!("voice '{name}' reference clip is not WAV, MP3, or FLAC audio");
            }
            let speaker = SpeechSpeaker {
                name: name.clone(),
                audio: Arc::from(audio),
            };
            voices.insert(name, speaker);
        }
        if let Some(default_voice) = default_voice.as_deref()
            && !voices.contains_key(default_voice)
        {
            bail!("default voice '{default_voice}' is not one of the configured voices");
        }
        Ok(Self {
            default_voice,
            voices,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.voices.is_empty()
    }

    /// Configured voice names in sorted order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.voices.keys().map(String::as_str)
    }

    pub fn default_voice(&self) -> Option<&str> {
        self.default_voice.as_deref()
    }

    /// Map a request's `voice` to a reference speaker and sampling seed.
    pub(in crate::frontend) fn resolve(
        &self,
        requested: &str,
    ) -> InferenceResult<ResolvedSpeechVoice> {
        let name = if requested == DEFAULT_SPEECH_VOICE {
            match self.default_voice.as_deref() {
                Some(name) => name,
                None => {
                    return Ok(ResolvedSpeechVoice {
                        speaker: None,
                        seed: u32::MAX,
                    });
                }
            }
        } else {
            requested
        };
        let Some(speaker) = self.voices.get(name) else {
            return Err(unknown_voice_error(requested, self.names()));
        };
        Ok(ResolvedSpeechVoice {
            speaker: Some(speaker.clone()),
            seed: voice_seed(name),
        })
    }
}

impl serde::Serialize for SpeechVoices {
    /// Report voice names, never clip bytes.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("SpeechVoices", 2)?;
        state.serialize_field("default_voice", &self.default_voice)?;
        state.serialize_field("voices", &self.voices.keys().collect::<Vec<_>>())?;
        state.end()
    }
}

/// Voice names are plain identifiers; `default` is reserved for the request
/// value that selects `default_voice`.
pub fn validate_speech_voice_name(name: &str) -> Result<()> {
    if name.is_empty()
        || !name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.'))
    {
        bail!("voice name '{name}' must be non-empty ASCII letters, digits, '-', '_' or '.'");
    }
    if name == DEFAULT_SPEECH_VOICE {
        bail!("voice name 'default' is reserved; name the voice and set default_voice instead");
    }
    Ok(())
}

/// The containers the projector's audio decoder reads, by magic bytes.
fn is_supported_audio_container(audio: &[u8]) -> bool {
    let wav = audio.len() >= 12 && &audio[..4] == b"RIFF" && &audio[8..12] == b"WAVE";
    let flac = audio.starts_with(b"fLaC");
    let mp3 = audio.starts_with(b"ID3")
        || (audio.len() >= 2 && audio[0] == 0xFF && (audio[1] & 0xE0) == 0xE0);
    wav || flac || mp3
}

/// FNV-1a of the voice name: stable across processes and releases.
fn voice_seed(name: &str) -> u32 {
    let hash = name.bytes().fold(0x811c_9dc5_u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    });
    // u32::MAX asks the sampler for a random seed.
    if hash == u32::MAX { 0 } else { hash }
}

fn unknown_voice_error<'a>(
    requested: &str,
    names: impl Iterator<Item = &'a str>,
) -> InferenceError {
    let available = std::iter::once(DEFAULT_SPEECH_VOICE)
        .chain(names)
        .map(|name| format!("'{name}'"))
        .collect::<Vec<_>>()
        .join(", ");
    InferenceError::unsupported(format!(
        "voice '{requested}' is not configured for this model; available voices: {available}"
    ))
    .with_param("voice")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav() -> Vec<u8> {
        let mut bytes = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        bytes.resize(64, 0);
        bytes
    }

    fn catalog(default_voice: Option<&str>) -> SpeechVoices {
        SpeechVoices::new(
            BTreeMap::from([("max".to_string(), wav())]),
            default_voice.map(str::to_string),
        )
        .unwrap()
    }

    #[test]
    fn default_without_a_default_voice_keeps_the_model_speaker() {
        let resolved = catalog(None).resolve("default").unwrap();
        assert_eq!(resolved.speaker, None);
        assert_eq!(resolved.seed, u32::MAX);
        let resolved = SpeechVoices::default().resolve("default").unwrap();
        assert_eq!(resolved.speaker, None);
    }

    #[test]
    fn named_and_default_voices_share_one_speaker_and_seed() {
        let voices = catalog(Some("max"));
        let named = voices.resolve("max").unwrap();
        let default = voices.resolve("default").unwrap();
        assert_eq!(named, default);
        assert_eq!(named.speaker.unwrap().name, "max");
        assert_ne!(named.seed, u32::MAX);
        assert_eq!(named.seed, voice_seed("max"));
    }

    #[test]
    /// Voice selection must not be silently repurposed as a language selector.
    fn unknown_voice_is_a_structured_voice_error() {
        for voice in ["alloy", "english", ""] {
            let error = catalog(None).resolve(voice).unwrap_err();
            let body = error.body().error;
            assert_eq!(body.code.as_deref(), Some("unsupported_model_feature"));
            assert_eq!(body.param.as_deref(), Some("voice"));
            assert!(
                body.message.contains("'default', 'max'"),
                "{}",
                body.message
            );
        }
        let error = SpeechVoices::default().resolve("max").unwrap_err();
        assert!(error.message().ends_with("available voices: 'default'"));
    }

    #[test]
    fn construction_rejects_bad_names_clips_and_defaults() {
        let clip = |name: &str, audio: Vec<u8>| BTreeMap::from([(name.to_string(), audio)]);
        assert!(SpeechVoices::new(clip("default", wav()), None).is_err());
        assert!(SpeechVoices::new(clip("two words", wav()), None).is_err());
        assert!(SpeechVoices::new(clip("max", b"not audio".to_vec()), None).is_err());
        assert!(SpeechVoices::new(clip("max", wav()), Some("other".into())).is_err());
        assert!(SpeechVoices::new(clip("max", b"fLaC....".to_vec()), None).is_ok());
    }

    #[test]
    fn serialization_names_voices_without_clip_bytes() {
        let json = serde_json::to_value(catalog(Some("max"))).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"default_voice": "max", "voices": ["max"]})
        );
    }
}
