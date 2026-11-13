//! Reference speakers for speech synthesis, decoded once per projector.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};

/// Reference audio whose speaker the projector's speaker encoder conditions
/// synthesis on.
#[derive(Clone, PartialEq, Eq)]
pub struct SpeechSpeaker {
    /// Stable name; the projector caches the decoded clip under it.
    pub name: String,
    /// Encoded clip (WAV, MP3 or FLAC), resampled to the projector's rate.
    pub audio: Arc<[u8]>,
}

impl fmt::Debug for SpeechSpeaker {
    /// Keep the clip's bytes out of logs and error reports.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SpeechSpeaker")
            .field("name", &self.name)
            .field("audio_bytes", &self.audio.len())
            .finish()
    }
}

/// A decoded reference clip. The native bitmap is immutable after decoding.
pub(crate) struct SpeakerBitmap(*mut skippy_ffi::MtmdBitmap);

// SAFETY: the bitmap is plain decoded PCM that native code only reads after
// construction, and it is freed once, when the last owner drops it.
unsafe impl Send for SpeakerBitmap {}
unsafe impl Sync for SpeakerBitmap {}

impl SpeakerBitmap {
    pub(crate) fn raw(&self) -> *mut skippy_ffi::MtmdBitmap {
        self.0
    }
}

impl Drop for SpeakerBitmap {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { skippy_ffi::mtmd_bitmap_free(self.0) };
        }
    }
}

/// One voice's clip and its decoded bitmap.
struct CachedSpeaker {
    audio: Arc<[u8]>,
    bitmap: Arc<SpeakerBitmap>,
}

/// Decoded reference clips keyed by voice name.
#[derive(Default)]
pub(crate) struct SpeakerCache {
    entries: Mutex<BTreeMap<String, CachedSpeaker>>,
}

impl SpeakerCache {
    /// Return the decoded clip for `speaker`, decoding it on first use and
    /// again only if the same name later arrives with different audio.
    pub(crate) fn bitmap(
        &self,
        projector: *mut skippy_ffi::MtmdContext,
        speaker: &SpeechSpeaker,
    ) -> Result<Arc<SpeakerBitmap>> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| anyhow!("speaker cache lock poisoned"))?;
        if let Some(cached) = entries.get(&speaker.name)
            && (Arc::ptr_eq(&cached.audio, &speaker.audio) || cached.audio == speaker.audio)
        {
            return Ok(cached.bitmap.clone());
        }
        let bitmap = Arc::new(decode_speaker(projector, speaker)?);
        entries.insert(
            speaker.name.clone(),
            CachedSpeaker {
                audio: speaker.audio.clone(),
                bitmap: bitmap.clone(),
            },
        );
        Ok(bitmap)
    }
}

/// Decode and resample one clip with the projector's audio front end.
fn decode_speaker(
    projector: *mut skippy_ffi::MtmdContext,
    speaker: &SpeechSpeaker,
) -> Result<SpeakerBitmap> {
    if speaker.audio.is_empty() {
        return Err(anyhow!(
            "voice '{}' has an empty reference clip",
            speaker.name
        ));
    }
    let wrapper = unsafe {
        skippy_ffi::mtmd_helper_bitmap_init_from_buf(
            projector,
            speaker.audio.as_ptr(),
            speaker.audio.len(),
            false,
            skippy_ffi::mtmd_helper_init_opt_default(),
        )
    };
    // Audio never opens a video context; free one defensively if it did.
    if !wrapper.video_ctx.is_null() {
        unsafe { skippy_ffi::mtmd_helper_video_free(wrapper.video_ctx) };
    }
    let bitmap = SpeakerBitmap(wrapper.bitmap);
    if bitmap.0.is_null() {
        return Err(anyhow!(
            "failed to decode the reference clip for voice '{}'",
            speaker.name
        ));
    }
    Ok(bitmap)
}
