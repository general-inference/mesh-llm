//! Bench only: dumps prefill activation and DFlash feature payloads that stage
//! 0 forwards, to measure how well they compress. Off unless
//! `MESH_LLM_BENCH_DUMP_DIR` names a directory.

use std::io::Write;
use std::sync::Mutex;

use skippy_runtime::ActivationFrame;

const CAP_BYTES: usize = 48 << 20;
const MIN_TOKENS: u32 = 512;

/// Bench only: the last prefill's stage-0 telemetry, reported in the next
/// response's timings as `bench_prefill`.
pub(super) static LAST_PREFILL: Mutex<Option<serde_json::Value>> = Mutex::new(None);

static WRITTEN: Mutex<(usize, usize)> = Mutex::new((0, 0));

pub(super) fn record(output: &ActivationFrame) {
    let Ok(dir) = std::env::var("MESH_LLM_BENCH_DUMP_DIR") else {
        return;
    };
    if output.desc.token_count < MIN_TOKENS {
        return;
    }
    let Ok(parts) = output.desc.parts() else {
        return;
    };
    let mut written = WRITTEN
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for (index, part) in parts.iter().enumerate() {
        let start = part.payload_offset as usize;
        let end = start.saturating_add(part.payload_bytes as usize);
        let Some(bytes) = output.payload.get(start..end) else {
            continue;
        };
        let (name, used) = if index == 0 {
            (
                format!("dump-activation-t{}.bin", part.ggml_type),
                &mut written.0,
            )
        } else {
            ("dump-features.bin".to_string(), &mut written.1)
        };
        if *used >= CAP_BYTES {
            continue;
        }
        let take = bytes.len().min(CAP_BYTES - *used);
        let path = std::path::Path::new(&dir).join(name);
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            && file.write_all(&bytes[..take]).is_ok()
        {
            *used += take;
        }
    }
}
