//! Target verification shared across independent speculative lanes.
use std::ptr;

use anyhow::{Context, Result, ensure};
use skippy_ffi::bindings::skippy::skippy_verify_request;

use crate::native_timing::native_call;
use crate::{SamplingConfig, StageSession, dflash::DecodedRows, error::ensure_ok};

/// One equal-length window on a distinct session of a complete terminal model.
pub struct VerifyBatchRequest<'a> {
    pub session: &'a mut StageSession,
    pub token_ids: &'a [i32],
    pub sampling: Option<&'a SamplingConfig>,
}

impl StageSession {
    /// Actual native capacity after backend allocation and batch clamping.
    /// The upstream context metadata functions share the optional draft API
    /// table; absent exports keep scheduler verification on its serial path.
    pub fn verification_batch_capacity(&self) -> Option<usize> {
        let api = skippy_ffi::llama_draft::llama_draft_api()?;
        let context = unsafe { skippy_ffi::skippy_session_llama_context(self.raw) }
            .cast::<skippy_ffi::Opaque>();
        if context.is_null() {
            return None;
        }
        let capacity = unsafe { (api.n_ubatch)(context).min((api.n_batch)(context)) };
        usize::try_from(capacity)
            .ok()
            .filter(|capacity| *capacity > 0)
    }

    /// Shares one target forward across compatible speculative windows.
    /// Each result stops at that lane's first rejection. The caller must trim
    /// or retire each window, as with single-session sampled verification.
    /// The native layer rejects unsupported shapes before decoding.
    pub fn verify_batch_sampled(requests: &mut [VerifyBatchRequest<'_>]) -> Result<Vec<Vec<i32>>> {
        let Some(first) = requests.first() else {
            return Ok(Vec::new());
        };
        let width = first.token_ids.len();
        ensure!(width > 0, "verification windows must not be empty");
        let ends = requests
            .iter()
            .map(|request| {
                ensure!(
                    request.token_ids.len() == width,
                    "verification windows must have equal length"
                );
                ensure!(
                    request.session.terminal_stage,
                    "shared verification requires terminal sessions"
                );
                request
                    .session
                    .token_count
                    .checked_add(width as u64)
                    .context("session position overflow")
            })
            .collect::<Result<Vec<_>>>()?;
        let sampling = requests
            .iter()
            .map(|request| request.sampling.map(SamplingConfig::as_raw).transpose())
            .collect::<Result<Vec<_>>>()?;
        let mut outputs = vec![vec![0; width]; requests.len()];
        let mut counts = vec![0; requests.len()];
        let raw: Vec<_> = requests
            .iter()
            .zip(&sampling)
            .zip(outputs.iter_mut().zip(&mut counts))
            .map(
                |((request, sampling), (output, count))| skippy_verify_request {
                    session: request.session.raw,
                    token_ids: request.token_ids.as_ptr(),
                    token_count: width,
                    sampling: sampling.as_ref().map_or(ptr::null(), ptr::from_ref),
                    output_tokens: output.as_mut_ptr(),
                    output_capacity: output.len(),
                    out_token_count: count,
                },
            )
            .collect();
        let mut error = ptr::null_mut();
        let status = unsafe {
            native_call!(skippy_verify_batch_sampled(
                raw.as_ptr(),
                raw.len(),
                &mut error
            ))
        };
        ensure_ok(status, error)?;
        for (index, request) in requests.iter_mut().enumerate() {
            ensure!(
                counts[index] > 0 && counts[index] <= width,
                "invalid native verification output count"
            );
            request.session.token_count = ends[index];
            request
                .session
                .dflash_after_decode(DecodedRows::trailing(index * width, width));
            outputs[index].truncate(counts[index]);
        }
        Ok(outputs)
    }
}
