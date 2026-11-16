//! Owns the stateless greedy chains attached to a draft context.

use std::{cell::Cell, ptr};

use anyhow::{Result, bail};
use skippy_ffi::Opaque;
use skippy_ffi::llama_draft::LlamaDraftSamplingApi;

pub(super) struct BackendReadout {
    api: LlamaDraftSamplingApi,
    ctx: *mut Opaque,
    samplers: Vec<*mut Opaque>,
    token_rows: Cell<u64>,
    fallback_rows: Cell<u64>,
}

impl BackendReadout {
    /// The context must outlive the readout. Each chain is initialized exactly
    /// once; resetting a draft's KV does not detach or reattach its sampler.
    pub(super) fn attach(api: LlamaDraftSamplingApi, ctx: *mut Opaque, lanes: i32) -> Option<Self> {
        let mut readout = Self {
            api,
            ctx,
            samplers: Vec::new(),
            token_rows: Cell::new(0),
            fallback_rows: Cell::new(0),
        };
        for seq in 0..lanes {
            unsafe {
                let chain = (api.chain_init)((api.chain_default_params)());
                if chain.is_null() {
                    return None;
                }
                let greedy = (api.init_greedy)();
                if greedy.is_null() {
                    (api.free)(chain);
                    return None;
                }
                (api.chain_add)(chain, greedy);
                readout.samplers.push(chain);
                if !(api.set_sampler)(ctx, seq, chain) {
                    return None;
                }
            }
        }
        Some(readout)
    }

    pub(super) fn token(&self, row: i32, n_vocab: usize) -> Result<Option<i32>> {
        let token = unsafe { (self.api.get_token)(self.ctx, row) };
        // Upstream can attach a chain whose backend prefix is empty. In that
        // case it returns processed logits instead of a sampled token, and its
        // logits getter supplies those rows to the existing CPU argmax path.
        if token == -1 {
            self.fallback_rows
                .set(self.fallback_rows.get().saturating_add(1));
            return Ok(None);
        }
        if usize::try_from(token)
            .ok()
            .is_none_or(|token| token >= n_vocab)
        {
            bail!("draft backend returned an invalid token for row {row}");
        }
        self.token_rows.set(self.token_rows.get().saturating_add(1));
        Ok(Some(token))
    }

    pub(super) fn rows(&self) -> (u64, u64) {
        (self.token_rows.get(), self.fallback_rows.get())
    }
}

impl Drop for BackendReadout {
    fn drop(&mut self) {
        for (seq, sampler) in self.samplers.iter().enumerate() {
            unsafe {
                (self.api.set_sampler)(self.ctx, seq as i32, ptr::null_mut());
                (self.api.free)(*sampler);
            }
        }
    }
}

/// Keep host readout available for matched benchmarks and backends where
/// sampling dispatch overhead outweighs the vocabulary transfer saved.
pub(super) fn enabled() -> bool {
    std::env::var("SKIPPY_DFLASH_GPU_READOUT").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_ffi::llama_draft::LlamaSamplerChainParams;

    struct Context {
        attached: Vec<*mut Opaque>,
        reject: Option<i32>,
        tokens: Vec<i32>,
    }

    struct Chain {
        greedy: *mut u8,
    }

    unsafe extern "C" fn defaults() -> LlamaSamplerChainParams {
        LlamaSamplerChainParams { no_perf: true }
    }

    unsafe extern "C" fn chain_init(_: LlamaSamplerChainParams) -> *mut Opaque {
        Box::into_raw(Box::new(Chain {
            greedy: ptr::null_mut(),
        }))
        .cast()
    }

    unsafe extern "C" fn greedy() -> *mut Opaque {
        Box::into_raw(Box::new(0u8)).cast()
    }

    unsafe extern "C" fn add(chain: *mut Opaque, greedy: *mut Opaque) {
        unsafe { (*chain.cast::<Chain>()).greedy = greedy.cast() };
    }

    unsafe extern "C" fn free(chain: *mut Opaque) {
        let chain = unsafe { Box::from_raw(chain.cast::<Chain>()) };
        if !chain.greedy.is_null() {
            drop(unsafe { Box::from_raw(chain.greedy) });
        }
    }

    unsafe extern "C" fn set(ctx: *mut Opaque, seq: i32, chain: *mut Opaque) -> bool {
        let ctx = unsafe { &mut *ctx.cast::<Context>() };
        if !chain.is_null() && ctx.reject == Some(seq) {
            return false;
        }
        ctx.attached[seq as usize] = chain;
        true
    }

    unsafe extern "C" fn token(ctx: *mut Opaque, row: i32) -> i32 {
        let ctx = unsafe { &*ctx.cast::<Context>() };
        ctx.tokens.get(row as usize).copied().unwrap_or(-1)
    }

    fn api() -> LlamaDraftSamplingApi {
        LlamaDraftSamplingApi {
            chain_default_params: defaults,
            chain_init,
            chain_add: add,
            init_greedy: greedy,
            free,
            set_sampler: set,
            get_token: token,
        }
    }

    #[test]
    fn reads_each_block_row_and_detaches_every_lane() {
        let mut ctx = Context {
            attached: vec![ptr::null_mut(); 2],
            reject: None,
            tokens: vec![5, 8, 9, -1, 10],
        };
        let readout = BackendReadout::attach(api(), (&raw mut ctx).cast(), 2).unwrap();
        assert!(ctx.attached.iter().all(|p| !p.is_null()));
        assert_eq!(readout.token(1, 10).unwrap(), Some(8));
        assert_eq!(readout.token(2, 10).unwrap(), Some(9));
        assert_eq!(readout.token(3, 10).unwrap(), None);
        assert!(readout.token(4, 10).is_err());
        assert_eq!(readout.rows(), (2, 1));
        drop(readout);
        assert!(ctx.attached.iter().all(|p| p.is_null()));
    }

    #[test]
    fn partial_backend_refusal_detaches_previously_attached_lanes() {
        let mut ctx = Context {
            attached: vec![ptr::null_mut(); 2],
            reject: Some(1),
            tokens: vec![],
        };
        assert!(BackendReadout::attach(api(), (&raw mut ctx).cast(), 2).is_none());
        assert!(ctx.attached.iter().all(|p| p.is_null()));
    }
}
