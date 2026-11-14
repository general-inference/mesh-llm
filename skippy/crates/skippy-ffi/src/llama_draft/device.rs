//! Optional upstream device-row injection surface.
use crate::{Opaque, bindings};

pub type LlamaProcessType = bindings::llama::llama_process_type;

#[derive(Clone, Copy)]
pub struct LlamaDraftDeviceApi {
    pub set_capture: unsafe extern "C" fn(*mut Opaque, bool),
    pub supports_rows: unsafe extern "C" fn(*const Opaque) -> bool,
    pub batch_init: unsafe extern "C" fn(*mut Opaque) -> *mut Opaque,
    pub batch_free: unsafe extern "C" fn(*mut Opaque),
    pub batch_clear: unsafe extern "C" fn(*mut Opaque),
    pub add_row: unsafe extern "C" fn(*mut Opaque, i32, i32) -> i32,
    pub set_pos: unsafe extern "C" fn(*mut Opaque, i32, *const i32) -> bool,
    pub set_output: unsafe extern "C" fn(*mut Opaque, i32, bool) -> bool,
    pub process: unsafe extern "C" fn(*mut Opaque, LlamaProcessType, *mut Opaque) -> i32,
    pub synchronize: unsafe extern "C" fn(*mut Opaque),
}

#[cfg(not(feature = "dynamic-runtime"))]
mod linked {
    use super::*;
    use bindings::llama as raw;

    unsafe extern "C" {
        #[cfg_attr(
            not(target_env = "msvc"),
            link_name = "_Z34llama_set_embeddings_layer_inp_devP13llama_contextb"
        )]
        #[cfg_attr(
            target_env = "msvc",
            link_name = "?llama_set_embeddings_layer_inp_dev@@YAXPEAUllama_context@@_N@Z"
        )]
        fn set_capture(ctx: *mut Opaque, enabled: bool);
        #[cfg_attr(
            not(target_env = "msvc"),
            link_name = "_Z23llama_supports_embd_rowPK13llama_context"
        )]
        #[cfg_attr(
            target_env = "msvc",
            link_name = "?llama_supports_embd_row@@YA_NPEBUllama_context@@@Z"
        )]
        fn supports_rows(ctx: *const Opaque) -> bool;
        #[cfg_attr(
            not(target_env = "msvc"),
            link_name = "_Z28llama_batch_ext_add_embd_rowP15llama_batch_extii"
        )]
        #[cfg_attr(
            target_env = "msvc",
            link_name = "?llama_batch_ext_add_embd_row@@YAHPEAUllama_batch_ext@@HH@Z"
        )]
        fn add_row(batch: *mut Opaque, seq: i32, row: i32) -> i32;
    }

    unsafe extern "C" fn batch_init(ctx: *mut Opaque) -> *mut Opaque {
        unsafe { raw::llama_batch_ext_init(ctx.cast()).cast() }
    }
    unsafe extern "C" fn batch_free(batch: *mut Opaque) {
        unsafe { raw::llama_batch_ext_free(batch.cast()) }
    }
    unsafe extern "C" fn batch_clear(batch: *mut Opaque) {
        unsafe { raw::llama_batch_ext_clear(batch.cast()) }
    }
    unsafe extern "C" fn set_pos(batch: *mut Opaque, idx: i32, pos: *const i32) -> bool {
        unsafe { raw::llama_batch_ext_set_pos(batch.cast(), idx, pos) }
    }
    unsafe extern "C" fn set_output(batch: *mut Opaque, idx: i32, value: bool) -> bool {
        unsafe { raw::llama_batch_ext_set_output_logits(batch.cast(), idx, value) }
    }
    unsafe extern "C" fn process(
        ctx: *mut Opaque,
        mode: LlamaProcessType,
        batch: *mut Opaque,
    ) -> i32 {
        unsafe { raw::llama_process(ctx.cast(), mode, batch.cast()) }
    }
    unsafe extern "C" fn synchronize(ctx: *mut Opaque) {
        unsafe { raw::llama_synchronize(ctx.cast()) }
    }

    pub(super) const API: LlamaDraftDeviceApi = LlamaDraftDeviceApi {
        set_capture,
        supports_rows,
        batch_init,
        batch_free,
        batch_clear,
        add_row,
        set_pos,
        set_output,
        process,
        synchronize,
    };
}

#[cfg(not(feature = "dynamic-runtime"))]
pub(super) const fn static_api() -> LlamaDraftDeviceApi {
    linked::API
}
