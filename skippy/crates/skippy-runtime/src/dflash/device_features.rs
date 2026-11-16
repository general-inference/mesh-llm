//! Device-owned target capture rows, consumed before the next target decode.
use anyhow::{Context, Result, bail};
use skippy_ffi::{
    Opaque,
    llama_draft::{LlamaDraftDeviceApi, LlamaProcessType},
};

pub(super) struct DeviceFeatures {
    api: LlamaDraftDeviceApi,
    target: *mut Opaque,
    draft: *mut Opaque,
    batch: *mut Opaque,
    rows: u64,
    shape: FeatureShape,
}

#[derive(Clone, Copy)]
pub(super) struct FeatureShape {
    pub(super) ubatch: usize,
    pub(super) capacity: usize,
    pub(super) mrope: bool,
}

impl DeviceFeatures {
    pub(super) fn attach(
        api: LlamaDraftDeviceApi,
        target: *mut Opaque,
        draft: *mut Opaque,
        shape: FeatureShape,
    ) -> Option<Self> {
        unsafe {
            (api.set_capture)(target, true);
            if !(api.supports_rows)(draft) {
                (api.set_capture)(target, false);
                return None;
            }
            let batch = (api.batch_init)(draft);
            if batch.is_null() {
                (api.set_capture)(target, false);
                return None;
            }
            Some(Self {
                api,
                target,
                draft,
                batch,
                rows: 0,
                shape,
            })
        }
    }

    pub(super) fn ingest(&mut self, seq: i32, first: usize, positions: &[i32]) -> Result<()> {
        let FeatureShape {
            ubatch,
            capacity,
            mrope,
        } = self.shape;
        if first
            .checked_add(positions.len())
            .is_none_or(|end| end > capacity)
        {
            bail!("draft device features exceed the target capture buffer");
        }
        for (chunk, positions) in positions.chunks(ubatch.max(1)).enumerate() {
            unsafe { (self.api.batch_clear)(self.batch) };
            for (offset, &position) in positions.iter().enumerate() {
                let row = first + chunk * ubatch.max(1) + offset;
                let row = i32::try_from(row).context("draft feature row exceeds i32")?;
                let idx = unsafe { (self.api.add_row)(self.batch, seq, row) };
                if idx < 0 {
                    bail!("draft device feature row refused: {idx}");
                }
                let axes = if mrope {
                    [position, position, position, 0]
                } else {
                    [position, 0, 0, 0]
                };
                if !unsafe { (self.api.set_pos)(self.batch, idx, axes.as_ptr()) }
                    || !unsafe { (self.api.set_output)(self.batch, idx, false) }
                {
                    bail!("draft device feature position/output refused");
                }
            }
            let rc =
                unsafe { (self.api.process)(self.draft, LlamaProcessType::Decode, self.batch) };
            if rc != 0 {
                bail!("draft device feature injection failed: {rc}");
            }
            self.rows = self.rows.saturating_add(positions.len() as u64);
        }
        Ok(())
    }

    pub(super) fn rows(&self) -> u64 {
        self.rows
    }
}

impl Drop for DeviceFeatures {
    fn drop(&mut self) {
        unsafe {
            // The target buffer must outlive all queued draft reads.
            (self.api.synchronize)(self.draft);
            (self.api.batch_free)(self.batch);
            (self.api.set_capture)(self.target, false);
        }
    }
}

pub(super) fn eligible(
    start: u32,
    end: u32,
    target_layers: i32,
    local: &[bool],
    ids: &[u32],
) -> bool {
    start == 0
        && i32::try_from(end).is_ok_and(|end| end >= target_layers)
        && !ids.is_empty()
        && local.len() == ids.len()
        && local.iter().all(|local| *local)
        && ids.windows(2).all(|ids| ids[0] < ids[1])
}

pub(super) fn enabled() -> bool {
    std::env::var("SKIPPY_DFLASH_DEVICE_FEATURES").as_deref() == Ok("1")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Fixture {
        capture: bool,
        supported: bool,
        batch_live: bool,
        rows: Vec<i32>,
        positions: Vec<[i32; 4]>,
        consumed: Vec<(i32, [i32; 4])>,
        calls: usize,
        waits: usize,
    }
    unsafe extern "C" fn capture(ctx: *mut Opaque, enabled: bool) {
        unsafe { (*ctx.cast::<Fixture>()).capture = enabled };
    }
    unsafe extern "C" fn supported(ctx: *const Opaque) -> bool {
        unsafe { (*ctx.cast::<Fixture>()).supported }
    }
    unsafe extern "C" fn init(ctx: *mut Opaque) -> *mut Opaque {
        unsafe { (*ctx.cast::<Fixture>()).batch_live = true };
        ctx
    }
    unsafe extern "C" fn free(ctx: *mut Opaque) {
        unsafe { (*ctx.cast::<Fixture>()).batch_live = false };
    }
    unsafe extern "C" fn clear(ctx: *mut Opaque) {
        let ctx = unsafe { &mut *ctx.cast::<Fixture>() };
        ctx.rows.clear();
        ctx.positions.clear();
    }
    unsafe extern "C" fn add(ctx: *mut Opaque, seq: i32, row: i32) -> i32 {
        assert_eq!(seq, 1);
        let ctx = unsafe { &mut *ctx.cast::<Fixture>() };
        let idx = ctx.rows.len() as i32;
        ctx.rows.push(row);
        ctx.positions.push([0; 4]);
        idx
    }
    unsafe extern "C" fn pos(ctx: *mut Opaque, idx: i32, axes: *const i32) -> bool {
        let ctx = unsafe { &mut *ctx.cast::<Fixture>() };
        ctx.positions[idx as usize] = unsafe { *axes.cast::<[i32; 4]>() };
        true
    }
    unsafe extern "C" fn output(_: *mut Opaque, _: i32, enabled: bool) -> bool {
        assert!(!enabled);
        true
    }
    unsafe extern "C" fn process(ctx: *mut Opaque, mode: LlamaProcessType, _: *mut Opaque) -> i32 {
        assert_eq!(mode, LlamaProcessType::Decode);
        let ctx = unsafe { &mut *ctx.cast::<Fixture>() };
        ctx.calls += 1;
        ctx.consumed
            .extend(ctx.rows.iter().copied().zip(ctx.positions.iter().copied()));
        0
    }
    unsafe extern "C" fn sync(ctx: *mut Opaque) {
        unsafe { (*ctx.cast::<Fixture>()).waits += 1 };
    }
    fn api() -> LlamaDraftDeviceApi {
        LlamaDraftDeviceApi {
            set_capture: capture,
            supports_rows: supported,
            batch_init: init,
            batch_free: free,
            batch_clear: clear,
            add_row: add,
            set_pos: pos,
            set_output: output,
            process,
            synchronize: sync,
        }
    }
    fn shape() -> FeatureShape {
        FeatureShape {
            ubatch: 2,
            capacity: 10,
            mrope: true,
        }
    }

    #[test]
    fn device_rows_preserve_offsets_positions_and_reader_lifetime() {
        let mut fixture = Fixture {
            supported: true,
            ..Default::default()
        };
        let ctx = (&raw mut fixture).cast();
        let mut device = DeviceFeatures::attach(api(), ctx, ctx, shape()).unwrap();
        device.ingest(1, 3, &[40, 41, 42]).unwrap();
        assert_eq!(
            fixture.consumed,
            vec![
                (3, [40, 40, 40, 0]),
                (4, [41, 41, 41, 0]),
                (5, [42, 42, 42, 0])
            ]
        );
        assert_eq!(fixture.calls, 2);
        assert_eq!(device.rows(), 3);
        assert!(device.ingest(1, 9, &[43, 44]).is_err());
        assert_eq!(fixture.calls, 2);
        assert!(fixture.capture && fixture.batch_live);
        drop(device);
        assert!(!fixture.capture && !fixture.batch_live);
        assert_eq!(fixture.waits, 1);
    }

    #[test]
    fn unsupported_device_restores_host_capture() {
        let mut fixture = Fixture::default();
        let ctx = (&raw mut fixture).cast();
        assert!(DeviceFeatures::attach(api(), ctx, ctx, shape()).is_none());
        assert!(!fixture.capture && !fixture.batch_live);
    }
    #[test]
    fn only_complete_local_ordered_captures_are_eligible() {
        assert!(eligible(0, 40, 40, &[true, true], &[3, 8]));
        assert!(!eligible(1, 40, 40, &[true, true], &[3, 8]));
        assert!(!eligible(0, 20, 40, &[true, true], &[3, 8]));
        assert!(!eligible(0, 40, 40, &[true, false], &[3, 8]));
        assert!(!eligible(0, 40, 40, &[true, true], &[8, 3]));
        assert!(!eligible(0, 40, 40, &[true, true], &[3, 3]));
        assert!(!eligible(0, 40, 40, &[], &[]));
    }
}
