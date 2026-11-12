//! Firecracker inside WSL2 (placeholder while the backend is written).

use anyhow::{Result, bail};

pub struct WslBackend;

impl WslBackend {
    pub async fn start(
        _settings: &crate::settings::Settings,
    ) -> Result<std::sync::Arc<dyn super::Backend>> {
        bail!("not yet")
    }
}
