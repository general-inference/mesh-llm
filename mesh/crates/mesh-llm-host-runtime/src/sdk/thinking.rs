//! Render-only thinking-control observations for locally loaded models.

use super::EmbeddedServeHandle;
use crate::api::RuntimeControlRequest;
use anyhow::Result;

pub use skippy_inference_api::thinking::{ThinkingControls, ThinkingToggle};

impl EmbeddedServeHandle {
    /// Queries this node's loaded model by its exact serving ID, in process.
    ///
    /// Returns the selected chat template's probe summary without HTTP or
    /// running inference. `None` means the model is absent, unprobed, or its
    /// loaded instances have different observations. It does not mean thinking
    /// is off. Effort values and their order are preserved from the probe.
    /// These observations do not guarantee that generation obeys the controls.
    ///
    /// # Errors
    /// Returns an error when the embedded runtime is shutting down or its
    /// control channel is unavailable.
    pub async fn thinking_controls(&self, serving_id: &str) -> Result<Option<ThinkingControls>> {
        let mut shutdown = self.shutdown.subscribe();
        anyhow::ensure!(
            !*shutdown.borrow_and_update(),
            "embedded runtime is shutting down"
        );
        let model = serving_id.to_owned();
        tokio::select! {
            biased;
            _ = shutdown.changed() => anyhow::bail!("embedded runtime is shutting down"),
            result = self.control(|resp| RuntimeControlRequest::ThinkingControls { model, resp }) => result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::{mpsc, watch};

    fn test_handle(
        control_tx: mpsc::UnboundedSender<RuntimeControlRequest>,
    ) -> EmbeddedServeHandle {
        EmbeddedServeHandle {
            api_base_url: "http://invalid.invalid/v1".into(),
            console_url: "http://invalid.invalid".into(),
            invite_token: None,
            control_tx: Some(control_tx),
            task: None,
            shutdown: watch::channel(false).0,
        }
    }

    #[tokio::test]
    async fn thinking_controls_use_the_typed_channel_without_http() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let handle = test_handle(tx);
        let summary = ThinkingControls {
            toggle: ThinkingToggle::NotObserved,
            reasoning_parser: Some(true),
            efforts: vec!["max".into(), "low".into()],
        };
        for expected in [Some(summary), None] {
            let request = handle.thinking_controls("exact-serving-id");
            let respond = async {
                let Some(RuntimeControlRequest::ThinkingControls { model, resp }) = rx.recv().await
                else {
                    panic!("expected typed thinking controls request");
                };
                assert_eq!(model, "exact-serving-id");
                resp.send(expected.clone()).unwrap();
            };
            let (actual, ()) = tokio::time::timeout(Duration::from_secs(1), async {
                tokio::join!(request, respond)
            })
            .await
            .expect("query must use the runtime control channel");
            assert_eq!(actual.unwrap(), expected);
        }
    }

    #[tokio::test]
    async fn thinking_controls_report_closed_or_dropped_control_channels() {
        let (tx, rx) = mpsc::unbounded_channel();
        let handle = test_handle(tx);
        drop(rx);
        assert!(
            handle
                .thinking_controls("model")
                .await
                .unwrap_err()
                .to_string()
                .contains("closed")
        );

        let (tx, mut rx) = mpsc::unbounded_channel();
        let handle = test_handle(tx);
        let request = handle.thinking_controls("model");
        let respond = async {
            let Some(RuntimeControlRequest::ThinkingControls { resp, .. }) = rx.recv().await else {
                panic!("expected typed thinking controls request");
            };
            drop(resp);
        };
        let (result, ()) = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(request, respond)
        })
        .await
        .expect("query must finish when its response is dropped");
        assert!(result.unwrap_err().to_string().contains("response dropped"));
    }

    #[tokio::test]
    async fn thinking_controls_do_not_wait_for_a_shutting_down_runtime() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let handle = test_handle(tx);
        let request = handle.thinking_controls("model");
        let shutdown = async {
            let Some(RuntimeControlRequest::ThinkingControls { resp, .. }) = rx.recv().await else {
                panic!("expected typed thinking controls request");
            };
            handle.shutdown.send_replace(true);
            resp
        };
        let (result, pending_response) = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(request, shutdown)
        })
        .await
        .expect("shutdown must release the pending query");
        assert!(result.unwrap_err().to_string().contains("shutting down"));
        assert!(pending_response.is_closed());
        assert!(handle.thinking_controls("model").await.is_err());
    }
}
