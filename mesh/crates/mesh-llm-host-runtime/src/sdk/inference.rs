//! In-process inference using a node's existing loaded model and frontend.

use anyhow::{Context, Result};
use axum::{body::Body, http::Request};
use tokio::sync::{mpsc, oneshot, watch};
use tower::ServiceExt;

use super::{EmbeddedOpenAiStream, EmbeddedServeHandle};
use crate::api::RuntimeControlRequest;

pub use skippy_inference_api::{ChatCompletionRequest, ChatMessage, MessageContent};

/// A cloneable client for this node's currently loaded local models.
///
/// Requests use the same frontend, guardrails and lifecycle observers as the
/// model's server without opening a connection or loading another model.
/// Model selection is refreshed per request; retaining this client does not
/// retain a loaded model across unload or node shutdown.
#[derive(Clone)]
pub struct InferenceClient {
    control_tx: mpsc::UnboundedSender<RuntimeControlRequest>,
    shutdown: watch::Receiver<bool>,
}

impl EmbeddedServeHandle {
    pub fn inference_client(&self) -> Result<InferenceClient> {
        Ok(InferenceClient {
            control_tx: self
                .control_tx
                .clone()
                .context("embedded control channel unavailable")?,
            shutdown: self.shutdown.subscribe(),
        })
    }
}

impl InferenceClient {
    /// Starts a streaming completion for an exact, locally loaded serving ID.
    /// Dropping the returned stream cancels its frontend request. Node shutdown
    /// also interrupts pending dispatch and stream reads.
    pub async fn chat_completion_stream(
        &self,
        mut request: ChatCompletionRequest,
    ) -> Result<EmbeddedOpenAiStream> {
        let mut shutdown = self.shutdown.clone();
        anyhow::ensure!(
            !*shutdown.borrow_and_update(),
            "embedded runtime is shutting down"
        );
        let dispatch = async {
            let (resp, response) = oneshot::channel();
            self.control_tx
                .send(RuntimeControlRequest::InferenceRouter {
                    model: request.model.clone(),
                    resp,
                })
                .context("embedded control channel closed")?;
            let router = response
                .await
                .context("embedded inference response dropped")?
                .context("model is not loaded locally for native inference")?;
            request.stream = true;
            let request = Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request)?))?;
            let response = router.oneshot(request).await?;
            Ok(EmbeddedOpenAiStream::from_response(response).with_shutdown(self.shutdown.clone()))
        };
        tokio::select! {
            biased;
            _ = shutdown.changed() => anyhow::bail!("embedded runtime is shutting down"),
            result = dispatch => result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_inference_api::{
        CancellationToken, ChatCompletionChunk, ChatCompletionResponse, ChatCompletionStream,
        InferenceBackend, InferenceRequestContext, InferenceResult, ModelObject,
    };
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    struct Backend {
        request: Mutex<Option<ChatCompletionRequest>>,
        token: Mutex<Option<CancellationToken>>,
        chunks: Mutex<Option<mpsc::UnboundedReceiver<InferenceResult<ChatCompletionChunk>>>>,
    }

    #[async_trait::async_trait]
    impl InferenceBackend for Backend {
        async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
            Ok(vec![ModelObject::new("exact-model")])
        }

        async fn chat_completion(
            &self,
            _: ChatCompletionRequest,
        ) -> InferenceResult<ChatCompletionResponse> {
            unreachable!("native client always streams")
        }

        async fn chat_completion_stream(
            &self,
            request: ChatCompletionRequest,
            context: InferenceRequestContext,
        ) -> InferenceResult<ChatCompletionStream> {
            *self.request.lock().unwrap() = Some(request);
            *self.token.lock().unwrap() = Some(context.cancellation_token());
            Ok(Box::pin(
                tokio_stream::wrappers::UnboundedReceiverStream::new(
                    self.chunks.lock().unwrap().take().unwrap(),
                ),
            ))
        }
    }

    fn request() -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "exact-model", "messages": [{"role": "user", "content": "hello"}],
            "chat_template_kwargs": {"enable_thinking": false}, "id_slot": 3,
        }))
        .unwrap()
    }

    #[tokio::test]
    async fn native_inference_streams_incrementally_and_drop_cancels_the_frontend() {
        let (control_tx, mut control_rx) = mpsc::unbounded_channel();
        let (_shutdown, shutdown) = watch::channel(false);
        let client = InferenceClient {
            control_tx,
            shutdown,
        };
        let (chunks, receiver) = mpsc::unbounded_channel();
        let backend = Arc::new(Backend {
            request: Mutex::new(None),
            token: Mutex::new(None),
            chunks: Mutex::new(Some(receiver)),
        });
        let serve = async {
            let Some(RuntimeControlRequest::InferenceRouter { model, resp }) =
                control_rx.recv().await
            else {
                panic!("expected native inference dispatch");
            };
            assert_eq!(model, "exact-model");
            resp.send(Some(skippy_inference_api::router_for(backend.clone())))
                .unwrap();
        };
        let (response, ()) = tokio::join!(client.chat_completion_stream(request()), serve);
        let mut response = response.unwrap();
        assert!(response.is_success());
        assert!(response.is_event_stream());
        let captured = backend.request.lock().unwrap().clone().unwrap();
        assert!(captured.stream);
        assert_eq!(captured.extra["id_slot"], 3);
        assert_eq!(
            captured.extra["chat_template_kwargs"]["enable_thinking"],
            false
        );
        chunks
            .send(Ok(ChatCompletionChunk::delta("exact-model", "first")))
            .unwrap();
        let data = tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let event = response.next_event().await.unwrap().unwrap();
                if event.data.contains("first") {
                    break event.data;
                }
            }
        })
        .await
        .expect("first chunk must arrive before backend completion");
        assert!(data.contains("first"));
        drop(response);
        assert!(
            backend
                .token
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .is_cancelled()
        );
        assert!(chunks.is_closed());
    }

    #[tokio::test]
    async fn native_inference_reselects_models_and_shutdown_releases_dispatch() {
        let (control_tx, mut control_rx) = mpsc::unbounded_channel();
        let (shutdown_tx, shutdown) = watch::channel(false);
        let client = InferenceClient {
            control_tx,
            shutdown,
        };
        let missing = async {
            let Some(RuntimeControlRequest::InferenceRouter { resp, .. }) = control_rx.recv().await
            else {
                panic!("expected native dispatch");
            };
            resp.send(None).unwrap();
        };
        let (response, ()) = tokio::join!(client.chat_completion_stream(request()), missing);
        assert!(
            response
                .err()
                .unwrap()
                .to_string()
                .contains("not loaded locally")
        );
        let stopping = async {
            let Some(RuntimeControlRequest::InferenceRouter { resp, .. }) = control_rx.recv().await
            else {
                panic!("each request must reselect its model");
            };
            shutdown_tx.send_replace(true);
            resp
        };
        let (response, pending) = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(client.chat_completion_stream(request()), stopping)
        })
        .await
        .unwrap();
        assert!(
            response
                .err()
                .unwrap()
                .to_string()
                .contains("shutting down")
        );
        assert!(pending.is_closed());
        assert!(client.chat_completion_stream(request()).await.is_err());
    }

    #[tokio::test]
    async fn node_shutdown_interrupts_an_idle_stream_read() {
        let (shutdown_tx, shutdown) = watch::channel(false);
        let body = Body::from_stream(futures_util::stream::pending::<
            Result<axum::body::Bytes, std::io::Error>,
        >());
        let response = axum::http::Response::builder()
            .header("content-type", "text/event-stream")
            .header("x-request-id", "native-request")
            .body(body)
            .unwrap();
        let mut stream = EmbeddedOpenAiStream::from_response(response).with_shutdown(shutdown);
        assert_eq!(stream.request_id(), Some("native-request"));
        let stop = async {
            tokio::task::yield_now().await;
            shutdown_tx.send_replace(true);
        };
        let (result, ()) = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(stream.next_event(), stop)
        })
        .await
        .expect("shutdown must interrupt a pending body read");
        assert!(result.err().unwrap().to_string().contains("shutting down"));
    }
}
