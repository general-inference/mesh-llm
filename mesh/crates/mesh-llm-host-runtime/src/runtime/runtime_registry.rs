use super::capacity::{
    RuntimeCapacityLedger, RuntimeCapacityPool, RuntimeCapacityRequest, RuntimeCapacityReservation,
    runtime_model_required_bytes,
};
use super::{
    StartupPinnedGpuTarget, add_serving_assignment, advertise_model_ready,
    remove_serving_assignment, set_advertised_model_context,
    set_runtime_verified_served_model_capabilities, withdraw_advertised_model,
};
use crate::{mesh, models};
use anyhow::Result;
use skippy_inference_api::thinking::ThinkingControls;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub(super) type RuntimeInstanceRegistry =
    Arc<tokio::sync::Mutex<HashMap<String, BTreeMap<String, RuntimeInstanceMetadata>>>>;

pub(super) struct RuntimeInstanceMetadata {
    inference_router: Option<axum::Router>,
    context_length: Option<u32>,
    thinking: Option<ThinkingControls>,
}

pub(super) struct RuntimeModelRegistration<'a> {
    pub inference_router: Option<axum::Router>,
    pub primary_model_name: &'a str,
    pub model_name: &'a str,
    pub instance_id: &'a str,
    pub context_length: Option<u32>,
    pub thinking: Option<&'a ThinkingControls>,
    pub capabilities: models::ModelCapabilities,
    pub workload_class: mesh::ModelWorkloadClass,
}

pub(super) fn next_runtime_instance_id(next_sequence: &mut u64) -> String {
    let instance_id = format!("runtime-{}", *next_sequence);
    *next_sequence = next_sequence.saturating_add(1);
    instance_id
}

pub(super) fn runtime_capacity_pool(
    pinned_gpu: Option<&StartupPinnedGpuTarget>,
) -> RuntimeCapacityPool {
    pinned_gpu
        .map(|gpu| RuntimeCapacityPool::PinnedGpu(gpu.stable_id.clone()))
        .unwrap_or(RuntimeCapacityPool::Node)
}

pub(super) fn runtime_capacity_request_for_model(
    instance_id: &str,
    model_name: &str,
    pinned_gpu: Option<&StartupPinnedGpuTarget>,
    capacity_bytes: u64,
    model_bytes: u64,
) -> RuntimeCapacityRequest {
    RuntimeCapacityRequest {
        instance_id: instance_id.to_string(),
        model_name: model_name.to_string(),
        pool: runtime_capacity_pool(pinned_gpu),
        capacity_bytes,
        required_bytes: runtime_model_required_bytes(model_bytes),
    }
}

pub(super) fn reserve_runtime_capacity_for_model(
    ledger: &RuntimeCapacityLedger,
    instance_id: &str,
    model_name: &str,
    pinned_gpu: Option<&StartupPinnedGpuTarget>,
    capacity_bytes: u64,
    model_bytes: u64,
) -> Result<RuntimeCapacityReservation> {
    ledger
        .reserve(runtime_capacity_request_for_model(
            instance_id,
            model_name,
            pinned_gpu,
            capacity_bytes,
            model_bytes,
        ))
        .map_err(Into::into)
}

pub(super) async fn register_runtime_instance(
    registry: &RuntimeInstanceRegistry,
    node: &mesh::Node,
    registration: RuntimeModelRegistration<'_>,
) {
    let RuntimeModelRegistration {
        inference_router,
        primary_model_name,
        model_name,
        instance_id,
        context_length,
        thinking,
        capabilities,
        workload_class,
    } = registration;
    let (was_empty, context_changed, next_context) = {
        let mut guard = registry.lock().await;
        let instances = guard.entry(model_name.to_string()).or_default();
        let previous_context = runtime_registry_model_context(instances);
        let was_empty = instances.is_empty();
        instances.insert(
            instance_id.to_string(),
            RuntimeInstanceMetadata {
                inference_router,
                context_length,
                thinking: thinking.cloned(),
            },
        );
        let next_context = runtime_registry_model_context(instances);
        (was_empty, previous_context != next_context, next_context)
    };

    if context_changed {
        set_advertised_model_context(node, model_name, next_context).await;
    }
    if was_empty {
        add_serving_assignment(node, primary_model_name, model_name).await;
        set_runtime_verified_served_model_capabilities(
            node,
            primary_model_name,
            model_name,
            capabilities,
            workload_class,
        )
        .await;
        advertise_model_ready(node, primary_model_name, model_name, "").await;
    }
}

pub(super) async fn unregister_runtime_instance(
    registry: &RuntimeInstanceRegistry,
    node: &mesh::Node,
    model_name: &str,
    instance_id: &str,
) -> bool {
    let (removed, became_empty, context_changed, next_context) = {
        let mut guard = registry.lock().await;
        let Some(instances) = guard.get_mut(model_name) else {
            return false;
        };
        let previous_context = runtime_registry_model_context(instances);
        let removed = instances.remove(instance_id).is_some();
        let next_context = runtime_registry_model_context(instances);
        let became_empty = instances.is_empty();
        if became_empty {
            guard.remove(model_name);
        }
        (
            removed,
            became_empty,
            previous_context != next_context,
            next_context,
        )
    };

    if !removed {
        return false;
    }
    if became_empty {
        set_advertised_model_context(node, model_name, None).await;
        withdraw_advertised_model(node, model_name, "").await;
        remove_serving_assignment(node, model_name).await;
        crate::inference::skippy::forget_local_thinking(model_name);
        true
    } else {
        if context_changed {
            set_advertised_model_context(node, model_name, next_context).await;
        }
        false
    }
}

pub(super) async fn runtime_registry_has_model(
    registry: &RuntimeInstanceRegistry,
    model_name: &str,
) -> bool {
    registry
        .lock()
        .await
        .get(model_name)
        .map(|instances| !instances.is_empty())
        .unwrap_or(false)
}

pub(super) fn runtime_registry_model_context(
    instances: &BTreeMap<String, RuntimeInstanceMetadata>,
) -> Option<u32> {
    instances
        .values()
        .filter_map(|instance| instance.context_length)
        .max()
}

pub(super) async fn runtime_registry_thinking_controls(
    registry: &RuntimeInstanceRegistry,
    model_name: &str,
) -> Option<ThinkingControls> {
    let guard = registry.lock().await;
    let instances = guard.get(model_name)?;
    let controls = instances.values().next()?.thinking.as_ref()?;
    // A model ID can route to several profiles. Only offer a control when the
    // same observations cover every instance that could receive the request.
    instances
        .values()
        .all(|instance| instance.thinking.as_ref() == Some(controls))
        .then(|| controls.clone())
}

pub(super) async fn runtime_registry_inference_router(
    registry: &RuntimeInstanceRegistry,
    model_name: &str,
) -> Option<axum::Router> {
    registry
        .lock()
        .await
        .get(model_name)?
        .values()
        .find_map(|instance| instance.inference_router.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_inference_api::thinking::ThinkingToggle;

    fn observed(efforts: &[&str]) -> RuntimeInstanceMetadata {
        RuntimeInstanceMetadata {
            inference_router: None,
            context_length: Some(8192),
            thinking: Some(ThinkingControls {
                toggle: ThinkingToggle::Observed,
                reasoning_parser: Some(true),
                efforts: efforts.iter().map(|effort| (*effort).into()).collect(),
            }),
        }
    }

    #[tokio::test]
    async fn thinking_controls_follow_the_current_model_and_node() {
        let first = observed(&["high", "low"]);
        let expected = first.thinking.clone();
        let registry = Arc::new(tokio::sync::Mutex::new(HashMap::from([
            ("first".into(), BTreeMap::from([("one".into(), first)])),
            (
                "second".into(),
                BTreeMap::from([("two".into(), observed(&["max"]))]),
            ),
            (
                "unprobed".into(),
                BTreeMap::from([(
                    "three".into(),
                    RuntimeInstanceMetadata {
                        inference_router: None,
                        context_length: None,
                        thinking: None,
                    },
                )]),
            ),
        ])));
        assert_eq!(
            runtime_registry_thinking_controls(&registry, "first").await,
            expected
        );
        let second = runtime_registry_thinking_controls(&registry, "second")
            .await
            .unwrap();
        assert_eq!(second.efforts, ["max"]);
        assert!(
            runtime_registry_thinking_controls(&registry, "unprobed")
                .await
                .is_none()
        );
        assert!(
            runtime_registry_thinking_controls(&registry, "missing")
                .await
                .is_none()
        );
        let other_node = Arc::default();
        assert!(
            runtime_registry_thinking_controls(&other_node, "first")
                .await
                .is_none()
        );

        registry
            .lock()
            .await
            .get_mut("first")
            .unwrap()
            .insert("one".into(), observed(&["medium"]));
        let reloaded = runtime_registry_thinking_controls(&registry, "first")
            .await
            .unwrap();
        assert_eq!(reloaded.efforts, ["medium"]);
        registry
            .lock()
            .await
            .get_mut("first")
            .unwrap()
            .get_mut("one")
            .unwrap()
            .thinking = None;
        assert!(
            runtime_registry_thinking_controls(&registry, "first")
                .await
                .is_none()
        );
        registry.lock().await.remove("second");
        assert!(
            runtime_registry_thinking_controls(&registry, "second")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn thinking_controls_require_agreement_across_loaded_instances() {
        let registry = Arc::new(tokio::sync::Mutex::new(HashMap::from([(
            "model".into(),
            BTreeMap::from([
                ("one".into(), observed(&["low", "high"])),
                ("two".into(), observed(&["low", "high"])),
            ]),
        )])));
        assert!(
            runtime_registry_thinking_controls(&registry, "model")
                .await
                .is_some()
        );
        registry
            .lock()
            .await
            .get_mut("model")
            .unwrap()
            .insert("two".into(), observed(&["max"]));
        assert!(
            runtime_registry_thinking_controls(&registry, "model")
                .await
                .is_none()
        );
        registry
            .lock()
            .await
            .get_mut("model")
            .unwrap()
            .get_mut("two")
            .unwrap()
            .thinking = None;
        assert!(
            runtime_registry_thinking_controls(&registry, "model")
                .await
                .is_none()
        );
        registry
            .lock()
            .await
            .get_mut("model")
            .unwrap()
            .remove("two");
        assert!(
            runtime_registry_thinking_controls(&registry, "model")
                .await
                .is_some()
        );
    }
}
