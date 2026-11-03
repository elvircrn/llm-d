# Pirate (RDU4) — Qwen3-4B-Instruct via the RHOAI operator

Single-GPU aggregated serving of `Qwen/Qwen3-4B-Instruct-2507` on the Pirate
(RDU4) OpenShift cluster's A100-SXM4-80GB nodes, deployed as an
`LLMInferenceService`. Consumed by the crucible-domains campaign
`campaign.qwen3-4b-a100-pirate.json`; the campaign stamps the pinned runtime
image from its config's `vllm_image` into the `main` container.

## Layout

- `base/` — the `LLMInferenceService` (1 replica, TP 1). The operator
  injects the storage-initializer, the workload Deployment, the EPP
  scheduler pod, the InferencePool, and the HTTPRoute on the shared
  `openshift-ai-inference` gateway in `openshift-ingress`.

## Cluster constraints

- Tenants hold no create on `InferencePool` or `Gateway`
  (`gateway.networking.k8s.io`), so the standalone llm-d router chart cannot
  deploy on this cluster; the operator-owned path above is the only one.
- The operator's webhook names the workload container `main`, injects the
  `vllm serve` command into it, and reads `VLLM_ADDITIONAL_ARGS` from it. An
  explicit image on that container is preserved.
- The template is a bare pod spec with no `metadata`, so pod labels are
  operator-chosen: workload pods carry `app.kubernetes.io/name` (the
  LLMInferenceService name) and `kserve.io/component=workload`. Campaigns
  select on those instead of `llm-d.ai/model`.
- Inference reaches the pods through the shared gateway Service at
  `http://openshift-ai-inference-openshift-default.openshift-ingress.svc.cluster.local/<namespace>/qwen3-4b/v1`.
- The RHOAI registry images are pinned by digest because they are the only
  vLLM images proven to satisfy the namespace's SecurityContextConstraints
  (non-root). Upstream `vllm/vllm-openai` images run as UID 0 and are
  refused admission.
