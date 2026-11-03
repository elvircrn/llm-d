# Pirate (RDU4) — Qwen3-4B-Instruct aggregated serving

Single-GPU aggregated vLLM serving of `Qwen/Qwen3-4B-Instruct-2507` on the
Pirate (RDU4) OpenShift cluster's A100-SXM4-80GB nodes, fronted by the
llm-d standalone router. Consumed by the crucible-domains campaign
`campaign.qwen3-4b-a100-pirate.json`; the campaign replaces the `vllm`
container image from its config's `vllm_image`.

## Layout

- `base/` — StatefulSet (1 replica, 1 GPU) plus its headless Service. The
  storage-initializer pulls the ungated model from Hugging Face into an
  emptyDir; no HF token Secret is required.
- `../../router/qwen3-4b.values.yaml` — router chart values selecting these
  pods via `llm-d.ai/model` and exposing the Envoy data plane on port 80.

## Cluster constraints

- The RHOAI registry images are pinned by digest because they are the only
  vLLM images proven to satisfy the namespace's SecurityContextConstraints
  (non-root). Upstream `vllm/vllm-openai` images run as UID 0 and are
  refused admission.
- Serving pods label themselves `llm-d.ai/inference-serving=true` and
  `llm-d.ai/model=Qwen3-4B-Instruct`; the router's InferencePool and the
  campaign's readiness check both select on those labels.
