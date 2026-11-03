# Pirate (RDU4) — Qwen3-4B-Instruct aggregated serving

Single-GPU aggregated vLLM serving of `Qwen/Qwen3-4B-Instruct-2507` on the
Pirate (RDU4) OpenShift cluster's A100-SXM4-80GB nodes, benchmarked through
the StatefulSet's own Service with no routing layer. The campaign path for
this cluster is the operator-owned `../../llmisvc/qwen3-4b/` overlay; this
direct variant exists for measuring the bare engine. The campaign replaces
the `vllm` container image from its config's `vllm_image`.

## Layout

- `base/` — StatefulSet (1 replica, 1 GPU) plus its headless Service. The
  storage-initializer pulls the ungated model from Hugging Face into an
  emptyDir; no HF token Secret is required.

## Cluster constraints

- The RHOAI registry images are pinned by digest because they are the only
  vLLM images proven to satisfy the namespace's SecurityContextConstraints
  (non-root). Upstream `vllm/vllm-openai` images run as UID 0 and are
  refused admission.
- Serving pods label themselves `llm-d.ai/inference-serving=true` and
  `llm-d.ai/model=Qwen3-4B-Instruct`; the router's InferencePool and the
  campaign's readiness check both select on those labels.
