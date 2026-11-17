# Qwen3-4B prefill/decode

A single-host prefill/decode pair for Qwen3-4B-Instruct-2507, one GPU per role,
as two StatefulSets rather than a `DisaggregatedSet`: the pair runs on clusters
that carry neither that CRD nor the controller behind it.

- `qwen3-4b-prefill` runs the prompt pass and publishes its KV cache over NIXL
  (`kv_role: kv_producer`).
- `qwen3-4b-decode` reads that cache (`kv_role: kv_consumer`) and generates. A
  routing sidecar listens on port 8000 and speaks the vLLM NIXL protocol.
- Service `qwen3-4b` is the front door and lands on the sidecar; service
  `qwen3-4b-prefill` is headless for the rank lookup.

Both ranks carry `llm-d.ai/role` (`prefill` / `decode`), `llm-d.ai/model`, and
`llm-d.ai/inference-serving=true`, and both name their serving container `vllm`,
which is what a profiler-driven campaign keys on to place per-role flags.

Client traffic must arrive through something that picks the prefill rank per
request — the llm-d EPP (`InferencePool` + `Gateway`) or the routing sidecar
paired with it. Posting directly to the `qwen3-4b` service without that picker
bypasses the prompt pass, which is a decode-only benchmark, not a P/D one.
