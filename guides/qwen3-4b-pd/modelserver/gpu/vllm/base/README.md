# Qwen3-4B prefill/decode

Qwen3-4B-Instruct-2507 served as a prefill/decode pair, one GPU per rank,
through the RHOAI operator: the `LLMInferenceService` declares the decode
template plus a `prefill` stanza, and the operator builds both workloads, the
EPP that pairs a prefill rank with each decode request, and the route on the
shared gateway.

The decode container's `args` is an empty list on purpose: a profiling campaign
appends its profiler flags there, and the template's args are passed through to
`vllm serve`. The prefill stanza carries its profiler flags inline, since only
the decode template is reachable from the campaign's overlay machinery.

Both ranks carry `llm-d.ai/role` (`prefill` / `decode`) and share the service
name in `app.kubernetes.io/name`, which is what a campaign selects pods on when
it collects traces from both of them.
