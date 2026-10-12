# Qwen3-30B-A3B on H200

## Overview

This guide deploys [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) (30.5B MoE,
3.3B activated) on H200 GPUs using a P/D-disaggregated DisaggregatedSet with NIXL for KV
transfer. Prefill and decode both run data-parallel experts (TP=1): DeepEP high-throughput
all-to-all for prefill, low-latency for decode. Each role takes 4 GPUs, so the whole
deployment — prefill and decode together — fits on a single 8-GPU H200 node.

Qwen3-30B-A3B is far smaller than the other wide-EP model servers
([GLM-5.2-FP8](../vllm-glm-5.2), [DeepSeek-V4-Pro](../vllm-deepseek-v4)), which makes it a
cheap way to exercise the wide-EP machinery — P/D disaggregation, rollouts, and the
aiperf concurrency-sweep automation — without reserving a large cluster. It has no MTP
layers and no KV-cache offloading, so there are no opt-in components: the base
configuration is the only configuration.

Tested on CoreWeave (CKS) with InfiniBand networking. This recipe reuses the
[wide-ep guide](../../../README.md) for the router/gateway and shared prerequisites
(namespace, HF token secret, LeaderWorkerSet controller with DisaggregatedSet enabled).

## Default Configuration

| Parameter               | Value                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------- |
| Model                   | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)                    |
| Accelerator             | NVIDIA H200 (8 GPUs per node)                                                      |
| DP model                | Supervisor (`--data-parallel-multi-port-external-lb`)                              |
| Prefill parallelism     | TP=1, DP=4, EP=4 (DEP4) — 4 GPUs, co-located with decode                            |
| Decode parallelism      | TP=1, DP=4, EP=4 (DEP4) — 4 GPUs, co-located with prefill                           |
| All-to-all (prefill)    | `deepep_high_throughput`                                                           |
| All-to-all (decode)     | `deepep_low_latency` (IBGDA + NVSHMEM)                                             |
| KV transfer             | NixlConnector                                                                      |
| Context length          | 32,768 native (`MAX_MODEL_LEN=auto` → model default 40,960)                        |
| Reasoning / tool-call   | `deepseek_r1` / `hermes`                                                           |

### P/D Deployment Options

| Deployment | Prefill                   | Decode                    | Nodes / GPUs |
| ---------- | ------------------------- | ------------------------- | ------------ |
| `p1w1d1w1` | 1 replica, DEP4 (4 GPUs)  | 1 replica, DEP4 (4 GPUs)  | 1 / 8        |

Prefill and decode are co-located on the same node: the deployment needs one node with 8
free GPUs and 8 `rdma/ib` devices.

### Supported Hardware Backends

| Backend             | Directory                                                      | Notes                                    |
| ------------------- | -------------------------------------------------------------- | ---------------------------------------- |
| NVIDIA GPU (vLLM)   | `wide-ep/modelserver/gpu/vllm-qwen3-30b-a3b/`                  | H200, P/D disaggregated                  |

## Prerequisites

In addition to the [wide-ep prerequisites](../../../README.md#prerequisites):

```bash
export KUBECONFIG=~/.kube/config
export NAMESPACE=<your-namespace>
export MODEL=Qwen/Qwen3-30B-A3B
```

## Deploy the Model Server

### P/D Disaggregated

Pick a deployment from the [P/D Deployment Options](#pd-deployment-options) table and apply:

```bash
kubectl apply -n ${NAMESPACE} -k deployments/<deployment>
```

Wait for pods to become ready (model load takes a few minutes; the startup probe allows up
to 10):

```bash
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/model=Qwen3-30B-A3B -w
```

## EPP Routing

The wide-ep router defaults advertise 8 DP rank ports (8000-8007) and scope the
DisaggregatedSet rollout screener to the DeepSeek-R1 model server. This guide runs DP=4 per
role on its own DisaggregatedSet, so include the Qwen3-30B-A3B router override after
`wide-ep.values.yaml` when installing the router (see the
[wide-ep router step](../../../README.md#1-deploy-the-llm-d-router)):

```bash
helm install ${GUIDE_NAME} \
    ${ROUTER_STANDALONE_CHART} \
    -f ${REPO_ROOT}/guides/recipes/router/base.values.yaml \
    -f ${REPO_ROOT}/guides/wide-ep/router/wide-ep.values.yaml \
    -f ${REPO_ROOT}/guides/wide-ep/router/qwen3-30b-a3b-overrides.values.yaml \
    -n ${NAMESPACE} --version ${ROUTER_CHART_VERSION}
```

The override restricts `targetPorts` to the 4 DP rank ports (8000-8003) that exist on both
roles and points the rollout screener at this guide's DisaggregatedSet
(`wide-ep-vllm-qwen3-30b-a3b`).

## Verification

### 1. Get the IP of the Proxy

```bash
export IP=$(kubectl get service wide-ep-epp -n ${NAMESPACE} -o jsonpath='{.spec.clusterIP}')
```

### 2. Send Test Requests

Open a temporary shell inside the cluster:

```bash
kubectl run curl-debug --rm -it \
    --image=cfmanteiga/alpine-bash-curl-jq \
    --env="IP=$IP" \
    --env="NAMESPACE=$NAMESPACE" \
    -- /bin/bash
```

Send a completion request:

```bash
curl -X POST http://${IP}/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "Qwen/Qwen3-30B-A3B",
        "prompt": "How are you today?"
    }' | jq
```

## Benchmarking

This model server exists to exercise the same [aiperf](https://github.com/ai-dynamo/aiperf)
concurrency-sweep automation that backs the
[agentic-serving GLM-5.2 guide](../../../../agentic-serving/glm-5-2-h200.md), on a smaller
and cheaper cluster footprint. Run from inside the cluster against the Kubernetes Gateway
Service (`llm-d-inference-gateway-istio`, from [Gateway Mode](../../../README.md#gateway-mode)
with `PROVIDER_NAME=istio`) and sweep `--concurrency` across the levels under test. The
dataset is the same `semianalysis_cc_traces_weka_with_subagents` aiperf preset, backed by
[`semianalysisai/cc-traces-weka-062126`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126)
on HuggingFace.

```bash
aiperf profile \
    --scenario 'inferencex-agentx-mvp' \
    --url 'http://llm-d-inference-gateway-istio:80/v1' \
    --model 'Qwen/Qwen3-30B-A3B' \
    --max-context-length 28000 \
    --endpoint-type 'chat' \
    --streaming \
    --use-server-token-count \
    --public-dataset 'semianalysis_cc_traces_weka_with_subagents' \
    --concurrency <1|8|32|64> \
    --random-seed 42 \
    --benchmark-duration 900 \
    --server-metrics 'http://llm-d-inference-gateway-istio:80/metrics' \
    --no-gpu-telemetry \
    --output-artifact-dir <path> \
    --ui 'simple'
```

Only the model and the context bound differ from the GLM-5.2 sweep. Qwen3-30B-A3B has a
32,768-token native context window (the model default `MAX_MODEL_LEN=auto` resolves to
40,960), so `--max-context-length 28000` truncates the trace dataset to requests that fit
with headroom for output. To replay longer traces, raise `MAX_MODEL_LEN` on both roles and
add YaRN rope scaling
(`--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'`,
see [Processing Long Texts](https://huggingface.co/Qwen/Qwen3-30B-A3B#processing-long-texts)).

A suggested shakeout sweep is `--concurrency` 1, 8, 32, 64 at `--benchmark-duration 900`.

## InfiniBand Networking

Both prefill and decode configure IB for multi-node communication:

| Variable                  | Value  | Purpose                                          |
| ------------------------- | ------ | ------------------------------------------------ |
| `NCCL_IB_HCA`            | `ibp`  | Filter IB HCAs for NCCL collectives              |
| `NVSHMEM_HCA_PREFIX`      | `ibp`  | Filter IB HCAs for NVSHMEM (decode low-latency)  |
| `NVSHMEM_REMOTE_TRANSPORT` | `ibgda` | GPUDirect Async for NVSHMEM                     |
| `rdma/ib`                 | `4`    | Request 4 RDMA/IB devices per pod (8 per node)  |

## Monitoring

Node-exporter sidecars on each pod collect InfiniBand, CPU, memory pressure, and network
retransmission metrics. Apply Prometheus scrape configs:

```bash
NAMESPACE=${NAMESPACE} bash guides/wide-ep/monitoring/apply-scrape-configs.sh
```

## Cleanup

```bash
kubectl delete -n ${NAMESPACE} -k deployments/<deployment>
```
