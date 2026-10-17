# Nemotron-H-47B-Reasoning-128K on H200

## Overview

This guide deploys [Nemotron-H-47B-Reasoning-128K](https://huggingface.co/nvidia/Nemotron-H-47B-Reasoning-128K)
(47B Mamba-Transformer hybrid) on one 8-GPU H200 node with TP=8. The hybrid Mamba state
cannot ride a KV transfer connector, so unlike the other wide-EP model servers there is no
prefill/decode split: a single LeaderWorkerSet serves both phases, and the router layers on
`recipes/router/base.values.yaml` instead of the wide-EP P/D plugin stack.

Serving requires vLLM's Model Runner V2 (`VLLM_USE_V2_MODEL_RUNNER=1`), which the
`nan-fault-tolerance` vLLM branch also requires. The `tp8-h200-nanft` deployment variant
appends that branch's `--enable-nan-fault-tolerance` via the base template's
`${VLLM_EXTRA_ARGS:-}` anchor.

Tested on CoreWeave (CKS). This recipe reuses the
[wide-ep guide](../../../README.md) for shared prerequisites (namespace, HF token secret,
LeaderWorkerSet controller) and pairs with
[router/nemotron-h-47b-overrides.values.yaml](../../router/nemotron-h-47b-overrides.values.yaml).

## Default Configuration

| Parameter    | Value                                                                                    |
| ------------ | ---------------------------------------------------------------------------------------- |
| Model        | [nvidia/Nemotron-H-47B-Reasoning-128K](https://huggingface.co/nvidia/Nemotron-H-47B-Reasoning-128K) |
| Accelerator  | NVIDIA H200 (8 GPUs per node)                                                             |
| Parallelism  | TP=8, single LeaderWorkerSet (size 1)                                                     |
| Model runner | V2 (`VLLM_USE_V2_MODEL_RUNNER=1`)                                                         |
| Context      | 128K native; the campaign runs `MAX_MODEL_LEN=16384`                                      |

## Deployments

| Directory       | Description                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------- |
| `tp8-h200`      | Baseline: one TP=8 worker on one 8-GPU H200 node                                                     |
| `tp8-h200-nanft`| Same, with `--enable-nan-fault-tolerance` from the vLLM `nan-fault-tolerance` branch                  |
