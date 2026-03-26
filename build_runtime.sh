#!/bin/bash
set -euo pipefail

VLLM_COMMIT_SHA="${1:?Usage: ./build_runtime.sh <vllm_commit_sha>}"

# Update Dockerfile with new commit
sed -i "s|VLLM_COMMIT_SHA=\".*\"|VLLM_COMMIT_SHA=\"${VLLM_COMMIT_SHA}\"|" docker/Dockerfile.cuda

IMAGE_TAG="quay.io/rh-ee-ecrncevi/llm-dev-cuda13:v0.5.0-arm64-upstream-${VLLM_COMMIT_SHA}"

echo "Building runtime with VLLM_COMMIT_SHA=${VLLM_COMMIT_SHA}"
echo "Image tag: ${IMAGE_TAG}"

TMPDIR=~/podman-tmp2 VLLM_PREBUILT=0 podman build --security-opt label=disable \
  --tmpdir ~/podman-tmp2 \
  --progress=plain \
  --platform linux/arm64 \
  --build-arg CUDA_MAJOR=13 \
  --build-arg CUDA_MINOR=0 \
  --build-arg TARGETOS=ubuntu \
  --build-arg BASE_IMAGE_SUFFIX=ubuntu24.04 \
  --build-arg BUILD_BASE_IMAGE_SUFFIX=ubuntu24.04 \
  --build-arg FINAL_BASE_IMAGE_SUFFIX=ubuntu24.04 \
  --build-arg USE_SCCACHE=true \
  --build-arg MAX_JOBS=40 \
  --build-arg TORCH_CUDA_ARCH_LIST="10.0a" \
  --build-arg VLLM_COMMIT_SHA="${VLLM_COMMIT_SHA}" \
  --secret id=aws_access_key_id,src=$HOME/.local/share/containers/secrets/aws_access_key_id \
  --secret id=aws_secret_access_key,src=$HOME/.local/share/containers/secrets/aws_secret_access_key \
  -t "${IMAGE_TAG}" \
  -f docker/Dockerfile.cuda .

TMPDIR=~/podman-tmp2 podman push "${IMAGE_TAG}"
