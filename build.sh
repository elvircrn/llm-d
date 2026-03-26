#!/bin/bash
set -euo pipefail

DEFAULT_REPO="https://github.com/vllm-project/vllm"

usage() {
  echo "Usage: ./build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  --latest            Use latest vllm main commit"
  echo "  --commit <sha>      Use a specific vllm commit"
  echo "  --repo <url>        Use a different vllm repo (default: upstream)"
  echo "  --precompiled       Use precompiled vllm binaries (faster build, no C++/CUDA editing)"
  echo "  --no-cache          Force rebuild all layers"
  echo ""
  echo "Examples:"
  echo "  ./build.sh                                          # use commit from Dockerfile"
  echo "  ./build.sh --latest                                 # latest vllm main"
  echo "  ./build.sh --commit abc123                          # specific commit"
  echo "  ./build.sh --repo https://github.com/user/vllm --commit abc123"
  echo "  ./build.sh --latest --precompiled                   # latest + precompiled (fastest)"
  exit 1
}

VLLM_USE_PRECOMPILED=0
NO_CACHE=""
COMMIT_MODE="dockerfile"
SPECIFIC_COMMIT=""
VLLM_REPO=""

while [ $# -gt 0 ]; do
  case "$1" in
    --latest) COMMIT_MODE="latest"; shift ;;
    --commit) COMMIT_MODE="specific"; SPECIFIC_COMMIT="${2:?--commit requires a sha}"; shift 2 ;;
    --repo) VLLM_REPO="${2:?--repo requires a url}"; shift 2 ;;
    --precompiled) VLLM_USE_PRECOMPILED=1; shift ;;
    --no-cache) NO_CACHE="--no-cache"; shift ;;
    --help|-h) usage ;;
    *) echo "Unknown option: $1"; usage ;;
  esac
done

# Determine repo: --repo flag > --latest defaults to upstream > Dockerfile value
if [ -n "$VLLM_REPO" ]; then
  sed -i "s|VLLM_REPO=\".*\"|VLLM_REPO=\"${VLLM_REPO}\"|" docker/Dockerfile.cuda
  echo "Using repo: ${VLLM_REPO}"
elif [ "$COMMIT_MODE" = "latest" ]; then
  VLLM_REPO="$DEFAULT_REPO"
  sed -i "s|VLLM_REPO=\".*\"|VLLM_REPO=\"${VLLM_REPO}\"|" docker/Dockerfile.cuda
  echo "Using upstream repo for --latest: ${VLLM_REPO}"
else
  VLLM_REPO=$(grep '^ARG VLLM_REPO=' docker/Dockerfile.cuda | cut -d'"' -f2)
fi

case "$COMMIT_MODE" in
  latest)
    VLLM_COMMIT_SHA=$(git ls-remote "${VLLM_REPO}" refs/heads/main | cut -f1)
    echo "Fetched latest main: ${VLLM_COMMIT_SHA}"
    sed -i "s|VLLM_COMMIT_SHA=\".*\"|VLLM_COMMIT_SHA=\"${VLLM_COMMIT_SHA}\"|" docker/Dockerfile.cuda
    ;;
  specific)
    VLLM_COMMIT_SHA="$SPECIFIC_COMMIT"
    sed -i "s|VLLM_COMMIT_SHA=\".*\"|VLLM_COMMIT_SHA=\"${VLLM_COMMIT_SHA}\"|" docker/Dockerfile.cuda
    ;;
  dockerfile)
    VLLM_COMMIT_SHA=$(grep '^ARG VLLM_COMMIT_SHA=' docker/Dockerfile.cuda | cut -d'"' -f2)
    ;;
esac

if [ -z "$VLLM_COMMIT_SHA" ]; then
  echo "ERROR: Could not extract VLLM_COMMIT_SHA"
  exit 1
fi

IMAGE_TAG="quay.io/rh-ee-ecrncevi/llm-dev-cuda13:v0.5.0-arm64-upstream-${VLLM_COMMIT_SHA}"

echo "Building with VLLM_COMMIT_SHA=${VLLM_COMMIT_SHA}"
echo "VLLM_USE_PRECOMPILED=${VLLM_USE_PRECOMPILED}"
echo "Image tag: ${IMAGE_TAG}"

TMPDIR=~/podman-tmp2 VLLM_PREBUILT=0 podman build --security-opt label=disable \
  --tmpdir ~/podman-tmp2 \
  ${NO_CACHE} \
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
  --build-arg VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED}" \
  --secret id=aws_access_key_id,src=$HOME/.local/share/containers/secrets/aws_access_key_id \
  --secret id=aws_secret_access_key,src=$HOME/.local/share/containers/secrets/aws_secret_access_key \
  -t "${IMAGE_TAG}" \
  -f docker/Dockerfile.cuda .

TMPDIR=~/podman-tmp2 podman push "${IMAGE_TAG}"
