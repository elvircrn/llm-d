#!/bin/bash
set -e

echo "Building CUDA 13 ARM64 Docker image for GB200..."

# Create temporary directory for secrets
SECRETS_DIR="/tmp/docker-secrets-$$"
mkdir -p "$SECRETS_DIR"

# Create empty secret files
echo "" > "$SECRETS_DIR/aws_access_key_id"
echo "" > "$SECRETS_DIR/aws_secret_access_key"
echo "" > "$SECRETS_DIR/subman_org"
echo "" > "$SECRETS_DIR/subman_activation_key"

echo "Building with platform: linux/arm64"
echo "Using Dockerfile: docker/Dockerfile.cuda"

# Build the Docker image
docker build --platform linux/arm64 \
  --build-arg USE_SCCACHE=false \
  --secret id=aws_access_key_id,src="$SECRETS_DIR/aws_access_key_id" \
  --secret id=aws_secret_access_key,src="$SECRETS_DIR/aws_secret_access_key" \
  --secret id=subman_org,src="$SECRETS_DIR/subman_org" \
  --secret id=subman_activation_key,src="$SECRETS_DIR/subman_activation_key" \
  -f docker/Dockerfile.cuda \
  -t llm-d:cuda13-gb200-ubuntu24.04 \
  .

echo "Build completed successfully!"
echo "Image tagged as: llm-d:cuda13-gb200-ubuntu24.04"