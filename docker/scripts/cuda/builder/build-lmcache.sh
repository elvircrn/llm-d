#!/bin/bash
set -Eeu

# builds and installs LMCache and Infinistore from source
#
# Required environment variables:
# - USE_SCCACHE: whether to use sccache (true/false)
# - VIRTUAL_ENV: path to Python virtual environment
# - INFINISTORE_REPO: git repo to build Infinistore from
# - INFINISTORE_VERSION: git ref to build Infinistore from
# - LMCACHE_REPO: git repo to build LMCache from
# - LMCACHE_VERSION: git ref to build LMCache from
# Optional environment variables:
# - TARGETPLATFORM: platform target (linux/arm64 or linux/amd64)
# - TARGETOS: OS type (ubuntu or rhel)

cd /tmp

# Disable sccache entirely for LMCache - torch cpp_extension is incompatible with sccache+nvcc
if [ -x /usr/local/bin/sccache ]; then
    sccache --stop-server 2>/dev/null || true
    mv /usr/local/bin/sccache /usr/local/bin/sccache.bak
fi
. "${VIRTUAL_ENV}/bin/activate"

# PyTorch cpp_extension doesn't recognize "10.0f" syntax, normalize to standard format
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//10.0f/10.0}"



git clone "${INFINISTORE_REPO}" infinistore && cd infinistore
git checkout -q "${INFINISTORE_VERSION}"
uv build --wheel --no-build-isolation --out-dir /wheels
cd ..
rm -rf infinistore

git clone "${LMCACHE_REPO}" lmcache && cd lmcache
git checkout -q "${LMCACHE_VERSION}"
uv build --wheel --no-build-isolation --out-dir /wheels  && \
cd ..
rm -rf lmcache

# Restore sccache for subsequent build steps
if [ -f /usr/local/bin/sccache.bak ]; then
    mv /usr/local/bin/sccache.bak /usr/local/bin/sccache
fi
