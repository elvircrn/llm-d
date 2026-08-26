# Common Humming MXFP4 weight + block-FP8 g128 activation MoE (vLLM PR 51332)
# build step, shared by base/decode.yaml and base/prefill.yaml (and kept in sync
# by hand with the inline copy in deployments/agg-tp8-ep4/serve_correct.yaml).
# Built from source with wheel caching on the shared-vast RWX PVC, keyed on the
# branch head SHA + BUILD_VARIANT, so only the first pod across BOTH the
# prefill and decode LWSes pays the compile; every other pod/restart reinstalls
# the cached wheel in seconds. Sourced (not exec'd) from each role's args so it
# runs in the caller's shell and leaves it able to `exec vllm serve` afterward.
#
# DeepEP: also built from source (humming-on-vllm-base fork), same
# wheel-cache scheme as the vLLM build above, kept in sync with
# agg-tp8-ep4/serve_correct.yaml.

BUILD_REPO=https://github.com/elvircrn/vllm.git
BUILD_BRANCH=humming-mxfp4-w4a8-block-fp8
# Bump when the build RECIPE changes (flags, prunes) -- part of the
# cache key, so a bump forces a rebuild for the same branch SHA.
BUILD_VARIANT=release-nofastmath-zeropad3
command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1; }
BUILD_SHA=$(git ls-remote "$BUILD_REPO" "$BUILD_BRANCH" | cut -f1)
if [ -z "$BUILD_SHA" ]; then
  echo "FATAL: could not resolve ${BUILD_BRANCH} via ls-remote."
  exit 1
fi
echo "Building ${BUILD_BRANCH}: ${BUILD_SHA}"
KEY="${BUILD_SHA:0:12}-${BUILD_VARIANT}"
CACHE=/shared/vllm-build/pr_build/${KEY}
WHEEL_DIR="${CACHE}/wheel"

command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv

SITE=$(cd / && python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
REUSE_DIRS="vllm_flash_attn third_party/deep_gemm third_party/flashmla"
REUSE_FILES="_flashmla_C.abi3.so _flashmla_extension_C.abi3.so"
rm -rf /tmp/base_reuse
for d in $REUSE_DIRS; do
  if [ -d "${SITE}/vllm/${d}" ]; then
    mkdir -p "/tmp/base_reuse/$(dirname "$d")"
    cp -r "${SITE}/vllm/${d}" "/tmp/base_reuse/${d}"
  fi
done
mkdir -p /tmp/base_reuse
for f in $REUSE_FILES; do
  [ -f "${SITE}/vllm/${f}" ] && cp "${SITE}/vllm/${f}" "/tmp/base_reuse/${f}"
done

NEED_BUILD=1
if ls "${WHEEL_DIR}"/vllm-*.whl >/dev/null 2>&1; then
  WHEEL=$(ls -t "${WHEEL_DIR}"/vllm-*.whl | head -1)
  if python3 -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$WHEEL') and zipfile.ZipFile('$WHEEL').testzip() is None else 1)" 2>/dev/null; then
    echo "Installing cached vLLM wheel for ${BUILD_BRANCH} (${BUILD_SHA}): $(basename "$WHEEL")"
    if uv pip install --system --force-reinstall --no-deps "$WHEEL"; then
      NEED_BUILD=0
    else
      echo "WARN: cached vLLM wheel install failed; removing and rebuilding: $WHEEL"
      rm -f "$WHEEL"
    fi
  else
    echo "WARN: cached vLLM wheel is corrupt; removing and rebuilding: $WHEEL"
    rm -f "$WHEEL"
  fi
fi
if [ "$NEED_BUILD" = 1 ]; then
  apt-get update -qq && apt-get install -y -qq ccache mold > /dev/null 2>&1
  rm -rf /tmp/vllm-pr
  mkdir -p /tmp/vllm-pr && cd /tmp/vllm-pr
  git init -q
  git fetch --depth=1 -q "$BUILD_REPO" "$BUILD_SHA" || { echo "FATAL: fetch ${BUILD_BRANCH} failed"; exit 1; }
  git checkout -q FETCH_HEAD || { echo "FATAL: checkout ${BUILD_BRANCH} failed"; exit 1; }
  echo "Building ${BUILD_BRANCH} (${BUILD_SHA}); SLOW (~30 min)..."
  uv pip install --system -q setuptools wheel setuptools_scm setuptools_rust ninja cmake || { echo "FATAL: build-deps install failed"; exit 1; }
  BASE_VLLM=$(cd / && python3 -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent)")
  cp -r "${BASE_VLLM}/vllm-rs" vllm/ 2>/dev/null || true
  cp "${BASE_VLLM}"/_rust_*.so vllm/ 2>/dev/null || true
  python3 - <<'PYEOF' || { echo "FATAL: setup.py prune failed (recipe drift?)"; exit 1; }
p = 'setup.py'
s = open(p).read()
s = s.replace(
    '    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa2_C"))\n',
    '')
s = s.replace(
    '        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))\n',
    '        pass\n')
s = s.replace(
    '        ext_modules.append(CMakeExtension(name="vllm._deep_gemm_C", optional=True))\n',
    '')
s = s.replace(
    '        ext_modules.append(CMakeExtension(name="vllm._flashmla_C", optional=True))\n'
    '        ext_modules.append(\n'
    '            CMakeExtension(name="vllm._flashmla_extension_C", optional=True)\n'
    '        )\n',
    '        pass\n')
open(p, 'w').write(s)
assert '_vllm_fa2_C")' not in s and '_vllm_fa3_C")' not in s, 'FA prune failed'
assert '_deep_gemm_C"' not in s, 'DeepGEMM prune failed'
assert 'name="vllm._flashmla_C"' not in s, 'FlashMLA prune failed'
print('Pruned FA2/FA3 + DeepGEMM + FlashMLA from setup.py ext_modules')
PYEOF
  NVRTC_LIB=$(ls /usr/local/cuda*/lib64/libnvrtc.so 2>/dev/null | head -1)
  [ -n "$NVRTC_LIB" ] || NVRTC_LIB=$(find /usr/local/cuda* \
    /usr/local/lib/python3.12/dist-packages/nvidia \
    -name 'libnvrtc.so*' 2>/dev/null | sort | head -1)
  if [ -z "$NVRTC_LIB" ]; then
    echo "FATAL: could not locate libnvrtc for the CUDA build."
    exit 1
  fi
  echo "Using libnvrtc: $NVRTC_LIB"
  export CMAKE_ARGS="${CMAKE_ARGS:-} -DCUDA_nvrtc_LIBRARY=${NVRTC_LIB}"
  if command -v mold >/dev/null 2>&1; then
    export CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_LINKER_TYPE=MOLD"
  fi
  export CMAKE_BUILD_TYPE=Release
  export NVCC_THREADS=${NVCC_THREADS:-8}
  # No --use_fast_math: it approximates div/sqrt/rsqrt and flushes denormals,
  # unsafe to stack under the already-lossy MXFP4/block-FP8 quant path.
  export CUDAFLAGS="${CUDAFLAGS:-} -Xptxas -O3 --extra-device-vectorization"
  mkdir -p "$WHEEL_DIR" "${CACHE}/logs"
  LOCAL_WHEEL=/tmp/vllm-wheel-out
  rm -rf "$LOCAL_WHEEL" && mkdir -p "$LOCAL_WHEEL"
  BUILD_LOG="${CACHE}/logs/build-${HOSTNAME:-$(hostname)}.log"
  set -o pipefail
  # Default to actual core count, not a hardcoded guess -- decode's cpu
  # request (32) is half of prefill's (64), so a shared fixed default
  # oversubscribes one of them against MAX_JOBS x NVCC_THREADS.
  if TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=${MAX_JOBS:-$(nproc)} \
       uv build --wheel --no-build-isolation -o "$LOCAL_WHEEL" . 2>&1 | tee "$BUILD_LOG"; then
    for w in "$LOCAL_WHEEL"/vllm-*.whl; do
      bn=$(basename "$w")
      cp "$w" "${WHEEL_DIR}/.${bn}.$$.tmp"
      mv -f "${WHEEL_DIR}/.${bn}.$$.tmp" "${WHEEL_DIR}/${bn}"
    done
    WHEEL=$(ls -t "${WHEEL_DIR}"/vllm-*.whl | head -1)
    uv pip install --system --force-reinstall --no-deps "$WHEEL" || { echo "FATAL: built wheel install failed"; exit 1; }
    echo "Built + published vLLM ${BUILD_BRANCH} (${BUILD_SHA}) -> ${WHEEL_DIR}"
  else
    echo "FATAL: vLLM ${BUILD_BRANCH} build failed; log at ${BUILD_LOG}; not publishing."
    exit 1
  fi
  cd
  rm -rf /tmp/vllm-pr "$LOCAL_WHEEL"
fi

for d in $REUSE_DIRS; do
  if [ -d "/tmp/base_reuse/${d}" ]; then
    rm -rf "${SITE}/vllm/${d}"
    mkdir -p "$(dirname "${SITE}/vllm/${d}")"
    cp -r "/tmp/base_reuse/${d}" "${SITE}/vllm/${d}"
    echo "Restored base-image ${d} into ${SITE}/vllm/${d}"
  else
    echo "WARN: no stashed base ${d} to restore (/tmp/base_reuse/${d} missing)"
  fi
done
for f in $REUSE_FILES; do
  if [ -f "/tmp/base_reuse/${f}" ]; then
    cp "/tmp/base_reuse/${f}" "${SITE}/vllm/${f}"
    echo "Restored base-image ${f} into ${SITE}/vllm/${f}"
  else
    echo "WARN: no stashed base ${f} to restore (/tmp/base_reuse/${f} missing)"
  fi
done
#
#DEEPEP_REPO=https://github.com/deepseek-ai/DeepEP.git
#DEEPEP_BRANCH=main
## Bump when the DeepEP build RECIPE changes (part of the cache key).
#DEEPEP_VARIANT=v2
#DEEPEP_SHA=$(git ls-remote "$DEEPEP_REPO" "$DEEPEP_BRANCH" | cut -f1)
#if [ -z "$DEEPEP_SHA" ]; then
#  echo "FATAL: could not resolve DeepEP ${DEEPEP_BRANCH} via ls-remote."
#  exit 1
#fi
#echo "DeepEP ${DEEPEP_BRANCH}: ${DEEPEP_SHA}"
#DEEPEP_KEY="${DEEPEP_SHA:0:12}-${DEEPEP_VARIANT}"
#DEEPEP_CACHE=/shared/vllm-build/deepep_build/${DEEPEP_KEY}
#DEEPEP_WHEEL_DIR="${DEEPEP_CACHE}/wheel"
#
#DEEPEP_NEED_BUILD=1
#if ls "${DEEPEP_WHEEL_DIR}"/deep_ep-*.whl >/dev/null 2>&1; then
#  DEEPEP_WHEEL=$(ls -t "${DEEPEP_WHEEL_DIR}"/deep_ep-*.whl | head -1)
#  if python3 -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('$DEEPEP_WHEEL') and zipfile.ZipFile('$DEEPEP_WHEEL').testzip() is None else 1)" 2>/dev/null; then
#    echo "Installing cached DeepEP wheel (${DEEPEP_SHA}): $(basename "$DEEPEP_WHEEL")"
#    if uv pip install --system --force-reinstall --no-deps "$DEEPEP_WHEEL"; then
#      DEEPEP_NEED_BUILD=0
#    else
#      echo "WARN: cached DeepEP wheel install failed; removing and rebuilding: $DEEPEP_WHEEL"
#      rm -f "$DEEPEP_WHEEL"
#    fi
#  else
#    echo "WARN: cached DeepEP wheel is corrupt; removing and rebuilding: $DEEPEP_WHEEL"
#    rm -f "$DEEPEP_WHEEL"
#  fi
#fi
#if [ "$DEEPEP_NEED_BUILD" = 1 ]; then
#  command -v git >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq git > /dev/null 2>&1; }
#  rm -rf /tmp/deepep-src
#  mkdir -p /tmp/deepep-src && cd /tmp/deepep-src
#  git init -q
#  git fetch --depth=1 -q "$DEEPEP_REPO" "$DEEPEP_SHA" || { echo "FATAL: DeepEP fetch failed"; exit 1; }
#  git checkout -q FETCH_HEAD || { echo "FATAL: DeepEP checkout failed"; exit 1; }
#  echo "Building DeepEP ${DEEPEP_BRANCH} (${DEEPEP_SHA})..."
#  uv pip install --system -q setuptools wheel setuptools_scm ninja cmake || { echo "FATAL: DeepEP build-deps install failed"; exit 1; }
#  mkdir -p "$DEEPEP_WHEEL_DIR" "${DEEPEP_CACHE}/logs"
#  DEEPEP_LOCAL=/tmp/deepep-wheel-out
#  rm -rf "$DEEPEP_LOCAL" && mkdir -p "$DEEPEP_LOCAL"
#  DEEPEP_LOG="${DEEPEP_CACHE}/logs/build-${HOSTNAME:-$(hostname)}.log"
#  # DeepEP's JIT wrapper (csrc/jit/compiler.hpp) #includes <nvrtc.h>
#  # and links libnvrtc, but the base image ships these only under the
#  # pip nvidia-cuda-nvrtc dir, not /usr/local/cuda/include. Add the
#  # header dir to CPATH (nvcc's host preprocessor honors it) and the
#  # lib dir to LIBRARY_PATH so the extension compiles + links.
#  NVRTC_HDR=$(find /usr/local/cuda* \
#    /usr/local/lib/python3.12/dist-packages/nvidia \
#    -name 'nvrtc.h' 2>/dev/null | head -1)
#  if [ -n "$NVRTC_HDR" ]; then
#    export CPATH="$(dirname "$NVRTC_HDR"):${CPATH}"
#    echo "Using nvrtc.h: $NVRTC_HDR"
#  else
#    echo "WARN: nvrtc.h not found; DeepEP build may fail."
#  fi
#  NVRTC_SO=$(find /usr/local/cuda* \
#    /usr/local/lib/python3.12/dist-packages/nvidia \
#    -name 'libnvrtc.so*' 2>/dev/null | sort | head -1)
#  if [ -n "$NVRTC_SO" ]; then
#    export LIBRARY_PATH="$(dirname "$NVRTC_SO"):${LIBRARY_PATH}"
#    export LD_LIBRARY_PATH="$(dirname "$NVRTC_SO"):${LD_LIBRARY_PATH}"
#    echo "Using libnvrtc: $NVRTC_SO"
#  fi
#  # DeepEP links NCCL by exact soname (-l:libnccl.so.2), but the
#  # base image may ship only a versioned libnccl.so.2.x.y or place
#  # it off the linker path, so ld can't resolve the bare soname.
#  # Find it and expose a canonical libnccl.so.2 symlink dir on
#  # LIBRARY_PATH (and the real dir on LD_LIBRARY_PATH for runtime).
#  DEEPEP_LINK_DIR=/tmp/deepep-link
#  rm -rf "$DEEPEP_LINK_DIR" && mkdir -p "$DEEPEP_LINK_DIR"
#  NCCL_SO=$(find /usr/local/lib/python3.12/dist-packages/nvidia \
#    /usr/local/cuda* /usr/lib/x86_64-linux-gnu /usr/lib \
#    -name 'libnccl.so.2*' 2>/dev/null | sort | head -1)
#  if [ -n "$NCCL_SO" ]; then
#    ln -sf "$NCCL_SO" "$DEEPEP_LINK_DIR/libnccl.so.2"
#    export LIBRARY_PATH="${DEEPEP_LINK_DIR}:${LIBRARY_PATH}"
#    export LD_LIBRARY_PATH="$(dirname "$NCCL_SO"):${LD_LIBRARY_PATH}"
#    echo "Using libnccl: $NCCL_SO"
#  else
#    echo "WARN: libnccl.so.2 not found; DeepEP link may fail."
#  fi
#  set -o pipefail
#  if TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=${MAX_JOBS:-$(nproc)} NVCC_THREADS=${NVCC_THREADS:-8} \
#       uv build --wheel --no-build-isolation -o "$DEEPEP_LOCAL" . 2>&1 | tee "$DEEPEP_LOG"; then
#    for w in "$DEEPEP_LOCAL"/deep_ep-*.whl; do
#      bn=$(basename "$w")
#      cp "$w" "${DEEPEP_WHEEL_DIR}/.${bn}.$$.tmp"
#      mv -f "${DEEPEP_WHEEL_DIR}/.${bn}.$$.tmp" "${DEEPEP_WHEEL_DIR}/${bn}"
#    done
#    DEEPEP_WHEEL=$(ls -t "${DEEPEP_WHEEL_DIR}"/deep_ep-*.whl | head -1)
#    uv pip install --system --force-reinstall --no-deps "$DEEPEP_WHEEL" || { echo "FATAL: built DeepEP wheel install failed"; exit 1; }
#    echo "Built + published DeepEP ${DEEPEP_BRANCH} (${DEEPEP_SHA}) -> ${DEEPEP_WHEEL_DIR}"
#  else
#    echo "FATAL: DeepEP build failed; log at ${DEEPEP_LOG}; not publishing."
#    exit 1
#  fi
#  cd
#  rm -rf /tmp/deepep-src "$DEEPEP_LOCAL"
#fi

# Bump the engine<->frontend startup handshake timeout (hardcoded
# 5 min in vllm/v1/engine/core.py -- no env/flag). Heavy multinode
# boot + slow W4A8 weight load can blow past 5 min before the
# front-end answers. Patch the installed file each boot (idempotent,
# works with the cached wheel too -- no rebuild needed).
python3 - <<'PYEOF' || { echo "FATAL: HANDSHAKE_TIMEOUT_MINS patch failed"; exit 1; }
import re, pathlib, vllm
p = pathlib.Path(vllm.__file__).parent / "v1/engine/core.py"
s = p.read_text()
s2 = re.sub(r'HANDSHAKE_TIMEOUT_MINS = \d+', 'HANDSHAKE_TIMEOUT_MINS = 60', s)
assert s2 != s, "HANDSHAKE_TIMEOUT_MINS patch did not match"
p.write_text(s2)
print("Patched HANDSHAKE_TIMEOUT_MINS -> 60")
PYEOF


uv pip install nixl