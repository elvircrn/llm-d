# Qwen3-4B prefill/decode, built vLLM

The same 1 GPU prefill / 1 GPU decode pair as `guides/qwen3-4b-pd`, served as
`qwen3-4b-pr` with each container carrying its own `bash -c` serve script so a
campaign wheel build can install a source-built vLLM into the pod at startup.
Requires a source-build capable campaign recipe; without a build the script
serves the image's own vLLM.
