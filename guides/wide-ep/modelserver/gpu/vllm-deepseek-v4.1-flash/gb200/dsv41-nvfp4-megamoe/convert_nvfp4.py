"""Convert DSV4.1 Flash routed-expert MXFP4 weights to NVFP4.

The MXFP4 checkpoint stores packed E2M1 expert weights with one UE8M0
scale per row per 32-element K group. The nvfp4_mega_moe kernel wants
packed E2M1 with one E4M3 scale per 16-element K group and unit global
scales, so this rewrites every ffn.experts.* tensor and adds the
per-expert weight_scale_2/input_scale globals the NVFP4 loader expects.
Every other tensor is copied byte for byte.
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from deep_gemm.utils import per_token_cast_to_nvfp4

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
EXPERT_RE = re.compile(r"^(.*\.ffn\.experts\.\d+\.(w[123]))\.(weight|scale)$")


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    vals = E2M1.to(codes.device)[(codes & 0x7).to(torch.long)]
    return torch.where((codes & 0x8) != 0, -vals, vals)


def dequant_mxfp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    n, k_half = weight.shape
    codes = weight.view(torch.uint8)
    x = torch.empty((n, k_half * 2), dtype=torch.float32, device=weight.device)
    x[:, 0::2] = _decode_e2m1(codes & 0x0F)
    x[:, 1::2] = _decode_e2m1(codes >> 4)
    block = torch.exp2(scale.view(torch.uint8).float() - 127.0)
    return x * block.repeat_interleave(32, dim=1)


def dequant_nvfp4(weight: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    n, k_half = weight.shape
    codes = weight.view(torch.uint8)
    x = torch.empty((n, k_half * 2), dtype=torch.float32, device=weight.device)
    x[:, 0::2] = _decode_e2m1(codes & 0x0F)
    x[:, 1::2] = _decode_e2m1(codes >> 4)
    group = sf.view(torch.uint8).view(torch.float8_e4m3fn).float()
    return x * group.repeat_interleave(16, dim=1)


def convert_shard(path: Path, dst: Path, device: str) -> dict:
    out: dict[str, torch.Tensor] = {}
    pending: dict[str, dict[str, torch.Tensor]] = {}
    errors: list[float] = []
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            m = EXPERT_RE.match(key)
            if m is None:
                out[key] = tensor
                continue
            prefix, shard = m.group(1), m.group(2)
            slot = pending.setdefault(prefix, {})
            slot[m.group(3)] = tensor
            if len(slot) < 2:
                continue
            x = dequant_mxfp4(slot["weight"].to(device), slot["scale"].to(device))
            packed, sf = per_token_cast_to_nvfp4(x, gran_k=16, use_packed_e4m3=True)
            roundtrip = dequant_nvfp4(packed, sf)
            errors.append(((roundtrip - x).norm() / x.norm()).item())
            out[f"{prefix}.weight"] = packed.contiguous().cpu()
            out[f"{prefix}.scale"] = sf.contiguous().cpu()
            out[f"{prefix}.weight_scale_2"] = torch.tensor(1.0)
            if shard == "w2":
                out[f"{prefix}.input_scale"] = torch.tensor(1.0)
            del pending[prefix]
    save_file(out, str(dst / path.name), metadata={"format": "pt"})
    return {
        "tensors": len(errors),
        "rel_err_mean": sum(errors) / len(errors) if errors else 0.0,
        "rel_err_max": max(errors) if errors else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for entry in sorted(src.iterdir()):
        if entry.suffix == ".safetensors" or entry.name == "model.safetensors.index.json":
            continue
        if entry.is_dir():
            shutil.copytree(entry, dst / entry.name, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dst / entry.name)

    shards = sorted(p for p in src.iterdir() if p.suffix == ".safetensors")
    weight_map: dict[str, str] = {}
    report: dict[str, dict] = {}
    for i, shard in enumerate(shards):
        stats = convert_shard(shard, dst, args.device)
        report[shard.name] = stats
        print(f"[{i + 1}/{len(shards)}] {shard.name}: {stats}", flush=True)
    for shard in shards:
        with safe_open(dst / shard.name, framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard.name
    total_size = sum((dst / s.name).stat().st_size for s in shards)
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    errs = [s["rel_err_mean"] for s in report.values()]
    summary = {
        "shards": len(shards),
        "expert_tensors": sum(s["tensors"] for s in report.values()),
        "rel_err_mean": sum(errs) / len(errs),
        "rel_err_max": max(s["rel_err_max"] for s in report.values()),
        "per_shard": report,
    }
    (dst / "conversion_report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_shard"}, indent=1))


if __name__ == "__main__":
    main()
