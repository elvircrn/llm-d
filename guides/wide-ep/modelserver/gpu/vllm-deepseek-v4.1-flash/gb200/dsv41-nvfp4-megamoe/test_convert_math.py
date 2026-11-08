"""Validate convert_nvfp4 dequant against deep_gemm's own quantizers."""

import torch
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_nvfp4

import convert_nvfp4 as c

torch.manual_seed(0)
x = torch.randn(64, 512, device="cuda") * 0.05
packed, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
sf_ckpt = sf.to(torch.float8_e8m0fnu)
x_rec = c.dequant_mxfp4(packed, sf_ckpt)
err_mx = ((x_rec - x).norm() / x.norm()).item()
packed_nv, sf_nv = per_token_cast_to_nvfp4(x_rec, gran_k=16, use_packed_e4m3=True)
x_rt = c.dequant_nvfp4(packed_nv, sf_nv)
err_nv = ((x_rt - x_rec).norm() / x_rec.norm()).item()
print(f"mxfp4 quant rel err:   {err_mx:.4f}")
print(f"nvfp4 added rel err:   {err_nv:.4f}")
print(f"packed: {packed_nv.dtype} {tuple(packed_nv.shape)}"
      f"  sf: {sf_nv.dtype} {tuple(sf_nv.shape)}")
assert packed_nv.dtype == torch.int8 and packed_nv.shape == (64, 256)
assert sf_nv.dtype == torch.int32 and sf_nv.shape == (64, 8)
assert err_mx < 0.2, err_mx
assert err_nv < 0.2, err_nv
print("self-test ok")
