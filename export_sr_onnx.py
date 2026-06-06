"""Export a 512 Generator + Real-ESRGAN x2plus SR to ONNX.

Submission contract:
    input  z      shape (B, 512), dtype float32
    output image  shape (B, 3, 1024, 1024), dtype float32, range [-1, 1]

Pipeline baked into the ONNX graph:
    z  →  z * psi  →  G(z, res=512, alpha=1.0)  →  [-1,1]→[0,1]
      →  RRDBNet x2  →  [0,1]→[-1,1]  →  (B, 3, 1024, 1024)

Usage:
    pip install realesrgan basicsr
    python export_sr_onnx.py \\
        --ckpt /path/to/ckpt.pt \\
        --out submission_sr.onnx \\
        --sr-weights weights/RealESRGAN_x2plus.pth \\
        --psi 0.7
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

from src.model import Generator, GeneratorConfig

# basicsr imports a removed torchvision submodule; shim before importing basicsr.
import torchvision.transforms.functional as _tv_functional
_functional_tensor_shim = types.ModuleType("torchvision.transforms.functional_tensor")
_functional_tensor_shim.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
sys.modules.setdefault("torchvision.transforms.functional_tensor", _functional_tensor_shim)

from basicsr.archs.rrdbnet_arch import RRDBNet


SR_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)


class SRSubmissionWrapper(nn.Module):
    """G(z*psi) at 512 → Real-ESRGAN x2 → 1024, output in [-1, 1]."""

    def __init__(self, G: nn.Module, sr_model: nn.Module, psi: float = 0.7) -> None:
        super().__init__()
        self.G = G
        self.sr = sr_model
        self.psi = psi

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.G(z * self.psi)              # (B, 512) → (B, 3, 512, 512), [-1, 1]
        x = (x + 1.0) / 2.0                   # [-1, 1] → [0, 1]
        x = self.sr(x)                         # (B, 3, 512, 512) → (B, 3, 1024, 1024)
        return (x * 2.0 - 1.0).clamp(-1, 1)   # [0, 1] → [-1, 1]


def load_generator(ckpt_path: Path) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "meta" in ckpt and isinstance(ckpt["meta"], dict) and "generator_config" in ckpt["meta"]:
        g_cfg = GeneratorConfig.from_dict(ckpt["meta"]["generator_config"])
        G = Generator(g_cfg)
    else:
        raise RuntimeError("Checkpoint has no meta.generator_config — use export_onnx.py for baseline ckpts.")
    state = ckpt.get("G_ema_state") or ckpt.get("G_state")
    if state is None:
        raise RuntimeError("Checkpoint has neither G_ema_state nor G_state.")
    G.load_state_dict(state)
    return G


def load_sr_model(sr_weights_path: Path) -> nn.Module:
    if not sr_weights_path.is_file():
        sr_weights_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading RealESRGAN_x2plus weights → {sr_weights_path}")
        torch.hub.download_url_to_file(SR_WEIGHTS_URL, str(sr_weights_path))

    sr_model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=2,
    )
    state_dict = torch.load(sr_weights_path, map_location="cpu")
    sr_model.load_state_dict(state_dict["params_ema"])
    return sr_model


def export_sr_onnx(
    ckpt_path: Path,
    out_path: Path,
    sr_weights_path: Path,
    *,
    psi: float = 0.7,
    opset: int = 17,
    batch_size: int = 1,
) -> None:
    G = load_generator(ckpt_path).eval()
    sr_model = load_sr_model(sr_weights_path).eval()

    wrapper = SRSubmissionWrapper(G, sr_model, psi=psi).eval()

    dummy_z = torch.randn(batch_size, 512)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        dummy_z,
        str(out_path),
        input_names=["z"],
        output_names=["image"],
        opset_version=opset,
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        dynamo=False,
    )

    with torch.no_grad():
        ref_out = wrapper(dummy_z)
    print(f"Saved ONNX → {out_path}")
    print(f"  input  z      (B, 512)")
    print(f"  output image  {tuple(ref_out.shape)}, range [{ref_out.min():.3f}, {ref_out.max():.3f}]")
    print(f"  psi={psi}, SR=RealESRGAN_x2plus")
    size_mb = out_path.stat().st_size / 1024 ** 2
    print(f"  file size: {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_sr.onnx"))
    parser.add_argument("--sr-weights", type=Path, default=Path("weights/RealESRGAN_x2plus.pth"))
    parser.add_argument("--psi", type=float, default=0.7)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    export_sr_onnx(
        args.ckpt,
        args.out,
        args.sr_weights,
        psi=args.psi,
        opset=args.opset,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
