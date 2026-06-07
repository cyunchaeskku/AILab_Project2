"""Generate faces with the SR strategy: native 512 from G, then 2x super
resolution (512 -> 1024) with Real-ESRGAN x2plus.

Why this exists:
    The progressive Generator can synthesize a clean native 512 image, but the
    submission contract needs 1024. Instead of letting G grow the last octave
    (the 1024 fade stage that is still rough), this renders 512 and hands the
    512 -> 1024 step to a pretrained SR network (RealESRGAN_x2plus). This script
    just generates images so you can eyeball whether SR detail beats a plain
    bicubic upscale -- no FID here, that comes later.

Faithfulness:
    Base faces are produced exactly like the submission path would --
    z*psi -> G(z, resolution=512, alpha=1.0) -- so what you see is what an
    SR-strategy submission would output, minus the ONNX wrapper.

Run from the repo root (where src/ and verify_truncation.py live):
    pip install realesrgan basicsr
    python verify_sr.py --ckpt /path/to/ckpt.pt
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.utils as vutils

from verify_truncation import load_generator

# basicsr still imports torchvision.transforms.functional_tensor, which newer
# torchvision removed. Shim it to the current location before importing basicsr.
import torchvision.transforms.functional as _tv_functional

_functional_tensor_shim = types.ModuleType("torchvision.transforms.functional_tensor")
_functional_tensor_shim.rgb_to_grayscale = _tv_functional.rgb_to_grayscale
sys.modules.setdefault("torchvision.transforms.functional_tensor", _functional_tensor_shim)

from basicsr.archs.rrdbnet_arch import RRDBNet


# ----------------------------- configuration ----------------------------- #
DEFAULT_CKPT_PATH = Path(
    "/content/drive/MyDrive/OpenAILab/project2/pggan_1024_v2/ckpt_000475176.pt"
)
OUT_DIR = Path("runs/sr_512to1024")

SEED = 42
NUM_SAMPLES = 16
GRID_NROW = 8
PREVIEW_RESOLUTION = 256          # downscale only for the comparison grid

# Base 512 generation (the submission path, before SR).
PSI = 0.7                         # truncation strength (match verify_truncation)
GEN_RESOLUTION = 512
GEN_ALPHA = 1.0                   # 1.0 = clean native 512; <1 if ckpt is mid-fade
GEN_BATCH_SIZE = 8

# Real-ESRGAN x2plus: 512 -> 1024.
SR_SCALE = 2
SR_BATCH_SIZE = 4                 # 1024 outputs are memory-heavy; keep small
SR_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.2.1/RealESRGAN_x2plus.pth"
)
SR_WEIGHTS_PATH = Path("weights/RealESRGAN_x2plus.pth")
# ------------------------------------------------------------------------- #


def load_sr_model(device: torch.device) -> RRDBNet:
    """Load the pretrained Real-ESRGAN x2plus generator (RRDBNet, scale=2)."""
    if not SR_WEIGHTS_PATH.is_file():
        SR_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(SR_WEIGHTS_URL, str(SR_WEIGHTS_PATH))

    sr_model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=SR_SCALE,
    )
    state_dict = torch.load(SR_WEIGHTS_PATH, map_location="cpu")
    sr_model.load_state_dict(state_dict["params_ema"])
    sr_model.eval().to(device)
    return sr_model


@torch.no_grad()
def generate_base_faces(generator: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Render NUM_SAMPLES native-512 faces in [0, 1], with truncation baked in."""
    z = torch.randn(
        NUM_SAMPLES, generator.z_dim,
        generator=torch.Generator().manual_seed(SEED),
    ).to(device)                                            # (N, 512)
    z = z * PSI                                             # (N, 512) -> (N, 512)

    faces = []
    for start in range(0, NUM_SAMPLES, GEN_BATCH_SIZE):
        z_batch = z[start:start + GEN_BATCH_SIZE]           # (b, 512)
        out = generator(
            z_batch, resolution=GEN_RESOLUTION, alpha=GEN_ALPHA
        )                                                  # (b, 512) -> (b, 3, 512, 512)
        faces.append(out.cpu())
    faces = torch.cat(faces, dim=0)                         # (N, 3, 512, 512)
    return ((faces + 1.0) / 2.0).clamp(0.0, 1.0)           # [-1, 1] -> [0, 1]


@torch.no_grad()
def super_resolve(
    sr_model: RRDBNet, faces_01: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Upscale [0, 1] faces 512 -> 1024 batch by batch."""
    outputs = []
    for start in range(0, faces_01.size(0), SR_BATCH_SIZE):
        batch = faces_01[start:start + SR_BATCH_SIZE].to(device)  # (b, 3, 512, 512)
        sr = sr_model(batch)                                      # (b, 3, 512, 512) -> (b, 3, 1024, 1024)
        outputs.append(sr.clamp(0.0, 1.0).cpu())
    return torch.cat(outputs, dim=0)                             # (N, 3, 1024, 1024)


def preview_grid(images_01: torch.Tensor) -> torch.Tensor:
    """Downscale a batch of 1024 images and tile them into one grid."""
    small = F.interpolate(
        images_01,
        size=(PREVIEW_RESOLUTION, PREVIEW_RESOLUTION),
        mode="bilinear",
        align_corners=False,
    )                                                      # (N, 3, 1024, 1024) -> (N, 3, 256, 256)
    return vutils.make_grid(small, nrow=GRID_NROW, padding=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    generator = load_generator(args.ckpt).to(device)
    print(f"Loaded G_ema from {args.ckpt.name} (z_dim={generator.z_dim})")
    sr_model = load_sr_model(device)
    print("Loaded Real-ESRGAN x2plus (512 -> 1024)")

    base_512 = generate_base_faces(generator, device)      # (N, 3, 512, 512)
    sr_1024 = super_resolve(sr_model, base_512, device)    # (N, 3, 1024, 1024)
    bicubic_1024 = F.interpolate(
        base_512, size=(1024, 1024), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)                                      # (N, 3, 512, 512) -> (N, 3, 1024, 1024)

    # Full-res individual faces for pixel-peeping the SR detail.
    for i in range(NUM_SAMPLES):
        vutils.save_image(sr_1024[i], OUT_DIR / f"sr_{i:02d}.png")
        vutils.save_image(bicubic_1024[i], OUT_DIR / f"bicubic_{i:02d}.png")

    # Side-by-side grids: SR vs bicubic on the SAME latents = the SR effect.
    vutils.save_image(preview_grid(sr_1024), OUT_DIR / "grid_sr.png")
    vutils.save_image(preview_grid(bicubic_1024), OUT_DIR / "grid_bicubic.png")

    print(f"[sr]      {NUM_SAMPLES} faces  -> {OUT_DIR}/sr_*.png, grid_sr.png")
    print(f"[bicubic] {NUM_SAMPLES} faces  -> {OUT_DIR}/bicubic_*.png, grid_bicubic.png")
    print("Compare grid_sr.png vs grid_bicubic.png to judge the SR effect.")


if __name__ == "__main__":
    main()
