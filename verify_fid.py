"""Sweep the truncation strength psi and measure FID for each, to pick the
submission psi by data instead of by eye.

Why this exists:
    Truncation (z *= psi, psi < 1) raises per-image fidelity but lowers
    diversity. FID penalizes BOTH low fidelity and lost diversity, so the
    psi that looks nicest is not necessarily the psi with the lowest FID.
    This script generates fakes at each psi and runs pytorch-fid against the
    same real set, so you can read off the FID-optimal psi.

Faithfulness to submission:
    Fakes are generated exactly as submission.onnx produces them --
    z*psi -> G(z, resolution=RESOLUTION, alpha=ALPHA) -- so the measured FID
    reflects the artifact you actually submit. Keep RESOLUTION/ALPHA in sync
    with verify_truncation.py's EXPORT_* values.

Reuses the tested validation pipeline from train.py (same seed, same FID call).

Run from the repo root:
    pip install pytorch-fid scipy
    python verify_fid.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from train import (
    extract_validation_subset,
    run_pytorch_fid,
    write_fake_validation_images,
)
from verify_truncation import load_generator


# ----------------------------- configuration ----------------------------- #
CKPT_PATH = Path(
    "/content/drive/MyDrive/OpenAILab/project2/pggan_1024_v2/ckpt_001150352.pt"
)
VALID_ZIP = "data/valid_10k_1024.zip"     # real 1024 images (from config)
OUT_DIR = Path("runs/fid_sweep")

# Must match the exported submission graph (verify_truncation.EXPORT_*).
RESOLUTION = 1024
ALPHA = 0.5

PSIS = [1.0, 0.85, 0.7, 0.6, 0.5]         # 1.0 = no truncation baseline

NUM_FAKE = 2000                            # >= 2000 keeps FID stable
NUM_REAL = 2000
BATCH_SIZE = 8
SEED = 2026                                # validation seed (from config)
# ------------------------------------------------------------------------- #


TARGET_RESOLUTION = 1024


class TruncatedGenerator(nn.Module):
    """Apply z *= psi, forward, then upsample to 1024 to match submission ONNX.

    Upsampling only applies when native output resolution < TARGET_RESOLUTION,
    so 1024 checkpoints pass through unchanged.
    """

    def __init__(self, generator: nn.Module, psi: float) -> None:
        super().__init__()
        self.generator = generator
        self.psi = float(psi)

    def forward(
        self,
        z: torch.Tensor,
        resolution: int | None = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        z = z * self.psi                                    # (B, 512) -> (B, 512)
        if resolution is None:
            x = self.generator(z)
        else:
            x = self.generator(z, resolution=resolution, alpha=alpha)
        if x.shape[-1] < TARGET_RESOLUTION:
            x = F.interpolate(
                x, size=(TARGET_RESOLUTION, TARGET_RESOLUTION),
                mode="bilinear", align_corners=False,
            )                                               # (B, 3, R, R) -> (B, 3, 1024, 1024)
        return x


def main() -> None:
    import shutil

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ckpt", type=Path, default=CKPT_PATH)
    parser.add_argument("--psis", type=float, nargs="+", default=PSIS)
    parser.add_argument("--valid-zip", type=str, default=VALID_ZIP)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--num-fake", type=int, default=NUM_FAKE)
    parser.add_argument("--num-real", type=int, default=NUM_REAL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    # Read resolution and alpha directly from checkpoint progressive_state
    raw_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    prog = raw_ckpt.get("progressive_state", {}) or {}
    resolution = int(prog.get("resolution", RESOLUTION))
    alpha = float(prog.get("alpha", 1.0))
    print(f"Checkpoint progressive_state: resolution={resolution}, alpha={alpha:.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_generator(args.ckpt).to(device)
    print(f"Loaded G_ema from {args.ckpt.name} (z_dim={generator.z_dim})")

    if not Path(args.valid_zip).is_file():
        raise FileNotFoundError(f"Real validation zip not found: {args.valid_zip}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    real_dir = args.out_dir / f"real_1024_{args.num_real}"
    n_real = extract_validation_subset(
        zip_path=args.valid_zip, out_dir=real_dir, max_images=args.num_real
    )
    print(f"Real set: {n_real} images -> {real_dir}")

    results: list[tuple[float, float]] = []
    for psi in args.psis:
        fake_dir = args.out_dir / f"fake_psi{psi:.2f}"
        print(f"\n[psi={psi:.2f}] generating {args.num_fake} fakes "
              f"(resolution={resolution}, alpha={alpha:.3f})...")
        write_fake_validation_images(
            G=TruncatedGenerator(generator, psi),
            out_dir=fake_dir,
            z_dim=generator.z_dim,
            n_images=args.num_fake,
            batch_size=args.batch_size,
            device=device,
            seed=SEED,
            resolution=resolution,
            alpha=alpha,
        )
        fid = run_pytorch_fid(fake_dir, real_dir, device=device)
        shutil.rmtree(fake_dir, ignore_errors=True)
        if fid is None:
            print(f"[psi={psi:.2f}] FID failed (install pytorch-fid scipy)")
            continue
        print(f"[psi={psi:.2f}] FID = {fid:.4f}")
        results.append((psi, fid))

    if not results:
        print("\nNo FID measured. Install with: pip install pytorch-fid scipy")
        return

    print("\n==================== FID sweep ====================")
    print(f"{'psi':>6} | {'FID':>10}")
    print("-" * 22)
    for psi, fid in results:
        print(f"{psi:>6.2f} | {fid:>10.4f}")
    best_psi, best_fid = min(results, key=lambda pair: pair[1])
    print("-" * 22)
    print(f"Lowest FID: psi={best_psi:.2f} (FID={best_fid:.4f})")
    print("Set verify_truncation.EXPORT_PSI to this, re-export, and submit.")


if __name__ == "__main__":
    main()
