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

from pathlib import Path

import torch
import torch.nn as nn

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


class TruncatedGenerator(nn.Module):
    """Apply z *= psi, then forward exactly like the bare Generator.

    Keeps the (z, resolution, alpha) signature so it drops straight into
    write_fake_validation_images without changing the tested pipeline.
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
            return self.generator(z)
        return self.generator(z, resolution=resolution, alpha=alpha)


def main() -> None:
    import shutil

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = load_generator(CKPT_PATH).to(device)
    print(f"Loaded G_ema from {CKPT_PATH.name} (z_dim={generator.z_dim})")

    if not Path(VALID_ZIP).is_file():
        raise FileNotFoundError(f"Real validation zip not found: {VALID_ZIP}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real_dir = OUT_DIR / f"real_{RESOLUTION}_{NUM_REAL}"
    n_real = extract_validation_subset(
        zip_path=VALID_ZIP, out_dir=real_dir, max_images=NUM_REAL
    )
    print(f"Real set: {n_real} images at {RESOLUTION} -> {real_dir}")

    results: list[tuple[float, float]] = []
    for psi in PSIS:
        fake_dir = OUT_DIR / f"fake_psi{psi:.2f}"
        print(f"\n[psi={psi:.2f}] generating {NUM_FAKE} fakes "
              f"(resolution={RESOLUTION}, alpha={ALPHA})...")
        write_fake_validation_images(
            G=TruncatedGenerator(generator, psi),
            out_dir=fake_dir,
            z_dim=generator.z_dim,
            n_images=NUM_FAKE,
            batch_size=BATCH_SIZE,
            device=device,
            seed=SEED,
            resolution=RESOLUTION,
            alpha=ALPHA,
        )
        fid = run_pytorch_fid(fake_dir, real_dir, device=device)
        shutil.rmtree(fake_dir, ignore_errors=True)        # free disk between psi
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
